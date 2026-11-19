"""The ACP transport, driven against a fake agent.

No Google account and no ``agy`` are needed: the agent is a small Python
script that speaks the same JSON-RPC over stdio.  What is under test is this
gateway's half — process lifecycle, the turn's chunk shapes, the sign-in
message, and the refusal that becomes a status code — not the protocol
implementation, which is vendored.
"""

from __future__ import annotations

import sys
import textwrap
from typing import Any

import pytest

from llm_rosetta_gateway.transport.acp import AcpTransport
from llm_rosetta_gateway.transport.provider_info import ProviderInfo, openai_auth

#: A minimal ACP agent: handshake, one session, text + thought + usage, stop.
#: ``MODE`` is read from the environment's first argument so one script can be
#: a working agent, a signed-out one, and one that fails mid-turn.
FAKE_AGENT = textwrap.dedent(
    """
    import json, os, sys

    MODE = os.environ.get("FAKE_ACP_MODE", "ok")

    def send(payload):
        sys.stdout.write(json.dumps(payload) + "\\n")
        sys.stdout.flush()

    def update(session_id, body):
        send({"jsonrpc": "2.0", "method": "session/update",
              "params": {"sessionId": session_id, "update": body}})

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        msg = json.loads(line)
        method = msg.get("method")
        req_id = msg.get("id")

        if method == "initialize":
            send({"jsonrpc": "2.0", "id": req_id, "result": {
                "protocolVersion": 1,
                "agentInfo": {"name": "fake-acp", "version": "0.1"},
                "agentCapabilities": {},
                "authMethods": [{"id": "oauth-personal", "name": "Sign in"}],
            }})
        elif method == "session/new":
            if MODE == "signed-out":
                send({"jsonrpc": "2.0", "id": req_id, "error": {
                    "code": -32603,
                    "message": "Failed to start Antigravity conversation",
                }})
                continue
            send({"jsonrpc": "2.0", "id": req_id, "result": {
                "sessionId": "sess-1",
                "configOptions": [{
                    "id": "model", "name": "Model", "type": "select",
                    "currentValue": "gemini-3.1-pro", "options": [],
                    "category": "model",
                }],
            }})
        elif method == "session/prompt":
            sid = msg["params"]["sessionId"]
            if MODE == "fails":
                send({"jsonrpc": "2.0", "id": req_id, "error": {
                    "code": -32603, "message": "model call failed",
                }})
                continue
            update(sid, {"sessionUpdate": "agent_thought_chunk",
                         "content": {"type": "text", "text": "thinking"}})
            update(sid, {"sessionUpdate": "agent_message_chunk",
                         "content": {"type": "text", "text": "Hello"}})
            update(sid, {"sessionUpdate": "usage_update", "used": 42,
                         "size": 1000})
            send({"jsonrpc": "2.0", "id": req_id,
                  "result": {"stopReason": "end_turn"}})
        elif method == "session/close":
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})
        elif req_id is not None:
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})
    """
)


def _provider(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Any, mode: str
) -> ProviderInfo:
    script = tmp_path / "fake_acp.py"
    script.write_text(FAKE_AGENT, encoding="utf-8")
    monkeypatch.setenv("FAKE_ACP_MODE", mode)
    return ProviderInfo(
        "acp",
        config_name="antigravity",
        api_key="",
        base_url="http://agy.local",
        auth_header_fn=openai_auth,
        url_template="{base_url}/acp-turn",
        keyless=True,
        transport="acp",
        acp_argv=(sys.executable, str(script)),
    )


TURN = {
    "model": "gemini-3.1-pro",
    "system": "",
    "prompt": [{"type": "text", "text": "[User]\nhi"}],
}


class TestTheStream:
    """What one turn yields, chunk by chunk."""

    async def test_a_turn_streams_text_thought_and_a_stop(self, monkeypatch, tmp_path):
        transport = AcpTransport()
        try:
            stream = await transport.send_streaming(
                _provider(monkeypatch, tmp_path, "ok"), "", TURN
            )
            assert stream.status_code == 200
            chunks = [chunk async for chunk in stream]
        finally:
            await transport.close()
        kinds = [chunk["type"] for chunk in chunks]
        assert kinds == ["session_start", "thought", "text", "usage", "stop"]
        assert chunks[0] == {
            "type": "session_start",
            "session_id": "sess-1",
            "model": "gemini-3.1-pro",
        }
        assert chunks[2]["text"] == "Hello"
        assert chunks[-1]["stop_reason"] == "end_turn"

    async def test_one_process_serves_two_turns(self, monkeypatch, tmp_path):
        """The agent is started once per provider, not once per request."""
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "ok")
        try:
            first = await transport.send(provider, "", TURN)
            second = await transport.send(provider, "", TURN)
        finally:
            await transport.close()
        assert first.body is not None and second.body is not None
        assert first.body["session_id"] == second.body["session_id"] == "sess-1"
        assert len(transport._agents) == 0  # closed, not leaked


class TestRefusals:
    """An agent that cannot serve the turn, and how it is reported."""

    async def test_a_signed_out_agent_is_a_503_naming_the_sign_in(
        self, monkeypatch, tmp_path
    ):
        transport = AcpTransport()
        try:
            resp = await transport.send(
                _provider(monkeypatch, tmp_path, "signed-out"), "", TURN
            )
        finally:
            await transport.close()
        assert resp.status_code == 503
        text = resp.error_text
        assert "could not open a session" in text
        assert "Sign the agent CLI in" in text
        # The agent's own words are kept: they are what an operator googles.
        assert "Failed to start Antigravity conversation" in text

    async def test_a_streamed_refusal_is_a_status_not_a_chunk(
        self, monkeypatch, tmp_path
    ):
        transport = AcpTransport()
        try:
            stream = await transport.send_streaming(
                _provider(monkeypatch, tmp_path, "signed-out"), "", TURN
            )
            assert stream.status_code == 503
            assert "sign" in (await stream.read_error()).lower()
            assert [chunk async for chunk in stream] == []
        finally:
            await transport.close()

    async def test_a_failed_prompt_reports_an_error_chunk(self, monkeypatch, tmp_path):
        transport = AcpTransport()
        try:
            stream = await transport.send_streaming(
                _provider(monkeypatch, tmp_path, "fails"), "", TURN
            )
            chunks = [chunk async for chunk in stream]
        finally:
            await transport.close()
        assert chunks[-1]["type"] == "error"
        assert "model call failed" in chunks[-1]["message"]
