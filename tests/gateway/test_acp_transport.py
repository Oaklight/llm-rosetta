"""The ACP transport, driven against a fake agent.

No Google account and no ``agy`` are needed: the agent is a small Python
script that speaks the same JSON-RPC over stdio.  What is under test is this
gateway's half — process lifecycle, the turn's chunk shapes, the sign-in
message, session reuse and the refusal that becomes a status code — not the
protocol implementation, which is vendored.

The fake agent appends every ``session/new``, ``session/resume``,
``session/load`` and ``session/prompt`` it receives to ``$FAKE_ACP_LOG`` as one
JSON object per line, so a test can assert what the gateway actually sent
rather than inferring it from the reply.
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import Any

import pytest

from llm_rosetta_gateway.transport.acp import AcpTransport
from llm_rosetta_gateway.transport.acp_ledger import (
    AcpSessionLedger,
    LedgerEntry,
    SessionKey,
)
from llm_rosetta_gateway.transport.provider_info import ProviderInfo, openai_auth

#: A minimal ACP agent: handshake, sessions, text + thought + usage, stop.
#: ``FAKE_ACP_MODE`` makes one script a working agent, a signed-out one, one
#: that fails mid-turn, and one that advertises ``session/resume``.
FAKE_AGENT = textwrap.dedent(
    """
    import json, os, sys

    MODE = os.environ.get("FAKE_ACP_MODE", "ok")
    LOG = os.environ.get("FAKE_ACP_LOG", "")
    SESSIONS = [0]

    def note(kind, **fields):
        if not LOG:
            return
        with open(LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": kind, **fields}) + "\\n")

    def send(payload):
        sys.stdout.write(json.dumps(payload) + "\\n")
        sys.stdout.flush()

    def update(session_id, body):
        send({"jsonrpc": "2.0", "method": "session/update",
              "params": {"sessionId": session_id, "update": body}})

    def capabilities():
        if MODE == "resume":
            return {"sessionCapabilities": {"resume": {}}}
        if MODE == "load":
            return {"loadSession": True}
        return {}

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
                "agentCapabilities": capabilities(),
                "authMethods": [{"id": "oauth-personal", "name": "Sign in"}],
            }})
        elif method == "session/new":
            if MODE == "signed-out":
                send({"jsonrpc": "2.0", "id": req_id, "error": {
                    "code": -32603,
                    "message": "Failed to start Antigravity conversation",
                }})
                continue
            SESSIONS[0] += 1
            sid = "sess-%d" % SESSIONS[0]
            note("new", session=sid)
            send({"jsonrpc": "2.0", "id": req_id, "result": {
                "sessionId": sid,
                "configOptions": [{
                    "id": "model", "name": "Model", "type": "select",
                    "currentValue": "gemini-3.1-pro", "options": [],
                    "category": "model",
                }],
            }})
        elif method in ("session/resume", "session/load"):
            sid = msg["params"]["sessionId"]
            note(method.split("/")[1], session=sid)
            if method == "session/load":
                update(sid, {"sessionUpdate": "agent_message_chunk",
                             "content": {"type": "text", "text": "replayed"}})
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})
        elif method == "session/prompt":
            sid = msg["params"]["sessionId"]
            text = "".join(
                block.get("text", "")
                for block in msg["params"].get("prompt") or []
                if block.get("type") == "text"
            )
            note("prompt", session=sid, text=text)
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
            note("close", session=msg["params"]["sessionId"])
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})
        elif req_id is not None:
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})
    """
)


def _provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    mode: str,
    conversation: str | None = None,
) -> ProviderInfo:
    script = tmp_path / "fake_acp.py"
    script.write_text(FAKE_AGENT, encoding="utf-8")
    monkeypatch.setenv("FAKE_ACP_MODE", mode)
    monkeypatch.setenv("FAKE_ACP_LOG", str(tmp_path / "agent.log"))
    provider = ProviderInfo(
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
    return provider.with_acp_conversation(conversation)


def _log(tmp_path: Any) -> list[dict[str, Any]]:
    """Everything the fake agent was asked to do, in order."""
    path = tmp_path / "agent.log"
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _kinds(tmp_path: Any) -> list[str]:
    return [entry["kind"] for entry in _log(tmp_path)]


TURN = {
    "model": "gemini-3.1-pro",
    "system": "",
    "prompt": [{"type": "text", "text": "[User]\nhi"}],
}


def _turn(*messages: str, cwd: str | None = None) -> dict[str, Any]:
    """A turn carrying a whole conversation, one block per message."""
    body: dict[str, Any] = {
        "model": "gemini-3.1-pro",
        "system": "",
        "prompt": [{"type": "text", "text": m} for m in messages],
    }
    if cwd is not None:
        body["cwd"] = cwd
    return body


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
        """The agent is started once per provider, not once per request.

        Two requests that declare no conversation share the process but not a
        session: each gets its own, because nothing identifies them as one
        conversation.
        """
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "ok")
        try:
            first = await transport.send(provider, "", TURN)
            second = await transport.send(provider, "", TURN)
            assert len(transport._agents) == 1
        finally:
            await transport.close()
        assert first.body is not None and second.body is not None
        assert first.body["session_id"] == "sess-1"
        assert second.body["session_id"] == "sess-2"
        assert len(transport._agents) == 0  # closed, not leaked


class TestSessionReuse:
    """A conversation that declares itself keeps one agent session."""

    async def test_the_tail_of_a_growing_history_is_all_that_is_sent(
        self, monkeypatch, tmp_path
    ):
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "ok", conversation="conv-1")
        try:
            await transport.send(provider, "", _turn("[User]\none"))
            await transport.send(provider, "", _turn("[User]\none", "[Assistant]\ntwo"))
            await transport.send(
                provider,
                "",
                _turn("[User]\none", "[Assistant]\ntwo", "[User]\nthree"),
            )
        finally:
            await transport.close()

        prompts = [e["text"] for e in _log(tmp_path) if e["kind"] == "prompt"]
        assert prompts == ["[User]\none", "[Assistant]\ntwo", "[User]\nthree"]
        # One session for the conversation, not one per request.
        assert _kinds(tmp_path).count("new") == 1

    async def test_the_same_history_again_replays_into_a_fresh_session(
        self, monkeypatch, tmp_path
    ):
        """A retry of an answered turn cannot trust what the agent recorded."""
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "ok", conversation="conv-1")
        try:
            await transport.send(provider, "", _turn("[User]\none"))
            await transport.send(provider, "", _turn("[User]\none"))
        finally:
            await transport.close()

        assert _kinds(tmp_path).count("new") == 2
        prompts = [e["text"] for e in _log(tmp_path) if e["kind"] == "prompt"]
        assert prompts == ["[User]\none", "[User]\none"]

    async def test_a_rewritten_history_replays_the_whole_conversation(
        self, monkeypatch, tmp_path
    ):
        """A compaction or an edited turn is not an extension of the record."""
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "ok", conversation="conv-1")
        try:
            await transport.send(provider, "", _turn("[User]\none", "[Assistant]\ntwo"))
            await transport.send(
                provider, "", _turn("[User]\none", "[Assistant]\nREWRITTEN")
            )
        finally:
            await transport.close()

        assert _kinds(tmp_path).count("new") == 2
        prompts = [e["text"] for e in _log(tmp_path) if e["kind"] == "prompt"]
        assert prompts[1] == "[User]\none\n\n[Assistant]\nREWRITTEN"

    async def test_a_conversation_that_declares_nothing_gets_a_session_per_request(
        self, monkeypatch, tmp_path
    ):
        """No declared id means no identity to key a session on."""
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "ok")
        try:
            await transport.send(provider, "", _turn("[User]\none"))
            await transport.send(provider, "", _turn("[User]\none"))
        finally:
            await transport.close()

        assert _kinds(tmp_path).count("new") == 2
        assert transport.ledger.entries_of("antigravity") == []

    async def test_a_failed_turn_leaves_no_session_to_reuse(
        self, monkeypatch, tmp_path
    ):
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "fails", conversation="conv-1")
        try:
            first = await transport.send(provider, "", _turn("[User]\none"))
            second = await transport.send(
                provider, "", _turn("[User]\none", "[Assistant]\ntwo")
            )
        finally:
            await transport.close()

        # A non-streaming client cannot tell "the model said nothing" from
        # "the agent broke", so a failed turn is a failure, not an empty 200.
        assert first.status_code == 502
        assert "model call failed" in first.error_text
        assert transport.ledger.entries_of("antigravity") == []
        assert _kinds(tmp_path).count("new") == 2
        # The retry opens its own session rather than continuing the failed one.
        assert second.status_code == 502


class TestTheSystemInstruction:
    """A chat lane must not drop the system instruction silently."""

    async def test_the_system_instruction_reaches_the_agent(
        self, monkeypatch, tmp_path
    ):
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "ok", conversation="conv-1")
        body = _turn("[User]\nhi")
        body["system"] = "SYSTEM-MARKER: be exact."
        try:
            await transport.send(provider, "", body)
        finally:
            await transport.close()

        prompts = [e["text"] for e in _log(tmp_path) if e["kind"] == "prompt"]
        assert "SYSTEM-MARKER" in prompts[0]
        # Labelled, so the model can tell instructions from the user's turn.
        assert prompts[0].startswith("[System]")

    async def test_it_is_sent_once_and_never_again_on_later_turns(
        self, monkeypatch, tmp_path
    ):
        """It is the first block, so a reused session already has it."""
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "ok", conversation="conv-1")
        first = _turn("[User]\none")
        first["system"] = "SYSTEM-MARKER"
        second = _turn("[User]\none", "[Assistant]\ntwo")
        second["system"] = "SYSTEM-MARKER"
        try:
            await transport.send(provider, "", first)
            await transport.send(provider, "", second)
        finally:
            await transport.close()

        prompts = [e["text"] for e in _log(tmp_path) if e["kind"] == "prompt"]
        assert "SYSTEM-MARKER" in prompts[0]
        assert "SYSTEM-MARKER" not in prompts[1]
        assert prompts[1] == "[Assistant]\ntwo"

    async def test_a_changed_system_instruction_reopens_the_session(
        self, monkeypatch, tmp_path
    ):
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "ok", conversation="conv-1")
        first = _turn("[User]\none")
        first["system"] = "one"
        second = _turn("[User]\none", "[Assistant]\ntwo")
        second["system"] = "two"
        try:
            await transport.send(provider, "", first)
            await transport.send(provider, "", second)
        finally:
            await transport.close()

        assert _kinds(tmp_path).count("new") == 2


class TestAnAbandonedStream:
    """A consumer that stops reading still gives its session back."""

    async def test_closing_the_stream_mid_turn_closes_the_session(
        self, monkeypatch, tmp_path
    ):
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "ok", conversation="conv-1")
        try:
            stream = await transport.send_streaming(provider, "", _turn("[User]\none"))
            iterator = stream.__aiter__()
            assert (await iterator.__anext__())["type"] == "session_start"
            await stream.close()
            remaining = [chunk async for chunk in iterator]
        finally:
            await transport.close()

        assert remaining == []
        assert transport.ledger.get(SessionKey("antigravity", "conv-1")) is None
        assert "close" in _kinds(tmp_path)

    async def test_a_consumer_that_never_iterates_still_releases_the_turn(
        self, monkeypatch, tmp_path
    ):
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "ok", conversation="conv-1")
        try:
            stream = await transport.send_streaming(provider, "", _turn("[User]\none"))
            await stream.close()
            # The lock the turn took is free again, so the next turn proceeds.
            second = await transport.send(
                provider, "", _turn("[User]\none", "[User]\ntwo")
            )
        finally:
            await transport.close()

        assert second.body is not None
        assert _kinds(tmp_path).count("new") == 2


class TestSessionResumeAndLoad:
    """A session that outlived its process is brought back, not replayed blind."""

    async def test_resume_continues_a_session_the_process_has_forgotten(
        self, monkeypatch, tmp_path
    ):
        """With ``session/resume`` advertised, a dead entry is brought back."""
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "resume", conversation="conv-1")
        transport.ledger.put(
            LedgerEntry(
                key=SessionKey("antigravity", "conv-1"),
                session_id="sess-9",
                cwd=str(tmp_path),
                model="gemini-3.1-pro",
                seen=(*_digests_for(["[User]\none"]),),
                live=False,
            )
        )
        try:
            await transport.send(
                provider,
                "",
                _turn("[User]\none", "[Assistant]\ntwo", cwd=str(tmp_path)),
            )
        finally:
            await transport.close()

        kinds = _kinds(tmp_path)
        assert "resume" in kinds
        assert "new" not in kinds
        prompts = [e["text"] for e in _log(tmp_path) if e["kind"] == "prompt"]
        assert prompts == ["[Assistant]\ntwo"]

    async def test_load_discards_the_history_the_agent_replays(
        self, monkeypatch, tmp_path
    ):
        """``session/load`` replays the transcript; the client did not ask."""
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "load", conversation="conv-1")
        transport.ledger.put(
            LedgerEntry(
                key=SessionKey("antigravity", "conv-1"),
                session_id="sess-9",
                cwd=str(tmp_path),
                model="gemini-3.1-pro",
                seen=_digests_for(["[User]\none"]),
                live=False,
            )
        )
        try:
            stream = await transport.send_streaming(
                provider,
                "",
                _turn("[User]\none", "[Assistant]\ntwo", cwd=str(tmp_path)),
            )
            chunks = [chunk async for chunk in stream]
        finally:
            await transport.close()

        assert "load" in _kinds(tmp_path)
        texts = [c.get("text") for c in chunks if c["type"] == "text"]
        assert "replayed" not in texts
        assert texts == ["Hello"]


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


class TestTurnSerialisation:
    """Two requests for one conversation must not interleave."""

    async def test_a_second_request_waits_for_the_first_turn(
        self, monkeypatch, tmp_path
    ):
        import asyncio

        transport = AcpTransport(ledger=AcpSessionLedger())
        provider = _provider(monkeypatch, tmp_path, "ok", conversation="conv-1")
        try:
            first = await transport.send_streaming(provider, "", _turn("[User]\none"))
            # The first turn holds the session lock until its stream drains.
            pending = asyncio.ensure_future(
                transport.send(provider, "", _turn("[User]\none", "[User]\ntwo"))
            )
            await _spin()
            assert not pending.done(), "the second turn ran inside the first"
            [chunk async for chunk in first]
            second = await pending
        finally:
            await transport.close()
        assert second.body is not None
        prompts = [e["text"] for e in _log(tmp_path) if e["kind"] == "prompt"]
        assert prompts == ["[User]\none", "[User]\ntwo"]


def _digests_for(messages: list[str]) -> tuple[str, ...]:
    from llm_rosetta_gateway.transport.acp_ledger import block_digests

    return block_digests({"type": "text", "text": m} for m in messages)


async def _spin() -> None:
    """Let the event loop run the other task as far as it can get."""
    import asyncio

    for _ in range(5):
        await asyncio.sleep(0)
