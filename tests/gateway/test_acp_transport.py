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

import asyncio
import json
import os
import sys
import textwrap
import urllib.request
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
    import json, os, sys, urllib.request

    MODE = os.environ.get("FAKE_ACP_MODE", "ok")
    LOG = os.environ.get("FAKE_ACP_LOG", "")
    SESSIONS = [0]
    MCP_URL = ""

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

    def mcp_call(method, params=None, rpc_id=1):
        payload = json.dumps({"jsonrpc": "2.0", "id": rpc_id,
                              "method": method, "params": params or {}})
        request = urllib.request.Request(
            MCP_URL, data=payload.encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            return json.loads(response.read() or b"{}")

    # One line per process, so a test can tell a replaced agent from a reused
    # one: a process that has to be thrown away shows up as a second pid.
    note("start", pid=os.getpid(), argv=sys.argv[1:],
         marker=os.environ.get("FAKE_ACP_MARKER"),
         has_path=bool(os.environ.get("PATH")))

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
            note("initialize", params=msg.get("params"))
            send({"jsonrpc": "2.0", "id": req_id, "result": {
                "protocolVersion": 1,
                "agentInfo": {"name": "fake-acp", "version": "0.1"},
                "agentCapabilities": capabilities(),
                "authMethods": [{"id": "oauth-personal", "name": "Sign in"}],
            }})
        elif method == "session/new":
            if MODE == "mute-new":
                continue
            if MODE == "signed-out":
                send({"jsonrpc": "2.0", "id": req_id, "error": {
                    "code": -32603,
                    "message": "Failed to start Antigravity conversation",
                }})
                continue
            SESSIONS[0] += 1
            sid = "sess-%d" % SESSIONS[0]
            note("new", session=sid)
            servers = msg["params"].get("mcpServers") or []
            MCP_URL = servers[0]["url"] if servers else ""
            if MCP_URL:
                note("mcp", url=MCP_URL)
            send({"jsonrpc": "2.0", "id": req_id, "result": {
                "sessionId": sid,
                "modes": None if MODE == "no-mode" else {
                    "currentModeId": "default",
                    "availableModes": (
                        [{"id": "default", "name": "Default"}]
                        if MODE == "modes-without-plan"
                        else [
                            {"id": "default", "name": "Default"},
                            {"id": "plan", "name": "Plan"},
                        ]
                    ),
                },
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
        elif method == "session/set_mode":
            note("mode-attempt", session=msg["params"]["sessionId"],
                 mode=msg["params"].get("modeId"))
            if MODE == "no-mode":
                note("mode-unsupported", session=msg["params"]["sessionId"])
                send({"jsonrpc": "2.0", "id": req_id, "error": {
                    "code": -32601, "message": "Method not found",
                }})
                continue
            if MODE == "mode-refused":
                note("mode-refused", session=msg["params"]["sessionId"])
                send({"jsonrpc": "2.0", "id": req_id, "error": {
                    "code": -32602, "message": "unknown mode",
                }})
                continue
            note("mode", session=msg["params"]["sessionId"],
                 mode=msg["params"].get("modeId"))
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})
        elif method == "session/set_config_option":
            if MODE == "config-stall":
                continue
            note(
                "config",
                config=msg["params"].get("configId"),
                value=msg["params"].get("value"),
            )
            send({"jsonrpc": "2.0", "id": req_id, "result": {}})
        elif method == "session/prompt":
            sid = msg["params"]["sessionId"]
            text = "".join(
                block.get("text", "")
                for block in msg["params"].get("prompt") or []
                if block.get("type") == "text"
            )
            note("prompt", session=sid, text=text)
            if MODE == "mcp-list":
                listed = []
                if MCP_URL:
                    listed = (
                        mcp_call("tools/list").get("result", {}).get("tools") or []
                    )
                names = [tool.get("name") for tool in listed]
                note("mcp-tools", names=names)
                update(sid, {"sessionUpdate": "agent_message_chunk",
                             "content": {"type": "text",
                                         "text": "tools: "
                                                 + (", ".join(names) or "none")}})
                send({"jsonrpc": "2.0", "id": req_id,
                      "result": {"stopReason": "end_turn"}})
                continue
            if MODE == "conformance":
                # #1629's whole contract in one turn: a request tool reaches
                # the agent over MCP, while a host file read and a host shell
                # command are refused.
                listed = mcp_call("tools/list").get("result", {}).get("tools") or []
                note("mcp-tools", names=[tool.get("name") for tool in listed])
                call = {"toolCallId": "conf-1",
                        "name": "mcp_llm_rosetta_tools_echo",
                        "title": "echo", "kind": "other",
                        "rawInput": {"text": "hello"}}
                update(sid, dict(call, sessionUpdate="tool_call",
                                 status="pending"))
                options = [
                    {"optionId": "a1", "name": "Allow", "kind": "allow_once"},
                    {"optionId": "r1", "name": "Reject", "kind": "reject_once"},
                ]
                send({"jsonrpc": "2.0", "id": 910,
                      "method": "session/request_permission",
                      "params": {"sessionId": sid, "toolCall": call,
                                 "options": options}})
                note("permission_reply", reply=json.loads(sys.stdin.readline()))
                result = mcp_call("tools/call", {"name": "echo",
                                                 "arguments": {"text": "hello"}})
                try:
                    answer = result["result"]["content"][0]["text"]
                except Exception:
                    answer = json.dumps(result)
                note("mcp-result", text=answer)
                send({"jsonrpc": "2.0", "id": 911,
                      "method": "fs/read_text_file",
                      "params": {"sessionId": sid, "path": "/etc/hostname"}})
                note("fs_reply", reply=json.loads(sys.stdin.readline()))
                shell = {"toolCallId": "conf-shell", "name": "shell",
                         "title": "Run shell command", "kind": "execute",
                         "rawInput": {"command": "touch shell-side-effect"}}
                send({"jsonrpc": "2.0", "id": 912,
                      "method": "session/request_permission",
                      "params": {"sessionId": sid, "toolCall": shell,
                                 "options": options}})
                shell_reply = json.loads(sys.stdin.readline())
                note("permission_reply", reply=shell_reply)
                outcome = (shell_reply.get("result") or {}).get("outcome") or {}
                if outcome.get("optionId") == "a1":
                    with open(os.path.join(os.path.dirname(LOG),
                                           "shell-side-effect"), "w") as handle:
                        handle.write("ran")
                update(sid, {"sessionUpdate": "agent_message_chunk",
                             "content": {"type": "text",
                                         "text": "bridged: " + answer}})
                send({"jsonrpc": "2.0", "id": req_id,
                      "result": {"stopReason": "end_turn"}})
                continue
            if MODE == "mcp-change":
                listed = mcp_call("tools/list").get("result", {}).get("tools") or []
                names = [tool.get("name") for tool in listed]
                note("mcp-tools", names=names)
                if "[Tool result]" not in text:
                    call = {"toolCallId": "mcp-change-1",
                            "name": "mcp_llm_rosetta_tools_echo",
                            "title": "echo", "kind": "other",
                            "rawInput": {"text": "hello"}}
                    update(sid, dict(call, sessionUpdate="tool_call",
                                     status="pending"))
                    options = [
                        {"optionId": "a1", "name": "Allow", "kind": "allow_once"},
                        {"optionId": "r1", "name": "Reject", "kind": "reject_once"},
                    ]
                    send({"jsonrpc": "2.0", "id": 903,
                          "method": "session/request_permission",
                          "params": {"sessionId": sid, "toolCall": call,
                                     "options": options}})
                    reply = json.loads(sys.stdin.readline())
                    note("permission_reply", reply=reply)
                    result = mcp_call("tools/call", {"name": "echo",
                                                     "arguments": {"text": "hello"}})
                    try:
                        answer = result["result"]["content"][0]["text"]
                    except Exception:
                        answer = json.dumps(result)
                    note("mcp-result", text=answer)
                    update(sid, {"sessionUpdate": "agent_message_chunk",
                                 "content": {"type": "text",
                                             "text": "tool said: " + answer}})
                    send({"jsonrpc": "2.0", "id": req_id,
                          "result": {"stopReason": "end_turn"}})
                    continue
                update(sid, {"sessionUpdate": "agent_message_chunk",
                             "content": {"type": "text",
                                         "text": "resumed with tools: "
                                                 + (", ".join(names) or "none")}})
                send({"jsonrpc": "2.0", "id": req_id,
                      "result": {"stopReason": "end_turn"}})
                continue
            if MODE == "mcp":
                listed = mcp_call("tools/list").get("result", {}).get("tools") or []
                note("mcp-tools", names=[tool.get("name") for tool in listed])
                call = {"toolCallId": "mcp-1",
                        "name": "mcp_llm_rosetta_tools_echo",
                        "title": "echo", "kind": "other",
                        "rawInput": {"text": "hello"}}
                update(sid, dict(call, sessionUpdate="tool_call",
                                 status="pending"))
                options = [
                    {"optionId": "a1", "name": "Allow", "kind": "allow_once"},
                    {"optionId": "r1", "name": "Reject", "kind": "reject_once"},
                ]
                send({"jsonrpc": "2.0", "id": 901,
                      "method": "session/request_permission",
                      "params": {"sessionId": sid, "toolCall": call,
                                 "options": options}})
                reply = json.loads(sys.stdin.readline())
                note("permission_reply", reply=reply)
                result = mcp_call("tools/call", {"name": "echo",
                                                 "arguments": {"text": "hello"}})
                try:
                    answer = result["result"]["content"][0]["text"]
                except Exception:
                    answer = json.dumps(result)
                note("mcp-result", text=answer)
                update(sid, {"sessionUpdate": "agent_message_chunk",
                             "content": {"type": "text",
                                         "text": "tool said: " + answer}})
                send({"jsonrpc": "2.0", "id": req_id,
                      "result": {"stopReason": "end_turn"}})
                continue
            if MODE == "mcp-claude":
                # Claude Code names the call once, then sends input deltas for
                # the same call id without a name.
                cid = "toolu_fake_1"
                update(sid, {"sessionUpdate": "tool_call_update",
                             "toolCallId": cid,
                             "name": "mcp__llm_rosetta_tools__echo",
                             "title": "llm_rosetta_tools: echo",
                             "kind": "other", "status": "pending",
                             "rawInput": {}})
                update(sid, {"sessionUpdate": "tool_call_update",
                             "toolCallId": cid,
                             "rawInput": {"text": "claude"}})
                send({"jsonrpc": "2.0", "id": 902,
                      "method": "session/request_permission",
                      "params": {
                          "sessionId": sid,
                          "title": "llm_rosetta_tools: echo",
                          "subject": {
                              "type": "tool_call",
                              "toolCall": {
                                  "toolCallId": cid,
                                  "name": "mcp__llm_rosetta_tools__echo",
                                  "rawInput": {"text": "claude"},
                                  "title": "llm_rosetta_tools: echo",
                                  "kind": "other",
                              },
                          },
                          "options": [
                              {"optionId": "allow-once", "name": "Yes",
                               "kind": "allow_once"},
                              {"optionId": "reject", "name": "No",
                               "kind": "reject_once"},
                          ],
                      }})
                reply = json.loads(sys.stdin.readline())
                note("permission_reply", reply=reply)
                result = mcp_call("tools/call", {"name": "echo",
                                                 "arguments": {"text": "claude"}})
                try:
                    answer = result["result"]["content"][0]["text"]
                except Exception:
                    answer = json.dumps(result)
                update(sid, {"sessionUpdate": "tool_call_update",
                             "toolCallId": cid, "status": "completed"})
                update(sid, {"sessionUpdate": "agent_message_chunk",
                             "content": {"type": "text",
                                         "text": "claude said: " + answer}})
                send({"jsonrpc": "2.0", "id": req_id,
                      "result": {"stopReason": "end_turn"}})
                continue
            if MODE == "mcp-parallel":
                import threading
                answers = {}

                def run_one(text):
                    result = mcp_call("tools/call", {
                        "name": "echo", "arguments": {"text": text}})
                    try:
                        answers[text] = result["result"]["content"][0]["text"]
                    except Exception:
                        answers[text] = json.dumps(result)

                threads = [threading.Thread(target=run_one, args=(value,))
                           for value in ("one", "two")]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join()
                note("mcp-parallel", answers=answers)
                update(sid, {"sessionUpdate": "agent_message_chunk",
                             "content": {"type": "text",
                                         "text": "tools said: " + ", ".join(
                                             answers.get(v, "?")
                                             for v in ("one", "two"))}})
                send({"jsonrpc": "2.0", "id": req_id,
                      "result": {"stopReason": "end_turn"}})
                continue
            if MODE in ("stalls", "wedged"):
                continue
            if MODE in ("tool", "mcp-tool"):
                # A frame for a tool the agent has started: the name is the
                # tool, the title and the raw input carry what a log must
                # never repeat.
                update(sid, {"sessionUpdate": "tool_call",
                             "toolCallId": "call_1",
                             "name": ("search_web" if MODE == "tool"
                                      else "mcp_bridge_echo"),
                             "title": "SECRET-TITLE", "kind": "search",
                             "status": "in_progress",
                             "rawInput": {"query": "SECRET-ARGUMENT"}})
                continue
            if MODE == "exits":
                sys.exit(3)
            if MODE == "fails":
                send({"jsonrpc": "2.0", "id": req_id, "error": {
                    "code": -32603, "message": "model call failed",
                }})
                continue
            if MODE in ("asks", "asks-allow-only", "asks-fs"):
                if MODE == "asks-fs":
                    send({"jsonrpc": "2.0", "id": 900,
                          "method": "fs/read_text_file",
                          "params": {"sessionId": sid, "path": "/etc/hostname"}})
                else:
                    options = [
                        {"optionId": "a1", "name": "Allow", "kind": "allow_once"},
                        {"optionId": "r1", "name": "Reject", "kind": "reject_once"},
                    ]
                    if MODE == "asks-allow-only":
                        options = options[:1]
                    send({"jsonrpc": "2.0", "id": 900,
                          "method": "session/request_permission",
                          "params": {"sessionId": sid,
                                     "toolCall": {"toolCallId": "t1",
                                                  "title": "Read /secret"},
                                     "options": options}})
                reply = json.loads(sys.stdin.readline())
                note("permission_reply", reply=reply)
            update(sid, {"sessionUpdate": "agent_thought_chunk",
                         "content": {"type": "text", "text": "thinking"}})
            update(sid, {"sessionUpdate": "agent_message_chunk",
                         "content": {"type": "text", "text": "Hello"}})
            update(sid, {"sessionUpdate": "usage_update", "used": 42,
                         "size": 1000})
            send({"jsonrpc": "2.0", "id": req_id,
                  "result": {"stopReason": "end_turn"}})
        elif method == "session/cancel":
            note("cancel", session=msg["params"]["sessionId"])
        elif method == "session/close":
            if MODE == "wedged":
                continue
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
    *,
    processes: int = 1,
    acp_env: dict[str, str] | None = None,
    agy: bool = False,
    wrapper: bool = False,
) -> ProviderInfo:
    if wrapper:
        # An agy the operator starts through a launcher: the executable's file
        # name says nothing, but the agent still advertises its modes.
        fake = tmp_path / "fake_acp.py"
        fake.write_text(FAKE_AGENT, encoding="utf-8")
        script = tmp_path / "run-agy.sh"
        script.write_text(
            f'#!/bin/sh\nexec "{sys.executable}" "{fake}"\n', encoding="utf-8"
        )
        script.chmod(0o755)
        argv: tuple[str, ...] = (str(script),)
    elif agy:
        # An agy-shaped test must start a file called ``agy``: the fallback
        # (an agent that advertises no modes) still decides by the name.
        script = tmp_path / "agy"
        script.write_text("#!/usr/bin/env python3\n" + FAKE_AGENT, encoding="utf-8")
        script.chmod(0o755)
        argv: tuple[str, ...] = (str(script),)
    else:
        script = tmp_path / "fake_acp.py"
        script.write_text(FAKE_AGENT, encoding="utf-8")
        argv = (sys.executable, str(script))
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
        acp_argv=argv,
        acp_argv_custom=agy or wrapper,
        acp_env=acp_env,
        acp_processes=processes,
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


async def _mcp_tool_names(url: str) -> list[str]:
    """Ask one session's MCP endpoint which tools it serves."""
    payload = json.dumps(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
    ).encode()

    def call() -> Any:
        request = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(request, timeout=5) as response:
            return json.loads(response.read() or b"{}")

    answer = await asyncio.to_thread(call)
    tools = answer.get("result", {}).get("tools") or []
    return [tool.get("name") for tool in tools]


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
            # Read inside the turn's life: close() drops the ledger, so an
            # assertion after it would pass whether or not a session was held.
            held = transport.ledger.entries_of("antigravity")
        finally:
            await transport.close()
        assert held == []

        assert _kinds(tmp_path).count("new") == 2

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
            # Inside the transport's life: close() drops the ledger.
            held = transport.ledger.entries_of("antigravity")
        finally:
            await transport.close()

        # A non-streaming client cannot tell "the model said nothing" from
        # "the agent broke", so a failed turn is a failure, not an empty 200.
        assert first.status_code == 502
        assert "model call failed" in first.error_text
        assert held == []
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
            held = transport.ledger.get(SessionKey("antigravity", "conv-1"))
        finally:
            await transport.close()

        assert remaining == []
        assert held is None
        assert "close" in _kinds(tmp_path)

    async def test_a_one_off_session_is_closed_when_its_consumer_leaves(
        self, monkeypatch, tmp_path
    ):
        """A request with no conversation carrier owns a session nobody keeps.

        Nothing will come back for it, so a consumer that disconnects after
        the first chunk must still see ``session/close`` sent for it.
        """
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "ok")
        try:
            stream = await transport.send_streaming(provider, "", _turn("[User]\none"))
            iterator = stream.__aiter__()
            first = await iterator.__anext__()
            assert first["type"] == "session_start"
            await stream.close()
            remaining = [chunk async for chunk in iterator]
            held = transport.ledger.entries_of("antigravity")
        finally:
            await transport.close()

        assert remaining == []
        assert held == []
        assert [e["session"] for e in _log(tmp_path) if e["kind"] == "close"] == [
            first["session_id"]
        ]

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


class TestAnAgentThatDies:
    """End of process is not end of turn.

    The vendored client signals both with the same sentinel, so a transport
    that trusted the end of its iteration would report a turn the model never
    finished as a clean stop (#1601's shape).  A dead process also keeps its
    slot unless it is taken out, and every later turn is then written into a
    pipe nobody reads.
    """

    async def test_a_process_that_exits_mid_turn_is_a_failure(
        self, monkeypatch, tmp_path
    ):
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "exits", conversation="conv-1")
        try:
            resp = await transport.send(provider, "", _turn("[User]\none"))
        finally:
            await transport.close()

        assert resp.status_code == 502
        assert resp.body is None

    async def test_the_stream_says_the_agent_went_away(self, monkeypatch, tmp_path):
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "exits", conversation="conv-1")
        try:
            stream = await transport.send_streaming(provider, "", _turn("[User]\none"))
            chunks = [chunk async for chunk in stream]
        finally:
            await transport.close()

        assert chunks[-1]["type"] == "error"
        # The whole sentence: an operator reads it, and it is not the
        # deadline's sentence.
        assert chunks[-1]["message"] == (
            "ACP agent for provider 'antigravity' ended the turn without "
            "reporting a stop; the conversation was closed. The agent process "
            "may have exited — check it on the gateway host."
        )
        # No stop chunk: the turn never reached one.
        assert not any(chunk["type"] == "stop" for chunk in chunks)

    async def test_the_dead_process_is_replaced_on_the_next_turn(
        self, monkeypatch, tmp_path
    ):
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "exits", conversation="conv-1")
        try:
            first = await transport.send(provider, "", _turn("[User]\none"))
            second = await transport.send(provider, "", _turn("[User]\ntwo"))
        finally:
            await transport.close()

        # Both are failures the transport reports, not exceptions out of send.
        assert first.status_code == 502
        assert second.status_code == 502
        # Each request reached a process of its own: the dead one left its slot.
        prompts = [entry for entry in _log(tmp_path) if entry["kind"] == "prompt"]
        assert len(prompts) == 2
        assert _kinds(tmp_path).count("new") == 2


class TestACancelledTurn:
    """Cancelling frees the slot even when the agent answers nothing.

    Cleanup is itself a request to the same agent, and the locks a turn holds
    are given back only after the cleanup returns.  An unbounded close made
    cancellation — and every timeout built on it — inert: the task parked in
    ``session/close`` instead of unwinding.
    """

    async def test_cancelling_a_turn_gives_its_locks_back(self, monkeypatch, tmp_path):
        from llm_rosetta_gateway.transport import acp as acp_module

        monkeypatch.setattr(acp_module, "CLEANUP_TIMEOUT", 0.05)
        transport = AcpTransport(timeout=30.0)
        provider = _provider(monkeypatch, tmp_path, "wedged", conversation="conv-1")

        async def consume() -> None:
            stream = await transport.send_streaming(provider, "", _turn("[User]\none"))
            async for _chunk in stream:
                pass

        try:
            task = asyncio.create_task(consume())
            await _await_prompt(tmp_path)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            held = transport.ledger.get(SessionKey("antigravity", "conv-1"))
            # The session lock is free: the next turn is acquired at once,
            # where a lock left behind would hold it past this timeout.
            second = await asyncio.wait_for(
                transport.send_streaming(provider, "", _turn("[User]\none")), 1.0
            )
            await second.close()
        finally:
            await transport.close()

        assert held is None


class TestASessionThatNeverOpens:
    """Setup spends the turn's budget, because setup is a request too.

    ``session/new``, ``session/resume``, ``session/load`` and
    ``set_config_option`` go to the same agent as the prompt.  A deadline that
    started after they returned left an agent that ignores them — the local
    CLI's "sick at startup" — with no bound at all, and with the process lock
    behind it.
    """

    async def test_a_session_new_that_never_answers_times_out(
        self, monkeypatch, tmp_path
    ):
        transport = AcpTransport(timeout=0.2)
        provider = _provider(monkeypatch, tmp_path, "mute-new", conversation="conv-1")
        body = _turn("[User]\none")
        body["tools"] = [ECHO_TOOL]
        try:
            resp = await asyncio.wait_for(transport.send(provider, "", body), 5.0)
            bridges = dict(transport._session_bridges)
            urls = dict(transport._session_urls)
            registrations = dict(transport._mcp._registrations)
        finally:
            await transport.close()

        assert resp.status_code == 503
        assert bridges == {}
        assert urls == {}
        # The URL handed out in ``session/new`` is no longer served: the
        # unregister on this failure path is what stops an orphan listener.
        assert registrations == {}
        assert resp.error_text == (
            "ACP agent for provider 'antigravity' did not open a session "
            "within 0.2s; the process was dropped. Check the agent on the "
            "gateway host."
        )

    async def test_the_stream_reports_it_as_a_status(self, monkeypatch, tmp_path):
        transport = AcpTransport(timeout=0.2)
        provider = _provider(monkeypatch, tmp_path, "mute-new", conversation="conv-1")
        try:
            stream = await asyncio.wait_for(
                transport.send_streaming(provider, "", _turn("[User]\none")), 5.0
            )
            error = await stream.read_error()
        finally:
            await transport.close()

        assert stream.status_code == 503
        assert error == (
            "ACP agent for provider 'antigravity' did not open a session "
            "within 0.2s; the process was dropped. Check the agent on the "
            "gateway host."
        )

    async def test_a_timed_out_session_new_drops_the_stalled_process(
        self, monkeypatch, tmp_path
    ):
        """The process that never opened a session leaves the pool.

        Left in its slot it stays half-read, and every later request on the
        lane queues behind its lock.
        """
        transport = AcpTransport(timeout=0.2)
        provider = _provider(monkeypatch, tmp_path, "mute-new", conversation="conv-1")
        try:
            first = await asyncio.wait_for(
                transport.send(provider, "", _turn("[User]\none")), 5.0
            )
            pooled = transport._pool("antigravity")
            second = await asyncio.wait_for(
                transport.send(provider, "", _turn("[User]\ntwo")), 5.0
            )
        finally:
            await transport.close()

        assert first.status_code == 503
        assert second.status_code == 503
        # The stalled agent is not in the pool after its timeout.
        assert pooled == []
        # Two processes, not one: the second request started its own.
        starts = [entry["pid"] for entry in _log(tmp_path) if entry["kind"] == "start"]
        assert len(starts) == 2, starts
        assert starts[0] != starts[1]


class TestAProcessThatStopped:
    """One timed-out turn must not take the provider with it.

    The process that let a turn run past its deadline keeps its half-read
    pipe.  While it stays in its slot, every later turn on that lane waits
    behind its lock — a first client gets a clean 502 and everyone after it
    hangs until the gateway restarts.
    """

    async def test_a_timed_out_process_is_replaced_for_the_next_turn(
        self, monkeypatch, tmp_path
    ):
        transport = AcpTransport(timeout=0.2)
        provider = _provider(monkeypatch, tmp_path, "stalls", conversation="conv-1")
        try:
            first = await asyncio.wait_for(
                transport.send(provider, "", _turn("[User]\none")), 5.0
            )
            second = await asyncio.wait_for(
                transport.send(provider, "", _turn("[User]\ntwo")), 5.0
            )
        finally:
            await transport.close()

        assert first.status_code == 502
        assert second.status_code == 502
        # Two processes, not one: the wedged one left its slot.
        starts = [entry["pid"] for entry in _log(tmp_path) if entry["kind"] == "start"]
        assert len(starts) == 2, starts
        assert starts[0] != starts[1]


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
        # The operator reads this sentence whole: the provider prefix, then
        # the agent's own words.
        assert (
            chunks[-1]["message"]
            == "ACP agent for provider 'antigravity': model call failed"
        )


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


class TestTheTurnDeadline:
    """An agent that stops answering must not hold the lane forever."""

    async def test_a_stalled_turn_ends_with_an_error_and_closes_the_session(
        self, monkeypatch, tmp_path
    ):
        transport = AcpTransport(timeout=0.2)
        provider = _provider(monkeypatch, tmp_path, "stalls", conversation="conv-1")
        try:
            stream = await transport.send_streaming(provider, "", _turn("[User]\none"))
            chunks = [chunk async for chunk in stream]
            held = transport.ledger.get(SessionKey("antigravity", "conv-1"))
        finally:
            await transport.close()

        assert chunks[-1]["type"] == "error"
        assert chunks[-1]["message"] == (
            "ACP agent for provider 'antigravity' did not finish the turn "
            "within 0.2s; the session was closed."
        )
        # A timed-out turn is not reusable: what the agent recorded is unknown.
        assert held is None
        assert "close" in _kinds(tmp_path)

    async def test_a_stalled_turn_is_a_failure_when_it_is_not_streamed(
        self, monkeypatch, tmp_path
    ):
        transport = AcpTransport(timeout=0.2)
        provider = _provider(monkeypatch, tmp_path, "stalls")
        try:
            resp = await transport.send(provider, "", _turn("[User]\none"))
        finally:
            await transport.close()

        assert resp.status_code == 502
        assert "did not finish the turn" in resp.error_text


class TestTheModelSwitch:
    """A request that names another model moves the session onto it."""

    async def test_a_new_session_is_switched_to_the_requested_model(
        self, monkeypatch, tmp_path
    ):
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "ok", conversation="conv-1")
        body = _turn("[User]\none")
        body["model"] = "gemini-3.1-flash"
        try:
            await transport.send(provider, "", body)
        finally:
            await transport.close()

        configs = [e for e in _log(tmp_path) if e["kind"] == "config"]
        assert configs == [
            {"kind": "config", "config": "model", "value": "gemini-3.1-flash"}
        ]

    async def test_a_reused_session_switches_when_the_model_changes(
        self, monkeypatch, tmp_path
    ):
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "ok", conversation="conv-1")
        try:
            await transport.send(provider, "", _turn("[User]\none"))
            body = _turn("[User]\none", "[Assistant]\ntwo")
            body["model"] = "gemini-3.1-flash"
            await transport.send(provider, "", body)
        finally:
            await transport.close()

        configs = [e for e in _log(tmp_path) if e["kind"] == "config"]
        assert configs[-1]["value"] == "gemini-3.1-flash"
        # The session was kept: only its model moved.
        assert _kinds(tmp_path).count("new") == 1

    async def test_a_turn_with_no_prompt_is_a_client_error(self, monkeypatch, tmp_path):
        transport = AcpTransport()
        provider = _provider(monkeypatch, tmp_path, "ok", conversation="conv-1")
        try:
            resp = await transport.send(provider, "", {"model": "m", "prompt": []})
        finally:
            await transport.close()

        assert resp.status_code == 400
        assert "carries no prompt" in resp.error_text


def _digests_for(messages: list[str]) -> tuple[str, ...]:
    from llm_rosetta_gateway.transport.acp_ledger import block_digests

    return block_digests({"type": "text", "text": m} for m in messages)


async def _spin() -> None:
    """Let the event loop run the other task as far as it can get."""
    import asyncio

    for _ in range(5):
        await asyncio.sleep(0)


async def _await_prompt(tmp_path: Any, timeout: float = 5.0) -> None:
    """Wait until the fake agent has received a prompt.

    At that instant the gateway is parked waiting for an update it will never
    receive, which is the state a cancellation test has to cancel from —
    sleeping a fixed interval would either race or slow the suite down.
    """
    loop = asyncio.get_running_loop()
    end = loop.time() + timeout
    while loop.time() < end:
        if any(entry["kind"] == "prompt" for entry in _log(tmp_path)):
            return
        await asyncio.sleep(0.02)
    raise AssertionError("the fake agent never received the prompt")


class TestAgentToolsAreOff:
    """Phase 1: the agent answers in text; every tool request is refused."""

    async def _turn_chunks(
        self, monkeypatch, tmp_path, mode, *, agy=False, wrapper=False
    ):
        transport = AcpTransport(timeout=5)
        try:
            stream = await transport.send_streaming(
                _provider(monkeypatch, tmp_path, mode, agy=agy, wrapper=wrapper),
                "",
                TURN,
            )
            return [chunk async for chunk in stream]
        finally:
            await transport.close()

    def _reply(self, tmp_path):
        return next(e for e in _log(tmp_path) if e["kind"] == "permission_reply")[
            "reply"
        ]

    async def test_a_permission_request_gets_the_reject_option_and_the_turn_ends(
        self, monkeypatch, tmp_path
    ):
        import logging

        records: list[logging.LogRecord] = []

        class _Keep(logging.Handler):
            def emit(self, record):
                records.append(record)

        gateway_log = logging.getLogger("llm-rosetta-gateway")
        handler = _Keep(level=logging.WARNING)
        gateway_log.addHandler(handler)
        try:
            chunks = await self._turn_chunks(monkeypatch, tmp_path, "asks")
        finally:
            gateway_log.removeHandler(handler)
        assert chunks[-1]["type"] == "stop"
        assert "text" in [c["type"] for c in chunks]
        assert self._reply(tmp_path)["result"] == {
            "outcome": {"outcome": "selected", "optionId": "r1"}
        }
        lines = [r.getMessage() for r in records if "refused" in r.getMessage()]
        assert lines == [
            "ACP provider 'antigravity': agent tool request refused "
            "(tool: Read /secret); agent tools are disabled"
        ]

    async def test_without_a_reject_option_the_outcome_is_cancelled(
        self, monkeypatch, tmp_path
    ):
        chunks = await self._turn_chunks(monkeypatch, tmp_path, "asks-allow-only")
        assert chunks[-1]["type"] == "stop"
        assert self._reply(tmp_path)["result"] == {"outcome": {"outcome": "cancelled"}}

    async def test_a_client_fs_request_is_an_error_not_a_file(
        self, monkeypatch, tmp_path
    ):
        chunks = await self._turn_chunks(monkeypatch, tmp_path, "asks-fs")
        assert chunks[-1]["type"] == "stop"
        reply = self._reply(tmp_path)
        assert "result" not in reply and "error" in reply

    async def test_initialize_advertises_no_fs_and_no_terminal(
        self, monkeypatch, tmp_path
    ):
        await self._turn_chunks(monkeypatch, tmp_path, "ok")
        params = next(e for e in _log(tmp_path) if e["kind"] == "initialize")["params"]
        # The wire form drops false flags, so an empty or absent block is
        # "nothing advertised".
        caps = params.get("clientCapabilities") or {}
        assert not caps.get("terminal")
        assert not any((caps.get("fs") or {}).values())

    async def test_every_new_session_is_put_in_the_asking_mode(
        self, monkeypatch, tmp_path
    ):
        await self._turn_chunks(monkeypatch, tmp_path, "ok", agy=True)
        modes = [e for e in _log(tmp_path) if e["kind"] == "mode"]
        assert [m["mode"] for m in modes] == ["plan"]
        kinds = _kinds(tmp_path)
        assert kinds.index("new") < kinds.index("mode") < kinds.index("prompt")

    async def test_an_agent_without_modes_runs_after_method_not_found(
        self, monkeypatch, tmp_path
    ):
        """A Claude-shaped agent advertises no modes and answers ``-32601``."""
        chunks = await self._turn_chunks(monkeypatch, tmp_path, "no-mode")
        kinds = _kinds(tmp_path)
        assert chunks[-1]["type"] == "stop"
        # The transport asked once, learned the agent has no modes, and left
        # the session as it started.
        assert "mode-attempt" in kinds
        assert "mode-unsupported" in kinds
        assert "mode" not in kinds

    async def test_an_agent_that_advertises_no_plan_keeps_its_mode(
        self, monkeypatch, tmp_path
    ):
        """An advertised mode list is authoritative: no probe, no override."""
        chunks = await self._turn_chunks(monkeypatch, tmp_path, "modes-without-plan")
        kinds = _kinds(tmp_path)
        assert chunks[-1]["type"] == "stop"
        assert "mode-attempt" not in kinds
        assert "mode" not in kinds

    async def test_an_agy_behind_a_wrapper_still_gets_the_asking_mode(
        self, monkeypatch, tmp_path
    ):
        """The mode is chosen from what the session advertises, not argv[0]."""
        chunks = await self._turn_chunks(monkeypatch, tmp_path, "ok", wrapper=True)
        assert chunks[-1]["type"] == "stop"
        assert [e.get("mode") for e in _log(tmp_path) if e["kind"] == "mode"] == [
            "plan"
        ]

    async def test_an_agy_that_advertises_no_modes_does_not_run(
        self, monkeypatch, tmp_path
    ):
        """An agy with no mode surface cannot be gated, so it must not run."""
        transport = AcpTransport(timeout=5)
        try:
            response = await transport.send(
                _provider(monkeypatch, tmp_path, "no-mode", agy=True), "", TURN
            )
            kinds = _kinds(tmp_path)
        finally:
            await transport.close()
        assert response.status_code == 502
        assert b"cannot be gated" in response.raw_content
        assert "mode-unsupported" in kinds
        assert "prompt" not in kinds

    async def test_a_session_whose_mode_cannot_be_set_does_not_run(
        self, monkeypatch, tmp_path
    ):
        transport = AcpTransport(timeout=5)
        try:
            response = await transport.send(
                _provider(monkeypatch, tmp_path, "mode-refused", agy=True), "", TURN
            )
            kinds = _kinds(tmp_path)
        finally:
            await transport.close()
        assert response.status_code == 502
        assert b"cannot be gated" in response.raw_content
        assert "prompt" not in kinds
        # The opened session is closed, not left behind for the process to hold.
        assert "close" in kinds

    async def test_a_failed_model_switch_releases_the_session_and_bridge(
        self, monkeypatch, tmp_path
    ):
        """A session that never becomes a turn must not keep its resources."""
        transport = AcpTransport(timeout=0.2)
        provider = _provider(
            monkeypatch, tmp_path, "config-stall", conversation="conv-config"
        )
        body = _turn("[User]\none")
        body["tools"] = [ECHO_TOOL]
        body["model"] = "gemini-3.1-flash"
        try:
            response = await transport.send(provider, "", body)
            bridges = dict(transport._session_bridges)
            urls = dict(transport._session_urls)
            registrations = dict(transport._mcp._registrations)
            port = transport._mcp.port
            kinds = _kinds(tmp_path)
        finally:
            await transport.close()

        assert response.status_code == 503
        assert bridges == {}
        assert urls == {}
        # The URL the agent received is no longer served, and the listener is
        # gone rather than left bound with an empty registration table.
        assert registrations == {}
        assert port == 0
        assert "prompt" not in kinds
        assert "close" in kinds

    def test_the_default_argv_turns_the_agent_auto_approve_off(self):
        from llm_rosetta_gateway.transport.acp import DEFAULT_AGENT_ARGV

        assert "--dangerously-skip-permissions=false" in DEFAULT_AGENT_ARGV

    def test_an_argv_override_is_kept(self):
        provider = ProviderInfo(
            "acp",
            config_name="x",
            api_key="",
            base_url="http://agy.local",
            auth_header_fn=openai_auth,
            url_template="{base_url}/acp-turn",
            keyless=True,
            transport="acp",
            acp_argv=("agy", "acp"),
            acp_argv_custom=True,
        )
        # Written by the operator: used as written, no profile, no flags.
        assert AcpTransport()._argv(provider) == ["agy", "acp"]


class TestTheAgentEnvironment:
    async def test_overrides_layer_over_the_gateway_environment(
        self, monkeypatch, tmp_path
    ):
        transport = AcpTransport(timeout=5)
        try:
            response = await transport.send(
                _provider(
                    monkeypatch,
                    tmp_path,
                    "ok",
                    acp_env={"FAKE_ACP_MARKER": "from-shim"},
                ),
                "",
                TURN,
            )
        finally:
            await transport.close()
        assert response.status_code == 200
        start = next(e for e in _log(tmp_path) if e["kind"] == "start")
        assert start["marker"] == "from-shim"
        # The process still sees the gateway's PATH, so a shim that declares
        # one variable does not strip the environment the agent needs.
        assert start["has_path"] is True


ECHO_TOOL = {
    "type": "function",
    "name": "echo",
    "description": "echo the text back",
    "parameters": {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    },
}

LOOKUP_TOOL = {
    "type": "function",
    "name": "lookup",
    "description": "look something up",
    "parameters": {
        "type": "object",
        "properties": {"query": {"type": "string"}},
        "required": ["query"],
    },
}


class TestClientTools:
    """A request's tools reach the agent, and its call comes back to the client."""

    async def test_a_bridged_call_suspends_and_the_result_resumes(
        self, monkeypatch, tmp_path
    ):
        transport = AcpTransport(timeout=10)
        provider = _provider(monkeypatch, tmp_path, "mcp", conversation="conv-tools")
        first = _turn("[User]\necho hello")
        first["tools"] = [ECHO_TOOL]
        first["tool_results"] = []
        try:
            stream = await transport.send_streaming(provider, "", first)
            assert stream.status_code == 200
            chunks = [chunk async for chunk in stream]

            assert [chunk["type"] for chunk in chunks] == [
                "session_start",
                "tool_call",
                "stop",
            ]
            call = chunks[1]
            assert call["tool_name"] == "echo"
            assert call["tool_input"] == {"text": "hello"}
            assert call["tool_call_id"].startswith("call_")
            assert chunks[2]["stop_reason"] == "tool_calls"
            assert transport._suspended  # the agent's prompt is still in flight

            tools = [e for e in _log(tmp_path) if e["kind"] == "mcp-tools"]
            assert tools and tools[0]["names"] == ["echo"]
            replies = [e for e in _log(tmp_path) if e["kind"] == "permission_reply"]
            assert replies and replies[0]["reply"]["result"]["outcome"] == {
                "outcome": "selected",
                "optionId": "a1",
            }

            second = _turn("[User]\necho hello")
            second["tools"] = [ECHO_TOOL]
            second["prompt"] = first["prompt"] + [
                {"type": "text", "text": "[Assistant]\n[called echo]"},
                {"type": "text", "text": "[Tool result]\nhello back"},
            ]
            second["tool_results"] = [
                {
                    "tool_call_id": call["tool_call_id"],
                    "result": "hello back",
                    "is_error": False,
                }
            ]
            response = await transport.send(provider, "", second)
            assert transport._suspended == {}
        finally:
            await transport.close()

        assert response.status_code == 200
        assert response.body is not None
        assert response.body["stop_reason"] == "end_turn"
        text = "".join(
            str(update.get("text") or "")
            for update in response.body["updates"]
            if update.get("type") == "text"
        )
        assert text == "tool said: hello back"
        # The same session continued; the whole conversation was not replayed.
        assert _kinds(tmp_path).count("new") == 1
        results = [e for e in _log(tmp_path) if e["kind"] == "mcp-result"]
        assert results and results[0]["text"] == "hello back"

    async def test_a_nameless_update_does_not_look_like_a_builtin_tool(
        self, monkeypatch, tmp_path
    ):
        """Claude Code names the call once, then sends input-only updates.

        The second frame has no name; without the call id it reads as a
        nameless built-in tool and the turn would stop before the MCP call.
        """
        transport = AcpTransport(timeout=10)
        provider = _provider(
            monkeypatch, tmp_path, "mcp-claude", conversation="conv-claude"
        )
        first = _turn("[User]\necho claude")
        first["tools"] = [ECHO_TOOL]
        try:
            stream = await transport.send_streaming(provider, "", first)
            assert stream.status_code == 200
            chunks = [chunk async for chunk in stream]
            assert [chunk["type"] for chunk in chunks] == [
                "session_start",
                "tool_call",
                "stop",
            ]
            call = chunks[1]
            assert call["tool_name"] == "echo"
            assert call["tool_input"] == {"text": "claude"}
            replies = [
                entry for entry in _log(tmp_path) if entry["kind"] == "permission_reply"
            ]
            assert replies and replies[-1]["reply"]["result"]["outcome"] == {
                "outcome": "selected",
                "optionId": "allow-once",
            }

            second = _turn("[User]\necho claude")
            second["tools"] = [ECHO_TOOL]
            second["prompt"] = first["prompt"] + [
                {"type": "text", "text": "[Assistant]\n[called echo]"},
                {"type": "text", "text": "[Tool result]\nclaude back"},
            ]
            second["tool_results"] = [
                {
                    "tool_call_id": call["tool_call_id"],
                    "result": "claude back",
                    "is_error": False,
                }
            ]
            response = await transport.send(provider, "", second)
        finally:
            await transport.close()

        assert response.status_code == 200
        assert response.body is not None
        assert response.body["stop_reason"] == "end_turn"
        text = "".join(
            str(update.get("text") or "")
            for update in response.body["updates"]
            if update.get("type") == "text"
        )
        assert text == "claude said: claude back"

    async def test_parallel_calls_resolved_in_one_request_are_not_asked_again(
        self, monkeypatch, tmp_path
    ):
        transport = AcpTransport(timeout=10)
        provider = _provider(
            monkeypatch, tmp_path, "mcp-parallel", conversation="conv-parallel"
        )
        first = _turn("[User]\necho two things")
        first["tools"] = [ECHO_TOOL]
        try:
            stream = await transport.send_streaming(provider, "", first)
            assert stream.status_code == 200
            chunks = [chunk async for chunk in stream]
            assert [chunk["type"] for chunk in chunks] == [
                "session_start",
                "tool_call",
                "stop",
            ]

            bridge = next(iter(transport._session_bridges.values()))
            calls: list[Any] = []
            for _ in range(400):
                calls = bridge.pending()
                if len(calls) == 2:
                    break
                await asyncio.sleep(0.005)
            assert len(calls) == 2

            second = _turn("[User]\necho two things")
            second["tools"] = [ECHO_TOOL]
            second["prompt"] = first["prompt"] + [
                {"type": "text", "text": "[Assistant]\n[called echo twice]"},
                {"type": "text", "text": "[Tool result]\none"},
                {"type": "text", "text": "[Tool result]\ntwo"},
            ]
            second["tool_results"] = [
                {
                    "tool_call_id": call.call_id,
                    "result": call.arguments["text"],
                    "is_error": False,
                }
                for call in calls
            ]
            response = await transport.send(provider, "", second)
        finally:
            await transport.close()

        assert response.status_code == 200
        assert response.body is not None
        assert response.body["stop_reason"] == "end_turn"
        # The second call was answered by this request, so it is not handed to
        # the client a second time.
        assert not any(
            update.get("type") == "tool_call" for update in response.body["updates"]
        )
        text = "".join(
            str(update.get("text") or "")
            for update in response.body["updates"]
            if update.get("type") == "text"
        )
        assert text == "tools said: one, two"

    async def test_a_one_off_request_cannot_declare_tools(self, monkeypatch, tmp_path):
        transport = AcpTransport(timeout=10)
        provider = _provider(monkeypatch, tmp_path, "mcp", conversation=None)
        body = _turn("[User]\necho hello")
        body["tools"] = [ECHO_TOOL]
        try:
            response = await transport.send(provider, "", body)
        finally:
            await transport.close()
        # A suspension is found by conversation id, so a one-off turn could
        # never carry the client's result back; the request is refused before
        # an agent starts.
        assert response.status_code == 400
        assert "declared conversation id" in response.error_text
        assert _kinds(tmp_path) == []

    async def test_two_processes_may_return_the_same_session_id(
        self, monkeypatch, tmp_path
    ):
        transport = AcpTransport(timeout=10)
        provider = _provider(monkeypatch, tmp_path, "mcp", processes=2)
        provider_a = provider.with_acp_conversation("conv-a")
        provider_b = provider.with_acp_conversation("conv-b")
        first_a = _turn("[User]\necho a")
        first_a["tools"] = [ECHO_TOOL]
        first_b = _turn("[User]\necho b")
        first_b["tools"] = [ECHO_TOOL]
        try:
            stream_a = await transport.send_streaming(provider_a, "", first_a)
            chunks_a = [chunk async for chunk in stream_a]
            stream_b = await transport.send_streaming(provider_b, "", first_b)
            chunks_b = [chunk async for chunk in stream_b]

            calls = [
                chunk for chunk in chunks_a + chunks_b if chunk["type"] == "tool_call"
            ]
            assert len(calls) == 2
            refs = set(transport._session_bridges)
            # Both agents use the same id, and each session keeps its own
            # bridge because the process slot is part of the key.
            assert len(refs) == 2
            assert {ref[1] for ref in refs} == {0, 1}
            assert {ref[2] for ref in refs} == {"sess-1"}

            responses = []
            for provider_x, call in (
                (provider_a, next(c for c in chunks_a if c["type"] == "tool_call")),
                (provider_b, next(c for c in chunks_b if c["type"] == "tool_call")),
            ):
                follow = _turn("[User]\necho again")
                follow["tools"] = [ECHO_TOOL]
                follow["tool_results"] = [
                    {
                        "tool_call_id": call["tool_call_id"],
                        "result": "back",
                        "is_error": False,
                    }
                ]
                responses.append(await transport.send(provider_x, "", follow))
            assert transport._suspended == {}
        finally:
            await transport.close()

        assert [response.status_code for response in responses] == [200, 200]
        for response in responses:
            assert response.body is not None
            assert response.body["stop_reason"] == "end_turn"

    async def test_two_conversations_get_separate_mcp_endpoints(
        self, monkeypatch, tmp_path
    ):
        """One session's tools must never be visible at another's URL."""
        transport = AcpTransport(timeout=10)
        provider = _provider(monkeypatch, tmp_path, "mcp-list")
        provider_a = provider.with_acp_conversation("conv-a")
        provider_b = provider.with_acp_conversation("conv-b")
        first_a = _turn("[User]\nlist")
        first_a["tools"] = [ECHO_TOOL]
        first_b = _turn("[User]\nlist")
        first_b["tools"] = [LOOKUP_TOOL]
        try:
            response_a = await transport.send(provider_a, "", first_a)
            response_b = await transport.send(provider_b, "", first_b)
            urls = dict(transport._session_urls)
            assert len(urls) == 2
            assert len(set(urls.values())) == 2
            served = sorted(
                [tuple(await _mcp_tool_names(url)) for url in urls.values()]
            )
        finally:
            await transport.close()

        assert response_a.status_code == 200
        assert response_b.status_code == 200
        assert response_a.body is not None and response_b.body is not None
        # Two sessions served by one process still have distinct endpoints.
        assert response_a.body["session_id"] != response_b.body["session_id"]
        assert served == [("echo",), ("lookup",)]


class TestToolChanges:
    """A request's tool set is a session fact, not a turn-local setting."""

    async def _two_turns(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Any,
        first_tools: list[dict[str, Any]],
        second_tools: list[dict[str, Any]],
    ) -> None:
        transport = AcpTransport(timeout=10)
        provider = _provider(
            monkeypatch, tmp_path, "mcp-list", conversation="conv-tools"
        )
        first = _turn("[User]\nhi")
        first["tools"] = first_tools
        second = _turn("[User]\nhi", "[Assistant]\nhello")
        second["tools"] = second_tools
        try:
            first_response = await transport.send(provider, "", first)
            second_response = await transport.send(provider, "", second)
        finally:
            await transport.close()
        assert first_response.status_code == 200
        assert second_response.status_code == 200

    @staticmethod
    def _tool_lists(tmp_path: Any) -> list[list[str]]:
        return [
            entry["names"] for entry in _log(tmp_path) if entry["kind"] == "mcp-tools"
        ]

    async def test_the_same_tools_reuse_the_session(self, monkeypatch, tmp_path):
        await self._two_turns(monkeypatch, tmp_path, [ECHO_TOOL], [ECHO_TOOL])

        assert _kinds(tmp_path).count("new") == 1
        assert _kinds(tmp_path).count("prompt") == 2
        assert self._tool_lists(tmp_path) == [["echo"], ["echo"]]

    async def test_a_tool_session_without_a_bridge_is_replayed(
        self, monkeypatch, tmp_path
    ):
        """A session that cannot serve its tools must not continue without one."""
        transport = AcpTransport(timeout=10)
        provider = _provider(
            monkeypatch, tmp_path, "mcp-list", conversation="conv-tools"
        )
        first = _turn("[User]\nhi")
        first["tools"] = [ECHO_TOOL]
        second = _turn("[User]\nhi", "[Assistant]\nhello")
        second["tools"] = [ECHO_TOOL]
        try:
            first_response = await transport.send(provider, "", first)
            # Simulate a half-resumed session: the ledger entry survives but
            # the bridge that answers its URL does not.
            transport._session_bridges.clear()
            second_response = await transport.send(provider, "", second)
            bridges = dict(transport._session_bridges)
        finally:
            await transport.close()

        assert first_response.status_code == 200
        assert second_response.status_code == 200
        assert first_response.body is not None and second_response.body is not None
        assert second_response.body["session_id"] != first_response.body["session_id"]
        assert _kinds(tmp_path).count("new") == 2
        assert _kinds(tmp_path).count("prompt") == 2
        urls = [entry["url"] for entry in _log(tmp_path) if entry["kind"] == "mcp"]
        assert len(urls) == 2
        assert urls[0] != urls[1]
        assert len(bridges) == 1

    async def test_reordering_the_same_tools_reuses_the_session(
        self, monkeypatch, tmp_path
    ):
        """A tool set is a set: another order is not a new session."""
        await self._two_turns(
            monkeypatch, tmp_path, [ECHO_TOOL, LOOKUP_TOOL], [LOOKUP_TOOL, ECHO_TOOL]
        )

        assert _kinds(tmp_path).count("new") == 1
        assert _kinds(tmp_path).count("prompt") == 2
        # The first session keeps the order it opened with.
        assert self._tool_lists(tmp_path) == [
            ["echo", "lookup"],
            ["echo", "lookup"],
        ]

    async def test_a_new_tool_opens_a_new_session(self, monkeypatch, tmp_path):
        await self._two_turns(
            monkeypatch, tmp_path, [ECHO_TOOL], [ECHO_TOOL, LOOKUP_TOOL]
        )

        assert _kinds(tmp_path).count("new") == 2
        assert self._tool_lists(tmp_path) == [["echo"], ["echo", "lookup"]]

    async def test_a_removed_tool_opens_a_new_session(self, monkeypatch, tmp_path):
        await self._two_turns(
            monkeypatch, tmp_path, [ECHO_TOOL, LOOKUP_TOOL], [ECHO_TOOL]
        )

        assert _kinds(tmp_path).count("new") == 2
        assert self._tool_lists(tmp_path) == [["echo", "lookup"], ["echo"]]

    async def test_removing_every_tool_opens_a_session_without_mcp(
        self, monkeypatch, tmp_path
    ):
        await self._two_turns(monkeypatch, tmp_path, [ECHO_TOOL], [])

        assert _kinds(tmp_path).count("new") == 2
        assert self._tool_lists(tmp_path) == [["echo"], []]

    async def test_a_refreshed_schema_opens_a_new_session(self, monkeypatch, tmp_path):
        refreshed = dict(ECHO_TOOL, description="echo back, now with a new contract")
        await self._two_turns(monkeypatch, tmp_path, [ECHO_TOOL], [refreshed])

        assert _kinds(tmp_path).count("new") == 2
        assert self._tool_lists(tmp_path) == [["echo"], ["echo"]]

    async def test_a_result_request_with_changed_tools_replays_instead(
        self, monkeypatch, tmp_path
    ):
        """The old bridge must not stay open when the request drops its tools."""
        transport = AcpTransport(timeout=10)
        provider = _provider(
            monkeypatch, tmp_path, "mcp-change", conversation="conv-change"
        )
        first = _turn("[User]\necho hello")
        first["tools"] = [ECHO_TOOL]
        try:
            stream = await transport.send_streaming(provider, "", first)
            assert stream.status_code == 200
            chunks = [chunk async for chunk in stream]
            call = next(chunk for chunk in chunks if chunk["type"] == "tool_call")

            second = _turn(
                "[User]\necho hello",
                "[Assistant]\n[called echo]",
                "[Tool result]\nhello back",
            )
            second["tools"] = [LOOKUP_TOOL]
            second["tool_results"] = [
                {
                    "tool_call_id": call["tool_call_id"],
                    "result": "hello back",
                    "is_error": False,
                }
            ]
            response = await transport.send(provider, "", second)
        finally:
            await transport.close()

        assert response.status_code == 200
        assert response.body is not None
        assert response.body["stop_reason"] == "end_turn"
        assert transport._suspended == {}
        # The old session is abandoned and the whole history is replayed into
        # a session whose MCP endpoint lists only the new tool.
        assert _kinds(tmp_path).count("new") == 2
        assert self._tool_lists(tmp_path) == [["echo"], ["lookup"]]
        text = "".join(
            str(update.get("text") or "")
            for update in response.body["updates"]
            if update.get("type") == "text"
        )
        assert "resumed with tools: lookup" in text

    async def test_a_result_request_with_reordered_tools_continues(
        self, monkeypatch, tmp_path
    ):
        """An order-only change must not abandon a pending call."""
        transport = AcpTransport(timeout=10)
        provider = _provider(
            monkeypatch, tmp_path, "mcp-change", conversation="conv-order"
        )
        first = _turn("[User]\necho hello")
        first["tools"] = [ECHO_TOOL, LOOKUP_TOOL]
        try:
            stream = await transport.send_streaming(provider, "", first)
            assert stream.status_code == 200
            chunks = [chunk async for chunk in stream]
            call = next(chunk for chunk in chunks if chunk["type"] == "tool_call")

            second = _turn(
                "[User]\necho hello",
                "[Assistant]\n[called echo]",
                "[Tool result]\nhello back",
            )
            second["tools"] = [LOOKUP_TOOL, ECHO_TOOL]
            second["tool_results"] = [
                {
                    "tool_call_id": call["tool_call_id"],
                    "result": "hello back",
                    "is_error": False,
                }
            ]
            response = await transport.send(provider, "", second)
        finally:
            await transport.close()

        assert response.status_code == 200
        assert response.body is not None
        assert response.body["stop_reason"] == "end_turn"
        assert transport._suspended == {}
        # The suspended turn continued: it was not replayed into a new session.
        assert _kinds(tmp_path).count("new") == 1
        assert self._tool_lists(tmp_path) == [["echo", "lookup"]]
        text = "".join(
            str(update.get("text") or "")
            for update in response.body["updates"]
            if update.get("type") == "text"
        )
        assert "tool said: hello back" in text


class TestAgentLaneConformance:
    """#1629: client tools work; the agent's host tools never do."""

    async def test_a_bridged_tool_and_the_two_host_tools(self, monkeypatch, tmp_path):
        """One turn proves the whole rule, not three unrelated paths."""
        transport = AcpTransport(timeout=10)
        provider = _provider(
            monkeypatch, tmp_path, "conformance", conversation="conv-conformance"
        )
        first = _turn("[User]\necho hello")
        first["tools"] = [ECHO_TOOL]
        try:
            stream = await transport.send_streaming(provider, "", first)
            assert stream.status_code == 200
            chunks = [chunk async for chunk in stream]
            call = next(chunk for chunk in chunks if chunk["type"] == "tool_call")
            assert call["tool_name"] == "echo"
            assert call["tool_input"] == {"text": "hello"}

            second = _turn(
                "[User]\necho hello",
                "[Assistant]\n[called echo]",
                "[Tool result]\nhello back",
            )
            second["tools"] = [ECHO_TOOL]
            second["tool_results"] = [
                {
                    "tool_call_id": call["tool_call_id"],
                    "result": "hello back",
                    "is_error": False,
                }
            ]
            response = await transport.send(provider, "", second)
        finally:
            await transport.close()

        assert response.status_code == 200
        assert response.body is not None
        assert response.body["stop_reason"] == "end_turn"
        text = "".join(
            str(update.get("text") or "")
            for update in response.body["updates"]
            if update.get("type") == "text"
        )
        assert "bridged: hello back" in text
        replies = [
            entry["reply"]
            for entry in _log(tmp_path)
            if entry["kind"] == "permission_reply"
        ]
        # The bridged call was allowed; the host shell was refused.
        assert replies[0]["result"]["outcome"] == {
            "outcome": "selected",
            "optionId": "a1",
        }
        assert replies[-1]["result"]["outcome"] == {
            "outcome": "selected",
            "optionId": "r1",
        }
        fs_reply = next(
            entry["reply"] for entry in _log(tmp_path) if entry["kind"] == "fs_reply"
        )
        assert "error" in fs_reply and "result" not in fs_reply
        assert not (tmp_path / "shell-side-effect").exists()
