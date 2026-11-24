"""The ``claude_cli`` transport, driven against a fake CLI.

No Claude install, no account and no network are needed: the "CLI" is a small
Python script that reads the gateway's NDJSON frames on stdin and writes the
events the real CLI writes on stdout.  What is under test is this gateway's
half — the frames it builds, the argv that keeps the CLI a model client, the
event mapping, the non-streaming fold, the error path and process cleanup.

The fake CLI appends what it saw to ``$FAKE_CLI_LOG`` as one JSON object per
line, so a test asserts the exact frames and argv the gateway sent rather than
inferring them from the reply.
"""

from __future__ import annotations

import json
import sys
import textwrap
import time
from pathlib import Path

import pytest

from llm_rosetta_gateway.transport import (
    ClaudeCliTransport,
    UpstreamConnectionError,
    UpstreamTimeoutError,
)
from llm_rosetta_gateway.transport.claude_cli import (
    build_frames,
    conversation_session_id,
)
from llm_rosetta_gateway.transport.provider_info import ProviderInfo, openai_auth

#: A minimal stand-in for ``claude``: it reads every frame, notes what it saw,
#: then answers according to ``FAKE_CLI_MODE``.
FAKE_CLI = textwrap.dedent(
    """
    import json, os, sys, time

    MODE = os.environ.get("FAKE_CLI_MODE", "stream")
    LOG = os.environ.get("FAKE_CLI_LOG", "")

    def note(kind, **fields):
        if not LOG:
            return
        with open(LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": kind, **fields}) + "\\n")

    frames = []
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        frames.append(json.loads(line))

    system_path = None
    argv = sys.argv[1:]
    if "--system-prompt-file" in argv:
        system_path = argv[argv.index("--system-prompt-file") + 1]
    system = ""
    if system_path:
        with open(system_path, encoding="utf-8") as handle:
            system = handle.read()

    note("start", argv=argv, frames=frames, system=system,
         removed=os.environ.get("FAKE_CLI_REMOVED"),
         kept=os.environ.get("FAKE_CLI_KEPT"),
         pid=os.getpid())

    def send(payload):
        sys.stdout.write(json.dumps(payload) + "\\n")
        sys.stdout.flush()

    ASSISTANT = {
        "id": "msg_test", "type": "message", "role": "assistant",
        "model": "claude-test",
        "content": [{"type": "text", "text": "Hello!"}],
        "stop_reason": None, "usage": {"input_tokens": 11, "output_tokens": 0},
    }
    RESULT = {
        "type": "result", "subtype": "success", "is_error": False,
        "num_turns": 1, "stop_reason": "end_turn",
        "usage": {"input_tokens": 11, "output_tokens": 3},
        "result": "Hello!",
    }

    if MODE == "hang":
        time.sleep(60)
        sys.exit(0)

    if MODE == "stream":
        send({"type": "stream_event", "event": {
            "type": "message_start",
            "message": {"id": "msg_test", "type": "message",
                        "role": "assistant", "model": "claude-test",
                        "content": [], "stop_reason": None,
                        "usage": {"input_tokens": 11, "output_tokens": 0}}}})
        send({"type": "stream_event", "event": {
            "type": "content_block_start", "index": 0,
            "content_block": {"type": "text", "text": ""}}})
        send({"type": "stream_event", "event": {
            "type": "content_block_delta", "index": 0,
            "delta": {"type": "text_delta", "text": "Hello"}}})
        send({"type": "stream_event", "event": {
            "type": "content_block_stop", "index": 0}})
        send({"type": "stream_event", "event": {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": 3}}})
        send({"type": "stream_event", "event": {"type": "message_stop"}})
        send({"type": "assistant", "message": ASSISTANT})
        send(RESULT)
    elif MODE == "whole":
        send({"type": "assistant", "message": ASSISTANT})
        send(RESULT)
    elif MODE == "blocks":
        send({"type": "assistant", "message": {
            "id": "msg_blocks", "type": "message", "role": "assistant",
            "model": "claude-test",
            "content": [
                {"type": "thinking", "thinking": "let me look"},
                {"type": "text", "text": "Calling."},
                {"type": "tool_use", "id": "toolu_9", "name": "lookup",
                 "input": {"q": "x"}},
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 5, "output_tokens": 7},
        }})
        send({"type": "result", "subtype": "success", "is_error": False,
              "num_turns": 1, "stop_reason": "tool_use",
              "usage": {"input_tokens": 5, "output_tokens": 7}})
    elif MODE == "error":
        send({"type": "result", "subtype": "error_during_execution",
              "is_error": True, "num_turns": 1,
              "errors": ["You've hit your 5-hour limit"]})
    """
)


def _write_cli(tmp_path: Path) -> tuple[str, ...]:
    """Write the fake CLI and return the argv that starts it."""
    script = tmp_path / "fake_claude.py"
    script.write_text(FAKE_CLI, encoding="utf-8")
    return (sys.executable, str(script))


def _provider(
    argv: tuple[str, ...],
    *,
    env: dict[str, str] | None = None,
    unset: tuple[str, ...] = (),
    args: tuple[str, ...] = (),
) -> ProviderInfo:
    """A keyless lane whose CLI is the fake one."""
    return ProviderInfo(
        "claude_cli",
        config_name="claude",
        api_key="",
        base_url="http://claude.local",
        auth_header_fn=openai_auth,
        url_template="{base_url}/v1/messages",
        keyless=True,
        transport="claude_cli",
        cli_argv=argv,
        cli_env=env,
        cli_env_unset=unset,
        cli_args=args,
    )


def _log(path: Path) -> list[dict]:
    """Every line the fake CLI noted."""
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _turn() -> dict:
    """A three-turn conversation, so replay frames are exercised."""
    return {
        "model": "claude-test",
        "max_tokens": 1024,
        "system": [{"type": "text", "text": "You are terse."}],
        "messages": [
            {"role": "user", "content": "name a fruit"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me look."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "lookup",
                        "input": {"q": "fruit"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": [{"type": "text", "text": "apple"}],
                    },
                    {"type": "text", "text": "now answer"},
                ],
            },
        ],
    }


class TestFrames:
    """The Messages body → CLI frames mapping."""

    def test_system_blocks_are_joined(self) -> None:
        system, _frames = build_frames(
            {
                "system": [
                    {"type": "text", "text": "one"},
                    {"type": "text", "text": "two"},
                ],
                "messages": [{"role": "user", "content": "hi"}],
            }
        )

        assert system == "one\n\ntwo"

    def test_roles_and_tool_ids_survive(self) -> None:
        _system, frames = build_frames(_turn())

        assert [frame["type"] for frame in frames] == ["user", "assistant", "user"]
        call = frames[1]["message"]["content"][1]
        assert call["type"] == "tool_use"
        assert call["id"] == "toolu_1"
        result = frames[2]["message"]["content"][0]
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "toolu_1"

    def test_history_frames_do_not_query(self) -> None:
        _system, frames = build_frames(_turn())

        assert frames[0]["shouldQuery"] is False
        assert frames[1]["shouldQuery"] is False
        assert "shouldQuery" not in frames[2]

    def test_assistant_prefill_is_refused(self) -> None:
        with pytest.raises(UpstreamConnectionError, match="assistant prefill"):
            build_frames(
                {
                    "messages": [
                        {"role": "user", "content": "hi"},
                        {"role": "assistant", "content": "sure"},
                    ]
                }
            )

    def test_session_id_is_stable_for_the_same_start(self) -> None:
        first = build_frames(_turn())
        second = build_frames(_turn())

        assert conversation_session_id(*first) == conversation_session_id(*second)

    def test_a_different_conversation_gets_another_session(self) -> None:
        first = conversation_session_id(*build_frames(_turn()))
        other = build_frames(
            {"messages": [{"role": "user", "content": "a different question"}]}
        )

        assert conversation_session_id(*other) != first


class TestTurn:
    """Driving one CLI process."""

    async def test_stream_events_are_forwarded(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FAKE_CLI_MODE", "stream")
        monkeypatch.setenv("FAKE_CLI_LOG", str(tmp_path / "cli.jsonl"))
        transport = ClaudeCliTransport(timeout=30)

        stream = await transport.send_streaming(
            _provider(_write_cli(tmp_path)), "", _turn()
        )
        try:
            events = [event async for event in stream]
        finally:
            await stream.close()
            await transport.close()

        kinds = [event["type"] for event in events]
        assert kinds == [
            "message_start",
            "content_block_start",
            "content_block_delta",
            "content_block_stop",
            "message_delta",
            "message_stop",
        ]
        assert events[2]["delta"] == {"type": "text_delta", "text": "Hello"}

    async def test_argv_keeps_the_cli_a_model_client(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FAKE_CLI_MODE", "stream")
        log = tmp_path / "cli.jsonl"
        monkeypatch.setenv("FAKE_CLI_LOG", str(log))
        transport = ClaudeCliTransport(timeout=30)

        stream = await transport.send_streaming(
            _provider(_write_cli(tmp_path), args=("--setting-sources", "user")),
            "",
            _turn(),
        )
        try:
            [event async for event in stream]
        finally:
            await stream.close()
            await transport.close()

        start = _log(log)[0]
        argv = start["argv"]
        assert argv[0] == "-p"
        assert argv[argv.index("--input-format") + 1] == "stream-json"
        assert argv[argv.index("--output-format") + 1] == "stream-json"
        assert argv[argv.index("--tools") + 1] == ""
        assert argv[argv.index("--model") + 1] == "claude-test"
        assert argv[-2:] == ["--setting-sources", "user"]
        assert "--include-partial-messages" in argv
        assert "--no-session-persistence" in argv
        assert start["system"] == "You are terse."

    async def test_the_conversation_arrives_as_frames(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FAKE_CLI_MODE", "stream")
        log = tmp_path / "cli.jsonl"
        monkeypatch.setenv("FAKE_CLI_LOG", str(log))
        transport = ClaudeCliTransport(timeout=30)

        stream = await transport.send_streaming(
            _provider(_write_cli(tmp_path)), "", _turn()
        )
        try:
            [event async for event in stream]
        finally:
            await stream.close()
            await transport.close()

        frames = _log(log)[0]["frames"]
        assert [frame["type"] for frame in frames] == ["user", "assistant", "user"]
        assert frames[0]["message"]["content"] == [
            {"type": "text", "text": "name a fruit"}
        ]

    async def test_a_whole_message_is_synthesised_into_events(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("FAKE_CLI_MODE", "whole")
        monkeypatch.setenv("FAKE_CLI_LOG", str(tmp_path / "cli.jsonl"))
        transport = ClaudeCliTransport(timeout=30)

        stream = await transport.send_streaming(
            _provider(_write_cli(tmp_path)), "", _turn()
        )
        try:
            events = [event async for event in stream]
        finally:
            await stream.close()
            await transport.close()

        kinds = [event["type"] for event in events]
        assert kinds[0] == "message_start"
        assert "content_block_delta" in kinds
        assert kinds[-2:] == ["message_delta", "message_stop"]
        delta = next(
            event for event in events if event["type"] == "content_block_delta"
        )
        assert delta["delta"]["text"] == "Hello!"

    async def test_thinking_and_tool_use_are_synthesised(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FAKE_CLI_MODE", "blocks")
        monkeypatch.setenv("FAKE_CLI_LOG", str(tmp_path / "cli.jsonl"))
        transport = ClaudeCliTransport(timeout=30)

        stream = await transport.send_streaming(
            _provider(_write_cli(tmp_path)), "", _turn()
        )
        try:
            events = [event async for event in stream]
        finally:
            await stream.close()
            await transport.close()

        starts = [
            event["content_block"]["type"]
            for event in events
            if event["type"] == "content_block_start"
        ]
        assert starts == ["thinking", "text", "tool_use"]
        tool = next(
            event["content_block"]
            for event in events
            if event["type"] == "content_block_start"
            and event["content_block"]["type"] == "tool_use"
        )
        assert tool["id"] == "toolu_9"
        assert tool["name"] == "lookup"
        delta = next(
            event
            for event in events
            if event["type"] == "content_block_delta"
            and event["delta"]["type"] == "input_json_delta"
        )
        assert json.loads(delta["delta"]["partial_json"]) == {"q": "x"}

    async def test_non_streaming_folds_the_turn(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FAKE_CLI_MODE", "whole")
        monkeypatch.setenv("FAKE_CLI_LOG", str(tmp_path / "cli.jsonl"))
        transport = ClaudeCliTransport(timeout=30)

        response = await transport.send(_provider(_write_cli(tmp_path)), "", _turn())
        await transport.close()

        assert response.status_code == 200
        assert response.body is not None
        assert response.body["content"] == [{"type": "text", "text": "Hello!"}]
        assert response.body["stop_reason"] == "end_turn"
        assert response.body["usage"]["input_tokens"] == 11
        assert response.body["usage"]["output_tokens"] == 3

    async def test_an_error_result_raises(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FAKE_CLI_MODE", "error")
        monkeypatch.setenv("FAKE_CLI_LOG", str(tmp_path / "cli.jsonl"))
        transport = ClaudeCliTransport(timeout=30)

        stream = await transport.send_streaming(
            _provider(_write_cli(tmp_path)), "", _turn()
        )
        try:
            with pytest.raises(UpstreamConnectionError, match="5-hour limit"):
                [event async for event in stream]
        finally:
            await stream.close()
            await transport.close()

    async def test_close_stops_a_hung_turn(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FAKE_CLI_MODE", "hang")
        monkeypatch.setenv("FAKE_CLI_LOG", str(tmp_path / "cli.jsonl"))
        transport = ClaudeCliTransport(timeout=30)

        stream = await transport.send_streaming(
            _provider(_write_cli(tmp_path)), "", _turn()
        )
        started = time.monotonic()
        await stream.close()
        await transport.close()

        assert time.monotonic() - started < 10

    async def test_silence_times_out(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FAKE_CLI_MODE", "hang")
        monkeypatch.setenv("FAKE_CLI_LOG", str(tmp_path / "cli.jsonl"))
        transport = ClaudeCliTransport(timeout=0.2)

        stream = await transport.send_streaming(
            _provider(_write_cli(tmp_path)), "", _turn()
        )
        try:
            with pytest.raises(UpstreamTimeoutError, match="silent"):
                [event async for event in stream]
        finally:
            await stream.close()
            await transport.close()

    async def test_env_overrides_and_removals_reach_the_child(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("FAKE_CLI_MODE", "stream")
        log = tmp_path / "cli.jsonl"
        monkeypatch.setenv("FAKE_CLI_LOG", str(log))
        monkeypatch.setenv("FAKE_CLI_REMOVED", "inherited-secret")
        provider = _provider(
            _write_cli(tmp_path),
            env={"FAKE_CLI_KEPT": "yes"},
            unset=("FAKE_CLI_REMOVED",),
        )
        transport = ClaudeCliTransport(timeout=30)

        stream = await transport.send_streaming(provider, "", _turn())
        try:
            [event async for event in stream]
        finally:
            await stream.close()
            await transport.close()

        start = _log(log)[0]
        assert start["removed"] is None
        assert start["kept"] == "yes"

    async def test_a_missing_cli_names_the_command(self, tmp_path):
        transport = ClaudeCliTransport(timeout=5)
        provider = _provider((str(tmp_path / "does-not-exist"),))

        with pytest.raises(UpstreamConnectionError, match="cannot start"):
            await transport.send_streaming(provider, "", _turn())
