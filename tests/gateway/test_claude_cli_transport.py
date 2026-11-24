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

import asyncio
import copy
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
    make_restorer,
)
from llm_rosetta_gateway.transport.claude_cli_relay import Relay, RelayError
from llm_rosetta_gateway.transport.provider_info import ProviderInfo, openai_auth

#: A minimal stand-in for ``claude``: it reads every frame, notes what it saw,
#: then answers according to ``FAKE_CLI_MODE``.
FAKE_CLI = textwrap.dedent(
    """
    import json, os, sys, time, urllib.error, urllib.request

    MODE = os.environ.get("FAKE_CLI_MODE", "stream")
    LOG = os.environ.get("FAKE_CLI_LOG", "")

    def note(kind, **fields):
        if not LOG:
            return
        with open(LOG, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"kind": kind, **fields}) + "\\n")

    def send(payload):
        sys.stdout.write(json.dumps(payload) + "\\n")
        sys.stdout.flush()

    frames = []

    def ack():
        send({"type": "result", "subtype": "success", "is_error": False,
              "num_turns": 0})

    def take(frame):
        \"\"\"Note one frame; answer a history turn.  True on the queried frame.\"\"\"
        frames.append(frame)
        note("frame", index=len(frames) - 1, at=time.monotonic())
        if frame.get("shouldQuery") is False:
            # The real CLI acknowledges a queued *user* frame and says nothing
            # about an appended assistant frame.
            if frame.get("type") == "user":
                ack()
                note("ack", index=len(frames) - 1, at=time.monotonic())
            return False
        return True

    if MODE == "chunks":
        # Read the raw pipe, one read at a time.  A transport that writes the
        # whole history at once puts several frames in a single read; one that
        # waits for the acknowledgment puts exactly one there.  The sleep lets
        # a bulk-writing transport finish before the first read.
        time.sleep(0.3)
        done = False
        while not done:
            data = os.read(0, 65536)
            if not data:
                break
            lines = [line for line in data.decode("utf-8").splitlines() if line.strip()]
            note("chunk", lines=len(lines))
            for line in lines:
                if take(json.loads(line)):
                    done = True
                    break
    else:
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            if take(json.loads(line)):
                break

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

    if MODE in ("relay-once", "relay-twice"):
        base = os.environ.get("ANTHROPIC_BASE_URL", "")
        # The CLI rebuilds the conversation from the frames it replayed and
        # appends its own annotation message.
        payload = json.dumps({
            "model": "claude-test",
            "messages": [frame["message"] for frame in frames]
            + [{"role": "system", "content": "# Environment\\n/tmp/cli"}],
        }).encode()
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        statuses = []
        attempts = 1 if MODE == "relay-once" else 2
        for _ in range(attempts):
            request = urllib.request.Request(
                base + "/v1/messages",
                data=payload,
                headers={"content-type": "application/json",
                         "authorization": "Bearer test-token",
                         "anthropic-version": "2023-06-01",
                         "anthropic-ratelimit-unified-status": "allowed"},
            )
            try:
                with opener.open(request, timeout=10) as response:
                    statuses.append(response.status)
                    response.read()
            except urllib.error.HTTPError as exc:
                statuses.append(exc.code)
                exc.read()
        note("relay", statuses=statuses, base=base)
        send({"type": "assistant", "message": ASSISTANT})
        send(RESULT)
        sys.exit(0)

    if MODE in ("stream", "chunks"):
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
    upstream: str | None = None,
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
        cli_upstream=upstream,
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


def _start(path: Path) -> dict:
    """The fake CLI's ``start`` note: its argv, the frames and the system text."""
    return next(entry for entry in _log(path) if entry["kind"] == "start")


class FakeUpstream:
    """A scripted Anthropic-shaped upstream that records what it received."""

    def __init__(self, responses: list[tuple[int, bytes]]) -> None:
        self._responses = list(responses)
        self.requests: list[dict] = []
        self._server: asyncio.AbstractServer | None = None

    async def start(self) -> str:
        """Bind a loopback port and return the URL the relay should use."""
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        socket = (self._server.sockets or [None])[0]
        assert socket is not None
        return f"http://127.0.0.1:{socket.getsockname()[1]}"

    async def close(self) -> None:
        """Stop serving."""
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            await reader.readline()
            headers: dict[str, str] = {}
            while True:
                line = await reader.readline()
                if not line or line in (b"\r\n", b"\n"):
                    break
                name, _, value = line.decode("latin-1").rstrip("\r\n").partition(":")
                headers[name.strip().lower()] = value.strip()
            body = b""
            length = headers.get("content-length")
            if length:
                body = await reader.readexactly(int(length))
            self.requests.append({"headers": headers, "body": body})
            status, payload = (
                self._responses.pop(0) if self._responses else (200, b'{"ok":true}')
            )
            head = (
                f"HTTP/1.1 {status} Scripted\r\n"
                "content-type: application/json\r\n"
                f"content-length: {len(payload)}\r\n"
                "anthropic-ratelimit-unified-status: allowed\r\n\r\n"
            )
            writer.write(head.encode("latin-1") + payload)
            await writer.drain()
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass


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


class TestRestore:
    """Putting the host's conversation back over the CLI's annotations."""

    @staticmethod
    def _cli_body(host: dict) -> dict:
        """The conversation as the CLI rebuilds it: an annotation per user turn."""
        messages: list[dict] = []
        for message in copy.deepcopy(host["messages"]):
            # The CLI rewrites string content as a text block.
            if isinstance(message["content"], str):
                message["content"] = [{"type": "text", "text": message["content"]}]
            messages.append(message)
            if message["role"] == "user":
                messages.append(
                    {"role": "system", "content": "# Environment\n/tmp/claude-cli-x"}
                )
        return {"model": "claude-test", "messages": messages}

    @staticmethod
    def _markers(body: dict) -> int:
        total = sum(
            1
            for block in body.get("system") or []
            if isinstance(block, dict) and block.get("cache_control")
        )
        for message in body["messages"]:
            content = message.get("content")
            if isinstance(content, list):
                total += sum(
                    1
                    for block in content
                    if isinstance(block, dict) and block.get("cache_control")
                )
        return total

    def test_the_annotation_messages_are_dropped(self) -> None:
        host = _turn()
        restore = make_restorer(host)

        body = restore(self._cli_body(host))

        assert [message["role"] for message in body["messages"]] == [
            "user",
            "assistant",
            "user",
        ]
        assert "# Environment" not in json.dumps(body)

    def test_a_reshaped_history_is_forwarded_unchanged(self) -> None:
        host = _turn()
        restore = make_restorer(host)
        cli_body = self._cli_body(host)
        # The CLI dropped a message (or merged two): the rewrite cannot put the
        # host's conversation back, so the request must go out as it is.
        cli_body["messages"] = [
            message
            for message in cli_body["messages"]
            if message["role"] != "assistant"
        ]

        assert restore(cli_body) is cli_body

    def test_the_breakpoint_moves_to_the_last_block(self) -> None:
        host = _turn()
        restore = make_restorer(host)

        body = restore(self._cli_body(host))

        assert body["messages"][-1]["content"][-1]["cache_control"] == {
            "type": "ephemeral"
        }

    def test_an_existing_breakpoint_is_left_alone(self) -> None:
        host = _turn()
        host["messages"][-1]["content"][-1]["cache_control"] = {"type": "ephemeral"}
        restore = make_restorer(host)

        body = restore(self._cli_body(host))

        assert self._markers(body) == 1

    def test_the_breakpoint_budget_is_four(self) -> None:
        host = _turn()
        restore = make_restorer(host)
        cli_body = self._cli_body(host)
        # Four markers, the API's limit, so the rewrite has to make room.
        for index, block in ((0, 0), (2, 0), (2, 1), (3, 0)):
            cli_body["messages"][index]["content"][block]["cache_control"] = {
                "type": "ephemeral"
            }

        body = restore(cli_body)

        assert self._markers(body) == 4
        # The oldest message-level marker made room for the new one.
        assert "cache_control" not in body["messages"][0]["content"][0]
        assert body["messages"][-1]["content"][-1]["cache_control"] == {
            "type": "ephemeral"
        }


class TestRelay:
    """The loopback relay between the CLI and the upstream."""

    async def test_one_request_reaches_the_upstream(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FAKE_CLI_MODE", "relay-once")
        log = tmp_path / "cli.jsonl"
        monkeypatch.setenv("FAKE_CLI_LOG", str(log))
        upstream = FakeUpstream([(200, b'{"ok":true}')])
        url = await upstream.start()
        transport = ClaudeCliTransport(timeout=30)
        stream = None
        try:
            provider = _provider(_write_cli(tmp_path), upstream=url)
            stream = await transport.send_streaming(provider, "", _turn())
            try:
                [event async for event in stream]
            finally:
                await stream.close()
                await transport.close()
        finally:
            await upstream.close()

        assert len(upstream.requests) == 1
        request = upstream.requests[0]
        assert request["headers"]["authorization"] == "Bearer test-token"
        assert request["headers"]["anthropic-version"] == "2023-06-01"
        body = json.loads(request["body"])
        assert body["model"] == "claude-test"
        # The CLI's own annotation message is dropped and the conversation
        # comes back as the host sent it, with the breakpoint moved last.
        assert [message["role"] for message in body["messages"]] == [
            "user",
            "assistant",
            "user",
        ]
        assert body["messages"][-1]["content"][-1]["cache_control"] == {
            "type": "ephemeral"
        }
        assert stream is not None
        assert [reading.status for reading in stream.readings] == [200]
        assert [reading.restored for reading in stream.readings] == [True]
        assert (
            stream.readings[0].rate_limit()["anthropic-ratelimit-unified-status"]
            == "allowed"
        )

    async def test_a_second_attempt_is_refused_locally(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FAKE_CLI_MODE", "relay-twice")
        log = tmp_path / "cli.jsonl"
        monkeypatch.setenv("FAKE_CLI_LOG", str(log))
        upstream = FakeUpstream([(200, b'{"ok":true}')])
        url = await upstream.start()
        transport = ClaudeCliTransport(timeout=30)
        try:
            provider = _provider(_write_cli(tmp_path), upstream=url)
            stream = await transport.send_streaming(provider, "", _turn())
            try:
                [event async for event in stream]
            finally:
                await stream.close()
                await transport.close()
            statuses = [entry for entry in _log(log) if entry["kind"] == "relay"][0][
                "statuses"
            ]
        finally:
            await upstream.close()

        assert statuses == [200, 429]
        assert len(upstream.requests) == 1

    async def test_a_401_lets_one_retry_through(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FAKE_CLI_MODE", "relay-twice")
        log = tmp_path / "cli.jsonl"
        monkeypatch.setenv("FAKE_CLI_LOG", str(log))
        upstream = FakeUpstream([(401, b'{"error":"expired"}'), (200, b'{"ok":true}')])
        url = await upstream.start()
        transport = ClaudeCliTransport(timeout=30)
        try:
            provider = _provider(_write_cli(tmp_path), upstream=url)
            stream = await transport.send_streaming(provider, "", _turn())
            try:
                [event async for event in stream]
            finally:
                await stream.close()
                await transport.close()
            statuses = [entry for entry in _log(log) if entry["kind"] == "relay"][0][
                "statuses"
            ]
        finally:
            await upstream.close()

        assert statuses == [401, 200]
        assert len(upstream.requests) == 2

    def test_a_plain_http_upstream_must_be_loopback(self):
        with pytest.raises(RelayError, match="loopback"):
            Relay("http://api.example.com")

    def test_a_hosted_upstream_must_be_a_url(self):
        with pytest.raises(RelayError, match="not a URL"):
            Relay("api.example.com")


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

        start = _start(log)
        argv = start["argv"]
        assert argv[0] == "-p"
        assert argv[argv.index("--input-format") + 1] == "stream-json"
        assert argv[argv.index("--output-format") + 1] == "stream-json"
        assert argv[argv.index("--tools") + 1] == ""
        assert argv[argv.index("--model") + 1] == "claude-test"
        assert argv[argv.index("--settings") + 1].endswith("settings.json")
        assert argv[-2:] == ["--setting-sources", "user"]
        assert "--include-partial-messages" in argv
        assert "--no-session-persistence" in argv
        assert start["system"] == "You are terse."

    async def test_a_context_window_tag_reaches_the_cli_verbatim(
        self, monkeypatch, tmp_path
    ):
        """``[1m]`` is the CLI's own 1M opt-in; the gateway must not strip it.

        The CLI resolves the tag itself: a bare model id gets the model's
        normal window, an id with ``[1m]`` asks for one million tokens.  An
        unknown id falls back to the CLI's 200K default.  Rewriting the id
        here would silently pin every route to the wrong window.
        """
        monkeypatch.setenv("FAKE_CLI_MODE", "stream")
        log = tmp_path / "cli.jsonl"
        monkeypatch.setenv("FAKE_CLI_LOG", str(log))
        transport = ClaudeCliTransport(timeout=30)

        turn = _turn()
        turn["model"] = "claude-sonnet-5[1m]"
        stream = await transport.send_streaming(
            _provider(_write_cli(tmp_path)), "", turn
        )
        try:
            [event async for event in stream]
        finally:
            await stream.close()
            await transport.close()

        argv = _start(log)["argv"]
        assert argv[argv.index("--model") + 1] == "claude-sonnet-5[1m]"

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

        frames = _start(log)["frames"]
        assert [frame["type"] for frame in frames] == ["user", "assistant", "user"]
        assert frames[0]["message"]["content"] == [
            {"type": "text", "text": "name a fruit"}
        ]

    async def test_the_history_is_written_one_frame_per_acknowledgment(
        self, monkeypatch, tmp_path
    ):
        monkeypatch.setenv("FAKE_CLI_MODE", "chunks")
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

        chunks = [entry["lines"] for entry in _log(log) if entry["kind"] == "chunk"]
        # Writing the whole history at once makes the CLI fold the later user
        # frames into the turn's own message, and the model answers the first
        # question.  The first read must therefore hold the first history frame
        # alone: it is written only once the CLI has acknowledged the previous
        # queued turn.
        assert chunks[0] == 1
        assert len(_start(log)["frames"]) == 3

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

        start = _start(log)
        assert start["removed"] is None
        assert start["kept"] == "yes"

    async def test_a_missing_cli_names_the_command(self, tmp_path):
        transport = ClaudeCliTransport(timeout=5)
        provider = _provider((str(tmp_path / "does-not-exist"),))

        with pytest.raises(UpstreamConnectionError, match="cannot start"):
            await transport.send_streaming(provider, "", _turn())
