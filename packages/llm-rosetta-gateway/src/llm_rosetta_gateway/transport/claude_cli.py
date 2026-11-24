"""The ``claude_cli`` transport: the official Claude Code CLI as a model client.

The lane has no URL and no gateway credential.  It runs ``claude`` in its
non-interactive stream-json mode, hands it the conversation as NDJSON frames
on stdin, and reads the CLI's own events on stdout.  The CLI builds and signs
the upstream request itself, so the account stays in the CLI's login store.

Wire direction:

* **in** — a target-format ``messages`` body.  :func:`build_frames` turns its
  system prompt and message list into the frames the CLI accepts.
* **out** — ``messages`` SSE events.  With ``--include-partial-messages`` the
  CLI emits the raw Anthropic stream events, so ``stream_event.event`` is
  forwarded as it is; a CLI that emits only whole ``assistant`` messages is
  synthesised into the same event sequence.

The CLI's own tool set is disabled (``--tools ""``) and its session
persistence is off: the gateway owns the conversation, and a request must not
leave state behind.

Every turn's child points ``ANTHROPIC_BASE_URL`` at a per-turn loopback relay
(:mod:`~.claude_cli_relay`).  The relay forwards the CLI's single upstream
request with the CLI's own headers, refuses the CLI's recovery attempts, and
keeps the upstream status and rate-limit headers for the gateway.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import shutil
import signal
import tempfile
import uuid
from collections import deque
from collections.abc import AsyncIterator, Callable
from typing import Any

from ._base import (
    UpstreamConnectionError,
    UpstreamResponse,
    UpstreamStream,
    UpstreamTimeoutError,
    UpstreamTransport,
)
from .claude_cli_relay import Relay, RelayError
from .provider_info import ProviderInfo

#: The command a shim that declares none gets.
DEFAULT_ARGV: tuple[str, ...] = ("claude",)

#: How long the CLI may stay silent before the turn is failed.  The gateway's
#: own read timeout wins when the factory passes one.
DEFAULT_TIMEOUT = 300.0

#: stdout line limit.  A single ``assistant`` event can carry a whole tool
#: input; the default 64 KiB stream limit is too small for a large schema.
_STREAM_LIMIT = 8 * 1024 * 1024

#: Namespace for the stable per-conversation session id.  A fixed UUID keeps
#: the id reproducible across processes.
_SESSION_NAMESPACE = uuid.UUID("6f5e2b1c-2f38-4c1e-9f2a-7c6d1e9a4b30")

#: Events whose payload is the raw Anthropic stream event.
_STREAM_EVENT = "stream_event"

#: Stop reason when the CLI reports none.
_DEFAULT_STOP_REASON = "end_turn"

#: Environment the transport needs whatever the shim declares.  A shim's own
#: ``cli_env`` value wins, because these are defaults.
_FIXED_ENV: dict[str, str] = {
    # Native's session-title request is its own upstream call, and it races
    # the turn for the relay's one admitted request.  The gateway names the
    # turn itself, so the title call is not wanted.
    "CLAUDE_CODE_DISABLE_TERMINAL_TITLE": "1",
    # The gateway owns the conversation and the system prompt.  Native's
    # memory and CLAUDE.md files would otherwise be injected into the replayed
    # prefix, changing it between turns and reading host notes as user text.
    "CLAUDE_CODE_DISABLE_AUTO_MEMORY": "1",
    "CLAUDE_CODE_DISABLE_CLAUDE_MDS": "1",
}


def _system_text(system: Any) -> str:
    """Flatten a Messages ``system`` value to the text the CLI takes.

    Args:
        system: A string, a list of content blocks, or ``None``.

    Returns:
        The joined text of every ``text`` block, or an empty string.

    Raises:
        UpstreamConnectionError: The value is neither a string nor a list.
    """
    if system is None:
        return ""
    if isinstance(system, str):
        return system
    if not isinstance(system, list):
        raise UpstreamConnectionError(
            "claude_cli: system must be a string or a list of content blocks"
        )
    parts = [
        str(block.get("text") or "")
        for block in system
        if isinstance(block, dict) and block.get("type") == "text"
    ]
    return "\n\n".join(part for part in parts if part)


def _content_blocks(content: Any) -> list[dict[str, Any]]:
    """Normalise one message's content to a list of blocks.

    Args:
        content: A string, a list of blocks, or ``None``.

    Returns:
        The message's blocks.

    Raises:
        UpstreamConnectionError: The value is neither a string nor a list.
    """
    if content is None:
        return []
    if isinstance(content, str):
        return [{"type": "text", "text": content}] if content else []
    if not isinstance(content, list):
        raise UpstreamConnectionError(
            "claude_cli: message content must be a string or a list of blocks"
        )
    return [dict(block) for block in content if isinstance(block, dict)]


def build_frames(body: dict[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    """Turn a Messages body into the CLI's system prompt and stdin frames.

    The CLI replays the conversation itself: every frame but the last is sent
    with ``shouldQuery: false`` so the CLI ingests it without answering, and
    the last user frame is the turn.  Preserving the original roles and
    ``tool_use`` / ``tool_result`` ids is what lets the CLI keep its history
    consistent across turns.

    Args:
        body: The target-format request body.

    Returns:
        The system prompt text (``""`` when the request carries none) and the
        frames to write, oldest first.

    Raises:
        UpstreamConnectionError: A role the CLI cannot replay, or a
            conversation that does not end with a non-empty user message
            (assistant prefill is not supported by the CLI).
    """
    system = _system_text(body.get("system"))
    frames: list[dict[str, Any]] = []
    for message in body.get("messages") or []:
        if not isinstance(message, dict):
            continue
        role = message.get("role")
        if role not in ("user", "assistant"):
            raise UpstreamConnectionError(
                f"claude_cli: unsupported message role {role!r}; only user and "
                "assistant can be replayed"
            )
        blocks = _content_blocks(message.get("content"))
        if not blocks:
            continue
        frames.append({"type": role, "message": {"role": role, "content": blocks}})
    if not frames or frames[-1]["type"] != "user":
        raise UpstreamConnectionError(
            "claude_cli: the conversation must end with a non-empty user "
            "message; assistant prefill is unsupported"
        )
    for frame in frames[:-1]:
        frame["shouldQuery"] = False
    return system, frames


def conversation_session_id(system: str, frames: list[dict[str, Any]]) -> str:
    """The CLI session id for this conversation.

    The CLI names a per-session scratch directory in the request it builds,
    so a fresh id per request would move the prompt-cache prefix every turn.
    The id is derived from what the conversation starts with, which is stable
    for the life of a conversation.

    Args:
        system: The system prompt text.
        frames: The frames the turn sends.

    Returns:
        A UUID-shaped session id.
    """
    seed = json.dumps(
        {"system": system, "first": frames[0] if frames else None},
        sort_keys=True,
        ensure_ascii=False,
        default=str,
    )
    return str(uuid.uuid5(_SESSION_NAMESPACE, seed))


#: The cache marker the Messages API accepts on a content block.
_BREAKPOINT: dict[str, str] = {"type": "ephemeral"}

#: Block kinds a cache breakpoint may sit on.
_BREAKPOINT_KINDS = frozenset({"text", "tool_use", "tool_result", "image"})

#: The API's limit on cache breakpoints in one request.
_MAX_BREAKPOINTS = 4


def _content_blocks_of(value: Any) -> list[dict[str, Any]]:
    """The block list of a message value, or an empty list."""
    content = value.get("content") if isinstance(value, dict) else None
    return content if isinstance(content, list) else []


def _breakpoint_blocks(body: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Every block of *body* that carries a cache breakpoint, in request order.

    Args:
        body: A Messages request body.

    Returns:
        ``(where, block)`` pairs, ``where`` being ``"system"`` or ``"messages"``.
    """
    found: list[tuple[str, dict[str, Any]]] = []
    for block in body.get("system") or []:
        if isinstance(block, dict) and block.get("cache_control"):
            found.append(("system", block))
    for message in body.get("messages") or []:
        for block in _content_blocks_of(message):
            if isinstance(block, dict) and block.get("cache_control"):
                found.append(("messages", block))
    return found


def _relocate_breakpoint(body: dict[str, Any], messages: list[dict[str, Any]]) -> None:
    """Move the newest cache breakpoint onto the turn's last block.

    The CLI puts its own breakpoint on the annotation message it appends, and
    that message is dropped.  Without a breakpoint at the end of the
    conversation the newest turn is never written to the cache, so every later
    turn re-reads the growing tail at full price.

    Args:
        body: The request body being built.
        messages: The conversation that will be sent.
    """
    content = _content_blocks_of(messages[-1]) if messages else []
    if not content:
        return
    block = content[-1]
    if not isinstance(block, dict) or block.get("cache_control"):
        return
    if block.get("type") not in _BREAKPOINT_KINDS:
        return
    markers = _breakpoint_blocks(body)
    if len(markers) >= _MAX_BREAKPOINTS:
        for where, existing in markers:
            if where == "messages":
                existing.pop("cache_control", None)
                break
    block["cache_control"] = dict(_BREAKPOINT)


def make_restorer(
    host_body: dict[str, Any],
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the rewrite that puts the host's conversation back.

    The CLI appends a ``role: "system"`` annotation message after every user
    message it replays: an environment reminder that names its own working
    directory, a date line, context reminders.  Those blocks sit inside the
    conversation prefix and differ between turns, so the upstream can never
    reuse the cached prefix, and they describe a directory the client never
    asked about.  They are the CLI's bookkeeping, not part of the conversation,
    so they are dropped before the request leaves the relay, and the cache
    breakpoint the CLI put on them is moved to the end of the turn.

    The rewrite fails open: a conversation whose message count does not come
    back as the host sent it is forwarded unchanged, which costs the cached
    prefix but never the answer.

    Args:
        host_body: The target-format body the gateway received.

    Returns:
        A function from the CLI's parsed request body to the body to forward.
    """
    expected = sum(
        1 for message in host_body.get("messages") or [] if isinstance(message, dict)
    )

    def restore(cli_body: dict[str, Any]) -> dict[str, Any]:
        messages = [
            message
            for message in cli_body.get("messages") or []
            if isinstance(message, dict)
        ]
        kept = [message for message in messages if message.get("role") != "system"]
        if len(kept) != expected:
            return cli_body
        body = {**cli_body, "messages": kept}
        _relocate_breakpoint(body, kept)
        return body

    return restore


def _spawn_argv(
    provider_info: ProviderInfo,
    session_id: str,
    system_path: str | None,
    model: str,
    settings_path: str,
) -> list[str]:
    """The CLI argv for one turn.

    Args:
        provider_info: The lane's resolved provider config.
        session_id: The conversation's CLI session id.
        system_path: File the system prompt was written to, if any.
        model: The model the client named, when it named one.
        settings_path: The private settings file that pins the relay URL.

    Returns:
        The full argv.
    """
    argv = list(provider_info.cli_argv or DEFAULT_ARGV)
    argv += [
        "-p",
        "--verbose",
        "--input-format",
        "stream-json",
        "--output-format",
        "stream-json",
        "--include-partial-messages",
        "--tools",
        "",
        "--setting-sources",
        "",
        "--strict-mcp-config",
        "--disable-slash-commands",
        "--max-turns",
        "1",
        "--permission-mode",
        "dontAsk",
        "--no-session-persistence",
    ]
    if system_path is not None:
        argv += ["--system-prompt-file", system_path]
    if model:
        argv += ["--model", model]
    argv += ["--settings", settings_path]
    argv += ["--session-id", session_id]
    argv += list(provider_info.cli_args)
    return argv


def _spawn_env(provider_info: ProviderInfo) -> dict[str, str]:
    """The CLI environment: the gateway's, minus the unset names, plus overrides.

    Args:
        provider_info: The lane's resolved provider config.

    Returns:
        The environment mapping for the child.
    """
    env = dict(os.environ)
    for name in provider_info.cli_env_unset:
        env.pop(name, None)
    env.update({str(key): str(value) for key, value in provider_info.cli_env.items()})
    for key, value in _FIXED_ENV.items():
        env.setdefault(key, value)
    return env


class _CliProcess:
    """One CLI child, its private working directory, and its stderr drain."""

    def __init__(
        self,
        process: asyncio.subprocess.Process,
        workdir: str,
        relay: Relay | None = None,
    ) -> None:
        self.process = process
        self.workdir = workdir
        self.relay = relay
        self.stderr: deque[str] = deque(maxlen=40)
        self._stderr_task = asyncio.ensure_future(self._drain_stderr())

    async def _drain_stderr(self) -> None:
        """Keep stderr flowing so a full pipe cannot stall the child."""
        stream = self.process.stderr
        if stream is None:
            return
        with contextlib.suppress(Exception):
            while True:
                line = await stream.readline()
                if not line:
                    return
                text = line.decode("utf-8", errors="replace").strip()
                if text:
                    self.stderr.append(text)

    @property
    def returncode(self) -> int | None:
        """The child's exit status, or ``None`` while it runs."""
        return self.process.returncode

    def stderr_tail(self) -> str:
        """The last stderr lines, for an error message."""
        return "\n".join(self.stderr)

    async def stop(self) -> None:
        """Terminate the whole process tree and remove the working directory."""
        process = self.process
        if process.returncode is None:
            with contextlib.suppress(ProcessLookupError, PermissionError):
                if os.name == "posix":
                    os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                else:
                    process.terminate()
            try:
                await asyncio.wait_for(process.wait(), 5)
            except (TimeoutError, asyncio.TimeoutError):
                with contextlib.suppress(ProcessLookupError, PermissionError):
                    if os.name == "posix":
                        os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                    else:
                        process.kill()
                with contextlib.suppress(Exception):
                    await process.wait()
        self._stderr_task.cancel()
        with contextlib.suppress(asyncio.CancelledError, Exception):
            await self._stderr_task
        if self.relay is not None:
            await self.relay.close()
        shutil.rmtree(self.workdir, ignore_errors=True)


class _CliStream(UpstreamStream):
    """A turn's ``messages`` events, produced from one CLI process."""

    status_code = 200

    def __init__(
        self,
        cli: _CliProcess,
        model: str,
        timeout: float,
        frames: list[dict[str, Any]],
    ) -> None:
        self._cli = cli
        self._model = model
        self._timeout = timeout
        self._frames = list(frames)
        self._next_frame = 0
        self._turn_sent = False
        self._queue: deque[dict[str, Any]] = deque()
        self._finished = False
        self._error: str | None = None
        self._saw_stream_event = False
        self._last_kind: str | None = None
        self._assistant: dict[str, Any] | None = None
        self._result: dict[str, Any] | None = None

    # -- UpstreamStream -----------------------------------------------------

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self

    async def __anext__(self) -> dict[str, Any]:
        while not self._queue:
            if self._error is not None:
                raise UpstreamConnectionError(self._error)
            if self._finished:
                raise StopAsyncIteration
            if self._turn_sent:
                await self._pump()
            else:
                await self._send_frame()
        return self._queue.popleft()

    async def read_error(self) -> str:
        """The failure text, for a caller that asked instead of iterating."""
        return self._error or self._cli.stderr_tail()

    async def close(self) -> None:
        """Stop the turn's process."""
        await self._cli.stop()

    async def aclose(self) -> None:
        """Alias for :meth:`close`, for an explicit async close."""
        await self.close()

    @property
    def assistant(self) -> dict[str, Any] | None:
        """The last whole ``assistant`` message the CLI emitted, if any."""
        return self._assistant

    @property
    def result(self) -> dict[str, Any] | None:
        """The ``result`` event that settled the turn, if any."""
        return self._result

    @property
    def readings(self) -> list[Any]:
        """The upstream calls the relay saw during this turn."""
        relay = self._cli.relay
        return list(relay.readings) if relay is not None else []

    # -- replay, one frame at a time ---------------------------------------

    async def _send_frame(self) -> None:
        """Write the next frame; wait for its acknowledgment when it has one.

        The CLI acknowledges a replayed *user* frame with a zero-turn
        ``result`` only after it has ingested it; a replayed assistant frame is
        appended to the history and answers nothing.  Sending the whole history
        at once makes the CLI fold the later user frames into the turn's own
        user message — the model then answers the first question in that merged
        block.  Waiting on the user frames is what keeps the history in order.
        """
        frame = self._frames[self._next_frame]
        self._next_frame += 1
        await self._write(frame)
        if frame.get("shouldQuery") is False and frame.get("type") == "user":
            await self._await_ack()
        elif frame.get("shouldQuery") is not False:
            self._turn_sent = True

    async def _write(self, frame: dict[str, Any]) -> None:
        """Write one NDJSON frame to the CLI's stdin."""
        stream = self._cli.process.stdin
        if stream is None:
            raise UpstreamConnectionError("claude_cli: the CLI has no stdin")
        payload = (json.dumps(frame, ensure_ascii=False, default=str) + "\n").encode(
            "utf-8"
        )
        try:
            stream.write(payload)
            await stream.drain()
        except (BrokenPipeError, ConnectionResetError):
            await self._cli.stop()
            raise UpstreamConnectionError(
                "claude_cli: the CLI closed its input before the turn started"
                + _stderr_suffix(self._cli)
            ) from None

    async def _await_ack(self) -> None:
        """Wait for the zero-turn ``result`` that ends a replayed frame."""
        while True:
            event = await self._next_event()
            if event is None:
                self._finish()
                return
            if event.get("type") != "result":
                # Setup notices during ingestion carry nothing this turn uses.
                continue
            if int(event.get("num_turns") or 0) == 0:
                return
            self._result = event
            self._finish()
            return

    # -- event pump ---------------------------------------------------------

    async def _next_event(self) -> dict[str, Any] | None:
        """Read one CLI event, or ``None`` at end of stream."""
        stream = self._cli.process.stdout
        if stream is None:
            return None
        try:
            line = await asyncio.wait_for(stream.readline(), self._timeout)
        except (TimeoutError, asyncio.TimeoutError) as exc:
            self._error = f"claude_cli: the CLI stayed silent for {self._timeout:.0f}s"
            await self._cli.stop()
            raise UpstreamTimeoutError(self._error) from exc
        if not line:
            return None
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            text = line.decode("utf-8", errors="replace")[:300]
            raise UpstreamConnectionError(
                f"claude_cli: the CLI wrote a non-JSON line: {text!r}"
            ) from exc
        return event if isinstance(event, dict) else None

    async def _pump(self) -> None:
        """Read one stdout line and, on end of stream, settle the turn."""
        event = await self._next_event()
        if event is None:
            self._finish()
        else:
            self._handle(event)

    def _handle(self, event: dict[str, Any]) -> None:
        """Dispatch one CLI event into the outgoing queue."""
        kind = event.get("type")
        if kind == _STREAM_EVENT:
            payload = event.get("event")
            if isinstance(payload, dict):
                self._saw_stream_event = True
                self._emit(payload)
            return
        if kind == "assistant":
            message = event.get("message")
            if isinstance(message, dict):
                self._assistant = message
            return
        if kind == "result":
            if int(event.get("num_turns") or 0) == 0:
                # The acknowledgment of a replayed history frame.
                return
            self._result = event
            self._finish()
            return
        # ``system``, ``user`` and vendor-specific events carry no content
        # this lane uses.

    def _emit(self, payload: dict[str, Any]) -> None:
        """Queue one outgoing event, remembering its kind."""
        self._last_kind = str(payload.get("type") or "")
        self._queue.append(payload)

    def _finish(self) -> None:
        """Settle the turn: synthesise events, or close an already-streamed one."""
        if self._finished:
            return
        self._finished = True
        result = self._result
        if result is None and self._assistant is None and not self._saw_stream_event:
            self._error = self._cli.stderr_tail() or (
                "claude_cli: the CLI ended without a response"
            )
            self._queue.clear()
            return
        if result is not None and result.get("is_error"):
            texts = _result_texts(self._assistant)
            if not texts and not self._saw_stream_event:
                self._error = _result_error(result) or "claude_cli: the turn failed"
                self._queue.clear()
                return
        if not self._saw_stream_event:
            self._synthesise()
        elif self._last_kind != "message_stop":
            self._emit({"type": "message_stop"})

    # -- synthesis ----------------------------------------------------------

    def _synthesise(self) -> None:
        """Build a full event sequence from one whole assistant message."""
        message = self._assistant or {}
        result = self._result or {}
        blocks = message.get("content")
        if not isinstance(blocks, list) or not blocks:
            text = str(result.get("result") or "")
            blocks = [{"type": "text", "text": text}] if text else []
        usage = _mapping(message.get("usage"))
        result_usage = _mapping(result.get("usage"))
        self._emit(
            {
                "type": "message_start",
                "message": {
                    "id": message.get("id") or f"msg_{uuid.uuid4().hex}",
                    "type": "message",
                    "role": "assistant",
                    "model": message.get("model") or self._model,
                    "content": [],
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {
                        "input_tokens": int(usage.get("input_tokens") or 0),
                        "output_tokens": 0,
                    },
                },
            }
        )
        for index, block in enumerate(blocks):
            if not isinstance(block, dict):
                continue
            kind = block.get("type")
            if kind == "text":
                self._emit(
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "text", "text": ""},
                    }
                )
                text = str(block.get("text") or "")
                if text:
                    self._emit(
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {"type": "text_delta", "text": text},
                        }
                    )
            elif kind == "thinking":
                self._emit(
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "thinking", "thinking": ""},
                    }
                )
                thinking = str(block.get("thinking") or "")
                if thinking:
                    self._emit(
                        {
                            "type": "content_block_delta",
                            "index": index,
                            "delta": {
                                "type": "thinking_delta",
                                "thinking": thinking,
                            },
                        }
                    )
            elif kind == "tool_use":
                self._emit(
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {
                            "type": "tool_use",
                            "id": block.get("id"),
                            "name": block.get("name"),
                            "input": {},
                        },
                    }
                )
                self._emit(
                    {
                        "type": "content_block_delta",
                        "index": index,
                        "delta": {
                            "type": "input_json_delta",
                            "partial_json": json.dumps(
                                block.get("input") or {}, ensure_ascii=False
                            ),
                        },
                    }
                )
            else:
                continue
            self._emit({"type": "content_block_stop", "index": index})
        self._emit(
            {
                "type": "message_delta",
                "delta": {
                    "stop_reason": result.get("stop_reason")
                    or message.get("stop_reason")
                    or _DEFAULT_STOP_REASON,
                    "stop_sequence": None,
                },
                "usage": {
                    key: value
                    for key, value in result_usage.items()
                    if key
                    in (
                        "input_tokens",
                        "output_tokens",
                        "cache_creation_input_tokens",
                        "cache_read_input_tokens",
                    )
                },
            }
        )
        self._emit({"type": "message_stop"})


def _mapping(value: Any) -> dict[str, Any]:
    """A defensive copy of *value* when it is a mapping, else an empty one."""
    return dict(value) if isinstance(value, dict) else {}


def _result_texts(assistant: dict[str, Any] | None) -> list[str]:
    """The assistant message's text blocks, if any."""
    if not assistant:
        return []
    return [
        str(block.get("text") or "")
        for block in assistant.get("content") or []
        if isinstance(block, dict) and block.get("type") == "text"
    ]


def _result_error(result: dict[str, Any]) -> str:
    """A failure text from a ``result`` event."""
    errors = result.get("errors")
    if isinstance(errors, list) and errors:
        return "claude_cli: " + "; ".join(str(item) for item in errors)
    text = result.get("result")
    if isinstance(text, str) and text:
        return f"claude_cli: {text}"
    subtype = result.get("subtype")
    return f"claude_cli: the turn failed ({subtype})" if subtype else ""


class ClaudeCliTransport(UpstreamTransport):
    """Drive the official Claude Code CLI as a model client.

    One process per request: the CLI has no long-lived session in this mode
    (``--no-session-persistence``), and a request must not be able to leave
    state behind for the next one.
    """

    def __init__(self, *, timeout: float = DEFAULT_TIMEOUT) -> None:
        self._timeout = float(timeout)

    # -- UpstreamTransport --------------------------------------------------

    async def send(
        self,
        provider_info: ProviderInfo,
        url: str,
        body: dict[str, Any],
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> UpstreamResponse:
        """Run one turn and return the whole Messages response body.

        Args:
            provider_info: The lane's resolved provider config.
            url: Unused; the lane dials nothing.
            body: The target-format request body.
            extra_headers: Unused; headers belong to the CLI.

        Returns:
            The turn's response.

        Raises:
            UpstreamConnectionError: The CLI failed the turn.
            UpstreamTimeoutError: The CLI stayed silent past the timeout.
        """
        stream = await self._turn(provider_info, body)
        assistant, result = await self._collect(stream)
        model = str(body.get("model") or "")
        return _response_body(assistant, result, model)

    async def send_streaming(
        self,
        provider_info: ProviderInfo,
        url: str,
        body: dict[str, Any],
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> _CliStream:
        """Run one turn and return its ``messages`` event stream.

        Args:
            provider_info: The lane's resolved provider config.
            url: Unused; the lane dials nothing.
            body: The target-format request body.
            extra_headers: Unused; headers belong to the CLI.

        Returns:
            The turn's event stream.

        Raises:
            UpstreamConnectionError: The CLI could not be started.
        """
        return await self._turn(provider_info, body)

    async def close(self) -> None:
        """Release transport resources.  Each turn owns its own process."""

    # -- turn ---------------------------------------------------------------

    async def _turn(
        self, provider_info: ProviderInfo, body: dict[str, Any]
    ) -> _CliStream:
        """Start one CLI process and write the conversation to it."""
        system, frames = build_frames(body)
        session_id = conversation_session_id(system, frames)
        workdir = tempfile.mkdtemp(prefix="llm-rosetta-claude-cli-")
        system_path: str | None = None
        if system:
            system_path = os.path.join(workdir, "system.md")
            with open(system_path, "w", encoding="utf-8") as handle:
                handle.write(system)
        argv = _spawn_argv(
            provider_info,
            session_id,
            system_path,
            str(body.get("model") or ""),
            os.path.join(workdir, "settings.json"),
        )
        env = _spawn_env(provider_info)
        try:
            relay = Relay(
                provider_info.cli_upstream,
                timeout=self._timeout,
                rewrite=make_restorer(body),
            )
            relay_url = await relay.start()
        except RelayError as exc:
            shutil.rmtree(workdir, ignore_errors=True)
            raise UpstreamConnectionError(str(exc)) from exc
        env["ANTHROPIC_BASE_URL"] = relay_url
        # A user settings file may set ``env.ANTHROPIC_BASE_URL`` itself, and
        # settings win over the process environment: the CLI would then call
        # its own host directly and the relay would see nothing.  ``--settings``
        # is the highest user-controlled layer, so the private file pins the
        # relay URL whatever the profile says.
        with open(
            os.path.join(workdir, "settings.json"), "w", encoding="utf-8"
        ) as handle:
            json.dump({"env": {"ANTHROPIC_BASE_URL": relay_url}}, handle)
        spawn_kwargs: dict[str, Any] = {}
        if os.name == "posix":
            spawn_kwargs["start_new_session"] = True
            spawn_kwargs["limit"] = _STREAM_LIMIT
        try:
            process = await asyncio.create_subprocess_exec(
                *argv,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                env=env,
                **spawn_kwargs,
            )
        except OSError as exc:
            await relay.close()
            shutil.rmtree(workdir, ignore_errors=True)
            raise UpstreamConnectionError(
                f"claude_cli: cannot start {argv[0]!r}: {exc}"
            ) from exc
        cli = _CliProcess(process, workdir, relay)
        return _CliStream(cli, str(body.get("model") or ""), self._timeout, frames)

    async def _collect(
        self, stream: _CliStream
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Consume the turn's stream and keep its whole-message facts."""
        async for _event in stream:
            pass
        assistant = stream.assistant
        result = stream.result
        await stream.close()
        return assistant, result


def _response_body(
    assistant: dict[str, Any] | None,
    result: dict[str, Any] | None,
    model: str,
) -> UpstreamResponse:
    """Build one Messages response body from the turn's facts."""
    result = result or {}
    message = assistant or {}
    blocks = message.get("content")
    if not isinstance(blocks, list) or not blocks:
        text = str(result.get("result") or "")
        blocks = [{"type": "text", "text": text}] if text else []
    usage = _mapping(message.get("usage"))
    result_usage = _mapping(result.get("usage"))
    merged_usage = {**usage, **result_usage}
    body: dict[str, Any] = {
        "id": message.get("id") or f"msg_{uuid.uuid4().hex}",
        "type": "message",
        "role": "assistant",
        "model": message.get("model") or model,
        "content": blocks,
        "stop_reason": result.get("stop_reason")
        or message.get("stop_reason")
        or _DEFAULT_STOP_REASON,
        "stop_sequence": None,
        "usage": {
            "input_tokens": int(merged_usage.get("input_tokens") or 0),
            "output_tokens": int(merged_usage.get("output_tokens") or 0),
            **{
                key: merged_usage[key]
                for key in (
                    "cache_creation_input_tokens",
                    "cache_read_input_tokens",
                )
                if key in merged_usage
            },
        },
    }
    raw = json.dumps(body, ensure_ascii=False).encode("utf-8")
    return UpstreamResponse(status_code=200, body=body, raw_content=raw)


def _stderr_suffix(cli: _CliProcess) -> str:
    """The child's last stderr lines, when it wrote any."""
    tail = cli.stderr_tail()
    return f"; stderr: {tail}" if tail else ""
