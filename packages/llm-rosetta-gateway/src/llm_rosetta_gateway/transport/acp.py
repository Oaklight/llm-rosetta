"""The ACP transport: drive a local agent CLI over stdio JSON-RPC.

ACP is not HTTP, so this transport does not POST a body anywhere.  It keeps
one agent process per configured provider, opens one session per request,
sends the request's prompt blocks, and turns ``session/update`` notifications
into the chunk shapes the ``acp`` converter reads.

The protocol implementation itself is vendored (``_vendor/acp.py``, from
zerodep): the client, the JSON-RPC framing and the session queues.  What
lives here is the part that is this gateway's own: how a provider's process is
started, which agent command serves it, how a refused session is reported, and
how the agent's events become chunks.

Phase 1 is deliberately stateless per request: each request opens a session,
sends the whole conversation, and closes it.  Session reuse across requests
depends on the agent's history agreeing with the client's, which needs a
ledger this slice does not keep yet; replaying the conversation is slower but
cannot silently lose a turn.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from .._vendor.acp import (
    ACPClient,
    ClientCapabilities,
    FsCapabilities,
    ImplementationInfo,
    InitializeParams,
)
from .._vendor.jsonrpc import JSONRPCException
from ..transport._base import UpstreamConnectionError, UpstreamResponse, UpstreamStream
from ..transport.provider_info import ProviderInfo

logger = logging.getLogger("llm-rosetta-gateway")

#: Chunk shapes this transport emits.  The ``acp`` converter reads exactly
#: these; keeping them in one place is what lets both sides agree without a
#: shared schema module.
CHUNK_SESSION_START = "session_start"
CHUNK_TEXT = "text"
CHUNK_THOUGHT = "thought"
CHUNK_USAGE = "usage"
CHUNK_STOP = "stop"
CHUNK_ERROR = "error"

#: Default agent argv.  A provider may override it with ``connection.acp_argv``
#: (a shim declares the command its vendor ships); ``agy acp`` is the
#: Antigravity one.
DEFAULT_AGENT_ARGV: tuple[str, ...] = ("agy", "acp")

#: A session that cannot be opened usually means the agent is not signed in.
#: The ACP error carries no code of its own, so the wording is the signal.
_SIGN_IN_HINTS = (
    "failed to start",
    "not signed in",
    "sign in",
    "authenticate",
    "login",
)


class AcpTurnError(UpstreamConnectionError):
    """The agent refused or failed the turn.

    Carries the HTTP status the gateway should report, so an agent that is
    not signed in reads as "this lane is not ready" rather than as a model
    failure.
    """

    def __init__(self, message: str, *, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class _Agent:
    """One provider's agent process and its handshake result."""

    client: ACPClient
    info: Any = None
    auth_methods: list[str] = field(default_factory=list)


@dataclass
class _Turn:
    """An open ACP session, ready to be prompted."""

    agent: _Agent
    session_id: str
    model: str


def _looks_like_sign_in(message: str) -> bool:
    lowered = message.lower()
    return any(hint in lowered for hint in _SIGN_IN_HINTS)


class AcpTransport:
    """Transport for providers whose upstream is a local ACP agent."""

    def __init__(self, *, timeout: float = 300.0) -> None:
        self._timeout = timeout
        self._agents: dict[str, _Agent] = {}
        self._locks: dict[str, asyncio.Lock] = {}

    # -- agent lifecycle ----------------------------------------------------

    def _argv(self, provider_info: ProviderInfo) -> list[str]:
        argv = getattr(provider_info, "acp_argv", None)
        return list(argv) if argv else list(DEFAULT_AGENT_ARGV)

    async def _agent(self, provider_info: ProviderInfo) -> _Agent:
        key = provider_info.config_name
        agent = self._agents.get(key)
        if agent is not None:
            return agent
        lock = self._locks.setdefault(key, asyncio.Lock())
        async with lock:
            agent = self._agents.get(key)
            if agent is not None:
                return agent
            argv = self._argv(provider_info)
            client = ACPClient(argv)
            try:
                await client.start()
                info = await client.initialize(
                    InitializeParams(
                        client_info=ImplementationInfo(
                            name="llm-rosetta-gateway", version=""
                        ),
                        client_capabilities=ClientCapabilities(
                            fs=FsCapabilities(),
                            terminal=False,
                        ),
                    )
                )
            except (OSError, JSONRPCException) as exc:
                await client.stop()
                raise AcpTurnError(
                    f"provider {provider_info.config_name!r}: could not start the "
                    f"ACP agent {argv!r}: {exc}",
                    status_code=503,
                ) from exc
            agent = _Agent(
                client=client,
                info=info,
                auth_methods=[m.id for m in (info.auth_methods or [])],
            )
            self._agents[key] = agent
            logger.info(
                "ACP agent for provider %r: %s %s (auth: %s)",
                key,
                getattr(info.agent_info, "name", "unknown"),
                getattr(info.agent_info, "version", ""),
                ", ".join(agent.auth_methods) or "none advertised",
            )
            return agent

    async def close(self) -> None:
        """Stop every agent process this transport started."""
        for key, agent in list(self._agents.items()):
            try:
                await agent.client.stop()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                logger.debug("ACP agent %r did not stop cleanly", key, exc_info=True)
        self._agents.clear()
        self._locks.clear()

    async def aclose_provider(self, provider_name: str) -> None:
        """Stop one provider's agent, if it is running."""
        agent = self._agents.pop(provider_name, None)
        self._locks.pop(provider_name, None)
        if agent is not None:
            await agent.client.stop()

    # -- turn execution -----------------------------------------------------

    async def _open_turn(
        self, provider_info: ProviderInfo, body: dict[str, Any]
    ) -> tuple[_Turn, dict[str, Any]]:
        """Start the agent and open a session; raise :class:`AcpTurnError`.

        Opening is separate from prompting so a streaming request can learn
        that the lane refuses *before* the response headers go out: a
        signed-out agent becomes a 503, not an empty 200 SSE stream.
        """
        agent = await self._agent(provider_info)
        client = agent.client
        prompt = body.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            raise AcpTurnError("the acp turn carries no prompt", status_code=400)
        cwd = str(
            getattr(provider_info, "acp_cwd", None)
            or body.get("cwd")
            or _default_workspace(provider_info)
        )
        try:
            session = await client.new_session(cwd, [])
        except JSONRPCException as exc:
            message = str(exc)
            raise AcpTurnError(
                _sign_in_message(provider_info, message),
                status_code=503 if _looks_like_sign_in(message) else 502,
            ) from exc

        session_id = getattr(session, "session_id", "") or ""
        model = _current_model(getattr(session, "config_options", None))
        requested = str(body.get("model") or "")
        if requested and requested != model:
            try:
                await client.set_config_option(session_id, "model", requested)
                model = requested
            except JSONRPCException as exc:
                logger.warning(
                    "ACP agent for provider %r refused model %r: %s",
                    provider_info.config_name,
                    requested,
                    exc,
                )
        start = {
            "type": CHUNK_SESSION_START,
            "session_id": session_id,
            "model": model or requested,
        }
        return _Turn(
            agent=agent, session_id=session_id, model=model or requested
        ), start

    async def _run_turn(
        self,
        provider_info: ProviderInfo,
        body: dict[str, Any],
        collect: list[dict[str, Any]] | None = None,
        *,
        opened: _Turn | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Open a session (unless already open), send the prompt, then close."""
        if opened is None:
            turn, start = await self._open_turn(provider_info, body)
        else:
            turn = opened
            start = {
                "type": CHUNK_SESSION_START,
                "session_id": turn.session_id,
                "model": turn.model,
            }
        client = turn.agent.client
        session_id = turn.session_id
        if collect is not None:
            collect.append(start)
        yield start

        text = _prompt_text(body.get("prompt") or [])
        images: list[Any] = [
            b
            for b in body.get("prompt") or []
            if isinstance(b, dict) and b.get("type") == "image"
        ]
        try:
            async for update in client.prompt(session_id, text, images or None):
                chunk = _chunk_of(update)
                if chunk is None:
                    continue
                if collect is not None:
                    collect.append(chunk)
                yield chunk
            result = client.prompt_result(session_id)
        except JSONRPCException as exc:
            message = str(exc)
            error = {
                "type": CHUNK_ERROR,
                "message": _sign_in_message(provider_info, message),
            }
            if collect is not None:
                collect.append(error)
            yield error
            await _close_quietly(client, session_id)
            return

        stop = {
            "type": CHUNK_STOP,
            "stop_reason": getattr(result, "stop_reason", "") or "end_turn",
        }
        if collect is not None:
            collect.append(stop)
        yield stop
        await _close_quietly(client, session_id)

    # -- UpstreamTransport --------------------------------------------------

    async def send(
        self,
        provider_info: ProviderInfo,
        url: str,
        body: dict[str, Any],
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> UpstreamResponse:
        """Run one turn to completion and return what it produced."""
        updates: list[dict[str, Any]] = []
        try:
            async for _chunk in self._run_turn(provider_info, body, collect=updates):
                pass
        except AcpTurnError as exc:
            return UpstreamResponse(
                status_code=exc.status_code,
                body=None,
                raw_content=str(exc).encode(),
            )
        payload = {
            "updates": updates,
            "stop_reason": _stop_reason_of(updates),
            "session_id": _field_of(updates, CHUNK_SESSION_START, "session_id"),
            "model": _field_of(updates, CHUNK_SESSION_START, "model"),
            "created": int(time.time()),
        }
        return UpstreamResponse(status_code=200, body=payload, raw_content=b"")

    async def send_streaming(
        self,
        provider_info: ProviderInfo,
        url: str,
        body: dict[str, Any],
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> UpstreamStream:
        """Open a turn and expose its chunks as they arrive.

        The session is opened here rather than on the first iteration: a
        refusal then becomes ``stream.status_code`` and ``read_error()``,
        which is what a caller needs before it commits to a 200 SSE response.
        """
        try:
            turn, _start = await self._open_turn(provider_info, body)
        except AcpTurnError as exc:
            return _AcpStream.refused(exc.status_code, str(exc))
        return _AcpStream(self, provider_info, body, turn)


class _AcpStream(UpstreamStream):
    """One in-flight ACP turn, adapted to the transport's stream contract."""

    def __init__(
        self,
        transport: AcpTransport,
        provider_info: ProviderInfo,
        body: dict[str, Any],
        turn: _Turn,
    ) -> None:
        self.status_code = 200
        self._transport = transport
        self._provider_info = provider_info
        self._body = body
        self._turn = turn
        self._error = ""

    @classmethod
    def refused(cls, status_code: int, message: str) -> _AcpStream:
        """A stream that never opened: an error status and its text."""
        stream = cls.__new__(cls)
        stream.status_code = status_code
        stream._transport = None  # ty: ignore[invalid-assignment]
        stream._provider_info = None  # ty: ignore[invalid-assignment]
        stream._body = {}
        stream._turn = None  # ty: ignore[invalid-assignment]
        stream._error = message
        return stream

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[dict[str, Any]]:
        if self._turn is None:
            return
        try:
            async for chunk in self._transport._run_turn(
                self._provider_info, self._body, opened=self._turn
            ):
                yield chunk
        except AcpTurnError as exc:
            self.status_code = exc.status_code
            self._error = str(exc)
            yield {"type": CHUNK_ERROR, "message": str(exc)}

    async def read_error(self) -> str:
        return self._error

    async def close(self) -> None:
        return None


# -- helpers ---------------------------------------------------------------


def _default_workspace(provider_info: ProviderInfo) -> str:
    """Where the agent runs when the operator named no directory.

    A scratch directory per provider under the system temp dir: the agent may
    write files, and pointing it at the gateway's own working directory would
    let a model write into the deployment.
    """
    import tempfile
    from pathlib import Path

    safe = "".join(
        c if c.isalnum() or c in "-_." else "_" for c in provider_info.config_name
    )
    path = Path(tempfile.gettempdir()) / "llm-rosetta-acp" / safe
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _prompt_text(prompt: list[dict[str, Any]]) -> str:
    """Join the prompt's text blocks; images travel beside them."""
    return "\n\n".join(
        str(block.get("text") or "")
        for block in prompt
        if isinstance(block, dict) and block.get("type") == "text"
    )


def _current_model(config_options: Any) -> str:
    """Return the session's current model, when it advertises one."""
    for option in config_options or []:
        if (
            getattr(option, "id", "") == "model"
            or getattr(option, "category", None) == "model"
        ):
            return str(getattr(option, "current_value", "") or "")
    return ""


def _chunk_of(update: dict[str, Any]) -> dict[str, Any] | None:
    """Translate one ``session/update`` payload into this transport's chunk.

    ``ACPClient.prompt`` yields the notification's *params*, so the update
    itself sits under ``"update"``.  Only the update kinds the ``acp``
    converter understands are returned.  Everything else — tool calls, plans,
    mode changes, available commands — is dropped for now: the converter has
    no IR event for them in this phase, and inventing one here would report a
    capability the lane does not have.
    """
    body = update.get("update")
    if isinstance(body, dict):
        update = body
    kind = update.get("sessionUpdate") or update.get("session_update") or ""
    if kind == "agent_message_chunk":
        content = update.get("content") or {}
        if isinstance(content, dict) and content.get("type") == "text":
            return {"type": CHUNK_TEXT, "text": str(content.get("text") or "")}
        return None
    if kind == "agent_thought_chunk":
        content = update.get("content") or {}
        if isinstance(content, dict) and content.get("type") == "text":
            return {"type": CHUNK_THOUGHT, "text": str(content.get("text") or "")}
        return None
    if kind == "usage_update":
        used = update.get("used")
        if isinstance(used, int):
            return {"type": CHUNK_USAGE, "used": used, "size": update.get("size")}
        return None
    return None


def _stop_reason_of(updates: list[dict[str, Any]]) -> str:
    for chunk in reversed(updates):
        if chunk.get("type") == CHUNK_STOP:
            return str(chunk.get("stop_reason") or "end_turn")
    return "end_turn"


def _field_of(updates: list[dict[str, Any]], kind: str, field_name: str) -> str:
    for chunk in updates:
        if chunk.get("type") == kind:
            return str(chunk.get(field_name) or "")
    return ""


def _sign_in_message(provider_info: ProviderInfo, message: str) -> str:
    """Explain a refused session in the operator's terms.

    An agent that cannot open a session is usually not signed in; saying so
    alongside the agent's own words is what lets an operator act.  Any other
    refusal keeps the agent's message as written.
    """
    if not _looks_like_sign_in(message):
        return f"ACP agent for provider {provider_info.config_name!r}: {message}"
    return (
        f"ACP agent for provider {provider_info.config_name!r} could not open a "
        f"session: {message}. Sign the agent CLI in on the gateway host "
        f"(for Antigravity: run `agy` in a terminal and complete the Google "
        f"sign-in), then retry."
    )


async def _close_quietly(client: ACPClient, session_id: str) -> None:
    """Close a session, ignoring an agent that has gone away."""
    if not session_id:
        return
    try:
        await client.close_session(session_id)
    except Exception:  # noqa: BLE001 - closing must not mask the turn
        logger.debug("ACP session %s did not close cleanly", session_id, exc_info=True)


__all__ = [
    "AcpTransport",
    "AcpTurnError",
    "DEFAULT_AGENT_ARGV",
]
