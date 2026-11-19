"""The ACP transport: drive a local agent CLI over stdio JSON-RPC.

ACP is not HTTP, so this transport does not POST a body anywhere.  It keeps
an agent process per configured provider, and a *session* per conversation:
the agent holds the transcript, so the next turn of that conversation carries
only what is new.

Two things decide what happens to a request:

* **Which conversation it belongs to.**  A client that declares one
  (:attr:`SessionIdentity.declared`) gets a session of its own, which survives
  across turns.  A request that declares none opens a session, uses it, and
  closes it — a guessed identity is worse than no reuse, because one agent
  session keyed on a guess would hand one conversation's transcript to
  another.
* **Whether its history still matches what the agent was shown.**  The ledger
  (``acp_ledger``) keeps one digest per prompt block the session has seen.  A
  history that extends it reuses the session and sends the tail; anything else
  closes the session and replays the whole conversation into a fresh one.

Turns of one session are serialised: two HTTP requests for one conversation
must not interleave inside one agent session.  Turns that share a process also
take that process's lock, because one stdio connection carries them.

The protocol implementation itself is vendored (``_vendor/acp.py``, from
zerodep): the client, the JSON-RPC framing and the session queues.  What
lives here is the part that is this gateway's own: how a provider's process is
started, which agent command serves it, when a session is reused or resumed,
how a refused session is reported, and how the agent's events become chunks.
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
from .acp_ledger import (
    EXTENDS,
    AcpSessionLedger,
    LedgerEntry,
    SessionKey,
    block_digests,
)

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
    #: Held for the length of a turn.  One stdio connection carries every
    #: session of this process, so two prompts must not be written into it at
    #: once even when their sessions differ.
    lock: asyncio.Lock = field(default_factory=asyncio.Lock, repr=False)

    def can_resume(self) -> bool:
        """Whether the agent advertises ``session/resume``."""
        return _capability(self.info, "resume") is not None

    def can_load(self) -> bool:
        """Whether the agent advertises ``session/load``."""
        caps = getattr(self.info, "agent_capabilities", None)
        return bool(getattr(caps, "load_session", False))


@dataclass
class _Turn:
    """An open ACP session, ready to be prompted."""

    agent: _Agent
    session_id: str
    model: str
    #: Where this request's own blocks start: everything before it is already
    #: in the agent's transcript.
    tail_start: int = 0
    #: The ledger entry this turn keeps up to date, when it has one.
    entry: LedgerEntry | None = None
    #: Locks this turn holds, released together when it ends.
    held: list[asyncio.Lock] = field(default_factory=list, repr=False)
    #: Digests of every block the agent will have seen once this turn lands.
    digests: tuple[str, ...] = ()
    released: bool = False

    def release(self) -> None:
        """Give back the locks this turn took, once."""
        if self.released:
            return
        self.released = True
        for lock in reversed(self.held):
            if lock.locked():
                lock.release()
        self.held.clear()


def _capability(info: Any, name: str) -> Any:
    """Return one of the agent's advertised session capabilities."""
    caps = getattr(info, "agent_capabilities", None)
    session_caps = getattr(caps, "session_capabilities", None)
    return getattr(session_caps, name, None)


def _looks_like_sign_in(message: str) -> bool:
    lowered = message.lower()
    return any(hint in lowered for hint in _SIGN_IN_HINTS)


class AcpTransport:
    """Transport for providers whose upstream is a local ACP agent."""

    def __init__(
        self,
        *,
        timeout: float = 300.0,
        ledger: AcpSessionLedger | None = None,
    ) -> None:
        self._timeout = timeout
        self._agents: dict[str, _Agent] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._ledger = ledger or AcpSessionLedger()

    @property
    def ledger(self) -> AcpSessionLedger:
        """The session ledger this transport keeps."""
        return self._ledger

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
                "ACP agent for provider %r: %s %s (auth: %s; resume: %s, load: %s)",
                key,
                getattr(info.agent_info, "name", "unknown"),
                getattr(info.agent_info, "version", ""),
                ", ".join(agent.auth_methods) or "none advertised",
                agent.can_resume(),
                agent.can_load(),
            )
            return agent

    async def close(self) -> None:
        """Stop every agent process this transport started."""
        for key, agent in list(self._agents.items()):
            try:
                await agent.client.stop()
            except Exception:  # noqa: BLE001 - shutdown must not raise
                logger.debug("ACP agent %r did not stop cleanly", key, exc_info=True)
            self._ledger.drop_provider(key)
        self._agents.clear()
        self._locks.clear()

    async def aclose_provider(self, provider_name: str) -> None:
        """Stop one provider's agent, and forget its sessions."""
        agent = self._agents.pop(provider_name, None)
        self._locks.pop(provider_name, None)
        self._ledger.drop_provider(provider_name)
        if agent is not None:
            await agent.client.stop()

    # -- the conversation a request belongs to ------------------------------

    def _key(self, provider_info: ProviderInfo) -> SessionKey | None:
        """The ledger key for this request, or ``None`` for a one-off session.

        A lane carries a conversation id only when the client declared one
        (see ``routing.session``); without it the request gets a session of
        its own, which is closed when the turn ends.
        """
        conversation = getattr(provider_info, "acp_conversation_id", None)
        if not conversation:
            return None
        return SessionKey(provider=provider_info.config_name, conversation=conversation)

    # -- acquiring a session ------------------------------------------------

    async def _acquire_turn(
        self, provider_info: ProviderInfo, body: dict[str, Any]
    ) -> tuple[_Turn, dict[str, Any]]:
        """Find or open the session this request belongs to.

        Raises:
            AcpTurnError: The prompt is empty, the agent could not start, or
                it refused to open a session.
        """
        prompt = body.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            raise AcpTurnError("the acp turn carries no prompt", status_code=400)
        agent = await self._agent(provider_info)
        digests = block_digests(prompt)
        key = self._key(provider_info)
        held: list[asyncio.Lock] = []

        try:
            if key is not None:
                session_lock = self._ledger.lock_for(key)
                await session_lock.acquire()
                held.append(session_lock)
            await agent.lock.acquire()
            held.append(agent.lock)
            if key is not None:
                turn = await self._reuse(provider_info, body, key, agent, digests, held)
                if turn is not None:
                    return turn, _start_of(turn)
                self._ledger.drop(key)
            turn = await self._open(provider_info, body, key, agent, digests, held)
            return turn, _start_of(turn)
        except BaseException:
            for lock in reversed(held):
                if lock.locked():
                    lock.release()
            raise

    async def _reuse(
        self,
        provider_info: ProviderInfo,
        body: dict[str, Any],
        key: SessionKey,
        agent: _Agent,
        digests: tuple[str, ...],
        held: list[asyncio.Lock],
    ) -> _Turn | None:
        """Continue the recorded session, when the request may ride on it."""
        entry = self._ledger.get(key)
        if entry is None:
            return None
        cwd = _workspace(provider_info, body)
        if entry.cwd != cwd:
            # A different working directory is a different workspace; the
            # agent's session cannot be moved, so it is not this request's.
            await self._forget(entry)
            return None
        if self._ledger.relationship_for(entry, digests) != EXTENDS:
            await self._forget(entry)
            return None
        if not entry.live:
            resumed = await self._resume(provider_info, agent, entry, cwd)
            if resumed is None:
                await self._forget(entry)
                return None
        model = await self._match_model(provider_info, agent, entry, body)
        entry.touch()
        return _Turn(
            agent=agent,
            session_id=entry.session_id,
            model=model,
            tail_start=len(entry.seen),
            entry=entry,
            held=held,
            digests=digests,
        )

    async def _resume(
        self, provider_info: ProviderInfo, agent: _Agent, entry: LedgerEntry, cwd: str
    ) -> str | None:
        """Bring a recorded session back, or report that it is gone.

        ``session/resume`` continues without replaying; ``session/load``
        replays the transcript first, and those updates are discarded here —
        the client is not asking for its own history back.
        """
        if agent.can_resume():
            try:
                await agent.client.resume_session(entry.session_id, cwd, [])
            except JSONRPCException as exc:
                logger.info(
                    "ACP provider %r could not resume session %s: %s",
                    provider_info.config_name,
                    entry.session_id,
                    exc,
                )
                return None
            entry.live = True
            return entry.session_id
        if agent.can_load():
            try:
                await agent.client.load_session(entry.session_id, cwd, [])
            except JSONRPCException as exc:
                logger.info(
                    "ACP provider %r could not load session %s: %s",
                    provider_info.config_name,
                    entry.session_id,
                    exc,
                )
                return None
            replayed = _drain_queue(agent.client, entry.session_id)
            if replayed:
                logger.debug(
                    "ACP provider %r replayed %d updates loading session %s",
                    provider_info.config_name,
                    replayed,
                    entry.session_id,
                )
            entry.live = True
            return entry.session_id
        logger.info(
            "ACP provider %r advertises neither session/resume nor session/load; "
            "conversation %s replays into a fresh session",
            provider_info.config_name,
            entry.key.conversation,
        )
        return None

    async def _open(
        self,
        provider_info: ProviderInfo,
        body: dict[str, Any],
        key: SessionKey | None,
        agent: _Agent,
        digests: tuple[str, ...],
        held: list[asyncio.Lock],
    ) -> _Turn:
        """Open a new session and send the whole conversation into it."""
        cwd = _workspace(provider_info, body)
        try:
            session = await agent.client.new_session(cwd, [])
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
            model = await self._switch_model(
                provider_info, agent, session_id, requested
            )
        turn = _Turn(
            agent=agent,
            session_id=session_id,
            model=model or requested,
            tail_start=0,
            held=held,
            digests=digests,
        )
        if key is not None:
            entry = LedgerEntry(
                key=key,
                session_id=session_id,
                cwd=cwd,
                model=turn.model,
                seen=tuple(digests),
                live=True,
            )
            turn.entry = entry
            for displaced in self._ledger.put(entry):
                await self._close_entry(displaced)
        return turn

    async def _match_model(
        self,
        provider_info: ProviderInfo,
        agent: _Agent,
        entry: LedgerEntry,
        body: dict[str, Any],
    ) -> str:
        """Bring the session onto the model this request names, if it differs."""
        requested = str(body.get("model") or "")
        if not requested or requested == entry.model:
            return entry.model or requested
        model = await self._switch_model(
            provider_info, agent, entry.session_id, requested
        )
        entry.model = model or entry.model
        return entry.model or requested

    async def _switch_model(
        self,
        provider_info: ProviderInfo,
        agent: _Agent,
        session_id: str,
        requested: str,
    ) -> str:
        """Ask the agent to change the session's model.

        A refusal is not fatal: the turn then runs on the model the session
        already has, which the caller reports as the response's model rather
        than pretending the requested one is in force.
        """
        try:
            await agent.client.set_config_option(session_id, "model", requested)
        except JSONRPCException as exc:
            logger.warning(
                "ACP agent for provider %r refused model %r: %s",
                provider_info.config_name,
                requested,
                exc,
            )
            return ""
        return requested

    # -- turn execution -----------------------------------------------------

    async def _run_turn(
        self,
        provider_info: ProviderInfo,
        body: dict[str, Any],
        collect: list[dict[str, Any]] | None = None,
        *,
        opened: _Turn | None = None,
        start: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Send this request's prompt blocks into the session, then settle."""
        turn = opened
        if turn is None:
            turn, start = await self._acquire_turn(provider_info, body)
        if start is None:
            start = _start_of(turn)
        client = turn.agent.client
        session_id = turn.session_id
        if collect is not None:
            collect.append(start)
        try:
            yield start

            blocks = [b for b in body.get("prompt") or [] if isinstance(b, dict)]
            sending = blocks[turn.tail_start :]
            text = _prompt_text(sending)
            images: list[Any] = [b for b in sending if b.get("type") == "image"]
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
                # What the agent recorded of a failed turn is unknown, so the
                # session cannot be reused: the next turn opens a fresh one.
                await self._forget(turn.entry)
                error = {
                    "type": CHUNK_ERROR,
                    "message": _sign_in_message(provider_info, str(exc)),
                }
                if collect is not None:
                    collect.append(error)
                yield error
                await _close_quietly(client, session_id)
                return
            self._settle(turn, provider_info)
            stop = {
                "type": CHUNK_STOP,
                "stop_reason": getattr(result, "stop_reason", "") or "end_turn",
            }
            if collect is not None:
                collect.append(stop)
            yield stop
        finally:
            turn.release()

    def _settle(self, turn: _Turn, provider_info: ProviderInfo) -> None:
        """Record what the agent has now seen, or close a one-off session."""
        entry = turn.entry
        if entry is None:
            return
        entry.remember(turn.digests, max_blocks=self._ledger.max_blocks)
        entry.live = True
        entry.touch()
        if not entry.seen:
            # Too long to track: the session is closed rather than left to
            # accumulate a record the next turn could never compare against.
            logger.info(
                "ACP provider %r: conversation %s is past %d blocks; "
                "its session will not be reused",
                provider_info.config_name,
                entry.key.conversation,
                self._ledger.max_blocks,
            )

    async def _forget(self, entry: LedgerEntry | None) -> None:
        """Close a session and forget it, so the next turn starts fresh."""
        if entry is None:
            return
        self._ledger.drop(entry.key)
        await self._close_entry(entry)

    async def _close_entry(self, entry: LedgerEntry) -> None:
        """Close one recorded session."""
        entry.live = False
        agent = self._agents.get(entry.key.provider)
        await _close_quietly(agent.client if agent else None, entry.session_id)

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

        The session is acquired here rather than on the first iteration: a
        refusal then becomes ``stream.status_code`` and ``read_error()``,
        which is what a caller needs before it commits to a 200 SSE response.
        """
        try:
            turn, start = await self._acquire_turn(provider_info, body)
        except AcpTurnError as exc:
            return _AcpStream.refused(exc.status_code, str(exc))
        return _AcpStream(self, provider_info, body, turn, start)


class _AcpStream(UpstreamStream):
    """One in-flight ACP turn, adapted to the transport's stream contract."""

    def __init__(
        self,
        transport: AcpTransport,
        provider_info: ProviderInfo,
        body: dict[str, Any],
        turn: _Turn,
        start: dict[str, Any],
    ) -> None:
        self.status_code = 200
        self._transport = transport
        self._provider_info = provider_info
        self._body = body
        self._turn = turn
        self._start = start
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
        stream._start = None
        stream._error = message
        return stream

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[dict[str, Any]]:
        if self._turn is None:
            return
        try:
            async for chunk in self._transport._run_turn(
                self._provider_info, self._body, opened=self._turn, start=self._start
            ):
                yield chunk
        except AcpTurnError as exc:
            self.status_code = exc.status_code
            self._error = str(exc)
            yield {"type": CHUNK_ERROR, "message": str(exc)}

    async def read_error(self) -> str:
        return self._error

    async def close(self) -> None:
        if self._turn is not None:
            self._turn.release()


# -- helpers ---------------------------------------------------------------


def _start_of(turn: _Turn) -> dict[str, Any]:
    """The ``session_start`` chunk for a turn that is already open."""
    return {
        "type": CHUNK_SESSION_START,
        "session_id": turn.session_id,
        "model": turn.model,
    }


def _workspace(provider_info: ProviderInfo, body: dict[str, Any]) -> str:
    """Where the agent runs for this request."""
    return str(
        getattr(provider_info, "acp_cwd", None)
        or body.get("cwd")
        or _default_workspace(provider_info)
    )


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


def _drain_queue(client: ACPClient, session_id: str) -> int:
    """Discard updates an agent replayed while loading a session.

    ``session/load`` replays the transcript as ``session/update``
    notifications before it answers, so by the time the call returns they are
    already queued.  The client is not asking for its own history back, and
    forwarding it would duplicate the whole conversation into the reply.
    """
    pending_of = getattr(client, "_existing_queue", None)
    if pending_of is None:
        return 0
    pending = pending_of(session_id)
    if pending is None:
        return 0
    drained = 0
    while not pending.empty():
        try:
            pending.get_nowait()
        except asyncio.QueueEmpty:  # pragma: no cover - empty() raced
            break
        drained += 1
    return drained


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


async def _close_quietly(client: ACPClient | None, session_id: str) -> None:
    """Close a session, ignoring an agent that has gone away."""
    if client is None or not session_id:
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
