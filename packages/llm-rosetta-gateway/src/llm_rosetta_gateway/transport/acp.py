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

#: How long a *cleanup* request — ``session/cancel``, ``session/close`` — may
#: take.  Cleanup runs on the way out of a turn, often because the caller was
#: cancelled, and an unresponsive agent must not be able to park it: the locks
#: the turn holds are only given back after the cleanup returns, so a cleanup
#: that waits forever is a slot that never comes back.
CLEANUP_TIMEOUT = 5.0


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
    slot: int = 0
    #: The provider config name this process belongs to.  Held here so a
    #: process that dies can be taken out of its own pool without the caller
    #: having to remember which pool that was.
    provider: str = ""
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
    #: Whether the turn reached its stop (or its error) chunk.  A turn that
    #: did not is abandoned: the agent may have recorded half of it, so the
    #: session is closed rather than reused.
    finished: bool = False

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


def _agent_gone(agent: _Agent) -> bool:
    """Whether the process behind *agent* has exited.

    Read from the vendored client's own state, which it does not expose as a
    property: the reader task ends at end-of-file, and the process carries a
    return code once it has exited.
    """
    task = getattr(agent.client, "_reader_task", None)
    if task is not None and task.done():
        return True
    process = getattr(agent.client, "_process", None)
    return process is not None and process.returncode is not None


class AcpTransport:
    """Transport for providers whose upstream is a local ACP agent."""

    def __init__(
        self,
        *,
        timeout: float = 300.0,
        ledger: AcpSessionLedger | None = None,
    ) -> None:
        #: How long one turn may take before the agent is asked to stop.
        #: ACP turns have no deadline of their own, and an agent that stops
        #: answering would otherwise hold this session's lock — and, with one
        #: process, the lane — until the transport closes.
        self._timeout = timeout
        # One *pool* per provider: the operator may run more than one agent
        # process for a provider, and a session is bound to the process that
        # holds it for as long as it lives.
        self._agents: dict[str, list[_Agent | None]] = {}
        self._starting: dict[str, asyncio.Lock] = {}
        self._ledger = ledger or AcpSessionLedger()

    @property
    def ledger(self) -> AcpSessionLedger:
        """The session ledger this transport keeps."""
        return self._ledger

    # -- agent lifecycle ----------------------------------------------------

    def _argv(self, provider_info: ProviderInfo) -> list[str]:
        argv = getattr(provider_info, "acp_argv", None)
        return list(argv) if argv else list(DEFAULT_AGENT_ARGV)

    def _processes(self, provider_info: ProviderInfo) -> int:
        """How many agent processes this provider may run."""
        return max(1, int(getattr(provider_info, "acp_processes", 1) or 1))

    async def _agent(self, provider_info: ProviderInfo, slot: int = 0) -> _Agent:
        """Start (once) and return the pool member *slot* for this provider."""
        key = provider_info.config_name
        pool = self._agents.setdefault(key, [])
        while len(pool) <= slot:
            pool.append(None)
        agent = pool[slot]
        if agent is not None:
            return agent
        lock = self._starting.setdefault(f"{key}#{slot}", asyncio.Lock())
        async with lock:
            agent = pool[slot]
            if agent is not None:
                return agent
            agent = await self._start(provider_info, slot)
            pool[slot] = agent
            return agent

    async def _start(self, provider_info: ProviderInfo, slot: int) -> _Agent:
        """Start one agent process and complete the handshake."""
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
            slot=slot,
            provider=provider_info.config_name,
            info=info,
            auth_methods=[m.id for m in (info.auth_methods or [])],
        )
        logger.info(
            "ACP agent for provider %r (process %d): %s %s (auth: %s; "
            "resume: %s, load: %s)",
            provider_info.config_name,
            slot,
            getattr(info.agent_info, "name", "unknown"),
            getattr(info.agent_info, "version", ""),
            ", ".join(agent.auth_methods) or "none advertised",
            agent.can_resume(),
            agent.can_load(),
        )
        return agent

    def _slot_for(self, provider_info: ProviderInfo, entry: LedgerEntry | None) -> int:
        """Which process serves this request.

        A session keeps the process that opened it.  A new session goes to the
        process with the fewest sessions bound to it, so a provider that runs
        several spreads its conversations rather than piling them on one.
        """
        processes = self._processes(provider_info)
        if entry is not None and entry.agent_slot < processes:
            return entry.agent_slot
        counts = [0] * processes
        for held in self._ledger.entries_of(provider_info.config_name):
            if held.agent_slot < processes:
                counts[held.agent_slot] += 1
        return counts.index(min(counts))

    def _pool(self, provider_name: str) -> list[_Agent]:
        return [a for a in self._agents.get(provider_name, []) if a is not None]

    async def close(self) -> None:
        """Stop every agent process this transport started."""
        for key, pool in list(self._agents.items()):
            for agent in pool:
                if agent is None:
                    continue
                try:
                    await agent.client.stop()
                except Exception:  # noqa: BLE001 - shutdown must not raise
                    logger.debug(
                        "ACP agent %r did not stop cleanly", key, exc_info=True
                    )
            self._ledger.drop_provider(key)
        self._agents.clear()
        self._starting.clear()

    async def aclose_provider(self, provider_name: str) -> None:
        """Stop one provider's agent processes, and forget its sessions.

        The counterpart of re-pointing a provider at another account: the
        processes were started with the old ``acp_argv`` and hold sessions of
        the old sign-in, so both go.  Nothing calls it yet — a config reload
        rebuilds the whole app today — and it stays because the reload slice
        needs a way to drop one lane without dropping the process that serves
        the others.
        """
        pool = self._agents.pop(provider_name, [])
        # ``<provider>#<slot>`` — split rather than prefix-match, so a provider
        # named ``anti`` does not clear the start locks of ``antigravity``.
        self._starting = {
            k: v
            for k, v in self._starting.items()
            if k.split("#", 1)[0] != provider_name
        }
        self._ledger.drop_provider(provider_name)
        for agent in pool:
            if agent is not None:
                await agent.client.stop()

    async def sweep(self) -> int:
        """Give back sessions nobody has used for their provider's TTL.

        Called periodically rather than only on the next claim: an idle
        gateway would otherwise keep every agent session — and the transcript
        the agent holds for it — alive until traffic returned.

        Returns:
            How many sessions were closed.
        """
        closed = 0
        for entry in self._ledger.expire():
            await self._close_entry(entry)
            closed += 1
        return closed

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
        self, provider_info: ProviderInfo, body: dict[str, Any], deadline: float
    ) -> tuple[_Turn, dict[str, Any]]:
        """Find or open the session this request belongs to.

        Session setup is part of the turn and runs against the same deadline:
        ``session/new``, ``session/resume``, ``session/load`` and
        ``set_config_option`` are requests to the same agent as the prompt, and
        an agent that is sick at session open would otherwise never reach the
        deadline meant to bound it.  A setup that runs out of time drops the
        process, the way an exited one is dropped.

        Raises:
            AcpTurnError: The prompt is empty, the agent could not start, it
                refused to open a session, or opening one ran past *deadline*.
        """
        prompt = body.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            raise AcpTurnError("the acp turn carries no prompt", status_code=400)
        # A process that exited between turns still occupies its slot; taking
        # it out first is what makes this request start a fresh one instead of
        # writing into a dead pipe.
        self._prune_gone(provider_info.config_name)
        key = self._key(provider_info)
        entry = self._ledger.get(key) if key is not None else None
        agent = await self._agent(provider_info, self._slot_for(provider_info, entry))
        digests = block_digests(_blocks_of(body))
        held: list[asyncio.Lock] = []

        try:
            if key is not None:
                session_lock = self._ledger.lock_for(key)
                await session_lock.acquire()
                held.append(session_lock)
            await agent.lock.acquire()
            held.append(agent.lock)
            if key is not None:
                turn = await self._reuse(
                    provider_info, body, key, agent, digests, held, deadline
                )
                if turn is not None:
                    return turn, _start_of(turn)
                self._ledger.drop(key)
            turn = await self._open(
                provider_info, body, key, agent, digests, held, deadline
            )
            return turn, _start_of(turn)
        except _TurnDeadline:
            for lock in reversed(held):
                if lock.locked():
                    lock.release()
            # A process that cannot open a session inside the turn's budget is
            # not usable for this request or the next: it stays half-read in
            # its slot, and every later turn queues behind it.
            self._drop_agent(agent)
            raise AcpTurnError(
                f"ACP agent for provider {provider_info.config_name!r} did not "
                f"open a session within {self._timeout:.1f}s; the process was "
                f"dropped. Check the agent on the gateway host.",
                status_code=503,
            ) from None
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
        deadline: float,
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
            resumed = await self._resume(provider_info, agent, entry, cwd, deadline)
            if resumed is None:
                await self._forget(entry)
                return None
        model = await self._match_model(provider_info, agent, entry, body, deadline)
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
        self,
        provider_info: ProviderInfo,
        agent: _Agent,
        entry: LedgerEntry,
        cwd: str,
        deadline: float,
    ) -> str | None:
        """Bring a recorded session back, or report that it is gone.

        ``session/resume`` continues without replaying; ``session/load``
        replays the transcript first, and those updates are discarded here —
        the client is not asking for its own history back.
        """
        if agent.can_resume():
            try:
                await _bounded(
                    agent.client.resume_session(entry.session_id, cwd, []), deadline
                )
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
                await _bounded(
                    agent.client.load_session(entry.session_id, cwd, []), deadline
                )
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
        deadline: float,
    ) -> _Turn:
        """Open a new session and send the whole conversation into it."""
        cwd = _workspace(provider_info, body)
        try:
            session = await _bounded(agent.client.new_session(cwd, []), deadline)
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
                provider_info, agent, session_id, requested, deadline
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
            self._ledger.set_limits(
                key.provider,
                max_sessions=getattr(provider_info, "acp_max_sessions", None),
                idle_ttl=getattr(provider_info, "acp_session_ttl", None),
            )
            entry = LedgerEntry(
                key=key,
                session_id=session_id,
                cwd=cwd,
                model=turn.model,
                seen=tuple(digests),
                agent_slot=agent.slot,
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
        deadline: float,
    ) -> str:
        """Bring the session onto the model this request names, if it differs."""
        requested = str(body.get("model") or "")
        if not requested or requested == entry.model:
            return entry.model or requested
        model = await self._switch_model(
            provider_info, agent, entry.session_id, requested, deadline
        )
        entry.model = model or entry.model
        return entry.model or requested

    async def _switch_model(
        self,
        provider_info: ProviderInfo,
        agent: _Agent,
        session_id: str,
        requested: str,
        deadline: float,
    ) -> str:
        """Ask the agent to change the session's model.

        A refusal is not fatal: the turn then runs on the model the session
        already has, which the caller reports as the response's model rather
        than pretending the requested one is in force.
        """
        try:
            await _bounded(
                agent.client.set_config_option(session_id, "model", requested),
                deadline,
            )
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
        deadline: float | None = None,
        opened: _Turn | None = None,
        start: dict[str, Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Send this request's prompt blocks into the session, then settle.

        *deadline* is the end of the whole turn's budget — setup included — as
        an event-loop time.  The caller computes it before acquiring a
        session, so an agent that is slow at session open cannot spend the
        prompt's time without being seen to.
        """
        if deadline is None:
            deadline = asyncio.get_running_loop().time() + self._timeout
        turn = opened
        if turn is None:
            turn, start = await self._acquire_turn(provider_info, body, deadline)
        if start is None:
            start = _start_of(turn)
        client = turn.agent.client
        session_id = turn.session_id
        if collect is not None:
            collect.append(start)
        try:
            yield start

            blocks = _blocks_of(body)
            sending = blocks[turn.tail_start :]
            text = _prompt_text(sending)
            images: list[Any] = [b for b in sending if b.get("type") == "image"]
            try:
                async for update in _prompt_updates(
                    client, session_id, text, images, deadline
                ):
                    chunk = _chunk_of(update)
                    if chunk is None:
                        continue
                    if collect is not None:
                        collect.append(chunk)
                    yield chunk
                result = client.prompt_result(session_id)
                if result is None:
                    # End of iteration is not a completion.  The vendored
                    # client signals end-of-process and end-of-turn with the
                    # same sentinel, and only a turn that reached its stop (or
                    # its ack) stores a result — so no result means the agent
                    # went away mid-turn.  A stop chunk here would tell the
                    # client the model finished answering.
                    turn.finished = True
                    await self._forget(turn.entry)
                    self._drop_agent(turn.agent)
                    message = (
                        f"ACP agent for provider {provider_info.config_name!r} "
                        f"ended the turn without reporting a stop; the "
                        f"conversation was closed. The agent process may have "
                        f"exited — check it on the gateway host."
                    )
                    yield _failure(collect, message)
                    return
            except _TurnDeadline:
                await self._forget(turn.entry)
                turn.finished = True
                # A process that let one turn run past its deadline keeps its
                # half-read pipe; leaving it in the pool makes every later
                # turn wait behind it.  Dropped, the next turn starts a fresh
                # process instead.
                self._drop_agent(turn.agent)
                await _cancel_quietly(client, session_id)
                message = (
                    f"ACP agent for provider {provider_info.config_name!r} did not "
                    f"finish the turn within {self._timeout:.1f}s; the session was "
                    f"closed."
                )
                yield _failure(collect, message)
                return
            except JSONRPCException as exc:
                # What the agent recorded of a failed turn is unknown, so the
                # session cannot be reused: the next turn opens a fresh one.
                turn.finished = True
                await self._forget(turn.entry)
                yield _failure(collect, _sign_in_message(provider_info, str(exc)))
                await _close_quietly(client, session_id)
                return
            turn.finished = True
            await self._settle(turn, provider_info)
            stop = {
                "type": CHUNK_STOP,
                "stop_reason": getattr(result, "stop_reason", "") or "end_turn",
            }
            if collect is not None:
                collect.append(stop)
            yield stop
        finally:
            try:
                if not turn.finished:
                    # The caller went away mid-turn — a closed tab, a
                    # cancelled response.  What the agent recorded is unknown,
                    # so the session is given back rather than reused with a
                    # hole in it.
                    await self._abandon(turn)
            finally:
                # Release even when the cleanup above is itself cancelled or
                # fails: the locks are the slot, and a slot that is never
                # given back takes the lane down with it.
                turn.release()

    async def _abandon(self, turn: _Turn) -> None:
        """Give back a session whose turn did not finish."""
        turn.finished = True
        await self._forget(turn.entry)
        await _close_quietly(turn.agent.client, turn.session_id)

    async def _settle(self, turn: _Turn, provider_info: ProviderInfo) -> None:
        """Record what the agent has now seen, or close a one-off session."""
        entry = turn.entry
        if entry is None:
            # A request that declared no conversation gets one session for its
            # own, and nothing will ever come back for it.
            await _close_quietly(turn.agent.client, turn.session_id)
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
        """Close one recorded session on the process that holds it."""
        entry.live = False
        agent = self._agent_at(entry.key.provider, entry.agent_slot)
        await _close_quietly(agent.client if agent else None, entry.session_id)

    def _agent_at(self, provider_name: str, slot: int) -> _Agent | None:
        """The pool member at *slot*, without starting one that is not running."""
        pool = self._agents.get(provider_name) or []
        if 0 <= slot < len(pool):
            return pool[slot]
        return None

    def _drop_agent(self, agent: _Agent) -> None:
        """Take a process that has gone away out of its pool.

        The vendored client reports end-of-process as the end of a turn's
        iteration, and a client whose process has exited answers every later
        request with a connection error while still occupying its slot.  The
        slot is emptied here so the next turn starts a fresh process, and the
        sessions it held are marked dead so they resume — or replay — instead
        of being prompted into a corpse.
        """
        pool = self._agents.get(agent.provider) or []
        if 0 <= agent.slot < len(pool) and pool[agent.slot] is agent:
            pool[agent.slot] = None
        for held in self._ledger.entries_of(agent.provider):
            if held.agent_slot == agent.slot:
                held.live = False

    def _prune_gone(self, provider_name: str) -> None:
        """Evict every process of *provider_name* whose process has exited."""
        for agent in self._pool(provider_name):
            if _agent_gone(agent):
                self._drop_agent(agent)

    # -- UpstreamTransport --------------------------------------------------

    async def send(
        self,
        provider_info: ProviderInfo,
        url: str,
        body: dict[str, Any],
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> UpstreamResponse:
        """Run one turn to completion and return what it produced.

        A turn that failed is reported as a failure, not as an empty answer: a
        non-streaming client cannot tell "the model said nothing" from "the
        agent broke", so the error chunk becomes the response's status and
        text, the same way a refused session does.
        """
        updates: list[dict[str, Any]] = []
        # One budget for the whole turn.  Computed before the session is
        # acquired, because opening one is a request to the same agent.
        deadline = asyncio.get_running_loop().time() + self._timeout
        try:
            async for _chunk in self._run_turn(
                provider_info, body, collect=updates, deadline=deadline
            ):
                pass
        except AcpTurnError as exc:
            return UpstreamResponse(
                status_code=exc.status_code,
                body=None,
                raw_content=str(exc).encode(),
            )
        except OSError as exc:
            # A write into an exited process raises here rather than as a turn
            # failure.  It must still leave through the response contract: an
            # exception out of ``send`` skips the status the proxy reports on,
            # and the dead process keeps its slot until the app restarts.
            self._prune_gone(provider_info.config_name)
            return UpstreamResponse(
                status_code=502,
                body=None,
                raw_content=(
                    f"ACP agent for provider {provider_info.config_name!r} "
                    f"could not be reached: {exc}"
                ).encode(),
            )
        failure = _error_of(updates)
        if failure is not None:
            return UpstreamResponse(
                status_code=503 if _looks_like_sign_in(failure) else 502,
                body=None,
                raw_content=failure.encode(),
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
            # The budget is set before the session is opened: opening one is a
            # request to the same agent, and it is passed on to the prompt so
            # both spend the same allowance.
            deadline = asyncio.get_running_loop().time() + self._timeout
            turn, start = await self._acquire_turn(provider_info, body, deadline)
        except AcpTurnError as exc:
            return _AcpStream.refused(exc.status_code, str(exc))
        except OSError as exc:
            self._prune_gone(provider_info.config_name)
            return _AcpStream.refused(
                502,
                f"ACP agent for provider {provider_info.config_name!r} could not "
                f"be reached: {exc}",
            )
        return _AcpStream(self, provider_info, body, turn, start, deadline)


class _AcpStream(UpstreamStream):
    """One in-flight ACP turn, adapted to the transport's stream contract."""

    def __init__(
        self,
        transport: AcpTransport,
        provider_info: ProviderInfo,
        body: dict[str, Any],
        turn: _Turn,
        start: dict[str, Any],
        deadline: float,
    ) -> None:
        self.status_code = 200
        self._transport = transport
        self._provider_info = provider_info
        self._body = body
        self._turn = turn
        self._start = start
        self._deadline = deadline
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
        stream._deadline = 0.0
        stream._error = message
        return stream

    def __aiter__(self) -> AsyncIterator[dict[str, Any]]:
        return self._iter()

    async def _iter(self) -> AsyncIterator[dict[str, Any]]:
        if self._turn is None:
            return
        try:
            async for chunk in self._transport._run_turn(
                self._provider_info,
                self._body,
                deadline=self._deadline,
                opened=self._turn,
                start=self._start,
            ):
                if self._turn is None:
                    # ``close()`` ran while this turn was in flight; the
                    # session has been given back, so nothing more is read.
                    return
                yield chunk
        except AcpTurnError as exc:
            self.status_code = exc.status_code
            self._error = str(exc)
            yield {"type": CHUNK_ERROR, "message": str(exc)}
        finally:
            # A consumer that stops reading — a closed tab, a cancelled
            # response — still has to give the session back, or the turn's
            # locks are held until the transport closes.
            await self.close()

    async def read_error(self) -> str:
        return self._error

    async def close(self) -> None:
        """Release the turn, giving its session back if it never finished."""
        turn = self._turn
        if turn is None:
            return
        self._turn = None
        if not turn.finished:
            await self._transport._abandon(turn)
        turn.release()


# -- helpers ---------------------------------------------------------------


async def _prompt_updates(
    client: ACPClient,
    session_id: str,
    text: str,
    images: list[Any],
    deadline: float,
) -> AsyncIterator[dict[str, Any]]:
    """Yield the turn's updates, giving up when the deadline passes.

    ACP has no turn timeout, and an agent that stops answering keeps the
    session — and the process lock that serialises the lane — forever.  The
    wait is per update rather than around the whole loop so the deadline is
    also a *stall* deadline: a turn that keeps producing is allowed to finish.

    Raises:
        _TurnDeadline: No update arrived before *deadline*.
    """
    updates = client.prompt(session_id, text, images or None).__aiter__()
    loop = asyncio.get_running_loop()
    try:
        while True:
            remaining = deadline - loop.time()
            if remaining <= 0:
                raise _TurnDeadline
            try:
                yield await asyncio.wait_for(updates.__anext__(), remaining)
            except StopAsyncIteration:
                return
            except asyncio.TimeoutError as exc:  # noqa: UP041 - 3.10 spelling
                raise _TurnDeadline from exc
    finally:
        # Leaving early — a deadline, or a caller that stopped reading — must
        # release the agent's queue, or the vendored client keeps a getter
        # waiting on a turn nobody will read.
        aclose = getattr(updates, "aclose", None)
        if aclose is not None:
            try:
                await aclose()
            except Exception:  # noqa: BLE001 - closing must not mask the turn
                logger.debug("ACP prompt stream did not close cleanly", exc_info=True)


class _TurnDeadline(Exception):
    """The agent stopped answering inside the turn's deadline."""


async def _bounded(coro: Any, deadline: float) -> Any:
    """Await one session-setup call within what is left of the turn's budget.

    Setup is a request to the same agent as the prompt, so it spends the same
    allowance; an agent that never answers ``session/new`` would otherwise
    hold the caller — and, through the process lock, every later turn on that
    lane — with no deadline in force at all.

    Raises:
        _TurnDeadline: *deadline* passed before the call answered.
    """
    remaining = deadline - asyncio.get_running_loop().time()
    try:
        return await asyncio.wait_for(coro, max(remaining, 0.0))
    except asyncio.TimeoutError as exc:  # noqa: UP041 - 3.10 spelling
        raise _TurnDeadline from exc


async def _cancel_quietly(client: ACPClient, session_id: str) -> None:
    """Ask the agent to stop the turn, then close the session.

    ``session/cancel`` is a *notification*: it never waits for a reply, so it
    cannot park on an agent that has stopped reading — the bound here only
    covers a write that blocks on a full pipe.  ``session/close`` is the
    request that can wait forever, and that is where the bound earns its
    keep; both are bounded because this runs on a lane whose turn has already
    failed or been cancelled and must not keep the turn's locks.
    """
    try:
        await asyncio.wait_for(client.cancel(session_id), CLEANUP_TIMEOUT)
    except Exception:  # noqa: BLE001 - a stuck agent must not raise here
        logger.debug("ACP session %s did not cancel cleanly", session_id)
    await _close_quietly(client, session_id)


def _blocks_of(body: dict[str, Any]) -> list[dict[str, Any]]:
    """The prompt blocks this turn sends, system instruction first.

    The converter keeps the system instruction in its own body field, because
    that is the shape an IR round-trip reads back; an agent needs it inside
    the prompt, where every other thing it is told lives.  Rendering it as the
    first block — labelled the way :func:`render_messages` labels a system
    message — is what stops a chat client's system instruction from vanishing:
    a lane that silently drops it is the one behaviour a chat lane must not
    have.

    Because the block is first and never moves, the ledger's digest of "what
    the agent has seen" covers it on every turn, and a session opened before
    it changed diverges rather than continuing.
    """
    blocks: list[dict[str, Any]] = []
    system = body.get("system")
    if isinstance(system, str) and system.strip():
        blocks.append({"type": "text", "text": f"[System]\n{system}"})
    blocks.extend(
        block for block in body.get("prompt") or [] if isinstance(block, dict)
    )
    return blocks


def _failure(collect: list[dict[str, Any]] | None, message: str) -> dict[str, Any]:
    """The turn's error chunk, recorded for a caller that collects chunks."""
    chunk = {"type": CHUNK_ERROR, "message": message}
    if collect is not None:
        collect.append(chunk)
    return chunk


def _error_of(updates: list[dict[str, Any]]) -> str | None:
    """The message of the turn's error chunk, if it produced one."""
    for chunk in updates:
        if chunk.get("type") == CHUNK_ERROR:
            return str(chunk.get("message") or "") or "the ACP turn failed"
    return None


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
    """Close a session, ignoring an agent that has gone away.

    Bounded for the same reason the cancel is: a close is a request to the
    same agent, so an agent that has stopped reading would otherwise keep the
    turn — and the slot behind it — until the transport closes.
    """
    if client is None or not session_id:
        return
    try:
        await asyncio.wait_for(client.close_session(session_id), CLEANUP_TIMEOUT)
    except Exception:  # noqa: BLE001 - closing must not mask the turn
        logger.debug("ACP session %s did not close cleanly", session_id, exc_info=True)


__all__ = [
    "AcpTransport",
    "AcpTurnError",
    "DEFAULT_AGENT_ARGV",
]
