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

Client tools take a second path.  A request's tool declarations are served to
the agent as a loopback MCP endpoint named on ``session/new`` (``acp_mcp``).
When the agent calls one, the turn *suspends*: the transport hands the API
client a tool call and ends the response, while the agent's ``session/prompt``
stays in flight and its MCP call stays unanswered.  The next request that
carries the matching result answers that MCP call and continues the same
prompt.  The agent's own built-in tools remain refused.
"""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import logging
import os
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from .._vendor.acp import (
    ACPClient,
    ClientCapabilities,
    FsCapabilities,
    ImplementationInfo,
    InitializeParams,
    PermissionOptionKind,
)
from .._vendor.jsonrpc import METHOD_NOT_FOUND, JSONRPCException
from ..transport._base import UpstreamConnectionError, UpstreamResponse, UpstreamStream
from ..transport.provider_info import ProviderInfo
from .acp_chunks import (
    CHUNK_ERROR,
    CHUNK_SESSION_START,
    CHUNK_STOP,
    CHUNK_TOOL_CALL,
    _blocks_of,
    _chunk_of,
    _error_of,
    _failure,
    _field_of,
    _prompt_text,
    _stop_reason_of,
)
from .acp_ledger import (
    EXTENDS,
    AcpSessionLedger,
    LedgerEntry,
    SessionKey,
    block_digests,
)
from .acp_mcp import AcpMcpServer, AcpToolBridge, MCP_SERVER_NAME, PendingCall
from .acp_profile import (
    ensure_profile,
    gemini_dir_of,
    harden_argv,
    is_agy,
    state_dir_for,
    tool_name_of,
)

logger = logging.getLogger("llm-rosetta-gateway")

#: Default agent argv.  A provider may override it with ``connection.acp_argv``
#: (a shim declares the command its vendor ships); ``agy acp`` is the
#: Antigravity one.
DEFAULT_AGENT_ARGV: tuple[str, ...] = (
    "agy",
    "acp",
    "--dangerously-skip-permissions=false",
)

#: The agent must never run one of its own tools (file read, write, shell,
#: web) on the gateway host for an API request.  Four layers enforce it (see
#: ``acp_profile``): a gateway-owned agent profile (``--gemini-dir``) with
#: ``toolPermission: strict``, a custom agent with no tools, a deny-all
#: ``PreToolUse`` hook, and this client, which refuses every
#: ``session/request_permission`` for a built-in tool and stops a turn at the
#: first built-in tool frame.
#:
#: Client tools are the exception: a tool the API request declared is served
#: by the gateway's own loopback MCP endpoint, so approving its permission
#: request runs the *client's* code through the bridge, not a host tool.  The
#: transport recognises that case by the bridge's pending call and lets it
#: through.  An operator-supplied ``agy`` command gets the same profile, so a
#: version switch does not reopen host tools; ``acp.harden: false`` turns it
#: off, and the command then runs as written with only the client-side
#: refusal and the tool-frame guard left.
DENY_AGENT_TOOLS = True

#: The agent mode that asks before *every* tool use, reads included.  Used only
#: for an agent the gateway does not give its own profile (an operator's
#: ``acp_argv``, a non-agy agent): agy's default mode reads files without
#: asking, and ``plan`` mode asks, and the client then refuses.  The hardened
#: profile does not use it: ``plan`` made the model write plan files first and
#: added no protection over ``toolPermission: strict``.
AGENT_TOOLS_OFF_MODE = "plan"

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
    #: Set when the agent answered "method not found" to ``set_mode``: the
    #: whole process has no mode surface, so a later session need not ask.
    no_modes: bool = False

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
    #: The tools this request declared, served to the agent over MCP.
    bridge: AcpToolBridge | None = None
    #: A digest of the tools this turn's session was opened with.  The agent's
    #: MCP server list is fixed for a session, so a request that declares a
    #: different set needs a fresh session even when it continues this turn.
    tools_digest: str = ""
    #: The suspended turn this request continues, when it carries a tool
    #: result for one.  Its runner owns the agent's in-flight prompt.
    suspension: _SuspendedTurn | None = None

    def release(self) -> None:
        """Give back the locks this turn took, once."""
        if self.released:
            return
        self.released = True
        for lock in reversed(self.held):
            if lock.locked():
                lock.release()
        self.held.clear()


@dataclass
class _SuspendedTurn:
    """One agent turn paused on a client tool call.

    The agent's ``session/prompt`` is still in flight: its MCP ``tools/call``
    is waiting, and the runner owns the update iterator that stays open until
    the next HTTP request supplies the result.
    """

    turn: _Turn
    bridge: AcpToolBridge
    runner: _TurnRunner
    pending: list[str] = field(default_factory=list)
    #: The session and process locks the paused turn holds.  A continuation
    #: adopts them instead of acquiring them again: the same turn is still in
    #: flight, so a second acquire would deadlock on its own lock.
    held: list[asyncio.Lock] = field(default_factory=list, repr=False)

    def release(self) -> None:
        """Give back the locks the suspended turn holds, once."""
        for lock in reversed(self.held):
            if lock.locked():
                lock.release()
        self.held.clear()


class _TurnRunner:
    """Consume one ``session/prompt`` in the background.

    A turn that stops on a client tool call must hand the API client a tool
    call and then let the agent wait.  The prompt iterator cannot be owned by
    the HTTP request that started it, because that request returns while the
    MCP call is still pending.  This runner reads the iterator and pushes
    chunks into a queue; whichever request is attached reads from that queue,
    and the queue survives between requests.

    Args:
        client: The agent client.
        session_id: The session the prompt belongs to.
        text: The prompt text.
        images: Extra prompt content blocks.
        bridge: The session's tool bridge.  A ``tools/call`` it records is
            pushed as a tool-call chunk.
        on_builtin: Called when the agent starts one of its own tools; returns
            the message the client receives.
        provider: The provider name, for messages the client reads.
        on_error: Turns an agent exception into the client-facing message.
    """

    def __init__(
        self,
        client: ACPClient,
        session_id: str,
        text: str,
        images: list[Any],
        bridge: AcpToolBridge | None,
        on_builtin: Callable[[str], Awaitable[str]],
        provider: str = "",
        on_error: Callable[[str], str] | None = None,
    ) -> None:
        self.queue: asyncio.Queue[dict[str, Any] | None] = asyncio.Queue()
        self.stop_reason = ""
        self.error = ""
        #: The agent process went away mid-turn (as opposed to refusing it).
        self.dead = False
        self._client = client
        self._session_id = session_id
        self._text = text
        self._images = images
        self._bridge = bridge
        self._on_builtin = on_builtin
        self._provider = provider
        self._on_error = on_error or (lambda message: message)
        #: Tool-call ids this turn has recognised as a client tool.  Agents
        #: that send the name once and then input deltas without it (Claude
        #: Code) would otherwise make the later frames look like a nameless
        #: built-in tool and stop the turn.
        self._client_calls: set[str] = set()
        self.task = asyncio.create_task(self._run(), name="acp-turn-runner")

    async def _run(self) -> None:
        updates = self._client.prompt(
            self._session_id, self._text, self._images or None
        ).__aiter__()
        bridge = self._bridge
        update_task: asyncio.Task[Any] | None = asyncio.ensure_future(
            updates.__anext__()
        )
        call_task: asyncio.Task[PendingCall] | None = (
            asyncio.ensure_future(bridge.next_call()) if bridge is not None else None
        )
        try:
            while True:
                waiters = [
                    task for task in (update_task, call_task) if task is not None
                ]
                if not waiters:
                    return
                done, _ = await asyncio.wait(
                    waiters, return_when=asyncio.FIRST_COMPLETED
                )
                if call_task is not None and call_task in done:
                    await self.queue.put(_tool_call_chunk(call_task.result()))
                    if bridge is not None:
                        call_task = asyncio.ensure_future(bridge.next_call())
                if update_task is not None and update_task in done:
                    update_task = await self._consume_update(updates, update_task)
                    if update_task is None:
                        return
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - one turn, one error chunk
            await self._fail(str(exc))
        finally:
            for task in (update_task, call_task):
                if task is not None and not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError, Exception):
                        await task
            aclose = getattr(updates, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:  # noqa: BLE001 - closing must not mask the turn
                    logger.debug(
                        "ACP prompt stream did not close cleanly", exc_info=True
                    )

    async def _consume_update(
        self,
        updates: AsyncIterator[dict[str, Any]],
        task: asyncio.Task[Any],
    ) -> asyncio.Task[Any] | None:
        """Handle one agent update; return the task for the next, or ``None``.

        ``None`` means this runner is done: the turn finished, failed, or used
        a built-in tool.
        """
        try:
            update = task.result()
        except StopAsyncIteration:
            await self._finish()
            return None
        except Exception as exc:  # noqa: BLE001 - reported to the client
            await self._fail(str(exc))
            return None
        builtin = self._builtin_of(update)
        if builtin is not None:
            self.error = await self._on_builtin(builtin)
            await self.queue.put({"type": CHUNK_ERROR, "message": self.error})
            await self.queue.put(None)
            return None
        chunk = _chunk_of(update)
        if chunk is not None:
            await self.queue.put(chunk)
        return asyncio.ensure_future(updates.__anext__())

    def _builtin_of(self, update: dict[str, Any]) -> str | None:
        """The name of the agent's own tool in *update*, or ``None``.

        A tool frame for a client tool the bridge is serving is not a
        built-in one; it is the call the bridge is already turning into a
        tool-call chunk.  The call id is remembered, because some agents send
        the name once and then send input deltas for the same call without it;
        a later frame must not look like a nameless built-in tool.
        """
        call = _tool_of(update)
        call_id = call.get("toolCallId") or call.get("tool_call_id")
        if isinstance(call_id, str) and call_id in self._client_calls:
            return None
        tool = tool_name_of(update)
        if tool is None:
            return None
        if self._bridge is not None and self._bridge.may_allow(call):
            if isinstance(call_id, str):
                self._client_calls.add(call_id)
            return None
        return tool

    async def _finish(self) -> None:
        result = self._client.prompt_result(self._session_id)
        if result is None:
            self.dead = True
            await self._fail_raw(
                f"ACP agent for provider {self._provider!r} ended the turn "
                f"without reporting a stop; the conversation was closed. The "
                f"agent process may have exited — check it on the gateway host."
            )
            return
        self.stop_reason = _reason_text(result)
        await self.queue.put(None)

    async def _fail(self, message: str) -> None:
        await self._fail_raw(self._on_error(message))

    async def _fail_raw(self, message: str) -> None:
        self.error = message
        await self.queue.put({"type": CHUNK_ERROR, "message": message})
        await self.queue.put(None)

    def cancel(self) -> None:
        """Stop reading the agent's turn."""
        self.task.cancel()


def _tool_of(update: dict[str, Any]) -> dict[str, Any]:
    """The ACP tool-call payload inside a ``session/update``."""
    body = update.get("update")
    return body if isinstance(body, dict) else update


def _tool_call_chunk(call: PendingCall) -> dict[str, Any]:
    """The transport chunk for one MCP tool call."""
    return {
        "type": CHUNK_TOOL_CALL,
        "tool_call_id": call.call_id,
        "tool_name": call.tool_name,
        "tool_input": call.arguments,
    }


def _capability(info: Any, name: str) -> Any:
    """Return one of the agent's advertised session capabilities."""
    caps = getattr(info, "agent_capabilities", None)
    session_caps = getattr(caps, "session_capabilities", None)
    return getattr(session_caps, name, None)


def _permission_tool(params: dict[str, Any]) -> dict[str, Any]:
    """The tool call a permission request names, in either ACP shape.

    Protocol v1 puts it at ``params.toolCall``; v2 wraps it as
    ``params.subject`` with ``type: "tool_call"``.  Claude Code sends the v2
    shape (measured live), so reading only the v1 key would refuse every
    client tool.
    """
    tool = params.get("toolCall")
    if isinstance(tool, dict) and tool:
        return tool
    subject = params.get("subject")
    if isinstance(subject, dict) and subject.get("type") == "tool_call":
        tool = subject.get("toolCall")
        if isinstance(tool, dict):
            return tool
    return {}


def _agent_request_handler(
    provider: str, bridge_for: Callable[[str], AcpToolBridge | None]
) -> Any:
    """Build the handler for requests the agent sends to this client.

    A permission request for a tool the client declared (a bridged MCP call)
    is approved once.  Every other permission request — the agent's own file,
    shell and web tools — is refused with ``reject_once`` (or
    ``reject_always``, or the protocol's ``cancelled`` outcome), so the agent
    can only use the tools the client offered.  Any other agent-to-client
    request (``fs/*``, ``terminal/*``) is refused with an error: none is
    advertised.  One WARNING line names the provider and the tool title;
    arguments and contents are never logged.
    """

    def handle(method: str, params: dict[str, Any]) -> dict[str, Any]:
        if method != "session/request_permission":
            raise RuntimeError(f"{method} is not available: agent tools are off")
        tool = _permission_tool(params)
        title = tool.get("title") or params.get("title")
        options = params.get("options") or []
        bridge = bridge_for(str(params.get("sessionId") or ""))
        if bridge is not None and bridge.may_allow(tool):
            logger.info(
                "ACP provider %r: client tool permission approved (tool: %s)",
                provider,
                str(title or "bridged tool")[:60],
            )
            return _permission_outcome(
                options,
                (PermissionOptionKind.ALLOW_ONCE, PermissionOptionKind.ALLOW_ALWAYS),
            )
        if not DENY_AGENT_TOOLS:
            # Relaxed mode for an operator's own agent argv: approve once what
            # the agent asks.  The default never takes this branch.
            return _permission_outcome(
                options,
                (PermissionOptionKind.ALLOW_ONCE, PermissionOptionKind.ALLOW_ALWAYS),
            )
        logger.warning(
            "ACP provider %r: agent tool request refused (tool: %s); "
            "agent tools are disabled",
            provider,
            # A shell tool's title is the command line: cap it.
            str(title or "unknown")[:60],
        )
        return _permission_outcome(
            options,
            (PermissionOptionKind.REJECT_ONCE, PermissionOptionKind.REJECT_ALWAYS),
        )

    return handle


def _permission_outcome(
    options: Any, wanted: tuple[PermissionOptionKind, ...]
) -> dict[str, Any]:
    """Pick the first offered option of a wanted kind, else ``cancelled``."""
    for kind in wanted:
        for option in options:
            if isinstance(option, dict) and option.get("kind") == kind.value:
                return {
                    "outcome": {
                        "outcome": "selected",
                        "optionId": option.get("optionId"),
                    }
                }
    return {"outcome": {"outcome": "cancelled"}}


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


#: One ACP session: the provider, the process slot that holds it, and the id
#: the agent returned.  The id alone is not unique across processes.
_SessionRef = tuple[str, int, str]


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
        # The loopback MCP endpoints that carry client tools to agents.
        self._mcp = AcpMcpServer()
        # Bridge and suspended runner by agent session id, so a permission
        # request from the agent can be answered without knowing the
        # conversation key.  The id alone is not unique: two processes of one
        # provider, or two providers, may both return ``sess-1``.  The key is
        # therefore ``(provider, process slot, session id)``.
        self._session_bridges: dict[_SessionRef, AcpToolBridge] = {}
        self._session_urls: dict[_SessionRef, str] = {}
        self._suspended: dict[SessionKey, _SuspendedTurn] = {}
        # Serialises the check-and-take of a suspended turn: two requests for
        # one conversation must not both decide that they continue it.
        self._suspend_guards: dict[SessionKey, asyncio.Lock] = {}

    @property
    def ledger(self) -> AcpSessionLedger:
        """The session ledger this transport keeps."""
        return self._ledger

    # -- agent lifecycle ----------------------------------------------------

    def _declared_argv(self, provider_info: ProviderInfo) -> list[str]:
        argv = getattr(provider_info, "acp_argv", None)
        return list(argv) if argv else list(DEFAULT_AGENT_ARGV)

    def _hardened(self, provider_info: ProviderInfo) -> bool:
        """Whether the gateway gives this provider's agent its own profile.

        The answer is resolved when the provider is built: the builder
        follows the command's name when the entry says nothing, and
        ``acp.harden: true`` forces it for a renamed binary or a wrapper that
        forwards ``agy``'s flags.  ``acp.harden: false`` runs the command as
        written.
        """
        return bool(getattr(provider_info, "acp_harden", False))

    def _argv(self, provider_info: ProviderInfo) -> list[str]:
        """The command that starts this provider's agent.

        For a hardened provider it carries ``--gemini-dir <profile>`` and the
        agent flag; the profile itself is written by :meth:`_profile`.
        """
        argv = self._declared_argv(provider_info)
        if not self._hardened(provider_info):
            return argv
        return harden_argv(argv, self._state_dir(provider_info))

    @staticmethod
    def _spawn_env(provider_info: ProviderInfo) -> dict[str, str] | None:
        """The agent's environment: this process's, plus the shim's switches.

        Passing only the overrides would drop ``PATH`` and the agent's login
        store, so the full environment is copied and the declared names win.
        """
        overrides = getattr(provider_info, "acp_env", None)
        if not overrides:
            return None
        return {**os.environ, **overrides}

    @staticmethod
    def _state_dir(provider_info: ProviderInfo) -> str:
        explicit = getattr(provider_info, "acp_state_dir", None)
        if explicit:
            return str(explicit)
        return state_dir_for(None, provider_info.config_name)

    def _profile(self, provider_info: ProviderInfo, argv: list[str]) -> None:
        """Write, or re-assert, the profile the agent will run in.

        Called before the agent starts and before every session: what the
        agent could change of its own profile is undone each time.
        """
        if not self._hardened(provider_info):
            return
        gemini_dir = gemini_dir_of(argv)
        if gemini_dir is None:  # pragma: no cover - harden_argv always sets it
            return
        try:
            drift = ensure_profile(gemini_dir)
        except OSError as exc:
            raise AcpTurnError(
                f"provider {provider_info.config_name!r}: the agent profile "
                f"{gemini_dir!r} cannot be written, so its tools cannot be "
                f"gated: {exc}",
                status_code=502,
            ) from exc
        if drift:
            logger.warning(
                "ACP provider %r: the agent's profile was changed since the "
                "gateway wrote it (%s); restored",
                provider_info.config_name,
                ", ".join(drift),
            )

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
        self._profile(provider_info, argv)
        client = ACPClient(argv, env=self._spawn_env(provider_info))
        client.set_request_handler(
            _agent_request_handler(
                provider_info.config_name,
                lambda session_id: self._bridge_for(
                    provider_info.config_name, slot, session_id
                ),
            )
        )
        try:
            await client.start()
            info = await client.initialize(
                InitializeParams(
                    client_info=ImplementationInfo(
                        name="llm-rosetta-gateway", version=""
                    ),
                    # No fs and no terminal: the agent must not be able to
                    # use the client as a tool.
                    client_capabilities=ClientCapabilities(
                        fs=FsCapabilities(read_text_file=False, write_text_file=False),
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
        for suspension in self._suspended.values():
            suspension.runner.cancel()
            suspension.bridge.close()
        self._suspended.clear()
        for bridge in self._session_bridges.values():
            bridge.close()
        self._session_bridges.clear()
        for url in self._session_urls.values():
            self._mcp.unregister(url)
        self._session_urls.clear()
        await self._mcp.close()

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
        for key, suspension in list(self._suspended.items()):
            if key.provider == provider_name:
                suspension.runner.cancel()
                suspension.bridge.close()
                self._suspended.pop(key, None)
        for ref, bridge in list(self._session_bridges.items()):
            if ref[0] != provider_name:
                continue
            url = self._session_urls.pop(ref, None)
            if url:
                self._mcp.unregister(url)
            bridge.close()
            self._session_bridges.pop(ref, None)
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
        key = self._key(provider_info)
        if _tool_defs(body) and key is None:
            # A tool call suspends the turn, and only a declared conversation
            # id can find that suspension when the result arrives.  A derived
            # id may name two conversations, so it must never key one.
            raise AcpTurnError(
                "this request declares tools, but an ACP lane needs a declared "
                "conversation id to carry a tool result back to the agent; "
                "send a session header (for example `x-session-id`) and retry.",
                status_code=400,
            )
        # A process that exited between turns still occupies its slot; taking
        # it out first is what makes this request start a fresh one instead of
        # writing into a dead pipe.
        self._prune_gone(provider_info.config_name)
        # Re-assert the agent's profile before every turn (new, resumed or
        # reused session alike): the agent can write files, and its own
        # profile is among them.
        self._profile(provider_info, self._argv(provider_info))
        digests = block_digests(_blocks_of(body))

        # A suspended turn still owns the session and process locks.  Taking
        # it is a check-and-take under a per-conversation guard: a second
        # request must not race the first to continue or abandon it.
        if key is not None:
            taken = await self._take_suspended(key, body, digests)
            if taken is not None:
                return taken

        entry = self._ledger.get(key) if key is not None else None
        agent = await self._agent(provider_info, self._slot_for(provider_info, entry))
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

    async def _take_suspended(
        self, key: SessionKey, body: dict[str, Any], digests: tuple[str, ...]
    ) -> tuple[_Turn, dict[str, Any]] | None:
        """Continue a suspended turn when this request carries its result.

        A request that does not carry one abandons the suspension: the agent
        is parked on an MCP call nobody will answer, and the next turn must
        not wait behind it.
        """
        guard = self._suspend_guards.setdefault(key, asyncio.Lock())
        async with guard:
            suspended = self._suspended.get(key)
            if suspended is not None and _suspension_is_dead(suspended):
                await self._abandon_suspension(key, suspended)
                suspended = None
            if suspended is None:
                return None
            if _carries_result_for(body, suspended.bridge.pending_ids()):
                if suspended.turn.tools_digest != _tools_digest(body):
                    # The agent's MCP server list is fixed for a session, so
                    # continuing with the old bridge would expose tools this
                    # request no longer declares (or hide newly declared
                    # ones).  The result is part of this request's history, so
                    # replay it into a fresh session with the new tools.
                    await self._abandon_suspension(key, suspended)
                    return None
                return self._continue_suspended(key, suspended, digests)
            await self._abandon_suspension(key, suspended)
            return None

    def _continue_suspended(
        self,
        key: SessionKey,
        suspended: _SuspendedTurn,
        digests: tuple[str, ...],
    ) -> tuple[_Turn, dict[str, Any]]:
        """Adopt a suspended turn for the request that carries its result.

        The paused turn holds the session and process locks through the
        suspension.  The continuation takes those locks over rather than
        acquiring them again — the same ``session/prompt`` is still in flight,
        so a second acquire would wait on itself.
        """
        held = suspended.held
        suspended.held = []
        suspended.turn.finished = True
        turn = _Turn(
            agent=suspended.turn.agent,
            session_id=suspended.turn.session_id,
            model=suspended.turn.model,
            entry=suspended.turn.entry,
            held=held,
            digests=digests,
            bridge=suspended.bridge,
            tools_digest=suspended.turn.tools_digest,
            suspension=suspended,
        )
        suspended.turn = turn
        self._suspended.pop(key, None)
        return turn, _start_of(turn)

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
        if entry.tools != _tools_digest(body):
            # The agent cannot gain or lose an MCP server inside one session;
            # the next turn would still see the old tool list.
            await self._forget(entry)
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
            try:
                resumed = await self._resume(provider_info, agent, entry, cwd, deadline)
            except BaseException:
                # The session is no longer known good; do not keep a ledger
                # entry pointing at a half-resumed agent.
                await self._forget(entry)
                raise
            if resumed is None:
                await self._forget(entry)
                return None
        bridge = self._bridge_for(agent.provider, agent.slot, entry.session_id)
        if entry.tools and bridge is None:
            # A session with client tools cannot continue without the endpoint
            # that serves them; replay it into a session that registers one.
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
            bridge=bridge,
            tools_digest=entry.tools,
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
                    agent.client.resume_session(
                        entry.session_id,
                        cwd,
                        self._mcp_servers_for(agent, entry.session_id),
                    ),
                    deadline,
                )
            except JSONRPCException as exc:
                logger.info(
                    "ACP provider %r could not resume session %s: %s",
                    provider_info.config_name,
                    entry.session_id,
                    exc,
                )
                return None
            await self._text_only(
                provider_info,
                agent,
                entry.session_id,
                deadline,
                allow_tools=bool(entry.tools),
                modes=entry.modes,
            )
            entry.live = True
            return entry.session_id
        if agent.can_load():
            try:
                await _bounded(
                    agent.client.load_session(
                        entry.session_id,
                        cwd,
                        self._mcp_servers_for(agent, entry.session_id),
                    ),
                    deadline,
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
            await self._text_only(
                provider_info,
                agent,
                entry.session_id,
                deadline,
                allow_tools=bool(entry.tools),
                modes=entry.modes,
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
        bridge: AcpToolBridge | None = None
        mcp_servers: list[dict[str, Any]] = []
        url = ""
        tool_defs = _tool_defs(body)
        if tool_defs:
            bridge = AcpToolBridge(
                tool_defs, session_label=key.conversation if key else ""
            )
            url = await self._mcp.register(bridge)
            mcp_servers = [
                {
                    "name": MCP_SERVER_NAME,
                    "type": "http",
                    "url": url,
                    "headers": [],
                }
            ]
        try:
            session = await _bounded(
                agent.client.new_session(cwd, mcp_servers), deadline
            )
        except BaseException as exc:
            if bridge is not None:
                self._mcp.unregister(url)
            if not isinstance(exc, JSONRPCException):
                raise
            message = str(exc)
            raise AcpTurnError(
                _sign_in_message(provider_info, message),
                status_code=503 if _looks_like_sign_in(message) else 502,
            ) from exc

        session_id = getattr(session, "session_id", "") or ""
        advertised_modes = _mode_ids(getattr(session, "modes", None))
        try:
            if bridge is not None:
                # The agent received this URL in ``session/new``; it may call
                # it at any point until the session is closed.
                ref = (agent.provider, agent.slot, session_id)
                self._session_bridges[ref] = bridge
                self._session_urls[ref] = url
            await self._text_only(
                provider_info,
                agent,
                session_id,
                deadline,
                allow_tools=bool(tool_defs),
                modes=getattr(session, "modes", None),
            )
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
                bridge=bridge,
                tools_digest=_tools_digest(body),
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
                    tools=_tools_digest(body),
                    modes=(
                        None
                        if advertised_modes is None
                        else tuple(sorted(advertised_modes))
                    ),
                )
                turn.entry = entry
                for displaced in self._ledger.put(entry):
                    await self._close_entry(displaced)
            return turn
        except BaseException:
            # The session opened but the request never became a turn.  Close
            # it and stop serving its bridge; otherwise both outlive the
            # request with nothing in the ledger to find them.  Shield the
            # close: the common way here is a cancelled request, and a plain
            # await would lose the close to the cancellation in flight.
            self._release_session_resources(agent.provider, agent.slot, session_id)
            await asyncio.shield(_close_quietly(agent.client, session_id))
            raise

    async def _text_only(
        self,
        provider_info: ProviderInfo,
        agent: _Agent,
        session_id: str,
        deadline: float,
        *,
        allow_tools: bool = False,
        modes: Any = None,
    ) -> None:
        """Put a session in the mode in which every tool use is asked for.

        ``plan`` is agy's mode, but it is not a name the transport may assume:
        an agent advertises its modes on ``session/new``, and an agent that
        advertises none may still answer ``set_mode`` with "Method not found"
        (the Claude adapter does).  The gate therefore asks the session, in
        this order:

        1. ``session/new`` named a mode list: use ``plan`` when it is there.
           When it is not, an agy must not run — its default mode reads files
           without asking — while another agent keeps its own mode and is
           covered by the client's permission refusal and tool-frame guard.
        2. The session named no mode list: try ``plan``.  ``-32601`` means the
           agent has no modes to set, so a non-agy agent runs as it started;
           an agy that refuses the mode still fails the turn.  Any other
           refusal fails the turn too.

        A wrapper-launched agy advertises its modes like any other agy, so
        the decision does not depend on the executable's file name.  The name
        check remains only to decide whether a missing ``plan`` is an error
        rather than another agent's normal surface.  An agent that names no
        modes and answers ``-32601`` cannot be told apart from a wrapped agy
        by its answer, so it runs ungated unless the name says agy; the
        client's permission refusal and tool-frame guard are what remain.

        For agy without the gateway's profile (``acp.harden: false``):
        agy's default mode reads files without asking (the permission flag
        does not cover reads); in ``plan`` mode a read asks too, and the
        client refuses.  Failing to set the mode fails the turn: a session
        whose tools cannot be shown to be gated must not run.  The hardened
        profile asks for everything through ``toolPermission: strict`` and
        needs no mode.

        A request that declares tools cannot use this mode: it would block
        the very tools the client offered.  Such a lane relies on the client
        refusing every built-in tool request, and the operator should set the
        agent's own permission mode to one that keeps host tools behind a
        prompt.
        """
        if not DENY_AGENT_TOOLS or allow_tools or self._hardened(provider_info):
            return
        argv = self._declared_argv(provider_info)
        available = _mode_ids(modes)
        if available is not None and AGENT_TOOLS_OFF_MODE not in available:
            if is_agy(argv):
                raise AcpTurnError(
                    f"provider {provider_info.config_name!r}: the ACP agent "
                    f"advertises no {AGENT_TOOLS_OFF_MODE!r} mode, so its tools "
                    "cannot be gated",
                    status_code=502,
                )
            return
        if available is None and agent.no_modes:
            # This process already answered ``-32601`` for an earlier session.
            # Modes are a property of the agent, not of one session.
            return
        try:
            await _bounded(
                agent.client.set_mode(session_id, AGENT_TOOLS_OFF_MODE), deadline
            )
        except JSONRPCException as exc:
            if (
                available is None
                and getattr(exc.error, "code", None) == METHOD_NOT_FOUND
                and not is_agy(argv)
            ):
                agent.no_modes = True
                logger.info(
                    "ACP provider %r: the agent has no session modes; leaving "
                    "the session as it started",
                    provider_info.config_name,
                )
                return
            raise AcpTurnError(
                f"provider {provider_info.config_name!r}: the ACP agent would not "
                f"enter its {AGENT_TOOLS_OFF_MODE!r} mode, so its tools cannot be "
                f"gated: {exc}",
                status_code=502,
            ) from exc

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
        key = self._key(provider_info)
        if collect is not None:
            collect.append(start)
        runner: _TurnRunner | None = None
        try:
            yield start

            if turn.suspension is not None:
                # The agent is still running the prompt that called a client
                # tool.  Hand it this request's result; there is no second
                # prompt to send.
                self._resolve_results(turn.bridge, body)
                runner = turn.suspension.runner
            else:
                blocks = _blocks_of(body)
                sending = blocks[turn.tail_start :]
                text = _prompt_text(sending)
                images: list[Any] = [b for b in sending if b.get("type") == "image"]
                runner = _TurnRunner(
                    client,
                    session_id,
                    text,
                    images,
                    turn.bridge,
                    on_builtin=lambda name: self._runner_tool_stop(
                        turn, provider_info, name
                    ),
                    provider=provider_info.config_name,
                    on_error=lambda message: _sign_in_message(provider_info, message),
                )
            try:
                async for chunk in self._forward_chunks(
                    provider_info, turn, runner, deadline, collect, key
                ):
                    yield chunk
                if turn.finished:
                    # ``_forward_chunks`` stopped the response at a client
                    # tool call; the turn is suspended, not abandoned.
                    return
                if runner.error:
                    # The runner already put the client-facing error chunk in
                    # the stream.  What the agent recorded of the turn is
                    # unknown, so the session is not reused; a process that
                    # went away is taken out of its pool.
                    turn.finished = True
                    await self._forget(turn.entry)
                    if runner.dead:
                        self._drop_agent(turn.agent)
                    return
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
                if runner is not None:
                    runner.cancel()
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
            turn.finished = True
            await self._settle(turn, provider_info)
            stop = {
                "type": CHUNK_STOP,
                "stop_reason": runner.stop_reason or _reason_text(result),
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

    async def _forward_chunks(
        self,
        provider_info: ProviderInfo,
        turn: _Turn,
        runner: _TurnRunner,
        deadline: float,
        collect: list[dict[str, Any]] | None,
        key: SessionKey | None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Relay the runner's chunks, ending the response at a tool call.

        A tool call is the last thing this response carries: the API client
        has to run it, so the agent's prompt is left suspended and the next
        request continues it.
        """
        async for chunk in _queued_chunks(runner, deadline):
            if chunk.get("type") != CHUNK_TOOL_CALL:
                if collect is not None:
                    collect.append(chunk)
                yield chunk
                continue
            bridge = turn.bridge
            if key is None or bridge is None:
                # No conversation to suspend, so the agent's prompt cannot be
                # continued.  Give the session back before telling the client
                # why: a one-off session has no ledger entry to close it later.
                await self._abandon(turn)
                yield _failure(
                    collect,
                    f"ACP agent for provider {provider_info.config_name!r} asked "
                    f"for a client tool, but this turn has no declared "
                    f"conversation to resume it on.",
                )
                return
            if chunk.get("tool_call_id") not in bridge.pending_ids():
                # The client already returned this call's result — a request
                # may carry the results of a parallel batch — so the agent has
                # continued past it; there is nothing to pause on.
                continue
            if collect is not None:
                collect.append(chunk)
            yield chunk
            self._suspend(key, turn, runner, bridge)
            stop = {"type": CHUNK_STOP, "stop_reason": "tool_calls"}
            if collect is not None:
                collect.append(stop)
            yield stop
            return

    def _suspend(
        self,
        key: SessionKey,
        turn: _Turn,
        runner: _TurnRunner,
        bridge: AcpToolBridge,
    ) -> None:
        """Pause *turn* on a client tool call without giving its locks back.

        The agent's ``session/prompt`` is still in flight.  The locks stay
        held by the suspension and are adopted by the request that carries
        the tool result (or released when the suspension is abandoned).
        """
        if turn.entry is not None:
            turn.entry.remember(turn.digests, max_blocks=self._ledger.max_blocks)
            turn.entry.live = True
            turn.entry.touch()
        held = turn.held
        turn.held = []
        turn.finished = True
        suspended = _SuspendedTurn(
            turn=turn,
            bridge=bridge,
            runner=runner,
            pending=bridge.pending_ids(),
            held=held,
        )
        self._suspended[key] = suspended

    async def _runner_tool_stop(
        self, turn: _Turn, provider_info: ProviderInfo, tool: str
    ) -> str:
        """Stop a turn the runner saw the agent use a built-in tool in."""
        return await self._stop_for_tool(turn, provider_info, tool)

    async def _stop_for_tool(
        self, turn: _Turn, provider_info: ProviderInfo, tool: str
    ) -> str:
        """End a turn in which the agent started one of its own tools.

        Only a tool the client declared may run.  A built-in one stops the
        turn here: cancelled, the session closed (what the tool did before
        the cancel is unknown, so the session is not reused).  The client is
        told, in a status it can act on, that the turn did not complete: an
        empty 200 would read as "the model said nothing".  The log line names
        the provider and the tool, never arguments or output.

        Returns:
            The message of the error the client receives.
        """
        turn.finished = True
        await self._forget(turn.entry)
        await _cancel_quietly(turn.agent.client, turn.session_id)
        logger.warning(
            "ACP provider %r: the agent started its built-in tool %r; "
            "agent tools are disabled, so the turn was stopped",
            provider_info.config_name,
            tool,
        )
        return (
            f"ACP agent for provider {provider_info.config_name!r} tried to "
            f"use its built-in tool {tool!r}; agent tools are disabled "
            f"on this lane, so the turn was stopped. Ask for a text answer "
            f"that needs no tool."
        )

    async def _abandon(self, turn: _Turn) -> None:
        """Give back a session whose turn did not finish."""
        turn.finished = True
        if turn.suspension is not None:
            turn.suspension.runner.cancel()
            key = turn.entry.key if turn.entry is not None else None
            if key is not None and self._suspended.get(key) is turn.suspension:
                self._suspended.pop(key, None)
            turn.suspension.held = []
        # Close the bridge first: an agent parked on a bridged call would not
        # read the session/close until that call is answered.
        self._release_session_resources(
            turn.agent.provider, turn.agent.slot, turn.session_id
        )
        await self._forget(turn.entry)
        await _close_quietly(turn.agent.client, turn.session_id)

    async def _abandon_suspension(
        self, key: SessionKey, suspended: _SuspendedTurn
    ) -> None:
        """Give back a turn whose tool result never came.

        The agent is parked on an MCP call nobody will answer, so its prompt
        is cancelled and the session closed: what it recorded of the partial
        turn is unknown.
        """
        if self._suspended.get(key) is suspended:
            self._suspended.pop(key, None)
        suspended.turn.finished = True
        suspended.runner.cancel()
        suspended.bridge.close()
        await self._forget(suspended.turn.entry)
        await _close_quietly(suspended.turn.agent.client, suspended.turn.session_id)
        suspended.release()

    async def _settle(self, turn: _Turn, provider_info: ProviderInfo) -> None:
        """Record what the agent has now seen, or close a one-off session."""
        entry = turn.entry
        if entry is None:
            # A request that declared no conversation gets one session for its
            # own, and nothing will ever come back for it.
            self._release_session_resources(
                turn.agent.provider, turn.agent.slot, turn.session_id
            )
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

    def _resolve_results(
        self, bridge: AcpToolBridge | None, body: dict[str, Any]
    ) -> None:
        """Hand this request's tool results to the bridge's pending calls."""
        if bridge is None:
            return
        for result in body.get("tool_results") or []:
            if not isinstance(result, dict):
                continue
            call_id = result.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                continue
            bridge.resolve(
                call_id,
                result.get("result", ""),
                is_error=bool(result.get("is_error")),
            )

    def _bridge_for(
        self, provider: str, slot: int, session_id: str
    ) -> AcpToolBridge | None:
        """The tool bridge of one agent process's session, if it has one."""
        return self._session_bridges.get((provider, slot, session_id))

    def _mcp_servers_for(self, agent: _Agent, session_id: str) -> list[dict[str, Any]]:
        """The MCP server entry that re-attaches a session's tool bridge."""
        url = self._session_urls.get((agent.provider, agent.slot, session_id))
        if not url:
            return []
        return [
            {
                "name": MCP_SERVER_NAME,
                "type": "http",
                "url": url,
                "headers": [],
            }
        ]

    def _release_session_resources(
        self, provider: str, slot: int, session_id: str
    ) -> None:
        """Close a session's tool bridge and stop serving its URL."""
        ref = (provider, slot, session_id)
        bridge = self._session_bridges.pop(ref, None)
        url = self._session_urls.pop(ref, None)
        if bridge is not None:
            bridge.close()
        if url:
            self._mcp.unregister(url)

    async def _forget(self, entry: LedgerEntry | None) -> None:
        """Close a session and forget it, so the next turn starts fresh."""
        if entry is None:
            return
        self._ledger.drop(entry.key)
        await self._close_entry(entry)

    async def _close_entry(self, entry: LedgerEntry) -> None:
        """Close one recorded session on the process that holds it."""
        entry.live = False
        self._release_session_resources(
            entry.key.provider, entry.agent_slot, entry.session_id
        )
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


async def _queued_chunks(
    runner: _TurnRunner, deadline: float
) -> AsyncIterator[dict[str, Any]]:
    """Yield the turn runner's chunks, giving up when the deadline passes.

    ACP has no turn timeout, and an agent that stops answering keeps the
    session — and the process lock that serialises the lane — forever.  The
    wait is per chunk rather than around the whole loop so the deadline is
    also a *stall* deadline: a turn that keeps producing may finish.  While a
    turn is suspended no request is reading; its runner keeps filling the
    queue, so a continuation picks the chunks up where the first request
    stopped.

    Raises:
        _TurnDeadline: No chunk arrived before *deadline*.
    """
    loop = asyncio.get_running_loop()
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            raise _TurnDeadline
        getter = asyncio.create_task(runner.queue.get())
        try:
            done, _ = await asyncio.wait({getter}, timeout=remaining)
        except BaseException:
            # The turn was cancelled or closed while waiting.  Cancelling the
            # getter here, rather than leaving it to the loop's shutdown,
            # keeps its coroutine from outliving the event loop.
            getter.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await getter
            raise
        if not done:
            getter.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await getter
            raise _TurnDeadline
        chunk = getter.result()
        if chunk is None:
            return
        yield chunk


class _TurnDeadline(Exception):
    """The agent stopped answering inside the turn's deadline."""


def _suspension_is_dead(suspended: _SuspendedTurn) -> bool:
    """Whether the process or bridge behind a suspended turn has gone."""
    return suspended.bridge.closed or _agent_gone(suspended.turn.agent)


def _tool_defs(body: dict[str, Any]) -> list[dict[str, Any]]:
    """The tool definitions a turn body carries, in MCP shape."""
    tools = body.get("tools")
    if not isinstance(tools, list):
        return []
    return [
        tool
        for tool in tools
        if isinstance(tool, dict) and isinstance(tool.get("name"), str)
    ]


def _tools_digest(body: dict[str, Any]) -> str:
    """A stable digest of the tools a session was opened with.

    An agent cannot gain or lose an MCP server inside one session, so a
    request whose tools differ needs a fresh session rather than a reused one.
    The digest is over the canonical definitions, order aside: the same tools
    in another order are the same tool set.
    """
    tools = _tool_defs(body)
    if not tools:
        return ""
    # Tool order is not part of a tool set: MCP serves a list, and what the
    # agent can call is the set of definitions.  Sorting the canonical form
    # keeps a client that reorders the same tools from closing a session that
    # would have shown the agent the same tools.
    canonical = sorted(
        json.dumps(tool, sort_keys=True, ensure_ascii=False, default=str)
        for tool in tools
    )
    blob = "\n".join(canonical)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


def _carries_result_for(body: dict[str, Any], pending_ids: list[str]) -> bool:
    """Whether this request returns a result for one of *pending_ids*."""
    if not pending_ids:
        return False
    wanted = set(pending_ids)
    for result in body.get("tool_results") or []:
        if not isinstance(result, dict):
            continue
        call_id = result.get("tool_call_id")
        if isinstance(call_id, str) and call_id in wanted:
            return True
    return False


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
    let a model write into the deployment.  An ``agy`` the gateway gives its own
    profile runs in an empty ``workspace`` directory beside that profile
    instead (mode 0700): a shared temp directory is somewhere another local
    user could plant ``.agents/`` customizations the agent would load.
    """
    import tempfile
    from pathlib import Path

    if getattr(provider_info, "acp_harden", False):
        base = Path(
            getattr(provider_info, "acp_state_dir", None)
            or state_dir_for(None, provider_info.config_name)
        )
        owned = base / "workspace"
        owned.mkdir(parents=True, exist_ok=True, mode=0o700)
        return str(owned)

    safe = "".join(
        c if c.isalnum() or c in "-_." else "_" for c in provider_info.config_name
    )
    path = Path(tempfile.gettempdir()) / "llm-rosetta-acp" / safe
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


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


def _mode_ids(modes: Any) -> set[str] | None:
    """The mode ids a session advertised, or ``None`` when it said nothing."""
    if isinstance(modes, (tuple, list, set, frozenset)):
        # A copy kept on the ledger entry after ``session/new``.
        return {str(mode) for mode in modes}
    available = getattr(modes, "available_modes", None)
    if available is None:
        return None
    ids = {str(getattr(mode, "id", "") or "") for mode in available}
    ids.discard("")
    return ids


def _current_model(config_options: Any) -> str:
    """Return the session's current model, when it advertises one."""
    for option in config_options or []:
        if (
            getattr(option, "id", "") == "model"
            or getattr(option, "category", None) == "model"
        ):
            return str(getattr(option, "current_value", "") or "")
    return ""


def _reason_text(result: Any) -> str:
    """The wire spelling of a prompt result's stop reason."""
    reason = getattr(result, "stop_reason", "")
    return str(getattr(reason, "value", reason) or "end_turn")


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
