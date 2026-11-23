"""An ACP client that adds a shim-declared ``_meta`` and reads turn usage.

The vendored client sends only the fields the ACP base schema defines.  Some
agents gate vendor behavior on the request's ``_meta`` object instead: the
Claude adapter, for example, keeps Claude Code's built-in tools unless
``session/new`` carries ``{"disableBuiltInTools": true}``.  A shim declares
that value; this subclass puts it on ``session/new``, ``session/load`` and
``session/resume``.

The same response object is the only place the adapter reports a per-turn
input/output split; its ``usage_update`` notifications carry just the total
context figure.  The vendored client's ``PromptResult`` keeps the stop reason
and drops ``usage``, so this subclass records the raw response by session and
exposes it as :meth:`prompt_usage`.

The session methods are overrides of the vendored client, not a copy of it:
each one calls ``ACPClient._call`` and reuses the vendored parsers, so a
protocol change upstream reaches this file through the same parsing code.
The metadata must accompany a resume as well as a new session because the
adapter folds it into its session fingerprint; resuming without it asks for a
session the agent no longer recognizes.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any, cast

from .._vendor.acp import (
    ACPClient,
    NewSessionResult,
    _parse_available_commands,
    _parse_config_options,
    _parse_session_modes,
    from_raw,
)


class _PromptTrackingTransport:
    """A JSON-RPC transport that records each prompt's session id.

    The prompt response itself carries no session id, and the vendored prompt
    iterator sends the request directly.  Delegating the transport is enough
    to learn the request id before the response arrives, without copying the
    iterator or its parsers.

    Args:
        inner: The transport the client would otherwise use.
        owner: The client whose prompt map receives the request id.
    """

    def __init__(self, inner: Any, owner: AcpMetaClient) -> None:
        self._inner = inner
        self._owner = owner

    async def send_request(
        self,
        method: str,
        params: Any = None,
        req_id: int | str | None = None,
    ) -> dict[str, Any]:
        """Send a request, remembering which session a prompt belongs to."""
        sent = await self._inner.send_request(method, params, req_id)
        session_id = params.get("sessionId") if isinstance(params, dict) else None
        if method == "session/prompt" and isinstance(session_id, str):
            self._owner._prompt_requests[sent.get("id")] = session_id
        return cast(dict[str, Any], sent)

    def __getattr__(self, name: str) -> Any:
        """Forward everything else to the real transport."""
        return getattr(self._inner, name)


class AcpMetaClient(ACPClient):
    """An ``ACPClient`` that repeats one ``_meta`` on every session request.

    Args:
        command: Command and arguments to spawn the agent process.
        env: Environment variables for the subprocess.
        session_meta: The ``_meta`` object to send.  An empty mapping keeps
            the vendored behavior (no ``_meta`` key at all).
    """

    def __init__(
        self,
        command: list[str],
        env: dict[str, str] | None = None,
        *,
        session_meta: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(command, env=env)
        self.session_meta = dict(session_meta or {})
        self._prompt_requests: dict[Any, str] = {}
        self._prompt_usage: dict[str, dict[str, Any]] = {}

    async def start(self) -> None:
        """Start the agent, then route prompt ids through the tracker."""
        await super().start()
        if self._transport is not None:
            self._transport = cast(Any, _PromptTrackingTransport(self._transport, self))

    def _resolve_pending_future(self, msg: dict[str, Any]) -> None:
        """File a prompt response's usage under the session that asked."""
        session_id = self._prompt_requests.pop(msg.get("id"), None)
        result = msg.get("result")
        usage = result.get("usage") if isinstance(result, dict) else None
        if session_id is not None and isinstance(usage, dict):
            self._prompt_usage[session_id] = usage
        super()._resolve_pending_future(msg)

    def _stop_reason_from_update(self, update: dict[str, Any]) -> str | None:
        """Record the turn outcome an ACP v2 ``idle`` carries.

        A v2 prompt response only acknowledges the message.  The stop reason,
        the usage and the quota update arrive together on the idle state
        update, so this is the one point that sees a v2 turn's bill.
        """
        payload = update.get("update")
        if (
            isinstance(payload, dict)
            and payload.get("sessionUpdate") == "state_update"
            and payload.get("state") == "idle"
        ):
            usage = payload.get("usage")
            session_id = update.get("sessionId")
            if isinstance(session_id, str) and isinstance(usage, dict):
                self._prompt_usage[session_id] = usage
        return super()._stop_reason_from_update(update)

    async def prompt(
        self,
        session_id: str,
        text: str,
        extra_content: list[Any] | None = None,
    ) -> AsyncIterator[dict[str, Any]]:
        """Send a prompt, forgetting the previous turn's usage."""
        self._prompt_usage.pop(session_id, None)
        async for update in super().prompt(session_id, text, extra_content):
            yield update

    def prompt_usage(self, session_id: str) -> dict[str, Any] | None:
        """The agent's own token tally for the last prompt, if it gave one.

        Returns:
            The raw ``usage`` object from ``session/prompt``: Claude reports
            ``inputTokens``, ``outputTokens``, ``cachedReadTokens``,
            ``cachedWriteTokens`` and ``totalTokens``.  ``None`` means the
            agent did not report one; the context figure from its
            ``usage_update`` is the only availability then.
        """
        return self._prompt_usage.get(session_id)

    def _with_meta(self, params: dict[str, Any]) -> dict[str, Any]:
        """Return *params* with the declared ``_meta``, when there is one."""
        if self.session_meta:
            params["_meta"] = dict(self.session_meta)
        return params

    async def new_session(
        self,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | None = None,
        additional_directories: list[str] | None = None,
    ) -> NewSessionResult:
        """Create a session, carrying the shim's ``_meta``."""
        params: dict[str, Any] = {"cwd": cwd, "mcpServers": mcp_servers or []}
        if additional_directories:
            params["additionalDirectories"] = additional_directories
        raw = await self._call("session/new", self._with_meta(params))
        result = from_raw(raw or {})
        return NewSessionResult(
            session_id=result.get("session_id", ""),
            modes=_parse_session_modes(result.get("modes")),
            config_options=_parse_config_options(result.get("config_options")),
            available_commands=_parse_available_commands(
                result.get("available_commands")
            ),
        )

    async def load_session(
        self,
        session_id: str,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | None = None,
        additional_directories: list[str] | None = None,
    ) -> None:
        """Load a session, carrying the shim's ``_meta``."""
        if self._negotiated_version is not None and self._negotiated_version >= 2:
            await self.resume_session(
                session_id,
                cwd,
                mcp_servers,
                additional_directories,
                replay_from={"type": "start"},
            )
            return
        params: dict[str, Any] = {
            "sessionId": session_id,
            "cwd": cwd,
            "mcpServers": mcp_servers or [],
        }
        if additional_directories:
            params["additionalDirectories"] = additional_directories
        await self._call("session/load", self._with_meta(params))

    async def resume_session(
        self,
        session_id: str,
        cwd: str,
        mcp_servers: list[dict[str, Any]] | None = None,
        additional_directories: list[str] | None = None,
        replay_from: dict[str, Any] | None = None,
    ) -> None:
        """Resume a session, carrying the shim's ``_meta``."""
        params: dict[str, Any] = {
            "sessionId": session_id,
            "cwd": cwd,
            "mcpServers": mcp_servers or [],
        }
        if additional_directories:
            params["additionalDirectories"] = additional_directories
        if replay_from is not None:
            params["replayFrom"] = replay_from
        await self._call("session/resume", self._with_meta(params))
