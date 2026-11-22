"""A loopback MCP server that carries a request's tools to an ACP agent.

An ACP agent cannot receive a tool through the ACP prompt: the prompt is
content blocks, and tools are an MCP concern.  What ``session/new`` does carry
is a list of MCP servers, so the gateway serves the request's tools as a small
HTTP MCP endpoint and gives the agent its URL.  The agent lists the tools and
calls one through MCP; this module turns that call into a value the transport
can return to the API client.

The call is deliberately **not** answered here.  The gateway's contract is the
usual tool-calling one: the model returns a tool call, the client runs it, and
the next request carries the result.  So :meth:`AcpToolBridge.begin_call`
records the call and leaves its MCP response pending until the next HTTP
request supplies the result through :meth:`AcpToolBridge.resolve`.  A call
whose result never arrives is resolved as an error when the session is closed
or its TTL expires.

Only loopback is bound, and each session gets an unguessable path token: the
endpoint exists to serve one agent process on the same host, not the network.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("llm-rosetta-gateway")

#: The name the agent sees for the gateway's loopback MCP endpoint.  An agent
#: that exposes MCP through a meta-tool (agy's ``call_mcp_tool``) addresses a
#: call by this name, so the bridge and the agent profile must agree on it.
MCP_SERVER_NAME = "llm_rosetta_tools"

#: Agent meta-tools that address this bridge rather than run a host tool.
#: ``call_mcp_tool`` carries ``ServerName`` and ``ToolName``; the resource and
#: prompt readers carry only ``ServerName``.
_MCP_META_TOOLS = frozenset(
    {
        "call_mcp_tool",
        "list_resources",
        "read_resource",
        "list_prompts",
        "get_prompt",
    }
)

#: The ways agents spell one of this server's tools directly.  Keep these
#: exact: a substring test would let a host tool whose name happens to contain
#: a declared tool's name (``search_echo`` for a declared ``echo``) through the
#: permission gate.
_BRIDGED_NAME_FORMS = (
    "mcp_{server}_{tool}",
    "mcp_{server}__{tool}",
    "mcp__{server}_{tool}",
    "mcp__{server}__{tool}",
    "{server}_{tool}",
    "{server}__{tool}",
    "{server}:{tool}",
    "{server}: {tool}",
    "{server}/{tool}",
)

#: How long one MCP ``tools/call`` POST may stay open waiting for the client's
#: tool result.  The session ledger's own TTL usually wins; this bound is what
#: keeps a wedged HTTP connection from outliving a forgotten session.
DEFAULT_CALL_TIMEOUT = 3600.0

_MCP_PROTOCOL = "2025-11-25"
_MCP_DISCOVERY_PROTOCOL = "2026-07-28"


def _text_of(value: Any) -> str:
    """Render a tool result as MCP text content."""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value)


@dataclass
class PendingCall:
    """One bridged tool call waiting for the API client's result.

    Attributes:
        call_id: The id the API client sees and returns with its result.
        tool_name: The declared tool's name.
        arguments: The JSON object the agent passed.
        rpc_id: The MCP JSON-RPC id, when the call arrived over that wire.
        created: Monotonic time the call was recorded.
        future: Resolved with the MCP result once the client answers.
    """

    call_id: str
    tool_name: str
    arguments: dict[str, Any]
    future: asyncio.Future[dict[str, Any]] = field(repr=False)
    rpc_id: Any = None
    created: float = field(default_factory=time.monotonic)


class AcpToolBridge:
    """The tools of one agent session, and its outstanding calls.

    Args:
        tools: Tool definitions in IR/OpenAI shape: ``name``, ``description``
            and ``parameters``.
        session_label: A short label for log lines.  Never contains a prompt.
    """

    def __init__(
        self,
        tools: list[dict[str, Any]],
        *,
        session_label: str = "",
        call_timeout: float = DEFAULT_CALL_TIMEOUT,
    ) -> None:
        self._tools = [self._mcp_tool(tool) for tool in tools if tool.get("name")]
        self._by_name = {tool["name"]: tool for tool in self._tools}
        self._session_label = session_label
        self._call_timeout = call_timeout
        self._pending: dict[str, PendingCall] = {}
        self._calls: asyncio.Queue[PendingCall | object] = asyncio.Queue()
        self._closed = False

    @staticmethod
    def _mcp_tool(tool: dict[str, Any]) -> dict[str, Any]:
        parameters = tool.get("parameters")
        if not isinstance(parameters, dict) or not parameters:
            parameters = {"type": "object", "properties": {}}
        return {
            "name": str(tool["name"]),
            "description": str(tool.get("description") or ""),
            "inputSchema": parameters,
        }

    @property
    def empty(self) -> bool:
        """Whether the request declared no tool this bridge can expose."""
        return not self._tools

    @property
    def closed(self) -> bool:
        """Whether the session that owns the bridge has gone."""
        return self._closed

    def tools(self) -> list[dict[str, Any]]:
        """The tools as MCP ``tools/list`` items."""
        return list(self._tools)

    def names(self) -> set[str]:
        """The declared tool names."""
        return set(self._by_name)

    def pending_ids(self) -> list[str]:
        """The call ids still waiting for a result, in call order."""
        return list(self._pending)

    def pending(self) -> list[PendingCall]:
        """The calls still waiting for a result, in call order."""
        return list(self._pending.values())

    def has_pending(self) -> bool:
        """Whether any call is waiting for the client's result."""
        return bool(self._pending)

    def may_allow(self, tool_call: dict[str, Any]) -> bool:
        """Whether a permission request belongs to a bridged tool.

        A client tool is the one thing this lane may let the agent run: the
        client asked for it by declaring the tool.  The agent's own tools stay
        refused.  Two shapes reach this method.  Some agents expose each MCP
        tool as a direct tool, so the call names the tool itself.  ``agy``
        instead uses the ``call_mcp_tool`` meta-tool and puts the server and
        tool in its arguments; its read-only resource and prompt readers carry
        only the server.  A meta-tool that addresses another server, or a tool
        this request did not declare, is not bridged.
        """
        call_id = tool_call.get("toolCallId") or tool_call.get("tool_call_id")
        if isinstance(call_id, str) and any(
            call.call_id == call_id or call.rpc_id == call_id
            for call in self._pending.values()
        ):
            return True
        name = tool_call.get("name")
        args = tool_call.get("args")
        if not isinstance(args, dict):
            args = tool_call.get("rawInput") or tool_call.get("raw_input")
        if isinstance(name, str) and name in _MCP_META_TOOLS and isinstance(args, dict):
            server = (
                args.get("ServerName")
                or args.get("serverName")
                or args.get("server_name")
            )
            if isinstance(server, str) and server == MCP_SERVER_NAME:
                if name != "call_mcp_tool":
                    return True
                tool = (
                    args.get("ToolName")
                    or args.get("toolName")
                    or args.get("tool_name")
                )
                return isinstance(tool, str) and tool in self._by_name
        for field_name in ("name", "title"):
            value = tool_call.get(field_name)
            if not isinstance(value, str):
                continue
            for declared in self._by_name:
                if value == declared:
                    return True
                names = {
                    form.format(server=MCP_SERVER_NAME, tool=declared)
                    for form in _BRIDGED_NAME_FORMS
                }
                if value in names:
                    return True
        return False

    async def next_call(self) -> PendingCall:
        """Wait for the next bridged tool call.

        Raises:
            asyncio.CancelledError: The reader was cancelled.
            RuntimeError: The bridge is closed and no call can arrive.
        """
        call = await self._calls.get()
        if not isinstance(call, PendingCall):
            raise RuntimeError("the ACP tool bridge is closed")
        return call

    async def begin_call(
        self, tool_name: str, arguments: Any, *, rpc_id: Any = None
    ) -> dict[str, Any]:
        """Record a tool call and wait for the API client's result.

        This coroutine is the body of one MCP ``tools/call`` request.  It does
        not return until :meth:`resolve` is called for the same ``call_id``.

        Args:
            tool_name: The MCP tool name the agent asked for.
            arguments: The call's arguments, as the agent sent them.
            rpc_id: The MCP JSON-RPC id, used to cancel this call if the agent
                withdraws it.

        Returns:
            An MCP ``CallToolResult``: text content plus ``isError``.
        """
        if self._closed:
            return _error_result("the ACP session closed before the tool could run")
        if tool_name not in self._by_name:
            return _error_result(f"unknown tool {tool_name!r} on this lane")
        args = arguments if isinstance(arguments, dict) else {}
        call_id = f"call_{secrets.token_hex(12)}"
        loop = asyncio.get_running_loop()
        future: asyncio.Future[dict[str, Any]] = loop.create_future()
        call = PendingCall(
            call_id=call_id,
            tool_name=tool_name,
            arguments=args,
            rpc_id=rpc_id,
            future=future,
        )
        self._pending[call_id] = call
        self._calls.put_nowait(call)
        try:
            return await asyncio.wait_for(future, self._call_timeout)
        except asyncio.TimeoutError:
            logger.info(
                "ACP tool call %s (%s) for %s timed out waiting for the client",
                call_id,
                tool_name,
                self._session_label or "a session",
            )
            return _error_result("the client did not return a tool result in time")
        finally:
            self._pending.pop(call_id, None)

    def resolve(self, call_id: str, result: Any, *, is_error: bool = False) -> bool:
        """Answer a pending call with the API client's tool result.

        Args:
            call_id: The id returned to the client with the tool call.
            result: The tool result, as IR carries it (a string, object or
                list; MCP text content is made from it).
            is_error: Whether the client reported the call as failed.

        Returns:
            True when a call by that id was waiting, False otherwise.
        """
        call = self._pending.get(call_id)
        if call is None or call.future.done():
            return False
        call.future.set_result(_content_result(result, is_error=is_error))
        return True

    def fail(self, call_id: str, message: str) -> bool:
        """Answer a pending call with an error."""
        call = self._pending.get(call_id)
        if call is None or call.future.done():
            return False
        call.future.set_result(_error_result(message))
        return True

    def cancel_rpc(self, rpc_id: Any, reason: str = "cancelled") -> bool:
        """Answer the call whose MCP JSON-RPC id is *rpc_id*."""
        for call in self._pending.values():
            if call.rpc_id == rpc_id and not call.future.done():
                call.future.set_result(_error_result(reason))
                return True
        return False

    def close(self, message: str = "the ACP session closed") -> None:
        """Fail every pending call and stop the call queue."""
        if self._closed:
            return
        self._closed = True
        for call in list(self._pending.values()):
            if not call.future.done():
                call.future.set_result(_error_result(message))
        self._calls.put_nowait(_CLOSED)


#: Sentinel placed on the call queue when the bridge closes.
_CLOSED = object()


def _content_result(result: Any, *, is_error: bool = False) -> dict[str, Any]:
    """Build an MCP ``CallToolResult`` from an IR tool result."""
    return {
        "content": [{"type": "text", "text": _text_of(result)}],
        "isError": bool(is_error),
    }


def _error_result(message: str) -> dict[str, Any]:
    """Build a failed MCP ``CallToolResult``."""
    return {"content": [{"type": "text", "text": message}], "isError": True}


@dataclass
class _Registration:
    """The bridge behind one path token."""

    token: str
    bridge: AcpToolBridge


class AcpMcpServer:
    """The loopback HTTP endpoint that serves every session's bridge.

    One server serves every provider and session; the path token selects the
    bridge.  The server starts on first use and binds ``127.0.0.1`` on an
    ephemeral port, so a gateway with no tool-enabled request opens no socket.
    """

    def __init__(self) -> None:
        self._server: asyncio.AbstractServer | None = None
        self._port = 0
        self._registrations: dict[str, _Registration] = {}
        self._lock = asyncio.Lock()

    @property
    def port(self) -> int:
        """The bound port, or 0 before the server has started."""
        return self._port

    async def register(self, bridge: AcpToolBridge, *, label: str = "") -> str:
        """Serve *bridge* and return the URL to put in ``session/new``.

        The token is unguessable and independent of the provider name, so a
        URL never becomes a handle to another session.
        """
        await self._ensure_started()
        token = secrets.token_urlsafe(24)
        self._registrations[token] = _Registration(token=token, bridge=bridge)
        return f"http://127.0.0.1:{self._port}/mcp/{token}"

    def unregister(self, url: str) -> None:
        """Stop serving the bridge behind *url*."""
        token = url.rsplit("/", 1)[-1]
        registration = self._registrations.pop(token, None)
        if registration is not None:
            registration.bridge.close()
            if not self._registrations:
                self._stop_soon()

    async def close(self) -> None:
        """Close every connection and the listener."""
        for registration in list(self._registrations.values()):
            registration.bridge.close()
        self._registrations.clear()
        if self._server is not None:
            self._server.close()
            with contextlib.suppress(Exception):
                await self._server.wait_closed()
            self._server = None
            self._port = 0

    def _stop_soon(self) -> None:
        """Stop the listener once no registration remains.

        The close itself is asynchronous; scheduling it here keeps
        :meth:`unregister` synchronous for the transport's cleanup path.  The
        listener and its port are cleared at once: a later :meth:`register`
        must start a fresh server rather than hand out the URL of a closed
        one.
        """
        server = self._server
        if server is None:
            return
        self._server = None
        self._port = 0
        server.close()

    async def _ensure_started(self) -> None:
        async with self._lock:
            if self._server is not None:
                return
            self._server = await asyncio.start_server(
                self._handle_connection, "127.0.0.1", 0
            )
            sockets = self._server.sockets or []
            self._port = int(sockets[0].getsockname()[1]) if sockets else 0

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Serve one keep-alive HTTP connection."""
        try:
            while True:
                request = await _read_request(reader)
                if request is None:
                    return
                method, path, _headers, body = request
                await self._dispatch(method, path, body, writer)
        except (asyncio.IncompleteReadError, ConnectionError):
            pass
        except Exception:  # noqa: BLE001 - one connection must not kill the server
            logger.debug("ACP MCP connection failed", exc_info=True)
        finally:
            with contextlib.suppress(Exception):
                writer.close()
                await writer.wait_closed()

    async def _dispatch(
        self,
        method: str,
        path: str,
        body: bytes,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Answer one HTTP request."""
        token = path.rsplit("/", 1)[-1]
        registration = self._registrations.get(token)
        if registration is None:
            await _send(writer, 404, b'{"error":"unknown session"}')
            return
        if method == "GET":
            # MCP's streamable-HTTP client probes with a GET for a standalone
            # SSE stream.  This server has no server-initiated messages, so it
            # answers 405 and the client stays on POST.  A 204 with an
            # event-stream type is read as a stream that closed at once and
            # retried.
            await _send(writer, 405, b'{"error":"only POST is supported"}')
            return
        if method != "POST":
            await _send(writer, 405, b'{"error":"method not allowed"}')
            return
        try:
            payload = json.loads(body or b"null")
        except ValueError:
            await _send(writer, 400, b'{"error":"invalid JSON"}')
            return
        if isinstance(payload, list):
            answers = [
                await self._answer(registration.bridge, item) for item in payload
            ]
            answers = [answer for answer in answers if answer is not None]
            if not answers:
                await _send(writer, 202, b"")
            else:
                await _send(writer, 200, json.dumps(answers).encode())
            return
        answer = await self._answer(registration.bridge, payload)
        if answer is None:
            await _send(writer, 202, b"")
        else:
            await _send(writer, 200, json.dumps(answer).encode())

    async def _answer(
        self, bridge: AcpToolBridge, message: Any
    ) -> dict[str, Any] | None:
        """Answer one JSON-RPC message, or ``None`` for a notification."""
        if not isinstance(message, dict):
            return _rpc_error(None, -32600, "invalid request")
        rpc_id = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        if not isinstance(method, str):
            return _rpc_error(rpc_id, -32600, "missing method")
        if rpc_id is None:
            if method == "notifications/cancelled":
                request_id = params.get("requestId")
                bridge.cancel_rpc(request_id, "the agent cancelled the tool call")
            return None
        if method == "server/discover":
            return _rpc_result(
                rpc_id,
                {
                    "protocolVersions": [_MCP_DISCOVERY_PROTOCOL, _MCP_PROTOCOL],
                    "serverInfo": _server_info(),
                    "capabilities": {"tools": {"listChanged": False}},
                },
            )
        if method == "initialize":
            requested = params.get("protocolVersion")
            version = requested if isinstance(requested, str) else _MCP_PROTOCOL
            return _rpc_result(
                rpc_id,
                {
                    "protocolVersion": version,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": _server_info(),
                },
            )
        if method == "ping":
            return _rpc_result(rpc_id, {})
        if method in ("tools/list", "tools/list_changed"):
            return _rpc_result(rpc_id, {"tools": bridge.tools()})
        if method == "tools/call":
            name = params.get("name")
            if not isinstance(name, str):
                return _rpc_error(rpc_id, -32602, "tools/call needs a name")
            return _rpc_result(
                rpc_id,
                await bridge.begin_call(name, params.get("arguments"), rpc_id=rpc_id),
            )
        if method in ("resources/list", "resources/templates/list", "prompts/list"):
            key = "resourceTemplates" if "templates" in method else method.split("/")[0]
            return _rpc_result(rpc_id, {key: []})
        if method == "logging/setLevel":
            return _rpc_result(rpc_id, {})
        return _rpc_error(rpc_id, -32601, f"method {method!r} is not supported")


def _server_info() -> dict[str, Any]:
    """The MCP server's identity."""
    return {"name": "llm-rosetta-gateway", "version": ""}


def _rpc_result(rpc_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rpc_id, "result": result}


def _rpc_error(rpc_id: Any, code: int, message: str) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": rpc_id, "error": {"code": code, "message": message}}


async def _read_request(
    reader: asyncio.StreamReader,
) -> tuple[str, str, dict[str, str], bytes] | None:
    """Read one HTTP/1.1 request from *reader*.

    Returns:
        ``(method, path, headers, body)``, or ``None`` at end of stream.
    """
    try:
        line = await reader.readline()
    except (asyncio.IncompleteReadError, ConnectionError):
        return None
    if not line:
        return None
    while line in (b"\r\n", b"\n"):
        line = await reader.readline()
        if not line:
            return None
    parts = line.decode("latin-1").strip().split(" ")
    if len(parts) < 2:
        return None
    method, path = parts[0], parts[1]
    headers: dict[str, str] = {}
    while True:
        raw = await reader.readline()
        if not raw or raw in (b"\r\n", b"\n"):
            break
        name, _, value = raw.decode("latin-1").partition(":")
        headers[name.strip().lower()] = value.strip()
    length = int(headers.get("content-length") or 0)
    body = await reader.readexactly(length) if length else b""
    return method, path, headers, body


async def _send(
    writer: asyncio.StreamWriter,
    status: int,
    body: bytes,
    *,
    content_type: str = "application/json",
) -> None:
    """Write one HTTP response and flush it."""
    reason = {
        200: "OK",
        202: "Accepted",
        204: "No Content",
        400: "Bad Request",
        404: "Not Found",
        405: "Method Not Allowed",
    }.get(status, "Error")
    head = (
        f"HTTP/1.1 {status} {reason}\r\n"
        f"Content-Type: {content_type}\r\n"
        f"Content-Length: {len(body)}\r\n"
        "Connection: keep-alive\r\n"
        "\r\n"
    ).encode("latin-1")
    writer.write(head + body)
    await writer.drain()


__all__ = [
    "AcpMcpServer",
    "AcpToolBridge",
    "DEFAULT_CALL_TIMEOUT",
    "MCP_SERVER_NAME",
    "PendingCall",
]
