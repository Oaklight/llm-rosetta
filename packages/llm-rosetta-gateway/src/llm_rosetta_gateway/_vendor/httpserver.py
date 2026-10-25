# /// zerodep
# version = "0.7.0"
# deps = []
# tier = "subsystem"
# category = "network"
# note = "Install/update via `zerodep add httpserver`"
# ///

"""Zero-dependency async HTTP server with decorator-based routing.

Part of zerodep: https://github.com/Oaklight/zerodep
Copyright (c) 2026 Peng Ding. MIT License.

Async HTTP/1.1 server built on ``asyncio.start_server()``.  Supports
decorator-based routing, JSON request/response, static file serving,
streaming responses (SSE), and graceful shutdown.

Usage::

    from httpserver import App, JSONResponse

    app = App()

    @app.route("/status")
    async def status(request):
        return JSONResponse({"state": "idle"})

    @app.route("/echo", methods=["POST"])
    def echo(request):
        return JSONResponse(request.json())

    app.run(host="127.0.0.1", port=8000)
"""

# ── Imports ──────────────────────────────────────────────────────────────────

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import json as _json
import logging
import mimetypes
import os
import re
import signal
import ssl
import sys
import uuid
from collections.abc import AsyncIterator, Callable, Mapping
from email.utils import formatdate
from http.cookies import CookieError, SimpleCookie
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import parse_qs, unquote, urlparse

__all__ = [
    # Application
    "App",
    # Request / Response
    "Request",
    "Response",
    "JSONResponse",
    "StreamingResponse",
    "FileResponse",
    "State",
    "RejectedRequest",
    # Exceptions
    "HTTPException",
    # Utilities
    "abort",
]

logger = logging.getLogger(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_MAX_BODY_SIZE = 1_048_576  # 1 MB
DEFAULT_READ_TIMEOUT = 30.0  # seconds
DEFAULT_KEEP_ALIVE_TIMEOUT = 5.0  # seconds between requests on one connection
DEFAULT_MAX_REQUESTS_PER_CONNECTION = 100
DEFAULT_SHUTDOWN_GRACE_TIMEOUT = 5.0  # seconds to let in-flight requests finish

_STATUS_REASONS: dict[int, str] = {
    100: "Continue",
    101: "Switching Protocols",
    200: "OK",
    201: "Created",
    202: "Accepted",
    204: "No Content",
    301: "Moved Permanently",
    302: "Found",
    303: "See Other",
    304: "Not Modified",
    307: "Temporary Redirect",
    308: "Permanent Redirect",
    400: "Bad Request",
    401: "Unauthorized",
    403: "Forbidden",
    404: "Not Found",
    405: "Method Not Allowed",
    406: "Not Acceptable",
    408: "Request Timeout",
    409: "Conflict",
    413: "Content Too Large",
    415: "Unsupported Media Type",
    422: "Unprocessable Entity",
    429: "Too Many Requests",
    500: "Internal Server Error",
    502: "Bad Gateway",
    503: "Service Unavailable",
    504: "Gateway Timeout",
}

# Type converters for path parameters
_PARAM_CONVERTERS: dict[str, tuple[str, Callable[[str], Any]]] = {
    "str": (r"([^/]+)", str),
    "int": (r"(-?\d+)", int),
    "float": (r"(-?[0-9]+(?:\.[0-9]+)?)", float),
    "path": (r"(.+)", str),
}

# Regex to find <type:name> or <name> segments in route patterns
_PARAM_RE = re.compile(r"<(?:(\w+):)?(\w+)>")

_SENTINEL = object()

# Seconds between disconnect checks while waiting for the next chunk from
# a streaming generator.  Only matters when the generator is blocked on
# slow upstream I/O; fast generators never hit the timeout.
_STREAMING_DISCONNECT_CHECK_INTERVAL: float = 30.0


# ── Exceptions ───────────────────────────────────────────────────────────────


class HTTPException(Exception):
    """HTTP error that maps to a specific status code.

    Raise directly or via :func:`abort` to short-circuit request handling.

    Args:
        status_code: HTTP status code (e.g. 404, 500).
        message: Optional human-readable error message.
    """

    __slots__ = ("status_code", "message")

    def __init__(self, status_code: int, message: str | None = None):
        self.status_code = status_code
        self.message = message or _STATUS_REASONS.get(status_code, "Error")
        super().__init__(self.message)


class _BadRequest(Exception):
    """Internal: malformed HTTP request from client."""


class _IdleTimeout(asyncio.TimeoutError):
    """No request line arrived within the keep-alive idle window.

    Subclasses :class:`asyncio.TimeoutError` so callers that only care about
    "the read timed out" keep working, while :meth:`App._handle_connection`
    can close a reused connection quietly instead of answering ``408``.
    """


class _ClientDisconnected(Exception):
    """Internal: the client closed the connection between requests."""


def abort(status_code: int, message: str | None = None) -> None:
    """Raise an :class:`HTTPException` with the given status code.

    Args:
        status_code: HTTP status code.
        message: Optional error message.
    """
    raise HTTPException(status_code, message)


# ── Request State ────────────────────────────────────────────────────────────


class State:
    """Mutable namespace for storing arbitrary per-request data.

    Middleware and handlers can attach attributes freely::

        @app.before_request
        async def start_timer(req):
            req.state.start_time = time.monotonic()

    Uses ``__dict__``-based attribute access (no ``__slots__``), following
    the same pattern as Starlette's ``State``.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.__dict__.update(kwargs)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, State):
            return NotImplemented
        return self.__dict__ == other.__dict__

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"State({items})"


# ── Request ──────────────────────────────────────────────────────────────────


class Request:
    """Parsed HTTP request.

    Attributes:
        method: Uppercase HTTP method (GET, POST, ...).
        path: URL path, percent-decoded.
        query_string: Raw query string (without leading ``?``).
        query_params: Parsed query string as ``{key: [values]}``.
        headers: Case-insensitive header dict (keys stored lowercase).
        body: Raw request body bytes.
        path_params: Parameters extracted from the route pattern.
        client_addr: Client ``(host, port)`` tuple.
        app: Reference to the :class:`App` instance handling this request.
        state: Per-request :class:`State` namespace for arbitrary data.
    """

    __slots__ = (
        "method",
        "path",
        "query_string",
        "query_params",
        "headers",
        "body",
        "path_params",
        "client_addr",
        "app",
        "state",
        "_json",
        "_cookies",
    )

    def __init__(
        self,
        method: str,
        path: str,
        query_string: str,
        headers: dict[str, str],
        body: bytes,
        client_addr: tuple[str, int],
        app: App | None = None,
    ):
        self.method = method
        self.path = path
        self.query_string = query_string
        self.query_params: dict[str, list[str]] = parse_qs(query_string)
        self.headers = headers
        self.body = body
        self.path_params: dict[str, Any] = {}
        self.client_addr = client_addr
        self.app = app
        self.state = State()
        self._json: Any = _SENTINEL
        self._cookies: Any = _SENTINEL

    @property
    def cookies(self) -> dict[str, str]:
        """Parse the Cookie request header into a ``{name: value}`` dict."""
        if self._cookies is _SENTINEL:
            raw = self.headers.get("cookie", "")
            if raw:
                sc = SimpleCookie()
                try:
                    sc.load(raw)
                except CookieError:
                    self._cookies = {}
                else:
                    self._cookies = {k: m.value for k, m in sc.items()}
            else:
                self._cookies = {}
        return self._cookies

    def json(self) -> Any:
        """Parse body as JSON (cached)."""
        if self._json is _SENTINEL:
            self._json = _json.loads(self.body)
        return self._json

    def text(self) -> str:
        """Decode body as UTF-8."""
        return self.body.decode("utf-8")

    def form(self) -> dict[str, list[str]]:
        """Parse URL-encoded form body."""
        return parse_qs(self.body.decode("utf-8"))


# ── Response Classes ─────────────────────────────────────────────────────────


def _build_set_cookie(
    name: str,
    value: str = "",
    *,
    max_age: int | None = None,
    expires: str | None = None,
    path: str | None = None,
    domain: str | None = None,
    secure: bool = False,
    httponly: bool = False,
    samesite: str | None = None,
) -> str:
    """Build a Set-Cookie header value string."""
    sc = SimpleCookie()
    sc[name] = value
    morsel = sc[name]
    if max_age is not None:
        morsel["max-age"] = str(max_age)
    if expires is not None:
        morsel["expires"] = expires
    if path is not None:
        morsel["path"] = path
    if domain is not None:
        morsel["domain"] = domain
    if secure:
        morsel["secure"] = True
    if httponly:
        morsel["httponly"] = True
    if samesite is not None:
        morsel["samesite"] = samesite
    return morsel.OutputString()


def _append_set_cookie(
    cookie_headers: list[str],
    name: str,
    value: str = "",
    *,
    max_age: int | None = None,
    expires: str | None = None,
    path: str | None = None,
    domain: str | None = None,
    secure: bool = False,
    httponly: bool = False,
    samesite: str | None = None,
) -> None:
    """Append a Set-Cookie header value to the list."""
    cookie_headers.append(
        _build_set_cookie(
            name,
            value,
            max_age=max_age,
            expires=expires,
            path=path,
            domain=domain,
            secure=secure,
            httponly=httponly,
            samesite=samesite,
        )
    )


def _append_delete_cookie(
    cookie_headers: list[str],
    name: str,
    *,
    path: str | None = None,
    domain: str | None = None,
) -> None:
    """Append a Set-Cookie header that expires the named cookie."""
    _append_set_cookie(
        cookie_headers,
        name,
        value="",
        max_age=0,
        expires="Thu, 01 Jan 1970 00:00:00 GMT",
        path=path,
        domain=domain,
    )


class Response:
    """HTTP response with a fixed body.

    Args:
        body: Response body (bytes or str).
        status_code: HTTP status code.
        headers: Extra response headers.
        content_type: Shorthand for ``Content-Type`` header.
    """

    __slots__ = ("status_code", "headers", "body", "_cookie_headers")

    def __init__(
        self,
        body: bytes | str = b"",
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        content_type: str | None = None,
    ):
        self.status_code = status_code
        self.headers: dict[str, str] = headers.copy() if headers else {}
        if isinstance(body, str):
            self.body = body.encode("utf-8")
        else:
            self.body = body
        if content_type is not None:
            self.headers["Content-Type"] = content_type
        self._cookie_headers: list[str] = []

    def set_cookie(
        self,
        name: str,
        value: str = "",
        *,
        max_age: int | None = None,
        expires: str | None = None,
        path: str | None = None,
        domain: str | None = None,
        secure: bool = False,
        httponly: bool = False,
        samesite: str | None = None,
    ) -> None:
        """Append a Set-Cookie header to the response.

        Args:
            name: Cookie name.
            value: Cookie value.
            max_age: Max age in seconds.
            expires: Expiry date string (HTTP date format).
            path: Cookie path scope.
            domain: Cookie domain scope.
            secure: Restrict to HTTPS.
            httponly: Restrict to HTTP (no JavaScript access).
            samesite: SameSite attribute (``"Strict"``, ``"Lax"``, or ``"None"``).
        """
        _append_set_cookie(
            self._cookie_headers,
            name,
            value,
            max_age=max_age,
            expires=expires,
            path=path,
            domain=domain,
            secure=secure,
            httponly=httponly,
            samesite=samesite,
        )

    def delete_cookie(
        self,
        name: str,
        *,
        path: str | None = None,
        domain: str | None = None,
    ) -> None:
        """Append a Set-Cookie header that expires the named cookie.

        Args:
            name: Cookie name to delete.
            path: Must match the path used when the cookie was set.
            domain: Must match the domain used when the cookie was set.
        """
        _append_delete_cookie(
            self._cookie_headers,
            name,
            path=path,
            domain=domain,
        )

    async def _write(
        self,
        writer: asyncio.StreamWriter,
        reader: asyncio.StreamReader | None = None,
    ) -> None:
        """Serialize and write the full HTTP response.

        Args:
            writer: Transport to write to.
            reader: Unused here; accepted so the dispatch site can pass the
                request reader uniformly to streaming responses.
        """
        reason = _STATUS_REASONS.get(self.status_code, "Unknown")
        self.headers.setdefault("Content-Length", str(len(self.body)))
        self.headers.setdefault("Content-Type", "text/plain; charset=utf-8")
        self.headers.setdefault("Date", _http_date())
        self.headers.setdefault("Connection", "close")

        buf = bytearray()
        buf.extend(f"HTTP/1.1 {self.status_code} {reason}\r\n".encode("latin-1"))
        for k, v in self.headers.items():
            buf.extend(f"{k}: {v}\r\n".encode("latin-1"))
        for cookie_line in self._cookie_headers:
            buf.extend(f"Set-Cookie: {cookie_line}\r\n".encode("latin-1"))
        buf.extend(b"\r\n")
        buf.extend(self.body)
        writer.write(bytes(buf))
        await writer.drain()


class JSONResponse(Response):
    """Response serialized as JSON.

    Non-finite floats (``NaN``, ``Infinity``) are rejected rather than
    written: they are not valid JSON, and a server that emits them hands
    clients a body ``JSON.parse`` and Go's ``encoding/json`` refuse. Starlette
    makes the same choice. Pass ``allow_nan=True`` to emit Python's
    non-standard constants anyway; an application that wants ``null`` for
    non-finite values should sanitize the payload before building the
    response.

    Args:
        data: Python object to serialize.
        status_code: HTTP status code.
        headers: Extra response headers.
        allow_nan: Emit ``NaN`` / ``Infinity`` instead of raising
            :class:`ValueError`. Defaults to ``False``.

    Raises:
        ValueError: If *data* contains a non-finite float and
            *allow_nan* is ``False``.
    """

    def __init__(
        self,
        data: Any,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        *,
        allow_nan: bool = False,
    ):
        body = _json.dumps(data, ensure_ascii=False, allow_nan=allow_nan).encode(
            "utf-8"
        )
        super().__init__(
            body=body,
            status_code=status_code,
            headers=headers,
            content_type="application/json; charset=utf-8",
        )


class StreamingResponse:
    """HTTP response streamed from an async generator.

    All streaming responses use ``Transfer-Encoding: chunked`` for
    maximum compatibility with reverse proxies and intermediaries.
    SSE (``text/event-stream``) responses additionally set
    ``Cache-Control: no-cache``.

    Args:
        generator: Async iterator yielding ``bytes`` or ``str`` chunks.
        status_code: HTTP status code.
        headers: Extra response headers.
        content_type: MIME type (default ``application/octet-stream``).
        background: Optional callable invoked after the stream completes
            (including client disconnect).  Accepts both sync and async
            callables.  Exceptions are logged and suppressed.
        disconnect_check_interval: Seconds between client-disconnect
            checks while the generator is blocked.  Defaults to 30.
        disconnect_on_eof: Treat the client's read-side EOF as a disconnect
            while the generator is blocked.  TCP cannot distinguish a
            half-close, where the client shut down its write side but is
            still reading -- legal, and what the request reader reports as
            EOF -- from a dropped connection.  Off by default so
            half-closing clients keep receiving the stream; enable it when
            clients never half-close (browsers and common HTTP libraries)
            to tear down a generator whose upstream never yields again.
    """

    __slots__ = (
        "_generator",
        "status_code",
        "headers",
        "content_type",
        "background",
        "_cookie_headers",
        "_disconnect_check_interval",
        "_disconnect_on_eof",
    )

    def __init__(
        self,
        generator: AsyncIterator[bytes | str],
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        content_type: str = "application/octet-stream",
        background: Callable[[], Any] | None = None,
        disconnect_check_interval: float | None = None,
        disconnect_on_eof: bool = False,
    ):
        self._generator = generator
        self.status_code = status_code
        self.headers: dict[str, str] = headers.copy() if headers else {}
        self.content_type = content_type
        self.background = background
        self._cookie_headers: list[str] = []
        self._disconnect_on_eof = disconnect_on_eof
        self._disconnect_check_interval = (
            disconnect_check_interval
            if disconnect_check_interval is not None
            else _STREAMING_DISCONNECT_CHECK_INTERVAL
        )

    def set_cookie(
        self,
        name: str,
        value: str = "",
        *,
        max_age: int | None = None,
        expires: str | None = None,
        path: str | None = None,
        domain: str | None = None,
        secure: bool = False,
        httponly: bool = False,
        samesite: str | None = None,
    ) -> None:
        """Append a Set-Cookie header to the response."""
        _append_set_cookie(
            self._cookie_headers,
            name,
            value,
            max_age=max_age,
            expires=expires,
            path=path,
            domain=domain,
            secure=secure,
            httponly=httponly,
            samesite=samesite,
        )

    def delete_cookie(
        self,
        name: str,
        *,
        path: str | None = None,
        domain: str | None = None,
    ) -> None:
        """Append a Set-Cookie header that expires the named cookie."""
        _append_delete_cookie(
            self._cookie_headers,
            name,
            path=path,
            domain=domain,
        )

    async def _write(
        self,
        writer: asyncio.StreamWriter,
        reader: asyncio.StreamReader | None = None,
    ) -> None:
        """Write status line, headers, then stream the body.

        Raises ``BrokenPipeError``, ``ConnectionResetError``, or
        ``ConnectionAbortedError`` on client disconnect.  The caller
        (``_handle_connection``) is responsible for handling these.
        Generator cleanup and the background callback still run via
        the ``finally`` block before the exception propagates.

        Args:
            writer: Transport to write the response to.
            reader: Request reader, when available.  A client that closes
                its side of the connection leaves ``writer.is_closing()``
                false on some platforms, but ``reader.at_eof()`` turns true;
                streaming uses both to tear down a blocked generator.
        """
        reason = _STATUS_REASONS.get(self.status_code, "Unknown")
        is_sse = self.content_type.startswith("text/event-stream")

        self.headers["Content-Type"] = self.content_type
        self.headers.setdefault("Transfer-Encoding", "chunked")
        if is_sse:
            self.headers.setdefault("Cache-Control", "no-cache")
        self.headers.setdefault("Date", _http_date())
        self.headers.setdefault("Connection", "close")

        buf = bytearray()
        buf.extend(f"HTTP/1.1 {self.status_code} {reason}\r\n".encode("latin-1"))
        for k, v in self.headers.items():
            buf.extend(f"{k}: {v}\r\n".encode("latin-1"))
        for cookie_line in self._cookie_headers:
            buf.extend(f"Set-Cookie: {cookie_line}\r\n".encode("latin-1"))
        buf.extend(b"\r\n")
        writer.write(bytes(buf))
        await writer.drain()

        try:
            aiter = self._generator.__aiter__()
            is_closing = getattr(writer, "is_closing", None)
            at_eof = (
                getattr(reader, "at_eof", None)
                if reader is not None and self._disconnect_on_eof
                else None
            )

            def disconnected() -> bool:
                """Whether the peer is gone, per the writer or the read side.

                ``is_closing()`` stays false on some platforms after the
                peer sends FIN, while ``at_eof()`` turns true.
                """
                if is_closing is not None and is_closing():
                    return True
                return at_eof is not None and at_eof()

            while True:
                # Detect client disconnect between generator yields.
                # When the generator blocks on slow upstream I/O (e.g.
                # an LLM API producing the next SSE chunk), plain
                # ``async for`` never checks the writer state.  We
                # race the next-chunk coroutine against a periodic
                # disconnect check so CLOSE_WAIT connections don't
                # accumulate.
                if is_closing is not None and is_closing():
                    raise BrokenPipeError("client disconnected (detected)")

                next_coro = aiter.__anext__()
                if is_closing is None and at_eof is None:
                    # Writer and reader cannot report a disconnect (mock) —
                    # fall back to plain await.
                    try:
                        chunk = await next_coro
                    except StopAsyncIteration:
                        break
                else:
                    next_task = asyncio.ensure_future(next_coro)
                    try:
                        chunk = await _anext_or_disconnect(
                            next_task,
                            disconnected,
                            self._disconnect_check_interval,
                        )
                    except StopAsyncIteration:
                        break
                    except (
                        BrokenPipeError,
                        ConnectionResetError,
                        ConnectionAbortedError,
                    ):
                        next_task.cancel()
                        try:
                            await next_task
                        except (asyncio.CancelledError, Exception):
                            pass
                        raise

                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")
                writer.write(f"{len(chunk):x}\r\n".encode("latin-1"))
                writer.write(chunk)
                writer.write(b"\r\n")
                await writer.drain()
            writer.write(b"0\r\n\r\n")
            await writer.drain()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            raise
        finally:
            aclose = getattr(self._generator, "aclose", None)
            if aclose is not None:
                try:
                    await aclose()
                except Exception:
                    logger.debug("Generator aclose failed", exc_info=True)
            if self.background is not None:
                try:
                    result = self.background()
                    if inspect.isawaitable(result):
                        await result
                except Exception:
                    logger.warning("Background callback failed", exc_info=True)


class FileResponse(Response):
    """Response serving a file from disk.

    Args:
        path: Path to the file.
        content_type: MIME type (auto-detected from extension if ``None``).
        status_code: HTTP status code.
    """

    def __init__(
        self,
        path: str | Path,
        content_type: str | None = None,
        status_code: int = 200,
    ):
        file_path = Path(path)
        body = file_path.read_bytes()
        if content_type is None:
            guessed, _ = mimetypes.guess_type(str(file_path))
            content_type = guessed or "application/octet-stream"
        headers = {
            "Last-Modified": _http_date(os.path.getmtime(file_path)),
        }
        super().__init__(
            body=body,
            status_code=status_code,
            headers=headers,
            content_type=content_type,
        )


# ── Routing ──────────────────────────────────────────────────────────────────


@dataclasses.dataclass(frozen=True, slots=True)
class _Route:
    """Internal route entry."""

    methods: list[str]  # uppercase, e.g. ["GET", "POST"], or ["*"]
    pattern: re.Pattern[str]
    handler: Callable[..., Any]
    is_async: bool
    param_names: list[str]
    param_converters: list[Callable[[str], Any]]


def _compile_route(
    path: str,
) -> tuple[re.Pattern[str], list[str], list[Callable[[str], Any]]]:
    """Compile a path pattern into a regex and parameter metadata.

    Supports ``<name>``, ``<int:name>``, ``<float:name>``, ``<path:name>``.

    Args:
        path: URL path pattern (e.g. ``/users/<int:id>``).

    Returns:
        Tuple of (compiled regex, param names, param converters).
    """
    param_names: list[str] = []
    param_converters: list[Callable[[str], Any]] = []
    regex_parts: list[str] = []
    last_end = 0

    for m in _PARAM_RE.finditer(path):
        regex_parts.append(re.escape(path[last_end : m.start()]))
        type_name = m.group(1) or "str"
        name = m.group(2)
        if type_name not in _PARAM_CONVERTERS:
            raise ValueError(f"Unknown path parameter type: {type_name!r}")
        regex_fragment, converter = _PARAM_CONVERTERS[type_name]
        regex_parts.append(regex_fragment)
        param_names.append(name)
        param_converters.append(converter)
        last_end = m.end()

    regex_parts.append(re.escape(path[last_end:]))
    full_regex = "^" + "".join(regex_parts) + "$"
    return re.compile(full_regex), param_names, param_converters


@dataclasses.dataclass(frozen=True, slots=True)
class RejectedRequest:
    """A request refused while it was being read, before any dispatch.

    Passed to :meth:`App.on_request_rejected` handlers.  Fields describing
    the request are best-effort: ``None`` (or ``0``) when the failure
    happened before that part of the request was parsed.

    Attributes:
        client_addr: Peer address as ``(host, port)``.
        status_code: Status of the error response being sent.
        reason: Stable token: ``"bad_request"``, ``"too_large"``,
            ``"timeout"``, ``"incomplete_body"`` or ``"read_error"``.
        exception: The exception that caused the rejection.
        request_id: Value sent as ``X-Request-Id``: the client's own when
            its headers were parsed far enough to carry one, otherwise a
            generated id.
        method: Request method, once the request line was parsed.
        path: Decoded request path, once the request line was parsed.
        headers: Read-only view of the lower-cased request headers parsed
            so far, or ``None``
            when the request line itself was not usable.
        content_length: Declared ``Content-Length``, when present and valid.
        bytes_received: Body bytes read before the failure.
    """

    client_addr: tuple[str, int]
    status_code: int
    reason: str
    exception: BaseException | None
    request_id: str
    method: str | None = None
    path: str | None = None
    headers: Mapping[str, str] | None = None
    content_length: int | None = None
    bytes_received: int = 0


class _ReadState:
    """What :func:`_read_request` had parsed when it stopped."""

    __slots__ = ("method", "path", "headers", "content_length", "bytes_received")

    def __init__(self) -> None:
        self.method: str | None = None
        self.path: str | None = None
        self.headers: dict[str, str] | None = None
        self.content_length: int | None = None
        self.bytes_received = 0


# A single token of URL-safe characters: no spaces, commas or semicolons that a
# proxy could read as separate values.
_REQUEST_ID_RE = re.compile(r"[A-Za-z0-9._~:/+=@-]{1,128}")


def _request_id_from(headers: dict[str, str] | None) -> str:
    """Return the client's ``X-Request-Id`` if usable, else a generated id."""
    candidate = (headers or {}).get("x-request-id", "").strip()
    if _REQUEST_ID_RE.fullmatch(candidate):
        return candidate
    return uuid.uuid4().hex


# ── HTTP Parsing ─────────────────────────────────────────────────────────────


def _read_failure_reason(exc: BaseException) -> str:
    """Stable ``RejectedRequest.reason`` token for a read-phase exception."""
    if isinstance(exc, _BadRequest):
        return "bad_request"
    # _read_request only raises HTTPException for 413.
    if isinstance(exc, HTTPException) and exc.status_code == 413:
        return "too_large"
    if isinstance(exc, asyncio.TimeoutError):
        return "timeout"
    if isinstance(exc, asyncio.IncompleteReadError):
        return "incomplete_body"
    return "read_error"


def _read_error_response(exc: Exception) -> Response:
    """Map an exception raised while reading a request to a response.

    Args:
        exc: The exception.

    Returns:
        The response to send before closing the connection.
    """
    if isinstance(exc, _BadRequest):
        return Response(
            body=str(exc),
            status_code=400,
            content_type="text/plain; charset=utf-8",
        )
    if isinstance(exc, HTTPException):
        return JSONResponse({"error": exc.message}, status_code=exc.status_code)
    if isinstance(exc, asyncio.TimeoutError):
        return JSONResponse({"error": "Request Timeout"}, status_code=408)
    if isinstance(exc, asyncio.IncompleteReadError):
        return JSONResponse({"error": "Bad Request: incomplete body"}, status_code=400)
    return JSONResponse({"error": "Internal Server Error"}, status_code=500)


async def _read_request(
    reader: asyncio.StreamReader,
    timeout: float,
    max_body_size: int,
    *,
    idle_timeout: float | None = None,
    on_request_start: Callable[[], None] | None = None,
    state: _ReadState | None = None,
) -> tuple[str, str, str, dict[str, str], bytes, str]:
    """Read and parse an HTTP request from a stream.

    Args:
        reader: Client stream.
        timeout: Budget for each header and body read once the request line
            has arrived.
        max_body_size: Largest body accepted, in bytes.
        idle_timeout: Budget for the request line itself, used for the wait
            between requests on a kept-alive connection. Defaults to
            *timeout*.
        on_request_start: Called once, as soon as a non-empty request line
            has arrived. :meth:`App._handle_connection` uses it to stop
            treating the connection as idle the moment a request begins,
            so shutdown drains a request that is still arriving instead of
            closing it mid-upload.
        state: Filled in as parsing progresses, so a caller can report what
            had been read when a failure is raised.

    Returns:
        Tuple of (method, path, query_string, headers, body, version).

    Raises:
        _ClientDisconnected: If the stream ended before a request line, which
            is how a client closes a kept-alive connection.
        _IdleTimeout: If no request line arrived within *idle_timeout*.
        _BadRequest: If the request is malformed, including a
            ``Content-Length`` or chunk size that is not a valid number.
        asyncio.TimeoutError: If reading headers or the body times out.
    """
    # -- Request line --
    # The wait for the first byte is the only one bounded by the idle window:
    # once a request has started, its headers and body get the full read
    # budget, so a client that uploads slowly after its headers is not cut off.
    request_line_timeout = timeout if idle_timeout is None else idle_timeout
    try:
        raw_line = await asyncio.wait_for(
            reader.readline(), timeout=request_line_timeout
        )
    except asyncio.TimeoutError:
        if idle_timeout is not None:
            raise _IdleTimeout from None
        raise
    if not raw_line:
        raise _ClientDisconnected
    # The request has started: it is no longer an idle connection, and a
    # shutdown from here on waits for it (up to the grace period).
    if on_request_start is not None:
        on_request_start()
    request_line = raw_line.decode("latin-1").rstrip("\r\n")
    parts = request_line.split(" ", 2)
    if len(parts) != 3:
        raise _BadRequest(f"Malformed request line: {request_line!r}")
    method, raw_url, version = parts

    # -- URL --
    parsed = urlparse(raw_url)
    path = unquote(parsed.path)
    query_string = parsed.query
    if state is not None:
        state.method = method.upper()
        state.path = path

    # -- Headers --
    headers: dict[str, str] = {}
    if state is not None:
        state.headers = headers
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        decoded = line.decode("latin-1").rstrip("\r\n")
        if not decoded:
            break
        if ":" in decoded:
            k, v = decoded.split(":", 1)
            headers[k.strip().lower()] = v.strip()

    # -- Body --
    body = b""
    cl = headers.get("content-length")
    if cl is not None:
        # 1*DIGIT only: int() would also take "1_0", "+5" and non-ASCII digits.
        if not (cl.isascii() and cl.isdigit()):
            raise _BadRequest(f"Invalid Content-Length: {cl!r}")
        length = int(cl)
        if state is not None:
            state.content_length = length
        if length > max_body_size:
            raise HTTPException(413, f"Request body too large ({length} bytes)")
        if length > 0:
            body = await asyncio.wait_for(
                _read_exact_body(reader, length, state), timeout=timeout
            )
    elif headers.get("transfer-encoding", "").lower() == "chunked":
        body = await _read_chunked_body(reader, timeout, max_body_size, state)

    return method.upper(), path, query_string, headers, body, version.upper()


_CHUNK_SIZE_RE = re.compile(r"[0-9A-Fa-f]+")  # 1*HEXDIG, no sign or 0x prefix


async def _read_exact_body(
    reader: asyncio.StreamReader,
    length: int,
    state: _ReadState | None,
) -> bytes:
    """Read *length* body bytes, counting them in *state* as they arrive.

    Unlike ``readexactly`` this keeps the byte count visible when the caller
    cancels the read on a timeout.

    Raises:
        asyncio.IncompleteReadError: If the stream ends first.
    """
    parts: list[bytes] = []
    remaining = length
    while remaining:
        chunk = await reader.read(min(remaining, 65536))
        if not chunk:
            raise asyncio.IncompleteReadError(b"".join(parts), length)
        parts.append(chunk)
        remaining -= len(chunk)
        if state is not None:
            state.bytes_received += len(chunk)
    return b"".join(parts)


async def _read_chunked_body(
    reader: asyncio.StreamReader,
    timeout: float,
    max_body_size: int,
    state: _ReadState | None = None,
) -> bytes:
    """Read a chunked transfer-encoded body."""
    parts: list[bytes] = []
    total = 0
    while True:
        size_line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        size_str = size_line.decode("latin-1").split(";")[0].strip()
        if not size_str:
            break
        if not _CHUNK_SIZE_RE.fullmatch(size_str):
            raise _BadRequest(f"Invalid chunk size: {size_str!r}")
        chunk_size = int(size_str, 16)
        if chunk_size == 0:
            await asyncio.wait_for(reader.readline(), timeout=timeout)
            break
        total += chunk_size
        if total > max_body_size:
            raise HTTPException(413, "Request body too large")
        data = await asyncio.wait_for(reader.readexactly(chunk_size), timeout=timeout)
        await asyncio.wait_for(reader.readline(), timeout=timeout)
        parts.append(data)
        if state is not None:
            state.bytes_received = total
    return b"".join(parts)


# ── Utilities ────────────────────────────────────────────────────────────────


async def _anext_or_disconnect(
    next_task: asyncio.Task[Any],
    is_disconnected: Callable[[], bool],
    interval: float,
) -> Any:
    """Await *next_task* while periodically checking for client disconnect.

    Polls *is_disconnected* every *interval* seconds.  If it reports the
    peer gone, cancels the pending ``__anext__`` task and raises
    ``BrokenPipeError`` so the caller can clean up the generator.

    The predicate may include the read side: after a peer FIN,
    ``writer.is_closing()`` can stay false while ``reader.at_eof()`` turns
    true.  Callers decide whether EOF counts, because a half-close (client
    stops sending, keeps reading) looks identical on the wire.

    If the task completes normally the chunk value is returned.  If it
    raises ``StopAsyncIteration`` (generator exhausted) that exception
    propagates to the caller unchanged.
    """
    while True:
        done, _ = await asyncio.wait({next_task}, timeout=interval)
        if done:
            # Propagates StopAsyncIteration to the caller unchanged.
            return next_task.result()
        if is_disconnected():
            next_task.cancel()
            try:
                await next_task
            except (asyncio.CancelledError, StopAsyncIteration, Exception):
                pass
            raise BrokenPipeError("client disconnected (detected)")
        # Generator is still working on the next chunk; keep waiting.


def _http_date(timestamp: float | None = None) -> str:
    """Format a timestamp as an HTTP-date (RFC 7231)."""
    return formatdate(timeval=timestamp, localtime=False, usegmt=True)


def _coerce_response(result: Any) -> Response | StreamingResponse:
    """Convert a handler return value into a Response object."""
    if isinstance(result, (Response, StreamingResponse)):
        return result

    if result is None:
        return Response(status_code=204)

    if isinstance(result, dict):
        return JSONResponse(result)

    if isinstance(result, tuple):
        if len(result) == 2:
            body, status = result
            resp = _coerce_response(body)
            resp.status_code = status
            return resp
        if len(result) == 3:
            body, status, extra_headers = result
            resp = _coerce_response(body)
            resp.status_code = status
            resp.headers.update(extra_headers)
            return resp
        raise ValueError(f"Unsupported tuple length: {len(result)}")

    if isinstance(result, bytes):
        return Response(body=result, content_type="application/octet-stream")

    if isinstance(result, str):
        return Response(body=result, content_type="text/plain; charset=utf-8")

    raise TypeError(
        f"Cannot coerce handler return type {type(result).__name__} to Response"
    )


# ── Static File Resolution ───────────────────────────────────────────────────


def _resolve_static_file(
    request_path: str,
    url_prefix: str,
    directory: str,
) -> FileResponse | None:
    """Try to resolve a static file request.

    Returns a FileResponse if the file exists and the path is safe,
    otherwise None.
    """
    if not request_path.startswith(url_prefix):
        return None

    relative = request_path[len(url_prefix) :].lstrip("/")
    if not relative:
        return None

    base = Path(directory).resolve()
    target = (base / relative).resolve()

    # Prevent directory traversal
    if not str(target).startswith(str(base)):
        return None

    if not target.is_file():
        return None

    return FileResponse(target)


# ── App ──────────────────────────────────────────────────────────────────────


class App:
    """Async HTTP server application.

    Request lifecycle::

        before_request → route handler → after_request
          → on_response_started → write → on_response_completed
                                    ↘ on_client_disconnect

    Args:
        max_body_size: Maximum request body size in bytes.
        read_timeout: Timeout for reading a single request (seconds).
        keep_alive: Serve more than one request per connection.  HTTP/1.1
            connections stay open unless the client or the response asks to
            close; HTTP/1.0 stays open only when the client asks.  Streaming
            responses always close.  Defaults to ``True``.
        keep_alive_timeout: Seconds an idle kept-alive connection may wait for
            the next request before the server closes it quietly.  ``0`` or a
            negative value disables reuse, so the response never advertises a
            connection the server will not keep.  Defaults to
            :data:`DEFAULT_KEEP_ALIVE_TIMEOUT`.
        max_requests_per_connection: Close a connection after this many
            requests even if the client would reuse it.  Defaults to
            :data:`DEFAULT_MAX_REQUESTS_PER_CONNECTION`.
        shutdown_grace_timeout: Seconds a shutdown waits for requests that
            are already being handled before cancelling them.  On Python
            3.12+ ``Server.wait_closed()`` provides the same guarantee, but
            it returns immediately on 3.10/3.11, where the wait is needed.
            Defaults to :data:`DEFAULT_SHUTDOWN_GRACE_TIMEOUT`.

    Example::

        app = App()

        @app.get("/hello")
        async def hello(request):
            return {"message": "Hello, world!"}

        app.run()
    """

    def __init__(
        self,
        *,
        max_body_size: int = DEFAULT_MAX_BODY_SIZE,
        read_timeout: float = DEFAULT_READ_TIMEOUT,
        keep_alive: bool = True,
        keep_alive_timeout: float = DEFAULT_KEEP_ALIVE_TIMEOUT,
        max_requests_per_connection: int = DEFAULT_MAX_REQUESTS_PER_CONNECTION,
        shutdown_grace_timeout: float = DEFAULT_SHUTDOWN_GRACE_TIMEOUT,
    ):
        self._routes: list[_Route] = []
        self._static_routes: list[tuple[str, str]] = []
        self._before_request_handlers: list[Callable[..., Any]] = []
        self._after_request_handlers: list[Callable[..., Any]] = []
        self._startup_handlers: list[Callable[[], Any]] = []
        self._shutdown_handlers: list[Callable[[], Any]] = []
        self._on_response_started_handlers: list[Callable[..., Any]] = []
        self._on_response_completed_handlers: list[Callable[..., Any]] = []
        self._on_client_disconnect_handlers: list[Callable[..., Any]] = []
        self._on_request_rejected_handlers: list[Callable[..., Any]] = []
        self._error_handlers: dict[int | type, Callable[..., Any]] = {}
        self._server: asyncio.Server | None = None
        self._shutdown_event: asyncio.Event | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        #: Connections currently waiting for their next request line.  A
        #: graceful shutdown closes these so ``Server.wait_closed()`` does
        #: not have to wait out ``keep_alive_timeout`` for every idle client
        #: (Python 3.12+ waits for all connections).
        self._idle_connections: set[asyncio.StreamWriter] = set()
        self.max_body_size = max_body_size
        self.read_timeout = read_timeout
        self.keep_alive = keep_alive
        self.keep_alive_timeout = keep_alive_timeout
        self.max_requests_per_connection = max_requests_per_connection
        self.shutdown_grace_timeout = shutdown_grace_timeout
        #: Connection handlers still running, so a shutdown can wait for a
        #: request that is already being handled (see
        #: :meth:`_drain_connection_tasks`).
        self._tasks: set[asyncio.Task[Any]] = set()
        self.port: int | None = None
        self.host: str | None = None

    # ── Route Registration ───────────────────────────────────────────────

    def route(
        self, url_pattern: str, methods: list[str] | None = None
    ) -> Callable[..., Any]:
        """Register a route handler.

        Args:
            url_pattern: URL pattern with optional parameters.
            methods: HTTP methods to handle (e.g. ``["GET", "POST"]``).
                Defaults to ``["GET"]``.

        Example::

            @app.route("/users/<int:id>", methods=["GET", "POST"])
            async def user(request, id):
                return {"id": id}
        """
        if methods is None:
            methods = ["GET"]

        def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
            pattern, param_names, converters = _compile_route(url_pattern)
            is_async = inspect.iscoroutinefunction(handler)
            self._routes.append(
                _Route(
                    methods=[m.upper() for m in methods],
                    pattern=pattern,
                    handler=handler,
                    is_async=is_async,
                    param_names=param_names,
                    param_converters=converters,
                )
            )
            return handler

        return decorator

    def get(self, path: str) -> Callable[..., Any]:
        """Shorthand for ``@app.route(path, methods=["GET"])``."""
        return self.route(path, methods=["GET"])

    def post(self, path: str) -> Callable[..., Any]:
        """Shorthand for ``@app.route(path, methods=["POST"])``."""
        return self.route(path, methods=["POST"])

    def put(self, path: str) -> Callable[..., Any]:
        """Shorthand for ``@app.route(path, methods=["PUT"])``."""
        return self.route(path, methods=["PUT"])

    def delete(self, path: str) -> Callable[..., Any]:
        """Shorthand for ``@app.route(path, methods=["DELETE"])``."""
        return self.route(path, methods=["DELETE"])

    def patch(self, path: str) -> Callable[..., Any]:
        """Shorthand for ``@app.route(path, methods=["PATCH"])``."""
        return self.route(path, methods=["PATCH"])

    def static(self, url_prefix: str, directory: str) -> None:
        """Register a directory for static file serving.

        Args:
            url_prefix: URL prefix (e.g. ``"/static"``).
            directory: Filesystem path to the directory.

        Example::

            app.static("/assets", "./public")
        """
        url_prefix = url_prefix.rstrip("/")
        abs_dir = os.path.abspath(directory)
        self._static_routes.append((url_prefix, abs_dir))

    # ── Middleware ────────────────────────────────────────────────────────

    def before_request(self, handler: Callable[..., Any]) -> Callable[..., Any]:
        """Register a before-request hook.

        The hook receives ``(request)`` and may return a ``Response`` to
        short-circuit the route handler.
        """
        self._before_request_handlers.append(handler)
        return handler

    def after_request(self, handler: Callable[..., Any]) -> Callable[..., Any]:
        """Register an after-request hook.

        The hook receives ``(request, response)`` and may return a new
        or modified ``Response``.
        """
        self._after_request_handlers.append(handler)
        return handler

    def errorhandler(self, code_or_exc: int | type) -> Callable[..., Any]:
        """Register an error handler.

        Args:
            code_or_exc: HTTP status code (int) or exception class.

        The handler receives ``(request, exception)`` and must return a
        ``Response``.
        """

        def decorator(handler: Callable[..., Any]) -> Callable[..., Any]:
            self._error_handlers[code_or_exc] = handler
            return handler

        return decorator

    # ── Lifespan Hooks ───────────────────────────────────────────────────

    def on_startup(self, handler: Callable[[], Any]) -> Callable[[], Any]:
        """Register a startup hook.

        The hook is called (with no arguments) before the server starts
        accepting connections.  Both sync and async callables are supported.
        Hooks run in registration order.

        If a startup hook raises, the error is logged and re-raised so the
        server does not start with failed initialization.

        Example::

            @app.on_startup
            async def init_pool():
                app.pool = await create_pool()
        """
        self._startup_handlers.append(handler)
        return handler

    def on_shutdown(self, handler: Callable[[], Any]) -> Callable[[], Any]:
        """Register a shutdown hook.

        The hook is called (with no arguments) after the server stops
        accepting connections.  Both sync and async callables are supported.
        Hooks run in **reverse** registration order (LIFO).

        If a shutdown hook raises, the error is logged and the remaining
        hooks still execute (best-effort cleanup).

        Example::

            @app.on_shutdown
            async def close_pool():
                await app.pool.close()
        """
        self._shutdown_handlers.append(handler)
        return handler

    # ── Request Lifecycle Signals ────────────────────────────────────────

    def on_response_started(self, handler: Callable[..., Any]) -> Callable[..., Any]:
        """Register a response-started signal.

        Fired just before the response is written to the client.  Useful
        for TTFB (time-to-first-byte) metrics.

        The handler receives ``(request, response)`` and its return value
        is ignored.  Both sync and async callables are supported.
        Exceptions are logged and suppressed.  Handlers run sequentially;
        keep them fast to avoid delaying the response write.

        Example::

            @app.on_response_started
            async def ttfb(request, response):
                request.state.response_start = time.monotonic()
        """
        self._on_response_started_handlers.append(handler)
        return handler

    def on_response_completed(self, handler: Callable[..., Any]) -> Callable[..., Any]:
        """Register a response-completed signal.

        Fired after the entire response body has been successfully sent
        to the client.  Useful for total transfer time and post-response
        logging.

        The handler receives ``(request, response)`` and its return value
        is ignored.  Both sync and async callables are supported.
        Exceptions are logged and suppressed.  Handlers run sequentially.

        Not fired when the client disconnects mid-response (see
        :meth:`on_client_disconnect` for that case).

        Example::

            @app.on_response_completed
            async def log_transfer(request, response):
                elapsed = time.monotonic() - request.state.response_start
                logger.info("Sent %d in %.3fs", response.status_code, elapsed)
        """
        self._on_response_completed_handlers.append(handler)
        return handler

    def on_client_disconnect(self, handler: Callable[..., Any]) -> Callable[..., Any]:
        """Register a client-disconnect signal.

        Fired when the client closes the connection during response
        delivery (broken pipe, connection reset).  Does not fire for
        disconnects during request dispatch (e.g. slow handler).
        Useful for cleanup, metrics, and cancelling expensive work.

        The handler receives ``(request)`` and its return value is ignored.
        Both sync and async callables are supported.  Exceptions are logged
        and suppressed.

        Example::

            @app.on_client_disconnect
            async def on_disconnect(request):
                logger.info("Client %s disconnected", request.client_addr)
        """
        self._on_client_disconnect_handlers.append(handler)
        return handler

    def on_request_rejected(self, handler: Callable[..., Any]) -> Callable[..., Any]:
        """Register a request-rejected signal.

        Fired when a request is refused while it is being read, before a
        :class:`Request` exists, so none of the other lifecycle hooks see
        it: malformed request line, ``Content-Length`` or chunk size (``400``), a body over
        ``max_body_size`` (``413``), a read timeout (``408``), a body
        shorter than ``Content-Length`` (``400``) and any other read error
        (``500``).  A keep-alive connection that goes idle or closes
        between requests is not a rejection.

        The handler receives one :class:`RejectedRequest` and its return
        value is ignored.  Both sync and async callables are supported.
        Exceptions are logged and suppressed.  Handlers run sequentially,
        just before the error response is written.

        Example::

            @app.on_request_rejected
            async def record(rejection):
                logger.warning(
                    "rejected %s (%s) id=%s",
                    rejection.status_code,
                    rejection.reason,
                    rejection.request_id,
                )
        """
        self._on_request_rejected_handlers.append(handler)
        return handler

    async def _fire_signal(
        self, name: str, handlers: list[Callable[..., Any]], *args: Any
    ) -> None:
        for hook in handlers:
            try:
                await _invoke(hook, *args)
            except Exception:
                logger.warning("%s hook failed", name, exc_info=True)

    # ── Request Dispatch ─────────────────────────────────────────────────

    def _match_route(
        self, request: Request
    ) -> tuple[_Route | None, re.Match[str] | None, bool]:
        """Find the first route matching the request path and method.

        Returns:
            (matched_route, regex_match, path_existed).  *path_existed* is
            True when a route matched the path but not the HTTP method.
        """
        # RFC 9110 §9.3.2: a server that supports GET must support HEAD.
        method = "GET" if request.method == "HEAD" else request.method
        path_existed = False
        for route in self._routes:
            m = route.pattern.match(request.path)
            if m is None:
                continue
            path_existed = True
            if "*" not in route.methods and method not in route.methods:
                continue
            return route, m, True
        return None, None, path_existed

    async def _invoke_route(
        self, route: _Route, match: re.Match[str], request: Request
    ) -> Response | StreamingResponse:
        """Extract path params, call the handler, return a response."""
        for name, converter, value in zip(
            route.param_names, route.param_converters, match.groups()
        ):
            request.path_params[name] = converter(value)

        try:
            result = await _invoke(route.handler, request, **request.path_params)
            return _coerce_response(result)
        except HTTPException as exc:
            return await self._handle_error(request, exc)
        except Exception as exc:
            logger.exception(
                "Unhandled exception in handler %s",
                getattr(route.handler, "__name__", repr(route.handler)),
            )
            return await self._handle_error(request, exc)

    async def _run_before_hooks(
        self, request: Request
    ) -> Response | StreamingResponse | None:
        """Run before-request hooks; a failing hook becomes an error response.

        A hook returns a response to short-circuit routing.  Its body is
        serialized here, outside the per-handler ``try`` that guards route
        handlers, so without the catch a hook that returns an unserializable
        body would drop the connection instead of answering.
        """
        for hook in self._before_request_handlers:
            try:
                result = await _invoke(hook, request)
                if result is not None:
                    return _coerce_response(result)
            except HTTPException as exc:
                return await self._handle_error(request, exc)
            except Exception as exc:
                logger.exception(
                    "Unhandled exception in before_request hook %s",
                    getattr(hook, "__name__", repr(hook)),
                )
                return await self._handle_error(request, exc)
        return None

    async def _run_after_hooks(
        self, request: Request, response: Response | StreamingResponse
    ) -> Response | StreamingResponse:
        """Run after-request hooks, allowing them to replace the response."""
        for hook in self._after_request_handlers:
            try:
                hook_result = await _invoke(hook, request, response)
                if hook_result is not None:
                    response = _coerce_response(hook_result)
            except HTTPException as exc:
                return await self._handle_error(request, exc)
            except Exception as exc:
                # Same reason as _run_before_hooks: this runs outside the
                # handler's guard, where an exception costs the response.
                logger.exception(
                    "Unhandled exception in after_request hook %s",
                    getattr(hook, "__name__", repr(hook)),
                )
                return await self._handle_error(request, exc)
        return response

    async def _dispatch(self, request: Request) -> Response | StreamingResponse:
        """Match a request to a route and invoke the handler."""
        # -- before_request hooks --
        short_circuit = await self._run_before_hooks(request)
        if short_circuit is not None:
            return short_circuit

        # -- Static file check --
        for prefix, directory in self._static_routes:
            file_resp = _resolve_static_file(request.path, prefix, directory)
            if file_resp is not None:
                return file_resp

        # -- Route matching --
        route, match, path_existed = self._match_route(request)

        if route is not None and match is not None:
            response = await self._invoke_route(route, match, request)
            return await self._run_after_hooks(request, response)

        # -- No route matched --
        if path_existed:
            allowed = sorted(
                {
                    m
                    for r in self._routes
                    if r.pattern.match(request.path)
                    for m in r.methods
                    if m != "*"
                }
            )
            exc = HTTPException(405, "Method Not Allowed")
            response = await self._handle_error(request, exc)
            if isinstance(response, Response):
                response.headers["Allow"] = ", ".join(allowed)
            return response

        exc = HTTPException(404, "Not Found")
        return await self._handle_error(request, exc)

    async def _run_error_handler(
        self, request: Request, handler: Any, exc: Exception
    ) -> Response | StreamingResponse | None:
        """Render an error through a registered handler.

        Returns ``None`` when the handler itself fails, so the caller falls
        back to the default response: a broken error page must not cost the
        client its response.
        """
        try:
            result = await _invoke(handler, request, exc)
            return _coerce_response(result)
        except Exception:
            logger.exception("Error handler %s failed while handling %s", handler, exc)
            return None

    async def _handle_error(
        self, request: Request, exc: Exception
    ) -> Response | StreamingResponse:
        """Resolve an error into a response, consulting registered handlers."""
        if isinstance(exc, HTTPException):
            handler = self._error_handlers.get(exc.status_code)
            if handler is not None:
                response = await self._run_error_handler(request, handler, exc)
                if response is not None:
                    return response

        for exc_cls, handler in self._error_handlers.items():
            if isinstance(exc_cls, type) and isinstance(exc, exc_cls):
                response = await self._run_error_handler(request, handler, exc)
                if response is not None:
                    return response

        if isinstance(exc, HTTPException):
            return JSONResponse(
                {"error": exc.message},
                status_code=exc.status_code,
            )
        return JSONResponse(
            {"error": "Internal Server Error"},
            status_code=500,
        )

    # ── Connection Handling ───────────────────────────────────────────────

    @staticmethod
    async def _send_error_and_close(
        writer: asyncio.StreamWriter,
        response: Response | JSONResponse,
    ) -> None:
        """Write an error response to the client and close the connection.

        Swallows broken-pipe / reset errors that occur when the client has
        already disconnected.
        """
        try:
            await response._write(writer)
        except (BrokenPipeError, ConnectionResetError):
            pass
        writer.close()

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Serve requests on one connection until it closes.

        A kept-alive connection is read, dispatched and written in a loop;
        the loop ends on ``Connection: close``, an HTTP/1.0 request that did
        not ask to stay, a streaming response, the request limit, an idle
        timeout, or a client disconnect.
        """
        peer = writer.get_extra_info("peername")
        client_addr = (peer[0], peer[1]) if peer else ("unknown", 0)
        served = 0
        task = asyncio.current_task()
        if task is not None:
            self._tasks.add(task)

        try:
            while True:
                idle_timeout = (
                    self.keep_alive_timeout
                    if served and self.keep_alive and self.keep_alive_timeout > 0
                    else None
                )
                # A connection is idle until a request line arrives: a probe
                # that says nothing, a half-sent line, and a socket still
                # sitting in the accept queue must not hold a shutdown for
                # the whole grace period (#220).  Once the request line is
                # in it leaves the set (``on_request_start``), so a body
                # that is still uploading is drained, not cut off.
                self._idle_connections.add(writer)
                read_state = _ReadState()
                try:
                    request_parts = await _read_request(
                        reader,
                        self.read_timeout,
                        self.max_body_size,
                        idle_timeout=idle_timeout,
                        on_request_start=lambda: self._idle_connections.discard(writer),
                        state=read_state,
                    )
                except _ClientDisconnected:
                    break
                except Exception as exc:
                    await self._reject_read(exc, writer, client_addr, read_state)
                    break
                finally:
                    self._idle_connections.discard(writer)

                method, path, qs, headers, body, version = request_parts
                keep_alive = self._keep_alive_requested(version, headers, served + 1)
                served += 1
                try:
                    if not await self._serve_one(
                        method,
                        path,
                        qs,
                        headers,
                        body,
                        client_addr,
                        writer,
                        keep_alive,
                        reader,
                    ):
                        break
                except Exception:
                    logger.exception("Error writing response to %s", client_addr)
                    break
        finally:
            if task is not None:
                self._tasks.discard(task)
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    async def _reject_read(
        self,
        exc: Exception,
        writer: asyncio.StreamWriter,
        client_addr: tuple,
        state: _ReadState | None = None,
    ) -> None:
        """Answer a failed request read, or close quietly on an idle timeout.

        A timeout before the request line is an idle kept-alive connection:
        the client simply stopped talking, so the connection closes without a
        response. A timeout after the request line means the client started a
        request and did not finish it, which gets a ``408``.

        Every response sent here carries an ``X-Request-Id`` and is reported
        to :meth:`on_request_rejected` handlers first.

        Args:
            exc: The exception raised while reading the request.
            writer: The client connection.
            client_addr: Client address for logging.
            state: What the read had parsed when it failed.
        """
        if isinstance(exc, _IdleTimeout):
            logger.debug("Keep-alive idle timeout from %s", client_addr)
            return
        state = state or _ReadState()
        response = _read_error_response(exc)
        reason = _read_failure_reason(exc)
        request_id = _request_id_from(state.headers)
        response.headers["X-Request-Id"] = request_id

        bytes_received = state.bytes_received
        if isinstance(exc, asyncio.IncompleteReadError):
            bytes_received = len(exc.partial)
        if reason == "read_error":
            logger.warning(
                "Failed to read request from %s (id=%s)",
                client_addr,
                request_id,
                exc_info=exc,
            )
        else:
            logger.warning(
                "Rejected request from %s: %s, status %d (id=%s)",
                client_addr,
                reason,
                response.status_code,
                request_id,
            )

        await self._fire_signal(
            "on_request_rejected",
            self._on_request_rejected_handlers,
            RejectedRequest(
                client_addr=(client_addr[0], client_addr[1]),
                status_code=response.status_code,
                reason=reason,
                exception=exc,
                request_id=request_id,
                method=state.method,
                path=state.path,
                headers=(
                    None
                    if state.headers is None
                    else MappingProxyType(dict(state.headers))
                ),
                content_length=state.content_length,
                bytes_received=bytes_received,
            ),
        )
        await self._send_error_and_close(writer, response)

    @staticmethod
    def _connection_tokens(headers: dict[str, str]) -> set[str]:
        """Connection header tokens, looked up case-insensitively.

        Args:
            headers: Response or request headers.

        Returns:
            Lower-cased token set; empty when there is no ``Connection``
            header under any spelling.
        """
        tokens: set[str] = set()
        for key, value in headers.items():
            if key.lower() == "connection":
                tokens.update(
                    token.strip().lower() for token in value.split(",") if token.strip()
                )
        return tokens

    @staticmethod
    def _set_connection_header(headers: dict[str, str], value: str) -> None:
        """Set ``Connection`` under its canonical spelling, dropping variants.

        A caller that wrote ``headers={"connection": "close"}`` should get
        one header that says ``close``, not two that disagree.
        """
        for key in [key for key in headers if key.lower() == "connection"]:
            del headers[key]
        headers["Connection"] = value

    async def _serve_one(
        self,
        method: str,
        path: str,
        qs: str,
        headers: dict[str, str],
        body: bytes,
        client_addr: tuple,
        writer: asyncio.StreamWriter,
        keep_alive: bool,
        reader: asyncio.StreamReader | None = None,
    ) -> bool:
        """Dispatch and write one request.

        Args:
            method: Request method.
            path: Decoded request path.
            qs: Query string.
            headers: Lower-cased request headers.
            body: Request body.
            client_addr: Client address for logging.
            writer: The client connection.
            keep_alive: Whether the connection may be reused afterwards.
            reader: Request reader, used to detect a peer FIN while a
                streaming generator is blocked.

        Returns:
            ``True`` when the connection stays open for another request.
        """
        request = Request(
            method=method,
            path=path,
            query_string=qs,
            headers=headers,
            body=body,
            client_addr=client_addr,
            app=self,
        )
        logger.debug("%s %s from %s", method, path, client_addr)

        try:
            response = await self._dispatch(request)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            logger.debug("Connection reset by %s during dispatch", client_addr)
            return False

        # A HEAD response has no body (RFC 9110 §9.3.2), and 204/304 carry
        # none either.  Writing one would leave stray bytes in the stream and
        # poison the next request on a kept-alive connection (#210 review).
        if isinstance(response, Response) and (
            method == "HEAD" or response.status_code in (204, 304)
        ):
            if method == "HEAD":
                # Same header fields as the GET would have sent, including
                # Content-Length, but no body (RFC 9110 §9.3.2).
                response.headers.setdefault("Content-Length", str(len(response.body)))
            response.body = b""

        # A request that is already being handled still gets its response, but
        # the connection must not be reused after shutdown has begun.
        if (
            keep_alive
            and self._shutdown_event is not None
            and self._shutdown_event.is_set()
        ):
            keep_alive = False

        keep_alive = keep_alive and not isinstance(response, StreamingResponse)
        if keep_alive and "close" not in self._connection_tokens(response.headers):
            self._set_connection_header(response.headers, "keep-alive")
        else:
            keep_alive = False
            self._set_connection_header(response.headers, "close")

        await self._fire_signal(
            "on_response_started",
            self._on_response_started_handlers,
            request,
            response,
        )

        try:
            await response._write(writer, reader)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            await self._fire_signal(
                "on_client_disconnect",
                self._on_client_disconnect_handlers,
                request,
            )
            logger.debug("Connection reset by %s during response", client_addr)
            return False

        await self._fire_signal(
            "on_response_completed",
            self._on_response_completed_handlers,
            request,
            response,
        )
        is_closing = getattr(writer, "is_closing", None)
        if is_closing is not None and is_closing():
            return False
        return keep_alive

    def _keep_alive_requested(
        self,
        version: str,
        headers: dict[str, str],
        served: int,
    ) -> bool:
        """Whether this request may be followed by another on the connection.

        Args:
            version: HTTP version from the request line.
            headers: Lower-cased request headers.
            served: Number of requests served including this one.

        Returns:
            ``True`` when the connection should stay open.
        """
        if (
            not self.keep_alive
            or self.keep_alive_timeout <= 0
            or served >= self.max_requests_per_connection
        ):
            return False
        tokens = {
            token.strip().lower()
            for token in headers.get("connection", "").split(",")
            if token.strip()
        }
        if "close" in tokens:
            return False
        if version == "HTTP/1.0":
            return "keep-alive" in tokens
        if not version.startswith("HTTP/1."):
            # Unknown or malformed protocol version: do not assume the
            # client can frame a second response.
            return False
        return True

    # ── Server Lifecycle ─────────────────────────────────────────────────

    async def _run_startup_hooks(self) -> None:
        """Run all registered startup hooks in registration order.

        If any hook raises, the exception is logged and re-raised so the
        server does not start with failed initialization.
        """
        for hook in self._startup_handlers:
            try:
                result = hook()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.error(
                    "Startup hook %r failed",
                    getattr(hook, "__name__", repr(hook)),
                    exc_info=True,
                )
                raise

    async def _run_shutdown_hooks(self) -> None:
        """Run all registered shutdown hooks in reverse registration order.

        Errors are logged and suppressed so that remaining hooks still
        execute (best-effort cleanup).
        """
        for hook in reversed(self._shutdown_handlers):
            try:
                result = hook()
                if inspect.isawaitable(result):
                    await result
            except Exception:
                logger.warning(
                    "Shutdown hook %r failed",
                    getattr(hook, "__name__", repr(hook)),
                    exc_info=True,
                )

    def run(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
        *,
        socket: str | None = None,
        ssl_context: ssl.SSLContext | None = None,
        backlog: int | None = None,
        reuse_address: bool | None = None,
        reuse_port: bool | None = None,
    ) -> None:
        """Start the server (blocking).

        Args:
            host: Bind address (ignored when *socket* is set).
            port: Bind port. Use ``0`` for OS-assigned port (ignored when
                *socket* is set).
            socket: Unix domain socket path. When set, the server listens
                on a Unix socket instead of TCP. The socket file permissions
                are restricted to owner-only (``0o600``) after creation.
                Only available on Unix-like systems.
            ssl_context: An :class:`ssl.SSLContext` for TLS termination.
                When provided, the server accepts HTTPS connections.
                The caller is responsible for configuring the context
                (loading certs, setting verify mode, etc.).
            backlog: Maximum number of queued connections passed to
                :func:`asyncio.start_server`. ``None`` leaves the OS
                default (typically 128).
            reuse_address: Sets ``SO_REUSEADDR``. ``None`` lets asyncio
                decide (``True`` on non-Windows).
            reuse_port: Sets ``SO_REUSEPORT`` for multi-process load
                balancing. ``None`` lets asyncio decide (``False``).
        """
        try:
            asyncio.run(
                self._serve(
                    host,
                    port,
                    socket=socket,
                    ssl_context=ssl_context,
                    backlog=backlog,
                    reuse_address=reuse_address,
                    reuse_port=reuse_port,
                )
            )
        except KeyboardInterrupt:
            pass

    async def _serve(
        self,
        host: str,
        port: int,
        *,
        socket: str | None = None,
        ssl_context: ssl.SSLContext | None = None,
        backlog: int | None = None,
        reuse_address: bool | None = None,
        reuse_port: bool | None = None,
    ) -> None:
        """Internal async server loop."""
        self._shutdown_event = asyncio.Event()
        self._loop = asyncio.get_running_loop()
        self._socket_path: str | None = None

        try:
            # Run startup hooks before accepting connections
            await self._run_startup_hooks()

            if socket:
                server = await self._start_unix_socket(
                    socket, ssl_context=ssl_context, backlog=backlog
                )
            else:
                kwargs: dict[str, Any] = {}
                if backlog is not None:
                    kwargs["backlog"] = backlog
                if reuse_address is not None:
                    kwargs["reuse_address"] = reuse_address
                if reuse_port is not None:
                    kwargs["reuse_port"] = reuse_port
                server = await asyncio.start_server(
                    self._handle_connection,
                    host,
                    port,
                    ssl=ssl_context,
                    **kwargs,
                )
                addrs = (
                    server.sockets[0].getsockname() if server.sockets else (host, port)
                )
                self.host = addrs[0]
                self.port = addrs[1]
                scheme = "https" if ssl_context else "http"
                logger.info("Serving on %s://%s:%d", scheme, self.host, self.port)

            self._server = server

            loop = asyncio.get_running_loop()
            if sys.platform != "win32":
                for sig in (signal.SIGINT, signal.SIGTERM):
                    loop.add_signal_handler(sig, self._shutdown_event.set)

            async with server:
                await self._shutdown_event.wait()
                logger.info("Shutting down server")
                # Stop accepting first, then let in-flight requests finish.
                # On 3.12+ ``wait_closed()`` waits for connections; on
                # 3.10/3.11 it returns immediately, so the drain is what
                # keeps the handler tasks from being cancelled by
                # ``asyncio.run``.
                server.close()
                self._close_idle_connections()
                if self._socket_path:
                    self._cleanup_socket()
                await self._drain_connection_tasks()
        finally:
            # Run shutdown hooks after server stops accepting connections
            await self._run_shutdown_hooks()

    async def _start_unix_socket(
        self,
        socket_path: str,
        *,
        ssl_context: ssl.SSLContext | None = None,
        backlog: int | None = None,
    ) -> asyncio.Server:
        """Start listening on a Unix domain socket.

        Handles stale socket cleanup, permission hardening (``0o600``),
        and registers the path for shutdown cleanup.

        Args:
            socket_path: Path for the Unix domain socket file.
            ssl_context: Optional :class:`ssl.SSLContext` for TLS over
                the Unix socket.
            backlog: Maximum number of queued connections. ``None``
                leaves the OS default.

        Returns:
            The ``asyncio.Server`` instance.

        Raises:
            SystemExit: If the path exists but is not a socket, or the
                parent directory does not exist.
        """
        import stat as stat_mod

        path = os.path.realpath(socket_path)

        # Remove stale socket if present
        if os.path.exists(path):
            try:
                st = os.stat(path)
                if stat_mod.S_ISSOCK(st.st_mode):
                    os.unlink(path)
                    logger.info("Removed stale socket: %s", path)
                else:
                    logger.error("Socket path exists and is not a socket: %s", path)
                    sys.exit(1)
            except OSError as exc:
                logger.error("Cannot remove stale socket %s: %s", path, exc)
                sys.exit(1)

        # Ensure parent directory exists
        parent = os.path.dirname(path)
        if not os.path.isdir(parent):
            logger.error("Socket parent directory does not exist: %s", parent)
            sys.exit(1)

        kwargs: dict[str, Any] = {}
        if backlog is not None:
            kwargs["backlog"] = backlog
        server = await asyncio.start_unix_server(
            self._handle_connection,
            path=path,
            ssl=ssl_context,
            **kwargs,
        )

        # Restrict permissions to owner-only
        if os.path.exists(path):
            os.chmod(path, 0o600)

        self._socket_path = path
        scheme = "https+unix" if ssl_context else "unix"
        logger.info("Serving on %s:%s (mode 0600)", scheme, path)
        return server

    def _cleanup_socket(self) -> None:
        """Remove the Unix socket file on shutdown."""
        if self._socket_path and os.path.exists(self._socket_path):
            try:
                os.unlink(self._socket_path)
                logger.info("Removed socket: %s", self._socket_path)
            except OSError:
                pass

    async def _drain_connection_tasks(self) -> None:
        """Wait for connection handlers that are still serving a request.

        Idle kept-alive connections have already been closed, and a request
        that completes after shutdown started is answered with
        ``Connection: close``, so every remaining task is finishing an
        in-flight request (or about to observe the closed socket).  The wait
        is bounded by ``shutdown_grace_timeout``; anything still running is
        cancelled, because an unresponsive client must not block exit.
        """
        current = asyncio.current_task()
        tasks = [
            task for task in self._tasks if task is not current and not task.done()
        ]
        if not tasks:
            return
        _done, pending = await asyncio.wait(tasks, timeout=self.shutdown_grace_timeout)
        for task in pending:
            logger.warning(
                "Cancelling a connection handler that outlived the shutdown "
                "grace period (%.1fs)",
                self.shutdown_grace_timeout,
            )
            task.cancel()

    def _close_idle_connections(self) -> None:
        """Close connections that are waiting for another request.

        A connection is idle while it waits for a request line: a
        kept-alive one until ``keep_alive_timeout`` expires, and a
        brand-new one until ``read_timeout`` does.  Since Python 3.12,
        ``Server.wait_closed()`` waits for every open connection, so leaving
        either open would hold a graceful shutdown for the idle window — or
        for ``shutdown_grace_timeout`` in the drain.  Requests that have
        been read are not idle: their handler finishes and closes its own
        writer.
        """
        writers = list(self._idle_connections)
        self._idle_connections.clear()
        for writer in writers:
            try:
                writer.close()
            except Exception:  # pragma: no cover - already closed
                pass

    def shutdown(self) -> None:
        """Request a graceful server shutdown.

        Safe to call from any thread or from a request handler.
        """
        if self._shutdown_event is None or self._loop is None:
            return
        if self._loop.is_closed():
            # Already stopped; a repeated shutdown is a no-op rather than an
            # ``Event loop is closed`` error from ``call_soon_threadsafe``.
            return
        try:
            self._loop.call_soon_threadsafe(self._shutdown_event.set)
            # Wake idle kept-alive connections so a host that waits for them
            # (``Server.wait_closed()`` on 3.12+) is not held up by the idle
            # window.  ``_serve`` does the same for signal-driven shutdowns.
            self._loop.call_soon_threadsafe(self._close_idle_connections)
        except RuntimeError:  # pragma: no cover - loop closed mid-call
            # The loop can finish between the ``is_closed()`` check and the
            # call when shutdown races the server thread's exit.
            pass


# ── Handler Invocation ───────────────────────────────────────────────────────


async def _invoke(handler: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Invoke a handler, wrapping sync functions with ``asyncio.to_thread``."""
    if inspect.iscoroutinefunction(handler):
        return await handler(*args, **kwargs)
    return await asyncio.to_thread(handler, *args, **kwargs)
