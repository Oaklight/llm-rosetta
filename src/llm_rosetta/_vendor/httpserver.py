# /// zerodep
# version = "0.1.0"
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
import sys
from collections.abc import AsyncIterator, Callable
from email.utils import formatdate
from pathlib import Path
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


def abort(status_code: int, message: str | None = None) -> None:
    """Raise an :class:`HTTPException` with the given status code.

    Args:
        status_code: HTTP status code.
        message: Optional error message.
    """
    raise HTTPException(status_code, message)


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
        "_json",
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
        self._json: Any = _SENTINEL

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


class Response:
    """HTTP response with a fixed body.

    Args:
        body: Response body (bytes or str).
        status_code: HTTP status code.
        headers: Extra response headers.
        content_type: Shorthand for ``Content-Type`` header.
    """

    __slots__ = ("status_code", "headers", "body")

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

    async def _write(self, writer: asyncio.StreamWriter) -> None:
        """Serialize and write the full HTTP response."""
        reason = _STATUS_REASONS.get(self.status_code, "Unknown")
        self.headers.setdefault("Content-Length", str(len(self.body)))
        self.headers.setdefault("Content-Type", "text/plain; charset=utf-8")
        self.headers.setdefault("Date", _http_date())
        self.headers.setdefault("Connection", "close")

        buf = bytearray()
        buf.extend(f"HTTP/1.1 {self.status_code} {reason}\r\n".encode("latin-1"))
        for k, v in self.headers.items():
            buf.extend(f"{k}: {v}\r\n".encode("latin-1"))
        buf.extend(b"\r\n")
        buf.extend(self.body)
        writer.write(bytes(buf))
        await writer.drain()


class JSONResponse(Response):
    """Response serialized as JSON.

    Args:
        data: Python object to serialize.
        status_code: HTTP status code.
        headers: Extra response headers.
    """

    def __init__(
        self,
        data: Any,
        status_code: int = 200,
        headers: dict[str, str] | None = None,
    ):
        body = _json.dumps(data, ensure_ascii=False).encode("utf-8")
        super().__init__(
            body=body,
            status_code=status_code,
            headers=headers,
            content_type="application/json; charset=utf-8",
        )


class StreamingResponse:
    """HTTP response streamed from an async generator.

    Writes chunks using ``Transfer-Encoding: chunked`` unless
    ``content_type`` is ``text/event-stream`` (SSE), in which case raw
    bytes are flushed directly for maximum compatibility with SSE clients.

    Args:
        generator: Async iterator yielding ``bytes`` or ``str`` chunks.
        status_code: HTTP status code.
        headers: Extra response headers.
        content_type: MIME type (default ``application/octet-stream``).
    """

    __slots__ = ("_generator", "status_code", "headers", "content_type")

    def __init__(
        self,
        generator: AsyncIterator[bytes | str],
        status_code: int = 200,
        headers: dict[str, str] | None = None,
        content_type: str = "application/octet-stream",
    ):
        self._generator = generator
        self.status_code = status_code
        self.headers: dict[str, str] = headers.copy() if headers else {}
        self.content_type = content_type

    async def _write(self, writer: asyncio.StreamWriter) -> None:
        """Write status line, headers, then stream the body."""
        reason = _STATUS_REASONS.get(self.status_code, "Unknown")
        is_sse = self.content_type.startswith("text/event-stream")

        self.headers["Content-Type"] = self.content_type
        if not is_sse:
            self.headers.setdefault("Transfer-Encoding", "chunked")
        else:
            self.headers.setdefault("Cache-Control", "no-cache")
        self.headers.setdefault("Date", _http_date())
        self.headers.setdefault("Connection", "close")

        buf = bytearray()
        buf.extend(f"HTTP/1.1 {self.status_code} {reason}\r\n".encode("latin-1"))
        for k, v in self.headers.items():
            buf.extend(f"{k}: {v}\r\n".encode("latin-1"))
        buf.extend(b"\r\n")
        writer.write(bytes(buf))
        await writer.drain()

        try:
            async for chunk in self._generator:
                if isinstance(chunk, str):
                    chunk = chunk.encode("utf-8")
                if is_sse:
                    writer.write(chunk)
                else:
                    writer.write(f"{len(chunk):x}\r\n".encode("latin-1"))
                    writer.write(chunk)
                    writer.write(b"\r\n")
                await writer.drain()
            if not is_sse:
                writer.write(b"0\r\n\r\n")
                await writer.drain()
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            logger.debug("Client disconnected during streaming")
        finally:
            aclose = getattr(self._generator, "aclose", None)
            if aclose is not None:
                await aclose()


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


# ── HTTP Parsing ─────────────────────────────────────────────────────────────


async def _read_request(
    reader: asyncio.StreamReader,
    timeout: float,
    max_body_size: int,
) -> tuple[str, str, str, dict[str, str], bytes]:
    """Read and parse an HTTP request from a stream.

    Returns:
        Tuple of (method, path, query_string, headers, body).

    Raises:
        _BadRequest: If the request is malformed.
        asyncio.TimeoutError: If reading times out.
    """
    # -- Request line --
    raw_line = await asyncio.wait_for(reader.readline(), timeout=timeout)
    if not raw_line:
        raise _BadRequest("Empty request")
    request_line = raw_line.decode("latin-1").rstrip("\r\n")
    parts = request_line.split(" ", 2)
    if len(parts) != 3:
        raise _BadRequest(f"Malformed request line: {request_line!r}")
    method, raw_url, _version = parts

    # -- URL --
    parsed = urlparse(raw_url)
    path = unquote(parsed.path)
    query_string = parsed.query

    # -- Headers --
    headers: dict[str, str] = {}
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
        length = int(cl)
        if length > max_body_size:
            raise HTTPException(413, f"Request body too large ({length} bytes)")
        if length > 0:
            body = await asyncio.wait_for(reader.readexactly(length), timeout=timeout)
    elif headers.get("transfer-encoding", "").lower() == "chunked":
        body = await _read_chunked_body(reader, timeout, max_body_size)

    return method.upper(), path, query_string, headers, body


async def _read_chunked_body(
    reader: asyncio.StreamReader,
    timeout: float,
    max_body_size: int,
) -> bytes:
    """Read a chunked transfer-encoded body."""
    parts: list[bytes] = []
    total = 0
    while True:
        size_line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        size_str = size_line.decode("latin-1").split(";")[0].strip()
        if not size_str:
            break
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
    return b"".join(parts)


# ── Utilities ────────────────────────────────────────────────────────────────


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

    Args:
        max_body_size: Maximum request body size in bytes.
        read_timeout: Timeout for reading a single request (seconds).

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
    ):
        self._routes: list[_Route] = []
        self._static_routes: list[tuple[str, str]] = []
        self._before_request_handlers: list[Callable[..., Any]] = []
        self._after_request_handlers: list[Callable[..., Any]] = []
        self._error_handlers: dict[int | type, Callable[..., Any]] = {}
        self._server: asyncio.Server | None = None
        self._shutdown_event: asyncio.Event | None = None
        self.max_body_size = max_body_size
        self.read_timeout = read_timeout
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

    # ── Request Dispatch ─────────────────────────────────────────────────

    def _match_route(
        self, request: Request
    ) -> tuple[_Route | None, re.Match[str] | None, bool]:
        """Find the first route matching the request path and method.

        Returns:
            (matched_route, regex_match, path_existed).  *path_existed* is
            True when a route matched the path but not the HTTP method.
        """
        path_existed = False
        for route in self._routes:
            m = route.pattern.match(request.path)
            if m is None:
                continue
            path_existed = True
            if "*" not in route.methods and request.method not in route.methods:
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

    async def _run_after_hooks(
        self, request: Request, response: Response | StreamingResponse
    ) -> Response | StreamingResponse:
        """Run after-request hooks, allowing them to replace the response."""
        for hook in self._after_request_handlers:
            hook_result = await _invoke(hook, request, response)
            if hook_result is not None:
                response = _coerce_response(hook_result)
        return response

    async def _dispatch(self, request: Request) -> Response | StreamingResponse:
        """Match a request to a route and invoke the handler."""
        # -- before_request hooks --
        for hook in self._before_request_handlers:
            result = await _invoke(hook, request)
            if result is not None:
                return _coerce_response(result)

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

    async def _handle_error(
        self, request: Request, exc: Exception
    ) -> Response | StreamingResponse:
        """Resolve an error into a response, consulting registered handlers."""
        if isinstance(exc, HTTPException):
            handler = self._error_handlers.get(exc.status_code)
            if handler is not None:
                result = await _invoke(handler, request, exc)
                return _coerce_response(result)

        for exc_cls, handler in self._error_handlers.items():
            if isinstance(exc_cls, type) and isinstance(exc, exc_cls):
                result = await _invoke(handler, request, exc)
                return _coerce_response(result)

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

    async def _handle_connection(
        self,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Handle a single client connection."""
        peer = writer.get_extra_info("peername")
        client_addr = (peer[0], peer[1]) if peer else ("unknown", 0)

        try:
            method, path, qs, headers, body = await _read_request(
                reader, self.read_timeout, self.max_body_size
            )
        except _BadRequest as exc:
            logger.debug("Bad request from %s: %s", client_addr, exc)
            response = Response(
                body=str(exc),
                status_code=400,
                content_type="text/plain; charset=utf-8",
            )
            try:
                await response._write(writer)
            except (BrokenPipeError, ConnectionResetError):
                pass
            writer.close()
            return
        except HTTPException as exc:
            response = JSONResponse({"error": exc.message}, status_code=exc.status_code)
            try:
                await response._write(writer)
            except (BrokenPipeError, ConnectionResetError):
                pass
            writer.close()
            return
        except (asyncio.TimeoutError, asyncio.IncompleteReadError):
            writer.close()
            return
        except Exception:
            writer.close()
            return

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
            await response._write(writer)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            logger.debug("Connection reset by %s during response", client_addr)
        except Exception:
            logger.exception("Error writing response to %s", client_addr)
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:
                pass

    # ── Server Lifecycle ─────────────────────────────────────────────────

    def run(
        self,
        host: str = DEFAULT_HOST,
        port: int = DEFAULT_PORT,
    ) -> None:
        """Start the server (blocking).

        Args:
            host: Bind address.
            port: Bind port. Use ``0`` for OS-assigned port.
        """
        try:
            asyncio.run(self._serve(host, port))
        except KeyboardInterrupt:
            pass

    async def _serve(self, host: str, port: int) -> None:
        """Internal async server loop."""
        self._shutdown_event = asyncio.Event()

        server = await asyncio.start_server(
            self._handle_connection,
            host,
            port,
        )

        self._server = server
        addrs = server.sockets[0].getsockname() if server.sockets else (host, port)
        self.host = addrs[0]
        self.port = addrs[1]
        logger.info("Serving on %s:%d", self.host, self.port)

        loop = asyncio.get_running_loop()
        if sys.platform != "win32":
            for sig in (signal.SIGINT, signal.SIGTERM):
                loop.add_signal_handler(sig, self._shutdown_event.set)

        async with server:
            await self._shutdown_event.wait()
            logger.info("Shutting down server")

    def shutdown(self) -> None:
        """Request a graceful server shutdown.

        Safe to call from a request handler.
        """
        if self._shutdown_event is not None:
            self._shutdown_event.set()


# ── Handler Invocation ───────────────────────────────────────────────────────


async def _invoke(handler: Callable[..., Any], *args: Any, **kwargs: Any) -> Any:
    """Invoke a handler, wrapping sync functions with ``asyncio.to_thread``."""
    if inspect.iscoroutinefunction(handler):
        return await handler(*args, **kwargs)
    return await asyncio.to_thread(handler, *args, **kwargs)
