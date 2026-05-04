# /// zerodep
# version = "0.3.1"
# deps = []
# tier = "subsystem"
# category = "network"
# note = "Install/update via `zerodep add httpclient`"
# ///

"""Zero-dependency sync + async HTTP REST client.

Part of zerodep: https://github.com/Oaklight/zerodep
Copyright (c) 2026 Peng Ding. MIT License.

Sync (http.client) and async (asyncio streams) HTTP/1.1 client
for REST API consumption. Thread-safe by design.

Sync usage::

    response = get("https://httpbin.org/get")
    response.json()

Async usage::

    response = await async_get("https://httpbin.org/get")
    response.json()

Session usage::

    with Client() as client:
        r = client.get("https://httpbin.org/get")

    async with AsyncClient() as client:
        r = await client.get("https://httpbin.org/get")
"""

# ── Imports ──

from __future__ import annotations

import asyncio
import base64
import hashlib
import http.client
import json as _json
import logging
import os
import ssl
import threading
import time
import warnings
import zlib
from collections.abc import AsyncIterator, Iterator
from typing import IO, Any
from urllib.parse import quote, urlencode, urlparse

__all__ = [
    # Constants
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_REDIRECTS",
    "DEFAULT_USER_AGENT",
    "DEFAULT_POOL_SIZE",
    "DEFAULT_POOL_IDLE_TIMEOUT",
    # Exceptions
    "HttpClientError",
    "HTTPError",
    "TooManyRedirects",
    "HttpConnectionError",
    "HttpTimeoutError",
    # Response classes
    "Response",
    "StreamingResponse",
    # Auth
    "Auth",
    "BasicAuth",
    "DigestAuth",
    # Sync convenience functions
    "get",
    "post",
    "put",
    "patch",
    "delete",
    "head",
    "options",
    # Async convenience functions
    "async_get",
    "async_post",
    "async_put",
    "async_patch",
    "async_delete",
    "async_head",
    "async_options",
    # Client classes
    "Client",
    "AsyncClient",
]

# ── Constants / Defaults ──

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT = 30.0
DEFAULT_MAX_REDIRECTS = 10
DEFAULT_USER_AGENT = "zerodep-http/0.1"
DEFAULT_POOL_SIZE = 10
DEFAULT_POOL_IDLE_TIMEOUT = 60.0


# ── Exceptions ──


class HttpClientError(Exception):
    """Base exception for all httpclient operations."""


class HTTPError(HttpClientError):
    """Raised on non-2xx status when raise_for_status() is called."""

    def __init__(self, status_code: int, body: str, url: str) -> None:
        self.status_code = status_code
        self.body = body
        self.url = url
        super().__init__(f"HTTP {status_code} for {url}")


class TooManyRedirects(HTTPError):
    """Raised when redirect limit is exceeded."""

    def __init__(self, url: str, max_redirects: int) -> None:
        super().__init__(0, "", url)
        self.max_redirects = max_redirects
        Exception.__init__(self, f"Too many redirects (>{max_redirects}) for {url}")


class HttpConnectionError(HttpClientError):
    """Raised on connection failures.

    Attributes:
        host: Remote hostname that the connection targeted.
        port: Remote port number.
        message: Human-readable error description.
    """

    def __init__(self, message: str, *, host: str = "", port: int = 0) -> None:
        self.host = host
        self.port = port
        self.message = message
        super().__init__(message)


class HttpTimeoutError(HttpClientError):
    """Raised on request timeout.

    Attributes:
        url: The URL that timed out.
        timeout: The timeout value in seconds that was exceeded.
        message: Human-readable error description.
    """

    def __init__(self, message: str, *, url: str = "", timeout: float = 0.0) -> None:
        self.url = url
        self.timeout = timeout
        self.message = message
        super().__init__(message)


# Backward-compatible aliases (deprecated: prefer HttpConnectionError/HttpTimeoutError)
ConnectionError = HttpConnectionError  # noqa: A001
TimeoutError = HttpTimeoutError  # noqa: A001


# ── Data Models (Response) ──


class Response:
    """HTTP response object.

    Attributes:
        status_code: HTTP status code.
        headers: Response headers as dict (last value wins for duplicates).
        content: Raw response body as bytes.
        url: Final URL after redirects.
    """

    __slots__ = ("status_code", "headers", "content", "url", "_text", "_json")

    def __init__(
        self,
        status_code: int,
        headers: dict[str, str],
        content: bytes,
        url: str,
    ) -> None:
        self.status_code = status_code
        self.headers = headers
        self.content = content
        self.url = url
        self._text: str | None = None
        self._json: Any = None

    @property
    def text(self) -> str:
        """Decode response body as text."""
        if self._text is None:
            encoding = self._guess_encoding()
            self._text = self.content.decode(encoding, errors="replace")
        return self._text

    def json(self) -> Any:
        """Parse response body as JSON."""
        if self._json is None:
            self._json = _json.loads(self.content)
        return self._json

    @property
    def ok(self) -> bool:
        """True if status_code is 2xx."""
        return 200 <= self.status_code < 300

    def raise_for_status(self) -> None:
        """Raise HTTPError if status is not 2xx."""
        if not self.ok:
            raise HTTPError(self.status_code, self.text, self.url)

    def _guess_encoding(self) -> str:
        return _guess_encoding_from_headers(self.headers)

    # ── Context managers (no-op, body is already fully read) ──

    def __enter__(self) -> Response:
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    async def __aenter__(self) -> Response:
        return self

    async def __aexit__(self, *args: Any) -> None:
        self.close()

    def close(self) -> None:
        """No-op close for a fully-read response."""

    async def aclose(self) -> None:
        """No-op async close for a fully-read response."""

    def __repr__(self) -> str:
        return f"<Response [{self.status_code}]>"


def _guess_encoding_from_headers(headers: dict[str, str]) -> str:
    """Extract charset from Content-Type header, default utf-8."""
    ct = headers.get("content-type", "")
    for part in ct.split(";"):
        part = part.strip()
        if part.startswith("charset="):
            return part[8:].strip().strip('"')
    return "utf-8"


# ── Auth (BasicAuth, DigestAuth) ──


class Auth:
    """Base class for HTTP authentication."""

    def auth_headers(self, method: str, url: str) -> dict[str, str]:
        """Return authorization headers.

        Args:
            method: HTTP method.
            url: Request URL.

        Returns:
            Dict of headers to add to the request.
        """
        raise NotImplementedError


class BasicAuth(Auth):
    """HTTP Basic authentication."""

    def __init__(self, username: str, password: str) -> None:
        self._username = username
        self._password = password

    def auth_headers(self, method: str, url: str) -> dict[str, str]:
        """Return Basic Authorization header."""
        credentials = f"{self._username}:{self._password}".encode()
        return {"Authorization": "Basic " + base64.b64encode(credentials).decode()}


class DigestAuth(Auth):
    """HTTP Digest authentication."""

    def __init__(self, username: str, password: str) -> None:
        self._username = username
        self._password = password
        self._nc = 0

    def auth_headers(self, method: str, url: str) -> dict[str, str]:
        """Not usable without a server challenge."""
        raise NotImplementedError("DigestAuth requires a server challenge")

    def auth_headers_from_challenge(
        self, method: str, path: str, challenge: str
    ) -> dict[str, str]:
        """Compute Digest auth headers from a WWW-Authenticate challenge.

        Args:
            method: HTTP method.
            path: Request path (URI).
            challenge: The WWW-Authenticate header value.

        Returns:
            Dict with the Authorization header.
        """
        params = _parse_digest_challenge(challenge)
        realm = params.get("realm", "")
        nonce = params.get("nonce", "")
        qop = params.get("qop", "")
        opaque = params.get("opaque", "")
        algorithm = params.get("algorithm", "MD5").upper()

        self._nc += 1
        nc_hex = f"{self._nc:08x}"
        cnonce = os.urandom(16).hex()

        if algorithm == "SHA-256":
            hash_fn = hashlib.sha256
        else:
            hash_fn = hashlib.md5

        ha1 = hash_fn(f"{self._username}:{realm}:{self._password}".encode()).hexdigest()
        ha2 = hash_fn(f"{method}:{path}".encode()).hexdigest()

        if qop == "auth":
            response = hash_fn(
                f"{ha1}:{nonce}:{nc_hex}:{cnonce}:{qop}:{ha2}".encode()
            ).hexdigest()
        else:
            response = hash_fn(f"{ha1}:{nonce}:{ha2}".encode()).hexdigest()

        header = (
            f'Digest username="{self._username}", realm="{realm}", '
            f'nonce="{nonce}", uri="{path}", response="{response}"'
        )
        if qop:
            header += f', qop={qop}, nc={nc_hex}, cnonce="{cnonce}"'
        if opaque:
            header += f', opaque="{opaque}"'
        header += f", algorithm={algorithm}"

        return {"Authorization": header}


def _normalize_auth(
    auth: tuple[str, str] | Auth | None,
) -> Auth | None:
    """Convert auth parameter to an Auth instance.

    Args:
        auth: A (username, password) tuple, an Auth subclass, or None.

    Returns:
        An Auth instance or None.
    """
    if auth is None:
        return None
    if isinstance(auth, tuple):
        return BasicAuth(str(auth[0]), str(auth[1]))
    return auth


def _parse_digest_challenge(header_value: str) -> dict[str, str]:
    """Parse a Digest WWW-Authenticate challenge into a dict.

    Args:
        header_value: The full WWW-Authenticate header value.

    Returns:
        Dict of challenge parameters.
    """
    if header_value.lower().startswith("digest "):
        header_value = header_value[7:]
    result: dict[str, str] = {}
    import re

    for match in re.finditer(r'(\w+)=(?:"([^"]*)"|([\w\-]+))', header_value):
        key = match.group(1)
        value = match.group(2) if match.group(2) is not None else match.group(3)
        result[key] = value
    return result


# ── Compression / Encoding helpers ──


def _decompress_body(body: bytes, encoding: str) -> bytes:
    """Decompress a response body based on Content-Encoding.

    Args:
        body: The raw response body bytes.
        encoding: The Content-Encoding value.

    Returns:
        Decompressed bytes, or original body if encoding is unsupported.
    """
    if encoding in ("gzip", "x-gzip"):
        return zlib.decompress(body, 16 + zlib.MAX_WBITS)
    if encoding == "deflate":
        try:
            return zlib.decompress(body, -zlib.MAX_WBITS)
        except zlib.error:
            return zlib.decompress(body)
    return body


def _make_decompressor(encoding: str) -> zlib._Decompress | None:
    """Create a streaming decompressor for the given encoding.

    Args:
        encoding: The Content-Encoding value.

    Returns:
        A zlib.decompressobj or None if encoding is unsupported.
    """
    if encoding in ("gzip", "x-gzip"):
        return zlib.decompressobj(16 + zlib.MAX_WBITS)
    if encoding == "deflate":
        return zlib.decompressobj(-zlib.MAX_WBITS)
    return None


# ── Streaming Response (state machine) ──


class StreamingResponse:
    """HTTP streaming response -- holds the connection open.

    Use as a context manager to ensure cleanup::

        with get(url, stream=True) as r:
            for chunk in r.iter_bytes():
                process(chunk)

        async with await async_get(url, stream=True) as r:
            async for line in r.aiter_lines():
                handle(line)
    """

    __slots__ = (
        "status_code",
        "headers",
        "url",
        "_encoding",
        "_decompressor",
        "_sync_resp",
        "_sync_conn",
        "_async_reader",
        "_async_writer",
        "_async_timeout",
        "_is_chunked",
        "_content_length",
        "_bytes_remaining",
        "_closed",
    )

    status_code: int
    headers: dict[str, str]
    url: str
    _encoding: str
    _decompressor: zlib._Decompress | None
    _sync_resp: http.client.HTTPResponse | None
    _sync_conn: http.client.HTTPConnection | None
    _async_reader: asyncio.StreamReader | None
    _async_writer: asyncio.StreamWriter | None
    _async_timeout: float | None
    _is_chunked: bool
    _content_length: int | None
    _bytes_remaining: int | None
    _closed: bool

    def __init__(self) -> None:
        raise TypeError("Use _from_sync() or _from_async()")

    @classmethod
    def _from_sync(
        cls,
        status_code: int,
        headers: dict[str, str],
        url: str,
        resp: http.client.HTTPResponse,
        conn: http.client.HTTPConnection,
        content_encoding: str = "",
    ) -> "StreamingResponse":
        obj = object.__new__(cls)
        obj.status_code = status_code
        obj.headers = headers
        obj.url = url
        obj._encoding = _guess_encoding_from_headers(headers)
        obj._decompressor = (
            _make_decompressor(content_encoding) if content_encoding else None
        )
        obj._sync_resp = resp
        obj._sync_conn = conn
        obj._async_reader = None
        obj._async_writer = None
        obj._async_timeout = None
        obj._is_chunked = False
        obj._content_length = None
        obj._bytes_remaining = None
        obj._closed = False
        return obj

    @classmethod
    def _from_async(
        cls,
        status_code: int,
        headers: dict[str, str],
        url: str,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
        is_chunked: bool,
        content_length: int | None,
        timeout: float,
        content_encoding: str = "",
    ) -> "StreamingResponse":
        obj = object.__new__(cls)
        obj.status_code = status_code
        obj.headers = headers
        obj.url = url
        obj._encoding = _guess_encoding_from_headers(headers)
        obj._decompressor = (
            _make_decompressor(content_encoding) if content_encoding else None
        )
        obj._sync_resp = None
        obj._sync_conn = None
        obj._async_reader = reader
        obj._async_writer = writer
        obj._async_timeout = timeout
        obj._is_chunked = is_chunked
        obj._content_length = content_length
        obj._bytes_remaining = content_length
        obj._closed = False
        return obj

    @property
    def ok(self) -> bool:
        """True if status_code is 2xx."""
        return 200 <= self.status_code < 300

    def raise_for_status(self) -> None:
        """Raise HTTPError if status is not 2xx."""
        if not self.ok:
            raise HTTPError(self.status_code, "", self.url)

    # ── Sync iteration ──

    def iter_bytes(self, chunk_size: int = 4096) -> Iterator[bytes]:
        """Yield response body in chunks."""
        if self._sync_resp is None:
            raise RuntimeError("iter_bytes() on async response")
        try:
            while True:
                chunk = self._sync_resp.read(chunk_size)
                if not chunk:
                    break
                if self._decompressor:
                    chunk = self._decompressor.decompress(chunk)
                yield chunk
            if self._decompressor:
                remaining = self._decompressor.flush()
                if remaining:
                    yield remaining
        except (OSError, http.client.HTTPException) as exc:
            raise HttpConnectionError(str(exc)) from exc

    def iter_lines(self) -> Iterator[str]:
        """Yield response body line by line (decoded)."""
        if self._sync_resp is None:
            raise RuntimeError("iter_lines() on async response")
        try:
            while True:
                line = self._sync_resp.readline()
                if not line:
                    break
                yield line.decode(self._encoding, errors="replace").rstrip("\r\n")
        except (OSError, http.client.HTTPException) as exc:
            raise HttpConnectionError(str(exc)) from exc

    def read(self) -> bytes:
        """Consume entire stream into bytes."""
        return b"".join(self.iter_bytes())

    # ── Async iteration ──

    async def aiter_bytes(self, chunk_size: int = 4096) -> AsyncIterator[bytes]:
        """Async yield response body in chunks."""
        if self._async_reader is None:
            raise RuntimeError("aiter_bytes() on sync response")
        try:
            raw_iter = self._select_raw_iterator(chunk_size)
            async for chunk in raw_iter:
                if self._decompressor:
                    chunk = self._decompressor.decompress(chunk)
                yield chunk
            if self._decompressor:
                remaining = self._decompressor.flush()
                if remaining:
                    yield remaining
        except asyncio.TimeoutError:
            raise HttpTimeoutError(
                f"Streaming read timed out for {self.url}",
                url=self.url,
                timeout=self._async_timeout or 0.0,
            )
        except OSError as exc:
            raise HttpConnectionError(str(exc)) from exc

    async def _select_raw_iterator(self, chunk_size: int) -> AsyncIterator[bytes]:
        """Select and yield from the appropriate raw byte iterator."""
        if self._is_chunked:
            async for chunk in self._aiter_chunked():
                yield chunk
        elif self._bytes_remaining is not None:
            async for chunk in self._aiter_fixed_length(chunk_size):
                yield chunk
        else:
            async for chunk in self._aiter_until_eof(chunk_size):
                yield chunk

    async def _aiter_fixed_length(self, chunk_size: int) -> AsyncIterator[bytes]:
        """Read a known-length response body in chunks."""
        assert self._async_reader is not None
        while self._bytes_remaining is not None and self._bytes_remaining > 0:
            to_read = min(chunk_size, self._bytes_remaining)
            data = await asyncio.wait_for(
                self._async_reader.read(to_read),
                timeout=self._async_timeout,
            )
            if not data:
                break
            self._bytes_remaining -= len(data)
            yield data

    async def _aiter_until_eof(self, chunk_size: int) -> AsyncIterator[bytes]:
        """Read response body until EOF in chunks."""
        assert self._async_reader is not None
        while True:
            data = await asyncio.wait_for(
                self._async_reader.read(chunk_size),
                timeout=self._async_timeout,
            )
            if not data:
                break
            yield data

    async def _aiter_chunked(self) -> AsyncIterator[bytes]:
        """Decode chunked transfer encoding from async reader."""
        assert self._async_reader is not None  # guaranteed by aiter_bytes guard
        reader = self._async_reader
        timeout = self._async_timeout
        while True:
            size_line = await asyncio.wait_for(reader.readline(), timeout=timeout)
            size_str = size_line.decode("latin-1").split(";")[0].strip()
            if not size_str:
                break
            chunk_size = int(size_str, 16)
            if chunk_size == 0:
                await asyncio.wait_for(
                    reader.readline(), timeout=timeout
                )  # trailing \r\n
                break
            data = await asyncio.wait_for(
                reader.readexactly(chunk_size), timeout=timeout
            )
            await asyncio.wait_for(reader.readline(), timeout=timeout)  # trailing \r\n
            yield data

    async def aiter_lines(self) -> AsyncIterator[str]:
        """Async yield response body line by line (decoded)."""
        buf = ""
        async for chunk in self.aiter_bytes():
            buf += chunk.decode(self._encoding, errors="replace")
            while "\n" in buf:
                line, buf = buf.split("\n", 1)
                yield line.rstrip("\r")
        if buf:
            yield buf.rstrip("\r")

    async def aread(self) -> bytes:
        """Async consume entire stream into bytes."""
        parts = []
        async for chunk in self.aiter_bytes():
            parts.append(chunk)
        return b"".join(parts)

    # ── Context managers ──

    def __enter__(self) -> "StreamingResponse":
        return self

    def __exit__(self, *args: Any) -> None:
        self.close()

    async def __aenter__(self) -> "StreamingResponse":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.aclose()

    def close(self) -> None:
        """Close the underlying sync connection."""
        if self._closed:
            return
        self._closed = True
        # Tier 2: best-effort observable -- active streaming resource
        if self._sync_resp is not None:
            try:
                self._sync_resp.close()
            except Exception:
                logger.debug(
                    "failed to close sync response for %s",
                    self.url,
                    exc_info=True,
                )
        if self._sync_conn is not None:
            try:
                self._sync_conn.close()
            except Exception:
                logger.debug(
                    "failed to close sync connection for %s",
                    self.url,
                    exc_info=True,
                )

    async def aclose(self) -> None:
        """Close the underlying async connection."""
        if self._closed:
            return
        self._closed = True
        # Tier 2: best-effort observable -- active streaming resource
        if self._async_writer is not None:
            try:
                self._async_writer.close()
                await self._async_writer.wait_closed()
            except Exception:
                logger.debug(
                    "failed to close async writer for %s",
                    self.url,
                    exc_info=True,
                )

    def __del__(self) -> None:
        if not self._closed:
            warnings.warn(
                f"Unclosed StreamingResponse for {self.url}",
                ResourceWarning,
                stacklevel=2,
            )
            self.close()

    def __repr__(self) -> str:
        return f"<StreamingResponse [{self.status_code}]>"


# ── Connection Pools (Sync + Async) ──


class _SyncConnectionPool:
    """Thread-safe connection pool for sync HTTP connections.

    Pool lifecycle rules:
        - A connection CAN be returned to the pool when a non-streaming
          request completes successfully AND the server did not send a
          ``Connection: close`` header.
        - A connection MUST be discarded (not returned) when:
          (a) an error occurred during the request/response cycle,
          (b) the server sent ``Connection: close``,
          (c) the request used a proxy (proxy connections are not pooled),
          (d) streaming mode is active (the connection is owned by
              StreamingResponse until it is closed/consumed).
        - Streaming responses bypass the pool entirely: the connection is
          handed to StreamingResponse, which closes it on ``close()`` or
          ``__del__``.  The connection is never returned to the pool.
    """

    def __init__(self, pool_size: int = DEFAULT_POOL_SIZE) -> None:
        self._pool: dict[
            tuple[str, int, bool],
            list[tuple[http.client.HTTPConnection, float]],
        ] = {}
        self._pool_size = pool_size
        self._lock = threading.Lock()

    def acquire(
        self,
        host: str,
        port: int,
        is_https: bool,
        timeout: float,
        verify: bool,
    ) -> http.client.HTTPConnection | None:
        """Acquire a connection from the pool if available.

        Args:
            host: Target hostname.
            port: Target port.
            is_https: Whether the connection uses TLS.
            timeout: Connection timeout.
            verify: Whether to verify TLS certificates.

        Returns:
            A reusable connection or None.
        """
        key = (host, port, is_https)
        now = time.monotonic()
        with self._lock:
            conns = self._pool.get(key, [])
            while conns:
                conn, timestamp = conns.pop()
                if now - timestamp > DEFAULT_POOL_IDLE_TIMEOUT:
                    # Tier 3: best-effort silent -- stale connection eviction
                    try:
                        conn.close()
                    except Exception:
                        pass
                    continue
                if conn.sock is not None and conn.sock.fileno() != -1:
                    return conn
                # Tier 3: best-effort silent -- dead connection eviction
                try:
                    conn.close()
                except Exception:
                    pass
        return None

    def release(
        self,
        host: str,
        port: int,
        is_https: bool,
        conn: http.client.HTTPConnection,
    ) -> None:
        """Return a connection to the pool.

        Args:
            host: Target hostname.
            port: Target port.
            is_https: Whether the connection uses TLS.
            conn: The connection to return.
        """
        key = (host, port, is_https)
        with self._lock:
            conns = self._pool.setdefault(key, [])
            if len(conns) < self._pool_size:
                conns.append((conn, time.monotonic()))
            else:
                # Tier 3: best-effort silent -- pool overflow discard
                try:
                    conn.close()
                except Exception:
                    pass

    def close_all(self) -> None:
        """Close all pooled connections."""
        with self._lock:
            for conns in self._pool.values():
                for conn, _ in conns:
                    # Tier 3: best-effort silent -- bulk shutdown
                    try:
                        conn.close()
                    except Exception:
                        pass
            self._pool.clear()


class _AsyncConnectionPool:
    """Async connection pool for async HTTP connections.

    Pool lifecycle rules:
        - A connection CAN be returned to the pool when a non-streaming
          request completes successfully AND the server did not send a
          ``Connection: close`` header.
        - A connection MUST be discarded (not returned) when:
          (a) an error occurred during the request/response cycle,
          (b) the server sent ``Connection: close``,
          (c) the request used a proxy (proxy connections are not pooled),
          (d) streaming mode is active (the connection is owned by
              StreamingResponse until it is closed/consumed).
        - Streaming responses bypass the pool entirely: the reader/writer
          pair is handed to StreamingResponse, which closes it on
          ``aclose()`` or ``__del__``.  The pair is never returned to
          the pool.
    """

    def __init__(self, pool_size: int = DEFAULT_POOL_SIZE) -> None:
        self._pool: dict[
            tuple[str, int, bool],
            list[
                tuple[
                    asyncio.StreamReader,
                    asyncio.StreamWriter,
                    float,
                ]
            ],
        ] = {}
        self._pool_size = pool_size
        self._lock = asyncio.Lock()

    async def acquire(
        self,
        host: str,
        port: int,
        is_https: bool,
        timeout: float,
        verify: bool,
    ) -> tuple[asyncio.StreamReader, asyncio.StreamWriter] | None:
        """Acquire a connection from the pool if available.

        Args:
            host: Target hostname.
            port: Target port.
            is_https: Whether the connection uses TLS.
            timeout: Connection timeout.
            verify: Whether to verify TLS certificates.

        Returns:
            A (reader, writer) tuple or None.
        """
        key = (host, port, is_https)
        now = time.monotonic()
        async with self._lock:
            conns = self._pool.get(key, [])
            while conns:
                reader, writer, timestamp = conns.pop()
                if now - timestamp > DEFAULT_POOL_IDLE_TIMEOUT:
                    # Tier 3: best-effort silent -- stale connection eviction
                    try:
                        writer.close()
                        await writer.wait_closed()
                    except Exception:
                        pass
                    continue
                if not reader.at_eof():
                    return reader, writer
                # Tier 3: best-effort silent -- dead connection eviction
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass
        return None

    async def release(
        self,
        host: str,
        port: int,
        is_https: bool,
        reader: asyncio.StreamReader,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Return a connection to the pool.

        Args:
            host: Target hostname.
            port: Target port.
            is_https: Whether the connection uses TLS.
            reader: The stream reader.
            writer: The stream writer.
        """
        key = (host, port, is_https)
        async with self._lock:
            conns = self._pool.setdefault(key, [])
            if len(conns) < self._pool_size:
                conns.append((reader, writer, time.monotonic()))
            else:
                # Tier 3: best-effort silent -- pool overflow discard
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass

    async def close_all(self) -> None:
        """Close all pooled connections."""
        async with self._lock:
            for conns in self._pool.values():
                for _, writer, _ in conns:
                    # Tier 3: best-effort silent -- bulk shutdown
                    try:
                        writer.close()
                        await writer.wait_closed()
                    except Exception:
                        pass
            self._pool.clear()


# ── Transport (request execution) ──

# -- Proxy helpers --


def _parse_proxy(proxy: str) -> tuple[str, int, str | None, str | None]:
    """Parse a proxy URL into components.

    Args:
        proxy: Proxy URL (e.g. "http://user:pass@host:port").

    Returns:
        Tuple of (hostname, port, username_or_None, password_or_None).
    """
    parsed = urlparse(proxy)
    hostname = parsed.hostname or ""
    port = parsed.port or 8080
    username = parsed.username or None
    password = parsed.password or None
    return hostname, port, username, password


def _proxy_auth_header(username: str, password: str) -> str:
    """Build a Proxy-Authorization Basic header value.

    Args:
        username: Proxy username.
        password: Proxy password.

    Returns:
        The header value string.
    """
    credentials = f"{username}:{password}".encode()
    return "Basic " + base64.b64encode(credentials).decode()


# -- URL helpers --


def _build_url(url: str, params: dict[str, Any] | None = None) -> str:
    """Append query parameters to URL."""
    if not params:
        return url
    sep = "&" if "?" in url else "?"
    encoded = urlencode(
        {k: v for k, v in params.items() if v is not None}, quote_via=quote
    )
    return f"{url}{sep}{encoded}"


def _parse_url(url: str) -> tuple[str, str, int, str, bool]:
    """Parse URL into (scheme, host, port, path, is_https)."""
    parsed = urlparse(url)
    is_https = parsed.scheme == "https"
    host = parsed.hostname or ""
    port = parsed.port or (443 if is_https else 80)
    path = parsed.path or "/"
    if parsed.query:
        path = f"{path}?{parsed.query}"
    return parsed.scheme, host, port, path, is_https


# -- Shared request preparation helpers --


def _prepare_request(
    method: str,
    url: str,
    headers: dict[str, str] | None,
    data: bytes | str | dict[str, str] | None,
    json_data: Any,
    files: dict[str, Any] | list[tuple[str, Any]] | None,
    params: dict[str, Any] | None,
    auth: tuple[str, str] | Auth | None,
) -> tuple[str, bytes | None, dict[str, str], Auth | None]:
    """Build URL, encode body, assemble headers, and normalize auth.

    Shared by _sync_request and _async_request (Phases 1-3).

    Returns:
        (final_url, body_bytes, request_headers, auth_object).
    """
    url = _build_url(url, params)
    body, content_type = _prepare_body(data, json_data, files)

    req_headers: dict[str, str] = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept-Encoding": "gzip, deflate",
    }
    if content_type:
        req_headers["Content-Type"] = content_type
    if body is not None:
        req_headers["Content-Length"] = str(len(body))
    req_headers.update(headers or {})

    auth_obj = _normalize_auth(auth)
    if isinstance(auth_obj, BasicAuth):
        req_headers.update(auth_obj.auth_headers(method, url))

    return url, body, req_headers, auth_obj


def _compute_redirect(
    status: int,
    method: str,
    body: bytes | None,
    req_headers: dict[str, str],
    resp_headers: dict[str, str],
    scheme: str,
    host: str,
    port: int,
    url: str,
    redirects: int,
    max_redirects: int,
) -> tuple[str, str, bytes | None]:
    """Compute redirect target and adjust method/body.

    Raises:
        TooManyRedirects: If redirect limit exceeded.

    Returns:
        (new_url, new_method, new_body).
    """
    redirects_now = redirects + 1
    if redirects_now > max_redirects:
        raise TooManyRedirects(url, max_redirects)
    location = resp_headers["location"]
    if location.startswith("/"):
        new_url = f"{scheme}://{host}:{port}{location}"
    else:
        new_url = location
    new_method = method
    new_body = body
    if status == 303 or (status in (301, 302) and method == "POST"):
        new_method = "GET"
        new_body = None
        req_headers.pop("Content-Type", None)
        req_headers.pop("Content-Length", None)
    return new_url, new_method, new_body


def _is_redirect(status: int, resp_headers: dict[str, str]) -> bool:
    """Check whether the response is a redirect with a location header."""
    return status in (301, 302, 303, 307, 308) and "location" in resp_headers


def _should_attempt_digest(
    auth_obj: Auth | None,
    status: int,
    resp_headers: dict[str, str],
    digest_attempted: bool,
) -> bool:
    """Return True if a Digest auth retry should be attempted."""
    return (
        isinstance(auth_obj, DigestAuth)
        and status == 401
        and not digest_attempted
        and resp_headers.get("www-authenticate", "").lower().startswith("digest")
    )


def _make_ssl_context(verify: bool) -> ssl.SSLContext:
    """Create an SSL context based on verification setting."""
    if verify:
        return ssl.create_default_context()
    return ssl._create_unverified_context()


# -- Sync transport helpers --


def _sync_connect_via_proxy(
    host: str,
    port: int,
    path: str,
    is_https: bool,
    timeout: float,
    verify: bool,
    proxy: str,
    req_headers: dict[str, str],
    url: str,
) -> tuple[http.client.HTTPConnection, str]:
    """Establish a sync connection through an HTTP proxy.

    For plain HTTP, connects to the proxy directly.
    For HTTPS, opens a CONNECT tunnel and wraps with TLS.

    Returns:
        (connection, request_path).
    """
    proxy_host, proxy_port, proxy_user, proxy_pass = _parse_proxy(proxy)
    if not is_https:
        conn = http.client.HTTPConnection(proxy_host, proxy_port, timeout=timeout)
        if proxy_user and proxy_pass:
            req_headers["Proxy-Authorization"] = _proxy_auth_header(
                proxy_user, proxy_pass
            )
        # For HTTP proxies, the full URL is used as the request path
        return conn, url

    # CONNECT tunnel for HTTPS through proxy
    tunnel_conn = http.client.HTTPConnection(proxy_host, proxy_port, timeout=timeout)
    connect_headers: dict[str, str] = {"Host": f"{host}:{port}"}
    if proxy_user and proxy_pass:
        connect_headers["Proxy-Authorization"] = _proxy_auth_header(
            proxy_user, proxy_pass
        )
    tunnel_conn.request("CONNECT", f"{host}:{port}", headers=connect_headers)
    tunnel_resp = tunnel_conn.getresponse()
    if tunnel_resp.status != 200:
        tunnel_conn.close()
        raise HttpConnectionError(
            f"CONNECT tunnel failed: {tunnel_resp.status}",
            host=host,
            port=port,
        )
    tunnel_resp.read()
    sock = tunnel_conn.sock
    ctx = _make_ssl_context(verify)
    wrapped = ctx.wrap_socket(sock, server_hostname=host)
    conn = http.client.HTTPSConnection(host, port, timeout=timeout, context=ctx)
    conn.sock = wrapped
    return conn, path


def _sync_acquire_connection(
    host: str,
    port: int,
    path: str,
    is_https: bool,
    timeout: float,
    verify: bool,
    proxy: str | None,
    _pool: _SyncConnectionPool | None,
    req_headers: dict[str, str],
    url: str,
) -> tuple[http.client.HTTPConnection, str]:
    """Acquire a sync HTTP connection via proxy, pool, or direct creation.

    Returns:
        (connection, request_path).
    """
    if proxy:
        return _sync_connect_via_proxy(
            host, port, path, is_https, timeout, verify, proxy, req_headers, url
        )

    if _pool:
        pooled_conn = _pool.acquire(host, port, is_https, timeout, verify)
        if pooled_conn is not None:
            req_headers["Connection"] = "keep-alive"
            return pooled_conn, path

    # Direct connection
    if is_https:
        ctx = _make_ssl_context(verify)
        conn = http.client.HTTPSConnection(host, port, timeout=timeout, context=ctx)
    else:
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
    return conn, path


def _sync_release_or_close(
    close_conn: bool,
    _pool: _SyncConnectionPool | None,
    proxy: str | None,
    stream: bool,
    resp_headers: dict[str, str],
    host: str,
    port: int,
    is_https: bool,
    conn: http.client.HTTPConnection,
) -> None:
    """Release a sync connection back to the pool or close it.

    Pool reuse decision: a connection is returned to the pool
    only when ALL of the following hold:
      - close_conn is True (not handed off to streaming)
      - a pool is active (_pool is not None)
      - the request did not use a proxy
      - the request is not streaming
      - the server did not send "Connection: close"
    Otherwise the connection is closed immediately.
    """
    if not close_conn:
        return
    if (
        _pool
        and not proxy
        and not stream
        and resp_headers.get("connection", "").lower() != "close"
    ):
        _pool.release(host, port, is_https, conn)
    else:
        conn.close()


def _build_sync_response(
    status: int,
    resp_headers: dict[str, str],
    url: str,
    resp: http.client.HTTPResponse,
    conn: http.client.HTTPConnection,
    stream: bool,
) -> tuple[Response | StreamingResponse, bool]:
    """Build a final sync response (streaming or buffered).

    Returns:
        (response, close_conn) -- close_conn is False when streaming.
    """
    content_encoding = resp_headers.get("content-encoding", "")
    if stream:
        return StreamingResponse._from_sync(
            status,
            resp_headers,
            url,
            resp,
            conn,
            content_encoding=content_encoding,
        ), False

    resp_body = resp.read()
    if content_encoding:
        resp_body = _decompress_body(resp_body, content_encoding)
    return Response(status, resp_headers, resp_body, url), True


def _wrap_sync_errors(
    exc: Exception,
    host: str,
    port: int,
    url: str,
    timeout: float,
) -> HttpClientError:
    """Translate low-level sync exceptions into httpclient exceptions."""
    if isinstance(exc, (HttpTimeoutError, HttpConnectionError, TooManyRedirects)):
        return exc
    if isinstance(exc, (OSError, http.client.HTTPException)):
        return HttpConnectionError(
            f"Connection to {host}:{port} failed: {exc}",
            host=host,
            port=port,
        )
    if "timed out" in str(exc).lower():
        msg = f"Request to {url} timed out after {timeout}s"
        return HttpTimeoutError(msg, url=url, timeout=timeout)
    return exc  # type: ignore[return-value]


# -- Sync transport --


def _sync_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: bytes | str | dict[str, str] | None = None,
    json: Any = None,
    files: dict[str, Any] | list[tuple[str, Any]] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    verify: bool = True,
    stream: bool = False,
    auth: tuple[str, str] | Auth | None = None,
    proxy: str | None = None,
    _pool: _SyncConnectionPool | None = None,
) -> Response | StreamingResponse:
    """Perform a synchronous HTTP request.

    Phase structure (mirrors _async_request):
        1-3. Request preparation (URL, body, headers, auth)
        4. Redirect loop:
           a. URL parsing + connection acquisition
           b. Send request + read response
           c. Handle redirects / digest auth / response construction
           d. Connection lifecycle (pool release or close)
    """
    url, body, req_headers, auth_obj = _prepare_request(
        method, url, headers, data, json, files, params, auth
    )

    redirects = 0
    _digest_attempted = False
    while True:
        scheme, host, port, path, is_https = _parse_url(url)
        close_conn = True
        resp_headers: dict[str, str] = {}
        try:
            conn, request_path = _sync_acquire_connection(
                host,
                port,
                path,
                is_https,
                timeout,
                verify,
                proxy,
                _pool,
                req_headers,
                url,
            )
            if _pool and not proxy:
                req_headers.setdefault("Connection", "keep-alive")

            try:
                conn.request(method, request_path, body=body, headers=req_headers)
                resp = conn.getresponse()
                resp_headers = {k.lower(): v for k, v in resp.getheaders()}
                status = resp.status

                if _is_redirect(status, resp_headers):
                    resp.read()
                    url, method, body = _compute_redirect(
                        status,
                        method,
                        body,
                        req_headers,
                        resp_headers,
                        scheme,
                        host,
                        port,
                        url,
                        redirects,
                        max_redirects,
                    )
                    redirects += 1
                    continue

                if _should_attempt_digest(
                    auth_obj, status, resp_headers, _digest_attempted
                ):
                    resp.read()
                    www_auth = resp_headers["www-authenticate"]
                    req_headers.update(
                        auth_obj.auth_headers_from_challenge(method, path, www_auth)
                    )
                    _digest_attempted = True
                    conn.close()
                    continue

                result, close_conn = _build_sync_response(
                    status,
                    resp_headers,
                    url,
                    resp,
                    conn,
                    stream,
                )
                return result
            finally:
                _sync_release_or_close(
                    close_conn,
                    _pool,
                    proxy,
                    stream,
                    resp_headers,
                    host,
                    port,
                    is_https,
                    conn,
                )
        except Exception as exc:
            wrapped = _wrap_sync_errors(exc, host, port, url, timeout)
            if wrapped is exc:
                raise
            raise wrapped from exc


# -- Async transport helpers --


async def _async_read_response_headers(
    reader: asyncio.StreamReader,
    timeout: float,
) -> tuple[int, dict[str, str]]:
    """Read HTTP status line and headers from an asyncio StreamReader.

    Does NOT consume the body -- the reader is left positioned at the
    start of the response body.

    Returns:
        (status_code, headers_dict).
    """
    # Status line: "HTTP/1.1 200 OK\r\n"
    status_line = await asyncio.wait_for(reader.readline(), timeout=timeout)
    status_str = status_line.decode("latin-1").rstrip("\r\n")
    parts = status_str.split(" ", 2)
    if len(parts) < 2:
        raise HttpConnectionError(f"Malformed status line: {status_str}")
    status_code = int(parts[1])

    # Headers until empty line
    headers: dict[str, str] = {}
    while True:
        line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        decoded = line.decode("latin-1").rstrip("\r\n")
        if not decoded:
            break
        if ":" in decoded:
            k, v = decoded.split(":", 1)
            headers[k.strip().lower()] = v.strip()

    return status_code, headers


async def _async_read_chunked_body(
    reader: asyncio.StreamReader,
    timeout: float,
) -> bytes:
    """Read a chunked transfer-encoded body from an asyncio StreamReader."""
    parts: list[bytes] = []
    while True:
        size_line = await asyncio.wait_for(reader.readline(), timeout=timeout)
        size_str = size_line.decode("latin-1").split(";")[0].strip()
        if not size_str:
            break
        chunk_size = int(size_str, 16)
        if chunk_size == 0:
            await asyncio.wait_for(reader.readline(), timeout=timeout)  # trailing \r\n
            break
        data = await asyncio.wait_for(reader.readexactly(chunk_size), timeout=timeout)
        await asyncio.wait_for(reader.readline(), timeout=timeout)  # trailing \r\n
        parts.append(data)
    return b"".join(parts)


async def _async_read_body(
    reader: asyncio.StreamReader,
    headers: dict[str, str],
    timeout: float,
) -> bytes:
    """Read the response body based on Content-Length or Transfer-Encoding.

    Falls back to reading until EOF when neither header is present.
    """
    te = headers.get("transfer-encoding", "")
    if te.lower() == "chunked":
        return await _async_read_chunked_body(reader, timeout)

    cl = headers.get("content-length")
    if cl is not None:
        length = int(cl)
        if length == 0:
            return b""
        return await asyncio.wait_for(reader.readexactly(length), timeout=timeout)

    # No Content-Length, no chunked -- read until EOF
    return await asyncio.wait_for(reader.read(), timeout=timeout)


# -- Async connection helpers --


async def _async_connect_via_proxy_plain(
    host: str,
    port: int,
    timeout: float,
    proxy: str,
    req_headers: dict[str, str],
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, str]:
    """Connect to a plain HTTP target through a proxy.

    Returns:
        (reader, writer, request_path) where request_path is the full URL.
    """
    proxy_host, proxy_port, proxy_user, proxy_pass = _parse_proxy(proxy)
    reader, writer = await asyncio.wait_for(
        asyncio.open_connection(proxy_host, proxy_port),
        timeout=timeout,
    )
    request_path = f"http://{host}:{port}/"  # will be overridden by caller
    if proxy_user and proxy_pass:
        req_headers["Proxy-Authorization"] = _proxy_auth_header(proxy_user, proxy_pass)
    return reader, writer, request_path


async def _async_connect_via_proxy_tunnel(
    host: str,
    port: int,
    timeout: float,
    verify: bool,
    proxy: str,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter]:
    """Open a CONNECT tunnel through a proxy and upgrade to TLS.

    Returns:
        (reader, writer) with TLS already established.
    """
    proxy_host, proxy_port, proxy_user, proxy_pass = _parse_proxy(proxy)
    proxy_reader, proxy_writer = await asyncio.wait_for(
        asyncio.open_connection(proxy_host, proxy_port),
        timeout=timeout,
    )
    connect_line = f"CONNECT {host}:{port} HTTP/1.1\r\n"
    connect_headers = f"Host: {host}:{port}\r\n"
    if proxy_user and proxy_pass:
        connect_headers += (
            f"Proxy-Authorization: {_proxy_auth_header(proxy_user, proxy_pass)}\r\n"
        )
    connect_headers += "\r\n"
    proxy_writer.write((connect_line + connect_headers).encode("latin-1"))
    await asyncio.wait_for(proxy_writer.drain(), timeout=timeout)
    tunnel_status, _ = await _async_read_response_headers(proxy_reader, timeout)
    if tunnel_status != 200:
        proxy_writer.close()
        # Tier 3: best-effort silent -- proxy tunnel teardown
        try:
            await proxy_writer.wait_closed()
        except Exception:
            pass
        raise HttpConnectionError(
            f"CONNECT tunnel failed: {tunnel_status}",
            host=host,
            port=port,
        )
    # Upgrade to TLS over the tunnel
    ctx = _make_ssl_context(verify)
    loop = asyncio.get_event_loop()
    transport = proxy_writer.transport
    new_transport = await loop.start_tls(
        transport, transport.get_protocol(), ctx, server_hostname=host
    )
    proxy_writer._transport = new_transport  # type: ignore[attr-defined]
    return proxy_reader, proxy_writer


async def _async_acquire_connection(
    host: str,
    port: int,
    path: str,
    is_https: bool,
    timeout: float,
    verify: bool,
    proxy: str | None,
    _pool: _AsyncConnectionPool | None,
    req_headers: dict[str, str],
    url: str,
) -> tuple[asyncio.StreamReader, asyncio.StreamWriter, str]:
    """Acquire an async connection via proxy, pool, or direct creation.

    Wraps connection errors into HttpConnectionError / HttpTimeoutError.

    Returns:
        (reader, writer, request_path).
    """
    try:
        if proxy:
            if not is_https:
                proxy_host, proxy_port, proxy_user, proxy_pass = _parse_proxy(proxy)
                reader, writer = await asyncio.wait_for(
                    asyncio.open_connection(proxy_host, proxy_port),
                    timeout=timeout,
                )
                if proxy_user and proxy_pass:
                    req_headers["Proxy-Authorization"] = _proxy_auth_header(
                        proxy_user, proxy_pass
                    )
                return reader, writer, url
            reader, writer = await _async_connect_via_proxy_tunnel(
                host, port, timeout, verify, proxy
            )
            return reader, writer, path

        if _pool:
            result = await _pool.acquire(host, port, is_https, timeout, verify)
            if result is not None:
                reader, writer = result
                req_headers["Connection"] = "keep-alive"
                return reader, writer, path

        # Direct connection
        ctx = _make_ssl_context(verify) if is_https else None
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port, ssl=ctx),
            timeout=timeout,
        )
        return reader, writer, path
    except asyncio.TimeoutError:
        msg = f"Connection to {host}:{port} timed out after {timeout}s"
        raise HttpTimeoutError(msg, url=url, timeout=timeout)
    except OSError as exc:
        raise HttpConnectionError(
            f"Connection to {host}:{port} failed: {exc}",
            host=host,
            port=port,
        ) from exc


def _build_raw_http_request(
    method: str,
    request_path: str,
    host: str,
    req_headers: dict[str, str],
    use_pool: bool,
    use_proxy: bool,
) -> bytes:
    """Construct raw HTTP/1.1 request bytes for async transport.

    Args:
        method: HTTP method.
        request_path: The request path or full URL (for proxy).
        host: Target hostname.
        req_headers: Request headers dict.
        use_pool: Whether connection pooling is active.
        use_proxy: Whether a proxy is being used.

    Returns:
        Encoded HTTP/1.1 request bytes (without body).
    """
    request_line = f"{method} {request_path} HTTP/1.1\r\n"
    header_lines = f"Host: {host}\r\n"
    for k, v in req_headers.items():
        header_lines += f"{k}: {v}\r\n"
    if not use_pool or use_proxy:
        header_lines += "Connection: close\r\n"
    header_lines += "\r\n"
    return (request_line + header_lines).encode("latin-1")


async def _async_release_or_close(
    close_writer: bool,
    _pool: _AsyncConnectionPool | None,
    proxy: str | None,
    stream: bool,
    resp_headers: dict[str, str],
    host: str,
    port: int,
    is_https: bool,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
) -> None:
    """Release an async connection back to the pool or close it.

    Pool reuse decision: a connection is returned to the pool
    only when ALL of the following hold:
      - close_writer is True (not handed off to streaming)
      - a pool is active (_pool is not None)
      - the request did not use a proxy
      - the request is not streaming
      - the server did not send "Connection: close"
    Otherwise the connection is closed immediately.
    """
    if not close_writer:
        return
    if (
        _pool
        and not proxy
        and not stream
        and resp_headers.get("connection", "").lower() != "close"
    ):
        await _pool.release(host, port, is_https, reader, writer)
    else:
        writer.close()
        # Tier 3: best-effort silent -- wait_closed on direct close
        try:
            await writer.wait_closed()
        except Exception:
            pass


async def _async_close_writer_silent(writer: asyncio.StreamWriter) -> None:
    """Close an async writer, ignoring errors on wait_closed."""
    writer.close()
    try:
        await writer.wait_closed()
    except Exception:
        pass


def _build_async_streaming_response(
    status: int,
    resp_headers: dict[str, str],
    url: str,
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    timeout: float,
) -> StreamingResponse:
    """Build an async StreamingResponse from response metadata."""
    content_encoding = resp_headers.get("content-encoding", "")
    te = resp_headers.get("transfer-encoding", "")
    is_chunked = te.lower() == "chunked"
    cl = resp_headers.get("content-length")
    content_length = int(cl) if cl else None
    return StreamingResponse._from_async(
        status,
        resp_headers,
        url,
        reader,
        writer,
        is_chunked,
        content_length,
        timeout,
        content_encoding=content_encoding,
    )


# -- Async transport --


async def _async_request(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    data: bytes | str | dict[str, str] | None = None,
    json: Any = None,
    files: dict[str, Any] | list[tuple[str, Any]] | None = None,
    params: dict[str, Any] | None = None,
    timeout: float = DEFAULT_TIMEOUT,
    max_redirects: int = DEFAULT_MAX_REDIRECTS,
    verify: bool = True,
    stream: bool = False,
    auth: tuple[str, str] | Auth | None = None,
    proxy: str | None = None,
    _pool: _AsyncConnectionPool | None = None,
) -> Response | StreamingResponse:
    """Perform an asynchronous HTTP request using asyncio streams.

    Phase structure (mirrors _sync_request):
        1-3. Request preparation (URL, body, headers, auth)
        4. Redirect loop:
           a. URL parsing + connection acquisition
           b. Send raw HTTP/1.1 request + read response headers
           c. Handle redirects / digest auth / response construction
           d. Connection lifecycle (pool release or close)
    """
    url, body, req_headers, auth_obj = _prepare_request(
        method, url, headers, data, json, files, params, auth
    )

    redirects = 0
    _digest_attempted = False
    while True:
        scheme, host, port, path, is_https = _parse_url(url)

        reader, writer, request_path = await _async_acquire_connection(
            host, port, path, is_https, timeout, verify, proxy, _pool, req_headers, url
        )
        if _pool and not proxy:
            req_headers.setdefault("Connection", "keep-alive")

        close_writer = True
        resp_headers: dict[str, str] = {}
        try:
            raw_request = _build_raw_http_request(
                method,
                request_path,
                host,
                req_headers,
                use_pool=bool(_pool),
                use_proxy=bool(proxy),
            )
            writer.write(raw_request)
            if body:
                writer.write(body)
            await asyncio.wait_for(writer.drain(), timeout=timeout)

            status, resp_headers = await _async_read_response_headers(reader, timeout)

            if _is_redirect(status, resp_headers):
                await _async_read_body(reader, resp_headers, timeout)
                url, method, body = _compute_redirect(
                    status,
                    method,
                    body,
                    req_headers,
                    resp_headers,
                    scheme,
                    host,
                    port,
                    url,
                    redirects,
                    max_redirects,
                )
                redirects += 1
                continue

            if _should_attempt_digest(
                auth_obj, status, resp_headers, _digest_attempted
            ):
                await _async_read_body(reader, resp_headers, timeout)
                www_auth = resp_headers["www-authenticate"]
                req_headers.update(
                    auth_obj.auth_headers_from_challenge(method, path, www_auth)
                )
                _digest_attempted = True
                await _async_close_writer_silent(writer)
                close_writer = False
                continue

            if stream:
                close_writer = False
                return _build_async_streaming_response(
                    status,
                    resp_headers,
                    url,
                    reader,
                    writer,
                    timeout,
                )

            content_encoding = resp_headers.get("content-encoding", "")
            resp_body = await _async_read_body(reader, resp_headers, timeout)
            if content_encoding:
                resp_body = _decompress_body(resp_body, content_encoding)
            return Response(status, resp_headers, resp_body, url)
        except asyncio.TimeoutError:
            raise HttpTimeoutError(
                f"Request to {url} timed out after {timeout}s",
                url=url,
                timeout=timeout,
            )
        finally:
            await _async_release_or_close(
                close_writer,
                _pool,
                proxy,
                stream,
                resp_headers,
                host,
                port,
                is_https,
                reader,
                writer,
            )


# ── Request Building helpers (multipart, headers) ──


def _prepare_body(
    data: bytes | str | dict[str, str] | None = None,
    json: Any = None,
    files: dict[str, Any] | list[tuple[str, Any]] | None = None,
) -> tuple[bytes | None, str | None]:
    """Prepare request body and content-type header.

    Priority: json > files > data.
    When files is provided and data is a dict, data fields are included
    as text parts in the multipart body.

    Returns:
        (body_bytes, content_type) tuple.
    """
    if json is not None:
        return _json.dumps(json, ensure_ascii=False).encode("utf-8"), "application/json"
    if files is not None:
        form_data = data if isinstance(data, dict) else None
        return _encode_multipart(form_data, files)
    if isinstance(data, str):
        return data.encode("utf-8"), "application/x-www-form-urlencoded"
    if isinstance(data, bytes):
        return data, "application/octet-stream"
    return None, None


def _read_file_content(value: bytes | IO[bytes]) -> bytes:
    """Read bytes from a file object or return bytes as-is."""
    if isinstance(value, bytes):
        return value
    return value.read()


def _get_filename(value: bytes | IO[bytes]) -> str:
    """Extract filename from a file object, or return a default."""
    if isinstance(value, bytes):
        return "upload"
    name = getattr(value, "name", None)
    if name:
        return os.path.basename(name)
    return "upload"


def _normalize_file_value(
    value: Any,
) -> tuple[str, bytes, str]:
    """Normalize a files parameter value to (filename, content, content_type).

    Accepted formats:
        bytes / file object      -> ("upload"/basename, content, octet-stream)
        (filename, content)      -> (filename, content, octet-stream)
        (filename, content, ct)  -> (filename, content, ct)
    """
    if isinstance(value, (bytes, IO)) or hasattr(value, "read"):
        fn = _get_filename(value)
        ct = "application/octet-stream"
        return fn, _read_file_content(value), ct
    if isinstance(value, (tuple, list)):
        if len(value) == 2:
            fname, content = value
            return fname, _read_file_content(content), "application/octet-stream"
        if len(value) == 3:
            fname, content, ct = value
            return fname, _read_file_content(content), ct
    raise ValueError(f"Invalid file value format: {type(value)}")


def _encode_multipart(
    data: dict[str, str] | None,
    files: dict[str, Any] | list[tuple[str, Any]],
) -> tuple[bytes, str]:
    """Encode multipart/form-data body.

    Args:
        data: Optional form fields to include as text parts.
        files: File fields as dict or list of (name, value) tuples.

    Returns:
        (body_bytes, content_type_with_boundary).
    """
    boundary = os.urandom(16).hex()
    parts: list[bytes] = []

    # Encode form data fields
    if data:
        for name, value in data.items():
            part = (
                f"--{boundary}\r\n"
                f'Content-Disposition: form-data; name="{name}"\r\n'
                f"\r\n"
                f"{value}\r\n"
            )
            parts.append(part.encode("utf-8"))

    # Encode file fields
    items: list[tuple[str, Any]]
    if isinstance(files, dict):
        items = list(files.items())
    else:
        items = list(files)

    for name, value in items:
        filename, content, content_type = _normalize_file_value(value)
        header = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'
            f"Content-Type: {content_type}\r\n"
            f"\r\n"
        )
        parts.append(header.encode("utf-8") + content + b"\r\n")

    # Final boundary
    parts.append(f"--{boundary}--\r\n".encode("utf-8"))

    body = b"".join(parts)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def _merge_headers(
    base: dict[str, str] | None,
    extra: dict[str, str] | None,
) -> dict[str, str]:
    """Merge header dicts (case-insensitive merge, last wins)."""
    merged: dict[str, str] = {}
    for h in (base, extra):
        if h:
            for k, v in h.items():
                merged[k] = v
    return merged


# ── Public API functions (get, post, put, etc.) ──

# -- Sync convenience functions --


def get(url: str, **kwargs: Any) -> Response | StreamingResponse:
    """Send a GET request."""
    return _sync_request("GET", url, **kwargs)


def post(url: str, **kwargs: Any) -> Response | StreamingResponse:
    """Send a POST request."""
    return _sync_request("POST", url, **kwargs)


def put(url: str, **kwargs: Any) -> Response | StreamingResponse:
    """Send a PUT request."""
    return _sync_request("PUT", url, **kwargs)


def patch(url: str, **kwargs: Any) -> Response | StreamingResponse:
    """Send a PATCH request."""
    return _sync_request("PATCH", url, **kwargs)


def delete(url: str, **kwargs: Any) -> Response | StreamingResponse:
    """Send a DELETE request."""
    return _sync_request("DELETE", url, **kwargs)


def head(url: str, **kwargs: Any) -> Response | StreamingResponse:
    """Send a HEAD request."""
    return _sync_request("HEAD", url, **kwargs)


def options(url: str, **kwargs: Any) -> Response | StreamingResponse:
    """Send an OPTIONS request."""
    return _sync_request("OPTIONS", url, **kwargs)


# -- Async convenience functions --


async def async_get(url: str, **kwargs: Any) -> Response | StreamingResponse:
    """Send an async GET request."""
    return await _async_request("GET", url, **kwargs)


async def async_post(url: str, **kwargs: Any) -> Response | StreamingResponse:
    """Send an async POST request."""
    return await _async_request("POST", url, **kwargs)


async def async_put(url: str, **kwargs: Any) -> Response | StreamingResponse:
    """Send an async PUT request."""
    return await _async_request("PUT", url, **kwargs)


async def async_patch(url: str, **kwargs: Any) -> Response | StreamingResponse:
    """Send an async PATCH request."""
    return await _async_request("PATCH", url, **kwargs)


async def async_delete(url: str, **kwargs: Any) -> Response | StreamingResponse:
    """Send an async DELETE request."""
    return await _async_request("DELETE", url, **kwargs)


async def async_head(url: str, **kwargs: Any) -> Response | StreamingResponse:
    """Send an async HEAD request."""
    return await _async_request("HEAD", url, **kwargs)


async def async_options(url: str, **kwargs: Any) -> Response | StreamingResponse:
    """Send an async OPTIONS request."""
    return await _async_request("OPTIONS", url, **kwargs)


# ── Client classes (Client, AsyncClient) ──


class Client:
    """Synchronous HTTP client session with connection pooling.

    Thread-safe: uses a threading.Lock internally.

    Usage::

        with Client(headers={"Authorization": "Bearer token"}) as c:
            r = c.get("https://api.example.com/data")
    """

    def __init__(
        self,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        verify: bool = True,
        auth: tuple[str, str] | Auth | None = None,
        proxy: str | None = None,
        pool_size: int = DEFAULT_POOL_SIZE,
    ) -> None:
        self._base_headers = headers or {}
        self._timeout = timeout
        self._max_redirects = max_redirects
        self._verify = verify
        self._auth = auth
        self._proxy = proxy
        self._pool = _SyncConnectionPool(pool_size)
        self._lock = threading.Lock()

    def request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> Response | StreamingResponse:
        """Send an HTTP request."""
        kwargs.setdefault("timeout", self._timeout)
        kwargs.setdefault("max_redirects", self._max_redirects)
        kwargs.setdefault("verify", self._verify)
        kwargs.setdefault("auth", self._auth)
        kwargs.setdefault("proxy", self._proxy)
        kwargs["_pool"] = self._pool
        kwargs["headers"] = _merge_headers(self._base_headers, kwargs.get("headers"))
        if kwargs.get("stream"):
            return _sync_request(method, url, **kwargs)
        with self._lock:
            return _sync_request(method, url, **kwargs)

    def get(self, url: str, **kwargs: Any) -> Response | StreamingResponse:
        return self.request("GET", url, **kwargs)

    def post(self, url: str, **kwargs: Any) -> Response | StreamingResponse:
        return self.request("POST", url, **kwargs)

    def put(self, url: str, **kwargs: Any) -> Response | StreamingResponse:
        return self.request("PUT", url, **kwargs)

    def patch(self, url: str, **kwargs: Any) -> Response | StreamingResponse:
        return self.request("PATCH", url, **kwargs)

    def delete(self, url: str, **kwargs: Any) -> Response | StreamingResponse:
        return self.request("DELETE", url, **kwargs)

    def head(self, url: str, **kwargs: Any) -> Response | StreamingResponse:
        return self.request("HEAD", url, **kwargs)

    def options(self, url: str, **kwargs: Any) -> Response | StreamingResponse:
        return self.request("OPTIONS", url, **kwargs)

    def close(self) -> None:
        """Close all pooled connections."""
        self._pool.close_all()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *args: Any) -> None:
        self._pool.close_all()


class AsyncClient:
    """Asynchronous HTTP client session with connection pooling.

    Safe to use from a single asyncio task; for concurrent requests
    from the same client, use asyncio.Lock internally.

    Usage::

        async with AsyncClient(headers={"Authorization": "Bearer token"}) as c:
            r = await c.get("https://api.example.com/data")
    """

    def __init__(
        self,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = DEFAULT_TIMEOUT,
        max_redirects: int = DEFAULT_MAX_REDIRECTS,
        verify: bool = True,
        auth: tuple[str, str] | Auth | None = None,
        proxy: str | None = None,
        pool_size: int = DEFAULT_POOL_SIZE,
    ) -> None:
        self._base_headers = headers or {}
        self._timeout = timeout
        self._max_redirects = max_redirects
        self._verify = verify
        self._auth = auth
        self._proxy = proxy
        self._pool = _AsyncConnectionPool(pool_size)
        self._lock = asyncio.Lock()

    async def request(
        self,
        method: str,
        url: str,
        **kwargs: Any,
    ) -> Response | StreamingResponse:
        """Send an async HTTP request."""
        kwargs.setdefault("timeout", self._timeout)
        kwargs.setdefault("max_redirects", self._max_redirects)
        kwargs.setdefault("verify", self._verify)
        kwargs.setdefault("auth", self._auth)
        kwargs.setdefault("proxy", self._proxy)
        kwargs["_pool"] = self._pool
        kwargs["headers"] = _merge_headers(self._base_headers, kwargs.get("headers"))
        if kwargs.get("stream"):
            return await _async_request(method, url, **kwargs)
        async with self._lock:
            return await _async_request(method, url, **kwargs)

    async def get(self, url: str, **kwargs: Any) -> Response | StreamingResponse:
        return await self.request("GET", url, **kwargs)

    async def post(self, url: str, **kwargs: Any) -> Response | StreamingResponse:
        return await self.request("POST", url, **kwargs)

    async def put(self, url: str, **kwargs: Any) -> Response | StreamingResponse:
        return await self.request("PUT", url, **kwargs)

    async def patch(self, url: str, **kwargs: Any) -> Response | StreamingResponse:
        return await self.request("PATCH", url, **kwargs)

    async def delete(self, url: str, **kwargs: Any) -> Response | StreamingResponse:
        return await self.request("DELETE", url, **kwargs)

    async def head(self, url: str, **kwargs: Any) -> Response | StreamingResponse:
        return await self.request("HEAD", url, **kwargs)

    async def options(self, url: str, **kwargs: Any) -> Response | StreamingResponse:
        return await self.request("OPTIONS", url, **kwargs)

    async def aclose(self) -> None:
        """Close all pooled connections."""
        await self._pool.close_all()

    async def __aenter__(self) -> AsyncClient:
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self._pool.close_all()
