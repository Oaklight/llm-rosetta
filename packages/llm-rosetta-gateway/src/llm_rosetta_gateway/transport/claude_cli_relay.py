"""The per-turn loopback relay the Claude Code CLI's base URL points at.

The CLI builds and signs its own upstream request, so the gateway cannot read
that request from a return value: it has to stand where the request goes.  The
transport sets ``ANTHROPIC_BASE_URL`` to this relay, the CLI posts its turn to
``/<token>/v1/messages`` here, and the relay forwards it unchanged to the real
upstream with the CLI's own headers.  The credential never leaves the proxy
path it already had; it is copied through memory and never logged.

Two jobs belong here beyond forwarding:

* **Single-request admission.**  ``--max-turns 1`` bounds the CLI's agent
  loop, not its recovery attempts: the CLI can retry or continue on its own
  and issue a second upstream request for one host request.  The relay admits
  exactly one and answers any further attempt locally with a 429, so one host
  request costs one upstream call.  An upstream 401 is the exception: the CLI
  refreshes its login and retries, and that retry is admitted once.
* **Capturing the wire reading.**  The upstream status and the
  ``anthropic-ratelimit-*`` headers are kept for the quota card and the
  request log, which the CLI's own stdout does not carry reliably.

Response framing is preserved as the upstream sent it (``Content-Length`` or
chunked), with ``Connection: close`` added, so the CLI reads exactly one
response.
"""

from __future__ import annotations

import asyncio
import json
import secrets
import ssl
from dataclasses import dataclass, field
from urllib.parse import urlsplit

#: Where the relay forwards when the shim names no upstream.
DEFAULT_UPSTREAM = "https://api.anthropic.com"

#: Headers the relay owns on the forwarded request.
_HOP_BY_HOP = frozenset(
    {
        "host",
        "connection",
        "proxy-connection",
        "keep-alive",
        "upgrade",
        "transfer-encoding",
        "content-length",
    }
)

#: A loopback host may be plain ``http``; anything else must be TLS.
_LOOPBACK = frozenset({"127.0.0.1", "localhost", "::1"})

#: Bytes copied per read when forwarding a body.
_CHUNK = 64 * 1024

#: The local answer to a second upstream attempt in one turn.
_REFUSAL = json.dumps(
    {
        "type": "error",
        "error": {
            "type": "rate_limit_error",
            "message": (
                "claude_cli: this turn already sent its upstream request; "
                "one gateway request admits one upstream call"
            ),
        },
    }
).encode("utf-8")


@dataclass
class RelayReading:
    """What one upstream call returned, as the gateway observed it."""

    status: int = 0
    headers: dict[str, str] = field(default_factory=dict)

    def rate_limit(self) -> dict[str, str]:
        """The ``anthropic-ratelimit-*`` headers, lower-cased."""
        return {
            key: value
            for key, value in self.headers.items()
            if key.startswith("anthropic-ratelimit-")
        }


class RelayError(RuntimeError):
    """The relay could not be started or reached its upstream."""


class Relay:
    """A one-turn loopback endpoint that forwards one request upstream."""

    def __init__(
        self,
        upstream: str | None = None,
        *,
        timeout: float = 300.0,
    ) -> None:
        self._upstream = urlsplit(upstream or DEFAULT_UPSTREAM)
        if (
            self._upstream.scheme not in ("http", "https")
            or not self._upstream.hostname
        ):
            raise RelayError(f"claude_cli: upstream {upstream!r} is not a URL")
        if self._upstream.scheme == "http" and self._upstream.hostname not in _LOOPBACK:
            raise RelayError(
                "claude_cli: the upstream must be https:// (http:// only for a "
                f"loopback host), got {upstream!r}"
            )
        self._timeout = float(timeout)
        self._token = secrets.token_urlsafe(18)
        self._server: asyncio.AbstractServer | None = None
        self._port = 0
        self._admitted = 0
        self._retried_after_401 = False
        self._last_status: int | None = None
        self._last_error: str | None = None
        self.readings: list[RelayReading] = []

    # -- lifecycle ----------------------------------------------------------

    @property
    def url(self) -> str:
        """The base URL to hand the CLI."""
        return f"http://127.0.0.1:{self._port}/{self._token}"

    @property
    def path_prefix(self) -> str:
        """The path every admitted request must start with."""
        return f"/{self._token}"

    async def start(self) -> str:
        """Bind an ephemeral loopback port and start serving."""
        self._server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        sockets = self._server.sockets or ()
        if not sockets:
            await self.close()
            raise RelayError("claude_cli: the relay could not bind a port")
        self._port = int(sockets[0].getsockname()[1])
        return self.url

    async def close(self) -> None:
        """Stop accepting connections.  In-flight requests are cancelled."""
        server, self._server = self._server, None
        if server is not None:
            server.close()
            try:
                await server.wait_closed()
            except Exception:  # noqa: BLE001 - closing must not raise
                pass

    # -- serving ------------------------------------------------------------

    async def _handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            await asyncio.wait_for(
                self._serve_one(reader, writer), timeout=self._timeout
            )
        except (TimeoutError, asyncio.TimeoutError):
            await _reply(writer, 504, b"relay timeout")
        except (ConnectionError, OSError):
            pass
        except RelayError as exc:
            await _reply(writer, 502, str(exc).encode("utf-8"))
        finally:
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def _serve_one(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        request = await _read_request(reader)
        if request is None:
            return
        method, target, headers, body = request
        prefix = self.path_prefix
        path = target.split("?", 1)[0]
        if method != "POST" or path != f"{prefix}/v1/messages":
            await _reply(writer, 404, b"not found")
            return
        if not self._admits():
            if self._last_error is not None:
                # The one attempt failed before it reached the upstream.  A
                # retry gets the real cause, not "already sent".
                await _reply(writer, 502, self._last_error.encode("utf-8"))
            elif self._last_status is not None and self._last_status >= 400:
                # The one attempt reached the upstream and it refused.  Repeat
                # that refusal, so the client reports the real status (an
                # expired login, a plan limit) instead of a gateway 429.
                await _reply(
                    writer,
                    self._last_status,
                    json.dumps(
                        {
                            "type": "error",
                            "error": {
                                "type": "api_error",
                                "message": (
                                    "claude_cli: the upstream answered "
                                    f"{self._last_status}; this turn admits one "
                                    "upstream request"
                                ),
                            },
                        }
                    ).encode("utf-8"),
                )
            else:
                await _reply(writer, 429, _REFUSAL)
            return
        # The relay's own token is not part of the upstream path.
        await self._forward(target[len(prefix) :], headers, body, writer)

    def _admits(self) -> bool:
        """Whether this attempt may reach the upstream."""
        if self._admitted == 0:
            self._admitted += 1
            return True
        if self._last_status == 401 and not self._retried_after_401:
            self._retried_after_401 = True
            self._admitted += 1
            return True
        return False

    async def _forward(
        self,
        target: str,
        headers: dict[str, str],
        body: bytes,
        writer: asyncio.StreamWriter,
    ) -> None:
        """Send one request upstream and copy its response back verbatim."""
        host = self._upstream.hostname or ""
        port = self._upstream.port or (443 if self._upstream.scheme == "https" else 80)
        context = (
            ssl.create_default_context() if self._upstream.scheme == "https" else None
        )
        try:
            up_reader, up_writer = await asyncio.wait_for(
                asyncio.open_connection(host, port, ssl=context),
                timeout=self._timeout,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            self._last_error = f"claude_cli: cannot reach {host}: {exc}"
            raise RelayError(self._last_error) from exc
        try:
            await self._exchange(target, headers, body, up_reader, up_writer, writer)
        finally:
            up_writer.close()
            try:
                await up_writer.wait_closed()
            except (ConnectionError, OSError):
                pass

    async def _exchange(
        self,
        target: str,
        headers: dict[str, str],
        body: bytes,
        up_reader: asyncio.StreamReader,
        up_writer: asyncio.StreamWriter,
        writer: asyncio.StreamWriter,
    ) -> None:
        path = target if target.startswith("/") else f"/{target}"
        prefix = self._upstream.path.rstrip("/")
        query = ""
        if "?" in path:
            path, query = path.split("?", 1)
            query = f"?{query}"
        forwarded = {
            key: value
            for key, value in headers.items()
            if key.lower() not in _HOP_BY_HOP
        }
        forwarded["host"] = self._upstream.hostname or ""
        forwarded["content-length"] = str(len(body))
        head = [f"POST {prefix}{path}{query} HTTP/1.1"]
        head += [f"{key}: {value}" for key, value in forwarded.items()]
        up_writer.write(("\r\n".join(head) + "\r\n\r\n").encode("latin-1"))
        if body:
            up_writer.write(body)
        await up_writer.drain()

        status_line = await _readline(up_reader)
        response_headers = await _read_headers(up_reader)
        try:
            status = int(status_line.split(" ", 2)[1])
        except (IndexError, ValueError) as exc:
            raise RelayError("claude_cli: the upstream sent no status line") from exc
        self._last_status = status
        self.readings.append(
            RelayReading(
                status=status,
                headers={k.lower(): v for k, v in response_headers.items()},
            )
        )
        out = [status_line]
        for key, value in response_headers.items():
            if key.lower() in ("connection", "transfer-encoding", "content-length"):
                continue
            out.append(f"{key}: {value}")
        for key in ("transfer-encoding", "content-length"):
            for name, value in response_headers.items():
                if name.lower() == key:
                    out.append(f"{key}: {value}")
        out.append("connection: close")
        writer.write(("\r\n".join(out) + "\r\n\r\n").encode("latin-1"))
        await _copy_body(up_reader, writer, response_headers)
        await writer.drain()


async def _readline(reader: asyncio.StreamReader) -> str:
    """One CRLF-terminated line, decoded as latin-1."""
    line = await reader.readline()
    if not line:
        raise RelayError("claude_cli: the upstream closed before a status line")
    return line.decode("latin-1").rstrip("\r\n")


async def _read_request(
    reader: asyncio.StreamReader,
) -> tuple[str, str, dict[str, str], bytes] | None:
    """Read one HTTP/1.1 request; ``None`` when the peer sent nothing."""
    line = await reader.readline()
    if not line:
        return None
    try:
        method, target, _version = line.decode("latin-1").split()
    except ValueError as exc:
        raise RelayError("claude_cli: malformed request line") from exc
    headers = await _read_headers(reader)
    body = await _read_body(reader, headers)
    return method, target, headers, body


async def _read_headers(reader: asyncio.StreamReader) -> dict[str, str]:
    """Read header lines until the blank one, preserving the order given."""
    headers: dict[str, str] = {}
    while True:
        line = await reader.readline()
        if not line or line in (b"\r\n", b"\n"):
            return headers
        text = line.decode("latin-1").rstrip("\r\n")
        if ":" not in text:
            continue
        name, _, value = text.partition(":")
        headers[name.strip()] = value.strip()


async def _read_body(reader: asyncio.StreamReader, headers: dict[str, str]) -> bytes:
    """Read a request body by ``Content-Length`` or chunked framing."""
    lowered = {key.lower(): value for key, value in headers.items()}
    if "chunked" in lowered.get("transfer-encoding", "").lower():
        return await _read_chunked(reader)
    length = lowered.get("content-length")
    if not length:
        return b""
    try:
        size = int(length)
    except ValueError as exc:
        raise RelayError("claude_cli: malformed Content-Length") from exc
    return await reader.readexactly(size) if size else b""


async def _read_chunked(reader: asyncio.StreamReader) -> bytes:
    """De-chunk a request body."""
    body = bytearray()
    while True:
        line = await reader.readline()
        if not line:
            raise RelayError("claude_cli: truncated chunked body")
        try:
            size = int(line.split(b";", 1)[0].strip(), 16)
        except ValueError as exc:
            raise RelayError("claude_cli: malformed chunk size") from exc
        if size == 0:
            while True:
                trailer = await reader.readline()
                if not trailer or trailer in (b"\r\n", b"\n"):
                    return bytes(body)
        body += await reader.readexactly(size)
        await reader.readexactly(2)


async def _copy_body(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    headers: dict[str, str],
) -> None:
    """Copy an upstream response body with its own framing."""
    lowered = {key.lower(): value for key, value in headers.items()}
    if "chunked" in lowered.get("transfer-encoding", "").lower():
        await _copy_chunked(reader, writer)
        return
    length = lowered.get("content-length")
    if length is not None:
        try:
            remaining = int(length)
        except ValueError:
            remaining = -1
        if remaining >= 0:
            await _copy_sized(reader, writer, remaining)
            return
    await _copy_until_eof(reader, writer)


async def _copy_chunked(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Copy a chunked body chunk by chunk, trailers included."""
    while True:
        line = await reader.readline()
        if not line:
            return
        writer.write(line)
        try:
            size = int(line.split(b";", 1)[0].strip(), 16)
        except ValueError:
            return
        if size == 0:
            await _copy_trailers(reader, writer)
            return
        writer.write(await reader.readexactly(size))
        writer.write(await reader.readexactly(2))
        await writer.drain()


async def _copy_trailers(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Copy the trailer section that ends a chunked body."""
    while True:
        trailer = await reader.readline()
        writer.write(trailer)
        if not trailer or trailer in (b"\r\n", b"\n"):
            return


async def _copy_sized(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, remaining: int
) -> None:
    """Copy exactly *remaining* bytes."""
    while remaining:
        piece = await reader.read(min(_CHUNK, remaining))
        if not piece:
            return
        writer.write(piece)
        remaining -= len(piece)
        await writer.drain()


async def _copy_until_eof(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Copy until the upstream closes the connection."""
    while True:
        piece = await reader.read(_CHUNK)
        if not piece:
            return
        writer.write(piece)
        await writer.drain()


async def _reply(
    writer: asyncio.StreamWriter,
    status: int,
    body: bytes,
    *,
    content_type: str = "application/json",
) -> None:
    """Answer the peer locally, without touching the upstream."""
    reason = {
        404: "Not Found",
        429: "Too Many Requests",
        502: "Bad Gateway",
        504: "Gateway Timeout",
    }.get(status, "Error")
    head = (
        f"HTTP/1.1 {status} {reason}\r\n"
        f"content-type: {content_type}\r\n"
        f"content-length: {len(body)}\r\n"
        "connection: close\r\n\r\n"
    )
    try:
        writer.write(head.encode("latin-1") + body)
        await writer.drain()
    except (ConnectionError, OSError):
        pass
