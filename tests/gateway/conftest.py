"""Shared fixtures for gateway tests.

Provides a minimal httpbin-compatible HTTP server, a SOCKS5 proxy server
(with optional auth), and related fixtures for integration testing.
"""

from __future__ import annotations

import inspect
import json
import logging
import os
import select
import socket
import socketserver
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from typing import Any

import pytest


@pytest.fixture(autouse=True)
def _no_leaked_extensions_dir():
    """The gateway points core's extensions directory at its data directory.

    That setting is process-wide, so a test that starts a gateway must not
    leave its temporary directory behind for the next one.
    """
    from llm_rosetta.api import configure_extensions_dir

    configure_extensions_dir(None)
    yield
    configure_extensions_dir(None)


# ---------------------------------------------------------------------------
# Minimal httpbin-compatible handler
# ---------------------------------------------------------------------------


class _HttpBinHandler(BaseHTTPRequestHandler):
    """Lightweight httpbin server for proxy integration tests."""

    def log_message(self, format, *args):  # noqa: A002
        pass

    def _send_json(self, data, status=200):
        body = json.dumps(data, ensure_ascii=False).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _headers_dict(self):
        return {k: v for k, v in self.headers.items()}

    def _request_url(self):
        return f"http://{self.headers['Host']}{self.path}"

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length) if length else b""

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/get":
            args = {k: v[0] for k, v in parse_qs(parsed.query).items()}
            self._send_json(
                {
                    "args": args,
                    "headers": self._headers_dict(),
                    "url": self._request_url(),
                }
            )
        elif path.startswith("/stream-bytes/"):
            n = int(path.rsplit("/", 1)[1])
            data = os.urandom(n)
            self.send_response(200)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(n))
            self.end_headers()
            self.wfile.write(data)
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path.rstrip("/")

        if path == "/post":
            raw = self._read_body()
            ct = self.headers.get("Content-Type", "")
            json_data = None
            if "application/json" in ct and raw:
                json_data = json.loads(raw)
            self._send_json(
                {
                    "args": {k: v[0] for k, v in parse_qs(parsed.query).items()},
                    "headers": self._headers_dict(),
                    "url": self._request_url(),
                    "json": json_data,
                }
            )
        else:
            self.send_response(404)
            self.send_header("Content-Length", "0")
            self.end_headers()


# ---------------------------------------------------------------------------
# SOCKS5 proxy server (RFC 1928 / 1929)
# ---------------------------------------------------------------------------


class _Socks5Server(socketserver.ThreadingTCPServer):
    """Minimal SOCKS5 server that optionally requires username/password auth."""

    allow_reuse_address = True

    def __init__(self, addr, handler_class, auth=None):
        super().__init__(addr, handler_class)
        self.socks5_auth = auth  # (username, password) or None


class _Socks5Handler(socketserver.StreamRequestHandler):
    """Minimal SOCKS5 proxy handler (RFC 1928 / 1929) for testing."""

    def handle(self):
        try:
            self._do_handshake()
        except Exception:
            return

    def _do_handshake(self):
        if not self._negotiate_method():
            return

        target_addr = self._read_connect_request()
        if target_addr is None:
            return
        target_host, target_port = target_addr

        try:
            target = socket.create_connection((target_host, target_port), timeout=10)
        except Exception:
            self._send_reply(0x05)
            return

        self._send_reply(0x00)
        self._relay(target)

    def _negotiate_method(self) -> bool:
        """SOCKS5 method negotiation (RFC 1928) + optional auth (RFC 1929)."""
        header = self.rfile.read(2)
        if len(header) < 2:
            return False
        ver, nmethods = struct.unpack("BB", header)
        if ver != 0x05:
            return False
        methods = self.rfile.read(nmethods)
        if len(methods) < nmethods:
            return False

        require_auth = self.server.socks5_auth is not None  # type: ignore

        if require_auth:
            if 0x02 not in methods:
                self.wfile.write(struct.pack("BB", 0x05, 0xFF))
                return False
            self.wfile.write(struct.pack("BB", 0x05, 0x02))
            return self._authenticate()

        self.wfile.write(struct.pack("BB", 0x05, 0x00))
        return True

    def _authenticate(self) -> bool:
        """RFC 1929 username/password sub-negotiation."""
        auth_header = self.rfile.read(2)
        if len(auth_header) < 2:
            return False
        _auth_ver, ulen = struct.unpack("BB", auth_header)
        uname = self.rfile.read(ulen)
        plen_byte = self.rfile.read(1)
        if not plen_byte:
            return False
        plen = struct.unpack("B", plen_byte)[0]
        passwd = self.rfile.read(plen)

        expected_user, expected_pass = self.server.socks5_auth  # type: ignore
        if uname.decode() != expected_user or passwd.decode() != expected_pass:
            self.wfile.write(struct.pack("BB", 0x01, 0x01))
            return False
        self.wfile.write(struct.pack("BB", 0x01, 0x00))
        return True

    def _read_connect_request(self) -> tuple[str, int] | None:
        """Parse SOCKS5 CONNECT request and return (host, port)."""
        req_header = self.rfile.read(4)
        if len(req_header) < 4:
            return None
        _ver, cmd, _rsv, atype = struct.unpack("BBBB", req_header)
        if cmd != 0x01:
            self._send_reply(0x07)
            return None

        if atype == 0x01:
            addr_bytes = self.rfile.read(4)
            target_host = socket.inet_ntoa(addr_bytes)
        elif atype == 0x03:
            addr_len = struct.unpack("B", self.rfile.read(1))[0]
            target_host = self.rfile.read(addr_len).decode()
        elif atype == 0x04:
            addr_bytes = self.rfile.read(16)
            target_host = socket.inet_ntop(socket.AF_INET6, addr_bytes)
        else:
            self._send_reply(0x08)
            return None

        target_port = struct.unpack("!H", self.rfile.read(2))[0]
        return target_host, target_port

    def _relay(self, target: socket.socket) -> None:
        """Bidirectional relay between client and target sockets."""
        client_sock = self.connection
        client_sock.setblocking(False)
        target.setblocking(False)
        try:
            while True:
                readable, _, _ = select.select([client_sock, target], [], [], 1.0)
                if not readable:
                    continue
                for sock in readable:
                    try:
                        data = sock.recv(8192)
                    except (BlockingIOError, ConnectionResetError):
                        data = b""
                    if not data:
                        return
                    if sock is client_sock:
                        target.sendall(data)
                    else:
                        client_sock.sendall(data)
        except Exception:
            pass
        finally:
            target.close()

    def _send_reply(self, reply_code):
        """Send a SOCKS5 reply with the given status code."""
        self.wfile.write(
            struct.pack("BBBB", 0x05, reply_code, 0x00, 0x01)
            + b"\x00\x00\x00\x00"  # bind addr (0.0.0.0)
            + struct.pack("!H", 0)  # bind port
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def httpbin_url():
    """Start a local httpbin-compatible server and yield its base URL."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), _HttpBinHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture(scope="session")
def socks5_url():
    """Start a local SOCKS5 proxy (no auth) and yield its URL."""
    server = _Socks5Server(("127.0.0.1", 0), _Socks5Handler, auth=None)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"socks5://127.0.0.1:{port}"
    server.shutdown()


@pytest.fixture(scope="session")
def socks5_auth_url():
    """Start a local SOCKS5 proxy with username/password auth."""
    server = _Socks5Server(
        ("127.0.0.1", 0), _Socks5Handler, auth=("testuser", "testpass")
    )
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"socks5://testuser:testpass@127.0.0.1:{port}"
    server.shutdown()


class _CollectingHandler(logging.Handler):
    """Keeps the gateway's own log records, whatever the run imported first."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def gateway_warnings():
    """Records what the gateway logger emits during the test.

    That logger is created with ``propagate = False``, so caplog — which
    listens on the root logger — does not see its records once
    ``llm_rosetta_gateway.observability.logging`` has been imported.  A handler on the
    logger itself is one path under every pytest version.
    """
    logger = logging.getLogger("llm-rosetta-gateway")
    handler = _CollectingHandler()
    logger.addHandler(handler)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)


@pytest.fixture(autouse=True)
def _acp_profile_stays_in_tmp(monkeypatch, tmp_path_factory):
    """Keep an ACP agent profile out of the real ``~/.config``.

    A provider with no data directory falls back to a per-user location; a
    test must never write there.
    """
    from llm_rosetta_gateway.transport.acp import profile as acp_profile

    root = tmp_path_factory.mktemp("acp-profiles")
    monkeypatch.setattr(acp_profile, "fallback_root", lambda: root)


# ---------------------------------------------------------------------------
# Stores a test opened and never closed
# ---------------------------------------------------------------------------


def _track_store_creation(monkeypatch, opened: list[Any]) -> None:
    """Record every store the gateway's two factories hand out during a test."""
    from llm_rosetta_gateway.keystore import KeyStore
    from llm_rosetta_gateway.observability import PersistenceManager

    for cls in (PersistenceManager, KeyStore):
        factory = cls.create.__func__  # type: ignore[attr-defined]

        async def tracked(klass, *args, _factory=factory, **kwargs):
            store = await _factory(klass, *args, **kwargs)
            opened.append(store)
            return store

        monkeypatch.setattr(cls, "create", classmethod(tracked))


@pytest.fixture
async def _close_stores_left_open(monkeypatch):
    """Close the SQLite stores an async test opened and did not close.

    ``create_app`` opens a ``keys.db`` and a ``gateway.db``, and a test that
    builds an app, or a ``PersistenceManager`` / ``KeyStore``, rarely closes
    them.  Each leaves a connection, its worker thread and two or three file
    descriptors behind until the interpreter collects it, and the gateway suite
    runs in one process: it reached the descriptor limit (``Too many open
    files``) on a machine with the common limit of 1024.  This closes them in
    the test's own loop, after every other fixture has been torn down.
    """
    opened: list[Any] = []
    _track_store_creation(monkeypatch, opened)
    yield
    for store in reversed(opened):
        await store.close()


@pytest.fixture(autouse=True)
def _stores_are_closed(request):
    """Attach the closer to every async test; a sync test has no loop to close in."""
    if inspect.iscoroutinefunction(getattr(request.node, "obj", None)):
        request.getfixturevalue("_close_stores_left_open")
