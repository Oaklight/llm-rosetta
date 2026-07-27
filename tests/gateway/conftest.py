"""Shared fixtures for gateway tests.

Provides a minimal httpbin-compatible HTTP server, a SOCKS5 proxy server
(with optional auth), and related fixtures for integration testing.
"""

from __future__ import annotations

import json
import os
import select
import socket
import socketserver
import struct
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

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
        methods = self._read_method_negotiation()
        if methods is None:
            return
        if not self._perform_auth(methods):
            return
        target_info = self._read_connect_request()
        if target_info is None:
            return
        target_host, target_port = target_info
        try:
            target = socket.create_connection((target_host, target_port), timeout=10)
        except Exception:
            self._send_reply(0x05)  # connection refused
            return
        self._send_reply(0x00)
        self._relay(target)

    def _read_method_negotiation(self):
        """Phase 1: read version + method list. Returns methods bytes or None."""
        header = self.rfile.read(2)
        if len(header) < 2:
            return None
        ver, nmethods = struct.unpack("BB", header)
        if ver != 0x05:
            return None
        methods = self.rfile.read(nmethods)
        if len(methods) < nmethods:
            return None
        return methods

    def _perform_auth(self, methods):
        """Phase 2: select method and optionally authenticate. Returns True on success."""
        if self.server.socks5_auth is None:  # type: ignore
            self.wfile.write(struct.pack("BB", 0x05, 0x00))  # no auth
            return True
        if 0x02 not in methods:
            self.wfile.write(struct.pack("BB", 0x05, 0xFF))
            return False
        self.wfile.write(struct.pack("BB", 0x05, 0x02))
        return self._read_userpass_auth()

    def _read_userpass_auth(self):
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
            self.wfile.write(struct.pack("BB", 0x01, 0x01))  # auth failure
            return False
        self.wfile.write(struct.pack("BB", 0x01, 0x00))  # auth success
        return True

    def _read_connect_request(self):
        """Phase 3: parse CONNECT request. Returns (host, port) or None on error."""
        req_header = self.rfile.read(4)
        if len(req_header) < 4:
            return None
        ver, cmd, _rsv, atype = struct.unpack("BBBB", req_header)
        if cmd != 0x01:  # only CONNECT supported
            self._send_reply(0x07)
            return None
        target_host = self._read_target_host(atype)
        if target_host is None:
            return None
        target_port = struct.unpack("!H", self.rfile.read(2))[0]
        return target_host, target_port

    def _read_target_host(self, atype):
        """Parse SOCKS5 address by type. Sends error reply on unsupported type."""
        if atype == 0x01:  # IPv4
            return socket.inet_ntoa(self.rfile.read(4))
        if atype == 0x03:  # domain
            addr_len = struct.unpack("B", self.rfile.read(1))[0]
            return self.rfile.read(addr_len).decode()
        if atype == 0x04:  # IPv6
            return socket.inet_ntop(socket.AF_INET6, self.rfile.read(16))
        self._send_reply(0x08)
        return None

    def _relay(self, target):
        """Bidirectional non-blocking relay between client and target."""
        client_sock = self.connection
        client_sock.setblocking(False)
        target.setblocking(False)
        try:
            while True:
                readable, _, _ = select.select([client_sock, target], [], [], 1.0)
                if not readable:
                    continue
                if not self._pump_readable(readable, client_sock, target):
                    return
        except Exception:
            pass
        finally:
            target.close()

    def _pump_readable(self, readable, client_sock, target):
        """Forward data for each readable socket. Returns False when EOF hit."""
        for sock in readable:
            try:
                data = sock.recv(8192)
            except (BlockingIOError, ConnectionResetError):
                data = b""
            if not data:
                return False
            if sock is client_sock:
                target.sendall(data)
            else:
                client_sock.sendall(data)
        return True

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
