"""Routing verdicts an unknown path must keep.

A catch-all OPTIONS route makes every path match, so a browser's ordinary GET
for a file it caches (``/sw.js``) was answered "405 Method Not Allowed".  The
preflight hook must not change the router's own verdict: unknown path is 404,
a known path with the wrong method is 405, and a real preflight is 204.
"""

from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any

import pytest

from llm_rosetta_gateway.app import create_app
from llm_rosetta_gateway.config import GatewayConfig


def _config(tmp_path: Any) -> GatewayConfig:
    return GatewayConfig(
        {
            "providers": {},
            "models": {},
            "server": {"data_dir": str(tmp_path / "data"), "open_on_no_keys": True},
        }
    )


async def _request(port: int, method: str, path: str, headers: dict[str, str]) -> Any:
    lines = [f"{method} {path} HTTP/1.1", "Host: t", "Connection: close"]
    lines.extend(f"{name}: {value}" for name, value in headers.items())
    raw = ("\r\n".join(lines) + "\r\n\r\n").encode()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(raw)
    await writer.drain()
    writer.write_eof()
    data = await reader.read()
    writer.close()
    head, _, body = data.partition(b"\r\n\r\n")
    first, *header_lines = head.decode().split("\r\n")
    status = int(first.split()[1])
    parsed = {}
    for line in header_lines:
        name, _, value = line.partition(":")
        parsed[name.lower()] = value.strip()
    return status, parsed, body


@pytest.fixture()
def serve(tmp_path):
    @asynccontextmanager
    async def _serve():
        app = await create_app(_config(tmp_path))
        listener = await asyncio.start_server(app._handle_connection, "127.0.0.1", 0)
        try:
            yield listener.sockets[0].getsockname()[1]
        finally:
            listener.close()
            await listener.wait_closed()

    return _serve


async def test_unknown_get_is_not_a_method_error(serve):
    async with serve() as port:
        status, _, _ = await _request(port, "GET", "/sw.js", {})

    assert status == 404


async def test_a_known_path_with_the_wrong_method_is_still_405(serve):
    async with serve() as port:
        status, headers, _ = await _request(port, "POST", "/health", {})

    assert status == 405
    assert headers.get("allow") == "GET"


async def test_preflight_answers_204_with_cors(serve):
    async with serve() as port:
        status, headers, _ = await _request(
            port,
            "OPTIONS",
            "/v1/chat/completions",
            {"Origin": "https://example.test"},
        )

    assert status == 204
    assert headers.get("access-control-allow-origin") == "*"
    assert headers.get("access-control-allow-methods") == "*"
