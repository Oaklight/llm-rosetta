"""An image sent to the Antigravity ACP lane arrives, or its loss is reported.

The shim declares ``text``, ``vision`` and ``reasoning`` by default, so the
vision pass leaves an inline image alone; a text-only per-model override still
strips it.  A remote image URL cannot ride in an ACP block, and that drop is
logged like every other conversion loss.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

import pytest

from llm_rosetta_gateway.app import create_app
from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.transport._base import UpstreamResponse

_PNG = "iVBORw0KGgo="
_REPLY = {
    "updates": [
        {"type": "session_start", "session_id": "s1", "model": "m"},
        {"type": "text", "text": "ok"},
        {"type": "stop", "stop_reason": "end_turn"},
    ],
    "stop_reason": "end_turn",
    "session_id": "s1",
    "model": "m",
}


class _FakeAgent:
    """An ACP lane that records the prompt blocks it receives."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def send(self, provider_info: Any, url: str, body: Any, **kw: Any) -> Any:
        self.calls.append(body)
        raw = json.dumps(_REPLY).encode()
        return UpstreamResponse(status_code=200, body=_REPLY, raw_content=raw)

    async def send_streaming(self, *args: Any, **kw: Any) -> Any:
        raise AssertionError("this test never streams")

    async def close(self) -> None:
        """Nothing to close."""


def _config(tmp_path: Any, model: Any) -> GatewayConfig:
    return GatewayConfig(
        {
            "providers": {"agy": {"type": "antigravity--acp"}},
            "models": {"m": model},
            "server": {"data_dir": str(tmp_path / "data"), "open_on_no_keys": True},
        }
    )


async def _request(
    port: int, method: str, path: str, body: dict[str, Any] | None = None
) -> tuple[int, Any]:
    raw = json.dumps(body).encode() if body is not None else b""
    request = (
        f"{method} {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n"
        f"Content-Type: application/json\r\nContent-Length: {len(raw)}\r\n\r\n"
    ).encode() + raw
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(request)
    await writer.drain()
    writer.write_eof()
    data = await reader.read()
    writer.close()
    head, _, payload = data.partition(b"\r\n\r\n")
    return int(head.decode().split("\r\n")[0].split()[1]), json.loads(payload or b"{}")


@pytest.fixture()
def serve(tmp_path):
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def _serve(model: Any):
        config = _config(tmp_path, model)
        # The shim's sign-in exchange would otherwise leave the entry waiting
        # for a credential no test agent has.
        for info in config.providers.values():
            info.key_ring.refresh("signed-in")
        app = await create_app(config)
        agent = _FakeAgent()
        wired: Any = app
        wired.transports = {"acp": agent}
        listener = await asyncio.start_server(app._handle_connection, "127.0.0.1", 0)
        try:
            yield listener.sockets[0].getsockname()[1], agent
        finally:
            listener.close()
            await listener.wait_closed()

    return _serve


def _messages(source: dict[str, Any]) -> dict[str, Any]:
    return {
        "model": "m",
        "max_tokens": 16,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "what colour?"},
                    {"type": "image", "source": source},
                ],
            }
        ],
    }


_B64 = {"type": "base64", "media_type": "image/png", "data": _PNG}
_URL = {"type": "url", "url": "https://example.com/cat.png"}


def _blocks(agent: _FakeAgent) -> list[dict[str, Any]]:
    assert len(agent.calls) == 1
    return agent.calls[0]["prompt"]


async def test_an_image_reaches_the_agent_with_no_capability_override(serve):
    async with serve("agy") as (port, agent):
        status, _ = await _request(port, "POST", "/v1/messages", _messages(_B64))

    assert status == 200
    assert {"type": "image", "data": _PNG, "mimeType": "image/png"} in _blocks(agent)


async def test_a_text_only_override_still_strips_the_image(serve, caplog):
    model = {"provider": "agy", "capabilities": ["text"]}
    with caplog.at_level(logging.WARNING):
        async with serve(model) as (port, agent):
            status, _ = await _request(port, "POST", "/v1/messages", _messages(_B64))

    assert status == 200
    assert all(b["type"] != "image" for b in _blocks(agent))
    assert "does not have vision capability" in caplog.text


async def test_models_listing_shows_the_shim_default(serve):
    async with serve("agy") as (port, _agent):
        _, listing = await _request(port, "GET", "/v1/models")

    entry = next(m for m in listing["data"] if m["id"] == "m")
    assert entry["capabilities"] == ["text", "vision", "reasoning"]


async def test_an_explicit_capabilities_list_overrides_the_shim_default(serve):
    model = {"provider": "agy", "capabilities": ["text", "tools"]}
    async with serve(model) as (port, _agent):
        _, listing = await _request(port, "GET", "/v1/models")

    entry = next(m for m in listing["data"] if m["id"] == "m")
    assert entry["capabilities"] == ["text", "tools"]


async def test_a_remote_image_url_drop_is_logged(serve, caplog):
    with caplog.at_level(logging.WARNING):
        async with serve("agy") as (port, agent):
            status, _ = await _request(port, "POST", "/v1/messages", _messages(_URL))

    assert status == 200
    assert all(b["type"] != "image" for b in _blocks(agent))
    assert "an image URL was dropped" in caplog.text
