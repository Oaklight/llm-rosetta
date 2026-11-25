"""An extension's transport at app creation: built once, scoped, served (epic #1703 E6)."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from _ext_wheels import build_wheel
from llm_rosetta.api import configure_extensions_dir, install_wheel, load_providers
from llm_rosetta_gateway.app import create_app
from llm_rosetta_gateway.app.factory import GatewayExtensions
from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.transport import registry
from llm_rosetta_gateway.transport._base import UpstreamResponse

TRANSPORT = """\
class Pigeon:
    def __init__(self, *, timeout=1.0):
        self.timeout = timeout

TRANSPORTS = {"pigeon": Pigeon}
"""

_CHAT = {"model": "m", "messages": [{"role": "user", "content": "hi"}]}


def _install(
    tmp_path: Path, transport_module: str = TRANSPORT, *, default: bool = True
) -> None:
    ext_dir = tmp_path / "extensions"
    ext_dir.mkdir()
    configure_extensions_dir(ext_dir)
    wheel = build_wheel(
        "acme",
        "0.1.0",
        ["acme--chat_completions"],
        members={"transport.py": transport_module},
        transport_of={"acme--chat_completions": "pigeon"} if default else None,
        transports=("pigeon",),
    )
    install_wheel(wheel, dest=ext_dir)
    load_providers()


def _config(tmp_path: Path, **provider: Any) -> GatewayConfig:
    return GatewayConfig(
        {
            "providers": {
                "p": {"type": "acme--chat_completions", "api_key": "k", **provider}
            },
            "models": {"m": {"providers": [{"name": "p"}]}},
            "server": {"data_dir": str(tmp_path / "data"), "open_on_no_keys": True},
        }
    )


async def _post(port: int, path: str, body: dict[str, Any]) -> tuple[int, Any]:
    raw = json.dumps(body).encode()
    request = (
        f"POST {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n"
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


def _imported() -> bool:
    return "llm_rosetta_ext.acme.transport" in sys.modules


async def test_the_app_builds_the_extensions_transport_and_serves_the_lane(tmp_path):
    _install(tmp_path)
    config = _config(tmp_path)
    assert config.providers["p"].transport_key == "acme:pigeon"
    assert not _imported()

    app: Any = await create_app(config, load_shim_sources=False)

    pigeon = app.transports["acme:pigeon"]
    assert type(pigeon).__name__ == "Pigeon"
    assert pigeon.timeout == config.upstream_timeout
    assert app.transports["acp"] is not pigeon
    calls: list[Any] = []

    async def send(provider_info, url, body, **kw):
        calls.append(provider_info.transport_key)
        payload = {
            "id": "r1",
            "object": "chat.completion",
            "choices": [
                {"index": 0, "message": {"role": "assistant", "content": "hi"}}
            ],
        }
        return UpstreamResponse(
            status_code=200, body=payload, raw_content=json.dumps(payload).encode()
        )

    class Http:
        async def send(self, *a, **k):
            raise AssertionError("the lane must not use the HTTP transport")

        async def close(self):
            return None

    pigeon.send = send
    app.transport = Http()
    listener = await asyncio.start_server(app._handle_connection, "127.0.0.1", 0)
    try:
        status, payload = await _post(
            listener.sockets[0].getsockname()[1], "/v1/chat/completions", _CHAT
        )
    finally:
        listener.close()
        await listener.wait_closed()

    assert status == 200 and calls == ["acme:pigeon"]
    assert payload["choices"][0]["message"]["content"] == "hi"


async def test_a_broken_transport_module_fails_startup_naming_the_extension(tmp_path):
    _install(tmp_path, "raise ImportError('no module named acme_lib')\n")

    with pytest.raises(registry.TransportPluginError, match="Extension 'acme'"):
        await create_app(_config(tmp_path), load_shim_sources=False)


async def test_an_embedder_transport_overrides_every_lane_and_imports_nothing(tmp_path):
    _install(tmp_path)
    mine = object()

    app: Any = await create_app(
        _config(tmp_path),
        extensions=GatewayExtensions(transport=mine),
        load_shim_sources=False,
    )

    assert app.transports["acme:pigeon"] is mine and app.transports["acp"] is mine
    assert not _imported()


async def test_a_provider_that_names_the_transport_in_config_gets_it_built(tmp_path):
    """The shim declares no transport: only the provider's config asks for it."""
    _install(tmp_path, default=False)
    config = _config(tmp_path, transport="pigeon")
    assert config.providers["p"].transport_key == "acme:pigeon"

    app: Any = await create_app(config, load_shim_sources=False)

    assert type(app.transports["acme:pigeon"]).__name__ == "Pigeon"


async def test_an_unreferenced_transport_is_not_imported_at_startup(tmp_path):
    _install(tmp_path, default=False)

    app: Any = await create_app(_config(tmp_path), load_shim_sources=False)

    assert set(app.transports) == {"acp"}
    assert not _imported()
