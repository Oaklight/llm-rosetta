"""The admin context wiring: the adapter, its protocol and the config seam."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any, Generic, Protocol, runtime_checkable
from unittest.mock import MagicMock

import pytest

from llm_rosetta_gateway import admin_api
from llm_rosetta_gateway.admin_api import AdminContext, admin_context
from llm_rosetta_gateway.app import create_app
from llm_rosetta_gateway.app.admin_context import GatewayAdminContext
from llm_rosetta_gateway.config import GatewayConfig


def _raw(tmp_path, *models: str) -> dict[str, Any]:
    return {
        "providers": {
            "Mock": {
                "type": "chat_completions",
                "base_url": "http://127.0.0.1:1/v1",
                "api_key": "x",
            }
        },
        "models": {name: {"provider": "Mock"} for name in models},
        "server": {"data_dir": str(tmp_path / "data"), "open_on_no_keys": True},
    }


def _protocol_members(protocol: type) -> set[str]:
    # Walk the MRO rather than read ``__protocol_attrs__``: that name is private
    # and absent on some supported Pythons.
    return {
        name
        for klass in protocol.__mro__
        if klass not in (Protocol, Generic, object)
        for name, value in vars(klass).items()
        if callable(value) and not name.startswith("_")
    }


def test_adapter_satisfies_the_protocol():
    assert isinstance(GatewayAdminContext(), AdminContext)


def test_adapter_implements_every_protocol_member():
    # ``isinstance`` against a runtime protocol checks member presence only, so
    # this derives the list from the protocol and names what is missing.
    members = _protocol_members(AdminContext)
    assert {"served_config", "publish_config"} <= members
    missing = sorted(
        m for m in members if not callable(getattr(GatewayAdminContext, m, None))
    )
    assert missing == [], f"GatewayAdminContext lacks AdminContext members: {missing}"


def test_a_protocol_member_the_adapter_lacks_is_noticed():
    @runtime_checkable
    class Wider(AdminContext, Protocol):
        def brand_new_member(self) -> None: ...

    assert not isinstance(GatewayAdminContext(), Wider)
    assert "brand_new_member" in _protocol_members(Wider)


@pytest.mark.asyncio
async def test_create_app_wires_the_admin_context(tmp_path):
    app: Any = await create_app(GatewayConfig(_raw(tmp_path, "m-one")))
    assert isinstance(app.admin, AdminContext)
    assert isinstance(app.admin, GatewayAdminContext)


def test_a_hand_built_app_gets_the_stock_adapter_and_one_warning(monkeypatch):
    log = MagicMock()
    monkeypatch.setattr(admin_api, "logger", log)
    monkeypatch.setattr(admin_api, "_fallback_reported", False)
    request = SimpleNamespace(app=SimpleNamespace())
    assert isinstance(admin_context(request), GatewayAdminContext)
    assert isinstance(admin_context(request), GatewayAdminContext)
    assert log.warning.call_count == 1
    wired = SimpleNamespace(app=SimpleNamespace(admin=GatewayAdminContext()))
    admin_context(wired)
    assert log.warning.call_count == 1


@pytest.mark.asyncio
async def test_published_config_is_the_one_served(tmp_path):
    app: Any = await create_app(GatewayConfig(_raw(tmp_path, "m-one")))
    new = GatewayConfig(_raw(tmp_path, "m-one", "m-two"))
    assert app.admin.served_config() is not new
    app.admin.publish_config(new)
    assert app.admin.served_config() is new


async def _get(port: int, path: str) -> dict[str, Any]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(
        f"GET {path} HTTP/1.1\r\nHost: t\r\nConnection: close\r\n\r\n".encode()
    )
    await writer.drain()
    writer.write_eof()
    data = await reader.read()
    writer.close()
    return json.loads(data.partition(b"\r\n\r\n")[2])


@pytest.mark.asyncio
async def test_admin_reload_changes_what_v1_models_serves(tmp_path):
    from llm_rosetta_gateway.admin.routes._shared import _reload_gateway_config

    app: Any = await create_app(GatewayConfig(_raw(tmp_path, "m-one")))
    reloaded = _raw(tmp_path, "m-one", "m-two")
    app.config_io = SimpleNamespace(load=lambda _path: reloaded)
    listener = await asyncio.start_server(app._handle_connection, "127.0.0.1", 0)
    port = listener.sockets[0].getsockname()[1]
    try:
        before = await _get(port, "/v1/models")
        assert [m["id"] for m in before["data"]] == ["m-one"]
        await _reload_gateway_config(SimpleNamespace(app=app), str(tmp_path / "c.json"))
        after = await _get(port, "/v1/models")
        assert sorted(m["id"] for m in after["data"]) == ["m-one", "m-two"]
    finally:
        listener.close()
        await listener.wait_closed()
