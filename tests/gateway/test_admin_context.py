"""The admin context wiring: the adapter, its protocol and the config seam."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Generic, Protocol, runtime_checkable

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


def test_an_app_without_an_admin_context_is_refused():
    request = SimpleNamespace(app=SimpleNamespace())
    with pytest.raises(RuntimeError, match="app.admin"):
        admin_context(request)


def test_the_context_the_app_carries_is_the_one_returned():
    context = GatewayAdminContext()
    request = SimpleNamespace(app=SimpleNamespace(admin=context))
    assert admin_context(request) is context


def test_admin_api_does_not_reach_into_app():
    source = Path(admin_api.__file__).read_text()
    assert "app.admin_context" not in source
    assert "from .app" not in source


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


class TestReloadConfig:
    """``AdminContext.reload_config``: what a reload swaps in, besides the config."""

    @staticmethod
    async def _app(tmp_path, raw: dict[str, Any]) -> Any:
        app: Any = await create_app(GatewayConfig(_raw(tmp_path, "m-one")))
        app.config_io = SimpleNamespace(load=lambda _path: raw)
        return app

    @pytest.mark.asyncio
    async def test_it_serves_the_new_config(self, tmp_path):
        app = await self._app(tmp_path, _raw(tmp_path, "m-one", "m-two"))
        new = await app.admin.reload_config(str(tmp_path / "c.json"))
        assert app.admin.served_config() is new
        assert app.gateway_config is new
        assert sorted(new.models) == ["m-one", "m-two"]

    @pytest.mark.asyncio
    async def test_it_applies_the_trusted_proxies_to_the_live_policy(self, tmp_path):
        raw = _raw(tmp_path, "m-one")
        raw["server"]["trusted_proxies"] = ["172.29.0.1/32"]
        app = await self._app(tmp_path, raw)
        assert "172.29.0.1" not in app.proxy_policy.trusted
        await app.admin.reload_config(str(tmp_path / "c.json"))
        assert "172.29.0.1" in app.proxy_policy.trusted

    @pytest.mark.asyncio
    async def test_it_applies_the_retention_caps_to_the_store(self, tmp_path):
        raw = _raw(tmp_path, "m-one")
        raw["server"]["request_log"] = {"success_max": 321, "error_dump_max": 12}
        raw["server"]["ops_log"] = {"info_max": 1000, "warn_max": 5}
        app = await self._app(tmp_path, raw)
        app.persistence = SimpleNamespace(
            success_max=0, dump_max=0, retention=SimpleNamespace()
        )
        await app.admin.reload_config(str(tmp_path / "c.json"))
        assert app.persistence.success_max == 321
        assert app.persistence.dump_max == 12
        assert app.persistence.retention.ops_info_max == 1000
        # Below the floor of 100, the floor wins.
        assert app.persistence.retention.ops_warn_max == 100

    @pytest.mark.asyncio
    async def test_it_syncs_the_admin_password_and_sessions(self, tmp_path):
        raw = _raw(tmp_path, "m-one")
        raw["server"]["admin_password"] = "new-secret"
        app = await self._app(tmp_path, raw)
        calls: list[Any] = []
        app.auth_state = SimpleNamespace(
            admin_password="old-secret",
            change_password=lambda value: calls.append(("password", value)),
            configure_sessions=lambda **kw: calls.append(("sessions", kw)),
        )
        await app.admin.reload_config(str(tmp_path / "c.json"))
        assert ("password", "new-secret") in calls
        assert any(kind == "sessions" for kind, _ in calls)

    @pytest.mark.asyncio
    async def test_the_rate_limiter_is_rebuilt_only_when_its_settings_changed(
        self, tmp_path
    ):
        raw = _raw(tmp_path, "m-one")
        app = await self._app(tmp_path, raw)
        rebuilt: list[Any] = []
        app.rate_limit_state = SimpleNamespace(rebuild=rebuilt.append)
        await app.admin.reload_config(str(tmp_path / "c.json"))
        first = len(rebuilt)
        await app.admin.reload_config(str(tmp_path / "c.json"))
        assert first == 1 and len(rebuilt) == 1, "an unchanged config must not rebuild"
        raw["server"]["rate_limit"] = {"enabled": True, "global": "5/minute"}
        await app.admin.reload_config(str(tmp_path / "c.json"))
        assert len(rebuilt) == 2

    @pytest.mark.asyncio
    async def test_the_reload_is_recorded_in_the_ops_log(self, tmp_path):
        from llm_rosetta_gateway.observability.ops_log import EVENT_CONFIG_RELOAD

        app = await self._app(tmp_path, _raw(tmp_path, "m-one"))
        added: list[Any] = []

        async def add(entry: Any, **_kw: Any) -> None:
            added.append(entry)

        app.ops_log = SimpleNamespace(add=add)
        await app.admin.reload_config(str(tmp_path / "c.json"))
        assert [e.event_type for e in added] == [EVENT_CONFIG_RELOAD]

    @pytest.mark.asyncio
    async def test_an_app_with_no_config_reader_cannot_reload(self, tmp_path):
        app: Any = await create_app(GatewayConfig(_raw(tmp_path, "m-one")))
        app.config_io = None
        with pytest.raises(RuntimeError, match="ConfigIO"):
            await app.admin.reload_config(str(tmp_path / "c.json"))


class TestTheServedConfigHasOneHome:
    """The config an app serves is ``app.gateway_config``, nothing else (#1830 A1)."""

    @pytest.mark.asyncio
    async def test_create_app_puts_the_config_on_the_app(self, tmp_path):
        config = GatewayConfig(_raw(tmp_path, "m-one"))
        app: Any = await create_app(config)
        assert app.gateway_config is config
        assert app.admin.served_config() is config

    @pytest.mark.asyncio
    async def test_publish_replaces_it_for_the_routes_and_the_panel(self, tmp_path):
        app: Any = await create_app(GatewayConfig(_raw(tmp_path, "m-one")))
        new = GatewayConfig(_raw(tmp_path, "m-one", "m-two"))
        app.admin.publish_config(new)
        assert app.gateway_config is new
        assert app.admin.served_config() is new

    @pytest.mark.asyncio
    async def test_two_apps_in_one_process_serve_their_own_config(self, tmp_path):
        first: Any = await create_app(GatewayConfig(_raw(tmp_path, "m-one")))
        second: Any = await create_app(GatewayConfig(_raw(tmp_path, "m-two")))
        assert first.admin.served_config() is first.gateway_config
        assert second.admin.served_config() is second.gateway_config
        assert first.gateway_config is not second.gateway_config

    def test_there_is_no_second_home(self):
        import importlib.util

        assert importlib.util.find_spec("llm_rosetta_gateway.app._state") is None

    def test_an_adapter_with_no_app_serves_nothing(self):
        assert GatewayAdminContext().served_config() is None


class TestEachAppKeepsItsOwnLaneAllowance:
    """What the lanes report belongs to the app that serves them (#1830 A9)."""

    @staticmethod
    def _lane_raw(tmp_path, name):
        return {
            "providers": {
                name: {
                    "type": "open_responses",
                    "base_url": "https://example.test/v1",
                    "api_key": "k",
                }
            },
            "models": {"m": name},
            "server": {"data_dir": str(tmp_path)},
        }

    @pytest.mark.asyncio
    async def test_every_provider_of_the_served_config_reports_to_the_apps_store(
        self, tmp_path
    ):
        config = GatewayConfig(self._lane_raw(tmp_path, "lane"))
        app: Any = await create_app(config)
        assert config.providers["lane"].lane_allowance is app.lane_allowance

    @pytest.mark.asyncio
    async def test_a_reload_binds_the_new_providers_to_the_same_store(self, tmp_path):
        app: Any = await create_app(GatewayConfig(self._lane_raw(tmp_path, "lane")))
        store = app.lane_allowance
        rebuilt = GatewayConfig(self._lane_raw(tmp_path, "lane"))
        assert rebuilt.providers["lane"].lane_allowance is None
        app.admin.publish_config(rebuilt)
        assert rebuilt.providers["lane"].lane_allowance is store

    @pytest.mark.asyncio
    async def test_two_apps_do_not_share_a_reading(self, tmp_path):
        first: Any = await create_app(GatewayConfig(self._lane_raw(tmp_path, "lane")))
        second: Any = await create_app(GatewayConfig(self._lane_raw(tmp_path, "lane")))
        update = {
            "sessionUpdate": "usage_update",
            "_meta": {"_claude/rateLimit": {"utilization": 0.5}},
        }
        lane = first.gateway_config.providers["lane"]
        assert first.lane_allowance is not second.lane_allowance
        assert lane.record_lane_reading(update) is True
        assert lane.lane_reading() is not None
        assert second.gateway_config.providers["lane"].lane_reading() is None


class TestTheServicesTheObservabilityRoutesRead:
    """``AdminServiceState``: each member is the service the app holds, else ``None``."""

    _MEMBERS = {
        "persistence": "persistence",
        "metrics": "metrics",
        "request_log": "request_log",
        "ops_log": "ops_log",
        "job_manager": "jobs",
        "rate_limit_state": "rate_limit_state",
        "profiler_state": "profiler_state",
        "capture_state": "capture_state",
    }

    @pytest.mark.parametrize("member", sorted(_MEMBERS))
    def test_a_member_returns_what_the_app_holds(self, member):
        marker = object()
        app = SimpleNamespace(**{self._MEMBERS[member]: marker})
        assert getattr(GatewayAdminContext(app), member)() is marker

    @pytest.mark.parametrize("member", sorted(_MEMBERS))
    def test_a_member_is_none_when_the_app_has_no_such_service(self, member):
        assert getattr(GatewayAdminContext(SimpleNamespace()), member)() is None
        assert getattr(GatewayAdminContext(), member)() is None

    def test_the_admin_password_is_set_only_when_it_is_non_empty(self):
        def context(**state):
            return GatewayAdminContext(
                SimpleNamespace(auth_state=SimpleNamespace(**state))
            )

        assert context(admin_password="pw").admin_password_is_set() is True
        assert context(admin_password="").admin_password_is_set() is False
        assert context(admin_password=None).admin_password_is_set() is False
        assert GatewayAdminContext(SimpleNamespace()).admin_password_is_set() is False

    def test_the_context_implements_the_protocol(self):
        from llm_rosetta_gateway.admin_api import AdminServiceState

        assert isinstance(GatewayAdminContext(SimpleNamespace()), AdminServiceState)


class TestTheStateTheSharedRoutesRead:
    """``AdminRuntime`` paths and ``AdminShimSourceState``: the app's attribute, else the empty answer."""

    @pytest.mark.parametrize("member", ["config_path", "config_io"])
    def test_a_runtime_member_returns_what_the_app_holds(self, member):
        marker = object()
        app = SimpleNamespace(**{member: marker})
        assert getattr(GatewayAdminContext(app), member)() is marker

    @pytest.mark.parametrize("member", ["config_path", "config_io"])
    def test_a_runtime_member_is_none_without_the_attribute(self, member):
        assert getattr(GatewayAdminContext(SimpleNamespace()), member)() is None
        assert getattr(GatewayAdminContext(), member)() is None

    @pytest.mark.parametrize(
        "member", ["shim_source_report", "shim_source_signature", "shim_update_check"]
    )
    def test_a_shim_source_member_returns_what_the_app_holds(self, member):
        marker = object()
        app = SimpleNamespace(**{member: marker})
        assert getattr(GatewayAdminContext(app), member)() is marker
        assert getattr(GatewayAdminContext(SimpleNamespace()), member)() is None

    def test_installing_a_report_drops_the_verdicts_cached_for_the_old_one(self):
        app = SimpleNamespace(shim_update_check={"stale": True})
        context = GatewayAdminContext(app)

        context.install_shim_report("report", "sig")

        assert (context.shim_source_report(), context.shim_source_signature()) == (
            "report",
            "sig",
        )
        assert context.shim_update_check() is None

    def test_the_update_verdicts_can_be_cached_and_dropped(self):
        context = GatewayAdminContext(SimpleNamespace())

        context.set_shim_update_check({"ok": 1})
        assert context.shim_update_check() == {"ok": 1}
        context.set_shim_update_check(None)
        assert context.shim_update_check() is None

    def test_the_warnings_are_a_copy_and_empty_when_none_were_recorded(self):
        held = [{"provider": "p"}]
        context = GatewayAdminContext(
            SimpleNamespace(provider_validation_warnings=held)
        )

        assert context.provider_validation_warnings() == held
        assert context.provider_validation_warnings() is not held
        assert (
            GatewayAdminContext(SimpleNamespace()).provider_validation_warnings() == []
        )
