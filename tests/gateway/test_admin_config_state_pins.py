"""Characterisation pins for the config routes that read the app (#1830).

``admin/routes/config/*`` read the served config, the shim-source report, the
update-check cache, the two shim indexes, the validation warnings, the
transports and the auth state from ``request.app``, and write the update check
and the two indexes back to it.  Each test records what a route answers on the
current code, so moving those reads and writes behind ``AdminContext`` cannot
change an answer unseen.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from _admin_app import fake_app

from llm_rosetta_gateway.app.admin_context import GatewayAdminContext
from llm_rosetta_gateway.admin.routes.config import (
    extensions,
    payload,
    providers,
    settings,
    shim_files,
    shim_manifests,
    shim_source_catalog,
    shims,
    upstream,
)


def _request(app: Any, body: Any = None, **query: str) -> Any:
    return SimpleNamespace(
        app=app,
        query_params={key: [value] for key, value in query.items()},
        path_params={},
        json=lambda: {} if body is None else body,
        headers={},
        cookies={},
        method="POST",
    )


def _body(response: Any) -> Any:
    return json.loads(response.body)


class TestTheServedConfigTheRoutesHand:
    def test_the_payload_helper_returns_the_config_the_app_serves(self):
        config = object()

        assert payload._get_gateway_config(
            _request(fake_app(gateway_config=config))
        ) is (config)
        assert payload._get_gateway_config(_request(fake_app())) is None

    @pytest.mark.asyncio
    async def test_the_extension_update_check_is_given_the_config_and_report(self):
        seen: list[Any] = []

        class _Context(GatewayAdminContext):
            async def check_extension_updates(
                self, config: Any, report: Any, ids: list[str] | None = None
            ) -> list[dict[str, Any]]:
                seen.append((config, report, ids))
                return []

        app = fake_app(gateway_config="config", shim_source_report="report")
        app.admin = _Context(app)

        response = await extensions.post_extension_check_updates(
            _request(app, {"ids": ["x"]})
        )

        assert _body(response) == {"updates": []}
        assert seen == [("config", "report", ["x"])]

    @pytest.mark.asyncio
    async def test_the_extension_update_is_given_the_config_and_report(
        self, monkeypatch
    ):
        seen: list[Any] = []

        class _Context(GatewayAdminContext):
            async def update_extension(
                self,
                ext_id: str,
                config: Any,
                report: Any,
                version: str | None = None,
            ) -> dict[str, Any]:
                seen.append((ext_id, config, report, version))
                return {"extension": ext_id}

        async def reloaded(request: Any) -> None:
            return None

        monkeypatch.setattr(extensions, "_reload_shims", reloaded)
        app = fake_app(gateway_config="config", shim_source_report="report")
        app.admin = _Context(app)

        response = await extensions.post_extension_update(
            _request(app, {"version": "1.0"}), "ext"
        )

        assert _body(response) == {"extension": "ext", "reload_error": None}
        assert seen == [("ext", "config", "report", "1.0")]

    @pytest.mark.asyncio
    async def test_the_source_catalogue_is_given_the_config_and_report(self):
        seen: list[Any] = []

        class _Context(GatewayAdminContext):
            def shim_source_catalogue(
                self, config: Any, report: Any, *, refresh: bool = False
            ) -> dict[str, Any]:
                seen.append((config, report, refresh))
                return {"sources": []}

        app = fake_app(gateway_config="config", shim_source_report="report")
        app.admin = _Context(app)

        response = await shim_source_catalog.get_shim_source_catalog(
            _request(app, refresh="1")
        )

        assert _body(response) == {"sources": []}
        assert seen == [("config", "report", True)]


class TestTheShimStatus:
    def test_no_report_answers_the_empty_status(self):
        assert shims._shim_status_payload(_request(fake_app())) == {
            "sources": [],
            "total_shims": 0,
            "rebuilt_at": 0.0,
            "failure_count": 0,
        }

    def test_a_report_answers_its_own_dict(self):
        report = SimpleNamespace(to_dict=lambda: {"total_shims": 3})

        assert shims._shim_status_payload(
            _request(fake_app(shim_source_report=report))
        ) == {"total_shims": 3}


class TestTheUpdateCheck:
    @pytest.mark.asyncio
    async def test_no_report_answers_nothing_checked(self):
        response = await shims.post_shims_check_updates(_request(fake_app()))

        assert _body(response) == {"sources": [], "checked_at": 0.0}

    @pytest.mark.asyncio
    async def test_a_fresh_cached_verdict_is_served_without_a_lookup(self, monkeypatch):
        cached = {"sources": ["cached"], "checked_at": 1e12}
        app = fake_app(shim_source_report=object(), shim_update_check=cached)

        response = await shims.post_shims_check_updates(_request(app))

        assert _body(response) == cached

    @pytest.mark.asyncio
    async def test_a_lookup_goes_through_the_configured_proxy_and_is_cached(
        self, monkeypatch
    ):
        calls: list[Any] = []

        class _Catalogue:
            def check_source_updates(self, report, *, proxy=None):
                calls.append((report, proxy))
                return ["looked-up"]

        monkeypatch.setattr(shims, "catalogue", lambda: _Catalogue())
        report = object()
        app = fake_app(
            shim_source_report=report,
            gateway_config=SimpleNamespace(proxy="http://p:1"),
            shim_update_check={"sources": ["stale"], "checked_at": 0.0},
        )

        response = await shims.post_shims_check_updates(_request(app))

        assert _body(response)["sources"] == ["looked-up"]
        assert calls == [(report, "http://p:1")]
        assert app.shim_update_check["sources"] == ["looked-up"]

    @pytest.mark.asyncio
    async def test_a_forced_check_ignores_a_fresh_cache(self, monkeypatch):
        class _Catalogue:
            def check_source_updates(self, report, *, proxy=None):
                return ["forced"]

        monkeypatch.setattr(shims, "catalogue", lambda: _Catalogue())
        app = fake_app(
            shim_source_report=object(),
            shim_update_check={"sources": ["cached"], "checked_at": 1e12},
        )

        response = await shims.post_shims_check_updates(
            _request(app, {"refresh": True})
        )

        assert _body(response)["sources"] == ["forced"]


class TestTheWarningsAReloadReports:
    @pytest.mark.asyncio
    async def test_the_config_reload_answers_the_recorded_warnings(self, monkeypatch):
        async def reloaded(request: Any, path: str, **kw: Any) -> Any:
            return SimpleNamespace(providers={"p": 1}, models={"m": 1})

        monkeypatch.setattr(shims, "_reload_gateway_config", reloaded)
        app = fake_app(
            config_path="/c.json", provider_validation_warnings=[{"provider": "p"}]
        )

        response = await shims.reload_config(_request(app))

        assert _body(response)["provider_warnings"] == [{"provider": "p"}]

    @pytest.mark.asyncio
    async def test_the_config_reload_with_no_warnings_answers_an_empty_list(
        self, monkeypatch
    ):
        async def reloaded(request: Any, path: str, **kw: Any) -> Any:
            return SimpleNamespace(providers={}, models={})

        monkeypatch.setattr(shims, "_reload_gateway_config", reloaded)

        response = await shims.reload_config(_request(fake_app(config_path="/c.json")))

        assert _body(response)["provider_warnings"] == []

    @pytest.mark.asyncio
    async def test_the_shim_reload_answers_the_recorded_warnings(self, monkeypatch):
        async def reloaded(request: Any, path: str, **kw: Any) -> Any:
            return SimpleNamespace(providers={"p": 1})

        monkeypatch.setattr(shims, "_reload_gateway_config", reloaded)
        app = fake_app(
            config_path="/c.json", provider_validation_warnings=[{"provider": "p"}]
        )

        response = await shims.post_shims_reload(_request(app))

        body = _body(response)
        assert body["provider_warnings"] == [{"provider": "p"}]
        assert body["status"]["total_shims"] == 0


class TestTheShimIndexes:
    def test_the_file_index_is_built_once_per_rebuild(self, monkeypatch, tmp_path):
        walks: list[int] = []

        def roots(request: Any) -> list[Any]:
            walks.append(1)
            return [("builtin", "", tmp_path / "nowhere")]

        monkeypatch.setattr(shim_files, "_shim_roots", roots)
        app = fake_app(shim_source_report=SimpleNamespace(rebuilt_at=1.0))

        first = shim_files._shim_index(_request(app))
        second = shim_files._shim_index(_request(app))

        assert second is first
        assert len(walks) == 1

    def test_a_new_rebuild_builds_the_file_index_again(self, monkeypatch, tmp_path):
        walks: list[int] = []

        def roots(request: Any) -> list[Any]:
            walks.append(1)
            return []

        monkeypatch.setattr(shim_files, "_shim_roots", roots)
        app = fake_app(shim_source_report=SimpleNamespace(rebuilt_at=1.0))
        shim_files._shim_index(_request(app))

        app.shim_source_report = SimpleNamespace(rebuilt_at=2.0)
        shim_files._shim_index(_request(app))

        assert len(walks) == 2

    def test_the_manifest_index_is_built_once_per_rebuild(self, monkeypatch):
        walks: list[int] = []

        def roots(request: Any) -> list[Any]:
            walks.append(1)
            return []

        monkeypatch.setattr(shim_manifests, "_shim_roots", roots)
        app = fake_app(shim_source_report=SimpleNamespace(rebuilt_at=1.0))

        first = shim_manifests._manifest_entries(_request(app))
        second = shim_manifests._manifest_entries(_request(app))

        assert second is first
        assert len(walks) == 1

    def test_a_new_rebuild_builds_the_manifest_index_again(self, monkeypatch):
        walks: list[int] = []

        def roots(request: Any) -> list[Any]:
            walks.append(1)
            return []

        monkeypatch.setattr(shim_manifests, "_shim_roots", roots)
        app = fake_app(shim_source_report=SimpleNamespace(rebuilt_at=1.0))
        shim_manifests._manifest_entries(_request(app))

        app.shim_source_report = SimpleNamespace(rebuilt_at=2.0)
        shim_manifests._manifest_entries(_request(app))

        assert len(walks) == 2

    def test_the_roots_follow_the_report_the_app_holds(self, tmp_path):
        source = SimpleNamespace(
            kind="local", location="here", ok=True, resolved_dir=str(tmp_path)
        )
        app = fake_app(shim_source_report=SimpleNamespace(sources=[source]))

        roots = shim_files._shim_roots(_request(app))

        assert [(origin, location) for origin, location, _ in roots][1:] == [
            ("local", "here")
        ]
        assert len(shim_files._shim_roots(_request(fake_app()))) == 1


class TestTheTransportsALaneRenameCloses:
    @pytest.mark.asyncio
    async def test_each_transport_closes_the_named_lanes_once(self):
        closed: list[str] = []

        class _Transport:
            async def aclose_provider(self, name: str) -> None:
                closed.append(name)

        shared = _Transport()
        app = fake_app(transports={"a": shared, "b": shared, "c": object()})

        await providers._close_lane_processes(_request(app), ["x", "y"])

        assert closed == ["x", "y"]

    @pytest.mark.asyncio
    async def test_an_app_with_no_transports_closes_nothing(self):
        await providers._close_lane_processes(_request(fake_app()), ["x"])


class TestTheAdminPasswordTheSettingsGate:
    @pytest.mark.asyncio
    async def test_revealing_credentials_needs_an_admin_password(self, tmp_path):
        class _Io:
            def load_raw(self, path: str) -> dict[str, Any]:
                return {}

            def save(self, path: str, data: dict[str, Any]) -> None:
                self.saved = data

        auth = SimpleNamespace(admin_password="")
        app = fake_app(
            config_path=str(tmp_path / "c.json"), config_io=_Io(), auth_state=auth
        )

        refused = await settings.put_server_settings(
            _request(app, {"credential_visible": True})
        )
        auth.admin_password = "secret"

        assert (refused.status_code, _body(refused)) == (
            403,
            {"error": "Credential reveal requires an admin password"},
        )
        # With a password set the gate no longer refuses (the later reload may).
        allowed = await settings.put_server_settings(
            _request(app, {"credential_visible": True})
        )
        assert allowed.status_code != 403


class TestTheConfigTheRoutingLoopCheckReads:
    """A running gateway keeps its config on ``gateway_config``; nothing sets ``config``."""

    @pytest.mark.asyncio
    async def test_the_served_config_is_checked_for_loops(self):
        config = SimpleNamespace(
            models={
                "a": SimpleNamespace(providers=[]),
                "b": SimpleNamespace(providers=[]),
            },
            model_upstream_names={"a": "b"},
        )

        response = await upstream.check_routing_loops(
            _request(fake_app(gateway_config=config))
        )

        [warning] = _body(response)["warnings"]
        assert "a" in warning and "b" in warning

    @pytest.mark.asyncio
    async def test_an_app_with_no_config_reports_no_loops(self):
        response = await upstream.check_routing_loops(_request(fake_app()))

        assert _body(response) == {"warnings": []}


class TestTheConfigPayloadState:
    @staticmethod
    async def _get_config(monkeypatch: Any, **state: Any) -> Any:
        monkeypatch.setattr(payload, "_get_config_path", lambda request: "/c.jsonc")
        monkeypatch.setattr(
            payload,
            "_get_config_io",
            lambda request: SimpleNamespace(load_raw=lambda path: {"providers": {}}),
        )
        monkeypatch.setattr(payload, "_shim_status_payload", lambda request: {})
        config = SimpleNamespace(
            providers={},
            credential_visible=False,
            api_keys_db=None,
            open_on_no_keys=True,
            admin_session_idle_minutes=30,
            admin_session_remember_days=30,
        )
        app = fake_app(gateway_config=config, **state)
        return _body(await payload.get_config(_request(app)))

    @pytest.mark.asyncio
    async def test_the_resolved_switch_comes_from_the_served_config(self, monkeypatch):
        body = await self._get_config(
            monkeypatch, auth_state=SimpleNamespace(admin_password=None)
        )

        assert body["server"]["open_on_no_keys"] is True

    @pytest.mark.asyncio
    async def test_the_payload_says_whether_an_admin_password_is_set(self, monkeypatch):
        with_password = await self._get_config(
            monkeypatch, auth_state=SimpleNamespace(admin_password="secret")
        )
        without = await self._get_config(
            monkeypatch, auth_state=SimpleNamespace(admin_password="")
        )

        assert with_password["has_admin_password"] is True
        assert without["has_admin_password"] is False
