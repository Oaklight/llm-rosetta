"""Characterisation pins for the shared admin helpers that read the app (#1830).

``admin/routes/_shared.py``, ``logos.py`` and ``updates.py`` read the disabled
tabs, the config file path and adapter, the shim-source report, signature and
update check, the provider validation warnings and the served config from
``request.app``.  Each test records what a helper answers on the current code,
so moving those reads behind ``AdminContext`` cannot change an answer unseen.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from _admin_app import fake_app

from llm_rosetta_gateway.admin.routes import _shared, logos, updates


def _request(app: Any, **query: str) -> Any:
    return SimpleNamespace(
        app=app, query_params={key: [value] for key, value in query.items()}
    )


async def _handler(request: Any, **kw: Any) -> str:
    return "ran"


class TestTheTabGuard:
    @pytest.mark.asyncio
    async def test_a_disabled_tab_answers_404(self):
        app = fake_app(disabled_tabs=["models"])

        response = await _shared._guard("models", _handler)(_request(app))

        assert response.status_code == 404
        assert json.loads(response.body) == {"error": "Tab disabled", "disabled": True}

    @pytest.mark.asyncio
    async def test_another_tab_runs_the_handler(self):
        app = fake_app(disabled_tabs=["models"])

        assert await _shared._guard("keys", _handler)(_request(app)) == "ran"

    @pytest.mark.asyncio
    async def test_an_app_with_no_disabled_tabs_runs_the_handler(self):
        assert await _shared._guard("models", _handler)(_request(fake_app())) == "ran"


class TestTheConfigFileHandles:
    def test_the_path_is_the_one_the_app_holds(self):
        assert _shared._get_config_path(_request(fake_app(config_path="/c.json"))) == (
            "/c.json"
        )

    def test_no_path_raises(self):
        with pytest.raises(RuntimeError, match="No config file path"):
            _shared._get_config_path(_request(fake_app()))

    def test_the_adapter_is_the_one_the_app_holds(self):
        io = object()

        assert _shared._get_config_io(_request(fake_app(config_io=io))) is io

    def test_no_adapter_raises(self):
        with pytest.raises(RuntimeError, match="No ConfigIO adapter"):
            _shared._get_config_io(_request(fake_app()))


class TestTheShimReport:
    def test_installing_a_report_replaces_it_and_drops_the_update_verdicts(self):
        app = fake_app(
            shim_source_report="old",
            shim_source_signature="old-sig",
            shim_update_check={"stale": True},
        )

        _shared._install_shim_report(_request(app), "new", "new-sig")

        assert app.shim_source_report == "new"
        assert app.shim_source_signature == "new-sig"
        assert app.shim_update_check is None


class _Io:
    def __init__(self) -> None:
        self.saved: list[Any] = []

    def load_raw(self, path: str) -> dict[str, Any]:
        return {"providers": {}}

    def save(self, path: str, data: dict[str, Any]) -> None:
        self.saved.append((path, data))


class TestTheWarningsAConfigSaveReports:
    @pytest.mark.asyncio
    async def test_a_committed_mutation_carries_the_warnings_the_reload_left(
        self, monkeypatch, tmp_path
    ):
        async def reload(request: Any, config_path: str, **kw: Any) -> str:
            request.app.provider_validation_warnings = [{"provider": "p"}]
            return "config"

        monkeypatch.setattr(_shared, "_reload_gateway_config", reload)
        path = str(tmp_path / "c.json")
        app = fake_app(config_path=path, config_io=_Io())

        async with _shared.config_mutate(_request(app)) as ctx:
            ctx.commit()

        assert ctx.new_config == "config"
        assert ctx.warnings == [{"provider": "p"}]
        assert app.config_io.saved == [(path, {"providers": {}})]

    @pytest.mark.asyncio
    async def test_no_recorded_warnings_gives_an_empty_list(
        self, monkeypatch, tmp_path
    ):
        async def reload(request: Any, config_path: str, **kw: Any) -> str:
            return "config"

        monkeypatch.setattr(_shared, "_reload_gateway_config", reload)
        path = str(tmp_path / "c.json")
        app = fake_app(config_path=path, config_io=_Io())

        async with _shared.config_mutate(_request(app)) as ctx:
            ctx.commit()

        assert ctx.warnings == []


class _Offline:
    def http_client(self, **options: Any) -> Any:
        raise OSError("offline")


class TestTheLogoSources:
    @pytest.mark.asyncio
    async def test_a_host_the_config_names_is_a_source(self, monkeypatch):
        monkeypatch.setattr(logos, "catalogue", lambda: _Offline())
        config = SimpleNamespace(
            _raw_providers={"p": {"logo": "https://cdn.example.org/p.svg"}}
        )
        app = fake_app(config=config)

        response = await logos.get_logo(
            _request(app, src="https://cdn.example.org/p.svg")
        )

        assert response.status_code == 502

    @pytest.mark.asyncio
    async def test_a_host_nothing_names_is_refused(self, monkeypatch):
        monkeypatch.setattr(logos, "catalogue", lambda: _Offline())
        config = SimpleNamespace(
            _raw_providers={"p": {"logo": "https://cdn.example.org/p.svg"}}
        )
        app = fake_app(config=config)

        response = await logos.get_logo(
            _request(app, src="https://elsewhere.example.net/p.svg")
        )

        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_an_app_with_no_config_names_no_hosts(self, monkeypatch):
        monkeypatch.setattr(logos, "catalogue", lambda: _Offline())

        response = await logos.get_logo(
            _request(fake_app(), src="https://cdn.example.org/p.svg")
        )

        assert response.status_code == 403


class _Redirected:
    """A client whose answer came from another host than the one asked for."""

    def __init__(self, url: str) -> None:
        self._url = url

    def http_client(self, **options: Any) -> Any:
        url = self._url

        class _Client:
            async def __aenter__(self) -> Any:
                return self

            async def __aexit__(self, *exc: Any) -> None:
                return None

            async def get(self, src: str) -> Any:
                return SimpleNamespace(
                    url=url,
                    status_code=200,
                    content=b"<svg/>",
                    headers={"content-type": "image/svg+xml"},
                )

        return _Client()

    def is_http_response(self, value: Any) -> bool:
        return True


class TestAnAllowedHostThatRedirects:
    @pytest.mark.asyncio
    async def test_the_url_the_bytes_came_from_is_checked_again(self, monkeypatch):
        monkeypatch.setattr(
            logos, "catalogue", lambda: _Redirected("https://elsewhere.example.net/x")
        )

        response = await logos.get_logo(
            _request(fake_app(), src="https://cdn.jsdelivr.net/x.svg")
        )

        assert response.status_code == 403

    @pytest.mark.asyncio
    async def test_an_answer_from_the_allowed_host_is_served(self, monkeypatch):
        monkeypatch.setattr(
            logos, "catalogue", lambda: _Redirected("https://cdn.jsdelivr.net/x.svg")
        )
        logos._cache.clear()

        response = await logos.get_logo(
            _request(fake_app(), src="https://cdn.jsdelivr.net/x.svg")
        )

        assert (response.status_code, response.body) == (200, b"<svg/>")
        logos._cache.clear()


class TestTheUpdateSwitch:
    @pytest.mark.asyncio
    async def test_a_config_that_turns_the_check_off_answers_disabled(self):
        app = fake_app(config=SimpleNamespace(update_check=False))

        response = await updates.get_updates(_request(app))

        assert json.loads(response.body) == {"checked": False, "reason": "disabled"}


class TestTheConfigAnAppServes:
    """A running gateway keeps its config on ``gateway_config``; nothing sets ``config``."""

    @pytest.mark.asyncio
    async def test_the_update_switch_is_not_read_from_the_served_config(self):
        app = fake_app(gateway_config=SimpleNamespace(update_check=False))

        response = await updates.get_updates(_request(app))

        assert json.loads(response.body) != {"checked": False, "reason": "disabled"}

    @pytest.mark.asyncio
    async def test_a_logo_host_the_served_config_names_is_not_a_source(
        self, monkeypatch
    ):
        monkeypatch.setattr(logos, "catalogue", lambda: _Offline())
        config = SimpleNamespace(
            _raw_providers={"p": {"logo": "https://cdn.example.org/p.svg"}}
        )
        app = fake_app(gateway_config=config)

        response = await logos.get_logo(
            _request(app, src="https://cdn.example.org/p.svg")
        )

        assert response.status_code == 403
