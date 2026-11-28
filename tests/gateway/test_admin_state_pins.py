"""Characterisation pins for the admin routes that read service state off the app (#1830).

``admin/routes/{observability,profiling,capture,routing,maintenance,jobs}.py``
read the metrics, persistence, request log, ops log, job manager, rate-limit,
profiler and capture state, the served config and the auth state from
``request.app``.  Each test records what a route answers on the current code, with
stub services that name the call they receive, so moving those reads behind
``AdminContext`` cannot change an answer unseen.  The apps are built with
``fake_app``, which gives them the admin context the migrated routes need.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest
from _admin_app import fake_app

from llm_rosetta_gateway.admin.routes import (
    jobs as jobs_routes,
    maintenance,
    observability,
)


def _request(app: Any, *, query: dict[str, str] | None = None, **attrs: Any) -> Any:
    fields: dict[str, Any] = dict(
        app=app,
        query_params={key: [value] for key, value in (query or {}).items()},
        path_params={},
        json=lambda: {},
        headers={},
        cookies={},
        client_addr=("127.0.0.1", 5000),
        method="GET",
    )
    fields.update(attrs)
    return SimpleNamespace(**fields)


def _body(response: Any) -> Any:
    return json.loads(response.body)


class _Metrics:
    def snapshot(self, series_seconds: int = 60) -> dict[str, Any]:
        return {"series_seconds": series_seconds}


class _Persistence:
    db_path = "/nonexistent/gateway.db"

    async def query_rolling_24h_tokens(self) -> dict[str, int]:
        return {"tokens_24h": 7}

    async def query_request_histogram(self, minutes: int, buckets: int) -> dict:
        return {"minutes": minutes, "buckets": buckets}

    async def query_token_usage_by_day(self, **kwargs: Any) -> dict[str, Any]:
        return {"days": [kwargs], "totals": {}}


class _Registry:
    def __init__(self, enabled: bool) -> None:
        self.enabled = enabled

    def get_all_states(self) -> dict[str, str]:
        return {"p": "closed"}


def _config(enabled: bool = False, **more: Any) -> Any:
    return SimpleNamespace(
        circuit_breaker_registry=_Registry(enabled), provider_types={}, **more
    )


# -- metrics, histogram, circuit breakers, rate limits, token usage ---------------


@pytest.mark.asyncio
async def test_metrics_clamp_the_window_and_leave_out_persistence_without_one() -> None:
    app = fake_app(metrics=_Metrics(), gateway_config=_config())
    response = await observability.get_metrics(_request(app, query={"seconds": "999"}))
    assert _body(response) == {"series_seconds": 300}


@pytest.mark.asyncio
async def test_metrics_add_the_rolling_tokens_and_the_breaker_states() -> None:
    app = fake_app(
        metrics=_Metrics(),
        persistence=_Persistence(),
        gateway_config=_config(enabled=True),
    )
    body = _body(await observability.get_metrics(_request(app)))
    assert body["tokens_24h"] == 7
    assert body["circuit_breakers"] == {"p": "closed"}
    assert body["series_seconds"] == 60


@pytest.mark.asyncio
async def test_metrics_without_a_served_config_have_no_breaker_section() -> None:
    app = fake_app(metrics=_Metrics(), gateway_config=None)
    assert "circuit_breakers" not in _body(
        await observability.get_metrics(_request(app))
    )


@pytest.mark.asyncio
async def test_histogram_without_persistence_is_404() -> None:
    response = await observability.get_request_histogram(_request(fake_app()))
    assert response.status_code == 404
    assert _body(response) == {"error": "Persistence is disabled"}


@pytest.mark.asyncio
async def test_histogram_passes_the_window_to_persistence() -> None:
    app = fake_app(persistence=_Persistence())
    response = await observability.get_request_histogram(
        _request(app, query={"minutes": "30", "buckets": "6"})
    )
    assert _body(response) == {"minutes": 30, "buckets": 6}
    bad = await observability.get_request_histogram(
        _request(app, query={"minutes": "x"})
    )
    assert bad.status_code == 400


@pytest.mark.asyncio
async def test_circuit_breaker_states_follow_the_served_config() -> None:
    off = await observability.get_circuit_breaker_states(
        _request(fake_app(gateway_config=None))
    )
    assert _body(off) == {"enabled": False, "providers": {}}
    on = await observability.get_circuit_breaker_states(
        _request(fake_app(gateway_config=_config(enabled=True)))
    )
    assert _body(on) == {"enabled": True, "providers": {"p": "closed"}}


@pytest.mark.asyncio
async def test_rate_limit_status_peeks_the_key_it_is_given() -> None:
    class _State:
        def snapshot(self, key: str) -> dict[str, str]:
            return {"key": key}

    assert _body(await observability.get_rate_limit_status(_request(fake_app()))) == {
        "enabled": False,
        "dimensions": {},
    }
    app = fake_app(rate_limit_state=_State())
    assert _body(await observability.get_rate_limit_status(_request(app))) == {
        "key": "__global__"
    }
    assert _body(
        await observability.get_rate_limit_status(_request(app, query={"key": "k1"}))
    ) == {"key": "k1"}


@pytest.mark.asyncio
async def test_token_usage_clamps_days_and_is_empty_without_persistence() -> None:
    empty = await observability.get_token_usage(_request(fake_app()))
    assert _body(empty) == {"days": [], "totals": {}}
    app = fake_app(persistence=_Persistence())
    body = _body(
        await observability.get_token_usage(
            _request(app, query={"days": "9999", "api_key_label": "a"})
        )
    )
    assert body["days"] == [{"days": 365, "api_key_label": "a", "model_type": None}]


# -- request log ---------------------------------------------------------------------


class _Log:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def get_api_key_labels(self) -> list[str]:
        return ["alice"]

    async def get_entries(self, **kwargs: Any) -> tuple[list[dict], int]:
        self.calls.append(kwargs)
        return [{"id": "e1"}], 1

    async def get_entry(self, entry_id: str) -> dict[str, str] | None:
        return {"id": entry_id} if entry_id == "e1" else None


@pytest.mark.asyncio
async def test_key_labels_and_single_entries_come_from_the_request_log() -> None:
    app = fake_app(request_log=_Log())
    assert _body(await observability.get_request_key_labels(_request(app))) == {
        "labels": ["alice"]
    }
    found = await observability.get_request_by_id(
        _request(app, path_params={"entry_id": "e1"})
    )
    assert _body(found) == {"id": "e1"}
    missing = await observability.get_request_by_id(
        _request(app, path_params={"entry_id": "zz"})
    )
    assert (missing.status_code, _body(missing)) == (404, {"error": "Not found"})


@pytest.mark.asyncio
async def test_a_provider_filter_is_resolved_to_its_type_through_the_served_config() -> (
    None
):
    log = _Log()
    config = _config()
    config.provider_types = {"lane": "messages"}
    app = fake_app(request_log=log, gateway_config=config)
    response = await observability.get_requests(
        _request(app, query={"provider": "lane", "limit": "5"})
    )
    assert _body(response) == {"entries": [{"id": "e1"}], "total": 1}
    assert log.calls[0]["provider"] == "lane"
    assert log.calls[0]["provider_type"] == "messages"
    assert log.calls[0]["limit"] == 5


# -- provider key reveal, diagnostics -------------------------------------------------


def _reveal_app(tmp_path: Any, *, password: str | None, visible: bool) -> Any:
    from llm_rosetta_gateway.config import JsoncConfigIO

    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"providers": {"p": {"api_key": "sk-secret"}}}), encoding="utf-8"
    )
    return fake_app(
        gateway_config=SimpleNamespace(credential_visible=visible),
        auth_state=SimpleNamespace(admin_password=password),
        config_path=str(path),
        config_io=JsoncConfigIO(),
    )


@pytest.mark.asyncio
async def test_the_provider_key_needs_an_admin_password(tmp_path: Any) -> None:
    app = _reveal_app(tmp_path, password=None, visible=True)
    response = await observability.get_provider_key(
        _request(app, path_params={"name": "p"})
    )
    assert response.status_code == 403
    assert _body(response) == {"error": "Credential reveal requires an admin password"}


@pytest.mark.asyncio
async def test_the_provider_key_needs_credential_visibility(tmp_path: Any) -> None:
    app = _reveal_app(tmp_path, password="pw", visible=False)
    response = await observability.get_provider_key(
        _request(app, path_params={"name": "p"})
    )
    assert response.status_code == 403
    assert _body(response) == {"error": "Credential visibility is disabled"}


@pytest.mark.asyncio
async def test_the_provider_key_is_revealed_when_both_allow_it(tmp_path: Any) -> None:
    app = _reveal_app(tmp_path, password="pw", visible=True)
    ok = await observability.get_provider_key(_request(app, path_params={"name": "p"}))
    assert _body(ok) == {"api_key": "sk-secret"}
    gone = await observability.get_provider_key(
        _request(app, path_params={"name": "nope"})
    )
    assert gone.status_code == 404


@pytest.mark.asyncio
async def test_diagnostics_use_the_served_configs_proxy(monkeypatch) -> None:
    seen: list[dict[str, Any]] = []

    class _Client:
        def __init__(self, **kwargs: Any) -> None:
            seen.append(kwargs)

        async def __aenter__(self) -> _Client:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def get(self, url: str) -> Any:
            raise OSError("offline")

    from llm_rosetta_gateway import admin_api

    monkeypatch.setattr(
        admin_api.catalogue(), "http_client", lambda **kw: _Client(**kw), raising=False
    )
    app = fake_app(gateway_config=SimpleNamespace(proxy="http://proxy.local:3128"))
    body = _body(await observability.network_diagnostics(_request(app)))
    assert body["proxy"] == "http://proxy.local:3128"
    assert seen and seen[0]["proxy"] == "http://proxy.local:3128"
    assert body["ip"] == {"ok": False, "error": "offline"}


# -- ops log, maintenance, jobs --------------------------------------------------------


@pytest.mark.asyncio
async def test_the_ops_log_is_empty_without_one_and_filters_with_one() -> None:
    empty = await observability.get_ops_log(_request(fake_app()))
    assert _body(empty) == {"entries": [], "total": 0}

    class _Ops:
        async def get_entries(self, **kwargs: Any) -> tuple[list, int]:
            return [kwargs], 1

    app = fake_app(ops_log=_Ops())
    body = _body(
        await observability.get_ops_log(
            _request(app, query={"source": "keys", "limit": "2"})
        )
    )
    assert body["entries"][0]["source"] == "keys"
    assert body["entries"][0]["limit"] == 2


@pytest.mark.asyncio
async def test_the_backfill_buttons_need_persistence() -> None:
    for handler in (
        maintenance.backfill_keys_last_used,
        maintenance.backfill_dump_log_ids,
    ):
        response = await handler(_request(fake_app()))
        assert response.status_code == 400
        assert _body(response) == {"error": "No persistence configured"}


@pytest.mark.asyncio
async def test_the_job_routes_say_when_there_is_no_manager() -> None:
    response = await jobs_routes.get_jobs(_request(fake_app()))
    assert response.status_code == 400
    assert _body(response) == {"error": "Background jobs require persistence"}
    cleared = await observability.clear_requests(_request(fake_app()))
    assert cleared.status_code == 400
