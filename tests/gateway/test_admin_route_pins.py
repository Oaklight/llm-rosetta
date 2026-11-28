"""Characterisation pins for admin route lines no other test reaches.

Each test records what a route answers today: the status code, the body and
the side effect (an ops-log row or a state change).  They exist so that moving
the routes behind ``admin_api`` cannot change a response unseen.  Handlers are
called directly with a ``SimpleNamespace`` request, as the neighbouring admin
tests do.
"""

from __future__ import annotations

import json
import zlib
from types import SimpleNamespace
from typing import Any

import pytest
from _admin_app import fake_app

from llm_rosetta_gateway._vendor import httpclient
from llm_rosetta_gateway.admin.routes import (
    auth,
    capture,
    keys,
    observability,
    profiling,
    quota,
    updates as updates_routes,
)
from llm_rosetta_gateway.admin.routes.config import model_providers, settings, upstream
from llm_rosetta_gateway.config import JsoncConfigIO
from llm_rosetta_gateway.jobs import JobManager
from llm_rosetta_gateway.keystore import KeyStore
from llm_rosetta_gateway.middleware.auth import ADMIN_COOKIE_NAME, AuthState
from llm_rosetta_gateway.observability import CaptureState, ProfilerState
from llm_rosetta_gateway.observability.capture import CapturedRequest
from llm_rosetta_gateway.observability.ops_log import (
    ALL_EVENT_TYPES,
    ALL_SOURCES,
    EVENT_ADMIN_SETUP,
    EVENT_KEY_CREATE,
    EVENT_KEY_DELETE,
    EVENT_KEY_ROTATE,
    EVENT_KEY_UPDATE,
    EVENT_PASSWORD_CHANGED,
    EVENT_SELF_UPDATE,
    EVENT_SESSION_LOGOUT_ALL,
    EVENT_TOKEN_ROTATED,
    OpsLog,
)
from llm_rosetta_gateway.observability.persistence import PersistenceManager
from llm_rosetta_gateway.ops import OpsContext
from llm_rosetta_gateway.transport.provider_info import ProviderInfo, openai_auth

PASSWORD = "hunter2"


def _request(app: Any, *, body: Any = None, **attrs: Any) -> SimpleNamespace:
    """A request carrying *body* as its JSON and *attrs* as the rest."""

    def _json() -> Any:
        if body is None:
            raise ValueError("no body")
        return body

    fields: dict[str, Any] = dict(
        app=app,
        json=_json,
        query_params={},
        path_params={},
        cookies={},
        headers={},
        client_addr=("127.0.0.1", 5000),
        method="POST",
    )
    fields.update(attrs)
    return SimpleNamespace(**fields)


def _body(response: Any) -> Any:
    return json.loads(response.body)


def _set_cookies(response: Any) -> list[str]:
    return list(response._cookie_headers)


async def _event_types(ops_log: OpsLog) -> list[str]:
    entries, _ = await ops_log.get_entries(limit=100)
    return [entry["event_type"] for entry in entries]


# ---------------------------------------------------------------------------
# auth.py
# ---------------------------------------------------------------------------


def _auth_app(tmp_path: Any, *, ops: bool) -> Any:
    path = tmp_path / "config.json"
    path.write_text(
        json.dumps({"server": {"admin_password": PASSWORD}, "providers": {}}),
        encoding="utf-8",
    )
    ops_log = OpsLog()
    app = fake_app(
        auth_state=AuthState(None, "rsk-internal-old", admin_password=PASSWORD),
        config_path=str(path),
        config_io=JsoncConfigIO(),
        gateway_config=None,
        provider_validation_warnings=[],
        shim_source_signature=None,
        rate_limit_state=None,
        internal_token="rsk-internal-old",
        ops_ctx=OpsContext(ops_log=ops_log) if ops else None,
    )
    app.test_ops_log = ops_log
    return app


async def test_change_password_persists_and_issues_a_session(tmp_path) -> None:
    app = _auth_app(tmp_path, ops=True)
    old_session = app.auth_state.create_session(ip="127.0.0.1")
    request = _request(
        app,
        body={"current_password": PASSWORD, "new_password": "newpass"},
        cookies={ADMIN_COOKIE_NAME: old_session},
    )

    response = await auth.change_password(request)

    assert response.status_code == 200
    assert _body(response) == {"ok": True}
    on_disk = json.loads(open(app.config_path, encoding="utf-8").read())
    assert on_disk["server"]["admin_password"] == "newpass"
    assert app.auth_state.admin_password == "newpass"
    # The old session died with the password; the response carries a new one.
    assert app.auth_state.get_session(old_session) is None
    cookies = _set_cookies(response)
    assert len(cookies) == 1
    assert cookies[0].startswith(f"{ADMIN_COOKIE_NAME}=")
    new_token = cookies[0].split(";")[0].split("=", 1)[1]
    assert app.auth_state.get_session(new_token) is not None
    assert "Max-Age" not in cookies[0]
    assert EVENT_PASSWORD_CHANGED in await _event_types(app.test_ops_log)


async def test_change_password_without_ops_ctx_writes_no_row(tmp_path) -> None:
    app = _auth_app(tmp_path, ops=False)
    request = _request(
        app, body={"current_password": PASSWORD, "new_password": "newpass"}
    )

    response = await auth.change_password(request)

    assert response.status_code == 200
    assert await _event_types(app.test_ops_log) == []


async def test_change_password_rejects_a_wrong_current_password(tmp_path) -> None:
    app = _auth_app(tmp_path, ops=True)
    request = _request(
        app, body={"current_password": "nope", "new_password": "newpass"}
    )

    response = await auth.change_password(request)

    assert response.status_code == 401
    assert _body(response) == {"error": "Current password is incorrect"}
    assert app.auth_state.admin_password == PASSWORD
    on_disk = json.loads(open(app.config_path, encoding="utf-8").read())
    assert on_disk["server"]["admin_password"] == PASSWORD
    assert await _event_types(app.test_ops_log) == []


async def test_change_password_rejects_a_short_new_password(tmp_path) -> None:
    app = _auth_app(tmp_path, ops=True)
    request = _request(app, body={"current_password": PASSWORD, "new_password": "abc"})

    response = await auth.change_password(request)

    assert response.status_code == 400
    assert _body(response) == {"error": "New password must be at least 4 characters"}
    assert app.auth_state.admin_password == PASSWORD


async def test_rotate_token_changes_the_internal_token(tmp_path) -> None:
    app = _auth_app(tmp_path, ops=True)
    session = app.auth_state.create_session()

    response = await auth.rotate_token(_request(app))

    assert response.status_code == 200
    assert _body(response) == {"ok": True}
    assert app.auth_state.internal_token != "rsk-internal-old"
    assert app.auth_state.internal_token.startswith("rsk-internal-")
    assert app.internal_token == app.auth_state.internal_token
    # Sessions do not depend on the token.
    assert app.auth_state.get_session(session) is not None
    assert await _event_types(app.test_ops_log) == [EVENT_TOKEN_ROTATED]


async def test_rotate_token_without_ops_ctx_writes_no_row(tmp_path) -> None:
    app = _auth_app(tmp_path, ops=False)

    response = await auth.rotate_token(_request(app))

    assert response.status_code == 200
    assert app.auth_state.internal_token != "rsk-internal-old"
    assert await _event_types(app.test_ops_log) == []


async def test_logout_all_sessions_counts_and_deletes_the_cookie(tmp_path) -> None:
    app = _auth_app(tmp_path, ops=True)
    for _ in range(3):
        app.auth_state.create_session()

    response = await auth.logout_all_sessions(_request(app))

    assert response.status_code == 200
    assert _body(response) == {"ok": True, "sessions_cleared": 3}
    assert app.auth_state.session_count == 0
    cookies = _set_cookies(response)
    assert len(cookies) == 1
    assert cookies[0].startswith(f"{ADMIN_COOKIE_NAME}=")
    assert "Max-Age=0" in cookies[0]
    entries, _ = await app.test_ops_log.get_entries()
    assert [e["event_type"] for e in entries] == [EVENT_SESSION_LOGOUT_ALL]
    assert entries[0]["details"] == {"sessions_cleared": 3}


async def test_logout_all_sessions_without_ops_ctx_writes_no_row(tmp_path) -> None:
    app = _auth_app(tmp_path, ops=False)

    response = await auth.logout_all_sessions(_request(app))

    assert _body(response) == {"ok": True, "sessions_cleared": 0}
    assert await _event_types(app.test_ops_log) == []


# ---------------------------------------------------------------------------
# capture.py
# ---------------------------------------------------------------------------


async def test_capture_lifecycle() -> None:
    app = fake_app(capture_state=CaptureState())

    status = await capture.get_capture_status(_request(app))
    assert _body(status) == {"enabled": False, "remaining": 0, "captured": 0}

    enabled = await capture.enable_capture(_request(app, body={"requests": 3}))
    assert _body(enabled) == {"enabled": True, "remaining": 3, "captured": 0}

    app.capture_state.record(
        CapturedRequest(request_id="r1", model="m", original_request={"a": 1})
    )
    results = _body(await capture.get_capture_results(_request(app)))
    assert results["total"] == 1
    assert results["results"][0]["request_id"] == "r1"
    assert "original_request" not in results["results"][0]

    detail = await capture.get_capture_result(_request(app, path_params={"index": "0"}))
    assert detail.status_code == 200
    assert _body(detail)["original_request"] == {"a": 1}

    disabled = await capture.disable_capture(_request(app))
    assert _body(disabled) == {"enabled": False, "remaining": 0, "captured": 1}

    cleared = await capture.clear_capture_results(_request(app))
    assert _body(cleared) == {"ok": True}
    assert _body(await capture.get_capture_results(_request(app))) == {
        "results": [],
        "total": 0,
    }


@pytest.mark.parametrize(
    ("body", "remaining"),
    [(None, 5), ({"requests": 0}, 1), ({"requests": 500}, 100), ({}, 5)],
)
async def test_enable_capture_clamps_the_request_count(body, remaining) -> None:
    app = fake_app(capture_state=CaptureState())

    response = await capture.enable_capture(_request(app, body=body))

    assert _body(response)["remaining"] == remaining


@pytest.mark.parametrize(
    ("index", "status", "error"),
    [
        ("abc", 400, "Invalid index"),
        ("-1", 404, "Index out of range"),
        ("7", 404, "Index out of range"),
    ],
)
async def test_capture_result_index_errors(index, status, error) -> None:
    app = fake_app(capture_state=CaptureState())

    response = await capture.get_capture_result(
        _request(app, path_params={"index": index})
    )

    assert response.status_code == status
    assert _body(response) == {"error": error}


async def test_capture_result_without_an_index_is_invalid() -> None:
    app = fake_app(capture_state=CaptureState())

    response = await capture.get_capture_result(_request(app, path_params={}))

    assert response.status_code == 400


# ---------------------------------------------------------------------------
# profiling.py
# ---------------------------------------------------------------------------


def _profile_result(model: str = "a/b:c") -> dict[str, Any]:
    return {
        "timestamp": "2026-11-01T10:20:30.123+00:00",
        "request_id": "r1",
        "model": model,
        "html": "<html>flame</html>",
        "text": "flat",
    }


async def test_profiling_lifecycle() -> None:
    app = fake_app(profiler_state=ProfilerState())

    status = _body(await profiling.get_profiling_status(_request(app)))
    assert status == {
        "enabled": False,
        "remaining": 0,
        "tracing": False,
        "results_count": 0,
        "max_results": 20,
    }

    enabled = await profiling.enable_profiling(
        _request(app, body={"requests": 500, "tracing": True})
    )
    assert _body(enabled)["remaining"] == 100
    assert _body(enabled)["tracing"] is True
    assert _body(enabled)["enabled"] is True

    disabled = _body(await profiling.disable_profiling(_request(app)))
    assert disabled["enabled"] is False
    assert disabled["remaining"] == 0

    default = await profiling.enable_profiling(_request(app))
    assert _body(default)["remaining"] == 5
    assert _body(default)["tracing"] is False


async def test_profiling_results_listing_hides_bodies() -> None:
    app = fake_app(profiler_state=ProfilerState())
    app.profiler_state.results.append(_profile_result())

    body = _body(await profiling.get_profiling_results(_request(app)))

    assert body["total"] == 1
    assert "html" not in body["results"][0]
    assert "text" not in body["results"][0]
    assert body["results"][0]["request_id"] == "r1"


async def test_profiling_result_detail_json_and_html() -> None:
    app = fake_app(profiler_state=ProfilerState())
    app.profiler_state.results.append(_profile_result())

    as_json = await profiling.get_profiling_result(
        _request(app, path_params={"index": "0"})
    )
    assert as_json.status_code == 200
    assert _body(as_json) == _profile_result()

    as_html = await profiling.get_profiling_result(
        _request(app, path_params={"index": "0"}, query_params={"format": ["html"]})
    )
    assert as_html.status_code == 200
    assert as_html.body == b"<html>flame</html>"
    assert as_html.headers["Content-Type"] == "text/html; charset=utf-8"


@pytest.mark.parametrize(
    ("index", "status", "error"),
    [("x", 400, "Invalid index"), ("3", 404, "Index out of range")],
)
async def test_profiling_result_index_errors(index, status, error) -> None:
    app = fake_app(profiler_state=ProfilerState())

    response = await profiling.get_profiling_result(
        _request(app, path_params={"index": index})
    )

    assert response.status_code == status
    assert _body(response) == {"error": error}


async def test_profiling_download_zip_and_clear() -> None:
    import io
    import zipfile

    app = fake_app(profiler_state=ProfilerState())

    empty = await profiling.download_profiling_results(_request(app))
    assert empty.status_code == 404
    assert _body(empty) == {"error": "No results to download"}

    app.profiler_state.results.append(_profile_result())
    response = await profiling.download_profiling_results(_request(app))
    assert response.status_code == 200
    assert response.headers["Content-Type"] == "application/zip"
    assert (
        response.headers["Content-Disposition"]
        == "attachment; filename=profiling-results.zip"
    )
    with zipfile.ZipFile(io.BytesIO(response.body)) as zf:
        assert zf.namelist() == ["profile-0-a_b_c-2026-11-01T102030.html"]
        assert zf.read(zf.namelist()[0]) == b"<html>flame</html>"

    cleared = await profiling.clear_profiling_results(_request(app))
    assert _body(cleared) == {"ok": True}
    assert app.profiler_state.results == []


# ---------------------------------------------------------------------------
# config/model_providers.py, config/settings.py
# ---------------------------------------------------------------------------


async def test_update_model_provider_rejects_an_unknown_capability() -> None:
    request = _request(
        SimpleNamespace(),
        body={"provider": "p", "capabilities": ["text", "telepathy"]},
        path_params={"name": "m"},
    )

    response = await model_providers.update_model_provider(request)

    assert response.status_code == 400
    error = _body(response)["error"]
    assert error.startswith("Unknown capabilities: telepathy; known: ")
    known = error.split("known: ")[1].split(", ")
    assert known == sorted(known)
    for name in ("text", "vision", "tools", "reasoning"):
        assert name in known


async def test_settings_reject_an_invalid_rate_limit_quota(tmp_path) -> None:
    # The quota is validated inside the config mutation, so the handler needs
    # a readable config; an empty one is enough.
    path = tmp_path / "config.json"
    path.write_text("{}", encoding="utf-8")
    app = fake_app(config_path=str(path), config_io=JsoncConfigIO())
    request = _request(app, body={"rate_limit": {"per_ip": "banana"}}, method="PUT")

    response = await settings.put_server_settings(request)

    assert response.status_code == 400
    assert _body(response)["error"].startswith("Invalid per_ip quota: ")
    assert "rate_limit" not in json.loads(path.read_text(encoding="utf-8")).get(
        "server", {}
    )


def test_apply_rate_limit_settings_rejects_a_bad_algorithm() -> None:
    server: dict[str, Any] = {}

    response = settings._apply_rate_limit_settings(server, {"algorithm": "nope"})

    assert response is not None
    assert response.status_code == 400
    assert _body(response)["error"].startswith("Invalid algorithm: 'nope'")


# ---------------------------------------------------------------------------
# config/upstream.py
# ---------------------------------------------------------------------------


class _Config:
    """The slice of ``GatewayConfig`` the upstream routes read."""

    def __init__(self, providers: dict[str, ProviderInfo]) -> None:
        self.providers = providers
        self.provider_types = {n: "chat_completions" for n in providers}
        self.provider_shim_names: dict[str, str] = {}
        self._raw_providers: dict[str, Any] = {}
        self.models: dict[str, Any] = {}
        self.model_upstream_names: dict[str, Any] = {}


def _pinfo() -> ProviderInfo:
    return ProviderInfo(
        name="up",
        api_key="sk-secret",
        base_url="https://example.com/v1",
        auth_header_fn=openai_auth,
        url_template="{base_url}/chat/completions",
    )


class _FakeClient:
    """Stands in for ``httpclient.AsyncClient``; records every GET."""

    calls: list[tuple[str, dict[str, str]]] = []
    reply: Any = None

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs

    async def get(self, url: str, headers: dict[str, str] | None = None) -> Any:
        type(self).calls.append((url, dict(headers or {})))
        reply = type(self).reply
        if isinstance(reply, Exception):
            raise reply
        return reply

    async def aclose(self) -> None:
        return None


def _http_response(status: int, content: bytes) -> httpclient.Response:
    return httpclient.Response(
        status, httpclient.CaseInsensitiveDict(), content, "https://example.com"
    )


@pytest.fixture
def fake_upstream(monkeypatch: pytest.MonkeyPatch) -> type[_FakeClient]:
    _FakeClient.calls = []
    _FakeClient.reply = None
    monkeypatch.setattr(httpclient, "AsyncClient", _FakeClient)
    # The route may also hold its own binding of the client class.
    monkeypatch.setattr(upstream, "AsyncClient", _FakeClient, raising=False)
    return _FakeClient


def _models_request(config: Any) -> SimpleNamespace:
    return _request(
        fake_app(gateway_config=config),
        path_params={"name": "up"},
        method="GET",
    )


async def test_fetch_upstream_models_lists_sorted_ids(fake_upstream) -> None:
    fake_upstream.reply = _http_response(
        200, json.dumps({"data": [{"id": "b-model"}, {"id": "a-model"}]}).encode()
    )

    response = await upstream.fetch_upstream_models(
        _models_request(_Config({"up": _pinfo()}))
    )

    assert response.status_code == 200
    assert _body(response) == {
        "provider": "up",
        "api_standard": "chat_completions",
        "models": ["a-model", "b-model"],
    }
    [(url, headers)] = fake_upstream.calls
    assert url == "https://example.com/v1/models"
    assert headers["Authorization"] == "Bearer sk-secret"


async def test_fetch_upstream_models_reports_upstream_errors(fake_upstream) -> None:
    fake_upstream.reply = _http_response(503, b"down")
    response = await upstream.fetch_upstream_models(
        _models_request(_Config({"up": _pinfo()}))
    )
    assert response.status_code == 200
    assert _body(response) == {
        "error": "Upstream returned HTTP 503. "
        "This provider may not support model listing."
    }

    fake_upstream.reply = _http_response(200, b"<html>")
    response = await upstream.fetch_upstream_models(
        _models_request(_Config({"up": _pinfo()}))
    )
    assert _body(response) == {"error": "Upstream returned non-JSON response"}

    fake_upstream.reply = ConnectionError("Connection refused")
    response = await upstream.fetch_upstream_models(
        _models_request(_Config({"up": _pinfo()}))
    )
    assert response.status_code == 200
    assert _body(response)["error"].startswith(
        "Connection refused at https://example.com/v1/models."
    )


async def test_fetch_upstream_models_unknown_provider_and_no_config() -> None:
    response = await upstream.fetch_upstream_models(_models_request(_Config({})))
    assert response.status_code == 404
    assert _body(response) == {"error": "Provider 'up' not found"}

    response = await upstream.fetch_upstream_models(_models_request(None))
    assert response.status_code == 500
    assert _body(response) == {"error": "Gateway config not loaded"}


async def test_check_routing_loops() -> None:
    app = SimpleNamespace(config=None)
    response = await upstream.check_routing_loops(_request(app, method="GET"))
    assert _body(response) == {"warnings": []}

    config = SimpleNamespace(
        models={"a": SimpleNamespace(providers=[]), "b": SimpleNamespace(providers=[])},
        model_upstream_names={"a": "b"},
    )
    app = SimpleNamespace(config=config)
    response = await upstream.check_routing_loops(_request(app, method="GET"))
    assert response.status_code == 200
    [warning] = _body(response)["warnings"]
    assert "a" in warning and "b" in warning


# ---------------------------------------------------------------------------
# keys.py
# ---------------------------------------------------------------------------


@pytest.fixture
async def keystore(tmp_path):
    store = await KeyStore.create(tmp_path / "keys.db")
    yield store
    await store.close()


def _keys_app(keystore: Any, *, ops: bool) -> Any:
    ops_log = OpsLog()
    app = fake_app(
        keystore=keystore,
        ops_ctx=OpsContext(ops_log=ops_log) if ops else None,
    )
    app.test_ops_log = ops_log
    return app


async def test_delete_api_key_removes_it_and_records_the_label(keystore) -> None:
    key_id, _ = await keystore.create_key(label="ci")
    app = _keys_app(keystore, ops=True)

    response = await keys.delete_api_key(
        _request(app, path_params={"key_id": key_id}, method="DELETE")
    )

    assert response.status_code == 200
    assert _body(response) == {"ok": True, "deleted": key_id}
    assert await keystore.list_keys() == []
    entries, _ = await app.test_ops_log.get_entries()
    assert [e["event_type"] for e in entries] == [EVENT_KEY_DELETE]
    assert "ci" in json.dumps(entries[0]["details"]) + entries[0]["message"]


async def test_delete_api_key_unknown_id_is_404(keystore) -> None:
    app = _keys_app(keystore, ops=True)

    response = await keys.delete_api_key(
        _request(app, path_params={"key_id": "nope"}, method="DELETE")
    )

    assert response.status_code == 404
    assert _body(response) == {"error": "Key 'nope' not found"}
    assert await _event_types(app.test_ops_log) == []


async def test_delete_api_key_without_ops_ctx_writes_no_row(keystore) -> None:
    key_id, _ = await keystore.create_key(label="ci")
    app = _keys_app(keystore, ops=False)

    response = await keys.delete_api_key(
        _request(app, path_params={"key_id": key_id}, method="DELETE")
    )

    assert response.status_code == 200
    assert await _event_types(app.test_ops_log) == []


async def test_rotate_api_key_returns_a_new_value(keystore) -> None:
    key_id, raw = await keystore.create_key(label="ci")
    app = _keys_app(keystore, ops=True)

    response = await keys.rotate_api_key(_request(app, path_params={"key_id": key_id}))

    assert response.status_code == 200
    body = _body(response)
    assert body["ok"] is True
    assert body["id"] == key_id
    assert body["key"] and body["key"] != raw
    assert [k["id"] for k in await keystore.list_keys()] == [key_id]
    assert await _event_types(app.test_ops_log) == [EVENT_KEY_ROTATE]


async def test_rotate_api_key_unknown_id_is_404(keystore) -> None:
    app = _keys_app(keystore, ops=True)

    response = await keys.rotate_api_key(_request(app, path_params={"key_id": "nope"}))

    assert response.status_code == 404
    assert _body(response) == {"error": "Key 'nope' not found"}
    assert await _event_types(app.test_ops_log) == []


async def test_rotate_api_key_without_ops_ctx_writes_no_row(keystore) -> None:
    key_id, _ = await keystore.create_key(label="ci")
    app = _keys_app(keystore, ops=False)

    response = await keys.rotate_api_key(_request(app, path_params={"key_id": key_id}))

    assert response.status_code == 200
    assert await _event_types(app.test_ops_log) == []


# ---------------------------------------------------------------------------
# observability.py
# ---------------------------------------------------------------------------


@pytest.fixture
async def pm(tmp_path):
    manager = await PersistenceManager.create(str(tmp_path), success_max=1000)
    yield manager
    await manager.close()


def _obs_request(jobs: Any, **attrs: Any) -> SimpleNamespace:
    return _request(fake_app(jobs=jobs), **attrs)


_NO_JOBS = {"error": "No persistence configured"}

# (handler name, request body, job kind, expected payload subset)
_JOB_ROUTES = [
    ("clear_requests", None, "data.clear", {"table": "request_log"}),
    ("clear_error_dumps", None, "data.clear", {"table": "error_dumps"}),
    ("clear_ops_log", None, "data.clear", {"table": "ops_log"}),
    ("db_vacuum", None, "data.vacuum", {}),
    ("db_trim", None, "data.trim", {}),
    (
        "db_cleanup",
        {"max_age_days": 7},
        "data.cleanup",
        {
            "mode": "age",
            "max_age_days": 7,
            "tables": ["request_log", "error_dumps", "ops_log"],
        },
    ),
    (
        "cleanup_requests_by_age",
        {"max_age_days": 3},
        "data.cleanup",
        {"mode": "age", "max_age_days": 3, "tables": ["request_log"]},
    ),
    (
        "cleanup_error_dumps_by_age",
        {"max_age_days": 4},
        "data.cleanup",
        {"mode": "age", "max_age_days": 4, "tables": ["error_dumps"]},
    ),
    (
        "cleanup_ops_log_by_age",
        {"max_age_days": 5},
        "data.cleanup",
        {"mode": "age", "max_age_days": 5, "tables": ["ops_log"]},
    ),
    (
        "db_cleanup_before",
        {"before": "2026-01-01T00:00:00Z", "target": "errors"},
        "data.cleanup",
        {"mode": "before", "before": "2026-01-01T00:00:00Z", "tables": ["error_dumps"]},
    ),
    (
        "db_cleanup_range",
        {"start": "2026-01-01T00:00:00Z", "end": "2026-02-01T00:00:00Z"},
        "data.cleanup",
        {
            "mode": "range",
            "start": "2026-01-01T00:00:00Z",
            "end": "2026-02-01T00:00:00Z",
            "tables": ["request_log", "error_dumps", "ops_log"],
        },
    ),
]


@pytest.mark.parametrize(("name", "body", "kind", "payload"), _JOB_ROUTES)
async def test_job_routes_answer_202_with_the_job(
    pm, name, body, kind, payload
) -> None:
    manager = JobManager(pm)

    response = await getattr(observability, name)(
        _obs_request(manager, body=body, method="POST")
    )

    assert response.status_code == 202
    result = _body(response)
    assert result["ok"] is True
    assert result["accepted"] is True
    assert result["job_id"] == result["job"]["id"]
    assert result["job"]["kind"] == kind
    for key, value in payload.items():
        assert result["job"]["payload"][key] == value
    assert result["job"]["payload"]["source"] == "admin"


async def test_rebuild_metrics_answers_202_with_a_rebuild_job(pm) -> None:
    response = await observability.rebuild_metrics(
        _obs_request(JobManager(pm), method="POST")
    )

    assert response.status_code == 202
    assert _body(response)["job"]["kind"] == "data.rebuild"


@pytest.mark.parametrize(
    ("name", "body"),
    [(n, b) for n, b, _, _ in _JOB_ROUTES if n != "db_vacuum"] + [("db_vacuum", None)],
)
async def test_job_routes_answer_400_without_a_job_manager(name, body) -> None:
    response = await getattr(observability, name)(
        _obs_request(None, body=body, method="POST")
    )

    assert response.status_code == 400
    expected = "No ops context configured" if name == "clear_requests" else None
    assert _body(response) == {"error": expected or _NO_JOBS["error"]}


async def test_rebuild_metrics_without_a_job_manager_is_400() -> None:
    response = await observability.rebuild_metrics(_obs_request(None))

    assert response.status_code == 400
    assert _body(response) == {"error": "No persistence configured (in-memory mode)"}


@pytest.mark.parametrize("value", [0, -1, "7", 1.5, None])
async def test_cleanup_by_age_rejects_a_bad_age(pm, value) -> None:
    response = await observability.cleanup_requests_by_age(
        _obs_request(JobManager(pm), body={"max_age_days": value})
    )

    assert response.status_code == 400
    assert _body(response) == {"error": "max_age_days must be a positive integer"}


async def test_db_cleanup_before_and_range_validate_their_bounds(pm) -> None:
    manager = JobManager(pm)

    response = await observability.db_cleanup_before(_obs_request(manager, body={}))
    assert response.status_code == 400
    assert _body(response) == {"error": "before must be a non-empty ISO-8601 string"}

    response = await observability.db_cleanup_range(
        _obs_request(manager, body={"end": "2026-02-01T00:00:00Z"})
    )
    assert response.status_code == 400
    assert _body(response) == {"error": "start must be a non-empty ISO-8601 string"}

    response = await observability.db_cleanup_range(
        _obs_request(manager, body={"start": "2026-01-01T00:00:00Z"})
    )
    assert response.status_code == 400
    assert _body(response) == {"error": "end must be a non-empty ISO-8601 string"}


async def test_delete_error_dump_without_persistence_is_400() -> None:
    app = fake_app(ops_ctx=OpsContext())

    response = await observability.delete_error_dump(
        _request(app, path_params={"dump_id": "d1"}, method="DELETE")
    )

    assert response.status_code == 400
    assert _body(response) == _NO_JOBS


async def test_delete_error_dump_without_ops_ctx_is_400() -> None:
    response = await observability.delete_error_dump(
        _request(fake_app(), path_params={"dump_id": "d1"}, method="DELETE")
    )

    assert response.status_code == 400
    assert _body(response) == _NO_JOBS


class _DumpStore:
    """Persistence stand-in holding error dumps and their bodies."""

    def __init__(self) -> None:
        self.dumps: dict[str, dict[str, Any]] = {}
        self.bodies: dict[str, bytes] = {}

    async def delete_error_dump(self, dump_id: str) -> bool:
        return self.dumps.pop(dump_id, None) is not None

    async def get_error_dump(self, dump_id: str) -> dict[str, Any] | None:
        entry = self.dumps.get(dump_id)
        return dict(entry) if entry is not None else None

    async def get_dump_body(self, body_hash: str) -> bytes | None:
        return self.bodies.get(body_hash)


def _dump_app(store: _DumpStore) -> Any:
    ops_log = OpsLog()
    app = fake_app(
        persistence=store, ops_ctx=OpsContext(ops_log=ops_log, persistence=store)
    )
    app.test_ops_log = ops_log
    return app


async def test_delete_error_dump_missing_is_404_and_existing_is_200() -> None:
    store = _DumpStore()
    store.dumps["d1"] = {"id": "d1"}
    app = _dump_app(store)

    missing = await observability.delete_error_dump(
        _request(app, path_params={"dump_id": "zzz"}, method="DELETE")
    )
    assert missing.status_code == 404
    assert _body(missing) == {"error": "Not found"}

    found = await observability.delete_error_dump(
        _request(app, path_params={"dump_id": "d1"}, method="DELETE")
    )
    assert found.status_code == 200
    assert _body(found) == {"ok": True}
    assert store.dumps == {}
    entries, _ = await app.test_ops_log.get_entries()
    assert [(e["details"]["dump_id"], e["details"]["found"]) for e in entries] == [
        ("d1", True),
        ("zzz", False),
    ]


async def test_error_dump_detail_decompresses_both_bodies() -> None:
    store = _DumpStore()
    store.dumps["d1"] = {
        "id": "d1",
        "body_hash": "h1",
        "converted_body_hash": "h2",
    }
    store.bodies["h1"] = zlib.compress(json.dumps({"model": "m"}).encode())
    store.bodies["h2"] = zlib.compress(json.dumps({"converted": True}).encode())
    app = _dump_app(store)

    response = await observability.get_error_dump_detail(
        _request(app, path_params={"dump_id": "d1"}, method="GET")
    )

    assert response.status_code == 200
    body = _body(response)
    assert body["request_body"] == {"model": "m"}
    assert body["converted_body"] == {"converted": True}


async def test_error_dump_detail_survives_a_corrupt_body() -> None:
    store = _DumpStore()
    store.dumps["d1"] = {"id": "d1", "body_hash": "h1", "converted_body_hash": "h2"}
    store.bodies["h1"] = b"not zlib"
    store.bodies["h2"] = b"also not zlib"
    app = _dump_app(store)

    response = await observability.get_error_dump_detail(
        _request(app, path_params={"dump_id": "d1"}, method="GET")
    )

    assert response.status_code == 200
    body = _body(response)
    assert body["request_body"] is None
    assert body["converted_body"] is None


async def test_error_dump_detail_errors() -> None:
    store = _DumpStore()
    app = _dump_app(store)
    request = _request(app, path_params={"dump_id": "d1"}, method="GET")

    missing = await observability.get_error_dump_detail(request)
    assert missing.status_code == 404
    assert _body(missing) == {"error": "Not found"}

    no_store = await observability.get_error_dump_detail(
        _request(fake_app(), path_params={"dump_id": "d1"}, method="GET")
    )
    assert no_store.status_code == 400
    assert _body(no_store) == _NO_JOBS


async def test_error_dump_body_download_is_the_decompressed_json() -> None:
    store = _DumpStore()
    store.dumps["d1"] = {"id": "d1", "body_hash": "h1"}
    store.dumps["d2"] = {"id": "d2"}
    store.bodies["h1"] = zlib.compress(b'{"model": "m"}')
    app = _dump_app(store)

    response = await observability.get_error_dump_body(
        _request(app, path_params={"dump_id": "d1"}, method="GET")
    )
    assert response.status_code == 200
    assert response.body == b'{"model": "m"}'
    assert (
        response.headers["Content-Disposition"]
        == 'attachment; filename="error-dump-d1.json"'
    )

    no_body = await observability.get_error_dump_body(
        _request(app, path_params={"dump_id": "d2"}, method="GET")
    )
    assert no_body.status_code == 404
    assert _body(no_body) == {"error": "No request body stored for this dump"}


async def test_ops_log_listings_return_the_known_names() -> None:
    request = _request(fake_app(), method="GET")

    types = await observability.get_ops_log_event_types(request)
    sources = await observability.get_ops_log_sources(request)

    assert types.status_code == 200
    assert _body(types) == {"event_types": ALL_EVENT_TYPES}
    assert sources.status_code == 200
    assert _body(sources) == {"sources": ALL_SOURCES}
    assert "key_create" in _body(types)["event_types"]
    assert "admin" in _body(sources)["sources"]


# ---------------------------------------------------------------------------
# quota.py
# ---------------------------------------------------------------------------


def _quota_request(config: Any, name: str = "up", **query: str) -> SimpleNamespace:
    return _request(
        fake_app(gateway_config=config),
        path_params={"name": name},
        query_params={k: [v] for k, v in query.items()},
        method="GET",
    )


async def test_provider_quota_happy_path_forwards_refresh(monkeypatch) -> None:
    seen: list[tuple[str, bool]] = []

    async def read(config: Any, name: str, *, force: bool = False) -> dict[str, Any]:
        seen.append((name, force))
        return {"provider": name, "supported": True, "status": "ok"}

    monkeypatch.setattr("llm_rosetta_gateway.quota.service.read_provider_quota", read)
    config = _Config({"up": _pinfo()})

    plain = await quota.get_provider_quota(_quota_request(config), "up")
    forced = await quota.get_provider_quota(_quota_request(config, refresh="1"), "up")

    assert plain.status_code == 200
    assert _body(plain) == {"provider": "up", "supported": True, "status": "ok"}
    assert forced.status_code == 200
    assert seen == [("up", False), ("up", True)]


async def test_provider_quota_failure_is_a_200_with_a_scrubbed_error(
    monkeypatch,
) -> None:
    async def read(config: Any, name: str, *, force: bool = False) -> dict[str, Any]:
        raise RuntimeError("probe failed with key sk-abcdefghijklmnopqrstuvwxyz1234")

    monkeypatch.setattr("llm_rosetta_gateway.quota.service.read_provider_quota", read)

    response = await quota.get_provider_quota(
        _quota_request(_Config({"up": _pinfo()})), "up"
    )

    assert response.status_code == 200
    assert _body(response) == {
        "provider": "up",
        "supported": True,
        "status": "error",
        "error": "probe failed with key [REDACTED]",
    }


async def test_provider_quota_unknown_provider_and_no_config() -> None:
    unknown = await quota.get_provider_quota(
        _quota_request(_Config({"up": _pinfo()}), "other"), "other"
    )
    assert unknown.status_code == 404
    assert _body(unknown) == {"error": "Provider 'other' not found"}

    no_config = await quota.get_provider_quota(_quota_request(None), "up")
    assert no_config.status_code == 500
    assert _body(no_config) == {"error": "No config loaded"}


# ---------------------------------------------------------------------------
# updates.py
# ---------------------------------------------------------------------------


async def test_get_updates_forwards_force_and_the_proxy(monkeypatch) -> None:
    from llm_rosetta_gateway import updates

    calls: list[tuple[Any, bool]] = []

    async def check(proxy: Any = None, *, force: bool = False) -> dict[str, Any]:
        calls.append((proxy, force))
        return {"checked": True, "update": None}

    monkeypatch.setattr(updates, "check", check)
    app = SimpleNamespace(config=SimpleNamespace(proxy="http://p:1", update_check=True))
    app = fake_app(config=app.config)

    plain = await updates_routes.get_updates(_request(app, method="GET"))
    forced = await updates_routes.get_updates(
        _request(app, method="GET", query_params={"force": ["1"]})
    )
    word = await updates_routes.get_updates(
        _request(app, method="GET", query_params={"force": ["true"]})
    )
    other = await updates_routes.get_updates(
        _request(app, method="GET", query_params={"force": ["yes"]})
    )

    assert plain.status_code == 200
    assert _body(plain) == {"checked": True, "update": None}
    assert _body(forced) == {"checked": True, "update": None}
    assert calls == [
        ("http://p:1", False),
        ("http://p:1", True),
        ("http://p:1", True),
        ("http://p:1", False),
    ]
    assert _body(word) == _body(other)


async def test_get_updates_without_a_config_probes_with_no_proxy(monkeypatch) -> None:
    from llm_rosetta_gateway import updates

    calls: list[tuple[Any, bool]] = []

    async def check(proxy: Any = None, *, force: bool = False) -> dict[str, Any]:
        calls.append((proxy, force))
        return {"checked": True}

    monkeypatch.setattr(updates, "check", check)

    response = await updates_routes.get_updates(_request(fake_app(), method="GET"))

    assert _body(response) == {"checked": True}
    assert calls == [(None, False)]


async def test_get_updates_disabled_skips_the_probe(monkeypatch) -> None:
    from llm_rosetta_gateway import updates

    async def check(proxy: Any = None, *, force: bool = False) -> dict[str, Any]:
        raise AssertionError("the probe must not run")

    monkeypatch.setattr(updates, "check", check)
    app = fake_app(config=SimpleNamespace(update_check=False))

    response = await updates_routes.get_updates(_request(app, method="GET"))

    assert response.status_code == 200
    assert _body(response) == {"checked": False, "reason": "disabled"}


@pytest.mark.parametrize(
    ("query", "language"),
    [({}, "en"), ({"lang": ["en"]}, "en"), ({"lang": ["zh"]}, "zh")],
)
async def test_get_changelog_returns_the_markdown(monkeypatch, query, language) -> None:
    from llm_rosetta_gateway import updates

    asked: list[str] = []

    def read_changelog(lang: str) -> str:
        asked.append(lang)
        return f"# changes ({lang})"

    monkeypatch.setattr(updates, "read_changelog", read_changelog)

    response = await updates_routes.get_changelog(
        _request(fake_app(), method="GET", query_params=query)
    )

    assert response.status_code == 200
    assert _body(response) == {"markdown": f"# changes ({language})"}
    assert asked == [language]


# ---------------------------------------------------------------------------
# Audit rows the first pass left unpinned
# ---------------------------------------------------------------------------


async def test_create_api_key_records_the_label(keystore) -> None:
    app = _keys_app(keystore, ops=True)

    response = await keys.create_api_key(_request(app, body={"label": "ci"}))

    assert response.status_code == 200
    key_id = _body(response)["key"]["id"]
    entries, _ = await app.test_ops_log.get_entries()
    assert [e["event_type"] for e in entries] == [EVENT_KEY_CREATE]
    assert key_id in json.dumps(entries[0]["details"]) + entries[0]["message"]
    assert "ci" in json.dumps(entries[0]["details"]) + entries[0]["message"]


async def test_create_api_key_without_ops_ctx_writes_no_row(keystore) -> None:
    app = _keys_app(keystore, ops=False)

    response = await keys.create_api_key(_request(app, body={"label": "ci"}))

    assert response.status_code == 200
    assert await _event_types(app.test_ops_log) == []


async def test_update_api_key_records_the_changed_fields(keystore) -> None:
    key_id, _ = await keystore.create_key(label="ci")
    app = _keys_app(keystore, ops=True)

    response = await keys.update_api_key(
        _request(
            app,
            body={"label": "renamed"},
            path_params={"key_id": key_id},
            method="PUT",
        )
    )

    assert response.status_code == 200
    entries, _ = await app.test_ops_log.get_entries()
    assert [e["event_type"] for e in entries] == [EVENT_KEY_UPDATE]
    assert "label" in json.dumps(entries[0]["details"])


async def test_update_api_key_without_ops_ctx_writes_no_row(keystore) -> None:
    key_id, _ = await keystore.create_key(label="ci")
    app = _keys_app(keystore, ops=False)

    response = await keys.update_api_key(
        _request(
            app,
            body={"label": "renamed"},
            path_params={"key_id": key_id},
            method="PUT",
        )
    )

    assert response.status_code == 200
    assert await _event_types(app.test_ops_log) == []


async def test_apply_update_records_the_version_it_installed(monkeypatch) -> None:
    from llm_rosetta_gateway import updates

    async def check(*_: Any, **__: Any) -> dict[str, Any]:
        return {
            "update": {"version": "9.9.9"},
            "self_update": {"allowed": True, "reason": None},
        }

    async def install(version: str, *_: Any, **__: Any) -> dict[str, Any]:
        return {"ok": True, "output": "", "dists": ["llm-rosetta"]}

    restarts: list[int] = []
    monkeypatch.setattr(updates, "check", check)
    monkeypatch.setattr(updates, "install", install)
    monkeypatch.setattr(updates, "restart_process", lambda: restarts.append(1))
    ops_log = OpsLog()
    app = fake_app(
        config=SimpleNamespace(update_check=True, self_update=True, proxy=None),
        ops_ctx=OpsContext(ops_log=ops_log),
    )

    response = await updates_routes.apply_update(
        _request(app, body={"version": "9.9.9"})
    )

    assert response.status_code == 200
    entries, _ = await ops_log.get_entries()
    assert [e["event_type"] for e in entries] == [EVENT_SELF_UPDATE]
    assert "9.9.9" in json.dumps(entries[0]["details"]) + entries[0]["message"]


async def test_setup_admin_records_the_admin_setup_row(tmp_path) -> None:
    from llm_rosetta_gateway.admin import setup_admin
    from llm_rosetta_gateway.app.admin_context import GatewayAdminContext
    from llm_rosetta_gateway.config import GatewayConfig

    config = GatewayConfig(
        {
            "providers": {"t": {"api_key": "k", "base_url": "http://x"}},
            "models": {"m": "t"},
        }
    )
    app = SimpleNamespace(persistence=None)
    app.admin = GatewayAdminContext(app)  # type: ignore[attr-defined]

    await setup_admin(
        app, config, str(tmp_path / "config.jsonc"), data_dir=str(tmp_path)
    )

    try:
        entries, _ = await app.ops_log.get_entries()
        assert EVENT_ADMIN_SETUP in [e["event_type"] for e in entries]
    finally:
        await app.persistence.close()
