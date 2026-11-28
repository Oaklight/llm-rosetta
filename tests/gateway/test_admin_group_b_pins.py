"""Characterisation pins for the admin reads of ``request.app`` that the A1 group B migration moves.

Each test records what a route or helper answers today for the state it reads
off the app: the panel's head, the auth state, the internal token, the stores,
the bind address, the transports and the deferred startup.  They run against a
``fake_app`` (the shared fake that has the ``admin`` context), so they hold
before and after the reads move behind ``AdminContext``.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace
from typing import Any

import pytest
from _admin_app import fake_app

from llm_rosetta_gateway.admin import apply_admin_theme
from llm_rosetta_gateway.admin.routes import auth, keys, testing
from llm_rosetta_gateway.keystore import KeyStore
from llm_rosetta_gateway.middleware.auth import ADMIN_COOKIE_NAME, AuthState


def _request(app: Any, **attrs: Any) -> SimpleNamespace:
    fields: dict[str, Any] = dict(
        app=app,
        query_params={},
        path_params={},
        cookies={},
        headers={},
        client_addr=("127.0.0.1", 5000),
        method="GET",
    )
    fields.update(attrs)
    return SimpleNamespace(**fields)


def _body(response: Any) -> Any:
    return json.loads(response.body)


# --- the panel's head --------------------------------------------------------


def test_apply_admin_theme_joins_the_stored_prefix_and_the_css() -> None:
    app = fake_app(_admin_head_prefix="<style>p{}</style>")
    apply_admin_theme(app, "body { color: red }")
    head = app.admin_custom_head
    assert head.startswith("<style>p{}</style>\n")
    assert '<style id="admin-custom-css">' in head
    assert "body { color: red }" in head


def test_apply_admin_theme_without_css_keeps_only_the_prefix() -> None:
    app = fake_app(_admin_head_prefix="<script>x</script>")
    apply_admin_theme(app, None)
    assert app.admin_custom_head == "<script>x</script>"


def test_apply_admin_theme_without_a_prefix_or_css_is_empty() -> None:
    app = fake_app()
    apply_admin_theme(app, "")
    assert app.admin_custom_head == ""


def test_apply_admin_theme_rebuilds_from_the_prefix_each_time() -> None:
    app = fake_app(_admin_head_prefix="P")
    apply_admin_theme(app, "a{}")
    apply_admin_theme(app, "b{}")
    assert app.admin_custom_head.count("admin-custom-css") == 1
    assert "b{}" in app.admin_custom_head
    assert "a{}" not in app.admin_custom_head


async def test_the_served_page_carries_the_apps_custom_head() -> None:
    auth._admin_html_cache.clear()
    marker = "<meta name=pin-marker>"
    with_head = await auth.serve_admin_html(
        _request(fake_app(admin_custom_head=marker))
    )
    without = await auth.serve_admin_html(_request(fake_app()))
    auth._admin_html_cache.clear()
    assert marker.encode() in with_head.body
    assert marker.encode() not in without.body


# --- the auth state and the internal token ----------------------------------


def _auth_app(**state: Any) -> Any:
    return fake_app(
        auth_state=AuthState(None, "rsk-internal", admin_password="pw", **state),
        internal_token="rsk-internal",
    )


async def test_admin_check_reports_what_the_auth_state_says() -> None:
    app = _auth_app(session_idle_minutes=7, session_remember_days=3)
    body = _body(await auth.admin_check(_request(app)))
    assert body == {
        "requires_auth": True,
        "idle_minutes": 7,
        "remember_days": 3,
        "remembered": False,
    }


async def test_admin_check_without_a_password_requires_no_auth() -> None:
    app = fake_app(auth_state=AuthState(None, "t"))
    assert _body(await auth.admin_check(_request(app)))["requires_auth"] is False


async def test_admin_check_marks_a_remembered_session() -> None:
    app = _auth_app()
    token = app.auth_state.create_session(remember=True)
    request = _request(app, cookies={ADMIN_COOKIE_NAME: token})
    assert _body(await auth.admin_check(request))["remembered"] is True


async def test_admin_logout_invalidates_the_session_and_clears_the_cookie() -> None:
    app = _auth_app()
    token = app.auth_state.create_session()
    response = await auth.admin_logout(
        _request(app, cookies={ADMIN_COOKIE_NAME: token})
    )
    assert _body(response) == {"ok": True}
    assert app.auth_state.get_session(token) is None
    assert any(ADMIN_COOKIE_NAME in c for c in response._cookie_headers)


async def test_admin_login_without_a_password_is_400() -> None:
    app = fake_app(auth_state=AuthState(None, "t"))
    response = await auth.admin_login(_request(app, method="POST"))
    assert response.status_code == 400
    assert _body(response) == {"error": "Admin password not configured"}


async def test_admin_login_with_the_password_issues_a_session() -> None:
    app = _auth_app()
    request = _request(app, method="POST", json=lambda: {"password": "pw"})
    response = await auth.admin_login(request)
    assert response.status_code == 200
    assert _body(response)["ok"] is True
    assert any(ADMIN_COOKIE_NAME in c for c in response._cookie_headers)


async def test_rotating_the_token_moves_the_apps_internal_token_with_it() -> None:
    app = _auth_app()
    app.ops_ctx = None
    response = await auth.rotate_token(_request(app, method="POST"))
    assert _body(response) == {"ok": True}
    assert app.internal_token == app.auth_state.internal_token
    assert app.internal_token != "rsk-internal"


async def test_the_internal_token_route_answers_the_apps_token() -> None:
    response = await keys.get_internal_token(_request(fake_app(internal_token="rsk-x")))
    assert _body(response) == {"token": "rsk-x"}


@pytest.mark.parametrize(
    "attrs", [{}, {"internal_token": ""}, {"internal_token": None}]
)
async def test_the_internal_token_route_without_a_token_is_500(attrs: Any) -> None:
    response = await keys.get_internal_token(_request(fake_app(**attrs)))
    assert response.status_code == 500
    assert _body(response) == {"error": "No internal token available"}


# --- the key store and the usage cache --------------------------------------


async def test_a_key_route_without_a_keystore_raises() -> None:
    with pytest.raises(RuntimeError, match="KeyStore not configured"):
        await keys.get_api_keys(_request(fake_app()))


async def test_key_usage_is_cached_on_the_app_for_five_seconds(tmp_path: Any) -> None:
    store = await KeyStore.create(tmp_path / "keys.db")
    try:
        app = fake_app(keystore=store)
        first = await keys.get_key_usage(_request(app))
        assert first.status_code == 200
        stamp, payload = app._usage_payload
        assert payload == _body(first)
        assert time.monotonic() - stamp < 5.0
        # A key created now does not show until the cache lapses.
        await store.create_key("late")
        again = await keys.get_key_usage(_request(app))
        assert _body(again) == payload
        # An expired entry is rebuilt.
        app._usage_payload = (time.monotonic() - 10.0, {"stale": True})
        rebuilt = await keys.get_key_usage(_request(app))
        assert "stale" not in _body(rebuilt)
        assert app._usage_payload[1] == _body(rebuilt)
    finally:
        await store.close()


async def test_key_usage_without_persistence_reports_no_traffic(tmp_path: Any) -> None:
    store = await KeyStore.create(tmp_path / "keys.db")
    try:
        response = await keys.get_key_usage(_request(fake_app(keystore=store)))
        assert response.status_code == 200
    finally:
        await store.close()


# --- the model test's self call ---------------------------------------------


async def _start(app: Any, monkeypatch: Any) -> tuple[Any, dict[str, Any], list[Any]]:
    seen: list[Any] = []

    async def fake_run(*args: Any) -> None:
        seen.append(args)

    monkeypatch.setattr(testing, "_run_test_task", fake_run)
    request = _request(
        app,
        method="POST",
        json=lambda: {"endpoint": "/v1/x", "payload": {"a": 1}},
    )
    response = await testing.start_test(request)
    tasks = dict(testing._test_tasks)
    await __import__("asyncio").sleep(0)
    return response, tasks, seen


async def test_start_test_without_an_internal_token_is_500(monkeypatch: Any) -> None:
    response, _tasks, seen = await _start(fake_app(), monkeypatch)
    assert response.status_code == 500
    assert _body(response) == {"error": "No internal token available"}
    assert seen == []


async def test_start_test_calls_back_on_the_bound_address(monkeypatch: Any) -> None:
    app = fake_app(internal_token="rsk-t", _bind_host="10.1.2.3", _bind_port=4242)
    response, tasks, seen = await _start(app, monkeypatch)
    task_id = _body(response)["task_id"]
    assert tasks[task_id]["_base_url"] == "http://10.1.2.3:4242"
    assert seen[0][3] == "rsk-t"


async def test_start_test_defaults_to_loopback_28765(monkeypatch: Any) -> None:
    response, tasks, _seen = await _start(fake_app(internal_token="t"), monkeypatch)
    assert tasks[_body(response)["task_id"]]["_base_url"] == "http://127.0.0.1:28765"


@pytest.mark.parametrize("host", ["0.0.0.0", "::"])
async def test_start_test_never_calls_a_wildcard_address(
    host: str, monkeypatch: Any
) -> None:
    app = fake_app(internal_token="t", _bind_host=host, _bind_port=1)
    response, tasks, _seen = await _start(app, monkeypatch)
    assert tasks[_body(response)["task_id"]]["_base_url"] == "http://127.0.0.1:1"


async def test_start_test_forwards_the_configs_proxy(monkeypatch: Any) -> None:
    app = fake_app(
        internal_token="t", gateway_config=SimpleNamespace(proxy="http://p:1")
    )
    _response, _tasks, seen = await _start(app, monkeypatch)
    assert seen[0][4] == "http://p:1"


async def test_start_test_without_a_config_has_no_proxy(monkeypatch: Any) -> None:
    _response, _tasks, seen = await _start(fake_app(internal_token="t"), monkeypatch)
    assert seen[0][4] is None
