"""Admin session lifetimes: idle sign-out and "stay signed in".

Two clocks, deliberately: an idle window that every request slides, and a
persistent-cookie lifetime that only a remembered sign-in gets.  Both are
enforced where a session is used or minted, so a restart of the process cannot
leave a stale decision behind.
"""

from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest

from llm_rosetta_gateway._vendor.httpserver import Response
from llm_rosetta_gateway.admin.routes.auth import _set_session_cookie, admin_check
from llm_rosetta_gateway.admin.routes.config.settings import put_server_settings
from llm_rosetta_gateway.config import GatewayConfig, JsoncConfigIO
from llm_rosetta_gateway.middleware.auth import (
    DEFAULT_SESSION_IDLE_MINUTES,
    DEFAULT_SESSION_REMEMBER_DAYS,
    AuthState,
    SessionInfo,
)


def _config(server: dict | None = None) -> GatewayConfig:
    return GatewayConfig({"providers": {}, "models": {}, "server": server or {}})


def _state(**kwargs) -> AuthState:
    return AuthState(keystore=None, internal_token="t", **kwargs)


def _session(state: AuthState, session_id: str) -> SessionInfo:
    """The session record, asserted present so the test can age it."""
    info = state.get_session(session_id)
    assert info is not None
    return info


def _cookie_header(max_age: int | None = None) -> str:
    """Set the session cookie on a real response and return its header."""
    resp = Response()
    _set_session_cookie(resp, "abc", max_age=max_age)
    # The vendored Response keeps Set-Cookie lines until the response is
    # written; the line itself is what this test is about.
    return resp._cookie_headers[0]


class TestConfigParsing:
    def test_defaults(self):
        cfg = _config()
        assert cfg.admin_session_idle_minutes == DEFAULT_SESSION_IDLE_MINUTES
        assert cfg.admin_session_remember_days == DEFAULT_SESSION_REMEMBER_DAYS

    def test_explicit_values(self):
        cfg = _config({"admin_session": {"idle_minutes": 0, "remember_days": 7}})
        assert cfg.admin_session_idle_minutes == 0
        assert cfg.admin_session_remember_days == 7

    def test_negative_is_rejected(self):
        with pytest.raises(ValueError, match="0 or greater"):
            _config({"admin_session": {"idle_minutes": -1}})

    def test_non_integer_is_rejected(self):
        with pytest.raises(ValueError, match="must be an integer"):
            _config({"admin_session": {"remember_days": "soon"}})


class TestIdleWindow:
    def test_fresh_session_validates(self):
        state = _state(session_idle_minutes=30)
        sid = state.create_session(ip="127.0.0.1")
        assert state.validate_session(sid) is True

    def test_idle_session_expires(self):
        state = _state(session_idle_minutes=30)
        sid = state.create_session()
        _session(state, sid).last_seen = time.time() - 31 * 60
        assert state.validate_session(sid) is False
        # and it is gone, not merely refused once
        assert state.get_session(sid) is None

    def test_zero_means_never(self):
        state = _state(session_idle_minutes=0)
        sid = state.create_session()
        _session(state, sid).last_seen = time.time() - 365 * 86400
        assert state.validate_session(sid) is True

    def test_a_request_slides_the_window(self):
        state = _state(session_idle_minutes=30)
        sid = state.create_session()
        _session(state, sid).last_seen = time.time() - 20 * 60
        assert state.validate_session(sid) is True
        # the clock restarted, so another 20 idle minutes still validate
        _session(state, sid).last_seen -= 20 * 60
        assert state.validate_session(sid) is True

    def test_shortening_the_window_applies_at_once(self):
        state = _state(session_idle_minutes=60)
        sid = state.create_session()
        _session(state, sid).last_seen = time.time() - 45 * 60
        state.configure_sessions(idle_minutes=15)
        assert state.validate_session(sid) is False


class TestRememberedSessions:
    def test_remember_flag_is_recorded(self):
        state = _state(session_remember_days=30)
        sid = state.create_session(remember=True)
        assert _session(state, sid).remember is True
        assert state.remember_seconds == 30 * 86400

    def test_remember_is_dropped_when_the_feature_is_off(self):
        state = _state(session_remember_days=0)
        sid = state.create_session(remember=True)
        assert _session(state, sid).remember is False
        assert state.remember_seconds == 0

    def test_cookie_without_remember_has_no_max_age(self):
        header = _cookie_header()
        assert "Max-Age" not in header
        assert "HttpOnly" in header
        assert "Path=/admin" in header
        assert "SameSite=Lax" in header

    def test_a_remembered_session_ignores_the_idle_window(self):
        # "Stay signed in" promises one thing; the idle window promises another.
        # The remembered session keeps the first and not the second.
        state = _state(session_idle_minutes=15, session_remember_days=7)
        sid = state.create_session(remember=True)
        _session(state, sid).last_seen = time.time() - 120 * 60
        assert state.validate_session(sid) is True

    def test_a_remembered_session_ends_with_its_lifetime(self):
        state = _state(session_idle_minutes=15, session_remember_days=7)
        sid = state.create_session(remember=True)
        _session(state, sid).created_at = time.time() - 8 * 86400
        assert state.validate_session(sid) is False

    def test_cookie_with_remember_is_persistent(self):
        header = _cookie_header(max_age=30 * 86400)
        assert "Max-Age=2592000" in header
        assert "HttpOnly" in header


class TestAuthCheck:
    @pytest.mark.asyncio
    async def test_reports_the_lifetimes(self):
        state = _state(session_idle_minutes=15, session_remember_days=7)
        state.admin_password = "pw"
        request = SimpleNamespace(app=SimpleNamespace(auth_state=state))

        payload = json.loads((await admin_check(request)).body)

        assert payload == {
            "requires_auth": True,
            "idle_minutes": 15,
            "remember_days": 7,
        }


def _request(tmp_path, raw: dict, body: dict) -> SimpleNamespace:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    app = SimpleNamespace(
        config_path=str(path),
        config_io=JsoncConfigIO(),
        gateway_config=None,
        provider_validation_warnings=[],
        shim_source_signature=None,
        rate_limit_state=None,
    )
    return SimpleNamespace(
        app=app, path_params={}, method="PUT", headers={}, json=lambda: body
    )


class TestSettingsRoute:
    @pytest.mark.asyncio
    async def test_saves_both_lifetimes(self, tmp_path):
        request = _request(
            tmp_path,
            {"providers": {}, "models": {}},
            {"admin_session": {"idle_minutes": 15, "remember_days": 7}},
        )

        saved = json.loads((await put_server_settings(request)).body)

        assert saved["ok"] is True
        on_disk = json.loads((tmp_path / "config.json").read_text())
        assert on_disk["server"]["admin_session"] == {
            "idle_minutes": 15,
            "remember_days": 7,
        }

    @pytest.mark.asyncio
    async def test_rejects_a_negative_value(self, tmp_path):
        request = _request(
            tmp_path,
            {"providers": {}, "models": {}},
            {"admin_session": {"idle_minutes": -5}},
        )

        response = await put_server_settings(request)

        assert response.status_code == 400

    @pytest.mark.asyncio
    async def test_rejects_a_non_object(self, tmp_path):
        request = _request(
            tmp_path, {"providers": {}, "models": {}}, {"admin_session": 30}
        )

        response = await put_server_settings(request)

        assert response.status_code == 400
