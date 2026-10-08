"""Gateway auth hook unit tests."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from llm_rosetta.gateway.middleware.auth import (
    AuthState,
    api_key_context_var,
    create_auth_hook,
)
from llm_rosetta.gateway.middleware.error_format import detect_api_format, is_admin_path
from llm_rosetta.gateway.keystore import KeyContext, KeyStore
from llm_rosetta.gateway.middleware.request_context import (
    RequestContext,
    request_context_var,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    path: str,
    method: str = "POST",
    headers: dict[str, str] | None = None,
    query_params: dict[str, list[str]] | None = None,
) -> MagicMock:
    """Build a minimal mock request matching httpserver conventions.

    Also populates ``request_context_var`` so that the auth hook can
    read ``api_format`` from the context (as it would in production
    where the context middleware runs first).
    """
    req = MagicMock()
    req.path = path
    req.method = method
    req.headers = headers or {}
    req.query_params = query_params or {}
    # Populate request context so auth can read api_format from it.
    admin = is_admin_path(path)
    api_format = None if admin else detect_api_format(path)
    request_context_var.set(
        RequestContext(
            request_id="test-id",
            client_ip="127.0.0.1",
            api_format=api_format,
            is_admin=admin,
        )
    )
    return req


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


async def _make_keystore_with_key(
    tmp_path, raw_key: str, label: str = ""
) -> tuple[KeyStore, str]:
    """Create a KeyStore with a single key and return (keystore, key_id)."""
    ks = await KeyStore.create(tmp_path / "keys.db")
    key_id, _ = await ks.create_key(label=label, manual_key=raw_key)
    return ks, key_id


async def _make_keystore_with_keys(tmp_path, keys: dict[str, str]) -> KeyStore:
    """Create a KeyStore with multiple keys {raw_key: label}."""
    ks = await KeyStore.create(tmp_path / "keys.db")
    for raw_key, label in keys.items():
        await ks.create_key(label=label, manual_key=raw_key)
    return ks


# ---------------------------------------------------------------------------
# No API keys configured
# ---------------------------------------------------------------------------


class TestNoApiKey:
    """When no api_key is configured, behavior depends on open_on_no_keys."""

    async def test_open_on_no_keys_allows_all(self):
        state = AuthState(keystore=None, internal_token=None, open_on_no_keys=True)
        hook = create_auth_hook(state)

        for path in [
            "/health",
            "/v1/chat/completions",
            "/v1/messages",
            "/admin/api/config",
        ]:
            resp = await hook(_make_request(path))
            assert resp is None, f"Expected pass-through for {path}"

    async def test_closed_on_no_keys_blocks_api(self):
        state = AuthState(
            keystore=None,
            internal_token=None,
            open_on_no_keys=False,
        )
        hook = create_auth_hook(state)

        resp = await hook(_make_request("/v1/chat/completions"))
        assert resp is not None
        assert resp.status_code == 403

    async def test_closed_on_no_keys_allows_health(self):
        state = AuthState(
            keystore=None,
            internal_token=None,
            open_on_no_keys=False,
        )
        hook = create_auth_hook(state)

        resp = await hook(_make_request("/health"))
        assert resp is None


# ---------------------------------------------------------------------------
# With API keys (KeyStore)
# ---------------------------------------------------------------------------


class TestWithApiKey:
    """When api_key is configured via KeyStore, requests must provide valid credentials."""

    KEY = "test-gateway-key-123"

    @pytest.fixture()
    async def hook(self, tmp_path):
        ks, _ = await _make_keystore_with_key(tmp_path, self.KEY)
        state = AuthState(keystore=ks, internal_token=None)
        yield create_auth_hook(state)
        await ks.close()

    # --- Health is always public ---
    async def test_health_no_auth(self, hook: Any):
        resp = await hook(_make_request("/health", method="GET"))
        assert resp is None

    # --- OpenAI Chat ---
    async def test_openai_chat_valid(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": f"Bearer {self.KEY}"},
        )
        assert await hook(req) is None

    async def test_openai_chat_missing(self, hook: Any):
        req = _make_request("/v1/chat/completions")
        resp = await hook(req)
        assert resp is not None
        assert resp.status_code == 401

    async def test_openai_chat_wrong(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": "Bearer wrong-key"},
        )
        resp = await hook(req)
        assert resp is not None
        assert resp.status_code == 401

    # --- OpenAI Responses ---
    async def test_openai_responses_valid(self, hook: Any):
        req = _make_request(
            "/v1/responses",
            headers={"authorization": f"Bearer {self.KEY}"},
        )
        assert await hook(req) is None

    # --- Anthropic ---
    async def test_anthropic_valid(self, hook: Any):
        req = _make_request(
            "/v1/messages",
            headers={"x-api-key": self.KEY},
        )
        assert await hook(req) is None

    async def test_anthropic_missing(self, hook: Any):
        req = _make_request("/v1/messages")
        resp = await hook(req)
        assert resp is not None
        assert resp.status_code == 401

    async def test_anthropic_wrong(self, hook: Any):
        req = _make_request(
            "/v1/messages",
            headers={"x-api-key": "wrong"},
        )
        resp = await hook(req)
        assert resp is not None
        assert resp.status_code == 401

    # --- Google GenAI (header) ---
    async def test_google_header_valid(self, hook: Any):
        req = _make_request(
            "/v1beta/models/gemini:generateContent",
            headers={"x-goog-api-key": self.KEY},
        )
        assert await hook(req) is None

    async def test_google_query_valid(self, hook: Any):
        req = _make_request(
            "/v1beta/models/gemini:generateContent",
            query_params={"key": [self.KEY]},
        )
        assert await hook(req) is None

    async def test_google_missing(self, hook: Any):
        req = _make_request("/v1beta/models/gemini:generateContent")
        resp = await hook(req)
        assert resp is not None
        assert resp.status_code == 401

    # --- Models list ---
    async def test_models_list_valid(self, hook: Any):
        req = _make_request(
            "/v1/models",
            method="GET",
            headers={"authorization": f"Bearer {self.KEY}"},
        )
        assert await hook(req) is None

    async def test_google_models_list_valid(self, hook: Any):
        req = _make_request(
            "/v1beta/models",
            method="GET",
            headers={"x-goog-api-key": self.KEY},
        )
        assert await hook(req) is None

    # --- Admin (no gateway-level auth) ---
    async def test_admin_html_no_auth(self, hook: Any):
        req = _make_request("/admin", method="GET")
        assert await hook(req) is None

    async def test_admin_api_no_auth(self, hook: Any):
        req = _make_request("/admin/api/config", method="GET")
        assert await hook(req) is None


# ---------------------------------------------------------------------------
# Multiple API keys
# ---------------------------------------------------------------------------


class TestMultiKey:
    """When multiple API keys are configured via KeyStore."""

    KEYS = {"key-alpha": "alpha", "key-beta": "beta", "key-gamma": "gamma"}

    @pytest.fixture()
    async def hook(self, tmp_path):
        ks = await _make_keystore_with_keys(tmp_path, self.KEYS)
        state = AuthState(keystore=ks, internal_token=None)
        yield create_auth_hook(state)
        await ks.close()

    async def test_first_key_valid(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": "Bearer key-alpha"},
        )
        assert await hook(req) is None

    async def test_second_key_valid(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": "Bearer key-beta"},
        )
        assert await hook(req) is None

    async def test_third_key_valid(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": "Bearer key-gamma"},
        )
        assert await hook(req) is None

    async def test_invalid_key_rejected(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": "Bearer wrong-key"},
        )
        resp = await hook(req)
        assert resp is not None
        assert resp.status_code == 401

    async def test_missing_key_rejected(self, hook: Any):
        req = _make_request("/v1/chat/completions")
        resp = await hook(req)
        assert resp is not None
        assert resp.status_code == 401

    async def test_anthropic_multi_key(self, hook: Any):
        req = _make_request(
            "/v1/messages",
            headers={"x-api-key": "key-beta"},
        )
        assert await hook(req) is None

    async def test_google_multi_key(self, hook: Any):
        req = _make_request(
            "/v1beta/models/gemini:generateContent",
            headers={"x-goog-api-key": "key-gamma"},
        )
        assert await hook(req) is None


# ---------------------------------------------------------------------------
# Internal token
# ---------------------------------------------------------------------------


class TestInternalToken:
    """Internal token bypasses API key auth for admin panel test requests."""

    KEY = "real-api-key"
    INTERNAL = "rsk-internal-abc123"

    @pytest.fixture()
    async def hook(self, tmp_path):
        ks, _ = await _make_keystore_with_key(tmp_path, self.KEY)
        state = AuthState(keystore=ks, internal_token=self.INTERNAL)
        yield create_auth_hook(state)
        await ks.close()

    async def test_internal_token_accepted(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": f"Bearer {self.INTERNAL}"},
        )
        assert await hook(req) is None

    async def test_real_key_still_works(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": f"Bearer {self.KEY}"},
        )
        assert await hook(req) is None

    async def test_wrong_key_still_rejected(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": "Bearer wrong"},
        )
        resp = await hook(req)
        assert resp is not None
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Key context tracking (replaces label tracking)
# ---------------------------------------------------------------------------


async def _run_and_get_context(hook: Any, req: Any) -> tuple[Any, KeyContext | None]:
    """Run the auth hook and return (response, context) in the same async context."""
    resp = await hook(req)
    return resp, api_key_context_var.get()


class TestKeyContextTracking:
    """API key context is attached to contextvars for logging."""

    KEYS = {"key-prod": "Production", "key-dev": "Development"}
    INTERNAL = "rsk-internal-test"

    @pytest.fixture()
    async def hook(self, tmp_path):
        ks = await _make_keystore_with_keys(tmp_path, self.KEYS)
        state = AuthState(keystore=ks, internal_token=self.INTERNAL)
        yield create_auth_hook(state)
        await ks.close()

    async def test_context_attached_for_prod_key(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": "Bearer key-prod"},
        )
        _, ctx = await _run_and_get_context(hook, req)
        assert ctx is not None
        assert ctx.label == "Production"

    async def test_context_attached_for_dev_key(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": "Bearer key-dev"},
        )
        _, ctx = await _run_and_get_context(hook, req)
        assert ctx is not None
        assert ctx.label == "Development"

    async def test_internal_token_context(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": f"Bearer {self.INTERNAL}"},
        )
        _, ctx = await _run_and_get_context(hook, req)
        assert ctx is not None
        assert ctx.label == "internal"
        assert ctx.allowed_shims == frozenset({"*"})

    async def test_anthropic_context(self, hook: Any):
        req = _make_request(
            "/v1/messages",
            headers={"x-api-key": "key-prod"},
        )
        _, ctx = await _run_and_get_context(hook, req)
        assert ctx is not None
        assert ctx.label == "Production"


# ---------------------------------------------------------------------------
# Admin session-based auth
# ---------------------------------------------------------------------------


class TestAdminSessionAuth:
    """Tests for session store and admin auth after HMAC removal."""

    def _make_auth_state(self) -> AuthState:
        return AuthState(
            keystore=None,
            internal_token="rsk-internal-test1234",
            admin_password="secret",
        )

    def _admin_request(
        self,
        path: str = "/admin/api/config",
        headers: dict[str, str] | None = None,
        cookies: dict[str, str] | None = None,
    ) -> MagicMock:
        req = _make_request(path, headers=headers or {})
        req.cookies = cookies or {}
        return req

    # -- Session store --

    async def test_create_and_validate_session(self):
        state = self._make_auth_state()
        sid = state.create_session(ip="1.2.3.4")
        assert state.validate_session(sid) is True
        assert state.session_count == 1

    async def test_validate_unknown_session(self):
        state = self._make_auth_state()
        assert state.validate_session("nonexistent") is False

    async def test_invalidate_session(self):
        state = self._make_auth_state()
        sid = state.create_session()
        assert state.invalidate_session(sid) is True
        assert state.validate_session(sid) is False
        assert state.invalidate_session(sid) is False

    async def test_invalidate_all_sessions(self):
        state = self._make_auth_state()
        state.create_session()
        state.create_session()
        state.create_session()
        count = state.invalidate_all_sessions()
        assert count == 3
        assert state.session_count == 0

    # -- change_password clears sessions --

    async def test_change_password_clears_sessions(self):
        state = self._make_auth_state()
        sid = state.create_session()
        state.change_password("new_secret")
        assert state.admin_password == "new_secret"
        assert state.validate_session(sid) is False
        assert state.session_count == 0

    # -- rotate does NOT clear sessions --

    async def test_rotate_preserves_sessions(self):
        state = self._make_auth_state()
        sid = state.create_session()
        old_token = state.internal_token
        new_token = state.rotate_internal_token()
        assert new_token != old_token
        assert state.internal_token == new_token
        assert state.validate_session(sid) is True

    # -- check_admin_auth with internal_token header --

    async def test_admin_auth_accepts_internal_token_header(self):
        from llm_rosetta.gateway.middleware.auth import check_admin_auth

        state = self._make_auth_state()
        req = self._admin_request(headers={"x-admin-token": "rsk-internal-test1234"})
        result = check_admin_auth(req, state)
        assert result is None  # allowed

    async def test_admin_auth_rejects_wrong_token_header(self):
        from llm_rosetta.gateway.middleware.auth import check_admin_auth

        state = self._make_auth_state()
        req = self._admin_request(headers={"x-admin-token": "wrong-token"})
        result = check_admin_auth(req, state)
        assert result is not None
        assert result.status_code == 401

    # -- check_admin_auth with session cookie --

    async def test_admin_auth_accepts_valid_session_cookie(self):
        from llm_rosetta.gateway.middleware.auth import (
            ADMIN_COOKIE_NAME,
            check_admin_auth,
        )

        state = self._make_auth_state()
        sid = state.create_session()
        req = self._admin_request(cookies={ADMIN_COOKIE_NAME: sid})
        result = check_admin_auth(req, state)
        assert result is None  # allowed

    async def test_admin_auth_rejects_invalid_cookie(self):
        from llm_rosetta.gateway.middleware.auth import (
            ADMIN_COOKIE_NAME,
            check_admin_auth,
        )

        state = self._make_auth_state()
        req = self._admin_request(cookies={ADMIN_COOKIE_NAME: "bogus"})
        result = check_admin_auth(req, state)
        assert result is not None
        assert result.status_code == 401

    # -- No password configured → pass through --

    async def test_admin_auth_no_password_allows_all(self):
        from llm_rosetta.gateway.middleware.auth import check_admin_auth

        state = AuthState(
            keystore=None, internal_token="rsk-internal-x", admin_password=None
        )
        req = self._admin_request()
        result = check_admin_auth(req, state)
        assert result is None

    # -- Login/logout/auth-check always allowed --

    async def test_admin_auth_always_allows_login(self):
        from llm_rosetta.gateway.middleware.auth import check_admin_auth

        state = self._make_auth_state()
        for path in (
            "/admin/api/login",
            "/admin/api/logout",
            "/admin/api/auth-check",
        ):
            req = self._admin_request(path=path)
            assert check_admin_auth(req, state) is None
