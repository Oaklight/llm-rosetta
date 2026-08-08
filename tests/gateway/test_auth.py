"""Gateway auth hook unit tests."""

from __future__ import annotations

import asyncio
import hashlib
from typing import Any
from unittest.mock import MagicMock

import pytest

from llm_rosetta.gateway.auth import (
    AuthState,
    _build_config_fallback,
    api_key_context_var,
    create_auth_hook,
)
from llm_rosetta.gateway.keystore import KeyContext, KeyStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_request(
    path: str,
    method: str = "POST",
    headers: dict[str, str] | None = None,
    query_params: dict[str, list[str]] | None = None,
) -> MagicMock:
    """Build a minimal mock request matching httpserver conventions."""
    req = MagicMock()
    req.path = path
    req.method = method
    req.headers = headers or {}
    req.query_params = query_params or {}
    return req


def _run(coro: Any) -> Any:
    return asyncio.run(coro)


def _make_keystore_with_key(
    tmp_path, raw_key: str, label: str = ""
) -> tuple[KeyStore, str]:
    """Create a KeyStore with a single key and return (keystore, key_id)."""
    ks = KeyStore(tmp_path / "keys.db")
    key_id, _ = ks.create(label=label, manual_key=raw_key)
    return ks, key_id


def _make_keystore_with_keys(tmp_path, keys: dict[str, str]) -> KeyStore:
    """Create a KeyStore with multiple keys {raw_key: label}."""
    ks = KeyStore(tmp_path / "keys.db")
    for raw_key, label in keys.items():
        ks.create(label=label, manual_key=raw_key)
    return ks


# ---------------------------------------------------------------------------
# No API keys configured
# ---------------------------------------------------------------------------


class TestNoApiKey:
    """When no api_key is configured, behavior depends on open_on_no_keys."""

    def test_open_on_no_keys_allows_all(self):
        state = AuthState(
            keystore=None, config_fallback={}, internal_token=None, open_on_no_keys=True
        )
        hook = create_auth_hook(state)

        for path in [
            "/health",
            "/v1/chat/completions",
            "/v1/messages",
            "/admin/api/config",
        ]:
            resp = _run(hook(_make_request(path)))
            assert resp is None, f"Expected pass-through for {path}"

    def test_closed_on_no_keys_blocks_api(self):
        state = AuthState(
            keystore=None,
            config_fallback={},
            internal_token=None,
            open_on_no_keys=False,
        )
        hook = create_auth_hook(state)

        resp = _run(hook(_make_request("/v1/chat/completions")))
        assert resp is not None
        assert resp.status_code == 403

    def test_closed_on_no_keys_allows_health(self):
        state = AuthState(
            keystore=None,
            config_fallback={},
            internal_token=None,
            open_on_no_keys=False,
        )
        hook = create_auth_hook(state)

        resp = _run(hook(_make_request("/health")))
        assert resp is None


# ---------------------------------------------------------------------------
# With API keys (KeyStore)
# ---------------------------------------------------------------------------


class TestWithApiKey:
    """When api_key is configured via KeyStore, requests must provide valid credentials."""

    KEY = "test-gateway-key-123"

    @pytest.fixture()
    def hook(self, tmp_path):
        ks, _ = _make_keystore_with_key(tmp_path, self.KEY)
        state = AuthState(keystore=ks, config_fallback={}, internal_token=None)
        yield create_auth_hook(state)
        ks.close()

    # --- Health is always public ---
    def test_health_no_auth(self, hook: Any):
        resp = _run(hook(_make_request("/health", method="GET")))
        assert resp is None

    # --- OpenAI Chat ---
    def test_openai_chat_valid(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": f"Bearer {self.KEY}"},
        )
        assert _run(hook(req)) is None

    def test_openai_chat_missing(self, hook: Any):
        req = _make_request("/v1/chat/completions")
        resp = _run(hook(req))
        assert resp is not None
        assert resp.status_code == 401

    def test_openai_chat_wrong(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": "Bearer wrong-key"},
        )
        resp = _run(hook(req))
        assert resp is not None
        assert resp.status_code == 401

    # --- OpenAI Responses ---
    def test_openai_responses_valid(self, hook: Any):
        req = _make_request(
            "/v1/responses",
            headers={"authorization": f"Bearer {self.KEY}"},
        )
        assert _run(hook(req)) is None

    # --- Anthropic ---
    def test_anthropic_valid(self, hook: Any):
        req = _make_request(
            "/v1/messages",
            headers={"x-api-key": self.KEY},
        )
        assert _run(hook(req)) is None

    def test_anthropic_missing(self, hook: Any):
        req = _make_request("/v1/messages")
        resp = _run(hook(req))
        assert resp is not None
        assert resp.status_code == 401

    def test_anthropic_wrong(self, hook: Any):
        req = _make_request(
            "/v1/messages",
            headers={"x-api-key": "wrong"},
        )
        resp = _run(hook(req))
        assert resp is not None
        assert resp.status_code == 401

    # --- Google GenAI (header) ---
    def test_google_header_valid(self, hook: Any):
        req = _make_request(
            "/v1beta/models/gemini:generateContent",
            headers={"x-goog-api-key": self.KEY},
        )
        assert _run(hook(req)) is None

    def test_google_query_valid(self, hook: Any):
        req = _make_request(
            "/v1beta/models/gemini:generateContent",
            query_params={"key": [self.KEY]},
        )
        assert _run(hook(req)) is None

    def test_google_missing(self, hook: Any):
        req = _make_request("/v1beta/models/gemini:generateContent")
        resp = _run(hook(req))
        assert resp is not None
        assert resp.status_code == 401

    # --- Models list ---
    def test_models_list_valid(self, hook: Any):
        req = _make_request(
            "/v1/models",
            method="GET",
            headers={"authorization": f"Bearer {self.KEY}"},
        )
        assert _run(hook(req)) is None

    def test_google_models_list_valid(self, hook: Any):
        req = _make_request(
            "/v1beta/models",
            method="GET",
            headers={"x-goog-api-key": self.KEY},
        )
        assert _run(hook(req)) is None

    # --- Admin (no gateway-level auth) ---
    def test_admin_html_no_auth(self, hook: Any):
        req = _make_request("/admin", method="GET")
        assert _run(hook(req)) is None

    def test_admin_api_no_auth(self, hook: Any):
        req = _make_request("/admin/api/config", method="GET")
        assert _run(hook(req)) is None


# ---------------------------------------------------------------------------
# Multiple API keys
# ---------------------------------------------------------------------------


class TestMultiKey:
    """When multiple API keys are configured via KeyStore."""

    KEYS = {"key-alpha": "alpha", "key-beta": "beta", "key-gamma": "gamma"}

    @pytest.fixture()
    def hook(self, tmp_path):
        ks = _make_keystore_with_keys(tmp_path, self.KEYS)
        state = AuthState(keystore=ks, config_fallback={}, internal_token=None)
        yield create_auth_hook(state)
        ks.close()

    def test_first_key_valid(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": "Bearer key-alpha"},
        )
        assert _run(hook(req)) is None

    def test_second_key_valid(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": "Bearer key-beta"},
        )
        assert _run(hook(req)) is None

    def test_third_key_valid(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": "Bearer key-gamma"},
        )
        assert _run(hook(req)) is None

    def test_invalid_key_rejected(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": "Bearer wrong-key"},
        )
        resp = _run(hook(req))
        assert resp is not None
        assert resp.status_code == 401

    def test_missing_key_rejected(self, hook: Any):
        req = _make_request("/v1/chat/completions")
        resp = _run(hook(req))
        assert resp is not None
        assert resp.status_code == 401

    def test_anthropic_multi_key(self, hook: Any):
        req = _make_request(
            "/v1/messages",
            headers={"x-api-key": "key-beta"},
        )
        assert _run(hook(req)) is None

    def test_google_multi_key(self, hook: Any):
        req = _make_request(
            "/v1beta/models/gemini:generateContent",
            headers={"x-goog-api-key": "key-gamma"},
        )
        assert _run(hook(req)) is None


# ---------------------------------------------------------------------------
# Internal token
# ---------------------------------------------------------------------------


class TestInternalToken:
    """Internal token bypasses API key auth for admin panel test requests."""

    KEY = "real-api-key"
    INTERNAL = "rsk-internal-abc123"

    @pytest.fixture()
    def hook(self, tmp_path):
        ks, _ = _make_keystore_with_key(tmp_path, self.KEY)
        state = AuthState(keystore=ks, config_fallback={}, internal_token=self.INTERNAL)
        yield create_auth_hook(state)
        ks.close()

    def test_internal_token_accepted(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": f"Bearer {self.INTERNAL}"},
        )
        assert _run(hook(req)) is None

    def test_real_key_still_works(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": f"Bearer {self.KEY}"},
        )
        assert _run(hook(req)) is None

    def test_wrong_key_still_rejected(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": "Bearer wrong"},
        )
        resp = _run(hook(req))
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
    def hook(self, tmp_path):
        ks = _make_keystore_with_keys(tmp_path, self.KEYS)
        state = AuthState(keystore=ks, config_fallback={}, internal_token=self.INTERNAL)
        yield create_auth_hook(state)
        ks.close()

    def test_context_attached_for_prod_key(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": "Bearer key-prod"},
        )
        _, ctx = _run(_run_and_get_context(hook, req))
        assert ctx is not None
        assert ctx.label == "Production"

    def test_context_attached_for_dev_key(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": "Bearer key-dev"},
        )
        _, ctx = _run(_run_and_get_context(hook, req))
        assert ctx is not None
        assert ctx.label == "Development"

    def test_internal_token_context(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": f"Bearer {self.INTERNAL}"},
        )
        _, ctx = _run(_run_and_get_context(hook, req))
        assert ctx is not None
        assert ctx.label == "internal"
        assert ctx.allowed_shims == frozenset({"*"})

    def test_anthropic_context(self, hook: Any):
        req = _make_request(
            "/v1/messages",
            headers={"x-api-key": "key-prod"},
        )
        _, ctx = _run(_run_and_get_context(hook, req))
        assert ctx is not None
        assert ctx.label == "Production"


# ---------------------------------------------------------------------------
# Config fallback
# ---------------------------------------------------------------------------


class TestConfigFallback:
    """Keys only in config (not KeyStore) still authenticate via fallback."""

    CONFIG_KEY = "config-only-key"

    @pytest.fixture()
    def hook(self, tmp_path):
        ks = KeyStore(tmp_path / "empty.db")
        fallback = _build_config_fallback(
            [
                {"key": self.CONFIG_KEY, "label": "from-config"},
            ]
        )
        state = AuthState(keystore=ks, config_fallback=fallback, internal_token=None)
        yield create_auth_hook(state)
        ks.close()

    def test_config_fallback_key_accepted(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": f"Bearer {self.CONFIG_KEY}"},
        )
        assert _run(hook(req)) is None

    def test_config_fallback_sets_context(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": f"Bearer {self.CONFIG_KEY}"},
        )
        _, ctx = _run(_run_and_get_context(hook, req))
        assert ctx is not None
        assert ctx.label == "from-config"

    def test_invalid_key_rejected_with_fallback(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": "Bearer wrong"},
        )
        resp = _run(hook(req))
        assert resp is not None
        assert resp.status_code == 401


# ---------------------------------------------------------------------------
# Build config fallback helper
# ---------------------------------------------------------------------------


class TestBuildConfigFallback:
    def test_builds_hash_map(self):
        keys = [{"key": "secret", "label": "test"}]
        fb = _build_config_fallback(keys)
        expected_hash = hashlib.sha256(b"secret").hexdigest()
        assert expected_hash in fb
        assert fb[expected_hash].label == "test"
        assert fb[expected_hash].allowed_shims == frozenset({"*"})

    def test_skips_empty_keys(self):
        keys = [{"key": "", "label": "empty"}]
        fb = _build_config_fallback(keys)
        assert len(fb) == 0
