"""Gateway auth hook unit tests."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest

from llm_rosetta.gateway.auth import (
    AuthState,
    api_key_label_var,
    create_auth_hook,
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
    """Build a minimal mock request matching httpserver conventions."""
    req = MagicMock()
    req.path = path
    req.method = method
    req.headers = headers or {}
    req.query_params = query_params or {}
    return req


def _run(coro: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(coro)


# ---------------------------------------------------------------------------
# No API keys configured
# ---------------------------------------------------------------------------


class TestNoApiKey:
    """When no api_key is configured, all requests pass through."""

    def test_all_requests_allowed(self):
        state = AuthState(frozenset(), {}, None)
        hook = create_auth_hook(state)

        for path in [
            "/health",
            "/v1/chat/completions",
            "/v1/messages",
            "/admin/api/config",
        ]:
            resp = _run(hook(_make_request(path)))
            assert resp is None, f"Expected pass-through for {path}"


# ---------------------------------------------------------------------------
# With API keys
# ---------------------------------------------------------------------------


class TestWithApiKey:
    """When api_key is configured, requests must provide valid credentials."""

    KEY = "test-gateway-key-123"

    @pytest.fixture()
    def hook(self):
        state = AuthState(frozenset({self.KEY}), {}, None)
        return create_auth_hook(state)

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
    """When multiple API keys are configured via api_key_set."""

    KEYS = {"key-alpha", "key-beta", "key-gamma"}

    @pytest.fixture()
    def hook(self):
        state = AuthState(frozenset(self.KEYS), {}, None)
        return create_auth_hook(state)

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
    def hook(self):
        state = AuthState(frozenset({self.KEY}), {}, self.INTERNAL)
        return create_auth_hook(state)

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
# Key label tracking
# ---------------------------------------------------------------------------


async def _run_and_get_label(hook: Any, req: Any) -> tuple[Any, str | None]:
    """Run the auth hook and return (response, label) in the same async context."""
    resp = await hook(req)
    return resp, api_key_label_var.get()


class TestKeyLabelTracking:
    """API key label is attached to contextvars for logging."""

    KEYS = {"key-prod", "key-dev"}
    LABELS = {"key-prod": "Production", "key-dev": "Development"}
    INTERNAL = "rsk-internal-test"

    @pytest.fixture()
    def hook(self):
        state = AuthState(frozenset(self.KEYS), self.LABELS, self.INTERNAL)
        return create_auth_hook(state)

    def test_label_attached_for_prod_key(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": "Bearer key-prod"},
        )
        _, label = _run(_run_and_get_label(hook, req))
        assert label == "Production"

    def test_label_attached_for_dev_key(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": "Bearer key-dev"},
        )
        _, label = _run(_run_and_get_label(hook, req))
        assert label == "Development"

    def test_internal_token_label(self, hook: Any):
        req = _make_request(
            "/v1/chat/completions",
            headers={"authorization": f"Bearer {self.INTERNAL}"},
        )
        _, label = _run(_run_and_get_label(hook, req))
        assert label == "internal"

    def test_anthropic_label(self, hook: Any):
        req = _make_request(
            "/v1/messages",
            headers={"x-api-key": "key-prod"},
        )
        _, label = _run(_run_and_get_label(hook, req))
        assert label == "Production"
