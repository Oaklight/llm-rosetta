"""Tests for the gateway rate-limiting middleware hooks."""

from __future__ import annotations

import asyncio
import json
from typing import Any, cast
from unittest.mock import MagicMock

from llm_rosetta.gateway.config import GatewayConfig
from llm_rosetta.gateway.ratelimit import (
    RateLimitState,
    _detect_format,
    _extract_client_ip,
    _extract_model,
    _rate_limit_response,
    create_rate_limit_after_hook,
    create_rate_limit_hook,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SENTINEL = object()


class FakeRequest:
    """Minimal request mock matching httpserver.Request interface."""

    def __init__(
        self,
        path: str = "/v1/chat/completions",
        method: str = "POST",
        body: dict | None = None,
        headers: dict[str, str] | None = None,
        client_addr: tuple[str, int] = ("127.0.0.1", 12345),
    ):
        self.path = path
        self.method = method
        self._body = body or {}
        self.headers = headers or {}
        self.client_addr = client_addr
        self._json_cache = _SENTINEL

    def json(self) -> Any:
        if self._json_cache is _SENTINEL:
            self._json_cache = self._body
        return self._json_cache


class FakeConfig:
    """Minimal GatewayConfig-like object for RateLimitState.rebuild()."""

    def __init__(
        self,
        enabled: bool = True,
        algorithm: str = "sliding_window",
        global_quota: str | None = None,
        per_ip: str | None = None,
        per_key: str | None = None,
        per_model: str | None = None,
        exclude: list[str] | None = None,
    ):
        self.rate_limit_enabled = enabled
        self.rate_limit_algorithm = algorithm
        self.rate_limit_global = global_quota
        self.rate_limit_per_ip = per_ip
        self.rate_limit_per_key = per_key
        self.rate_limit_per_model = per_model
        self.rate_limit_exclude = (
            exclude if exclude is not None else ["/health", "/admin"]
        )


def _run(coro):
    return asyncio.run(coro)


# ---------------------------------------------------------------------------
# Format detection
# ---------------------------------------------------------------------------


class TestDetectFormat:
    def test_openai_chat(self):
        assert _detect_format("/v1/chat/completions") == "openai"

    def test_openai_responses(self):
        assert _detect_format("/v1/responses") == "openai"

    def test_anthropic(self):
        assert _detect_format("/v1/messages") == "anthropic"

    def test_google(self):
        assert _detect_format("/v1beta/models/gemini:generateContent") == "google"

    def test_unknown_defaults_openai(self):
        assert _detect_format("/unknown/path") == "openai"


# ---------------------------------------------------------------------------
# Client IP extraction
# ---------------------------------------------------------------------------


class TestExtractClientIp:
    def test_x_forwarded_for_single(self):
        req = FakeRequest(headers={"x-forwarded-for": "1.2.3.4"})
        assert _extract_client_ip(req) == "1.2.3.4"

    def test_x_forwarded_for_chain(self):
        req = FakeRequest(headers={"x-forwarded-for": "1.2.3.4, 5.6.7.8"})
        assert _extract_client_ip(req) == "1.2.3.4"

    def test_x_real_ip(self):
        req = FakeRequest(headers={"x-real-ip": "10.0.0.1"})
        assert _extract_client_ip(req) == "10.0.0.1"

    def test_client_addr_fallback(self):
        req = FakeRequest(client_addr=("192.168.1.1", 9999))
        assert _extract_client_ip(req) == "192.168.1.1"


# ---------------------------------------------------------------------------
# Model extraction
# ---------------------------------------------------------------------------


class TestExtractModel:
    def test_from_body(self):
        req = FakeRequest(body={"model": "gpt-4"})
        assert _extract_model(req) == "gpt-4"

    def test_from_google_url(self):
        req = FakeRequest(path="/v1beta/models/gemini-pro:generateContent", body={})
        assert _extract_model(req) == "gemini-pro"

    def test_no_model(self):
        req = FakeRequest(path="/v1/chat/completions", body={})
        assert _extract_model(req) is None


# ---------------------------------------------------------------------------
# Rate limit response format
# ---------------------------------------------------------------------------


class TestRateLimitResponse:
    def _make_result(self):
        from llm_rosetta.gateway.ratelimit import RateLimitResult

        return RateLimitResult(
            allowed=False,
            limit=100,
            remaining=0,
            reset_at=1000.5,
            retry_after=5.3,
        )

    def test_openai_format(self):
        resp = _rate_limit_response(
            "/v1/chat/completions", self._make_result(), "per_ip"
        )
        assert resp.status_code == 429
        body = json.loads(resp.body)
        assert body["error"]["type"] == "rate_limit_error"
        assert body["error"]["code"] == "rate_limit_exceeded"
        assert "per_ip" in body["error"]["message"]

    def test_anthropic_format(self):
        resp = _rate_limit_response("/v1/messages", self._make_result(), "global")
        body = json.loads(resp.body)
        assert body["type"] == "error"
        assert body["error"]["type"] == "rate_limit_error"

    def test_google_format(self):
        resp = _rate_limit_response(
            "/v1beta/models/gemini:gen", self._make_result(), "per_model"
        )
        body = json.loads(resp.body)
        assert body["error"]["code"] == 429
        assert body["error"]["status"] == "RESOURCE_EXHAUSTED"

    def test_headers(self):
        resp = _rate_limit_response("/v1/chat/completions", self._make_result(), "test")
        assert resp.headers["Retry-After"] == "6"  # ceil(5.3)
        assert resp.headers["X-RateLimit-Limit"] == "100"
        assert resp.headers["X-RateLimit-Remaining"] == "0"
        assert "X-RateLimit-Reset" in resp.headers

    def test_cors_headers_on_api_path(self):
        resp = _rate_limit_response("/v1/chat/completions", self._make_result(), "test")
        assert resp.headers["Access-Control-Allow-Origin"] == "*"

    def test_no_cors_on_admin_path(self):
        resp = _rate_limit_response("/admin/api/test", self._make_result(), "test")
        assert "Access-Control-Allow-Origin" not in resp.headers


# ---------------------------------------------------------------------------
# RateLimitState
# ---------------------------------------------------------------------------


class TestRateLimitState:
    def test_default_disabled(self):
        state = RateLimitState()
        assert state.enabled is False
        assert state._global is None

    def test_rebuild_enabled(self):
        state = RateLimitState()
        config = FakeConfig(enabled=True, global_quota="100/m", per_ip="60/m")
        state.rebuild(cast(GatewayConfig, config))
        assert state.enabled is True
        assert state._global is not None
        assert state._per_ip is not None
        assert state._per_key is None
        assert state._per_model is None

    def test_rebuild_disabled(self):
        state = RateLimitState()
        state.rebuild(
            cast(GatewayConfig, FakeConfig(enabled=True, global_quota="100/m"))
        )
        assert state._global is not None
        state.rebuild(cast(GatewayConfig, FakeConfig(enabled=False)))
        assert state.enabled is False
        assert state._global is None

    def test_rebuild_resets_counters(self):
        state = RateLimitState()
        config = FakeConfig(enabled=True, global_quota="2/m")
        state.rebuild(cast(GatewayConfig, config))
        assert state._global is not None
        state._global.acquire("k")
        state._global.acquire("k")
        r = state._global.acquire("k")
        assert not r.allowed
        state.rebuild(cast(GatewayConfig, config))
        assert state._global is not None
        r = state._global.acquire("k")
        assert r.allowed


# ---------------------------------------------------------------------------
# Hook: before_request
# ---------------------------------------------------------------------------


class TestRateLimitHook:
    def _make_state(self, **kwargs):
        state = RateLimitState()
        state.rebuild(cast(GatewayConfig, FakeConfig(**kwargs)))
        return state

    def test_disabled_returns_none(self):
        state = self._make_state(enabled=False)
        hook = create_rate_limit_hook(state)
        req = FakeRequest()
        result = _run(hook(req))
        assert result is None

    def test_options_skipped(self):
        state = self._make_state(enabled=True, global_quota="1/m")
        hook = create_rate_limit_hook(state)
        req = FakeRequest(method="OPTIONS")
        result = _run(hook(req))
        assert result is None

    def test_health_excluded(self):
        state = self._make_state(enabled=True, global_quota="1/m")
        hook = create_rate_limit_hook(state)
        req = FakeRequest(path="/health")
        result = _run(hook(req))
        assert result is None

    def test_admin_excluded(self):
        state = self._make_state(enabled=True, global_quota="1/m")
        hook = create_rate_limit_hook(state)
        req = FakeRequest(path="/admin/api/config")
        result = _run(hook(req))
        assert result is None

    def test_allows_under_limit(self):
        state = self._make_state(enabled=True, global_quota="100/m")
        hook = create_rate_limit_hook(state)
        req = FakeRequest()
        result = _run(hook(req))
        assert result is None

    def test_denies_over_global_limit(self):
        state = self._make_state(enabled=True, global_quota="2/m")
        hook = create_rate_limit_hook(state)
        _run(hook(FakeRequest()))
        _run(hook(FakeRequest()))
        result = _run(hook(FakeRequest()))
        assert result is not None
        assert result.status_code == 429

    def test_per_ip_isolation(self):
        state = self._make_state(enabled=True, per_ip="1/m")
        hook = create_rate_limit_hook(state)
        r1 = _run(hook(FakeRequest(client_addr=("1.1.1.1", 1))))
        assert r1 is None
        r2 = _run(hook(FakeRequest(client_addr=("2.2.2.2", 1))))
        assert r2 is None
        r3 = _run(hook(FakeRequest(client_addr=("1.1.1.1", 1))))
        assert r3 is not None
        assert r3.status_code == 429

    def test_per_model_isolation(self):
        state = self._make_state(enabled=True, per_model="1/m")
        hook = create_rate_limit_hook(state)
        r1 = _run(hook(FakeRequest(body={"model": "gpt-4"})))
        assert r1 is None
        r2 = _run(hook(FakeRequest(body={"model": "claude"})))
        assert r2 is None
        r3 = _run(hook(FakeRequest(body={"model": "gpt-4"})))
        assert r3 is not None

    def test_no_model_skips_per_model(self):
        state = self._make_state(enabled=True, per_model="1/m")
        hook = create_rate_limit_hook(state)
        r1 = _run(hook(FakeRequest(path="/v1/models", method="GET", body={})))
        assert r1 is None
        r2 = _run(hook(FakeRequest(path="/v1/models", method="GET", body={})))
        assert r2 is None

    def test_per_key_uses_label(self):
        from llm_rosetta.gateway.auth import api_key_context_var
        from llm_rosetta.gateway.keystore import KeyContext

        state = self._make_state(enabled=True, per_key="1/m")
        hook = create_rate_limit_hook(state)
        ctx = KeyContext(label="user-a", allowed_shims=frozenset())
        token = api_key_context_var.set(ctx)
        try:
            r1 = _run(hook(FakeRequest()))
            assert r1 is None
            r2 = _run(hook(FakeRequest()))
            assert r2 is not None
            assert r2.status_code == 429
        finally:
            api_key_context_var.reset(token)

    def test_no_key_skips_per_key(self):
        from llm_rosetta.gateway.auth import api_key_context_var

        state = self._make_state(enabled=True, per_key="1/m")
        hook = create_rate_limit_hook(state)
        token = api_key_context_var.set(None)
        try:
            r1 = _run(hook(FakeRequest()))
            assert r1 is None
            r2 = _run(hook(FakeRequest()))
            assert r2 is None
        finally:
            api_key_context_var.reset(token)


# ---------------------------------------------------------------------------
# Hook: after_request (headers on success)
# ---------------------------------------------------------------------------


class TestRateLimitAfterHook:
    def test_adds_headers_when_result_set(self):
        from llm_rosetta.gateway.ratelimit import (
            _rate_limit_result_var,
            RateLimitResult,
        )

        result = RateLimitResult(
            allowed=True, limit=100, remaining=95, reset_at=1000.0, retry_after=None
        )
        token = _rate_limit_result_var.set(result)
        try:
            hook = create_rate_limit_after_hook()
            response = MagicMock()
            response.headers = {}
            _run(hook(FakeRequest(), response))
            assert response.headers["X-RateLimit-Limit"] == "100"
            assert response.headers["X-RateLimit-Remaining"] == "95"
        finally:
            _rate_limit_result_var.reset(token)

    def test_no_headers_when_no_result(self):
        from llm_rosetta.gateway.ratelimit import _rate_limit_result_var

        token = _rate_limit_result_var.set(None)
        try:
            hook = create_rate_limit_after_hook()
            response = MagicMock()
            response.headers = {}
            _run(hook(FakeRequest(), response))
            assert "X-RateLimit-Limit" not in response.headers
        finally:
            _rate_limit_result_var.reset(token)


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestRateLimitConfig:
    def test_defaults(self):
        from llm_rosetta.gateway.config import GatewayConfig

        config = GatewayConfig(
            {
                "providers": {"test": {"api_key": "k", "base_url": "http://x"}},
                "models": {"m": "test"},
            }
        )
        assert config.rate_limit_enabled is False
        assert config.rate_limit_algorithm == "sliding_window"
        assert config.rate_limit_global is None
        assert config.rate_limit_per_ip is None
        assert config.rate_limit_exclude == ["/health", "/admin"]

    def test_parses_all_fields(self):
        from llm_rosetta.gateway.config import GatewayConfig

        raw = {
            "providers": {"test": {"api_key": "k", "base_url": "http://x"}},
            "models": {"m": "test"},
            "server": {
                "rate_limit": {
                    "enabled": True,
                    "algorithm": "token_bucket",
                    "global": "600/m",
                    "per_ip": "120/m",
                    "per_key": "60/m",
                    "per_model": "30/m",
                    "exclude_paths": ["/health"],
                }
            },
        }
        config = GatewayConfig(raw)
        assert config.rate_limit_enabled is True
        assert config.rate_limit_algorithm == "token_bucket"
        assert config.rate_limit_global == "600/m"
        assert config.rate_limit_per_ip == "120/m"
        assert config.rate_limit_per_key == "60/m"
        assert config.rate_limit_per_model == "30/m"
        assert config.rate_limit_exclude == ["/health"]

    def test_invalid_algorithm_falls_back(self):
        from llm_rosetta.gateway.config import GatewayConfig

        raw = {
            "providers": {"test": {"api_key": "k", "base_url": "http://x"}},
            "models": {"m": "test"},
            "server": {"rate_limit": {"algorithm": "invalid"}},
        }
        config = GatewayConfig(raw)
        assert config.rate_limit_algorithm == "sliding_window"
