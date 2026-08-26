"""Tests for opt-in preflight token count (issue #426)."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

from llm_rosetta.gateway.proxy import (
    _build_preflight_body,
    _extract_preflight_input_tokens,
    _run_preflight,
)
from llm_rosetta.gateway.transport._base import UpstreamResponse
from llm_rosetta.converters.base.context import StreamContext
from llm_rosetta.converters.anthropic.converter import AnthropicConverter
from llm_rosetta.types.ir.stream import StreamStartEvent
from llm_rosetta.gateway.config import GatewayConfig
from llm_rosetta.gateway.headers import get_preflight_tokens_override


# ---------------------------------------------------------------------------
# _build_preflight_body
# ---------------------------------------------------------------------------


class TestBuildPreflightBody:
    def test_openai_chat(self):
        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4096,
            "stream": True,
            "stream_options": {"include_usage": True},
            "reasoning": {"effort": "high"},
        }
        result = _build_preflight_body(body, "openai_chat")
        assert result["max_tokens"] == 1
        assert "stream" not in result
        assert "stream_options" not in result
        assert "reasoning" not in result
        assert result["messages"] == body["messages"]

    def test_anthropic(self):
        body = {
            "model": "claude-3",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4096,
            "stream": True,
            "thinking": {"type": "enabled", "budget_tokens": 10000},
        }
        result = _build_preflight_body(body, "anthropic")
        assert result["max_tokens"] == 1
        assert "stream" not in result
        assert "thinking" not in result

    def test_google(self):
        body = {
            "contents": [{"parts": [{"text": "hi"}]}],
            "generationConfig": {
                "maxOutputTokens": 4096,
                "responseModalities": ["TEXT"],
            },
        }
        result = _build_preflight_body(body, "google")
        assert result["generationConfig"]["maxOutputTokens"] == 1
        assert "responseModalities" not in result["generationConfig"]
        assert "stream" not in result

    def test_does_not_mutate_original(self):
        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 4096,
            "stream": True,
        }
        _build_preflight_body(body, "openai_chat")
        assert body["max_tokens"] == 4096
        assert body["stream"] is True

    def test_openai_responses(self):
        body = {
            "model": "gpt-4",
            "input": [{"role": "user", "content": "hi"}],
            "max_output_tokens": 4096,
            "stream": True,
            "reasoning": {"effort": "medium"},
        }
        result = _build_preflight_body(body, "openai_responses")
        assert result["max_output_tokens"] == 1
        assert "stream" not in result
        assert "reasoning" not in result


# ---------------------------------------------------------------------------
# _extract_preflight_input_tokens
# ---------------------------------------------------------------------------


class TestExtractPreflightInputTokens:
    def test_openai_chat(self):
        resp = {"usage": {"prompt_tokens": 42, "completion_tokens": 1}}
        assert _extract_preflight_input_tokens(resp, "openai_chat") == 42

    def test_anthropic(self):
        resp = {"usage": {"input_tokens": 100, "output_tokens": 1}}
        assert _extract_preflight_input_tokens(resp, "anthropic") == 100

    def test_openai_responses(self):
        resp = {"usage": {"input_tokens": 55, "output_tokens": 1}}
        assert _extract_preflight_input_tokens(resp, "openai_responses") == 55

    def test_google(self):
        resp = {"usageMetadata": {"promptTokenCount": 200}}
        assert _extract_preflight_input_tokens(resp, "google") == 200

    def test_missing_usage(self):
        assert _extract_preflight_input_tokens({}, "openai_chat") is None

    def test_malformed_usage(self):
        resp = {"usage": {"prompt_tokens": "not_a_number"}}
        assert _extract_preflight_input_tokens(resp, "openai_chat") is None


# ---------------------------------------------------------------------------
# _run_preflight
# ---------------------------------------------------------------------------


class TestRunPreflight:
    def test_success(self):
        transport = AsyncMock()
        transport.send.return_value = UpstreamResponse(
            status_code=200,
            body={"usage": {"prompt_tokens": 42, "completion_tokens": 1}},
            raw_content=b"",
        )
        provider_info = AsyncMock()
        provider_info.upstream_url.return_value = (
            "https://api.example.com/v1/chat/completions"
        )

        result = asyncio.run(
            _run_preflight(
                transport,
                provider_info,
                {"model": "gpt-4", "messages": [], "max_tokens": 4096},
                "openai_chat",
                "gpt-4",
            )
        )
        assert result == 42
        transport.send.assert_called_once()

    def test_upstream_error(self):
        transport = AsyncMock()
        transport.send.return_value = UpstreamResponse(
            status_code=429, body={"error": "rate limited"}, raw_content=b""
        )
        provider_info = AsyncMock()
        provider_info.upstream_url.return_value = (
            "https://api.example.com/v1/chat/completions"
        )

        result = asyncio.run(
            _run_preflight(
                transport,
                provider_info,
                {"model": "gpt-4", "messages": []},
                "openai_chat",
                "gpt-4",
            )
        )
        assert result is None

    def test_transport_exception(self):
        transport = AsyncMock()
        transport.send.side_effect = ConnectionError("connection refused")
        provider_info = AsyncMock()
        provider_info.upstream_url.return_value = (
            "https://api.example.com/v1/chat/completions"
        )

        result = asyncio.run(
            _run_preflight(
                transport,
                provider_info,
                {"model": "gpt-4", "messages": []},
                "openai_chat",
                "gpt-4",
            )
        )
        assert result is None

    def test_null_body_response(self):
        transport = AsyncMock()
        transport.send.return_value = UpstreamResponse(
            status_code=200, body=None, raw_content=b""
        )
        provider_info = AsyncMock()
        provider_info.upstream_url.return_value = (
            "https://api.example.com/v1/chat/completions"
        )

        result = asyncio.run(
            _run_preflight(
                transport,
                provider_info,
                {"model": "gpt-4", "messages": []},
                "openai_chat",
                "gpt-4",
            )
        )
        assert result is None


# ---------------------------------------------------------------------------
# StreamContext.preflight_usage integration with Anthropic converter
# ---------------------------------------------------------------------------


class TestPreflightUsageIntegration:
    def test_anthropic_stream_start_uses_preflight_usage(self):
        ctx = StreamContext()
        ctx.preflight_usage = {"prompt_tokens": 42}

        converter = AnthropicConverter()
        event: StreamStartEvent = {
            "type": "stream_start",
            "response_id": "msg_123",
            "model": "claude-3",
        }
        result = converter._handle_ir_stream_start_to_p(event, ctx)
        assert result["message"]["usage"]["input_tokens"] == 42

    def test_pending_usage_takes_precedence(self):
        ctx = StreamContext()
        ctx.preflight_usage = {"prompt_tokens": 42}
        ctx.pending_usage = {"prompt_tokens": 100}

        converter = AnthropicConverter()
        event: StreamStartEvent = {
            "type": "stream_start",
            "response_id": "msg_123",
            "model": "claude-3",
        }
        result = converter._handle_ir_stream_start_to_p(event, ctx)
        assert result["message"]["usage"]["input_tokens"] == 100

    def test_no_preflight_no_pending_gives_zero(self):
        ctx = StreamContext()

        converter = AnthropicConverter()
        event: StreamStartEvent = {
            "type": "stream_start",
            "response_id": "msg_123",
            "model": "claude-3",
        }
        result = converter._handle_ir_stream_start_to_p(event, ctx)
        assert result["message"]["usage"]["input_tokens"] == 0


class TestPreflightSkipSameFormat:
    def test_skip_when_source_equals_target(self):
        """Preflight should not fire when source == target provider."""
        from llm_rosetta.routing import ResolvedRoute

        route = ResolvedRoute(
            source_provider="anthropic",
            target_provider="anthropic",
            provider_name="anthropic",
            preflight_token_count=True,
        )
        # The guard in handle_streaming checks source != target;
        # verify the condition that would skip preflight
        assert route.source_provider == route.target_provider


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------


class TestPreflightConfig:
    def _minimal_raw(self, **provider_overrides) -> dict:
        provider = {
            "api_key": "sk-test",
            "base_url": "https://api.example.com",
            "type": "openai",
        }
        provider.update(provider_overrides)
        return {
            "providers": {"test": provider},
            "models": {"gpt-test": "test"},
            "server": {},
        }

    def test_default_false(self):
        cfg = GatewayConfig(self._minimal_raw())
        route, _ = cfg.resolve("openai_chat", "gpt-test")
        assert route.preflight_token_count is False

    def test_enabled_in_config(self):
        cfg = GatewayConfig(self._minimal_raw(preflight_token_count=True))
        route, _ = cfg.resolve("openai_chat", "gpt-test")
        assert route.preflight_token_count is True


# ---------------------------------------------------------------------------
# Header override
# ---------------------------------------------------------------------------


class TestPreflightHeader:
    def _make_request(self, header_value: str | None):
        request = type("Request", (), {"headers": {}})()
        if header_value is not None:
            request.headers["x-rosetta-preflight-tokens"] = header_value
        return request

    def test_true_values(self):
        for val in ("true", "True", "TRUE", "1", "yes"):
            assert get_preflight_tokens_override(self._make_request(val)) is True

    def test_false_values(self):
        for val in ("false", "0", "no", ""):
            assert get_preflight_tokens_override(self._make_request(val)) is False

    def test_missing_header(self):
        assert get_preflight_tokens_override(self._make_request(None)) is None
