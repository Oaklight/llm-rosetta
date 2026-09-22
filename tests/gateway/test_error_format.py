"""Tests for the shared gateway error response formatting module."""

from __future__ import annotations

import json

from llm_rosetta.gateway.error_format import (
    apply_cors_headers,
    detect_api_format,
    detect_api_format_from_provider,
    format_error_response,
    is_admin_path,
)


# ---------------------------------------------------------------------------
# detect_api_format (path-based)
# ---------------------------------------------------------------------------


class TestDetectApiFormat:
    def test_openai_chat(self):
        assert detect_api_format("/v1/chat/completions") == "openai"

    def test_openai_responses(self):
        assert detect_api_format("/v1/responses") == "openai"

    def test_openai_models(self):
        assert detect_api_format("/v1/models") == "openai"

    def test_anthropic(self):
        assert detect_api_format("/v1/messages") == "anthropic"

    def test_google(self):
        assert detect_api_format("/v1beta/models/gemini:generateContent") == "google"

    def test_google_list(self):
        assert detect_api_format("/v1beta/models") == "google"

    def test_unknown_defaults_openai(self):
        assert detect_api_format("/unknown/path") == "openai"

    def test_empty_path(self):
        assert detect_api_format("") == "openai"

    def test_admin_path(self):
        assert detect_api_format("/admin/api/config") == "openai"


# ---------------------------------------------------------------------------
# detect_api_format_from_provider
# ---------------------------------------------------------------------------


class TestDetectApiFormatFromProvider:
    def test_openai_chat(self):
        assert detect_api_format_from_provider("openai_chat") == "openai"

    def test_openai_responses(self):
        assert detect_api_format_from_provider("openai_responses") == "openai"

    def test_open_responses(self):
        assert detect_api_format_from_provider("open_responses") == "openai"

    def test_anthropic(self):
        assert detect_api_format_from_provider("anthropic") == "anthropic"

    def test_google(self):
        assert detect_api_format_from_provider("google") == "google"

    def test_google_generate(self):
        assert detect_api_format_from_provider("google_generate") == "google"

    def test_google_interactions(self):
        assert detect_api_format_from_provider("google_interactions") == "google"

    def test_unknown_defaults_openai(self):
        assert detect_api_format_from_provider("some_unknown") == "openai"


# ---------------------------------------------------------------------------
# is_admin_path
# ---------------------------------------------------------------------------


class TestIsAdminPath:
    def test_admin_root(self):
        assert is_admin_path("/admin") is True

    def test_admin_subpath(self):
        assert is_admin_path("/admin/api/config") is True

    def test_admin_trailing_slash(self):
        assert is_admin_path("/admin/") is True

    def test_non_admin(self):
        assert is_admin_path("/v1/chat/completions") is False

    def test_health(self):
        assert is_admin_path("/health") is False

    def test_admin_lookalike(self):
        # "/administrator" should NOT be treated as admin
        assert is_admin_path("/administrator") is False


# ---------------------------------------------------------------------------
# format_error_response — OpenAI format
# ---------------------------------------------------------------------------


class TestFormatErrorResponseOpenAI:
    def test_default_openai_envelope(self):
        resp = format_error_response("openai", 400, "bad request")
        assert resp.status_code == 400
        body = json.loads(resp.body)
        assert body == {
            "error": {
                "message": "bad request",
                "type": "invalid_request_error",
                "code": None,
            }
        }

    def test_custom_error_type_and_code(self):
        resp = format_error_response(
            "openai",
            401,
            "invalid key",
            error_type="invalid_request_error",
            error_code="invalid_api_key",
        )
        body = json.loads(resp.body)
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["code"] == "invalid_api_key"

    def test_rate_limit_error_type(self):
        resp = format_error_response(
            "openai",
            429,
            "rate limited",
            error_type="rate_limit_error",
            error_code="rate_limit_exceeded",
        )
        body = json.loads(resp.body)
        assert body["error"]["type"] == "rate_limit_error"
        assert body["error"]["code"] == "rate_limit_exceeded"


# ---------------------------------------------------------------------------
# format_error_response — Anthropic format
# ---------------------------------------------------------------------------


class TestFormatErrorResponseAnthropic:
    def test_default_anthropic_envelope(self):
        resp = format_error_response("anthropic", 400, "bad request")
        assert resp.status_code == 400
        body = json.loads(resp.body)
        assert body == {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": "bad request"},
        }

    def test_auth_error_type(self):
        resp = format_error_response(
            "anthropic",
            401,
            "invalid key",
            error_type="authentication_error",
        )
        body = json.loads(resp.body)
        assert body["error"]["type"] == "authentication_error"
        assert body["type"] == "error"

    def test_rate_limit_error_type(self):
        resp = format_error_response(
            "anthropic", 429, "rate limited", error_type="rate_limit_error"
        )
        body = json.loads(resp.body)
        assert body["error"]["type"] == "rate_limit_error"

    def test_error_code_ignored(self):
        """Anthropic format should not include error_code in the body."""
        resp = format_error_response(
            "anthropic",
            400,
            "test",
            error_code="should_not_appear",
        )
        body = json.loads(resp.body)
        assert "code" not in body["error"]


# ---------------------------------------------------------------------------
# format_error_response — Google format
# ---------------------------------------------------------------------------


class TestFormatErrorResponseGoogle:
    def test_default_google_envelope(self):
        resp = format_error_response("google", 400, "bad request")
        assert resp.status_code == 400
        body = json.loads(resp.body)
        assert body == {
            "error": {
                "code": 400,
                "message": "bad request",
                "status": "INVALID_ARGUMENT",
            }
        }

    def test_google_status_code_in_body(self):
        """Google format embeds the HTTP status code inside the error body."""
        resp = format_error_response("google", 429, "rate limited")
        body = json.loads(resp.body)
        assert body["error"]["code"] == 429

    def test_custom_google_status(self):
        resp = format_error_response(
            "google",
            401,
            "unauthenticated",
            google_status="UNAUTHENTICATED",
        )
        body = json.loads(resp.body)
        assert body["error"]["status"] == "UNAUTHENTICATED"

    def test_resource_exhausted(self):
        resp = format_error_response(
            "google",
            429,
            "rate limited",
            google_status="RESOURCE_EXHAUSTED",
        )
        body = json.loads(resp.body)
        assert body["error"]["status"] == "RESOURCE_EXHAUSTED"


# ---------------------------------------------------------------------------
# CORS headers
# ---------------------------------------------------------------------------


class TestCorsHeaders:
    def test_cors_disabled_by_default(self):
        resp = format_error_response("openai", 400, "test")
        assert "Access-Control-Allow-Origin" not in resp.headers

    def test_cors_enabled(self):
        resp = format_error_response("openai", 400, "test", cors=True)
        assert resp.headers["Access-Control-Allow-Origin"] == "*"
        assert resp.headers["Access-Control-Allow-Methods"] == "*"
        assert resp.headers["Access-Control-Allow-Headers"] == "*"

    def test_cors_disabled(self):
        resp = format_error_response("openai", 400, "test", cors=False)
        assert "Access-Control-Allow-Origin" not in resp.headers

    def test_apply_cors_headers_standalone(self):
        from llm_rosetta._vendor.httpserver import Response

        resp = Response(body=b"", status_code=200)
        apply_cors_headers(resp)
        assert resp.headers["Access-Control-Allow-Origin"] == "*"
        assert resp.headers["Access-Control-Allow-Methods"] == "*"
        assert resp.headers["Access-Control-Allow-Headers"] == "*"


# ---------------------------------------------------------------------------
# Extra headers
# ---------------------------------------------------------------------------


class TestExtraHeaders:
    def test_extra_headers_applied(self):
        resp = format_error_response(
            "openai",
            429,
            "rate limited",
            extra_headers={
                "Retry-After": "5",
                "X-RateLimit-Limit": "100",
            },
        )
        assert resp.headers["Retry-After"] == "5"
        assert resp.headers["X-RateLimit-Limit"] == "100"

    def test_extra_headers_with_cors(self):
        resp = format_error_response(
            "openai",
            429,
            "rate limited",
            extra_headers={"Retry-After": "5"},
            cors=True,
        )
        assert resp.headers["Retry-After"] == "5"
        assert resp.headers["Access-Control-Allow-Origin"] == "*"

    def test_no_extra_headers(self):
        resp = format_error_response("openai", 400, "test")
        assert "Retry-After" not in resp.headers


# ---------------------------------------------------------------------------
# Behavioral parity — proxy.py (error_response_for_source)
# ---------------------------------------------------------------------------


class TestProxyParity:
    """Verify that the shared module produces identical output to the
    old ``error_response_for_source`` for every provider type."""

    def test_openai_chat(self):
        resp = format_error_response(
            detect_api_format_from_provider("openai_chat"), 400, "test"
        )
        body = json.loads(resp.body)
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["code"] is None

    def test_openai_responses(self):
        resp = format_error_response(
            detect_api_format_from_provider("openai_responses"), 400, "test"
        )
        body = json.loads(resp.body)
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["code"] is None

    def test_anthropic(self):
        resp = format_error_response(
            detect_api_format_from_provider("anthropic"), 400, "test"
        )
        body = json.loads(resp.body)
        assert body["type"] == "error"
        assert body["error"]["type"] == "invalid_request_error"

    def test_google(self):
        resp = format_error_response(
            detect_api_format_from_provider("google"), 502, "upstream error"
        )
        body = json.loads(resp.body)
        assert body["error"]["code"] == 502
        assert body["error"]["status"] == "INVALID_ARGUMENT"

    def test_unknown_provider_uses_openai(self):
        resp = format_error_response(
            detect_api_format_from_provider("something_unknown"), 500, "err"
        )
        body = json.loads(resp.body)
        assert "error" in body
        assert body["error"]["type"] == "invalid_request_error"


# ---------------------------------------------------------------------------
# Behavioral parity — auth.py (_error_for_path)
# ---------------------------------------------------------------------------


class TestAuthParity:
    """Verify the shared module reproduces auth.py's _error_for_path output."""

    def test_openai_auth_error(self):
        fmt = detect_api_format("/v1/chat/completions")
        resp = format_error_response(
            fmt,
            401,
            "Invalid or missing API key",
            error_type="invalid_request_error",
            error_code="invalid_api_key",
            cors=True,
        )
        body = json.loads(resp.body)
        assert body["error"]["type"] == "invalid_request_error"
        assert body["error"]["code"] == "invalid_api_key"
        assert resp.headers["Access-Control-Allow-Origin"] == "*"

    def test_anthropic_auth_error(self):
        fmt = detect_api_format("/v1/messages")
        resp = format_error_response(
            fmt,
            401,
            "Invalid or missing API key",
            error_type="authentication_error",
            cors=True,
        )
        body = json.loads(resp.body)
        assert body["type"] == "error"
        assert body["error"]["type"] == "authentication_error"
        assert resp.headers["Access-Control-Allow-Origin"] == "*"

    def test_google_auth_error(self):
        fmt = detect_api_format("/v1beta/models/gemini:generateContent")
        resp = format_error_response(
            fmt,
            401,
            "Invalid or missing API key",
            google_status="UNAUTHENTICATED",
            cors=True,
        )
        body = json.loads(resp.body)
        assert body["error"]["code"] == 401
        assert body["error"]["status"] == "UNAUTHENTICATED"
        assert resp.headers["Access-Control-Allow-Origin"] == "*"

    def test_admin_path_no_cors(self):
        fmt = detect_api_format("/admin/api/config")
        resp = format_error_response(fmt, 403, "forbidden", cors=False)
        assert "Access-Control-Allow-Origin" not in resp.headers


# ---------------------------------------------------------------------------
# Behavioral parity — ratelimit.py (_rate_limit_response)
# ---------------------------------------------------------------------------


class TestRateLimitParity:
    """Verify the shared module reproduces ratelimit.py's output."""

    def test_openai_rate_limit(self):
        fmt = detect_api_format("/v1/chat/completions")
        resp = format_error_response(
            fmt,
            429,
            "Rate limit exceeded (per_ip). Please retry after 6s.",
            error_type="rate_limit_error",
            error_code="rate_limit_exceeded",
            extra_headers={"Retry-After": "6", "X-RateLimit-Limit": "100"},
            cors=True,
        )
        body = json.loads(resp.body)
        assert body["error"]["type"] == "rate_limit_error"
        assert body["error"]["code"] == "rate_limit_exceeded"
        assert resp.headers["Retry-After"] == "6"
        assert resp.headers["X-RateLimit-Limit"] == "100"
        assert resp.headers["Access-Control-Allow-Origin"] == "*"

    def test_anthropic_rate_limit(self):
        fmt = detect_api_format("/v1/messages")
        resp = format_error_response(
            fmt,
            429,
            "Rate limit exceeded (global). Please retry after 5s.",
            error_type="rate_limit_error",
            google_status="RESOURCE_EXHAUSTED",
            cors=True,
        )
        body = json.loads(resp.body)
        assert body["type"] == "error"
        assert body["error"]["type"] == "rate_limit_error"

    def test_google_rate_limit(self):
        fmt = detect_api_format("/v1beta/models/gemini:gen")
        resp = format_error_response(
            fmt,
            429,
            "Rate limit exceeded (per_model). Please retry after 5s.",
            error_type="rate_limit_error",
            error_code="rate_limit_exceeded",
            google_status="RESOURCE_EXHAUSTED",
            cors=True,
        )
        body = json.loads(resp.body)
        assert body["error"]["code"] == 429
        assert body["error"]["status"] == "RESOURCE_EXHAUSTED"

    def test_admin_rate_limit_no_cors(self):
        fmt = detect_api_format("/admin/api/test")
        resp = format_error_response(
            fmt,
            429,
            "rate limited",
            error_type="rate_limit_error",
            cors=False,
        )
        assert "Access-Control-Allow-Origin" not in resp.headers
