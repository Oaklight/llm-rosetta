"""Tests for credential sanitization: header redaction and pattern scrubbing."""

from __future__ import annotations

from llm_rosetta.gateway.sanitize import (
    sanitize_headers,
    sanitize_upstream_error,
    scrub_credential_patterns,
)


class TestSanitizeHeaders:
    def test_redacts_authorization(self):
        headers = {
            "Authorization": "Bearer sk-abc123",
            "Content-Type": "application/json",
        }
        result = sanitize_headers(headers)
        assert result["Authorization"] == "[REDACTED]"
        assert result["Content-Type"] == "application/json"

    def test_redacts_x_api_key(self):
        result = sanitize_headers({"x-api-key": "sk-ant-secret"})
        assert result["x-api-key"] == "[REDACTED]"

    def test_redacts_x_goog_api_key(self):
        result = sanitize_headers({"x-goog-api-key": "AIzaSyAbcdefghijklmnop"})
        assert result["x-goog-api-key"] == "[REDACTED]"

    def test_redacts_proxy_authorization(self):
        result = sanitize_headers({"Proxy-Authorization": "Basic dXNlcjpwYXNz"})
        assert result["Proxy-Authorization"] == "[REDACTED]"

    def test_case_insensitive(self):
        result = sanitize_headers({"AUTHORIZATION": "Bearer token", "X-API-KEY": "key"})
        assert result["AUTHORIZATION"] == "[REDACTED]"
        assert result["X-API-KEY"] == "[REDACTED]"

    def test_preserves_non_auth_headers(self):
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "test/1.0",
            "x-request-id": "abc-123",
        }
        result = sanitize_headers(headers)
        assert result == headers

    def test_returns_copy(self):
        headers = {"Authorization": "Bearer token"}
        result = sanitize_headers(headers)
        assert result is not headers
        assert headers["Authorization"] == "Bearer token"


class TestScrubCredentialPatterns:
    def test_scrubs_bearer_token(self):
        text = "error: Authorization: Bearer sk-proj-abc123def456"
        result = scrub_credential_patterns(text)
        assert "sk-proj-abc123def456" not in result
        assert "[REDACTED]" in result

    def test_scrubs_openai_key(self):
        text = "Invalid API key: sk-abcdefghijklmnopqrstuvwx"
        result = scrub_credential_patterns(text)
        assert "sk-abcdefghijklmnopqrstuvwx" not in result
        assert "[REDACTED]" in result

    def test_scrubs_anthropic_key(self):
        text = "key=sk-ant-api03-abcdefghijklmnopqrstuvwx"
        result = scrub_credential_patterns(text)
        assert "sk-ant-api03" not in result
        assert "[REDACTED]" in result

    def test_scrubs_google_key(self):
        text = "x-goog-api-key: AIzaSyB1234567890abcdefghijklmnopqrst"
        result = scrub_credential_patterns(text)
        assert "AIzaSyB1234567890" not in result
        assert "[REDACTED]" in result

    def test_scrubs_multiple_patterns(self):
        text = "key1=sk-abc123defghijklmnopqrst key2=Bearer token123456789012"
        result = scrub_credential_patterns(text)
        assert "sk-abc123" not in result
        assert "token123456789012" not in result

    def test_no_false_positives_on_normal_json(self):
        text = '{"error": {"message": "Context length exceeded", "type": "invalid_request_error", "code": "context_length_exceeded"}}'
        result = scrub_credential_patterns(text)
        assert result == text

    def test_no_false_positives_on_short_sk(self):
        text = "sk-short"
        result = scrub_credential_patterns(text)
        assert result == text

    def test_preserves_surrounding_text(self):
        text = "before Bearer sk-proj-abc123def456ghijklmno after"
        result = scrub_credential_patterns(text)
        assert result.startswith("before ")
        assert result.endswith(" after")


class TestSanitizeUpstreamError:
    def test_str_input_returns_str(self):
        text = "error with Bearer sk-abc123def456ghijklmno"
        result = sanitize_upstream_error(text)
        assert isinstance(result, str)
        assert "sk-abc123" not in result

    def test_bytes_input_returns_bytes(self):
        raw = b"error with Bearer sk-abc123def456ghijklmno"
        result = sanitize_upstream_error(raw)
        assert isinstance(result, bytes)
        assert b"sk-abc123" not in result

    def test_non_utf8_bytes_returned_unchanged(self):
        raw = b"\x80\x81\x82\xff"
        result = sanitize_upstream_error(raw)
        assert result == raw

    def test_empty_string(self):
        assert sanitize_upstream_error("") == ""

    def test_empty_bytes(self):
        assert sanitize_upstream_error(b"") == b""

    def test_real_provider_error_body(self):
        body = '{"error":{"message":"Incorrect API key provided: sk-proj-abc123defghijklmnop. You can find your API key at https://platform.openai.com/account/api-keys.","type":"invalid_request_error","param":null,"code":"invalid_api_key"}}'
        result = sanitize_upstream_error(body)
        assert isinstance(result, str)
        assert "sk-proj-abc123" not in result
        assert "invalid_api_key" in result
