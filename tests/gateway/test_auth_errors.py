"""Tests for upstream auth error classification and ALCF session-policy detection."""

from __future__ import annotations

from llm_rosetta.gateway.transport.auth_errors import (
    AuthErrorKind,
    classify_auth_error,
    rewrite_session_policy_error,
)


class TestClassifyAuthError:
    def test_non_401_returns_not_auth_error(self):
        for code in (400, 403, 404, 500, 502):
            assert classify_auth_error(code, "anything") == AuthErrorKind.NOT_AUTH_ERROR

    def test_plain_401_returns_normal(self):
        assert (
            classify_auth_error(401, '{"error": "unauthorized"}')
            == AuthErrorKind.NORMAL_401
        )

    def test_empty_body_returns_normal(self):
        assert classify_auth_error(401, "") == AuthErrorKind.NORMAL_401

    def test_internal_policies_returns_session_policy(self):
        body = "The token was rejected due to internal policies of the resource server"
        assert classify_auth_error(401, body) == AuthErrorKind.SESSION_POLICY_401

    def test_high_assurance_hyphenated_returns_session_policy(self):
        body = "This resource requires high-assurance authentication"
        assert classify_auth_error(401, body) == AuthErrorKind.SESSION_POLICY_401

    def test_high_assurance_no_hyphen_returns_session_policy(self):
        body = "This resource requires high assurance authentication"
        assert classify_auth_error(401, body) == AuthErrorKind.SESSION_POLICY_401

    def test_case_insensitive(self):
        body = "Rejected due to Internal Policies"
        assert classify_auth_error(401, body) == AuthErrorKind.SESSION_POLICY_401

    def test_mixed_case_high_assurance(self):
        body = "HIGH-ASSURANCE policy enforced"
        assert classify_auth_error(401, body) == AuthErrorKind.SESSION_POLICY_401


class TestRewriteSessionPolicyError:
    def test_mentions_alcf_token_login(self):
        msg = rewrite_session_policy_error("some error body")
        assert "alcf-token.py --login" in msg

    def test_mentions_globus_logout(self):
        msg = rewrite_session_policy_error("some error body")
        assert "globus.org" in msg

    def test_mentions_30_day(self):
        msg = rewrite_session_policy_error("some error body")
        assert "30-day" in msg
