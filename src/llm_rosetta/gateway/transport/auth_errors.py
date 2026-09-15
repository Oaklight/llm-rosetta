"""Upstream auth error classification and ALCF session-policy detection.

Classifies 401 responses so the proxy can decide whether to:
- retry after a reactive token refresh (normal token expiry), or
- return an actionable error immediately (ALCF 30-day session policy).
"""

from __future__ import annotations

import enum


class AuthErrorKind(enum.Enum):
    """Classification of an upstream error for auth-retry decisions."""

    NOT_AUTH_ERROR = "not_auth_error"
    NORMAL_401 = "normal_401"
    SESSION_POLICY_401 = "session_policy"


# Substrings in ALCF's Globus 401 body that indicate the 30-day
# forced re-authentication policy (distinct from token expiry).
_SESSION_POLICY_MARKERS = (
    "internal policies",
    "high-assurance",
    "high assurance",
)


def classify_auth_error(status_code: int, error_body: str) -> AuthErrorKind:
    """Classify an upstream error response for auth-retry decisions."""
    if status_code != 401:
        return AuthErrorKind.NOT_AUTH_ERROR
    body_lower = error_body.lower()
    for marker in _SESSION_POLICY_MARKERS:
        if marker in body_lower:
            return AuthErrorKind.SESSION_POLICY_401
    return AuthErrorKind.NORMAL_401


def rewrite_session_policy_error() -> str:
    """Produce a user-friendly error for ALCF session-policy 401s."""
    return (
        "ALCF session expired (30-day Globus re-authentication policy). "
        "Token refresh will not help — you must re-authenticate interactively. "
        "Steps: "
        "1) Visit https://app.globus.org/logout to end your Globus session. "
        "2) Re-run: python3 scripts/alcf-token.py --login"
    )
