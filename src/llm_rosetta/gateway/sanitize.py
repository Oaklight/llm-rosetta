"""Credential sanitization for upstream error bodies and headers."""

from __future__ import annotations

import re

_AUTH_HEADER_NAMES = frozenset(
    {
        "authorization",
        "x-api-key",
        "x-goog-api-key",
        "proxy-authorization",
    }
)

_CREDENTIAL_PATTERNS = re.compile(
    r"|".join(
        [
            r"Bearer\s+\S+",
            r"sk-ant-[A-Za-z0-9_-]{20,}",
            r"sk-[A-Za-z0-9_-]{20,}",
            r"AIza[A-Za-z0-9_-]{30,}",
        ]
    )
)


def sanitize_headers(headers: dict[str, str]) -> dict[str, str]:
    """Return a copy of *headers* with auth values redacted."""
    result = {}
    for k, v in headers.items():
        if k.lower() in _AUTH_HEADER_NAMES:
            result[k] = "[REDACTED]"
        else:
            result[k] = v
    return result


def scrub_credential_patterns(text: str) -> str:
    """Replace credential-shaped strings in *text* with ``[REDACTED]``."""
    return _CREDENTIAL_PATTERNS.sub("[REDACTED]", text)


def sanitize_upstream_error(raw: str | bytes) -> str | bytes:
    """Scrub credentials from an upstream error body, preserving type."""
    if isinstance(raw, bytes):
        try:
            decoded = raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw
        return scrub_credential_patterns(decoded).encode("utf-8")
    return scrub_credential_patterns(raw)
