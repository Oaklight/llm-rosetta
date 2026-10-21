"""Credential redaction for reports.

Three mechanisms, strongest first:

1. **Exact values.**  The harness knows which credentials a suite resolved, so
   those strings are replaced wherever they appear — in a header value, a query
   string, an upstream error or a bare message.  Pattern matching alone can
   only catch shapes it recognises.
2. **Shape-aware mapping.**  A value under a sensitive key (``x-api-key``,
   ``authorization``, ``token`` …) is redacted regardless of its value, which
   covers headers echoed back as a dict.
3. **Patterns.**  The credential patterns shared with
   :func:`llm_rosetta.observability.sanitize.scrub_credential_patterns` plus
   URL userinfo and ``key=value`` parameters.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

from llm_rosetta.observability.sanitize import scrub_credential_patterns

#: Mapping keys whose value is a credential, matched case-insensitively.
_SENSITIVE_KEYS = frozenset(
    {
        "authorization",
        "proxy-authorization",
        "x-api-key",
        "x-goog-api-key",
        "api-key",
        "api_key",
        "apikey",
        "token",
        "access_token",
        "refresh_token",
        "secret",
        "client_secret",
        "password",
    }
)

#: Shorter values are too likely to be ordinary words to replace everywhere.
_MIN_SECRET_LENGTH = 4

_USERINFO = re.compile(r"(?i)\b(https?://)[^/@\s]+:[^/@\s]+@")

_ASSIGNMENT = re.compile(
    r"(?i)\b("
    r"(?:api[_-]?key|x-api-key|x-goog-api-key|authorization|"
    r"access[_-]?token|refresh[_-]?token|token|secret|password|key)"
    r"\s*[:=]\s*)"
    r"[^\s,;\"'&]+"
)


class Redactor:
    """Redact known secret values, sensitive mapping keys and patterns."""

    def __init__(self, secrets: Iterable[str] = ()) -> None:
        self._secrets = tuple(
            sorted(
                {
                    secret
                    for secret in secrets
                    if secret and len(secret) >= _MIN_SECRET_LENGTH
                },
                key=len,
                reverse=True,
            )
        )

    def text(self, text: str) -> str:
        for secret in self._secrets:
            text = text.replace(secret, "[REDACTED]")
        # The shared patterns run before the assignment rule: ``Authorization:
        # Bearer <token>`` must lose ``Bearer <token>`` as a whole, not just
        # the word ``Bearer``.
        text = scrub_credential_patterns(text)
        text = _USERINFO.sub(r"\1[REDACTED]@", text)
        return _ASSIGNMENT.sub(r"\1[REDACTED]", text)

    def value(self, value: Any) -> Any:
        """Return *value* with credentials removed, preserving its shape."""
        if isinstance(value, str):
            return self.text(value)
        if isinstance(value, dict):
            return {
                key: (
                    "[REDACTED]"
                    if isinstance(key, str) and key.strip().lower() in _SENSITIVE_KEYS
                    else self.value(item)
                )
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if isinstance(value, tuple):
            return tuple(self.value(item) for item in value)
        return value


def redact_text(text: str, *, secrets: Iterable[str] = ()) -> str:
    return Redactor(secrets).text(text)


def redact(value: Any, *, secrets: Iterable[str] = ()) -> Any:
    return Redactor(secrets).value(value)
