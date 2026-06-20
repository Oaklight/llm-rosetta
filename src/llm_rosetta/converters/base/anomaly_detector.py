"""Anomalous response content detection for the converter pipeline.

Detects JavaScript runtime errors, HTML error pages, and raw stack traces
that upstream gateways occasionally leak into response content fields.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detection patterns
# ---------------------------------------------------------------------------

# JavaScript runtime errors (e.g. from Argo middleware)
_JS_ERROR_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"Cannot read propert(?:y|ies) of (?:undefined|null)", re.IGNORECASE),
    re.compile(r"TypeError:", re.IGNORECASE),
    re.compile(r"ReferenceError:", re.IGNORECASE),
    re.compile(r"SyntaxError:", re.IGNORECASE),
    re.compile(r"is not a function", re.IGNORECASE),
    re.compile(r"is not defined", re.IGNORECASE),
    re.compile(r"\bat\s+\w+\s*\(.*:\d+:\d+\)", re.IGNORECASE),  # JS stack frame
]

# HTML error pages
_HTML_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"<!DOCTYPE html", re.IGNORECASE),
    re.compile(r"<html[\s>]", re.IGNORECASE),
    re.compile(r"<body[\s>]", re.IGNORECASE),
    re.compile(r"<title>.*error.*</title>", re.IGNORECASE),
    re.compile(r"nginx/", re.IGNORECASE),
    re.compile(r"apache/", re.IGNORECASE),
]

# Python/generic stack traces
_STACKTRACE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"Traceback \(most recent call last\)", re.IGNORECASE),
    re.compile(r'File ".*", line \d+', re.IGNORECASE),
    re.compile(r"at (?:Object\.|Function\.|async )?[\w.<>]+\s*\(", re.IGNORECASE),
]

_ALL_PATTERN_GROUPS: list[tuple[str, list[re.Pattern[str]]]] = [
    ("javascript_error", _JS_ERROR_PATTERNS),
    ("html_error_page", _HTML_PATTERNS),
    ("stack_trace", _STACKTRACE_PATTERNS),
]

# Minimum text length to bother checking (skip very short strings)
_MIN_CHECK_LENGTH = 20


def _detect_anomaly(text: str) -> str | None:
    """Return anomaly category name if text looks anomalous, else None."""
    if len(text) < _MIN_CHECK_LENGTH:
        return None
    for category, patterns in _ALL_PATTERN_GROUPS:
        for pattern in patterns:
            if pattern.search(text):
                return category
    return None


def check_ir_response_content(
    ir_response: Any,
    *,
    provider: str = "unknown",
    model: str | None = None,
    request_id: str = "-",
) -> None:
    """Scan IR response content parts for anomalous text and log warnings.

    Checks all text content parts in the response. Logs a WARNING with
    the anomaly category, provider, model, and a short excerpt so the
    issue is traceable without logging full response bodies.

    Args:
        ir_response: IR response dict (or any dict with a ``content`` key).
        provider: Upstream provider name for context.
        model: Model name for context.
        request_id: Request ID for log correlation.
    """
    if not isinstance(ir_response, dict):
        return

    # IRResponse has choices[].message.content[]
    choices = ir_response.get("choices", [])
    if isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            message = choice.get("message", {})
            if not isinstance(message, dict):
                continue
            content = message.get("content", [])
            _check_content_parts(
                content, provider=provider, model=model, request_id=request_id
            )

    # Also check a flat ``content`` key (used by some internal helpers / tests)
    flat_content = ir_response.get("content", [])
    if isinstance(flat_content, list):
        _check_content_parts(
            flat_content, provider=provider, model=model, request_id=request_id
        )


def _check_content_parts(
    content: Any,
    *,
    provider: str,
    model: str | None,
    request_id: str,
) -> None:
    """Check a list of content parts for anomalous text."""
    if not isinstance(content, list):
        return
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") != "text":
            continue
        text = part.get("text", "")
        if not isinstance(text, str):
            continue
        category = _detect_anomaly(text)
        if category is not None:
            excerpt = text[:120].replace("\n", " ")
            logger.warning(
                "[%s] anomalous response content detected "
                "(category=%s provider=%s model=%s): %r",
                request_id,
                category,
                provider,
                model or "unknown",
                excerpt,
            )
