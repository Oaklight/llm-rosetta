"""Tool call ID sanitization for provider compatibility.

Ensures ``tool_call_id`` values conform to provider-specific constraints
before they are written into outgoing requests or streaming events.

Provider constraints:
    - OpenAI Chat / Responses: max 64 characters
    - Anthropic Messages: must match ``^[a-zA-Z0-9_-]+$``
    - Google GenAI: generates its own IDs (not affected)

The sanitization is applied at the IR → provider boundary so that the IR
itself preserves the original ID for internal correlation.
"""

from __future__ import annotations

import hashlib
import re

_INVALID_CHARS = re.compile(r"[^a-zA-Z0-9_-]")

MAX_TOOL_CALL_ID_LENGTH = 64
_HASH_SUFFIX_LEN = 8


def sanitize_tool_call_id(
    raw_id: str, max_length: int = MAX_TOOL_CALL_ID_LENGTH
) -> str:
    """Sanitize a tool call ID for provider compatibility.

    1. Replace characters outside ``[a-zA-Z0-9_-]`` with ``_``.
    2. If the result exceeds *max_length*, truncate and append a
       deterministic hash suffix to preserve uniqueness.

    The function is pure and deterministic — the same input always
    produces the same output, so tool_call and tool_result pairs
    remain correlated without a lookup table.

    Args:
        raw_id: The original tool call ID (may contain invalid chars
            or exceed length limits).
        max_length: Maximum allowed length (default 64, matching
            OpenAI's constraint).

    Returns:
        A sanitized ID that satisfies both character-set and length
        constraints.  Returns the input unchanged if it already
        conforms.
    """
    if not raw_id:
        return raw_id or ""

    sanitized = _INVALID_CHARS.sub("_", raw_id)

    if len(sanitized) <= max_length:
        return sanitized

    digest = hashlib.sha256(sanitized.encode()).hexdigest()[:_HASH_SUFFIX_LEN]
    truncated_len = max_length - _HASH_SUFFIX_LEN - 1  # 1 for separator
    return f"{sanitized[:truncated_len]}_{digest}"
