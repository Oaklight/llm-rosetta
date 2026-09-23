"""Fit an identifier into a length budget without losing uniqueness.

Providers cap several distinct identifiers at the same 64 characters —
tool call IDs and tool names among them — and a plain truncation makes
two different values collide.  Appending a digest of the full value
keeps them apart, and keeping the digest a pure function of the input
means the result is stable across requests: callers that correlate two
identifiers by equality, or that rely on prompt caching, still work.
"""

from __future__ import annotations

import hashlib

HASH_SUFFIX_LEN = 8


def truncate_with_digest(text: str, max_length: int) -> str:
    """Shorten *text* to *max_length* with a deterministic digest suffix.

    Args:
        text: The value to shorten.  Returned unchanged if it already fits.
        max_length: The budget the result must fit in.  Must leave room for
            the digest and its separator.

    Returns:
        *text* itself when it fits, otherwise ``{prefix}_{digest}``.
    """
    if len(text) <= max_length:
        return text

    digest = hashlib.sha256(text.encode()).hexdigest()[:HASH_SUFFIX_LEN]
    truncated_len = max_length - HASH_SUFFIX_LEN - 1  # 1 for separator
    return f"{text[:truncated_len]}_{digest}"
