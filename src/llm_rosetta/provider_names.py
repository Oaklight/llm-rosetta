"""Canonical provider/format names and legacy alias resolution.

This module is the **single source of truth** for translating legacy
provider or format names into their canonical spelling.  Every boundary
that accepts a provider/format name from the outside world (gateway
JSONC config, request auto-detection, library ``source_provider`` /
``target_provider`` arguments, shim resolution) routes the name through
:func:`normalize_provider_name` before comparing it as a string.

The canonical names are the ones understood internally; legacy names stay
accepted forever for backward compatibility but emit a
:class:`DeprecationWarning` pointing at the replacement.

Adding a new rename is a one-line change to :data:`LEGACY_PROVIDER_ALIASES`
(and, where relevant, the ``ProviderType`` literal in
:mod:`llm_rosetta.auto_detect`).

Example:
    >>> from llm_rosetta import normalize_provider_name
    >>> normalize_provider_name("google-genai")
    'google_generate'
    >>> normalize_provider_name("anthropic")
    'anthropic'
"""

from __future__ import annotations

import warnings

# ---------------------------------------------------------------------------
# Single alias table — legacy name -> canonical name
# ---------------------------------------------------------------------------
#
# Keys are legacy spellings still accepted at the edges; values are the
# canonical names used internally.  A name absent from this table is either
# canonical or unknown, and passes through unchanged.
#
# Do not add alias tables anywhere else — extend this mapping instead.
LEGACY_PROVIDER_ALIASES: dict[str, str] = {
    # Google generateContent: ``google_generate`` is the canonical provider
    # name; ``google`` was the short spelling and ``google-genai`` the
    # SDK-style hyphenated form.  Both legacy spellings migrate to the
    # canonical name; ``google`` is not retained as a standalone name.
    "google": "google_generate",
    "google-genai": "google_generate",
    # OpenAI Chat Completions: ``chat_completions`` is the canonical,
    # API-standard name.  ``openai_chat`` was the old base/converter name and
    # stays accepted (deprecated); the hyphenated tool-ops spelling maps
    # straight to the canonical name.
    "openai_chat": "chat_completions",
    # Hyphenated spellings accepted by the tool-ops convenience API.
    "openai-chat": "chat_completions",
    "openai-responses": "openai_responses",
    "open-responses": "open_responses",
    "google-interactions": "google_interactions",
    # --- Reserved for upcoming renames (kept here so the mechanism is
    #     exercised by a single table).  Add the line when the rename lands:
    #       "openai_responses": "open_responses",   # #909
}


def normalize_provider_name(name: str) -> str:
    """Resolve a legacy provider/format name to its canonical spelling.

    Args:
        name: Provider type, shim name, or format identifier.

    Returns:
        The canonical name if *name* is a known legacy alias, otherwise
        *name* unchanged (canonical or unknown names pass through).

    Warns:
        DeprecationWarning: When *name* is a known legacy alias.  The
            message names the canonical replacement.

    Examples:
        >>> normalize_provider_name("openai-responses")
        'openai_responses'
        >>> normalize_provider_name("openai_chat")
        'chat_completions'
    """
    canonical = LEGACY_PROVIDER_ALIASES.get(name)
    if canonical is None:
        return name
    warnings.warn(
        f"Provider name {name!r} is deprecated; use {canonical!r} instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return canonical


__all__ = ["LEGACY_PROVIDER_ALIASES", "normalize_provider_name"]
