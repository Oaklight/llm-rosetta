"""Canonical provider/format names and legacy alias resolution.

This module is the **single source of truth** for translating legacy
provider or format names into their canonical spelling.  Every boundary
that accepts a provider/format name from the outside world (gateway
JSONC config, request auto-detection, library ``source_provider`` /
``target_provider`` arguments, shim resolution) routes the name through
:func:`canonical_provider_name` before comparing it as a string.  Public
entry points emit the deprecation warning themselves via
:func:`warn_legacy_provider_names` (or :func:`normalize_provider_name` for
direct callers); internal lookups stay silent.

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
    # Responses: ``open_responses`` is the canonical, vendor-neutral base
    # type.  The OpenAI profile is a *shim* over it, named
    # ``openai--open_responses`` (provider--base); the OpenAI-flavoured legacy
    # spellings target that shim so they resolve to the same OpenAI provider
    # rather than silently downgrading to the neutral base.  ``open-responses``
    # (the spec spelling, no ``ai``) is the neutral base.
    "openai_responses": "openai--open_responses",
    # Hyphenated spellings accepted by the tool-ops convenience API.
    "openai-chat": "chat_completions",
    "openai-responses": "openai--open_responses",
    "open-responses": "open_responses",
    "google-interactions": "google_interactions",
}


def canonical_provider_name(name: str) -> str:
    """Resolve a legacy provider/format name to its canonical spelling.

    This is the pure, non-warning lookup.  Use it for internal lookups and
    after a public boundary has already emitted the deprecation warning, so
    a single legacy input does not produce a cascade of warnings through the
    call stack.  Callers that accept a name from the outside world and want
    the deprecation signal should use :func:`normalize_provider_name`.

    Args:
        name: Provider type, shim name, or format identifier.

    Returns:
        The canonical name if *name* is a known legacy alias, otherwise
        *name* unchanged (canonical or unknown names pass through).

    Examples:
        >>> canonical_provider_name("openai-responses")
        'openai--open_responses'
        >>> canonical_provider_name("openai_chat")
        'chat_completions'
    """
    return LEGACY_PROVIDER_ALIASES.get(name, name)


def normalize_provider_name(name: str) -> str:
    """Resolve a legacy provider/format name and warn when it is deprecated.

    Public helper for callers that accept a provider/format name directly
    from the outside world.  Internally, prefer :func:`canonical_provider_name`
    so a boundary that has already warned does not warn again.

    Args:
        name: Provider type, shim name, or format identifier.

    Returns:
        The canonical name if *name* is a known legacy alias, otherwise
        *name* unchanged (canonical or unknown names pass through).

    Warns:
        DeprecationWarning: When *name* is a known legacy alias, attributed
            to the caller's frame.  The message names the canonical
            replacement.

    Examples:
        >>> normalize_provider_name("openai-responses")
        'openai--open_responses'
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


def warn_legacy_provider_names(*names: str | None) -> None:
    """Emit one :class:`DeprecationWarning` per distinct legacy *name*.

    Public boundaries (:func:`llm_rosetta.convert`,
    :func:`llm_rosetta.convert_response`,
    :class:`llm_rosetta.ConversionPipeline`) call this with the names the
    user supplied, then resolve them with :func:`canonical_provider_name`.
    A name supplied in two roles (e.g. the same legacy spelling as both
    ``source_provider`` and ``target_provider``) warns only once.

    ``stacklevel=3`` attributes the warning to the caller of the public
    boundary: :func:`warnings.warn` → this helper → boundary → user frame.

    Args:
        *names: Provider/format names supplied by the user.  ``None``
            entries are ignored.
    """
    seen: set[str] = set()
    for name in names:
        if name is None or name in seen:
            continue
        seen.add(name)
        canonical = LEGACY_PROVIDER_ALIASES.get(name)
        if canonical is not None:
            warnings.warn(
                f"Provider name {name!r} is deprecated; use {canonical!r} instead.",
                DeprecationWarning,
                stacklevel=3,
            )


__all__ = [
    "LEGACY_PROVIDER_ALIASES",
    "canonical_provider_name",
    "normalize_provider_name",
    "warn_legacy_provider_names",
]
