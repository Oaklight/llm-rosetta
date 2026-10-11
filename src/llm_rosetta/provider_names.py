"""Canonical provider/format names and legacy alias resolution.

This module is the **single source of truth** for translating legacy
provider or format names into their canonical spelling.  Every boundary
that accepts a provider/format name from the outside world (gateway
JSONC config, request auto-detection, library ``source_provider`` /
``target_provider`` arguments, shim resolution) routes the name through
:func:`canonical_provider_name` before comparing it as a string.  Public
entry points emit the deprecation warning themselves via
:func:`warn_legacy_provider_names` (or :func:`normalize_provider_name` for
direct callers); internal lookups stay silent.  Legacy *shim* names live in a
separate table (:data:`LEGACY_SHIM_ALIASES`) and warn through
:func:`warn_legacy_shim_names`; shim-registry lookups stay silent too.

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

# ---------------------------------------------------------------------------
# Shim-name aliases — legacy shim name -> canonical shim name
# ---------------------------------------------------------------------------
#
# Kept separate from :data:`LEGACY_PROVIDER_ALIASES` on purpose.  Some legacy
# shim names (``google_generate``, ``google_interactions``, ``openai``) are also
# *base* type names or vendor spellings that flow through
# :func:`canonical_provider_name` at provider/format boundaries; rewriting them
# there would misroute converters and gateway formats.  This table is consulted
# only by the shim registry, so a shim registered under its canonical
# ``<provider>--<base>`` name stays reachable under the old spelling.
#
# The ``<provider>--<old base>`` entries are removed in 0.16.0 (#919); the
# bare-name entries are the ``openai`` / ``google_*`` shims folded into the
# grouped layout in #917.
LEGACY_SHIM_ALIASES: dict[str, str] = {
    # Bare shim names → grouped ``<provider>--<base>`` names.
    "openai": "openai--chat_completions",
    "google_generate": "google--google_generate",
    "google_interactions": "google--google_interactions",
    # ``<provider>--<old base>`` → ``<provider>--<canonical base>``.
    "argo--openai_chat": "argo--chat_completions",
    "asksage--openai_chat": "asksage--chat_completions",
    "deepseek--openai_chat": "deepseek--chat_completions",
    "kilo--openai_chat": "kilo--chat_completions",
    "minimax--openai_chat": "minimax--chat_completions",
    "openrouter--openai_chat": "openrouter--chat_completions",
    "volcengine--openai_chat": "volcengine--chat_completions",
    "argo--openai_responses": "argo--open_responses",
    "asksage--openai_responses": "asksage--open_responses",
    "deepseek--openai_responses": "deepseek--open_responses",
    "volcengine--openai_responses": "volcengine--open_responses",
}


def canonical_shim_name(name: str) -> str:
    """Resolve a legacy *shim* name to its canonical ``<provider>--<base>`` form.

    Unlike :func:`canonical_provider_name`, this only rewrites shim identities,
    never base converter types: ``google_generate`` here means the Google
    generate shim (``google--google_generate``), whereas the provider-name
    lookup leaves the base type ``google_generate`` untouched.

    Args:
        name: A shim name, possibly a legacy spelling.

    Returns:
        The canonical shim name if *name* is a known legacy shim alias,
        otherwise *name* unchanged.
    """
    return LEGACY_SHIM_ALIASES.get(name, name)


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


def warn_legacy_shim_names(*names: str | None) -> None:
    """Emit one :class:`DeprecationWarning` per distinct legacy *shim* name.

    Shim-name analogue of :func:`warn_legacy_provider_names`.  Public
    boundaries that accept a shim identity from the outside world
    (``gateway`` JSONC config, :class:`llm_rosetta.ConversionPipeline`'s
    ``source_shim`` / ``target_shim``) call this with the raw shim strings the
    user supplied, then resolve them with :func:`canonical_shim_name`.  A name
    supplied in two roles (e.g. the same legacy shim for both shims) warns only
    once.

    Only names present in :data:`LEGACY_SHIM_ALIASES` warn; a canonical shim
    name (``deepseek--chat_completions``), a base format name
    (``chat_completions``), or an unknown name passes through silently.  This
    is deliberately separate from :func:`warn_legacy_provider_names` so a
    legacy shim name that is also a base type (``google_generate``) is only
    warned where the context is unambiguously a shim.

    ``stacklevel=3`` attributes the warning to the caller of the public
    boundary: :func:`warnings.warn` → this helper → boundary → user frame.

    Args:
        *names: Shim names supplied by the user.  ``None`` entries are ignored.
    """
    seen: set[str] = set()
    for name in names:
        if name is None or name in seen:
            continue
        seen.add(name)
        canonical = LEGACY_SHIM_ALIASES.get(name)
        if canonical is not None:
            warnings.warn(
                f"Shim name {name!r} is deprecated; use {canonical!r} instead.",
                DeprecationWarning,
                stacklevel=3,
            )


__all__ = [
    "LEGACY_PROVIDER_ALIASES",
    "LEGACY_SHIM_ALIASES",
    "canonical_provider_name",
    "canonical_shim_name",
    "normalize_provider_name",
    "warn_legacy_provider_names",
    "warn_legacy_shim_names",
]
