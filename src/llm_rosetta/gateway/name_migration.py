"""One-time correction of legacy provider/shim names in the gateway config.

This is deliberately **not** part of the versioned ``config_version`` schema
migrations (:mod:`llm_rosetta.gateway.migrations`): it is a scoped, idempotent
rewrite of *names* only.  It runs at load time and the config is written back
to disk the first time it changes anything, so a config only carries canonical
names afterwards.

After correction the gateway resolves shim names canonically and **does not**
fall back to legacy shim aliases at runtime.  A ``shim`` / ``type`` value that
is still a known legacy spelling therefore means the config could not be
corrected, and is rejected with a pointer to the docs.
"""

from __future__ import annotations

import logging
from typing import Any

from llm_rosetta.provider_names import (
    LEGACY_PROVIDER_ALIASES,
    LEGACY_SHIM_ALIASES,
    canonical_provider_name,
    canonical_shim_name,
)

logger = logging.getLogger("llm-rosetta-gateway")

#: Where to send a user whose config still holds a name we cannot correct.
DOCS_URL = "https://llm-rosetta.readthedocs.io/en/latest/guide/providers/"


def _corrected_value(value: str, *, shim_slot: bool) -> str:
    """Canonicalise one ``shim`` / ``type`` value.

    A ``shim:`` value is unambiguously a shim identity.  A ``type:`` value names
    a *format*, so it is first run through the provider/format aliases and, only
    if that leaves it unchanged and it is not a base format, through the shim
    aliases (``type: openai`` → ``openai--chat_completions``; ``type:
    google_generate`` stays a format).
    """
    if shim_slot:
        return canonical_shim_name(value)
    canonical = canonical_provider_name(value)
    if canonical != value:
        return canonical
    from .providers import BASE_FORMATS

    if value in BASE_FORMATS:
        return value
    return canonical_shim_name(value)


def correct_legacy_names(raw: dict[str, Any]) -> bool:
    """Rewrite legacy ``providers.*.{shim,type}`` names to canonical in place.

    Args:
        raw: The raw config mapping (mutated in place).

    Returns:
        ``True`` if any name was rewritten.
    """
    providers = raw.get("providers")
    if not isinstance(providers, dict):
        return False
    changed = False
    for cfg in providers.values():
        if not isinstance(cfg, dict):
            continue
        for key, shim_slot in (("shim", True), ("type", False)):
            value = cfg.get(key)
            if not isinstance(value, str):
                continue
            corrected = _corrected_value(value, shim_slot=shim_slot)
            if corrected != value:
                cfg[key] = corrected
                changed = True
                logger.info(
                    "config: corrected provider %s %r → %r", key, value, corrected
                )
    return changed


def legacy_name_error(providers: dict[str, Any]) -> str | None:
    """Return an error message if *providers* still holds a legacy name.

    Guards load paths that bypass :func:`correct_legacy_names`: the gateway
    rejects the config rather than silently resolving a legacy name at runtime.
    """
    from .providers import BASE_FORMATS

    for name, cfg in providers.items():
        if not isinstance(cfg, dict):
            continue
        shim = cfg.get("shim")
        if isinstance(shim, str) and shim in LEGACY_SHIM_ALIASES:
            return (
                f"provider {name!r} uses legacy shim name {shim!r}; "
                f"use {canonical_shim_name(shim)!r} instead — see {DOCS_URL}"
            )
        typ = cfg.get("type")
        if isinstance(typ, str):
            if typ in LEGACY_PROVIDER_ALIASES:
                return (
                    f"provider {name!r} uses legacy name {typ!r}; "
                    f"use {canonical_provider_name(typ)!r} instead — see {DOCS_URL}"
                )
            if typ in LEGACY_SHIM_ALIASES and typ not in BASE_FORMATS:
                return (
                    f"provider {name!r} uses legacy shim name {typ!r}; "
                    f"use {canonical_shim_name(typ)!r} instead — see {DOCS_URL}"
                )
    return None
