"""One-time correction of legacy provider/shim names in the gateway config.

This is deliberately **not** part of the versioned ``config_version`` schema
migrations (:mod:`llm_rosetta_gateway.maintenance.migrations`): it is a scoped, idempotent
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
    from ..providers import BASE_FORMATS

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


def _retired_error(name: str, value: str, kinds: tuple[str, ...]) -> str | None:
    """The message for *value* when it is retired under one of *kinds*.

    With the aliases on, the message is the historical "use X instead" form;
    with them off it is the retired-names diagnostic, which also names the
    standalone package's preset.
    """
    from llm_rosetta.api import retired_names

    for kind in kinds:
        row = retired_names.lookup(value, kind)
        if row is None:
            continue
        noun = "name" if kind == retired_names.PROVIDER_TYPE else "shim name"
        if retired_names.ALIASES_ENABLED:
            return (
                f"provider {name!r} uses legacy {noun} {value!r}; "
                f"use {row.replacement!r} instead — see {DOCS_URL}"
            )
        return (
            f"provider {name!r}: "
            f"{retired_names.retired_name_message(value, kind)} — see {DOCS_URL}"
        )
    return None


def legacy_name_error(providers: dict[str, Any]) -> str | None:
    """Return an error message if *providers* still holds a retired name.

    Guards load paths that bypass :func:`correct_legacy_names`: the gateway
    rejects the config rather than silently resolving a legacy name at runtime.
    The names come from the retired-names table
    (:mod:`llm_rosetta.retired_names`), so the guard keeps working once the
    alias tables are gone, and names the replacement (and the preset of a
    standalone package) when it fires.
    """
    from llm_rosetta.api import retired_names

    from ..providers import BASE_FORMATS

    for name, cfg in providers.items():
        if not isinstance(cfg, dict):
            continue
        shim = cfg.get("shim")
        if isinstance(shim, str):
            error = _retired_error(name, shim, (retired_names.SHIM,))
            if error is not None:
                return error
        typ = cfg.get("type")
        if isinstance(typ, str):
            error = _retired_error(name, typ, (retired_names.PROVIDER_TYPE,))
            if error is not None:
                return error
            if typ not in BASE_FORMATS:
                error = _retired_error(name, typ, (retired_names.SHIM,))
                if error is not None:
                    return error
    return None
