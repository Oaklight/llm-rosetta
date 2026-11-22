"""The guard that rejects a retired provider or shim name in the gateway config.

A retired name is an *error*, not an alias.  The gateway used to rewrite such a
name to its canonical spelling at load time and write the config back; that
correction is gone.  A ``shim`` / ``type`` value that is a retired
spelling is rejected with the retired-names diagnostic
(:mod:`llm_rosetta.retired_names`): it names the replacement, and for a shim
that moved to a standalone package it names the ``shims.install`` entry too.
"""

from __future__ import annotations

from typing import Any


def _retired_error(
    name: str, slot: str, value: str, kinds: tuple[str, ...]
) -> str | None:
    """The message for *value* when it is retired under one of *kinds*.

    The message is the retired-names diagnostic: it names the replacement, the
    standalone package's ``shims.install`` entry when there is one, and ends with the
    ``llm-rosetta-migrate`` command.
    """
    from llm_rosetta.api import retired_names

    for kind in kinds:
        row = retired_names.lookup(value, kind)
        if row is None:
            continue
        message = retired_names.retired_name_message(value, kind)
        return f"providers.{name}.{slot}: provider {name!r}: {message}"
    return None


def legacy_name_error(providers: dict[str, Any]) -> str | None:
    """Return an error message if *providers* still holds a retired name.

    The gateway rejects the config rather than resolving a retired name at
    runtime.  The names come from the retired-names table
    (:mod:`llm_rosetta.retired_names`); the message names the replacement (and
    the ``shims.install`` entry of a standalone package).
    """
    from llm_rosetta.api import retired_names

    from ..config.providers import BASE_FORMATS

    for name, cfg in providers.items():
        if not isinstance(cfg, dict):
            continue
        shim = cfg.get("shim")
        if isinstance(shim, str):
            error = _retired_error(name, "shim", shim, (retired_names.SHIM,))
            if error is not None:
                return error
        typ = cfg.get("type")
        if isinstance(typ, str):
            error = _retired_error(name, "type", typ, (retired_names.PROVIDER_TYPE,))
            if error is not None:
                return error
            if typ not in BASE_FORMATS:
                error = _retired_error(name, "type", typ, (retired_names.SHIM,))
                if error is not None:
                    return error
    return None
