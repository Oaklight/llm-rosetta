"""Config discovery and the load/write facade.

A thin layer over :mod:`config.io`: find the file, read it through the default
reader, write it back atomically, and build the runtime config object.
"""

from __future__ import annotations

import os
from typing import Any

from .core import GatewayConfig
from .io import JsoncConfigIO


PATHS_TO_TRY = [
    "./config.jsonc",
    os.path.expanduser("~/.config/llm-rosetta-gateway/config.jsonc"),
    os.path.expanduser("~/.llm-rosetta-gateway/config.jsonc"),
]


def load_config(path: str) -> dict[str, Any]:
    """Load a JSONC config file for use, with env-var substitution.

    The document comes back through the load-time maintenance steps
    (:class:`JsoncConfigIO`), so a caller gets the shape the current code
    expects rather than whatever was written to disk, however long ago.  A step
    that changed something writes the migrated document back and leaves a
    ``.pre-migration.bak`` beside it; a step that cannot run raises, and there
    is no config to serve.  Env-var placeholders are substituted *and* kept in
    the written-back copy, so a round-trip does not bake a secret into the file.

    The registry the steps ran belongs to the reader that ran them, which this
    one-shot call drops.  A caller that serves an admin surface wants it —
    :func:`load_gateway_config` is that caller's entry point.
    """
    return JsoncConfigIO().load(path)


def load_config_raw(path: str) -> dict[str, Any]:
    """Load a JSONC config file for editing, without env-var substitution.

    The same steps as :func:`load_config` run, for the same reason: a document
    that is about to be written back should be brought forward first.  Env-var
    placeholders are left as they are, so editing a config does not replace them
    with their values.
    """
    return JsoncConfigIO().load_raw(path)


def discover_config(explicit_path: str | None = None) -> str | None:
    """Find the first existing config file.

    If *explicit_path* is given, return it unconditionally (caller is
    responsible for handling missing files).  Otherwise search
    ``PATHS_TO_TRY`` in order and return the first hit, or ``None``.
    """
    if explicit_path is not None:
        return explicit_path
    for path in PATHS_TO_TRY:
        if os.path.isfile(path):
            return path
    return None


def load_gateway_config(path: str) -> GatewayConfig:
    """Read a config document and build the config the gateway serves.

    The load-time steps run as part of the read, and the config carries the
    registry they ran — which is what the admin surface reports the config's
    maintenance from.  That is the whole difference between this and
    ``GatewayConfig(load_config(path))``, which has run the steps without
    keeping any trace of them.
    """
    reader = JsoncConfigIO()
    raw = reader.load(path)
    return GatewayConfig(raw, maintenance=reader.last_load_registry)
