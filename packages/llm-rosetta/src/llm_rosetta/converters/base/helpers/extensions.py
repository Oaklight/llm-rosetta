"""Shared handling for ``IRRequest.provider_extensions``.

Extension keys beginning with ``_`` are converter-internal markers: they carry
state between a converter's provider→IR and IR→provider halves (e.g.
``_text_verbosity``) and must never be merged into a provider request body.
They would otherwise escape onto the wire when a *different* converter performs
the IR→provider half, since provider extensions are passed through verbatim.
"""

from __future__ import annotations

from typing import Any


def wire_extensions(ir_request: Any) -> dict[str, Any]:
    """Return the provider extensions that may be merged into a request body.

    Drops ``_``-prefixed keys, which are converter-internal and not wire fields.
    """
    extensions = ir_request.get("provider_extensions")
    if not extensions:
        return {}
    return {k: v for k, v in extensions.items() if not k.startswith("_")}
