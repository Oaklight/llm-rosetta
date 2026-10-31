"""Shim-declared request defaults, applied at the IR boundary.

A provider shim may declare parameter defaults for its own model type (an
``embedding:``, ``rerank:`` or ``decision:`` section in ``provider.yaml``).
They are IR field names, so they are applied to the IR request after the
source converter has read the client's body and before the target converter
writes the upstream one.  A value the client supplied always wins.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypeVar, cast

#: The IR request a converter hands over; each family has its own TypedDict.
_IR = TypeVar("_IR", bound=Mapping[str, Any])


def apply_ir_defaults(ir: _IR, defaults: Mapping[str, Any] | None) -> _IR:
    """Fill absent or ``None`` IR fields from *defaults*.

    Args:
        ir: The IR request the source converter produced.
        defaults: A shim's declared defaults, keyed by IR field name.

    Returns:
        The IR request to hand to the target converter.  The input mapping is
        not mutated, because the converter owns it.
    """
    if not defaults:
        return ir
    merged = dict(ir)
    for key, value in defaults.items():
        if merged.get(key) is None:
            merged[key] = value
    return cast(_IR, merged)


__all__ = ["apply_ir_defaults"]
