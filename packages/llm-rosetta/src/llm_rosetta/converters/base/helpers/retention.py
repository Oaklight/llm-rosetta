"""The prompt-cache retention value space, spelled each way.

The OpenAI wire spells the value ``in_memory``; the IR spells the same value
``in-memory``.  Copying the raw value across the boundary therefore either
fails IR validation (a client's ``in_memory``) or sends a value the API does
not define (the IR's ``in-memory``), so each direction translates.

A value outside the table passes through unchanged: a vendor extension is the
shim layer's business, and dropping it here would hide it.
"""

from __future__ import annotations

from typing import Any

#: IR spelling → the OpenAI wire spelling.
_IR_TO_WIRE = {"in-memory": "in_memory"}
_WIRE_TO_IR = {wire: ir for ir, wire in _IR_TO_WIRE.items()}


def retention_to_ir(value: Any) -> Any:
    """Translate a wire retention value into the IR's spelling.

    Args:
        value: The value as the provider request carries it.

    Returns:
        The IR spelling, or *value* unchanged when it is not a known one.
    """
    return _WIRE_TO_IR.get(value, value)


def retention_to_wire(value: Any) -> Any:
    """Translate an IR retention value into the OpenAI wire spelling.

    Args:
        value: The value as the IR carries it.

    Returns:
        The wire spelling, or *value* unchanged when it is not a known one.
    """
    return _IR_TO_WIRE.get(value, value)
