"""LLM-Rosetta - Tool strict-mode helpers

The ``strict`` flag asks the provider to validate a tool call's arguments
against the tool's JSON Schema.  Three of the five formats carry it — OpenAI
Chat (``function.strict``), OpenAI Responses (``strict``) and Anthropic
Messages (``strict``) — while the two Google formats have no equivalent.

The IR states it once, as the first-class ``ToolDefinition.strict`` field.
``metadata["strict"]`` was the earlier spelling, written by the Responses
converter alone, so a reader accepts it as a fallback and a writer never
produces it again.

Absent is not ``False``: a source that said nothing leaves the field off, and
each target then emits its own default.  Both helpers here keep that
distinction, and both refuse a value that is not a real boolean, since the IR
field is typed ``bool`` and a stray ``None`` would fail IR validation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = ["ir_strict", "write_ir_strict"]


def ir_strict(ir_tool: Mapping[str, Any]) -> bool | None:
    """Read a tool's strict flag off an IR tool definition.

    Args:
        ir_tool: IR ``ToolDefinition``.

    Returns:
        The flag, or ``None`` when the source never stated one.
    """
    strict = ir_tool.get("strict")
    if strict is None:
        metadata = ir_tool.get("metadata")
        if isinstance(metadata, Mapping):
            strict = metadata.get("strict")
    return strict if isinstance(strict, bool) else None


def write_ir_strict(ir_tool: dict[str, Any], provider_strict: Any) -> None:
    """Copy a provider's strict flag onto an IR tool definition.

    Writes only a real boolean, and only when the provider stated one, so a
    missing field and an explicit ``false`` stay apart on the IR side.

    Args:
        ir_tool: IR ``ToolDefinition`` to write onto, in place.
        provider_strict: The provider's value, of any shape.
    """
    if isinstance(provider_strict, bool):
        ir_tool["strict"] = provider_strict
