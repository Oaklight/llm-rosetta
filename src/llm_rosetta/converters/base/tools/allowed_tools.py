"""LLM-Rosetta - Tool allow-list helpers

A tool choice may be narrowed from "any tool" to a named subset.  The wire
spellings disagree on both the wrapper and the entry:

- OpenAI Chat and Responses wrap an object in ``tool_choice``
  (``{"type": "allowed_tools", "mode": ..., "tools": [...]}``).  A Chat entry
  nests the name one level deeper (``{"type": "function", "function": {"name"}}``,
  or ``custom``) than a Responses one (``{"type": "function", "name": ...,
  "namespace": ...}``).
- Google Interactions wraps a flat list of names the same way.
- Google generateContent keeps the names alone in
  ``functionCallingConfig.allowedFunctionNames`` and has no wrapper object;
  its converter reads its own field and calls :func:`allowed_tool_refs`.

The IR states the concept once, as ``ToolChoice.allowed_tools``, a list of
identities — see ``types/ir/tools.py``.  These helpers move between the two
shapes; the flattening they do is deliberate, since only the Responses family
carries a group beside a name in this position and the IR keeps it when it is
there.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

from ....types.ir.tools import ToolChoice, ToolRef

__all__ = [
    "allowed_tool_names",
    "allowed_tool_refs",
    "provider_allowed_tools_to_ir",
]

#: Wire ``mode`` spellings seen across the dialects that wrap an allow-list.
#: Chat and Responses say ``required`` where Interactions says ``any``; both
#: mean the model must call one of the listed tools.
_ALLOWED_MODE_TO_IR: dict[str, str] = {
    "auto": "auto",
    "required": "any",
    "any": "any",
    "none": "none",
}


def _entry_to_ref(entry: Any) -> ToolRef | None:
    """Read one allow-list entry as a tool identity.

    Accepts what the dialects send: a bare name string, a flat
    ``{"name": ..., "namespace": ...}`` object, or a Chat-style nested object
    under ``function`` / ``custom``.  Anything else — an entry naming a
    server tool, say — has no identity the IR can hold.

    Args:
        entry: One element of a wire ``tools`` array.

    Returns:
        The identity, or ``None`` when the entry names nothing.
    """
    if isinstance(entry, str):
        return cast(ToolRef, {"name": entry}) if entry else None
    if not isinstance(entry, Mapping):
        return None

    name = entry.get("name")
    if not isinstance(name, str) or not name:
        nested = entry.get("function") or entry.get("custom")
        name = nested.get("name") if isinstance(nested, Mapping) else None
    if not isinstance(name, str) or not name:
        return None

    ref: dict[str, Any] = {"name": name}
    # The Responses family may echo the container beside the name; a flat
    # dialect has no such key and the name is the whole identity.  No upstream
    # we measured accepts such a key in this position, so it is read from a
    # client body rather than from anything the API would have produced.
    group = entry.get("namespace")
    if isinstance(group, str) and group:
        ref["tool_group"] = group
    return cast(ToolRef, ref)


def _describe_entry(entry: Any) -> str:
    """Name an allow-list entry for a warning, whatever shape it arrived in."""
    if isinstance(entry, Mapping):
        for key in ("name", "type"):
            value = entry.get(key)
            if isinstance(value, str) and value:
                return repr(value)
        return "an unnamed entry"
    return repr(entry) if not isinstance(entry, str) or entry else "an empty entry"


def allowed_tool_refs(
    entries: Any,
    *,
    warnings: list[str] | None = None,
    selector: str = "allowed_tools",
) -> list[ToolRef]:
    """Read a list of wire allow-list entries as IR identities.

    An entry that names nothing is dropped — the IR has no identity to hold it
    under — and reported on *warnings* rather than silently: dropping it
    *widens* the restriction, which is the one direction this library reports.

    Args:
        entries: The wire ``tools`` array, or the bare name list Google
            generateContent keeps.
        warnings: Optional warning sink, one entry per dropped entry.
        selector: Name of the selector for the warning text.

    Returns:
        The identities, in wire order, with entries that name nothing
        dropped.
    """
    if not isinstance(entries, list):
        return []
    refs: list[ToolRef] = []
    for entry in entries:
        ref = _entry_to_ref(entry)
        if ref is None:
            if warnings is not None:
                warnings.append(
                    f"{selector} lists {_describe_entry(entry)}, which names no "
                    "tool this library can restrict to, so the entry was "
                    "dropped and the restriction is wider than the source asked "
                    "for"
                )
            continue
        refs.append(ref)
    return refs


def _wire_mode_to_ir(
    wire_mode: str, *, warnings: list[str] | None, selector: str
) -> str:
    """Map a wire mode onto the IR's, reporting one we do not know.

    An unknown mode reads as ``auto`` — the widest of the four — so it is
    reported rather than absorbed: silently widening a restriction is what
    this module exists to stop doing.
    """
    ir_mode = _ALLOWED_MODE_TO_IR.get(wire_mode)
    if ir_mode is None:
        if warnings is not None:
            warnings.append(
                f"{selector} carried an unrecognized mode {wire_mode!r}; it was "
                "read as 'auto', which lets the model call any declared tool"
            )
        return "auto"
    return ir_mode


def provider_allowed_tools_to_ir(
    provider_tool_choice: Mapping[str, Any],
    *,
    warnings: list[str] | None = None,
    selector: str = "allowed_tools",
) -> ToolChoice:
    """Read a wrapped allow-list tool choice as an IR tool choice.

    Args:
        provider_tool_choice: A ``{"type": "allowed_tools", ...}`` object.
        warnings: Optional warning sink.
        selector: Name of the selector for the warning text.

    Returns:
        IR tool choice carrying the same mode and the listed identities.  An
        object listing nothing readable is just its mode, since the IR leaves
        ``allowed_tools`` off rather than restricting to no tools at all.
    """
    wire_mode = str(provider_tool_choice.get("mode") or "auto")
    choice: dict[str, Any] = {
        "mode": _wire_mode_to_ir(wire_mode, warnings=warnings, selector=selector),
        "tool_name": "",
    }
    refs = allowed_tool_refs(
        provider_tool_choice.get("tools"), warnings=warnings, selector=selector
    )
    if refs:
        choice["allowed_tools"] = refs
    return cast(ToolChoice, choice)


def allowed_tool_names(ir_tool_choice: Mapping[str, Any]) -> list[str]:
    """Read the names off an IR tool choice's restriction.

    The names are whatever spelling the IR currently holds — the pipeline
    re-spells them to the target's own before a converter writes them — so a
    writer uses this rather than reaching into the identities itself.

    Args:
        ir_tool_choice: IR tool choice.

    Returns:
        The names, in order.
    """
    refs = ir_tool_choice.get("allowed_tools")
    if not isinstance(refs, list):
        return []
    names: list[str] = []
    for ref in refs:
        if isinstance(ref, Mapping) and isinstance(ref.get("name"), str):
            names.append(ref["name"])
    return names
