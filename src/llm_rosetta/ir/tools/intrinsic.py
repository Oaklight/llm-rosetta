"""Intrinsic tool helpers — shared utilities for provider-hosted tool handling.

Intrinsic tools are provider-native server-side capabilities (web search,
code execution, file search, etc.) represented in the IR with
``tool_type="intrinsic"`` (parts/events) or ``type="intrinsic"``
(definitions) and a first-class ``intrinsic_kind`` field identifying the
specific capability.

``intrinsic_kind`` follows the same promotion as ``tool_group`` (#999): the
kind used to live under ``provider_metadata["intrinsic_kind"]`` on a part or
``metadata["intrinsic_kind"]`` on a definition, and now has a top-level field
on every carrier.  :func:`intrinsic_kind_of` reads the first-class field
first and still accepts the legacy metadata spelling, so IR serialized before
the field existed keeps working.
"""

from collections.abc import Mapping
from typing import Any


def is_intrinsic_part(part: Any) -> bool:
    """Check whether an IR content part is an intrinsic tool call or result."""
    return (
        isinstance(part, dict)
        and part.get("type") in ("tool_call", "tool_result")
        and part.get("tool_type") == "intrinsic"
    )


def is_intrinsic(x: Any) -> bool:
    """Check whether an IR object is an intrinsic tool call, result or definition.

    Covers the part/event carriers (``tool_type == "intrinsic"``) as well as
    the definition carrier (``type == "intrinsic"``), so callers that handle
    either side of the tool identity can ask once.
    """
    if not isinstance(x, dict):
        return False
    if x.get("type") == "intrinsic":
        return True
    return (
        x.get("type")
        in (
            "tool_call",
            "tool_result",
            "tool_call_start",
        )
        and x.get("tool_type") == "intrinsic"
    )


def intrinsic_kind_of(x: Any) -> str:
    """Read the canonical intrinsic kind from an IR part, event or definition.

    The kind lives in the first-class ``intrinsic_kind`` field.  IR built
    before that field existed stored it under ``provider_metadata`` (a part or
    event) or ``metadata`` (a definition); those legacy spellings are still
    read as a fallback so an old serialized payload keeps working.  A missing,
    non-string, or empty value reads as ``""``.

    The tool name is deliberately never consulted: an intrinsic kind comes
    from the provider's own wire type or from this field, not by inferring one
    from a client tool name.

    Works on anything carrying these fields: an IR ``ToolCallPart`` /
    ``ToolResultPart``, a ``tool_call_start`` / ``tool_result`` stream event,
    a ``ToolDefinition``, or a dict about to become one.

    Args:
        x: The IR object to read.

    Returns:
        The canonical kind, or ``""`` when the object carries none.
    """
    if not isinstance(x, Mapping):
        return ""
    kind = x.get("intrinsic_kind")
    if isinstance(kind, str) and kind:
        return kind
    for key in ("provider_metadata", "metadata"):
        holder = x.get(key)
        if isinstance(holder, Mapping):
            legacy = holder.get("intrinsic_kind")
            if isinstance(legacy, str) and legacy:
                return legacy
    return ""


def get_intrinsic_kind(part: Any) -> str:
    """Deprecated alias of :func:`intrinsic_kind_of`.

    Kept for one release so existing importers keep working.  New code should
    call :func:`intrinsic_kind_of`.
    """
    return intrinsic_kind_of(part)


def make_intrinsic_tool_call(
    *,
    tool_call_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    intrinsic_kind: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an IR ToolCallPart with intrinsic tool_type.

    The kind is written to the first-class ``intrinsic_kind`` field; the
    legacy ``provider_metadata["intrinsic_kind"]`` spelling is not written.
    """
    part: dict[str, Any] = {
        "type": "tool_call",
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_type": "intrinsic",
        "intrinsic_kind": intrinsic_kind,
    }
    if extra_metadata:
        part["provider_metadata"] = extra_metadata
    return part


def make_intrinsic_tool_result(
    *,
    tool_call_id: str,
    result: Any,
    intrinsic_kind: str,
    is_error: bool = False,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an IR ToolResultPart with intrinsic tool_type.

    The kind is written to the first-class ``intrinsic_kind`` field; the
    legacy ``provider_metadata["intrinsic_kind"]`` spelling is not written.
    """
    part: dict[str, Any] = {
        "type": "tool_result",
        "tool_call_id": tool_call_id,
        "result": result,
        "tool_type": "intrinsic",
        "intrinsic_kind": intrinsic_kind,
    }
    if is_error:
        part["is_error"] = True
    if extra_metadata:
        part["provider_metadata"] = extra_metadata
    return part


# ---------------------------------------------------------------------------
# Intrinsic tool definitions
# ---------------------------------------------------------------------------
#
# An IR ToolDefinition may be ``type="intrinsic"`` for a provider-hosted
# server tool (web search, code execution, …).  The kind lives in the
# first-class ``intrinsic_kind`` field.  Whether a provider enables it is
# declared by that provider's shim (``ToolsConfig.intrinsic_tools``).


def make_intrinsic_tool_definition(
    kind: str,
    *,
    description: str = "",
    parameters: dict[str, Any] | None = None,
    native: dict[str, Any] | None = None,
    native_base: str | None = None,
) -> dict[str, Any]:
    """Build an IR ToolDefinition of ``type="intrinsic"`` from a kind.

    The kind is written to the first-class ``intrinsic_kind`` field; the
    legacy ``metadata["intrinsic_kind"]`` spelling is not written.

    ``native`` (with ``native_base``) stores the provider's original tool
    payload so a same-provider round-trip can restore it verbatim (extra
    config such as search filters is otherwise lost when the def is
    reduced to a canonical kind).
    """
    definition: dict[str, Any] = {
        "type": "intrinsic",
        "name": kind,
        "description": description,
        "parameters": parameters or {},
        "intrinsic_kind": kind,
    }
    if native is not None:
        definition["metadata"] = {"_native": {"base": native_base, "tool": native}}
    return definition


def get_definition_kind(ir_tool: Any) -> str:
    """Deprecated alias of :func:`intrinsic_kind_of`.

    Kept for one release so existing importers keep working.  New code should
    call :func:`intrinsic_kind_of`.
    """
    return intrinsic_kind_of(ir_tool)


def get_native_definition(ir_tool: Any, base: str) -> dict[str, Any] | None:
    """Return the stored native payload if it came from ``base``, else None."""
    meta = ir_tool.get("metadata") or {}
    native = meta.get("_native")
    if isinstance(native, dict) and native.get("base") == base:
        tool = native.get("tool")
        return tool if isinstance(tool, dict) else None
    return None


def history_native_calls(
    messages: Any,
    tool_ops: Any,
    *,
    id_key: Any = None,
) -> dict[str, bool]:
    """Map each intrinsic call id to whether the target writes it natively.

    A call and its result live in different messages, so the decision is made
    once over the whole history and every result writer consults the same map:
    a result pairs with the call that is actually written.  The judge is the
    target's ``intrinsic_renderable`` on the history leg, the same predicate
    the pipeline's history pass uses.

    Args:
        messages: IR message list.
        tool_ops: Target ``BaseToolOps`` subclass.
        id_key: Optional transform applied to ids (e.g. a dialect's id
            sanitizer), so the map is keyed as the writer emits them.

    Returns:
        ``tool_call_id`` → ``True`` when the call is written natively.
    """
    native: dict[str, bool] = {}
    for item in messages:
        if not isinstance(item, dict):
            continue
        content = item.get("content")
        if not isinstance(content, list):
            continue
        for part in content:
            if not is_intrinsic_part(part) or part.get("type") != "tool_call":
                continue
            fields = part.get("tool_input")
            cid = part.get("tool_call_id", "")
            native[id_key(cid) if id_key else cid] = tool_ops.intrinsic_renderable(
                intrinsic_kind_of(part),
                fields if isinstance(fields, dict) else {},
                leg="history",
            )
    return native
