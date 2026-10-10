"""Intrinsic tool helpers — shared utilities for provider-hosted tool handling.

Intrinsic tools are provider-native server-side capabilities (web search,
code execution, file search, etc.) represented in the IR with
``tool_type="intrinsic"`` and a ``provider_metadata["intrinsic_kind"]``
string identifying the specific capability.
"""

import json as _json
from typing import Any


def is_intrinsic_part(part: Any) -> bool:
    """Check whether an IR content part is an intrinsic tool call or result."""
    return (
        isinstance(part, dict)
        and part.get("type") in ("tool_call", "tool_result")
        and part.get("tool_type") == "intrinsic"
    )


def get_intrinsic_kind(part: Any, fallback: str = "") -> str:
    """Read ``intrinsic_kind`` from a part's provider_metadata."""
    pm = part.get("provider_metadata") or {}
    return pm.get("intrinsic_kind", fallback)


def set_intrinsic_kind(part: Any, kind: str) -> None:
    """Write ``intrinsic_kind`` into a part's provider_metadata (in place)."""
    part.setdefault("provider_metadata", {})["intrinsic_kind"] = kind


def make_intrinsic_tool_call(
    *,
    tool_call_id: str,
    tool_name: str,
    tool_input: dict[str, Any],
    intrinsic_kind: str,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an IR ToolCallPart with intrinsic tool_type."""
    pm: dict[str, Any] = {"intrinsic_kind": intrinsic_kind}
    if extra_metadata:
        pm.update(extra_metadata)
    return {
        "type": "tool_call",
        "tool_call_id": tool_call_id,
        "tool_name": tool_name,
        "tool_input": tool_input,
        "tool_type": "intrinsic",
        "provider_metadata": pm,
    }


def make_intrinsic_tool_result(
    *,
    tool_call_id: str,
    result: Any,
    intrinsic_kind: str,
    is_error: bool = False,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build an IR ToolResultPart with intrinsic tool_type."""
    pm: dict[str, Any] = {"intrinsic_kind": intrinsic_kind}
    if extra_metadata:
        pm.update(extra_metadata)
    part: dict[str, Any] = {
        "type": "tool_result",
        "tool_call_id": tool_call_id,
        "result": result,
        "tool_type": "intrinsic",
        "provider_metadata": pm,
    }
    if is_error:
        part["is_error"] = True
    return part


# ---------------------------------------------------------------------------
# Cross-format translation: degrade intrinsic parts to function equivalents
# ---------------------------------------------------------------------------


def _flatten_result(result: Any) -> str:
    """Normalise an intrinsic tool result to a plain string.

    Lists of content blocks are joined with newlines (non-text blocks are
    JSON-serialised), dicts are JSON-serialised, and everything else —
    including ``None`` and numbers — falls back to ``str()``.

    Note: non-text blocks (e.g. base64 image data) are serialised verbatim,
    which can produce a large string.  Degrading them to text is still
    preferable to dropping the tool result entirely.
    """
    if isinstance(result, str):
        return result
    if isinstance(result, list):
        parts: list[str] = []
        for item in result:
            if isinstance(item, dict):
                text = item.get("text")
                if text is not None:
                    parts.append(str(text))
                else:
                    parts.append(_json.dumps(item, ensure_ascii=False))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    if isinstance(result, dict):
        return _json.dumps(result, ensure_ascii=False)
    return str(result)


def _translated_metadata(part: dict[str, Any], kind: str) -> dict[str, Any]:
    """Build provider_metadata for a translated part.

    Preserves any original metadata keys (other than ``intrinsic_kind``,
    which no longer applies once the part is a function call) and records
    the translation lineage.
    """
    meta = {
        k: v
        for k, v in (part.get("provider_metadata") or {}).items()
        if k != "intrinsic_kind"
    }
    meta["_translated_from"] = "intrinsic"
    meta["_original_intrinsic_kind"] = kind
    return meta


def intrinsic_call_to_function(part: dict[str, Any]) -> dict[str, Any]:
    """Degrade an intrinsic tool_call to a function tool_call.

    Used during cross-format conversion to preserve conversation context
    that would otherwise be stripped.  The ``_intrinsic--`` prefix on the
    tool name prevents collisions with user-defined functions.
    """
    kind = get_intrinsic_kind(part, part.get("tool_name", "unknown"))
    translated: dict[str, Any] = {
        "type": "tool_call",
        "tool_call_id": part["tool_call_id"],
        "tool_name": f"_intrinsic--{kind}",
        "tool_input": part.get("tool_input", {}),
        "tool_type": "function",
        "provider_metadata": _translated_metadata(part, kind),
    }
    if "cache_hint" in part:
        translated["cache_hint"] = part["cache_hint"]
    return translated


def intrinsic_result_to_function(part: dict[str, Any]) -> dict[str, Any]:
    """Degrade an intrinsic tool_result to a function tool_result.

    Result content is flattened to a plain string so every target
    converter can serialize it.
    """
    kind = get_intrinsic_kind(part, "unknown")
    translated: dict[str, Any] = {
        "type": "tool_result",
        "tool_call_id": part["tool_call_id"],
        "result": _flatten_result(part.get("result", "")),
        "tool_type": "function",
        "provider_metadata": _translated_metadata(part, kind),
    }
    if part.get("is_error"):
        translated["is_error"] = True
    if "cache_hint" in part:
        translated["cache_hint"] = part["cache_hint"]
    return translated


# ---------------------------------------------------------------------------
# Intrinsic tool definitions
# ---------------------------------------------------------------------------
#
# An IR ToolDefinition may be ``type="intrinsic"`` for a provider-hosted
# server tool (web search, code execution, …).  The kind is stored in
# ``metadata["intrinsic_kind"]``.  Whether a provider enables it is declared
# by that provider's shim (``ToolsConfig.intrinsic_tools``).


def make_intrinsic_tool_definition(
    kind: str,
    *,
    description: str = "",
    parameters: dict[str, Any] | None = None,
    native: dict[str, Any] | None = None,
    native_base: str | None = None,
) -> dict[str, Any]:
    """Build an IR ToolDefinition of ``type="intrinsic"`` from a kind.

    ``native`` (with ``native_base``) stores the provider's original tool
    payload so a same-provider round-trip can restore it verbatim (extra
    config such as search filters is otherwise lost when the def is
    reduced to a canonical kind).
    """
    meta: dict[str, Any] = {"intrinsic_kind": kind}
    if native is not None:
        meta["_native"] = {"base": native_base, "tool": native}
    return {
        "type": "intrinsic",
        "name": kind,
        "description": description,
        "parameters": parameters or {},
        "metadata": meta,
    }


def get_definition_kind(ir_tool: Any) -> str:
    """Read the intrinsic kind from an IR ToolDefinition's metadata."""
    meta = ir_tool.get("metadata") or {}
    return meta.get("intrinsic_kind") or ""


def get_native_definition(ir_tool: Any, base: str) -> dict[str, Any] | None:
    """Return the stored native payload if it came from ``base``, else None."""
    meta = ir_tool.get("metadata") or {}
    native = meta.get("_native")
    if isinstance(native, dict) and native.get("base") == base:
        tool = native.get("tool")
        return tool if isinstance(tool, dict) else None
    return None
