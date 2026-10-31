"""OpenCode Zen free-lane request and response shims.

The free lane refuses a request that does not declare a tool named exactly
``read``.  The declaration is a gate, not a feature: it need not be called,
and ``bash``/``glob``/``grep`` are not required (measured 2026-10-16, see
issue #884).

A client that already declares ``read`` keeps its own tool untouched.  When
the gateway injects the declaration, a call to it cannot be answered by the
client, so the response leg removes such calls instead of forwarding a tool
the client never declared.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any

# Whether this request's `read` tool was injected by the gateway.  Both the
# request and the response leg run in the same request task, so one context
# variable carries the answer between them without shared mutable state.
_INJECTED: ContextVar[bool] = ContextVar(
    "opencode_zen_free_read_injected", default=False
)

# Per-stream set of tool-call indexes that name the injected tool, keyed by
# the upstream chunk id.  Argument fragments for those indexes arrive in later
# chunks and must be dropped with the call.
_DROPPED: ContextVar[dict[str, set[int]] | None] = ContextVar(
    "opencode_zen_free_dropped_calls", default=None
)

_READ_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read",
        "description": "Read a file from the local filesystem.",
        "parameters": {
            "type": "object",
            "properties": {
                "filePath": {
                    "type": "string",
                    "description": "Path of the file to read.",
                }
            },
            "required": ["filePath"],
        },
    },
}


def _is_read_tool(tool: Any) -> bool:
    return (
        isinstance(tool, dict)
        and isinstance(tool.get("function"), dict)
        and tool["function"].get("name") == "read"
    )


def _is_read_call(call: Any) -> bool:
    return (
        isinstance(call, dict)
        and isinstance(call.get("function"), dict)
        and call["function"].get("name") == "read"
    )


def ensure_read_tool(body: dict[str, Any]) -> dict[str, Any]:
    """Declare the `read` tool the free lane requires, if it is absent."""
    if not isinstance(body, dict):
        return body
    tools = body.get("tools")
    if isinstance(tools, list) and any(_is_read_tool(t) for t in tools):
        _INJECTED.set(False)
        return body
    body = dict(body)
    body["tools"] = [*(tools if isinstance(tools, list) else []), dict(_READ_TOOL)]
    _INJECTED.set(True)
    return body


def _drops_for(chunk_id: str) -> set[int]:
    state = _DROPPED.get()
    if state is None:
        state = {}
        _DROPPED.set(state)
    return state.setdefault(chunk_id, set())


def _drop_stream_chunk(body: dict[str, Any]) -> dict[str, Any]:
    chunk_id = str(body.get("id") or "")
    drops = _drops_for(chunk_id)
    for choice in body.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        delta = choice.get("delta")
        if not isinstance(delta, dict):
            continue
        calls = delta.get("tool_calls")
        if not isinstance(calls, list):
            continue
        kept = []
        for call in calls:
            if not isinstance(call, dict):
                kept.append(call)
                continue
            index = call.get("index")
            function = call.get("function")
            name = function.get("name") if isinstance(function, dict) else None
            if name == "read":
                if isinstance(index, int):
                    drops.add(index)
                continue
            if isinstance(index, int) and index in drops:
                # An argument fragment for a call already dropped.
                continue
            kept.append(call)
        delta["tool_calls"] = kept
    return body


def drop_injected_read_calls(body: dict[str, Any]) -> dict[str, Any]:
    """Remove calls to the injected `read` tool from a response."""
    if not _INJECTED.get() or not isinstance(body, dict):
        return body
    if isinstance(body.get("choices"), list):
        # A clump of deltas is a stream chunk when it carries `delta`; a full
        # response carries `message`.  Distinguish on the choice's own keys.
        if any(isinstance(c, dict) and "delta" in c for c in body["choices"]):
            return _drop_stream_chunk(body)
        for choice in body["choices"]:
            message = choice.get("message") if isinstance(choice, dict) else None
            if isinstance(message, dict) and isinstance(
                message.get("tool_calls"), list
            ):
                kept = [c for c in message["tool_calls"] if not _is_read_call(c)]
                if len(kept) != len(message["tool_calls"]):
                    message["tool_calls"] = kept
    return body


post_ir_transforms = (ensure_read_tool,)
response_body_transforms = (drop_injected_read_calls,)
