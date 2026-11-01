"""OpenCode Zen free-lane request and response shims.

The free lane refuses a request that does not declare a tool named exactly
``read``.  The declaration is a gate, not a feature: it need not be called,
and ``bash``/``glob``/``grep`` are not required (measured 2026-10-16, see
issue #884).

A client that already declares ``read`` keeps its own tool untouched.  When
the gateway injects the declaration, a call to it cannot be answered by the
client, so the response leg removes such calls instead of forwarding a tool
the client never declared.  The injected declaration says it is a placeholder
so the model prefers the client's own tools; if it calls it anyway and no
kept call remains, ``finish_reason`` becomes ``stop`` rather than leaving the
client waiting for a tool result that will never arrive.
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


# Per-request tool-call bookkeeping, keyed by the choice's own index: two
# choices number their calls independently, so one choice's dropped index must
# not delete the other's call.  The two sets live in one state object so a
# reset cannot clear one and keep the other; ``ensure_read_tool`` resets it at
# the start of every request, because a reused context would otherwise carry a
# verdict from the previous one.
class _StreamState:
    """One stream's tool-call indexes, per choice."""

    __slots__ = ("dropped", "kept")

    def __init__(self) -> None:
        self.dropped: dict[int, set[int]] = {}
        self.kept: dict[int, set[int]] = {}


_STATE: ContextVar[_StreamState | None] = ContextVar(
    "opencode_zen_free_stream_state", default=None
)


def _state() -> _StreamState:
    """This request's stream state, created on first use."""
    state = _STATE.get()
    if state is None:
        state = _StreamState()
        _STATE.set(state)
    return state


def _reset_calls() -> None:
    """Start a request with no call state (a context may be reused)."""
    _STATE.set(None)


_READ_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "read",
        "description": (
            "Placeholder this transport requires; it is not connected to any "
            "implementation. Calling it does nothing and the call is "
            "discarded. Never call it — use the tools the client declared."
        ),
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
    _reset_calls()
    tools = body.get("tools")
    if isinstance(tools, list) and any(_is_read_tool(t) for t in tools):
        _INJECTED.set(False)
        return body
    body = dict(body)
    body["tools"] = [*(tools if isinstance(tools, list) else []), dict(_READ_TOOL)]
    _INJECTED.set(True)
    return body


def _is_stream_chunk(choices: list[Any]) -> bool:
    """Whether these choices belong to a stream rather than a whole response.

    A full response carries ``message`` on its choices; a stream chunk carries
    ``delta``, and the finish chunk may carry only ``finish_reason``.
    """
    return not any(isinstance(c, dict) and "message" in c for c in choices)


def _keep_stream_call(call: Any, drops: set[int], kept: set[int]) -> bool:
    """Return whether one streamed call survives, recording its index.

    A call naming the injected tool is dropped and its index remembered, so
    the argument fragments that follow are dropped with it.  An unindexed
    survivor records the ``-1`` sentinel: the turn did forward a call.
    """
    if not isinstance(call, dict):
        return True
    index = call.get("index")
    function = call.get("function")
    name = function.get("name") if isinstance(function, dict) else None
    if name == "read":
        if isinstance(index, int):
            drops.add(index)
        return False
    if isinstance(index, int) and index in drops:
        return False
    kept.add(index if isinstance(index, int) else -1)
    return True


def _rewrite_finish(choice: dict[str, Any], kept: Any) -> None:
    """A turn with no surviving call must not finish as `tool_calls`."""
    if choice.get("finish_reason") == "tool_calls" and not kept:
        choice["finish_reason"] = "stop"


def _drop_stream_chunk(body: dict[str, Any]) -> dict[str, Any]:
    state = _state()
    drops_by_choice = state.dropped
    kept_by_choice = state.kept
    for choice in body.get("choices") or []:
        if not isinstance(choice, dict):
            continue
        index = choice.get("index")
        choice_index = index if isinstance(index, int) else 0
        delta = choice.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("tool_calls"), list):
            drops = drops_by_choice.setdefault(choice_index, set())
            kept = kept_by_choice.setdefault(choice_index, set())
            delta["tool_calls"] = [
                call
                for call in delta["tool_calls"]
                if _keep_stream_call(call, drops, kept)
            ]
        _rewrite_finish(choice, kept_by_choice.get(choice_index, set()))
    return body


def _drop_response_body(body: dict[str, Any]) -> dict[str, Any]:
    for choice in body["choices"]:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not (
            isinstance(message, dict) and isinstance(message.get("tool_calls"), list)
        ):
            continue
        survivors = [c for c in message["tool_calls"] if not _is_read_call(c)]
        if len(survivors) != len(message["tool_calls"]):
            message["tool_calls"] = survivors
            _rewrite_finish(choice, survivors)
    return body


def drop_injected_read_calls(body: dict[str, Any]) -> dict[str, Any]:
    """Remove calls to the injected `read` tool from a response."""
    if not _INJECTED.get() or not isinstance(body, dict):
        return body
    choices = body.get("choices")
    if not isinstance(choices, list):
        return body
    # A full response carries `message` on its choices; anything else is a
    # stream chunk, including the finish chunk that carries only a reason.
    if _is_stream_chunk(choices):
        return _drop_stream_chunk(body)
    return _drop_response_body(body)


def _zero_price(value: Any) -> bool:
    """Whether a price field states zero (a boolean is not a price)."""
    return (
        isinstance(value, (int, float)) and not isinstance(value, bool) and value == 0
    )


def _is_free_model(entry: dict[str, Any]) -> bool:
    """Whether the listing marks *entry* as free.

    The catalogue states a per-token input and output price; a model whose
    both are zero is the free slice.  A block that states one side only is
    not a free statement.  A listing with no cost block at all falls back to
    the ``-free`` suffix, so a metadata change cannot hide every model from
    the panel — though that fallback would miss a zero-cost model without the
    suffix (``big-pickle`` today).
    """
    cost = entry.get("cost")
    text = cost.get("text") if isinstance(cost, dict) else None
    if isinstance(text, dict):
        input_cost = text.get("input")
        output_cost = text.get("output")
        price_in = input_cost.get("price") if isinstance(input_cost, dict) else None
        price_out = output_cost.get("price") if isinstance(output_cost, dict) else None
        if price_in is not None or price_out is not None:
            return _zero_price(price_in) and _zero_price(price_out)
    return str(entry.get("id", "")).endswith("-free")


def model_list_transform(entries: list[Any]) -> tuple[list[str], dict[str, str]]:
    """Keep only the models the free lane serves.

    The lane's ``/models`` lists the whole Zen catalogue while the keyless
    lane serves only the free slice; offering the rest would route a model the
    lane refuses.  Ids pass through unchanged, so the upstream map is empty.
    """
    ids = [
        entry["id"]
        for entry in entries
        if isinstance(entry, dict)
        and isinstance(entry.get("id"), str)
        and entry["id"]
        and _is_free_model(entry)
    ]
    return ids, {}


post_ir_transforms = (ensure_read_tool,)
response_body_transforms = (drop_injected_read_calls,)
