"""Chat-completion delta merging: stream chunks to one response."""

from __future__ import annotations

from typing import Any


from ..logging import (
    get_logger,
)

logger = get_logger()


def _merge_tool_call_delta(merged_calls: list[dict[str, Any]], call: Any) -> None:
    """Fold one streamed tool-call fragment into the accumulated call list."""
    if not isinstance(call, dict):
        return
    position = call.get("index")
    if not isinstance(position, int):
        position = len(merged_calls)
    while len(merged_calls) <= position:
        merged_calls.append(
            {
                "id": "",
                "type": "function",
                "function": {"name": "", "arguments": ""},
            }
        )
    target = merged_calls[position]
    if call.get("id"):
        target["id"] = call["id"]
    if call.get("type"):
        target["type"] = call["type"]
    function = call.get("function")
    if isinstance(function, dict):
        if function.get("name"):
            target["function"]["name"] = function["name"]
        if isinstance(function.get("arguments"), str):
            target["function"]["arguments"] += function["arguments"]


def _merge_choice_delta(acc: dict[str, Any], delta: Any) -> None:
    """Fold one streamed delta into an accumulated choice."""
    if not isinstance(delta, dict):
        return
    if delta.get("role"):
        acc["message"]["role"] = delta["role"]
    for field_name in ("content", "reasoning_content", "refusal"):
        value = delta.get(field_name)
        if isinstance(value, str):
            acc["message"][field_name] = acc["message"].get(field_name, "") + value
    calls = delta.get("tool_calls")
    if isinstance(calls, list):
        merged_calls = acc["message"].setdefault("tool_calls", [])
        for call in calls:
            _merge_tool_call_delta(merged_calls, call)


def merge_chat_completion_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Reassemble a non-streaming Chat Completions reply from stream events.

    Only the fields a client reads are merged: identity, per-choice message
    content and reasoning, tool calls by index, the finish reason and the
    usage block.  Vendor echo fields are not invented — a non-streaming
    request that was served by a streaming upstream is a synthesized reply,
    and claiming fields the vendor never sent would be worse than omitting
    them.

    Args:
        events: Source-format ``chat.completion.chunk`` events, in order.

    Returns:
        A ``chat.completion`` object.
    """
    first: dict[str, Any] = {}
    choices: dict[int, dict[str, Any]] = {}
    usage: dict[str, Any] | None = None
    for event in events:
        if not isinstance(event, dict):
            continue
        if not first:
            first = event
        if isinstance(event.get("usage"), dict):
            usage = event["usage"]
        for choice in event.get("choices") or []:
            if not isinstance(choice, dict):
                continue
            index = choice.get("index", 0)
            acc = choices.setdefault(
                index,
                {
                    "index": index,
                    "message": {"role": "assistant"},
                    "finish_reason": None,
                },
            )
            _merge_choice_delta(acc, choice.get("delta"))
            if choice.get("finish_reason") is not None:
                acc["finish_reason"] = choice["finish_reason"]

    merged: dict[str, Any] = {
        "id": first.get("id", ""),
        "object": "chat.completion",
        "created": first.get("created", 0),
        "model": first.get("model", ""),
        "choices": [choices[key] for key in sorted(choices)],
    }
    if isinstance(first.get("system_fingerprint"), str):
        merged["system_fingerprint"] = first["system_fingerprint"]
    if usage is not None:
        merged["usage"] = usage
    return merged
