"""OpenAI Responses converter utilities — shared helpers for stream conversion."""

from __future__ import annotations

from typing import Any

from ..base.context import StreamContext
from ._constants import ResponsesEventType, generate_message_id
from .stream_context import OpenAIResponsesStreamContext


def resolve_call_id(chunk: dict[str, Any], context: StreamContext | None) -> str:
    """Resolve a tool call_id from a Responses API event chunk.

    The Responses API may provide either ``call_id`` or ``item_id`` on
    function-call-related events.  When only ``item_id`` is present and
    the context carries an item-to-call-id mapping, the call_id is
    resolved via that mapping.

    Args:
        chunk: Responses API event dict.
        context: Stream context (must be ``OpenAIResponsesStreamContext``
            for item_id resolution to work).

    Returns:
        The resolved call_id, or ``""`` if unresolvable.
    """
    call_id = chunk.get("call_id", "")
    if not call_id and isinstance(context, OpenAIResponsesStreamContext):
        item_id = chunk.get("item_id", "")
        if item_id:
            call_id = context.item_id_to_call_id.get(item_id, "")
    return call_id


def build_message_preamble_events(
    context: OpenAIResponsesStreamContext,
    output_index: int = 0,
) -> list[dict[str, Any]]:
    """Build output_item.added + content_part.added for a new message item.

    Marks the context as having emitted the output item and generates
    the item_id.  Must only be called when ``context.output_item_emitted``
    is ``False``.

    Args:
        context: Responses stream context.
        output_index: The output array index for the item.

    Returns:
        Two-element list of SSE event dicts.
    """
    context.output_item_emitted = True
    item_id = generate_message_id(context.response_id)
    context.item_id = item_id
    return [
        {
            "type": ResponsesEventType.OUTPUT_ITEM_ADDED,
            "output_index": output_index,
            "item": {
                "id": item_id,
                "type": "message",
                "role": "assistant",
                "status": "in_progress",
                "content": [],
            },
        },
        {
            "type": ResponsesEventType.CONTENT_PART_ADDED,
            "item_id": item_id,
            "output_index": output_index,
            "content_index": 0,
            "part": {
                "type": "output_text",
                "text": "",
                "annotations": [],
                "logprobs": [],
            },
        },
    ]
