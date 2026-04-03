"""OpenAI Responses API stream context with provider-specific state."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..base.stream_context import StreamContext


@dataclass
class OpenAIResponsesStreamContext(StreamContext):
    """Stream context with OpenAI Responses API specific state.

    Extends the base StreamContext with fields needed for Responses API
    stream conversion, including output item tracking, text accumulation,
    and item-to-call-id resolution.

    Attributes:
        item_id_to_call_id: Reverse mapping from Responses item_id to
            tool call_id for function call argument delta resolution.
        output_item_emitted: Whether the initial output_item.added and
            content_part.added events have been emitted.
        item_id: Current output item ID for the response message.
        accumulated_text: Accumulated text deltas for the final
            response.completed payload.
        content_part_done_emitted: Whether content_part.done has been
            emitted (prevents duplicate emission).
    """

    item_id_to_call_id: dict[str, str] = field(default_factory=dict)
    output_item_emitted: bool = False
    item_id: str = ""
    accumulated_text: str = ""
    content_part_done_emitted: bool = False

    def register_tool_call_item(self, tool_call_id: str, item_id: str) -> None:
        """Register tool call item with reverse item_id mapping.

        Extends the base implementation to also populate
        ``item_id_to_call_id`` for Responses API delta resolution.

        Args:
            tool_call_id: The stable tool correlation identifier.
            item_id: The Responses output item identifier for the function call.
        """
        super().register_tool_call_item(tool_call_id, item_id)
        if tool_call_id and item_id:
            self.item_id_to_call_id[item_id] = tool_call_id
