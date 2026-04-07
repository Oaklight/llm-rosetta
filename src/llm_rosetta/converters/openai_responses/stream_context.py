"""OpenAI Responses API stream context with provider-specific state."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..base.context import StreamContext


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

    @classmethod
    def from_base(cls, base: StreamContext) -> OpenAIResponsesStreamContext:
        """Create from a base StreamContext, preserving existing state.

        Args:
            base: The base StreamContext whose state should be carried over.

        Returns:
            A new OpenAIResponsesStreamContext with the base state copied.
        """
        ctx = cls()
        # Copy base StreamContext fields
        ctx.warnings = base.warnings
        ctx.options = base.options
        ctx.metadata = base.metadata
        ctx.response_id = base.response_id
        ctx.model = base.model
        ctx.created = base.created
        ctx.current_block_index = base.current_block_index
        ctx.tool_call_id_map = base.tool_call_id_map
        ctx.tool_call_item_id_map = base.tool_call_item_id_map
        ctx.pending_usage = base.pending_usage
        ctx.pending_finish = base.pending_finish
        ctx.pending_response = base.pending_response
        ctx._started = base._started
        ctx._ended = base._ended
        ctx._tool_call_args = base._tool_call_args
        ctx._tool_call_order = base._tool_call_order
        return ctx

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
