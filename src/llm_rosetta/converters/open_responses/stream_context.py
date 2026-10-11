"""OpenAI Responses API stream context with provider-specific state."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..base.context import StreamContext


@dataclass
class ReasoningItemState:
    """One reasoning item being assembled on the outbound stream.

    ``encrypted_content`` is cryptographically bound to ``item_id``
    upstream, so the two must stay together: replaying a blob under
    another item's id fails verification on the next turn.
    """

    item_id: str
    output_index: int = -1
    accumulated_text: str = ""
    encrypted_content: str = ""


@dataclass
class OpenResponsesStreamContext(StreamContext):
    """Stream context with OpenAI Responses API specific state.

    Extends the base StreamContext with fields needed for Responses API
    stream conversion, including output item tracking, text accumulation,
    and item-to-call-id resolution.

    Attributes:
        item_id_to_call_id: Reverse mapping from Responses item_id to
            tool call_id for function call argument delta resolution.
        output_item_emitted: Whether the initial message output_item.added
            and content_part.added events have been emitted.
        item_id: Current output item ID for the response message.
        accumulated_text: Accumulated text deltas for the final
            response.completed payload.
        content_part_done_emitted: Whether content_part.done has been
            emitted (prevents duplicate emission).
    """

    # Message item tracking
    item_id_to_call_id: dict[str, str] = field(default_factory=dict)
    output_item_emitted: bool = False
    item_id: str = ""
    accumulated_text: str = ""
    accumulated_refusal: str = ""
    content_part_done_emitted: bool = False
    _sequence_number: int = 0

    # Unified output item counter — allocates output_index for all item
    # types (reasoning, message, tool call).  Call next_output_index()
    # ONLY when emitting output_item.added.
    _output_item_counter: int = 0
    _message_output_index: int = -1

    # Tool call output_index storage (call_id → assigned index)
    _tool_call_output_indices: dict[str, int] = field(default_factory=dict, repr=False)

    # Namespace of each tool call (call_id → namespace), for tools that were
    # flattened out of a `namespace` container on the request leg.  Recorded
    # at tool_call_start so the later completed/done items can restore it.
    _tool_call_namespaces: dict[str, str] = field(default_factory=dict, repr=False)

    # Reasoning items being assembled outbound (IR → Responses), in
    # arrival order.  A turn can hold several — one per stretch of
    # thinking between tool calls — and each carries its own
    # encrypted_content, so this cannot collapse to a single item.
    _reasoning_items: list[ReasoningItemState] = field(default_factory=list, repr=False)

    # Id of the reasoning item currently streaming inbound (Responses →
    # IR), used to tag deltas with their source item.  Separate from
    # _reasoning_items: inbound and outbound use different contexts.
    _inbound_reasoning_item_id: str = ""

    def reasoning_item(self, item_id: str) -> ReasoningItemState | None:
        """Return the outbound reasoning item with this id, if started."""
        for item in self._reasoning_items:
            if item.item_id == item_id:
                return item
        return None

    def start_reasoning_item(
        self, item_id: str, output_index: int
    ) -> ReasoningItemState:
        """Begin a new outbound reasoning item and return its state."""
        item = ReasoningItemState(item_id=item_id, output_index=output_index)
        self._reasoning_items.append(item)
        return item

    @property
    def current_reasoning_item(self) -> ReasoningItemState | None:
        """The most recently started outbound reasoning item."""
        return self._reasoning_items[-1] if self._reasoning_items else None

    def next_output_index(self) -> int:
        """Allocate the next output_index.

        Call ONLY when emitting an output_item.added event.  Delta and
        done events must reference the index stored at added-time.
        """
        idx = self._output_item_counter
        self._output_item_counter += 1
        return idx

    @property
    def next_sequence_number(self) -> int:
        """The sequence number the next emitted event should carry.

        Responses numbers its stream events monotonically, so a
        synthesized terminal event has to continue the run the client
        has already consumed rather than restart it.
        """
        return self._sequence_number + 1

    @classmethod
    def from_base(cls, base: StreamContext) -> OpenResponsesStreamContext:
        """Create from a base StreamContext, preserving existing state.

        Args:
            base: The base StreamContext whose state should be carried over.

        Returns:
            A new OpenResponsesStreamContext with the base state copied.
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
        ctx._finished_choice_indexes = base._finished_choice_indexes
        ctx._tool_call_args = base._tool_call_args
        ctx._tool_call_order = base._tool_call_order
        ctx._tool_call_types = base._tool_call_types
        ctx._tool_call_index = base._tool_call_index
        # Subclass-only fields below start empty and are populated during
        # streaming, so they are not copied from `base` (which is a plain
        # StreamContext).  Listed here so a new field is not silently lost.
        # _tool_call_output_indices: allocated during streaming
        # _tool_call_namespaces: populated at tool_call_start events
        # _reasoning_items: appended as reasoning items start
        # _inbound_reasoning_item_id: set at reasoning output_item events
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


# Backward-compatible alias (deprecated): the OpenAI Responses profile reuses
# the same stream context as the vendor-neutral Open Responses base.
OpenAIResponsesStreamContext = OpenResponsesStreamContext
