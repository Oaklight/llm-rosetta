"""
LLM-Rosetta - Messages API stream reader (provider SSE event -> IR events).

The per-event handlers :class:`MessagesConverter` dispatches to through
``_P_TO_IR_DISPATCH``; kept here so the converter module stays focused on the
request/response interfaces.
"""

from typing import TYPE_CHECKING, Any

from llm_rosetta.ir.model.stream import (
    CitationDeltaEvent,
    ContentBlockEndEvent,
    ContentBlockStartEvent,
    IRStreamEvent,
    ReasoningDeltaEvent,
    StreamEndEvent,
    StreamStartEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallStartEvent,
    ToolResultEvent,
    UsageEvent,
)
from llm_rosetta.ir.context import (
    StreamContext,
    carry_registered_call_identity,
)
from ...base.helpers.opaque import (
    set_opaque_origin,
)
from ._constants import (
    ANTHROPIC_REASON_FROM_PROVIDER,
)
from .tool_ops import MessagesToolOps, canonical_server_tool_kind


class MessagesStreamReader:
    """Mixin holding the Messages SSE event -> IR stream event handlers."""

    if TYPE_CHECKING:
        # Provided by ``MessagesConverter`` / ``BaseConverter`` at runtime.
        content_ops: Any

        @staticmethod
        def _build_p_usage_to_ir(p_usage: dict[str, Any]) -> Any: ...

        def _get_response_id_prefix(self, context: Any = None) -> str: ...

        @staticmethod
        def strip_response_id_prefix(raw_id: str, prefix: str) -> str: ...

    def _handle_p_message_start_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Handle message_start → UsageEvent + StreamStartEvent.

        UsageEvent is emitted first so that downstream ir→p converters
        have ``pending_usage`` populated when they process the
        StreamStartEvent (needed for ``message_start.usage.input_tokens``).
        """
        message = chunk.get("message", {})

        # Emit usage BEFORE stream_start so ir→p can read input_tokens
        # from context.pending_usage when building message_start.
        usage = message.get("usage")
        if usage:
            start_usage = dict(usage)
            start_usage["output_tokens"] = 0
            events.append(
                UsageEvent(
                    type="usage",
                    usage=self._build_p_usage_to_ir(start_usage),
                )
            )

        if context is not None:
            response_id = message.get("id", "")
            model = message.get("model", "")
            prefix = self._get_response_id_prefix(context)
            context.response_id = self.strip_response_id_prefix(response_id, prefix)
            context.model = model
            context.mark_started()
            events.append(
                StreamStartEvent(
                    type="stream_start",
                    response_id=response_id,
                    model=model,
                )
            )

    def _handle_p_content_block_start_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Handle content_block_start → ContentBlockStartEvent + optional ToolCallStartEvent."""
        content_block = chunk.get("content_block", {})
        block_type = content_block.get("type", "")
        block_index = chunk.get("index", 0)

        if context is not None:
            context.next_block_index()
            events.append(
                ContentBlockStartEvent(
                    type="content_block_start",
                    block_index=block_index,
                    block_type=block_type,
                )
            )

        if block_type in ("tool_use", "server_tool_use"):
            tool_call_id = content_block.get("id", "")
            tool_name = content_block.get("name", "")
            if block_type == "server_tool_use":
                # Same fold as the block reader: a newer code-execution version
                # reports itself as ``bash_code_execution``, a kind no shim
                # declares.  Folding only on the non-streaming path left the
                # two disagreeing for the same response.
                tool_name = canonical_server_tool_kind(tool_name)

            start_evt = ToolCallStartEvent(
                type="tool_call_start",
                tool_call_id=tool_call_id,
                tool_name=tool_name,
            )
            if block_type == "server_tool_use":
                # The event has to say which it was, exactly as
                # ``p_tool_call_to_ir`` does for the non-streaming part: mark
                # it intrinsic and carry the canonical kind, so the writer
                # restores ``server_tool_use`` without inferring it from the
                # name.  ``tool_name`` is already folded above.
                start_evt["tool_type"] = "intrinsic"
                start_evt["intrinsic_kind"] = tool_name
            # The same two fields ``p_tool_call_to_ir`` keeps on the
            # non-streaming part, so a streamed call does not lose what the
            # block carries: a cache breakpoint, and a non-``direct`` caller
            # (the default caller is not re-emitted, as on the other path).
            cache_control = content_block.get("cache_control")
            if cache_control is not None:
                start_evt["cache_hint"] = cache_control
            caller = content_block.get("caller")
            if isinstance(caller, dict) and caller.get("type") != "direct":
                start_evt.setdefault("provider_metadata", {})["anthropic_caller"] = (
                    caller
                )
            if context is not None:
                # Register what the block said, not only its name: the deltas
                # that follow are keyed on the call id alone, so the kind (a
                # provider-executed call's whole identity) and the metadata the
                # block carried are read back from here onto each delta.
                context.register_tool_call(
                    tool_call_id,
                    tool_name,
                    tool_type=start_evt.get("tool_type", "function"),
                    provider_metadata=start_evt.get("provider_metadata"),
                    intrinsic_kind=start_evt.get("intrinsic_kind", ""),
                )
                context.block_index_to_tool_call_id[block_index] = tool_call_id
                start_evt["tool_call_index"] = context.tool_call_count - 1
            events.append(start_evt)

        elif isinstance(block_type, str) and block_type.endswith("_tool_result"):
            # Anthropic executes server tools and returns their results whole
            # in a single ``content_block_start`` (no deltas follow).  Build the
            # IR part with the same reader the non-streaming path uses so the
            # payload shape matches, then carry it as a one-shot ToolResultEvent.
            part = MessagesToolOps.p_server_tool_result_to_ir(content_block)
            result_evt = ToolResultEvent(
                type="tool_result",
                tool_call_id=part["tool_call_id"],
                result=part["result"],
                is_error=part.get("is_error", False),
            )
            tool_type = part.get("tool_type")
            if tool_type is not None:
                result_evt["tool_type"] = tool_type
            # Carry the canonical kind on the event's first-class field, the
            # same carrier ``p_server_tool_result_to_ir`` puts it on.
            if "intrinsic_kind" in part:
                result_evt["intrinsic_kind"] = part["intrinsic_kind"]
            if "cache_hint" in part:
                result_evt["cache_hint"] = part["cache_hint"]
            if "provider_metadata" in part:
                result_evt["provider_metadata"] = part["provider_metadata"]
            result_evt["block_index"] = block_index
            events.append(result_evt)

    def _handle_p_content_block_delta_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Handle content_block_delta → TextDeltaEvent, ToolCallDeltaEvent, or ReasoningDeltaEvent."""
        delta = chunk.get("delta", {})
        delta_type = delta.get("type", "")
        # Preserve provider block index on IR events for lossless round-trip (#246)
        chunk_block_index: int | None = chunk.get("index")

        if delta_type == "text_delta":
            evt = TextDeltaEvent(
                type="text_delta",
                text=delta.get("text", ""),
            )
            if chunk_block_index is not None:
                evt["block_index"] = chunk_block_index
            events.append(evt)
        elif delta_type == "input_json_delta":
            tool_call_id = ""
            if context is not None and chunk_block_index is not None:
                tool_call_id = context.block_index_to_tool_call_id.get(
                    chunk_block_index, ""
                )
            if not tool_call_id and context is not None and context.tool_call_id_map:
                tool_call_id = list(context.tool_call_id_map.keys())[-1]

            partial_json = delta.get("partial_json", "")
            delta_evt = ToolCallDeltaEvent(
                type="tool_call_delta",
                tool_call_id=tool_call_id,
                arguments_delta=partial_json,
            )
            if chunk_block_index is not None:
                delta_evt["block_index"] = chunk_block_index
            if context is not None and tool_call_id:
                # The arguments arrive as fragments that carry no identity of
                # their own, so each repeats what the call registered (kind and
                # metadata), as a start event would have carried it.
                carry_registered_call_identity(delta_evt, context, tool_call_id)
            events.append(delta_evt)

            if context is not None and tool_call_id:
                context.append_tool_call_args(tool_call_id, partial_json)
        elif delta_type == "thinking_delta":
            evt_r = ReasoningDeltaEvent(
                type="reasoning_delta",
                reasoning=delta.get("thinking", ""),
            )
            if chunk_block_index is not None:
                evt_r["block_index"] = chunk_block_index
            events.append(evt_r)
        elif delta_type == "citations_delta":
            self._citations_delta_to_ir(delta, chunk_block_index, events)
        elif delta_type == "signature_delta":
            sig_delta = delta.get("signature", "")
            evt_s = ReasoningDeltaEvent(
                type="reasoning_delta",
                reasoning="",
                signature=sig_delta,
            )
            if sig_delta:
                set_opaque_origin(evt_s, signature="messages")
            if chunk_block_index is not None:
                evt_s["block_index"] = chunk_block_index
            events.append(evt_s)

    def _citations_delta_to_ir(
        self,
        delta: dict[str, Any],
        block_index: int | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Anthropic's own citation delta → one IR citation event.

        The citation arrives before the text it supports, and the block it
        annotates opened with an empty ``citations`` array.
        """
        citation = delta.get("citation")
        if not isinstance(citation, dict):
            return
        event = CitationDeltaEvent(
            type="citation_delta",
            citation=self.content_ops.p_citation_to_ir(citation),
        )
        if block_index is not None:
            event["block_index"] = block_index
        events.append(event)

    def _handle_p_content_block_stop_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Handle content_block_stop → ContentBlockEndEvent."""
        if context is not None:
            block_index = chunk.get("index", 0)
            events.append(
                ContentBlockEndEvent(
                    type="content_block_end",
                    block_index=block_index,
                )
            )

    def _handle_p_message_delta_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Handle message_delta → UsageEvent + FinishEvent."""
        delta = chunk.get("delta", {})
        stop_reason = delta.get("stop_reason")

        # Emit UsageEvent before FinishEvent
        usage = chunk.get("usage")
        if usage:
            events.append(
                UsageEvent(
                    type="usage",
                    usage=self._build_p_usage_to_ir(usage),
                )
            )

        if stop_reason:
            reason = ANTHROPIC_REASON_FROM_PROVIDER.get(stop_reason, "stop")
            finish_reason: dict[str, Any] = {"reason": reason}
            # The body carries the matched sequence beside ``stop_reason``;
            # the stream's ``message_delta`` is where it rides, so read it back
            # the same way or the two legs disagree on the value (#1245).
            stop_sequence = delta.get("stop_sequence")
            if stop_sequence:
                finish_reason["stop_sequence"] = stop_sequence
            finish: dict[str, Any] = {
                "type": "finish",
                "finish_reason": finish_reason,
            }
            # Structured refusal: extract explanation and category
            if stop_reason == "refusal":
                stop_details = delta.get("stop_details") or {}
                # No canned sentence: a refusal that carried no words came
                # back with none, rather than with words nobody said.
                explanation = stop_details.get("explanation", "")
                category = stop_details.get("category")
                pm: dict[str, Any] = {"refusal_text": explanation}
                if category:
                    pm["anthropic_refusal_category"] = category
                finish["provider_metadata"] = pm
            events.append(finish)  # ty: ignore[invalid-argument-type]

    def _handle_p_message_stop_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Handle message_stop → StreamEndEvent."""
        if context is not None:
            context.mark_ended()
            events.append(StreamEndEvent(type="stream_end"))
