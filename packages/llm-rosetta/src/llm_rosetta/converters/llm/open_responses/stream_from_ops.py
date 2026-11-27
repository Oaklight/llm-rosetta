"""Provider-to-IR streaming for the Open Responses converter."""

import json
from typing import Any

from llm_rosetta.ir.model.passthrough import ProviderPassthroughEvent
from llm_rosetta.ir.model.stream import (
    CitationDeltaEvent,
    ContentBlockEndEvent,
    ContentBlockStartEvent,
    FinishEvent,
    IRStreamEvent,
    ReasoningDeltaEvent,
    RefusalDeltaEvent,
    StreamEndEvent,
    StreamStartEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallStartEvent,
    ToolResultEvent,
    UsageEvent,
)
from ...base import BaseConverter
from llm_rosetta.ir.context import StreamContext
from ...base.helpers.opaque import (
    set_opaque_origin,
)
from .stream_context import OpenResponsesStreamContext
from ._mixin_host import OpenResponsesMixinHost
from ._constants import (
    ResponsesEventType,
)
from .tool_ops import (
    _INTRINSIC_RESULT_TYPES,
    _ITEM_TO_INTRINSIC_KIND,
    _server_item_tool_input,
    OpenResponsesToolOps,
)
from .utils import resolve_call_id
from ._wire_helpers import (
    _stream_finish_reason,
    _response_failure_metadata,
)


#: Server-tool output item types this format reads (``web_search_call``,
#: ``code_interpreter_call``, …).  A streamed item of one of these types is a
#: provider-executed call; the value is its canonical intrinsic kind.
_SERVER_ITEM_INTRINSIC_KINDS: frozenset[str] = frozenset(_ITEM_TO_INTRINSIC_KIND)

#: Server *result* item types (``code_interpreter_call_output``, …).  A
#: streamed item of one of these types is a provider-executed result, carried
#: as a one-shot ``ToolResultEvent``.
_SERVER_RESULT_ITEM_TYPES: frozenset[str] = frozenset(_INTRINSIC_RESULT_TYPES)


class OpenResponsesStreamFromMixin(OpenResponsesMixinHost):
    """Handlers that turn Responses SSE events into IR stream events."""

    def stream_response_from_provider(
        self,
        chunk: dict[str, Any] | None,
        context: StreamContext | None = None,
    ) -> list[IRStreamEvent]:
        """Convert an OpenAI Responses SSE event to IR stream events.

        OpenAI Responses API uses fine-grained SSE events with a ``type`` field
        (e.g. ``response.output_text.delta``) instead of the ``choices[].delta``
        structure used by Chat Completions.

        A single event typically produces zero or one IR events, but
        ``response.completed`` may produce both a ``FinishEvent`` and a
        ``UsageEvent``.

        When a ``context`` is provided, lifecycle events (``StreamStartEvent``,
        ``ContentBlockStartEvent``, ``ContentBlockEndEvent``,
        ``StreamEndEvent``) are emitted and cross-event state is tracked.
        Without a context the behaviour is identical to the previous
        implementation (backward compatible).

        Pass ``None`` as the chunk to signal end-of-stream; nothing of this
        dialect is held back across chunks, so it flushes nothing.

        Args:
            chunk: OpenAI Responses SSE event dict (or SDK object), or
                ``None`` to signal end-of-stream.
            context: Optional stream context for stateful conversions.

        Returns:
            List of IR stream events extracted from the event.
        """
        if chunk is None:
            # The EOF sentinel: this dialect holds nothing back across chunks
            # (the event that ends its stream carries the usage the Chat
            # reader waits for), so there is nothing to flush.  Callers pass
            # it — ``BaseConverter.iter_stream_from_provider`` and the
            # gateway's per-chunk ``StreamProcessor`` both do — so it must not
            # reach the normalizing below.
            return []

        chunk = self._normalize(chunk)
        events: list[IRStreamEvent] = []
        event_type = chunk.get("type", "")

        handler_name = self._P_TO_IR_DISPATCH.get(event_type)
        if handler_name is not None:
            getattr(self, handler_name)(chunk, context, events)

        # All other event types (response.in_progress, response.queued,
        # response.output_text.done, etc.) are dropped — response.in_progress
        # is redundant with response.created and is re-synthesised by
        # _handle_ir_stream_start_to_p on the to_provider side.

        # ``_stream_finish_reason`` reads the completed frame's own output
        # list, which this dialect may omit or contradict.  The frame's list
        # decides when it names the call, and the opened call decides when
        # the list says nothing about it: the context remembers every client
        # call the stream opened, and the finalizer applies that to the
        # finish this chunk produced (#1601).  A stream that names its call
        # only in ``output_item.done`` stays a normal stop — ``added``
        # precedes ``done`` in this dialect, so that stream is out of spec
        # and no ``tool_call_start`` was emitted for the memory to hold.
        return self.finalize_stream_finish(events, context)

    # --- from_provider handlers ---

    def _handle_p_response_created_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        if context is not None:
            response = chunk.get("response", {})
            response_id = self.strip_response_id_prefix(
                response.get("id", ""),
                self._get_response_id_prefix(context),
            )
            model = response.get("model", "")
            created = int(response.get("created_at", 0))
            context.response_id = response_id
            context.model = model
            context.created = created
            context.mark_started()
            start_event: StreamStartEvent = {
                "type": "stream_start",
                "response_id": response_id,
                "model": model,
            }
            if created:
                start_event["created"] = created
            events.append(start_event)

            # Preserve mode: capture echo fields from the initial response
            if context.metadata_mode == "preserve" and isinstance(response, dict):
                extras = {
                    k: v
                    for k, v in response.items()
                    if k in self._get_preserve_fields(context)
                }
                if extras:
                    context.store_response_extras(extras)

    def _handle_p_output_text_delta_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        events.append(
            TextDeltaEvent(
                type="text_delta",
                text=chunk.get("delta", ""),
            )
        )

    def _handle_p_output_text_annotation_added_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """A streamed annotation is a citation of the text just sent."""
        annotation = chunk.get("annotation")
        if isinstance(annotation, dict):
            events.append(
                CitationDeltaEvent(
                    type="citation_delta",
                    citation=self.content_ops.p_citation_to_ir(annotation),
                )
            )

    def _handle_p_reasoning_delta_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        event = ReasoningDeltaEvent(
            type="reasoning_delta",
            reasoning=chunk.get("delta", ""),
        )
        if (
            isinstance(context, OpenResponsesStreamContext)
            and context._inbound_reasoning_item_id
        ):
            event["provider_metadata"] = {
                "responses_reasoning_id": context._inbound_reasoning_item_id,
            }
        events.append(event)

    def _handle_p_refusal_delta_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        events.append(
            RefusalDeltaEvent(
                type="refusal_delta",
                refusal=chunk.get("delta", ""),
            )
        )

    def _handle_p_output_item_added_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        item = chunk.get("item", {})
        if isinstance(item, dict):
            item_type = item.get("type", "")

            if item_type in ("function_call", "custom_tool_call"):
                self._emit_client_call_start_ir(item, chunk, context, events)

            elif item_type in _SERVER_ITEM_INTRINSIC_KINDS:
                self._emit_server_call_start_ir(item, chunk, context, events)

            elif item_type in _SERVER_RESULT_ITEM_TYPES:
                self._emit_server_result_ir(item, chunk, events)

            elif item_type == "reasoning" and isinstance(
                context, OpenResponsesStreamContext
            ):
                # Capture early so deltas can carry the source ID;
                # done event re-confirms authoritatively.
                item_id = item.get("id", "")
                if item_id:
                    context._inbound_reasoning_item_id = item_id

            elif item_type == "message":
                # Message-level output item — no IR event needed.
                # The actual content block is signaled by the subsequent
                # ``response.content_part.added`` event which produces its
                # own ``ContentBlockStartEvent``.  Emitting a
                # ``ContentBlockStartEvent`` here would cause duplicate
                # ``response.content_part.added`` events on round-trip.
                #
                # Remember which item this is instead: the blocks that
                # follow carry its id and phase, which is how the write leg
                # tells a turn's several message items apart (#997).
                if isinstance(context, OpenResponsesStreamContext):
                    context._inbound_message_item_id = item.get("id", "")
                    side = item.get("phase")
                    context._inbound_message_phase = (
                        side if isinstance(side, str) else ""
                    )

            elif self._is_opaque_item(item):
                events.append(self._opaque_item_event(chunk, item))

    def _opaque_item_event(
        self, chunk: dict[str, Any], item: dict[str, Any]
    ) -> ProviderPassthroughEvent:
        """The passthrough event for one frame of an opaque item.

        The payload is the whole frame, for the writer to replay; ``item`` and
        ``position`` let a fold keep the item itself (see
        :class:`ProviderPassthroughEvent`).
        """
        event = ProviderPassthroughEvent(
            type="provider_passthrough",
            provider=self._CONVERTER_TAG,
            payload=dict(chunk),
            item=dict(item),
        )
        position = chunk.get("output_index")
        if isinstance(position, int):
            event["position"] = position
        return event

    def _is_opaque_item(self, item: dict[str, Any]) -> bool:
        """Whether a streamed item is one the body leg keeps as passthrough.

        The body leg keeps every item type it does not rebuild
        (:meth:`is_portable_response_output_item`): ``tool_search_*``,
        ``compaction`` and the implementor-prefixed types the spec allows
        (``acme:search_result``).  The stream leg keeps the same set (#1740),
        less the server result items it turns into IR events itself.  An item
        with no type is malformed, not an unknown type, and is dropped.
        """
        item_type = item.get("type")
        return (
            isinstance(item_type, str)
            and bool(item_type)
            and not self.message_ops.is_portable_response_output_item(item)
            and item_type not in _SERVER_RESULT_ITEM_TYPES
        )

    @staticmethod
    def _emit_client_call_start_ir(
        item: dict[str, Any],
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Emit the IR ``tool_call_start`` for a client tool call item.

        Covers the upstream's own render of a client call — a
        ``function_call`` or a ``custom_tool_call``.  The ``namespace`` the
        item echoes beside the name is carried onto the event as the
        first-class ``tool_group``, so the source leg can hand it back to a
        client that grouped the tool.
        """
        call_id = item.get("call_id", "")
        item_id = item.get("id", "")
        tool_type = "custom" if item.get("type") == "custom_tool_call" else "function"

        # Register tool call in context
        if context is not None:
            context.register_tool_call(call_id, item.get("name", ""), tool_type)
            context.register_tool_call_item(call_id, item_id)

        start_event = ToolCallStartEvent(
            type="tool_call_start",
            tool_call_id=call_id,
            tool_name=item.get("name", ""),
            tool_type=tool_type,
        )
        namespace = item.get("namespace")
        if isinstance(namespace, str) and namespace:
            start_event["tool_group"] = namespace
        if item_id and item_id != call_id:
            start_event["provider_metadata"] = {"responses_item_id": item_id}
        output_index = chunk.get("output_index")
        if output_index is not None:
            start_event["tool_call_index"] = output_index
        events.append(start_event)

    @staticmethod
    def _emit_server_call_start_ir(
        item: dict[str, Any],
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Emit an intrinsic ToolCallStartEvent for a provider-executed call.

        The kind comes from the item's own type, never the name, so a server
        call never reads as a client ``function_call``.
        """
        intrinsic_kind = _ITEM_TO_INTRINSIC_KIND[item["type"]]
        call_id = item.get("call_id") or item.get("id", "")
        item_id = item.get("id", "")
        if context is not None:
            context.register_tool_call(call_id, intrinsic_kind, "intrinsic")
            context.register_tool_call_item(call_id, item_id)
        start_event = ToolCallStartEvent(
            type="tool_call_start",
            tool_call_id=call_id,
            tool_name=intrinsic_kind,
            tool_type="intrinsic",
        )
        start_event["intrinsic_kind"] = intrinsic_kind
        if item_id and item_id != call_id:
            start_event.setdefault("provider_metadata", {})["responses_item_id"] = (
                item_id
            )
        output_index = chunk.get("output_index")
        if output_index is not None:
            start_event["tool_call_index"] = output_index
        events.append(start_event)

    def _emit_server_result_ir(
        self,
        item: dict[str, Any],
        chunk: dict[str, Any],
        events: list[IRStreamEvent],
    ) -> None:
        """Emit a one-shot ToolResultEvent for a provider-executed result.

        The item is built with the same reader the non-streaming path uses, so
        the payload shape matches; the provider's whole item is kept on the
        event so a same-dialect writer can replay it verbatim.
        """
        part = OpenResponsesToolOps.p_tool_result_to_ir(item)
        result_event = ToolResultEvent(
            type="tool_result",
            tool_call_id=part["tool_call_id"],
            result=part["result"],
        )
        if part.get("tool_type") is not None:
            result_event["tool_type"] = part["tool_type"]
        # Carry the canonical kind on the event's first-class field, the same
        # carrier ``p_tool_result_to_ir`` puts it on.
        if "intrinsic_kind" in part:
            result_event["intrinsic_kind"] = part["intrinsic_kind"]
        if "is_error" in part:
            result_event["is_error"] = part["is_error"]
        pm = dict(part.get("provider_metadata") or {})
        pm["_native"] = {"base": self._CONVERTER_TAG, "item": dict(item)}
        result_event["provider_metadata"] = pm
        output_index = chunk.get("output_index")
        if output_index is not None:
            result_event["block_index"] = output_index
        events.append(result_event)

    def _handle_p_content_part_added_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        if context is not None:
            part = chunk.get("part", {})
            part_type = part.get("type", "") if isinstance(part, dict) else ""
            block_type = "text"
            if part_type == "output_text":
                block_type = "text"
            elif part_type == "summary_text":
                block_type = "thinking"
            elif part_type == "refusal":
                block_type = "refusal"
            block_index = context.next_block_index()
            event = ContentBlockStartEvent(
                type="content_block_start",
                block_index=block_index,
                block_type=block_type,
            )
            # A text or refusal block belongs to the message item that
            # announced it: carrying that item's id and phase keeps a turn's
            # several message items separable on the write leg (#997).
            if (
                block_type in ("text", "refusal")
                and isinstance(context, OpenResponsesStreamContext)
                and context._inbound_message_item_id
            ):
                metadata: dict[str, Any] = {
                    "responses_item_id": context._inbound_message_item_id,
                }
                if context._inbound_message_phase:
                    metadata["responses_phase"] = context._inbound_message_phase
                event["provider_metadata"] = metadata
            events.append(event)

    def _handle_p_content_part_done_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        if context is not None:
            events.append(
                ContentBlockEndEvent(
                    type="content_block_end",
                    block_index=context.current_block_index,
                )
            )

    def _handle_p_output_item_done_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        # For function_call items, store the completed item in context
        # so it can be included in the response.completed output array.
        # Message-level done is a no-op (content_part.done handles it).
        item = chunk.get("item", {})
        if not isinstance(item, dict):
            return
        item_type = item.get("type", "")
        if item_type == "function_call":
            if context is not None:
                call_id = item.get("call_id", "")
                if call_id:
                    context.set_tool_call_args(call_id, item.get("arguments", ""))
        elif item_type == "custom_tool_call":
            if context is not None:
                call_id = item.get("call_id", "")
                if call_id:
                    context.set_tool_call_args(call_id, item.get("input", ""))
        elif item_type in _SERVER_ITEM_INTRINSIC_KINDS:
            # The full server item only arrives on `done` (its `action` / `code`
            # is absent at `added` time).  Move the payload into the IR as a
            # single residual delta so the write leg rebuilds the same item.
            call_id = item.get("call_id") or item.get("id", "")
            if context is not None and call_id and not context.get_tool_name(call_id):
                context.register_tool_call(
                    call_id, _ITEM_TO_INTRINSIC_KIND[item_type], "intrinsic"
                )
            tool_input = _server_item_tool_input(item, item_type)
            self._emit_residual_tool_call_delta(
                json.dumps(tool_input), call_id, chunk, context, events
            )
        elif item_type == "reasoning" and isinstance(
            context, OpenResponsesStreamContext
        ):
            item_id = item.get("id", "")
            if item_id:
                context._inbound_reasoning_item_id = item_id
            encrypted_content = item.get("encrypted_content")
            if encrypted_content:
                # Zero-text delta carries encrypted_content through IR;
                # suppressed on the outbound side when reasoning is empty.
                event = ReasoningDeltaEvent(
                    type="reasoning_delta",
                    reasoning="",
                    encrypted_content=str(encrypted_content),
                )
                if item_id:
                    event["provider_metadata"] = {
                        "responses_reasoning_id": item_id,
                    }
                set_opaque_origin(event, redacted_data="open_responses")
                events.append(event)
        elif self._is_opaque_item(item):
            events.append(self._opaque_item_event(chunk, item))

    def _handle_p_function_call_args_delta_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        delta_text = chunk.get("delta", "")
        call_id = resolve_call_id(chunk, context)
        delta_event = ToolCallDeltaEvent(
            type="tool_call_delta",
            tool_call_id=call_id,
            arguments_delta=delta_text,
        )
        output_index = chunk.get("output_index")
        if output_index is not None:
            delta_event["tool_call_index"] = output_index

        # Accumulate arguments in context
        if context is not None and call_id:
            context.append_tool_call_args(call_id, delta_text)

        events.append(delta_event)

    def _emit_residual_tool_call_delta(
        self,
        final_text: str,
        call_id: str,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Emit a ToolCallDeltaEvent for any portion of *final_text* not yet
        delivered by preceding ``.delta`` events, then store *final_text* as
        the authoritative value in *context*.

        Some upstreams deliver the complete payload only on the ``.done``
        event with no preceding ``.delta`` — ARGO's native /v1/responses does
        this for cached turns (#800).  The delta handler is the only path that
        moves arguments into the IR stream, so without this guard the payload
        is silently dropped.  See #800 and #803.
        """
        if context is not None and call_id:
            accumulated = context._tool_call_args.get(call_id, "")
        else:
            accumulated = final_text
        if final_text and final_text != accumulated:
            residual = (
                final_text[len(accumulated) :]
                if final_text.startswith(accumulated)
                else final_text
            )
            if residual:
                delta_event = ToolCallDeltaEvent(
                    type="tool_call_delta",
                    tool_call_id=call_id,
                    arguments_delta=residual,
                )
                output_index = chunk.get("output_index")
                if output_index is not None:
                    delta_event["tool_call_index"] = output_index
                events.append(delta_event)

        if context is not None and call_id:
            context.set_tool_call_args(call_id, final_text)

    def _handle_p_function_call_args_done_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        call_id = resolve_call_id(chunk, context)
        self._emit_residual_tool_call_delta(
            chunk.get("arguments", ""), call_id, chunk, context, events
        )

    def _handle_p_custom_tool_call_input_delta_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Handle custom_tool_call_input.delta — same as function_call args delta."""
        delta_text = chunk.get("delta", "")
        call_id = resolve_call_id(chunk, context)
        delta_event = ToolCallDeltaEvent(
            type="tool_call_delta",
            tool_call_id=call_id,
            arguments_delta=delta_text,
        )
        output_index = chunk.get("output_index")
        if output_index is not None:
            delta_event["tool_call_index"] = output_index

        # Accumulate input in context
        if context is not None and call_id:
            context.append_tool_call_args(call_id, delta_text)

        events.append(delta_event)

    def _handle_p_custom_tool_call_input_done_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Handle custom_tool_call_input.done — store final input text."""
        call_id = resolve_call_id(chunk, context)
        self._emit_residual_tool_call_delta(
            chunk.get("input", ""), call_id, chunk, context, events
        )

    def _handle_p_response_completed_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        response = chunk.get("response", chunk)

        # response.incomplete and response.completed share this handler; the
        # event type is part of the decision, not only the body (#1612).
        reason = _stream_finish_reason(response, chunk.get("type"))

        # Emit UsageEvent before FinishEvent so that downstream
        # converters can store usage in context.pending_usage before
        # FinishEvent builds the terminal response.completed event.
        usage = response.get("usage")
        if isinstance(usage, dict):
            events.append(
                UsageEvent(
                    type="usage",
                    usage=self._build_p_usage_to_ir(usage),
                )
            )

        finish = FinishEvent(
            type="finish",
            finish_reason={"reason": reason},  # ty: ignore[invalid-argument-type]
        )
        if reason in {"error", "cancelled"}:
            metadata = _response_failure_metadata(response, context)
            if metadata:
                finish["provider_metadata"] = metadata
        events.append(finish)

        # Emit StreamEndEvent after other events
        if context is not None:
            context.mark_ended()
            events.append(StreamEndEvent(type="stream_end"))

    def _handle_p_response_failed_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        finish: FinishEvent = FinishEvent(
            type="finish",
            finish_reason={"reason": "error"},
        )
        metadata = _response_failure_metadata(chunk.get("response", chunk), context)
        if metadata:
            finish["provider_metadata"] = metadata
        events.append(finish)

        # Emit StreamEndEvent after FinishEvent
        if context is not None:
            context.mark_ended()
            events.append(StreamEndEvent(type="stream_end"))

    _P_TO_IR_DISPATCH: dict[str, str] = {
        ResponsesEventType.RESPONSE_CREATED: "_handle_p_response_created_to_ir",
        ResponsesEventType.OUTPUT_TEXT_DELTA: "_handle_p_output_text_delta_to_ir",
        ResponsesEventType.OUTPUT_TEXT_ANNOTATION_ADDED: (
            "_handle_p_output_text_annotation_added_to_ir"
        ),
        ResponsesEventType.REASONING_SUMMARY_TEXT_DELTA: "_handle_p_reasoning_delta_to_ir",
        ResponsesEventType.OUTPUT_ITEM_ADDED: "_handle_p_output_item_added_to_ir",
        ResponsesEventType.CONTENT_PART_ADDED: "_handle_p_content_part_added_to_ir",
        ResponsesEventType.CONTENT_PART_DONE: "_handle_p_content_part_done_to_ir",
        ResponsesEventType.OUTPUT_ITEM_DONE: "_handle_p_output_item_done_to_ir",
        ResponsesEventType.FUNCTION_CALL_ARGS_DELTA: "_handle_p_function_call_args_delta_to_ir",
        ResponsesEventType.FUNCTION_CALL_ARGS_DONE: "_handle_p_function_call_args_done_to_ir",
        ResponsesEventType.CUSTOM_TOOL_CALL_INPUT_DELTA: "_handle_p_custom_tool_call_input_delta_to_ir",
        ResponsesEventType.CUSTOM_TOOL_CALL_INPUT_DONE: "_handle_p_custom_tool_call_input_done_to_ir",
        ResponsesEventType.RESPONSE_COMPLETED: "_handle_p_response_completed_to_ir",
        # An incomplete turn may close with either terminal event, and both
        # carry the same status/incomplete_details payload, so one handler
        # serves them.
        ResponsesEventType.RESPONSE_INCOMPLETE: "_handle_p_response_completed_to_ir",
        ResponsesEventType.RESPONSE_FAILED: "_handle_p_response_failed_to_ir",
        ResponsesEventType.REFUSAL_DELTA: "_handle_p_refusal_delta_to_ir",
    }

    _IR_TO_P_DISPATCH: dict[str, str] = {
        **BaseConverter._IR_TO_P_DISPATCH,
        "refusal_delta": "_handle_ir_refusal_delta_to_p",
        "tool_result": "_handle_ir_tool_result_to_p",
    }
