"""IR-to-provider streaming (content, reasoning, tools) for Open Responses."""

import time
from typing import Any, cast

from llm_rosetta.ir.model.stream import (
    CitationDeltaEvent,
    ContentBlockEndEvent,
    ContentBlockStartEvent,
    IRStreamEvent,
    ReasoningDeltaEvent,
    RefusalDeltaEvent,
    StreamEndEvent,
    StreamStartEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallStartEvent,
    ToolResultEvent,
)
from ...base.context import StreamContext
from ...base.helpers.citations import report_dropped_citation
from ...base.helpers.opaque import (
    get_opaque_origin,
    is_own_origin,
)
from llm_rosetta.ir.primitives.tool_group import (
    tool_call_group,
)
from ...base.tools import (
    intrinsic_kind_of,
    sanitize_tool_call_id,
)
from .stream_context import MessageItemState, OpenResponsesStreamContext
from ._mixin_host import OpenResponsesMixinHost
from ._constants import (
    RESPONSES_FAILURE_ERROR,
    RESPONSES_TERMINAL_EVENT_BY_STATUS,
    ResponsesEventType,
    generate_reasoning_id,
)
from .tool_ops import (
    _INTRINSIC_KIND_TO_ITEM,
)
from .utils import build_message_preamble_events


class OpenResponsesStreamToMixin(OpenResponsesMixinHost):
    """Handlers that turn IR stream events into Responses SSE events."""

    def stream_response_to_provider(
        self,
        event: IRStreamEvent,
        context: StreamContext | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert IR stream event with automatic context upgrade.

        If a base ``StreamContext`` is passed, it is automatically upgraded
        to ``OpenResponsesStreamContext`` (preserving existing state) so
        that callers do not need to know about the provider-specific subclass.
        """
        # Auto-upgrade base StreamContext to the provider-specific subclass.
        # Cache the upgraded instance in metadata so state persists across calls.
        if context is not None and not isinstance(context, OpenResponsesStreamContext):
            cached = context.metadata.get("_responses_stream_ctx")
            if cached is None:
                cached = OpenResponsesStreamContext.from_base(context)
                context.metadata["_responses_stream_ctx"] = cached
            context = cached

        return super().stream_response_to_provider(event, context)

    def _post_process_ir_to_p(
        self,
        result: dict[str, Any] | list[dict[str, Any]],
        event: IRStreamEvent,
        context: StreamContext | None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Inject sequence_number into emitted Responses events."""
        if isinstance(context, OpenResponsesStreamContext):
            if isinstance(result, list):
                for r in result:
                    if isinstance(r, dict) and "type" in r:
                        context._sequence_number += 1
                        r["sequence_number"] = context._sequence_number
            elif isinstance(result, dict) and "type" in result:
                context._sequence_number += 1
                result["sequence_number"] = context._sequence_number
        return result

    # --- to_provider handlers ---

    def _handle_ir_stream_start_to_p(
        self,
        event: StreamStartEvent,
        context: StreamContext | None,
    ) -> list[dict[str, Any]]:
        response_id_stem = event["response_id"]

        # Store metadata in context if provided (stem, without prefix)
        if context is not None:
            context.response_id = response_id_stem
            context.model = event["model"]
            context.created = event.get("created", 0)
            context.mark_started()

        prefixed_id = self.add_response_id_prefix(
            response_id_stem, self._get_response_id_prefix(context)
        )
        response: dict[str, Any] = {
            "id": prefixed_id,
            "object": "response",
            "model": event["model"],
            "status": "in_progress",
            "output": [],
        }
        created = event.get("created", 0)
        response["created_at"] = created or int(time.time())

        # Preserve mode: include echo fields in response.created
        if context is not None and context.metadata_mode == "preserve":
            echo = context.get_echo_fields()
            core_keys = {
                "id",
                "object",
                "created_at",
                "model",
                "output",
                "status",
                "usage",
            }
            for k, v in self._get_required_defaults(context).items():
                if k not in core_keys and k not in response:
                    response[k] = v
            for k, v in echo.items():
                if k not in core_keys:
                    response[k] = v
            # response.created must include usage: null (not yet available)
            response.setdefault("usage", None)

        # Both events share the same response snapshot — safe because
        # nothing between here and json.dumps serialisation mutates it.
        return [
            {
                "type": ResponsesEventType.RESPONSE_CREATED,
                "response": response,
            },
            {
                "type": ResponsesEventType.RESPONSE_IN_PROGRESS,
                "response": response,
            },
        ]

    def _handle_ir_stream_end_to_p(
        self,
        event: StreamEndEvent,
        context: StreamContext | None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        if context is not None:
            context.mark_ended()
            # Emit the deferred terminal event if FinishEvent
            # stored one.  This ensures UsageEvents that arrive
            # between FinishEvent and StreamEndEvent (e.g. OpenAI
            # Chat sends usage in a separate chunk after
            # finish_reason) are merged into the response.
            if context.pending_response is not None:
                resp = context.pending_response
                context.pending_response = None
                # Merge any usage that arrived after FinishEvent
                if context.pending_usage is not None and "usage" not in resp:
                    resp["usage"] = self._build_finish_usage(context.pending_usage)
                terminal = self._terminal_events(resp)
                # A completed turn is a single event, and this path has always
                # answered with one dict; only a failure grows to two.
                return terminal[0] if len(terminal) == 1 else terminal
        return {}

    @staticmethod
    def _terminal_events(response: dict[str, Any]) -> list[dict[str, Any]]:
        """Build the events that close a stream, from the response it closes.

        The spec allows three terminal events and the choice is not free: it
        has to agree with the ``status`` the payload carries.  A failed
        response is preceded by an ``error`` event carrying the same envelope,
        so a client that only reads the error event still learns the code.
        """
        event_type = RESPONSES_TERMINAL_EVENT_BY_STATUS.get(
            response.get("status", ""), ResponsesEventType.RESPONSE_COMPLETED
        )
        events: list[dict[str, Any]] = []
        if event_type == ResponsesEventType.RESPONSE_FAILED:
            error = response.get("error")
            events.append(
                {
                    "type": ResponsesEventType.ERROR,
                    "error": dict(
                        error if isinstance(error, dict) else RESPONSES_FAILURE_ERROR
                    ),
                }
            )
        events.append({"type": event_type, "response": response})
        return events

    def _handle_ir_content_block_start_to_p(
        self,
        event: ContentBlockStartEvent,
        context: OpenResponsesStreamContext | None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        block_type = event["block_type"]
        if context is not None:
            context.current_block_type = block_type
        if block_type in ("text", "refusal"):
            content_part_type = "refusal" if block_type == "refusal" else "output_text"
            from .utils import _build_content_part

            if not isinstance(context, OpenResponsesStreamContext):
                # No context, so no item tracking: the block is reported as a
                # part on its own (backward compatible).
                return {
                    "type": ResponsesEventType.CONTENT_PART_ADDED,
                    "item_id": "",
                    "output_index": 0,
                    "content_index": 0,
                    "part": _build_content_part(content_part_type),
                }

            phase, source_item_id = self._block_message_item(event)
            current = context.current_message_item
            results: list[dict[str, Any]] = []
            if current is None or not self._continues_message_item(
                current, phase, source_item_id
            ):
                # A new upstream message item: close the one being written
                # before opening its successor, or the client reads both as
                # the same item and renders the reply twice (#997).
                if current is not None:
                    results.extend(self._close_message_item_events(current))
                results.extend(
                    build_message_preamble_events(
                        context,
                        content_part_type,
                        phase=phase,
                        source_item_id=source_item_id,
                    )
                )
                return results
            # Another content part of the item already open (Responses
            # carries several per item, e.g. output text then a refusal).
            return {
                "type": ResponsesEventType.CONTENT_PART_ADDED,
                "item_id": current.item_id,
                "output_index": current.output_index,
                "content_index": 0,
                "part": _build_content_part(content_part_type),
            }
        # Other block types are no-ops for now
        return {}

    @staticmethod
    def _block_message_item(event: ContentBlockStartEvent) -> tuple[str, str]:
        """Read the upstream message item a text block belongs to.

        The read leg tags a text block with the message item that announced
        it, which is how the write leg tells a continuation from a new item.
        Both fields are absent for a dialect with no such notion, and then
        every block belongs to the item already open.

        Args:
            event: IR content block start event.

        Returns:
            ``(phase, source_item_id)``; either may be ``""``.
        """
        metadata = event.get("provider_metadata") or {}
        phase = metadata.get("responses_phase", "")
        source_item_id = metadata.get("responses_item_id", "")
        return (
            phase if isinstance(phase, str) else "",
            source_item_id if isinstance(source_item_id, str) else "",
        )

    @staticmethod
    def _continues_message_item(
        item: MessageItemState, phase: str, source_item_id: str
    ) -> bool:
        """Whether a text block continues the message item being written.

        An upstream item id decides it when both sides have one; the phase is
        the fallback for a stream that reported no ids, since two items that
        report different phases are different items however they are keyed.
        """
        if source_item_id and item.source_item_id:
            return source_item_id == item.source_item_id
        return phase == item.phase

    def _handle_ir_content_block_end_to_p(
        self,
        event: ContentBlockEndEvent,
        context: OpenResponsesStreamContext | None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        if not isinstance(context, OpenResponsesStreamContext):
            return {
                "type": ResponsesEventType.CONTENT_PART_DONE,
                "part": {
                    "type": "output_text",
                },
            }
        item = context.current_message_item
        if item is None:
            return {}
        item.content_part_type = (
            "refusal" if context.current_block_type == "refusal" else "output_text"
        )
        return self._content_part_done_events(item)

    @staticmethod
    def _content_part_done_events(item: MessageItemState) -> list[dict[str, Any]]:
        """The events that close one message item's content part.

        Responses ends a part with a type-specific ``*.done`` and a
        ``content_part.done`` carrying the whole part.

        Args:
            item: Message item state whose part is ending.

        Returns:
            Two SSE event dicts.
        """
        item.content_part_done_emitted = True
        is_refusal = item.content_part_type == "refusal"
        accumulated = item.accumulated_refusal if is_refusal else item.accumulated_text
        if is_refusal:
            return [
                {
                    "type": ResponsesEventType.REFUSAL_DONE,
                    "item_id": item.item_id,
                    "output_index": item.output_index,
                    "content_index": 0,
                    "refusal": accumulated,
                },
                {
                    "type": ResponsesEventType.CONTENT_PART_DONE,
                    "item_id": item.item_id,
                    "output_index": item.output_index,
                    "content_index": 0,
                    "part": {
                        "type": "refusal",
                        "refusal": accumulated,
                    },
                },
            ]
        return [
            {
                "type": ResponsesEventType.OUTPUT_TEXT_DONE,
                "item_id": item.item_id,
                "output_index": item.output_index,
                "content_index": 0,
                "text": accumulated,
                "logprobs": [],
            },
            {
                "type": ResponsesEventType.CONTENT_PART_DONE,
                "item_id": item.item_id,
                "output_index": item.output_index,
                "content_index": 0,
                "part": {
                    "type": "output_text",
                    "text": accumulated,
                    "annotations": [],
                    "logprobs": [],
                },
            },
        ]

    def _handle_ir_citation_delta_to_p(
        self, event: CitationDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Handle CitationDeltaEvent → response.output_text.annotation.added."""
        annotation = self.content_ops.ir_citation_to_p(event["citation"])
        if annotation is None:
            report_dropped_citation(
                event["citation"], warnings=self._citation_warnings(context, {})
            )
            return {}
        stream_context = (
            context if isinstance(context, OpenResponsesStreamContext) else None
        )
        # The annotation belongs to the message item being written, which is
        # where the text it annotates went.
        item = stream_context.current_message_item if stream_context else None
        return {
            "type": ResponsesEventType.OUTPUT_TEXT_ANNOTATION_ADDED,
            "item_id": item.item_id if item is not None else "",
            "output_index": item.output_index if item is not None else 0,
            "content_index": 0,
            "annotation": annotation,
        }

    def _handle_ir_text_delta_to_p(
        self,
        event: TextDeltaEvent,
        context: OpenResponsesStreamContext | None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        text = event["text"]

        # Emit output_item.added + content_part.added before the first
        # text delta so clients (e.g. Codex CLI) can register the item.
        preamble: list[dict[str, Any]] = []
        if context is not None and context.current_message_item is None:
            preamble = build_message_preamble_events(context)

        item = context.current_message_item if context is not None else None

        # Accumulate text on the item for its done events and for
        # response.completed's output array
        if item is not None:
            item.accumulated_text += text

        delta_event: dict[str, Any] = {
            "type": ResponsesEventType.OUTPUT_TEXT_DELTA,
            "item_id": item.item_id if item is not None else "",
            "output_index": item.output_index if item is not None else 0,
            "content_index": 0,
            "delta": text,
            "logprobs": [],
        }

        if preamble:
            return preamble + [delta_event]
        return delta_event

    def _handle_ir_refusal_delta_to_p(
        self,
        event: RefusalDeltaEvent,
        context: StreamContext | None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        refusal = event["refusal"]
        ctx = context if isinstance(context, OpenResponsesStreamContext) else None

        preamble: list[dict[str, Any]] = []
        if ctx is not None and ctx.current_message_item is None:
            preamble = build_message_preamble_events(ctx, "refusal")

        item = ctx.current_message_item if ctx is not None else None
        if item is not None:
            item.accumulated_refusal += refusal
        if ctx is not None and refusal:
            # The part carries words now, so a finish repeating them in
            # ``refusal_text`` must not open a second one.  Recorded on the
            # words rather than on the block: a refusal block that opened and
            # stayed empty writes nothing, and suppressing the finish's
            # synthesis for it dropped the words entirely.
            ctx._refusal_part_written = True

        delta_event: dict[str, Any] = {
            "type": ResponsesEventType.REFUSAL_DELTA,
            "item_id": item.item_id if item is not None else "",
            "output_index": item.output_index if item is not None else 0,
            "content_index": 0,
            "delta": refusal,
        }

        if preamble:
            return preamble + [delta_event]
        return delta_event

    def _handle_ir_reasoning_delta_to_p(
        self,
        event: ReasoningDeltaEvent,
        context: StreamContext | None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        ctx = context if isinstance(context, OpenResponsesStreamContext) else None
        provider_metadata = event.get("provider_metadata") or {}
        source_item_id = provider_metadata.get("responses_reasoning_id", "")
        if not isinstance(source_item_id, str):
            source_item_id = ""

        item = None
        if ctx is not None:
            # Deltas with no provenance all belong to one item; a new
            # source id starts a new one, so each item keeps the
            # encrypted_content that was minted against its own id.
            item = (
                ctx.reasoning_item(source_item_id)
                if source_item_id
                else ctx.current_reasoning_item
            )

        if ctx is not None and item is None:
            output_index = ctx.next_output_index()
            item = ctx.start_reasoning_item(
                source_item_id or generate_reasoning_id(ctx.response_id, output_index),
                output_index,
            )

            results.append(
                {
                    "type": ResponsesEventType.OUTPUT_ITEM_ADDED,
                    "output_index": item.output_index,
                    "item": {
                        "id": item.item_id,
                        "type": "reasoning",
                        "summary": [],
                        "status": "in_progress",
                    },
                }
            )
            results.append(
                {
                    "type": ResponsesEventType.REASONING_SUMMARY_PART_ADDED,
                    "item_id": item.item_id,
                    "output_index": item.output_index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": ""},
                }
            )

        enc = event.get("encrypted_content", "")
        if enc and not is_own_origin(
            get_opaque_origin(event, "redacted_data"), "open_responses"
        ):
            # Foreign opaque blob — not valid for the Responses API.
            enc = ""
        if item is not None and source_item_id and enc:
            item.encrypted_content = enc
            if not event["reasoning"]:
                return results

        reasoning_text = event["reasoning"]
        if item is not None:
            item.accumulated_text += reasoning_text

        item_id = item.item_id if item is not None else ""
        r_idx = item.output_index if item is not None else 0
        results.append(
            {
                "type": ResponsesEventType.REASONING_SUMMARY_TEXT_DELTA,
                "item_id": item_id,
                "output_index": r_idx,
                "summary_index": 0,
                "delta": reasoning_text,
            }
        )
        return results

    def _handle_ir_tool_call_start_to_p(
        self,
        event: ToolCallStartEvent,
        context: StreamContext | None,
    ) -> dict[str, Any]:
        call_id = sanitize_tool_call_id(event["tool_call_id"])
        tool_name = event["tool_name"]
        tool_type = event.get("tool_type", "function")
        pm = event.get("provider_metadata") or {}
        item_id = pm.get("responses_item_id") or call_id

        # A provider-executed call re-emits the provider's own server item
        # type, never a client ``function_call``.  The kind comes from the
        # event's ``intrinsic_kind`` carrier, never from the tool name; a kind
        # this dialect cannot render is dropped rather than downgraded.
        intrinsic_item: dict[str, Any] | None = None
        intrinsic_kind = intrinsic_kind_of(event)
        if tool_type == "intrinsic":
            intrinsic_item = self.tool_ops.stream_intrinsic_item(
                intrinsic_kind, item_id, status="in_progress"
            )
            if intrinsic_item is None:
                # A kind this dialect cannot render as a server item: nothing
                # opens the item, so nothing may follow it either.  Record the
                # drop so the argument deltas — and the completed item the
                # finish flush would otherwise build from them — are dropped
                # with it instead of leaking a nameless ``function_call``.
                if context is not None and call_id:
                    context.drop_intrinsic_call(call_id)
                return {}

        # Register in context for later done events, carrying the intrinsic
        # kind through so the done/completed items restore it without
        # re-deriving it from the name.
        if context is not None and call_id:
            context.register_tool_call(
                call_id,
                tool_name,
                tool_type,
                provider_metadata=pm or None,
                intrinsic_kind=intrinsic_kind,
            )
            context.register_tool_call_item(call_id, item_id)

        stream_ctx = (
            context if isinstance(context, OpenResponsesStreamContext) else None
        )

        if stream_ctx is not None:
            output_index = stream_ctx.next_output_index()
            stream_ctx._tool_call_output_indices[call_id] = output_index
            # Record the group once, so the later done/completed items can be
            # rebuilt with it straight from the context.
            namespace = tool_call_group(event)
            if namespace and call_id:
                stream_ctx._tool_call_groups[call_id] = namespace
        else:
            tc_index = event.get("tool_call_index")
            output_index = tc_index if tc_index is not None else 0

        item: dict[str, Any]
        if tool_type == "intrinsic":
            assert intrinsic_item is not None
            item = intrinsic_item
        elif tool_type == "custom":
            item = self._build_stream_custom_tool_call_item(
                stream_ctx,
                item_id=item_id,
                call_id=call_id,
                tool_name=tool_name,
                input_str="",
                status="in_progress",
            )
        else:
            item = self._build_stream_function_call_item(
                stream_ctx,
                item_id=item_id,
                call_id=call_id,
                tool_name=tool_name,
                arguments="",
                status="in_progress",
            )

        result: dict[str, Any] = {
            "type": ResponsesEventType.OUTPUT_ITEM_ADDED,
            "output_index": output_index,
            "item": item,
        }
        return result

    def _handle_ir_tool_call_delta_to_p(
        self,
        event: ToolCallDeltaEvent,
        context: StreamContext | None,
    ) -> dict[str, Any]:
        call_id = sanitize_tool_call_id(event["tool_call_id"])
        delta = event["arguments_delta"]
        tc_index = event.get("tool_call_index")

        if (
            context is not None
            and call_id
            and context.is_intrinsic_call_dropped(call_id)
        ):
            # The call's ``output_item.added`` was dropped, so its arguments
            # belong to no item: they are not accumulated either, which is what
            # keeps the finish flush from rebuilding the call without a name.
            return {}

        # Defense-in-depth: resolve empty tool_call_id by index.
        # Some upstream providers (e.g. certain Chat Completions
        # implementations) only send tool_call_id on the first chunk.
        if not call_id and context is not None and tc_index is not None:
            if tc_index < context.tool_call_count:
                call_id = context.resolve_tool_call_id_by_index(tc_index)

        # Accumulate arguments in context for done events
        if context is not None and call_id:
            context.append_tool_call_args(call_id, delta)

        # Use item_id per Responses API spec
        item_id = ""
        if context is not None and call_id:
            item_id = context.get_tool_call_item_id(call_id)
        if not item_id and call_id:
            item_id = call_id

        if isinstance(context, OpenResponsesStreamContext) and call_id:
            output_index = context._tool_call_output_indices.get(call_id, 0)
        else:
            output_index = tc_index if tc_index is not None else 0

        # Determine event type based on tool type (custom vs function)
        tool_type = (
            context.get_tool_type(call_id)
            if context is not None and call_id
            else "function"
        )
        if tool_type == "intrinsic":
            # A server item carries its payload whole (``action`` / ``code``),
            # not as an arguments delta the way a ``function_call`` does, so
            # the accumulated bytes are kept for the done/completed item and
            # no provider delta event is emitted.
            return {}
        event_type = (
            ResponsesEventType.CUSTOM_TOOL_CALL_INPUT_DELTA
            if tool_type == "custom"
            else ResponsesEventType.FUNCTION_CALL_ARGS_DELTA
        )

        result: dict[str, Any] = {
            "type": event_type,
            "item_id": item_id,
            "output_index": output_index,
            "delta": delta,
        }
        return result

    def _handle_ir_tool_result_to_p(
        self,
        event: ToolResultEvent,
        context: StreamContext | None,
    ) -> dict[str, Any]:
        """Handle ToolResultEvent → a provider server result item.

        A result read from this same dialect is replayed verbatim from the
        ``_native`` item the reader stored.  Otherwise the item is rebuilt from
        the IR fields for a kind that has its own ``*_call_output`` item; a
        ``web_search`` result rides on its call item and needs no item of its
        own, and a server kind this dialect cannot render at all is reported
        through the warning sink rather than dropped without a word.  A server
        result is never re-encoded as a client ``function_call_output``.
        """
        pm = event.get("provider_metadata") or {}
        native = pm.get("_native")
        if isinstance(native, dict) and native.get("base") == self._CONVERTER_TAG:
            native_item = native.get("item")
            if isinstance(native_item, dict):
                return self._stream_result_chunk(dict(native_item), event, context)

        item = self.tool_ops.stream_result_item(
            intrinsic_kind_of(event),
            sanitize_tool_call_id(event["tool_call_id"]),
            event.get("result", ""),
        )
        if item is None:
            if (
                event.get("tool_type") == "intrinsic"
                and intrinsic_kind_of(event) not in _INTRINSIC_KIND_TO_ITEM
            ):
                # A server kind with no item at all is a loss, and the body
                # leg reports it; a stream that dropped it silently would make
                # the two legs answer a caller differently.  A client result is
                # not this writer's to render — a body writes it, a stream
                # never has.
                self.tool_ops.dropped_intrinsic_result(
                    cast(Any, event),
                    leg="response",
                    # An empty list is not a sink: naming one would swallow
                    # the report instead of letting the conversion's own
                    # sink hear it.
                    warnings=getattr(context, "warnings", None) or None,
                )
            return {}
        return self._stream_result_chunk(item, event, context)

    @staticmethod
    def _stream_result_chunk(
        item: dict[str, Any],
        event: ToolResultEvent,
        context: StreamContext | None,
    ) -> dict[str, Any]:
        """Wrap a server result item in an ``output_item.added`` chunk."""
        result: dict[str, Any] = {
            "type": ResponsesEventType.OUTPUT_ITEM_ADDED,
            "item": item,
        }
        output_index = event.get("block_index")
        if isinstance(context, OpenResponsesStreamContext):
            if output_index is None:
                output_index = context.next_output_index()
            result["output_index"] = output_index
            # The item has no deltas to follow it, so the completed response
            # learns about it here or not at all.
            context._result_items.append((output_index, item))
        elif output_index is not None:
            result["output_index"] = output_index
        return result
