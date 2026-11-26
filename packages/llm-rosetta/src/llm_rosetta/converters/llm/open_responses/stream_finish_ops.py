"""IR-to-provider streaming finish assembly for Open Responses."""

import json
import time
from typing import Any, cast

from llm_rosetta.ir.model.stream import (
    FinishEvent,
    UsageEvent,
)
from ...base.context import StreamContext
from .stream_context import MessageItemState, OpenResponsesStreamContext
from ._mixin_host import OpenResponsesMixinHost
from ._constants import (
    RESPONSES_FAILURE_ERROR,
    RESPONSES_REASON_TO_INCOMPLETE_REASON,
    RESPONSES_REASON_TO_STATUS,
    ResponsesEventType,
)


class OpenResponsesStreamFinishMixin(OpenResponsesMixinHost):
    """Finish, tool-done and usage handlers that close a Responses stream."""

    def _handle_ir_finish_to_p(
        self,
        event: FinishEvent,
        context: OpenResponsesStreamContext | None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        reason = event["finish_reason"]["reason"]
        status = RESPONSES_REASON_TO_STATUS.get(reason, "completed")

        # A failed turn carries the error envelope it was given, so the
        # failure leaves with a code and message rather than a placeholder.
        provider_metadata = event.get("provider_metadata") or {}
        carried_error = provider_metadata.get("responses_error")

        # A refusal whose words arrived only on the finish event has no content
        # part yet: open one and write them, so a reader sees the words as
        # content instead of a status with nothing behind it.  The spec carries
        # a refusal as an output part, and every dialect that reads it back
        # reads it there (#1314).  When the turn already wrote the part — a
        # folded response carries one — the words are already on the wire and
        # opening a second part doubled them.
        opening: list[dict[str, Any]] = []
        refusal_text = provider_metadata.get("refusal_text")
        if (
            context is not None
            and reason == "refusal"
            and refusal_text
            and not context._refusal_part_written
        ):
            block_index = context.next_block_index()
            produced_events = [
                self._handle_ir_content_block_start_to_p(
                    cast(
                        Any,
                        {
                            "type": "content_block_start",
                            "block_index": block_index,
                            "block_type": "refusal",
                        },
                    ),
                    context,
                ),
                self._handle_ir_refusal_delta_to_p(
                    cast(Any, {"type": "refusal_delta", "refusal": refusal_text}),
                    context,
                ),
                self._handle_ir_content_block_end_to_p(
                    cast(
                        Any, {"type": "content_block_end", "block_index": block_index}
                    ),
                    context,
                ),
            ]
            for produced in produced_events:
                items = produced if isinstance(produced, list) else [produced]
                opening.extend(item for item in items if item)

        response = self._build_finish_response(
            status,
            context,
            reason,
            error=carried_error if isinstance(carried_error, dict) else None,
        )

        # Emit done events before response.completed
        results: list[dict[str, Any]] = list(opening)

        if context is not None:
            self._emit_reasoning_done_events(context, results)
            self._emit_text_done_events(context, results)
            self._emit_tool_call_done_events(context, results)

        # With context: defer response.completed to StreamEndEvent
        # so that any UsageEvent arriving after FinishEvent (e.g.
        # OpenAI Chat sends usage in a separate chunk) can still be
        # merged into the response.
        if context is not None:
            context.pending_response = response
            return results

        # Without context: emit immediately (backward compatible)
        results.extend(self._terminal_events(response))
        return results

    def _build_finish_response(
        self,
        status: str,
        context: OpenResponsesStreamContext | None,
        finish_reason: str = "length",
        error: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build the response dict for a FinishEvent."""
        output = self._collect_finish_output(context)
        response: dict[str, Any] = {"status": status, "output": output}

        # Populate id, object, model, created_at from context so clients can
        # parse the completed response envelope.
        if context is not None:
            response["id"] = self.add_response_id_prefix(
                context.response_id, self._get_response_id_prefix(context)
            )
            response["object"] = "response"
            response["model"] = context.model
            response["created_at"] = context.created or int(time.time())

        # Set completed_at timestamp for completed responses.
        if status == "completed":
            response["completed_at"] = self._completion_timestamp()
        else:
            response["completed_at"] = None

        if status == "incomplete":
            incomplete_reason = RESPONSES_REASON_TO_INCOMPLETE_REASON.get(
                finish_reason, "max_output_tokens"
            )
            response["incomplete_details"] = {"reason": incomplete_reason}

        if status == "failed":
            response["error"] = error or RESPONSES_FAILURE_ERROR

        # Merge pending usage from context if available
        if context is not None and context.pending_usage is not None:
            response["usage"] = self._build_finish_usage(context.pending_usage)

        # Preserve mode: inject echo fields into the response.completed payload
        if context is not None and context.metadata_mode == "preserve":
            self._apply_finish_echo(response, context)

        return response

    def _collect_finish_output(
        self, context: OpenResponsesStreamContext | None
    ) -> list[dict[str, Any]]:
        """Collect text and tool call output items from stream context."""
        output: list[dict[str, Any]] = []
        if context is None:
            return output

        # (output_index, item) so the array can be restored to emission
        # order below — a reasoning item has to stay immediately before
        # the tool call it produced, which front-loading would break
        # once a turn holds more than one.
        entries: list[tuple[int, dict[str, Any]]] = []

        for item in context._reasoning_items:
            reasoning_item: dict[str, Any] = {
                "id": item.item_id,
                "type": "reasoning",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": item.accumulated_text,
                    }
                ],
                "status": "completed",
            }
            if item.encrypted_content:
                reasoning_item["encrypted_content"] = item.encrypted_content
            entries.append((item.output_index, reasoning_item))

        for item in context._message_items:
            msg_item = self._finish_message_item(item, context)
            if msg_item is not None:
                entries.append((item.output_index, msg_item))

        for call_id in context.tool_call_ids:
            tool_name = context.get_tool_name(call_id)
            arguments = context._tool_call_args.get(call_id, "")
            tc_item_id = context.get_tool_call_item_id(call_id) or call_id
            tool_type = context.get_tool_type(call_id)
            tc_index = context._tool_call_output_indices.get(call_id, -1)

            if tool_type == "intrinsic":
                intrinsic_item = self._build_stream_intrinsic_item(
                    context,
                    call_id=call_id,
                    item_id=tc_item_id,
                    kind=self._stream_intrinsic_kind(context, call_id),
                    arguments=arguments,
                    status="completed",
                )
                if intrinsic_item is not None:
                    entries.append((tc_index, intrinsic_item))
            elif tool_type == "custom":
                entries.append(
                    (
                        tc_index,
                        self._build_stream_custom_tool_call_item(
                            context,
                            item_id=tc_item_id,
                            call_id=call_id,
                            tool_name=tool_name,
                            input_str=arguments,
                            status="completed",
                        ),
                    )
                )
            else:
                entries.append(
                    (
                        tc_index,
                        self._build_stream_function_call_item(
                            context,
                            item_id=tc_item_id,
                            call_id=call_id,
                            tool_name=tool_name,
                            arguments=arguments,
                            status="completed",
                        ),
                    )
                )

        # Server results are whole items, announced once with no delta to
        # follow; they carry their own index.
        entries.extend(context._result_items)

        # Only reorder when every item knows its index; otherwise keep the
        # historical reasoning → message → tool-calls order.  sort() is
        # stable, so ties stay as appended.
        if all(index >= 0 for index, _ in entries):
            entries.sort(key=lambda entry: entry[0])
        output.extend(item for _, item in entries)
        return output

    @staticmethod
    def _finish_message_item(
        item: MessageItemState, context: OpenResponsesStreamContext
    ) -> dict[str, Any] | None:
        """Build one message item for the completed response.

        A message item is the text it accumulated plus, when the turn carried
        one, its refusal: the item's own ``output_item.done`` announces both,
        so the completed payload must not be the one place a refusal
        disappears.

        Args:
            item: Message item state the stream accumulated.
            context: The stream context, for the preserve-mode shape.

        Returns:
            The item, or ``None`` when it holds no content at all.
        """
        if not item.accumulated_text and not item.accumulated_refusal:
            return None
        parts: list[dict[str, Any]] = []
        if item.accumulated_text:
            parts.append({"type": "output_text", "text": item.accumulated_text})
        if item.accumulated_refusal:
            parts.append({"type": "refusal", "refusal": item.accumulated_refusal})
        msg_item: dict[str, Any] = {
            "id": item.item_id,
            "type": "message",
            "role": "assistant",
            "content": parts,
        }
        if item.phase:
            msg_item["phase"] = item.phase
        if context.metadata_mode == "preserve":
            msg_item["status"] = "completed"
            for part in msg_item["content"]:
                part.setdefault("annotations", [])
                part.setdefault("logprobs", [])
        return msg_item

    @staticmethod
    def _build_stream_function_call_item(
        context: OpenResponsesStreamContext | None,
        *,
        item_id: str,
        call_id: str,
        tool_name: str,
        arguments: str,
        status: str,
    ) -> dict[str, Any]:
        """Build a streamed ``function_call`` item.

        Shared by the three points that emit one — ``output_item.added``,
        ``output_item.done``, and the ``response.completed`` output list —
        so they cannot drift apart.  Re-attaches the tool's namespace when
        it was flattened out of a namespace container on the request leg.
        """
        item: dict[str, Any] = {
            "id": item_id,
            "type": "function_call",
            "call_id": call_id,
            "name": tool_name,
            "arguments": arguments,
            "status": status,
        }
        if context is not None:
            namespace = context._tool_call_groups.get(call_id)
            if namespace:
                item["namespace"] = namespace
        return item

    @staticmethod
    def _build_stream_custom_tool_call_item(
        context: OpenResponsesStreamContext | None,
        *,
        item_id: str,
        call_id: str,
        tool_name: str,
        input_str: str,
        status: str,
    ) -> dict[str, Any]:
        """Build a streamed ``custom_tool_call`` item.

        The custom-tool counterpart of
        :meth:`_build_stream_function_call_item`, shared by the same three
        emit points and carrying the namespace for the same reason.
        """
        item: dict[str, Any] = {
            "id": item_id,
            "type": "custom_tool_call",
            "call_id": call_id,
            "name": tool_name,
            "input": input_str,
            "status": status,
        }
        if context is not None:
            namespace = context._tool_call_groups.get(call_id)
            if namespace:
                item["namespace"] = namespace
        return item

    def _build_stream_intrinsic_item(
        self,
        context: OpenResponsesStreamContext | None,
        *,
        call_id: str,
        item_id: str,
        kind: str,
        arguments: str,
        status: str,
    ) -> dict[str, Any] | None:
        """Rebuild a streamed server item from the accumulated payload.

        The counterpart of :meth:`_build_stream_function_call_item` for a
        provider-executed call: it restores the provider's own item type
        (``web_search_call`` / ``code_interpreter_call``, …) with the payload
        the reader moved through a ``ToolCallDeltaEvent``.  ``None`` when the
        kind cannot be rendered as a server item.
        """
        fields: dict[str, Any] = {}
        if arguments:
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                fields = parsed
        return self.tool_ops.stream_intrinsic_item(kind, item_id, fields, status=status)

    @staticmethod
    def _stream_intrinsic_kind(context: StreamContext | None, call_id: str) -> str:
        """Read the intrinsic kind the start event registered, if any.

        The kind is recorded on the context when the start event registers the
        call, so the done and completed items restore it without re-deriving
        it from the tool name.  ``""`` means the kind was never registered,
        which those writers treat as an unrenderable kind.
        """
        if context is None:
            return ""
        return context.get_tool_call_intrinsic_kind(call_id)

    @staticmethod
    def _build_finish_usage(pending_usage: dict[str, Any]) -> dict[str, Any]:
        """Build usage dict for the finish response from pending IR usage."""
        usage: dict[str, Any] = {
            "input_tokens": pending_usage.get("prompt_tokens") or 0,
            "output_tokens": pending_usage.get("completion_tokens") or 0,
            "total_tokens": pending_usage.get("total_tokens") or 0,
        }
        cache_read = pending_usage.get("cache_read_tokens")
        usage["input_tokens_details"] = {
            "cached_tokens": cache_read if cache_read is not None else 0
        }
        reasoning = pending_usage.get("reasoning_tokens")
        usage["output_tokens_details"] = {
            "reasoning_tokens": reasoning if reasoning is not None else 0
        }
        # The body and stream terminals are written here, so this is where the
        # field survives a non-stream round trip.
        if "compute_units" in pending_usage:
            usage["compute_units"] = pending_usage["compute_units"]
        return usage

    def _apply_finish_echo(
        self, response: dict[str, Any], context: OpenResponsesStreamContext
    ) -> None:
        """Inject preserve-mode echo fields into the finish response."""
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

    def _emit_reasoning_done_events(
        self,
        context: OpenResponsesStreamContext,
        results: list[dict[str, Any]],
    ) -> None:
        """Emit reasoning lifecycle done events, one set per item."""
        for item in context._reasoning_items:
            accumulated = item.accumulated_text
            results.append(
                {
                    "type": ResponsesEventType.REASONING_SUMMARY_TEXT_DONE,
                    "item_id": item.item_id,
                    "output_index": item.output_index,
                    "summary_index": 0,
                    "text": accumulated,
                }
            )
            results.append(
                {
                    "type": ResponsesEventType.REASONING_SUMMARY_PART_DONE,
                    "item_id": item.item_id,
                    "output_index": item.output_index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": accumulated},
                }
            )
            reasoning_item: dict[str, Any] = {
                "id": item.item_id,
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": accumulated}],
                "status": "completed",
            }
            if item.encrypted_content:
                reasoning_item["encrypted_content"] = item.encrypted_content
            results.append(
                {
                    "type": ResponsesEventType.OUTPUT_ITEM_DONE,
                    "output_index": item.output_index,
                    "item": reasoning_item,
                }
            )

    def _emit_text_done_events(
        self,
        context: OpenResponsesStreamContext,
        results: list[dict[str, Any]],
    ) -> None:
        """Emit done events for every message item not yet closed."""
        for item in context._message_items:
            if not item.output_item_done_emitted:
                results.extend(self._close_message_item_events(item))

    def _close_message_item_events(
        self, item: MessageItemState
    ) -> list[dict[str, Any]]:
        """Emit the events that end one message item.

        A turn's items close one at a time: the item that was being written
        when the next one opened is closed at that split, and whatever is
        still open is closed by the terminal flush.  Both go through here so
        an item cannot be closed twice or left open.

        Args:
            item: Message item state to close.

        Returns:
            The ``*.done`` and ``output_item.done`` events for the item.
        """
        results: list[dict[str, Any]] = []
        # Only emit output_text.done + content_part.done if not already
        # emitted by a prior ContentBlockEndEvent
        if not item.content_part_done_emitted:
            results.extend(self._content_part_done_events(item))
        item.output_item_done_emitted = True
        results.append(
            {
                "type": ResponsesEventType.OUTPUT_ITEM_DONE,
                "output_index": item.output_index,
                "item": self._message_item_done_payload(item),
            }
        )
        return results

    @staticmethod
    def _message_item_done_payload(item: MessageItemState) -> dict[str, Any]:
        """Build the message item an ``output_item.done`` event carries.

        Args:
            item: Message item state being closed.

        Returns:
            The completed item, with its own id, phase and parts.
        """
        content: list[dict[str, Any]] = []
        if item.accumulated_text:
            content.append(
                {
                    "type": "output_text",
                    "text": item.accumulated_text,
                    "annotations": [],
                    "logprobs": [],
                }
            )
        if item.accumulated_refusal:
            content.append({"type": "refusal", "refusal": item.accumulated_refusal})
        payload: dict[str, Any] = {
            "id": item.item_id,
            "type": "message",
            "role": "assistant",
            "status": "completed",
            "content": content,
        }
        if item.phase:
            payload["phase"] = item.phase
        return payload

    def _emit_tool_call_done_events(
        self,
        context: OpenResponsesStreamContext,
        results: list[dict[str, Any]],
    ) -> None:
        """Emit done events for each tool call."""
        for tc_idx, call_id in enumerate(context.tool_call_ids):
            tool_name = context.get_tool_name(call_id)
            arguments = context._tool_call_args.get(call_id, "")
            item_id = context.get_tool_call_item_id(call_id) or call_id
            output_index = context._tool_call_output_indices.get(call_id, tc_idx)
            tool_type = context.get_tool_type(call_id)

            if tool_type == "intrinsic":
                # A server item has no ``*_arguments.done`` event; the item
                # itself carries the payload, so only the output_item.done is
                # emitted.
                intrinsic_item = self._build_stream_intrinsic_item(
                    context,
                    call_id=call_id,
                    item_id=item_id,
                    kind=self._stream_intrinsic_kind(context, call_id),
                    arguments=arguments,
                    status="completed",
                )
                if intrinsic_item is None:
                    continue
                results.append(
                    {
                        "type": ResponsesEventType.OUTPUT_ITEM_DONE,
                        "output_index": output_index,
                        "item": intrinsic_item,
                    }
                )
            elif tool_type == "custom":
                # response.custom_tool_call_input.done
                results.append(
                    {
                        "type": ResponsesEventType.CUSTOM_TOOL_CALL_INPUT_DONE,
                        "item_id": item_id,
                        "output_index": output_index,
                        "input": arguments,
                    }
                )

                # response.output_item.done for the custom_tool_call
                results.append(
                    {
                        "type": ResponsesEventType.OUTPUT_ITEM_DONE,
                        "output_index": output_index,
                        "item": self._build_stream_custom_tool_call_item(
                            context,
                            item_id=item_id,
                            call_id=call_id,
                            tool_name=tool_name,
                            input_str=arguments,
                            status="completed",
                        ),
                    }
                )
            else:
                # response.function_call_arguments.done
                results.append(
                    {
                        "type": ResponsesEventType.FUNCTION_CALL_ARGS_DONE,
                        "item_id": item_id,
                        "output_index": output_index,
                        "arguments": arguments,
                    }
                )

                # response.output_item.done for the function_call
                results.append(
                    {
                        "type": ResponsesEventType.OUTPUT_ITEM_DONE,
                        "output_index": output_index,
                        "item": self._build_stream_function_call_item(
                            context,
                            item_id=item_id,
                            call_id=call_id,
                            tool_name=tool_name,
                            arguments=arguments,
                            status="completed",
                        ),
                    }
                )

    def _handle_ir_usage_to_p(
        self,
        event: UsageEvent,
        context: StreamContext | None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        usage = event["usage"]

        # With context: store usage for later merging, avoid duplicate
        # response.completed
        if context is not None:
            context.buffer_usage(usage)
            return {}

        # Without context: preserve backward-compatible behavior
        output_usage: dict[str, Any] = {
            "input_tokens": usage.get("prompt_tokens") or 0,
            "output_tokens": usage.get("completion_tokens") or 0,
            "total_tokens": usage.get("total_tokens") or 0,
        }
        if "compute_units" in usage:
            output_usage["compute_units"] = usage["compute_units"]
        resp: dict[str, Any] = {
            "status": "completed",
            "output": [],
            "usage": output_usage,
        }
        terminal = self._terminal_events(resp)
        return terminal[0] if len(terminal) == 1 else terminal
