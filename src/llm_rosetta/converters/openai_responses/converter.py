"""
LLM-Rosetta - OpenAI Responses API Converter

Top-level converter implementing the 6 explicit interfaces.
Composes ContentOps, ToolOps, MessageOps, and ConfigOps for full bidirectional
conversion between IR and OpenAI Responses API format.

Note: Responses API uses a flat list of items (input/output) instead of
nested messages. The converter handles this structural difference.
"""

from collections.abc import Sequence
from typing import Any, cast

from ...types.ir import (
    ExtensionItem,
    Message,
    is_text_part,
    is_tool_call_part,
    is_reasoning_part,
)
from ...types.ir.request import IRRequest
from ...types.ir.response import IRResponse
from ...types.ir.stream import (
    ContentBlockEndEvent,
    ContentBlockStartEvent,
    FinishEvent,
    IRStreamEvent,
    ReasoningDeltaEvent,
    StreamEndEvent,
    StreamStartEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallStartEvent,
    UsageEvent,
)
from ..base import BaseConverter
from ..base.stream_context import StreamContext
from ..base.tools import fix_orphaned_tool_calls_ir, strip_orphaned_tool_config
from .stream_context import OpenAIResponsesStreamContext
from ._constants import (
    RESPONSES_INCOMPLETE_REASON_TO_IR,
    RESPONSES_REASON_TO_INCOMPLETE_REASON,
    RESPONSES_REASON_TO_STATUS,
    RESPONSES_STATUS_TO_REASON,
    ResponsesEventType,
    generate_message_id,
)
from .config_ops import OpenAIResponsesConfigOps
from .content_ops import OpenAIResponsesContentOps
from .message_ops import OpenAIResponsesMessageOps
from .tool_ops import OpenAIResponsesToolOps


class OpenAIResponsesConverter(BaseConverter):
    """OpenAI Responses API converter.

    Implements the 6 explicit conversion interfaces defined by BaseConverter.

    Uses composition of Ops classes for modular, testable conversion logic.

    Note: Responses API uses ``input`` for request items and ``output`` for
    response items, with a flat item list structure.
    """

    content_ops_class = OpenAIResponsesContentOps
    tool_ops_class = OpenAIResponsesToolOps
    message_ops_class = OpenAIResponsesMessageOps
    config_ops_class = OpenAIResponsesConfigOps

    def __init__(self):
        self.content_ops = self.content_ops_class()
        self.tool_ops = self.tool_ops_class()
        self.message_ops = self.message_ops_class(self.content_ops, self.tool_ops)
        self.config_ops = self.config_ops_class()

    @classmethod
    def create_stream_context(cls) -> OpenAIResponsesStreamContext:
        """Create a stream context with Responses API specific state."""
        return OpenAIResponsesStreamContext()

    # ==================== Top-level Interfaces ====================

    def request_to_provider(
        self,
        ir_request: IRRequest,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], list[str]]:
        """Convert IRRequest to OpenAI Responses API request parameters.

        Orchestrates all Ops classes to build the complete provider request.

        Args:
            ir_request: IR request.

        Returns:
            Tuple of (provider request dict, warnings list).
        """
        warnings: list[str] = []
        result: dict[str, Any] = {"model": ir_request["model"]}

        # 1. System instruction → instructions field
        system_instruction = ir_request.get("system_instruction")
        if system_instruction:
            result["instructions"] = system_instruction

        # 2. Messages → input items — fix orphaned tool_calls at IR level
        # before conversion.  OpenAI Responses API strictly requires every
        # function_call to have a matching function_call_output.
        ir_messages = fix_orphaned_tool_calls_ir(ir_request.get("messages", []))
        warnings.extend(strip_orphaned_tool_config(ir_request))
        items, msg_warnings = self.message_ops.ir_messages_to_p(ir_messages)
        warnings.extend(msg_warnings)
        result["input"] = items

        # 3. Tools
        tools = ir_request.get("tools")
        if tools:
            result["tools"] = [self.tool_ops.ir_tool_definition_to_p(t) for t in tools]

        # 4. Tool choice
        tool_choice = ir_request.get("tool_choice")
        if tool_choice:
            result["tool_choice"] = self.tool_ops.ir_tool_choice_to_p(tool_choice)

        # 5. Tool config
        tool_config = ir_request.get("tool_config")
        if tool_config:
            tc_fields = self.tool_ops.ir_tool_config_to_p(tool_config)
            result.update(tc_fields)

        # 6. Generation config
        gen_config = ir_request.get("generation")
        if gen_config:
            gen_fields = self.config_ops.ir_generation_config_to_p(gen_config)
            result.update(gen_fields)

        # 7. Response format
        resp_format = ir_request.get("response_format")
        if resp_format:
            rf_fields = self.config_ops.ir_response_format_to_p(resp_format)
            result.update(rf_fields)

        # 8. Stream config
        stream = ir_request.get("stream")
        if stream:
            stream_fields = self.config_ops.ir_stream_config_to_p(stream)
            result.update(stream_fields)

        # 9. Reasoning config
        reasoning = ir_request.get("reasoning")
        if reasoning:
            reasoning_fields = self.config_ops.ir_reasoning_config_to_p(reasoning)
            result.update(reasoning_fields)

        # 10. Cache config
        cache = ir_request.get("cache")
        if cache:
            cache_fields = self.config_ops.ir_cache_config_to_p(cache)
            result.update(cache_fields)

        # 11. Provider extensions (pass-through)
        extensions = ir_request.get("provider_extensions")
        if extensions:
            result.update(extensions)

        return result, warnings

    def request_from_provider(
        self,
        provider_request: dict[str, Any],
        **kwargs: Any,
    ) -> IRRequest:
        """Convert OpenAI Responses API request to IRRequest.

        Args:
            provider_request: OpenAI Responses request dict (or SDK object).

        Returns:
            IR request.
        """
        provider_request = self._normalize(provider_request)

        ir_request: dict[str, Any] = {
            "model": provider_request.get("model", ""),
            "messages": [],
        }

        # 1. Instructions → system_instruction
        instructions = provider_request.get("instructions")
        if instructions:
            ir_request["system_instruction"] = instructions

        # 2. Input items → messages
        input_items = provider_request.get("input", [])
        if isinstance(input_items, str):
            input_items = [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": input_items}],
                }
            ]
        if isinstance(input_items, list):
            ir_messages = self.message_ops.p_messages_to_ir(input_items)
            ir_request["messages"] = ir_messages

        # 3. Tools
        tools = provider_request.get("tools")
        if tools:
            ir_tools = []
            for t in tools:
                # Skip disabled tools (e.g. web_search with external_web_access=false)
                if isinstance(t, dict) and t.get("external_web_access") is False:
                    continue
                ir_tools.append(self.tool_ops.p_tool_definition_to_ir(t))
            if ir_tools:
                ir_request["tools"] = ir_tools

        # 4. Tool choice
        tool_choice = provider_request.get("tool_choice")
        if tool_choice is not None:
            ir_request["tool_choice"] = self.tool_ops.p_tool_choice_to_ir(tool_choice)

        # 5. Tool config
        tool_config_fields = {}
        if "parallel_tool_calls" in provider_request:
            tool_config_fields["parallel_tool_calls"] = provider_request[
                "parallel_tool_calls"
            ]
        if "max_tool_calls" in provider_request:
            tool_config_fields["max_tool_calls"] = provider_request["max_tool_calls"]
        if tool_config_fields:
            ir_request["tool_config"] = self.tool_ops.p_tool_config_to_ir(
                tool_config_fields
            )

        # 6. Generation config
        gen_config = self.config_ops.p_generation_config_to_ir(provider_request)
        if gen_config:
            ir_request["generation"] = gen_config

        # 7. Response format (text field in Responses API)
        text_format = provider_request.get("text")
        if text_format:
            ir_request["response_format"] = self.config_ops.p_response_format_to_ir(
                text_format
            )

        # 8. Reasoning config
        reasoning = provider_request.get("reasoning")
        if reasoning:
            ir_request["reasoning"] = self.config_ops.p_reasoning_config_to_ir(
                {"reasoning": reasoning}
            )

        # 9. Stream config
        stream = provider_request.get("stream")
        stream_options = provider_request.get("stream_options")
        if stream is not None or stream_options:
            ir_request["stream"] = self.config_ops.p_stream_config_to_ir(
                {"stream": stream, "stream_options": stream_options}
            )

        # 10. Cache config
        cache_fields = {}
        if "prompt_cache_key" in provider_request:
            cache_fields["prompt_cache_key"] = provider_request["prompt_cache_key"]
        if "prompt_cache_retention" in provider_request:
            cache_fields["prompt_cache_retention"] = provider_request[
                "prompt_cache_retention"
            ]
        if cache_fields:
            ir_request["cache"] = self.config_ops.p_cache_config_to_ir(cache_fields)

        return cast(IRRequest, ir_request)

    def response_from_provider(
        self,
        provider_response: dict[str, Any],
        **kwargs: Any,
    ) -> IRResponse:
        """Convert OpenAI Responses API response to IRResponse.

        Args:
            provider_response: OpenAI Responses response dict (or SDK object).

        Returns:
            IR response.
        """
        provider_response = self._normalize(provider_response)

        choices = []
        output_items = provider_response.get("output", [])

        # Determine finish reason from status
        status = provider_response.get("status")
        if status == "incomplete":
            incomplete_details = provider_response.get("incomplete_details", {})
            inc_reason = (
                incomplete_details.get("reason", "")
                if isinstance(incomplete_details, dict)
                else ""
            )
            finish_reason_val = RESPONSES_INCOMPLETE_REASON_TO_IR.get(
                inc_reason, "stop"
            )
        else:
            finish_reason_val = RESPONSES_STATUS_TO_REASON.get(status or "", "stop")

        # Convert output items to IR message content
        ir_items = self.message_ops.p_messages_to_ir(output_items)

        # Collect all content parts into a single assistant message
        message_content: list[dict[str, Any]] = []
        for ir_item in ir_items:
            if isinstance(ir_item, dict) and "role" in ir_item:
                # It's a message - extract content
                content = ir_item.get("content", [])
                message_content.extend(cast(list, content))
            elif isinstance(ir_item, dict) and "type" in ir_item:
                # It's an extension item (system_event etc.) - skip for choices
                pass

        if message_content:
            choices.append(
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": message_content},
                    "finish_reason": {"reason": finish_reason_val},
                }
            )

        ir_response: dict[str, Any] = {
            "id": provider_response.get("id", ""),
            "object": "response",
            "created": int(provider_response.get("created_at", 0)),
            "model": provider_response.get("model", ""),
            "choices": choices,
        }

        # Handle usage
        p_usage = provider_response.get("usage")
        if p_usage:
            usage_info: dict[str, Any] = {
                "prompt_tokens": p_usage.get("input_tokens") or 0,
                "completion_tokens": p_usage.get("output_tokens") or 0,
                "total_tokens": p_usage.get("total_tokens") or 0,
            }

            # Handle detailed statistics
            p_input_details = p_usage.get("input_tokens_details")
            if p_input_details:
                if "cached_tokens" in p_input_details:
                    usage_info["cache_read_tokens"] = p_input_details["cached_tokens"]

            p_output_details = p_usage.get("output_tokens_details")
            if p_output_details:
                if "reasoning_tokens" in p_output_details:
                    usage_info["reasoning_tokens"] = p_output_details[
                        "reasoning_tokens"
                    ]

            ir_response["usage"] = usage_info

        if "service_tier" in provider_response:
            ir_response["service_tier"] = provider_response["service_tier"]

        return cast(IRResponse, ir_response)

    def response_to_provider(
        self,
        ir_response: IRResponse,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convert IRResponse to OpenAI Responses API response.

        Args:
            ir_response: IR response.

        Returns:
            OpenAI Responses response dict.
        """
        provider_response: dict[str, Any] = {
            "id": ir_response.get("id", ""),
            "object": "response",
            "created_at": ir_response.get("created", 0),
            "model": ir_response.get("model", ""),
            "output": [],
            "status": "completed",
        }

        for choice in ir_response.get("choices", []):
            message = choice.get("message")
            if not message:
                continue

            content_parts = message.get("content", [])

            for part in content_parts:
                if is_text_part(part):
                    provider_response["output"].append(
                        {
                            "type": "message",
                            "role": "assistant",
                            "content": [{"type": "output_text", "text": part["text"]}],
                        }
                    )
                elif is_tool_call_part(part):
                    provider_response["output"].append(
                        self.tool_ops.ir_tool_call_to_p(part)
                    )
                elif is_reasoning_part(part):
                    provider_response["output"].append(
                        self.content_ops.ir_reasoning_to_p(part)
                    )

            # Set finish reason
            finish_reason = choice.get("finish_reason", {}).get("reason", "stop")
            status = RESPONSES_REASON_TO_STATUS.get(finish_reason, "completed")
            provider_response["status"] = status
            incomplete_reason = RESPONSES_REASON_TO_INCOMPLETE_REASON.get(finish_reason)
            if incomplete_reason:
                provider_response["incomplete_details"] = {"reason": incomplete_reason}

        # Usage
        ir_usage = ir_response.get("usage")
        if ir_usage:
            usage: dict[str, Any] = {
                "input_tokens": ir_usage.get("prompt_tokens") or 0,
                "output_tokens": ir_usage.get("completion_tokens") or 0,
                "total_tokens": ir_usage.get("total_tokens") or 0,
            }

            if "cache_read_tokens" in ir_usage:
                usage["input_tokens_details"] = {
                    "cached_tokens": ir_usage["cache_read_tokens"]
                }

            if "reasoning_tokens" in ir_usage:
                usage["output_tokens_details"] = {
                    "reasoning_tokens": ir_usage["reasoning_tokens"]
                }

            provider_response["usage"] = usage

        if "service_tier" in ir_response:
            provider_response["service_tier"] = ir_response["service_tier"]

        return provider_response

    def messages_to_provider(
        self,
        messages: Sequence[Message | ExtensionItem],
        **kwargs: Any,
    ) -> tuple[list[Any], list[str]]:
        """Convert IR message list to OpenAI Responses input items.

        Delegates to message_ops.

        Args:
            messages: IR messages (may contain ExtensionItems).

        Returns:
            Tuple of (converted items, warnings).
        """
        return self.message_ops.ir_messages_to_p(messages, **kwargs)

    def messages_from_provider(
        self,
        provider_messages: list[Any],
        **kwargs: Any,
    ) -> list[Message | ExtensionItem]:
        """Convert OpenAI Responses items to IR message list.

        Delegates to message_ops.

        Args:
            provider_messages: OpenAI Responses items.

        Returns:
            IR messages.
        """
        return self.message_ops.p_messages_to_ir(provider_messages, **kwargs)

    # ==================== Backward Compatibility ====================
    # These methods maintain backward compatibility with the old API

    def to_provider(self, ir_input, tools=None, tool_choice=None, **kwargs):
        """Backward-compatible conversion method.

        Handles both IRRequest dicts and plain message lists.

        Args:
            ir_input: Either an IRRequest dict or a list of IR messages.
            tools: Optional tool definitions.
            tool_choice: Optional tool choice config.

        Returns:
            Tuple of (provider request dict, warnings list).
        """
        # Check if it's an IRRequest (has "messages" key)
        if isinstance(ir_input, dict) and "messages" in ir_input:
            return self.request_to_provider(ir_input, **kwargs)

        # It's a plain message list - wrap in a minimal request
        items, warnings = self.message_ops.ir_messages_to_p(ir_input)
        result: dict[str, Any] = {"input": items}

        if tools:
            result["tools"] = [self.tool_ops.ir_tool_definition_to_p(t) for t in tools]

        if tool_choice:
            result["tool_choice"] = self.tool_ops.ir_tool_choice_to_p(tool_choice)

        return result, warnings

    # ==================== Compatibility Aliases ====================

    def _convert_image_to_responses(self, image_part):
        """Convert IR image to Responses API format (compatibility alias)."""
        return self.content_ops.ir_image_to_p(image_part)

    def _convert_file_to_responses(self, file_part):
        """Convert IR file to Responses API format (compatibility alias)."""
        return self.content_ops.ir_file_to_p(file_part)

    def _convert_image_from_responses(self, image_part):
        """Convert Responses API image to IR format (compatibility alias)."""
        return self.content_ops.p_image_to_ir(image_part)

    def _convert_file_from_responses(self, file_part):
        """Convert Responses API file to IR format (compatibility alias)."""
        return self.content_ops.p_file_to_ir(file_part)

    # ==================== Stream Support ====================

    def stream_response_from_provider(
        self,
        chunk: dict[str, Any],
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

        Args:
            chunk: OpenAI Responses SSE event dict (or SDK object).
            context: Optional stream context for stateful conversions.

        Returns:
            List of IR stream events extracted from the event.
        """
        chunk = self._normalize(chunk)
        events: list[IRStreamEvent] = []
        event_type = chunk.get("type", "")

        handler_name = self._FROM_P_DISPATCH.get(event_type)
        if handler_name is not None:
            getattr(self, handler_name)(chunk, context, events)

        # All other event types (response.in_progress,
        # response.output_text.done, etc.) are ignored.

        return events

    # --- from_provider handlers ---

    def _handle_response_created_from_p(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        if context is not None:
            response = chunk.get("response", {})
            response_id = response.get("id", "")
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

    def _handle_output_text_delta_from_p(
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

    def _handle_reasoning_delta_from_p(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        events.append(
            ReasoningDeltaEvent(
                type="reasoning_delta",
                reasoning=chunk.get("delta", ""),
            )
        )

    def _handle_output_item_added_from_p(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        item = chunk.get("item", {})
        if isinstance(item, dict):
            item_type = item.get("type", "")

            if item_type == "function_call":
                call_id = item.get("call_id", "")
                item_id = item.get("id", "")

                # Register tool call in context
                if context is not None:
                    context.register_tool_call(call_id, item.get("name", ""))
                    context.register_tool_call_item(call_id, item_id)

                start_event_tc = ToolCallStartEvent(
                    type="tool_call_start",
                    tool_call_id=call_id,
                    tool_name=item.get("name", ""),
                )
                output_index = chunk.get("output_index")
                if output_index is not None:
                    start_event_tc["tool_call_index"] = output_index
                events.append(start_event_tc)

            elif item_type == "message":
                # Message-level output item — no IR event needed.
                # The actual content block is signaled by the subsequent
                # ``response.content_part.added`` event which produces its
                # own ``ContentBlockStartEvent``.  Emitting a
                # ``ContentBlockStartEvent`` here would cause duplicate
                # ``response.content_part.added`` events on round-trip.
                pass

    def _handle_content_part_added_from_p(
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
            block_index = context.next_block_index()
            events.append(
                ContentBlockStartEvent(
                    type="content_block_start",
                    block_index=block_index,
                    block_type=block_type,
                )
            )

    def _handle_content_part_done_from_p(
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

    def _handle_output_item_done_from_p(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        # For function_call items, store the completed item in context
        # so it can be included in the response.completed output array.
        # Message-level done is a no-op (content_part.done handles it).
        item = chunk.get("item", {})
        if isinstance(item, dict) and item.get("type") == "function_call":
            if context is not None:
                call_id = item.get("call_id", "")
                if call_id:
                    context.set_tool_call_args(call_id, item.get("arguments", ""))

    def _handle_function_call_args_delta_from_p(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        delta_text = chunk.get("delta", "")
        # Upstream may send call_id or item_id — resolve to call_id
        call_id = chunk.get("call_id", "")
        if not call_id and isinstance(context, OpenAIResponsesStreamContext):
            item_id = chunk.get("item_id", "")
            if item_id:
                call_id = context.item_id_to_call_id.get(item_id, "")
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

    def _handle_function_call_args_done_from_p(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        # Resolve call_id from item_id if needed
        call_id = chunk.get("call_id", "")
        if not call_id and isinstance(context, OpenAIResponsesStreamContext):
            item_id = chunk.get("item_id", "")
            if item_id:
                call_id = context.item_id_to_call_id.get(item_id, "")
        arguments = chunk.get("arguments", "")
        # Store final arguments in context
        if context is not None and call_id:
            context.set_tool_call_args(call_id, arguments)

    def _handle_response_completed_from_p(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        response = chunk.get("response", chunk)

        # Determine finish reason from status
        status = response.get("status", "completed")
        if status == "incomplete":
            incomplete_details = response.get("incomplete_details", {})
            inc_reason = (
                incomplete_details.get("reason", "")
                if isinstance(incomplete_details, dict)
                else ""
            )
            reason = RESPONSES_INCOMPLETE_REASON_TO_IR.get(inc_reason, "stop")
        else:
            reason = RESPONSES_STATUS_TO_REASON.get(status, "stop")

        # Check if any output item is a function_call to set tool_calls reason
        output_items = response.get("output", [])
        if isinstance(output_items, list):
            for item in output_items:
                if isinstance(item, dict) and item.get("type") == "function_call":
                    reason = "tool_calls"
                    break

        # Emit UsageEvent before FinishEvent so that downstream
        # converters can store usage in context.pending_usage before
        # FinishEvent builds the terminal response.completed event.
        usage = response.get("usage")
        if isinstance(usage, dict):
            events.append(
                UsageEvent(
                    type="usage",
                    usage={
                        "prompt_tokens": usage.get("input_tokens") or 0,
                        "completion_tokens": usage.get("output_tokens") or 0,
                        "total_tokens": usage.get("total_tokens") or 0,
                    },
                )
            )

        events.append(
            FinishEvent(
                type="finish",
                finish_reason={"reason": reason},
            )
        )

        # Emit StreamEndEvent after other events
        if context is not None:
            context.mark_ended()
            events.append(StreamEndEvent(type="stream_end"))

    def _handle_response_failed_from_p(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        events.append(
            FinishEvent(
                type="finish",
                finish_reason={"reason": "error"},
            )
        )

        # Emit StreamEndEvent after FinishEvent
        if context is not None:
            context.mark_ended()
            events.append(StreamEndEvent(type="stream_end"))

    _FROM_P_DISPATCH: dict[str, str] = {
        ResponsesEventType.RESPONSE_CREATED: "_handle_response_created_from_p",
        ResponsesEventType.OUTPUT_TEXT_DELTA: "_handle_output_text_delta_from_p",
        ResponsesEventType.REASONING_SUMMARY_TEXT_DELTA: "_handle_reasoning_delta_from_p",
        ResponsesEventType.OUTPUT_ITEM_ADDED: "_handle_output_item_added_from_p",
        ResponsesEventType.CONTENT_PART_ADDED: "_handle_content_part_added_from_p",
        ResponsesEventType.CONTENT_PART_DONE: "_handle_content_part_done_from_p",
        ResponsesEventType.OUTPUT_ITEM_DONE: "_handle_output_item_done_from_p",
        ResponsesEventType.FUNCTION_CALL_ARGS_DELTA: "_handle_function_call_args_delta_from_p",
        ResponsesEventType.FUNCTION_CALL_ARGS_DONE: "_handle_function_call_args_done_from_p",
        ResponsesEventType.RESPONSE_COMPLETED: "_handle_response_completed_from_p",
        ResponsesEventType.RESPONSE_FAILED: "_handle_response_failed_from_p",
    }

    def stream_response_to_provider(
        self,
        event: IRStreamEvent,
        context: StreamContext | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert an IR stream event to an OpenAI Responses SSE event.

        When a ``context`` is provided, ``UsageEvent`` stores usage info in
        the context instead of emitting a duplicate ``response.completed``,
        and ``FinishEvent`` merges any pending usage into its
        ``response.completed`` output.

        Args:
            event: IR stream event.
            context: Optional stream context for stateful conversions.

        Returns:
            OpenAI Responses SSE event dict, or a list of dicts.
        """
        handler_name = self._TO_P_DISPATCH.get(event["type"])
        if handler_name is not None:
            return getattr(self, handler_name)(event, context)
        return {}

    # --- to_provider handlers ---

    def _handle_stream_start_to_p(
        self,
        event: StreamStartEvent,
        context: StreamContext | None,
    ) -> dict[str, Any]:
        # Store metadata in context if provided
        if context is not None:
            context.response_id = event["response_id"]
            context.model = event["model"]
            context.created = event.get("created", 0)
            context.mark_started()
        return {
            "type": ResponsesEventType.RESPONSE_CREATED,
            "response": {
                "id": event["response_id"],
                "object": "response",
                "model": event["model"],
                "status": "in_progress",
                "output": [],
            },
        }

    def _handle_stream_end_to_p(
        self,
        event: StreamEndEvent,
        context: StreamContext | None,
    ) -> dict[str, Any]:
        if context is not None:
            context.mark_ended()
            # Emit the deferred response.completed if FinishEvent
            # stored one.  This ensures UsageEvents that arrive
            # between FinishEvent and StreamEndEvent (e.g. OpenAI
            # Chat sends usage in a separate chunk after
            # finish_reason) are merged into the response.
            if context.pending_response is not None:
                resp = context.pending_response
                context.pending_response = None
                # Merge any usage that arrived after FinishEvent
                if context.pending_usage is not None and "usage" not in resp:
                    resp["usage"] = {
                        "input_tokens": context.pending_usage.get("prompt_tokens") or 0,
                        "output_tokens": context.pending_usage.get("completion_tokens")
                        or 0,
                        "total_tokens": context.pending_usage.get("total_tokens") or 0,
                    }
                return {
                    "type": ResponsesEventType.RESPONSE_COMPLETED,
                    "response": resp,
                }
        return {}

    def _handle_content_block_start_to_p(
        self,
        event: ContentBlockStartEvent,
        context: OpenAIResponsesStreamContext | None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        block_type = event["block_type"]
        if block_type == "text":
            # With context: emit output_item.added + content_part.added
            # and mark the item as emitted so the first TextDelta doesn't
            # re-emit them.
            if context is not None and not context.output_item_emitted:
                context.output_item_emitted = True
                item_id = generate_message_id(context.response_id)
                context.item_id = item_id
                return [
                    {
                        "type": ResponsesEventType.OUTPUT_ITEM_ADDED,
                        "output_index": 0,
                        "item": {
                            "id": item_id,
                            "type": "message",
                            "role": "assistant",
                            "content": [],
                        },
                    },
                    {
                        "type": ResponsesEventType.CONTENT_PART_ADDED,
                        "output_index": 0,
                        "content_index": 0,
                        "part": {
                            "type": "output_text",
                            "text": "",
                        },
                    },
                ]
            # Fallback: just emit content_part.added (e.g. no context, or
            # output item already emitted by a prior ContentBlockStartEvent)
            return {
                "type": ResponsesEventType.CONTENT_PART_ADDED,
                "part": {
                    "type": "output_text",
                    "text": "",
                },
            }
        # Other block types are no-ops for now
        return {}

    def _handle_content_block_end_to_p(
        self,
        event: ContentBlockEndEvent,
        context: OpenAIResponsesStreamContext | None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        if context is not None:
            context.content_part_done_emitted = True
            accumulated = context.accumulated_text
            # Emit output_text.done before content_part.done (matches
            # OpenAI's event ordering)
            return [
                {
                    "type": ResponsesEventType.OUTPUT_TEXT_DONE,
                    "output_index": 0,
                    "content_index": 0,
                    "text": accumulated,
                },
                {
                    "type": ResponsesEventType.CONTENT_PART_DONE,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": accumulated},
                },
            ]
        return {
            "type": ResponsesEventType.CONTENT_PART_DONE,
            "part": {
                "type": "output_text",
            },
        }

    def _handle_text_delta_to_p(
        self,
        event: TextDeltaEvent,
        context: OpenAIResponsesStreamContext | None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        choice_index = event.get("choice_index", 0)
        text = event["text"]
        delta_event: dict[str, Any] = {
            "type": ResponsesEventType.OUTPUT_TEXT_DELTA,
            "delta": text,
            "output_index": choice_index,
            "content_index": 0,
        }

        # Accumulate text in context for response.completed output
        if context is not None:
            context.accumulated_text += text

        # Emit output_item.added + content_part.added before the first
        # text delta so clients (e.g. Codex CLI) can register the item.
        if context is not None and not context.output_item_emitted:
            context.output_item_emitted = True
            item_id = generate_message_id(context.response_id)
            context.item_id = item_id
            return [
                {
                    "type": ResponsesEventType.OUTPUT_ITEM_ADDED,
                    "output_index": choice_index,
                    "item": {
                        "id": item_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                    },
                },
                {
                    "type": ResponsesEventType.CONTENT_PART_ADDED,
                    "output_index": choice_index,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": "",
                    },
                },
                delta_event,
            ]

        return delta_event

    def _handle_reasoning_delta_to_p(
        self,
        event: ReasoningDeltaEvent,
        context: StreamContext | None,
    ) -> dict[str, Any]:
        return {
            "type": ResponsesEventType.REASONING_SUMMARY_TEXT_DELTA,
            "delta": event["reasoning"],
        }

    def _handle_tool_call_start_to_p(
        self,
        event: ToolCallStartEvent,
        context: StreamContext | None,
    ) -> dict[str, Any]:
        call_id = event["tool_call_id"]
        tool_name = event["tool_name"]
        item_id = call_id

        # Register in context for later done events
        if context is not None and call_id:
            context.register_tool_call(call_id, tool_name)
            context.register_tool_call_item(call_id, item_id)

        result: dict[str, Any] = {
            "type": ResponsesEventType.OUTPUT_ITEM_ADDED,
            "item": {
                "id": item_id,
                "type": "function_call",
                "call_id": call_id,
                "name": tool_name,
            },
        }
        tc_index = event.get("tool_call_index")
        if tc_index is not None:
            result["output_index"] = tc_index
        return result

    def _handle_tool_call_delta_to_p(
        self,
        event: ToolCallDeltaEvent,
        context: StreamContext | None,
    ) -> dict[str, Any]:
        call_id = event["tool_call_id"]
        delta = event["arguments_delta"]
        tc_index = event.get("tool_call_index")

        # Defense-in-depth: resolve empty tool_call_id by index.
        # Some upstream providers (e.g. certain Chat Completions
        # implementations) only send tool_call_id on the first chunk.
        if not call_id and context is not None and tc_index is not None:
            if tc_index < len(context._tool_call_order):
                call_id = context._tool_call_order[tc_index]

        # Accumulate arguments in context for done events
        if context is not None and call_id:
            context.append_tool_call_args(call_id, delta)

        # Use item_id per Responses API spec
        item_id = ""
        if context is not None and call_id:
            item_id = context.get_tool_call_item_id(call_id)
        if not item_id and call_id:
            item_id = call_id

        result: dict[str, Any] = {
            "type": ResponsesEventType.FUNCTION_CALL_ARGS_DELTA,
            "item_id": item_id,
            "delta": delta,
        }
        if tc_index is not None:
            result["output_index"] = tc_index
        return result

    def _handle_finish_to_p(
        self,
        event: FinishEvent,
        context: OpenAIResponsesStreamContext | None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        reason = event["finish_reason"]["reason"]
        status = RESPONSES_REASON_TO_STATUS.get(reason, "completed")

        response = self._build_finish_response(status, context, reason)

        # Emit done events before response.completed
        results: list[dict[str, Any]] = []

        if context is not None:
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
        results.append(
            {
                "type": ResponsesEventType.RESPONSE_COMPLETED,
                "response": response,
            }
        )
        return results

    def _build_finish_response(
        self,
        status: str,
        context: OpenAIResponsesStreamContext | None,
        finish_reason: str = "length",
    ) -> dict[str, Any]:
        """Build the response dict for a FinishEvent."""
        output: list[dict[str, Any]] = []
        if context is not None:
            accumulated = context.accumulated_text
            item_id = context.item_id
            if accumulated:
                output.append(
                    {
                        "id": item_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [{"type": "output_text", "text": accumulated}],
                    }
                )

            # Add accumulated function calls to output
            for call_id in context._tool_call_order:
                tool_name = context.get_tool_name(call_id)
                arguments = context._tool_call_args.get(call_id, "")
                tc_item_id = context.get_tool_call_item_id(call_id) or call_id
                output.append(
                    {
                        "id": tc_item_id,
                        "type": "function_call",
                        "call_id": call_id,
                        "name": tool_name,
                        "arguments": arguments,
                        "status": "completed",
                    }
                )

        response: dict[str, Any] = {"status": status, "output": output}

        # Populate id, object, model from context so clients can parse
        # the completed response envelope.
        if context is not None:
            response["id"] = context.response_id
            response["object"] = "response"
            response["model"] = context.model

        if status == "incomplete":
            incomplete_reason = RESPONSES_REASON_TO_INCOMPLETE_REASON.get(
                finish_reason, "max_output_tokens"
            )
            response["incomplete_details"] = {"reason": incomplete_reason}

        # Merge pending usage from context if available
        if context is not None and context.pending_usage is not None:
            response["usage"] = {
                "input_tokens": context.pending_usage.get("prompt_tokens") or 0,
                "output_tokens": context.pending_usage.get("completion_tokens") or 0,
                "total_tokens": context.pending_usage.get("total_tokens") or 0,
            }

        return response

    def _emit_text_done_events(
        self,
        context: OpenAIResponsesStreamContext,
        results: list[dict[str, Any]],
    ) -> None:
        """Emit text done events if we had text output."""
        if not context.output_item_emitted:
            return

        accumulated = context.accumulated_text
        item_id = context.item_id

        # Only emit output_text.done + content_part.done if not
        # already emitted by a prior ContentBlockEndEvent
        if not context.content_part_done_emitted:
            results.append(
                {
                    "type": ResponsesEventType.OUTPUT_TEXT_DONE,
                    "output_index": 0,
                    "content_index": 0,
                    "text": accumulated,
                }
            )
            results.append(
                {
                    "type": ResponsesEventType.CONTENT_PART_DONE,
                    "output_index": 0,
                    "content_index": 0,
                    "part": {"type": "output_text", "text": accumulated},
                }
            )
        results.append(
            {
                "type": ResponsesEventType.OUTPUT_ITEM_DONE,
                "output_index": 0,
                "item": {
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": accumulated}],
                },
            }
        )

    def _emit_tool_call_done_events(
        self,
        context: OpenAIResponsesStreamContext,
        results: list[dict[str, Any]],
    ) -> None:
        """Emit done events for each tool call."""
        for tc_idx, call_id in enumerate(context._tool_call_order):
            tool_name = context.get_tool_name(call_id)
            arguments = context._tool_call_args.get(call_id, "")
            item_id = context.get_tool_call_item_id(call_id) or call_id
            output_index = tc_idx + (1 if context.output_item_emitted else 0)

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
                    "item": {
                        "id": item_id,
                        "type": "function_call",
                        "call_id": call_id,
                        "name": tool_name,
                        "arguments": arguments,
                        "status": "completed",
                    },
                }
            )

    def _handle_usage_to_p(
        self,
        event: UsageEvent,
        context: StreamContext | None,
    ) -> dict[str, Any]:
        usage = event["usage"]

        # With context: store usage for later merging, avoid duplicate
        # response.completed
        if context is not None:
            context.pending_usage = dict(usage)
            return {}

        # Without context: preserve backward-compatible behavior
        resp: dict[str, Any] = {
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": usage.get("prompt_tokens") or 0,
                "output_tokens": usage.get("completion_tokens") or 0,
                "total_tokens": usage.get("total_tokens") or 0,
            },
        }
        return {
            "type": ResponsesEventType.RESPONSE_COMPLETED,
            "response": resp,
        }

    _TO_P_DISPATCH: dict[str, str] = {
        "stream_start": "_handle_stream_start_to_p",
        "stream_end": "_handle_stream_end_to_p",
        "content_block_start": "_handle_content_block_start_to_p",
        "content_block_end": "_handle_content_block_end_to_p",
        "text_delta": "_handle_text_delta_to_p",
        "reasoning_delta": "_handle_reasoning_delta_to_p",
        "tool_call_start": "_handle_tool_call_start_to_p",
        "tool_call_delta": "_handle_tool_call_delta_to_p",
        "finish": "_handle_finish_to_p",
        "usage": "_handle_usage_to_p",
    }

    # ==================== Backward Compatibility ====================

    def validate_ir_input(self, ir_input):
        """Validate IR input for backward compatibility.

        Args:
            ir_input: IR input to validate.

        Returns:
            List of validation errors, empty if valid.
        """
        return self.message_ops.validate_messages(ir_input)
