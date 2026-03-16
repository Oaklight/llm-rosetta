"""
LLM-Rosetta - OpenAI Responses API Converter

Top-level converter implementing the 6 explicit interfaces.
Composes ContentOps, ToolOps, MessageOps, and ConfigOps for full bidirectional
conversion between IR and OpenAI Responses API format.

Note: Responses API uses a flat list of items (input/output) instead of
nested messages. The converter handles this structural difference.
"""

from typing import Any, cast
from collections.abc import Iterable

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
from ...types.ir.type_guards import (
    is_content_block_end_event,
    is_content_block_start_event,
    is_finish_event,
    is_reasoning_delta_event,
    is_stream_end_event,
    is_stream_start_event,
    is_text_delta_event,
    is_tool_call_delta_event,
    is_tool_call_start_event,
    is_usage_event,
)
from ..base import BaseConverter
from ..base.stream_context import StreamContext
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
            if isinstance(system_instruction, str):
                result["instructions"] = system_instruction
            elif isinstance(system_instruction, list):
                text_parts = []
                for part in system_instruction:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                result["instructions"] = " ".join(text_parts)

        # 2. Messages → input items
        ir_messages = ir_request.get("messages", [])
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
            ir_request["tools"] = [
                self.tool_ops.p_tool_definition_to_ir(t) for t in tools
            ]

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
        finish_reason_val = "stop"
        if status == "completed":
            finish_reason_val = "stop"
        elif status == "incomplete":
            incomplete_details = provider_response.get("incomplete_details", {})
            if isinstance(incomplete_details, dict):
                reason = incomplete_details.get("reason")
                if reason == "max_output_tokens":
                    finish_reason_val = "length"
                elif reason == "content_filter":
                    finish_reason_val = "content_filter"
        elif status == "failed":
            finish_reason_val = "error"
        elif status == "cancelled":
            finish_reason_val = "cancelled"

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
            if finish_reason == "length":
                provider_response["status"] = "incomplete"
                provider_response["incomplete_details"] = {
                    "reason": "max_output_tokens"
                }
            elif finish_reason == "error":
                provider_response["status"] = "failed"

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
        messages: Iterable[Message | ExtensionItem],
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

        # --- Response created (session start) ---
        if event_type == "response.created":
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

        # --- Text delta ---
        elif event_type == "response.output_text.delta":
            events.append(
                TextDeltaEvent(
                    type="text_delta",
                    text=chunk.get("delta", ""),
                )
            )

        # --- Reasoning summary delta ---
        elif event_type == "response.reasoning_summary_text.delta":
            events.append(
                ReasoningDeltaEvent(
                    type="reasoning_delta",
                    reasoning=chunk.get("delta", ""),
                )
            )

        # --- Output item added ---
        elif event_type == "response.output_item.added":
            item = chunk.get("item", {})
            if isinstance(item, dict):
                item_type = item.get("type", "")

                if item_type == "function_call":
                    # Register tool call in context
                    if context is not None:
                        context.register_tool_call(
                            item.get("call_id", ""), item.get("name", "")
                        )

                    start_event_tc = ToolCallStartEvent(
                        type="tool_call_start",
                        tool_call_id=item.get("call_id", ""),
                        tool_name=item.get("name", ""),
                    )
                    output_index = chunk.get("output_index")
                    if output_index is not None:
                        start_event_tc["tool_call_index"] = output_index
                    events.append(start_event_tc)

                elif item_type == "message" and context is not None:
                    # Message item added → ContentBlockStartEvent
                    block_index = context.next_block_index()
                    events.append(
                        ContentBlockStartEvent(
                            type="content_block_start",
                            block_index=block_index,
                            block_type="text",
                        )
                    )

        # --- Content part added ---
        elif event_type == "response.content_part.added":
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

        # --- Content part done ---
        elif event_type == "response.content_part.done":
            if context is not None:
                events.append(
                    ContentBlockEndEvent(
                        type="content_block_end",
                        block_index=context.current_block_index,
                    )
                )

        # --- Output item done ---
        elif event_type == "response.output_item.done":
            if context is not None:
                item = chunk.get("item", {})
                if isinstance(item, dict) and item.get("type") == "message":
                    events.append(
                        ContentBlockEndEvent(
                            type="content_block_end",
                            block_index=context.current_block_index,
                        )
                    )

        # --- Tool call arguments delta ---
        elif event_type == "response.function_call_arguments.delta":
            delta_event = ToolCallDeltaEvent(
                type="tool_call_delta",
                tool_call_id=chunk.get("call_id", ""),
                arguments_delta=chunk.get("delta", ""),
            )
            output_index = chunk.get("output_index")
            if output_index is not None:
                delta_event["tool_call_index"] = output_index
            events.append(delta_event)

        # --- Response completed ---
        elif event_type == "response.completed":
            response = chunk.get("response", chunk)

            # Determine finish reason from status
            status = response.get("status", "completed")
            if status == "completed":
                reason = "stop"
            elif status == "incomplete":
                incomplete_details = response.get("incomplete_details", {})
                inc_reason = (
                    incomplete_details.get("reason", "")
                    if isinstance(incomplete_details, dict)
                    else ""
                )
                if inc_reason == "max_output_tokens":
                    reason = "length"
                elif inc_reason == "content_filter":
                    reason = "content_filter"
                else:
                    reason = "stop"
            else:
                reason = "stop"

            # Check if any output item is a function_call to set tool_calls reason
            output_items = response.get("output", [])
            if isinstance(output_items, list):
                for item in output_items:
                    if isinstance(item, dict) and item.get("type") == "function_call":
                        reason = "tool_calls"
                        break

            events.append(
                FinishEvent(
                    type="finish",
                    finish_reason={"reason": reason},
                )
            )

            # Extract usage if present
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

            # Emit StreamEndEvent after other events
            if context is not None:
                context.mark_ended()
                events.append(StreamEndEvent(type="stream_end"))

        # --- Response failed ---
        elif event_type == "response.failed":
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

        # All other event types (response.in_progress,
        # response.output_text.done,
        # response.function_call_arguments.done, etc.) are ignored.

        return events

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
        if is_stream_start_event(event):
            # Store metadata in context if provided
            if context is not None:
                context.response_id = event["response_id"]
                context.model = event["model"]
                context.created = event.get("created", 0)
                context.mark_started()
            return {
                "type": "response.created",
                "response": {
                    "id": event["response_id"],
                    "object": "response",
                    "model": event["model"],
                    "status": "in_progress",
                    "output": [],
                },
            }

        elif is_stream_end_event(event):
            if context is not None:
                context.mark_ended()
            return {}

        elif is_content_block_start_event(event):
            block_type = event["block_type"]
            if block_type == "text":
                return {
                    "type": "response.content_part.added",
                    "part": {
                        "type": "output_text",
                        "text": "",
                    },
                }
            # Other block types are no-ops for now
            return {}

        elif is_content_block_end_event(event):
            return {
                "type": "response.content_part.done",
                "part": {
                    "type": "output_text",
                },
            }

        elif is_text_delta_event(event):
            return {
                "type": "response.output_text.delta",
                "delta": event["text"],
            }

        elif is_reasoning_delta_event(event):
            return {
                "type": "response.reasoning_summary_text.delta",
                "delta": event["reasoning"],
            }

        elif is_tool_call_start_event(event):
            result: dict[str, Any] = {
                "type": "response.output_item.added",
                "item": {
                    "type": "function_call",
                    "call_id": event["tool_call_id"],
                    "name": event["tool_name"],
                },
            }
            tc_index = event.get("tool_call_index")
            if tc_index is not None:
                result["output_index"] = tc_index
            return result

        elif is_tool_call_delta_event(event):
            result: dict[str, Any] = {
                "type": "response.function_call_arguments.delta",
                "call_id": event["tool_call_id"],
                "delta": event["arguments_delta"],
            }
            tc_index = event.get("tool_call_index")
            if tc_index is not None:
                result["output_index"] = tc_index
            return result

        elif is_finish_event(event):
            reason = event["finish_reason"]["reason"]
            status = "completed"
            response: dict[str, Any] = {"status": status}

            if reason == "length":
                response["status"] = "incomplete"
                response["incomplete_details"] = {"reason": "max_output_tokens"}
            elif reason == "error":
                response["status"] = "failed"

            # Merge pending usage from context if available
            if context is not None and context.pending_usage is not None:
                response["usage"] = {
                    "input_tokens": context.pending_usage.get("prompt_tokens") or 0,
                    "output_tokens": context.pending_usage.get("completion_tokens")
                    or 0,
                    "total_tokens": context.pending_usage.get("total_tokens") or 0,
                }

            return {
                "type": "response.completed",
                "response": response,
            }

        elif is_usage_event(event):
            usage = event["usage"]

            # With context: store usage for later merging, avoid duplicate
            # response.completed
            if context is not None:
                context.pending_usage = dict(usage)
                return {}

            # Without context: preserve backward-compatible behavior
            return {
                "type": "response.completed",
                "response": {
                    "status": "completed",
                    "usage": {
                        "input_tokens": usage.get("prompt_tokens") or 0,
                        "output_tokens": usage.get("completion_tokens") or 0,
                        "total_tokens": usage.get("total_tokens") or 0,
                    },
                },
            }

        return {}

    # ==================== Backward Compatibility ====================

    def validate_ir_input(self, ir_input):
        """Validate IR input for backward compatibility.

        Args:
            ir_input: IR input to validate.

        Returns:
            List of validation errors, empty if valid.
        """
        return self.message_ops.validate_messages(ir_input)
