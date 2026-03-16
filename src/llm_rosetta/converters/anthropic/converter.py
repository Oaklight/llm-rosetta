"""
LLM-Rosetta - Anthropic Messages API Converter

Top-level converter implementing the 6 explicit interfaces + 2 stream methods.
Composes ContentOps, ToolOps, MessageOps, and ConfigOps for full bidirectional
conversion between IR and Anthropic Messages API format.

Key Anthropic differences from OpenAI:
- ``max_tokens`` is required (default 4096)
- System messages via top-level ``system`` parameter
- Single response message (not choices list)
- No ``created`` timestamp (uses ``time.time()``)
- Tool call arguments are Dict (not JSON string)
- Thinking/reasoning with ``signature`` field
"""

import time
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
from .config_ops import AnthropicConfigOps
from .content_ops import AnthropicContentOps
from .message_ops import AnthropicMessageOps
from .tool_ops import AnthropicToolOps


class AnthropicConverter(BaseConverter):
    """Anthropic Messages API converter.

    Implements the 6 explicit conversion interfaces defined by BaseConverter,
    plus 2 stream methods for SSE event-level conversion.

    Uses composition of Ops classes for modular, testable conversion logic.
    """

    content_ops_class = AnthropicContentOps
    tool_ops_class = AnthropicToolOps
    message_ops_class = AnthropicMessageOps
    config_ops_class = AnthropicConfigOps

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
        """Convert IRRequest to Anthropic Messages API request parameters.

        Orchestrates all Ops classes to build the complete provider request.

        Args:
            ir_request: IR request.

        Returns:
            Tuple of (provider request dict, warnings list).
        """
        warnings: list[str] = []
        result: dict[str, Any] = {"model": ir_request["model"]}

        # 1. System instruction → top-level system parameter
        system_instruction = ir_request.get("system_instruction")
        if system_instruction:
            if isinstance(system_instruction, str):
                result["system"] = system_instruction
            elif isinstance(system_instruction, list):
                text_parts = []
                for part in system_instruction:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                result["system"] = " ".join(text_parts)

        # 2. Messages (system messages in IR are extracted to system param)
        ir_messages = ir_request.get("messages", [])

        # Extract system messages from message list
        for item in ir_messages:
            if isinstance(item, dict) and item.get("role") == "system":
                content = item.get("content", [])
                text_parts = []
                for part in content:
                    if is_text_part(part):
                        text_parts.append(part["text"])
                if text_parts and "system" not in result:
                    result["system"] = " ".join(text_parts)

        converted_msgs, msg_warnings = self.message_ops.ir_messages_to_p(ir_messages)
        warnings.extend(msg_warnings)
        result["messages"] = converted_msgs

        # 3. Generation config (must come before tools since max_tokens is required)
        gen_config = ir_request.get("generation")
        if gen_config:
            gen_fields = self.config_ops.ir_generation_config_to_p(gen_config)
            result.update(gen_fields)
        else:
            # Anthropic requires max_tokens
            result["max_tokens"] = 4096

        # 4. Tools
        tools = ir_request.get("tools")
        if tools:
            result["tools"] = [self.tool_ops.ir_tool_definition_to_p(t) for t in tools]

        # 5. Tool choice
        tool_choice = ir_request.get("tool_choice")
        if tool_choice:
            result["tool_choice"] = self.tool_ops.ir_tool_choice_to_p(tool_choice)

        # 6. Tool config (disable_parallel_tool_use merges into tool_choice)
        tool_config = ir_request.get("tool_config")
        if tool_config:
            tc_fields = self.tool_ops.ir_tool_config_to_p(tool_config)
            if tc_fields:
                if "tool_choice" not in result:
                    result["tool_choice"] = {"type": "auto"}
                result["tool_choice"].update(tc_fields)
            if "max_calls" in tool_config:
                warnings.append("Anthropic does not support max_tool_calls, ignored")

        # 7. Response format (not supported)
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

        # 10. Cache config (block-level, warning)
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
        """Convert Anthropic Messages API request to IRRequest.

        Args:
            provider_request: Anthropic request dict (or SDK object).

        Returns:
            IR request.
        """
        provider_request = self._normalize(provider_request)

        ir_request: dict[str, Any] = {
            "model": provider_request.get("model", ""),
            "messages": [],
        }

        # 1. System instruction
        system_content = provider_request.get("system")
        if system_content:
            if isinstance(system_content, str):
                ir_request["system_instruction"] = system_content
            elif isinstance(system_content, list):
                text_parts = []
                for part in system_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append({"type": "text", "text": part["text"]})
                ir_request["system_instruction"] = text_parts

        # 2. Messages
        messages = provider_request.get("messages", [])
        ir_messages = self.message_ops.p_messages_to_ir(messages)
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

            # Extract tool config from tool_choice
            tc_config = self.tool_ops.p_tool_config_to_ir(tool_choice)
            if tc_config:
                ir_request["tool_config"] = tc_config

        # 5. Generation config
        gen_config = self.config_ops.p_generation_config_to_ir(provider_request)
        if gen_config:
            ir_request["generation"] = gen_config

        # 6. Reasoning config
        if "thinking" in provider_request:
            ir_request["reasoning"] = self.config_ops.p_reasoning_config_to_ir(
                {"thinking": provider_request["thinking"]}
            )

        # 7. Stream config
        stream = provider_request.get("stream")
        if stream is not None:
            ir_request["stream"] = self.config_ops.p_stream_config_to_ir(
                {"stream": stream}
            )

        return cast(IRRequest, ir_request)

    def response_from_provider(
        self,
        provider_response: dict[str, Any],
        **kwargs: Any,
    ) -> IRResponse:
        """Convert Anthropic Messages API response to IRResponse.

        Anthropic returns a single message (not choices list).
        We wrap it as ``choices[0]``.

        Args:
            provider_response: Anthropic response dict (or SDK object).

        Returns:
            IR response.
        """
        provider_response = self._normalize(provider_response)

        # Convert the response message to IR
        ir_message = self.message_ops._p_message_to_ir(provider_response)

        # Map stop_reason to IR finish_reason
        stop_reason_val = provider_response.get("stop_reason")
        reason_map: dict[str, str] = {
            "end_turn": "stop",
            "max_tokens": "length",
            "tool_use": "tool_calls",
            "stop_sequence": "stop",
            "refusal": "refusal",
        }

        finish_reason = (
            reason_map.get(str(stop_reason_val), "stop") if stop_reason_val else "stop"
        )
        choice_info: dict[str, Any] = {
            "index": 0,
            "message": ir_message,
            "finish_reason": {"reason": finish_reason},
        }

        if provider_response.get("stop_sequence"):
            choice_info["finish_reason"]["stop_sequence"] = provider_response[
                "stop_sequence"
            ]

        ir_response: dict[str, Any] = {
            "id": provider_response.get("id", ""),
            "object": "response",
            "created": int(time.time()),  # Anthropic doesn't provide timestamp
            "model": provider_response.get("model", ""),
            "choices": [choice_info],
        }

        # Usage
        p_usage = provider_response.get("usage")
        if p_usage:
            input_tokens = p_usage.get("input_tokens") or 0
            output_tokens = p_usage.get("output_tokens") or 0
            usage_info: dict[str, Any] = {
                "prompt_tokens": input_tokens,
                "completion_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
            }

            if "cache_read_input_tokens" in p_usage:
                usage_info["cache_read_tokens"] = p_usage["cache_read_input_tokens"]

            ir_response["usage"] = usage_info

        return cast(IRResponse, ir_response)

    def response_to_provider(
        self,
        ir_response: IRResponse,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convert IRResponse to Anthropic Messages API response.

        Args:
            ir_response: IR response.

        Returns:
            Anthropic response dict.
        """
        # Anthropic response is a single message
        provider_response: dict[str, Any] = {
            "id": ir_response.get("id", ""),
            "type": "message",
            "model": ir_response.get("model", ""),
            "content": [],
        }

        # Get the first choice (Anthropic only has one)
        choices = ir_response.get("choices", [])
        if choices:
            choice = choices[0]
            message = choice.get("message")
            if message:
                provider_response["role"] = message.get("role", "assistant")

                content_parts = message.get("content", [])
                anthropic_content: list[dict[str, Any]] = []

                for part in content_parts:
                    if is_text_part(part):
                        anthropic_content.append(self.content_ops.ir_text_to_p(part))
                    elif is_tool_call_part(part):
                        anthropic_content.append(self.tool_ops.ir_tool_call_to_p(part))
                    elif is_reasoning_part(part):
                        anthropic_content.append(
                            self.content_ops.ir_reasoning_to_p(part)
                        )

                provider_response["content"] = anthropic_content

            # Map finish_reason back to stop_reason
            finish_reason = choice.get("finish_reason", {})
            reason = finish_reason.get("reason", "stop")
            reason_map = {
                "stop": "end_turn",
                "length": "max_tokens",
                "tool_calls": "tool_use",
                "content_filter": "end_turn",
                "refusal": "refusal",
            }
            provider_response["stop_reason"] = reason_map.get(reason, "end_turn")

            if "stop_sequence" in finish_reason:
                provider_response["stop_sequence"] = finish_reason["stop_sequence"]

        # Usage
        ir_usage = ir_response.get("usage")
        if ir_usage:
            usage: dict[str, Any] = {
                "input_tokens": ir_usage.get("prompt_tokens") or 0,
                "output_tokens": ir_usage.get("completion_tokens") or 0,
            }

            if "cache_read_tokens" in ir_usage:
                usage["cache_read_input_tokens"] = ir_usage["cache_read_tokens"]

            provider_response["usage"] = usage

        return provider_response

    def messages_to_provider(
        self,
        messages: Iterable[Message | ExtensionItem],
        **kwargs: Any,
    ) -> tuple[list[Any], list[str]]:
        """Convert IR message list to Anthropic message format.

        Delegates to message_ops.

        Args:
            messages: IR messages (may contain ExtensionItems).

        Returns:
            Tuple of (converted messages, warnings).
        """
        return self.message_ops.ir_messages_to_p(messages, **kwargs)

    def messages_from_provider(
        self,
        provider_messages: list[Any],
        **kwargs: Any,
    ) -> list[Message | ExtensionItem]:
        """Convert Anthropic messages to IR message list.

        Delegates to message_ops.

        Args:
            provider_messages: Anthropic messages.

        Returns:
            IR messages.
        """
        return self.message_ops.p_messages_to_ir(provider_messages, **kwargs)

    # ==================== Stream Support ====================

    def stream_response_from_provider(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None = None,
    ) -> list[IRStreamEvent]:
        """Convert an Anthropic SSE event to IR stream events.

        Anthropic SSE event types:
        - ``message_start`` → extract usage (optional)
        - ``content_block_start`` → ToolCallStartEvent (if tool_use)
        - ``content_block_delta`` → TextDeltaEvent or ToolCallDeltaEvent
        - ``content_block_stop`` → ignored
        - ``message_delta`` → FinishEvent (contains stop_reason)
        - ``message_stop`` → ignored
        - ``ping`` → ignored

        Args:
            chunk: Anthropic SSE event dict (or SDK object).

        Returns:
            List of IR stream events extracted from the event.
        """
        chunk = self._normalize(chunk)
        events: list[IRStreamEvent] = []

        event_type = chunk.get("type", "")

        if event_type == "message_start":
            message = chunk.get("message", {})

            # Emit StreamStartEvent when context is provided
            if context is not None:
                response_id = message.get("id", "")
                model = message.get("model", "")
                context.response_id = response_id
                context.model = model
                context.mark_started()
                events.append(
                    StreamStartEvent(
                        type="stream_start",
                        response_id=response_id,
                        model=model,
                    )
                )

            # Extract initial usage if available
            usage = message.get("usage")
            if usage:
                events.append(
                    UsageEvent(
                        type="usage",
                        usage={
                            "prompt_tokens": usage.get("input_tokens") or 0,
                            "completion_tokens": 0,
                            "total_tokens": usage.get("input_tokens") or 0,
                        },
                    )
                )

        elif event_type == "content_block_start":
            content_block = chunk.get("content_block", {})
            block_type = content_block.get("type", "")
            block_index = chunk.get("index", 0)

            # Emit ContentBlockStartEvent when context is provided
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

                # Register tool call in context
                if context is not None:
                    context.register_tool_call(tool_call_id, tool_name)

                events.append(
                    ToolCallStartEvent(
                        type="tool_call_start",
                        tool_call_id=tool_call_id,
                        tool_name=tool_name,
                    )
                )

        elif event_type == "content_block_delta":
            delta = chunk.get("delta", {})
            delta_type = delta.get("type", "")

            if delta_type == "text_delta":
                events.append(
                    TextDeltaEvent(
                        type="text_delta",
                        text=delta.get("text", ""),
                    )
                )
            elif delta_type == "input_json_delta":
                # Try to get tool_call_id from context if available
                tool_call_id = ""
                if context is not None and context.tool_call_id_map:
                    # Use the last registered tool_call_id (current block)
                    tool_call_id = list(context.tool_call_id_map.keys())[-1]

                events.append(
                    ToolCallDeltaEvent(
                        type="tool_call_delta",
                        tool_call_id=tool_call_id,
                        arguments_delta=delta.get("partial_json", ""),
                    )
                )
            elif delta_type == "thinking_delta":
                # Thinking deltas map to ReasoningDeltaEvent
                events.append(
                    ReasoningDeltaEvent(
                        type="reasoning_delta",
                        reasoning=delta.get("thinking", ""),
                    )
                )
            elif delta_type == "signature_delta":
                # Signature deltas for thinking block verification
                events.append(
                    ReasoningDeltaEvent(
                        type="reasoning_delta",
                        reasoning="",
                        signature=delta.get("signature", ""),
                    )
                )

        elif event_type == "content_block_stop":
            # Emit ContentBlockEndEvent when context is provided
            if context is not None:
                block_index = chunk.get("index", 0)
                events.append(
                    ContentBlockEndEvent(
                        type="content_block_end",
                        block_index=block_index,
                    )
                )

        elif event_type == "message_delta":
            delta = chunk.get("delta", {})
            stop_reason = delta.get("stop_reason")

            if stop_reason:
                reason_map = {
                    "end_turn": "stop",
                    "max_tokens": "length",
                    "tool_use": "tool_calls",
                    "stop_sequence": "stop",
                }
                events.append(
                    FinishEvent(
                        type="finish",
                        finish_reason={"reason": reason_map.get(stop_reason, "stop")},
                    )
                )

            # Final usage
            usage = chunk.get("usage")
            if usage:
                input_tokens = usage.get("input_tokens") or 0
                output_tokens = usage.get("output_tokens") or 0
                events.append(
                    UsageEvent(
                        type="usage",
                        usage={
                            "prompt_tokens": input_tokens,
                            "completion_tokens": output_tokens,
                            "total_tokens": input_tokens + output_tokens,
                        },
                    )
                )

        elif event_type == "message_stop":
            # Emit StreamEndEvent when context is provided
            if context is not None:
                context.mark_ended()
                events.append(StreamEndEvent(type="stream_end"))

        # ping → ignored

        return events

    def stream_response_to_provider(
        self,
        event: IRStreamEvent,
        context: StreamContext | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert an IR stream event to an Anthropic SSE event.

        Args:
            event: IR stream event.

        Returns:
            Anthropic SSE event dict.
        """
        if is_stream_start_event(event):
            # Store metadata in context if provided
            if context is not None:
                context.response_id = event["response_id"]
                context.model = event["model"]
                context.mark_started()
            return {
                "type": "message_start",
                "message": {
                    "id": event["response_id"],
                    "type": "message",
                    "role": "assistant",
                    "model": event["model"],
                    "content": [],
                    "stop_reason": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            }

        elif is_stream_end_event(event):
            if context is not None:
                context.mark_ended()
            return {"type": "message_stop"}

        elif is_content_block_start_event(event):
            block_index = event["block_index"]
            block_type = event["block_type"]

            if context is not None:
                context.next_block_index()

            if block_type == "text":
                return {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "text", "text": ""},
                }
            elif block_type == "thinking":
                return {
                    "type": "content_block_start",
                    "index": block_index,
                    "content_block": {"type": "thinking", "thinking": ""},
                }
            else:
                # tool_use content_block_start is handled by ToolCallStartEvent
                return {}

        elif is_content_block_end_event(event):
            return {
                "type": "content_block_stop",
                "index": event["block_index"],
            }

        elif is_text_delta_event(event):
            result: dict[str, Any] = {
                "type": "content_block_delta",
                "delta": {
                    "type": "text_delta",
                    "text": event["text"],
                },
            }
            if context is not None and context.current_block_index >= 0:
                result["index"] = context.current_block_index
            return result

        elif is_reasoning_delta_event(event):
            signature = event.get("signature")
            rd_result: dict[str, Any]
            if signature is not None:
                rd_result = {
                    "type": "content_block_delta",
                    "delta": {
                        "type": "signature_delta",
                        "signature": signature,
                    },
                }
            else:
                rd_result = {
                    "type": "content_block_delta",
                    "delta": {
                        "type": "thinking_delta",
                        "thinking": event["reasoning"],
                    },
                }
            if context is not None and context.current_block_index >= 0:
                rd_result["index"] = context.current_block_index
            return rd_result

        elif is_tool_call_start_event(event):
            result2: dict[str, Any] = {
                "type": "content_block_start",
                "content_block": {
                    "type": "tool_use",
                    "id": event["tool_call_id"],
                    "name": event["tool_name"],
                    "input": {},
                },
            }
            if context is not None and context.current_block_index >= 0:
                result2["index"] = context.current_block_index
            return result2

        elif is_tool_call_delta_event(event):
            result3: dict[str, Any] = {
                "type": "content_block_delta",
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": event["arguments_delta"],
                },
            }
            if context is not None and context.current_block_index >= 0:
                result3["index"] = context.current_block_index
            return result3

        elif is_finish_event(event):
            reason = event["finish_reason"]["reason"]
            reason_map = {
                "stop": "end_turn",
                "length": "max_tokens",
                "tool_calls": "tool_use",
                "content_filter": "end_turn",
            }
            return {
                "type": "message_delta",
                "delta": {
                    "stop_reason": reason_map.get(reason, "end_turn"),
                },
            }

        elif is_usage_event(event):
            usage = event["usage"]
            return {
                "type": "message_delta",
                "usage": {
                    "input_tokens": usage.get("prompt_tokens") or 0,
                    "output_tokens": usage.get("completion_tokens") or 0,
                },
            }

        return {}
