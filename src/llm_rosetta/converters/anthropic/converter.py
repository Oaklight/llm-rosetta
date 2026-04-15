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
from collections.abc import Sequence
from typing import Any

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
from ..base.context import ConversionContext, StreamContext
from ..base.tools import fix_orphaned_tool_calls_ir, strip_orphaned_tool_config
from ._constants import (
    ANTHROPIC_REASON_FROM_PROVIDER,
    ANTHROPIC_REASON_TO_PROVIDER,
    AnthropicEventType,
)
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
        *,
        context: ConversionContext | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], list[str]]:
        """Convert IRRequest to Anthropic Messages API request parameters.

        Orchestrates all Ops classes to build the complete provider request.

        Args:
            ir_request: IR request.

        Returns:
            Tuple of (provider request dict, warnings list).
        """
        ctx = context if context is not None else ConversionContext()
        result: dict[str, Any] = {"model": ir_request["model"]}

        # 1. System instruction → top-level system parameter
        system_instruction = ir_request.get("system_instruction")
        if system_instruction:
            result["system"] = system_instruction

        # 2. Messages — fix orphaned tool_calls/results at IR level before
        #    conversion.  Anthropic strictly requires bidirectional pairing.
        ir_messages = fix_orphaned_tool_calls_ir(ir_request.get("messages", []))
        ctx.warnings.extend(strip_orphaned_tool_config(ir_request))

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
        ctx.warnings.extend(msg_warnings)
        result["messages"] = converted_msgs

        # 3. Generation config (must come before tools since max_tokens is required)
        gen_config = ir_request.get("generation")
        if gen_config:
            gen_fields = self.config_ops.ir_generation_config_to_p(gen_config)
            result.update(gen_fields)
        else:
            # Anthropic requires max_tokens
            result["max_tokens"] = 4096

        # 4-6. Tools, tool choice, tool config
        self._apply_tool_config(ir_request, result, ctx)

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

        return result, ctx.warnings

    def request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext | None = None,
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
                        text_parts.append(part["text"])
                if text_parts:
                    ir_request["system_instruction"] = " ".join(text_parts)

        # 2. Messages
        messages = provider_request.get("messages", [])
        ir_messages = self.message_ops.p_messages_to_ir(messages)
        ir_request["messages"] = ir_messages

        # 3. Tools
        tools = provider_request.get("tools")
        if tools:
            ir_request["tools"] = self._convert_tools_from_p(tools)

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
        reasoning = self.config_ops.p_reasoning_config_to_ir(provider_request)
        if reasoning:
            ir_request["reasoning"] = reasoning

        # 7. Stream config
        stream = provider_request.get("stream")
        if stream is not None:
            ir_request["stream"] = self.config_ops.p_stream_config_to_ir(
                {"stream": stream}
            )

        return self._validate_ir_request(ir_request)

    def response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext | None = None,
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

        finish_reason = (
            ANTHROPIC_REASON_FROM_PROVIDER.get(str(stop_reason_val), "stop")
            if stop_reason_val
            else "stop"
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
            ir_response["usage"] = self._build_ir_usage(p_usage)

        # Preserve mode: capture extra fields for lossless round-trip
        ctx = context if context is not None else ConversionContext()
        if ctx.metadata_mode == "preserve":
            self._capture_preserve_metadata(provider_response, ctx)

        return self._validate_ir_response(ir_response)

    def response_to_provider(
        self,
        ir_response: IRResponse,
        *,
        context: ConversionContext | None = None,
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
            provider_response["stop_reason"] = ANTHROPIC_REASON_TO_PROVIDER.get(
                reason, "end_turn"
            )

            if "stop_sequence" in finish_reason:
                provider_response["stop_sequence"] = finish_reason["stop_sequence"]

        # Usage
        ir_usage = ir_response.get("usage")
        if ir_usage:
            provider_response["usage"] = self._build_provider_usage(ir_usage)

        # Preserve mode: inject captured extra fields
        ctx = context if context is not None else ConversionContext()
        if ctx.metadata_mode == "preserve":
            self._apply_preserve_metadata(provider_response, ctx)

        return provider_response

    # ------------------------------------------------------------------
    # Cross-provider consistency helpers
    # ------------------------------------------------------------------

    def _apply_tool_config(
        self,
        ir_request: IRRequest,
        result: dict[str, Any],
        ctx: ConversionContext,
    ) -> None:
        """Apply tools, tool_choice, and tool_config to provider request."""
        tools = ir_request.get("tools")
        if tools:
            result["tools"] = [self.tool_ops.ir_tool_definition_to_p(t) for t in tools]

        tool_choice = ir_request.get("tool_choice")
        if tool_choice:
            result["tool_choice"] = self.tool_ops.ir_tool_choice_to_p(tool_choice)

        tool_config = ir_request.get("tool_config")
        if tool_config:
            tc_fields = self.tool_ops.ir_tool_config_to_p(tool_config)
            if tc_fields:
                if "tool_choice" not in result:
                    result["tool_choice"] = {"type": "auto"}
                result["tool_choice"].update(tc_fields)
            if "max_calls" in tool_config:
                ctx.warnings.append(
                    "Anthropic does not support max_tool_calls, ignored"
                )

    @staticmethod
    def _build_ir_usage(p_usage: dict[str, Any]) -> dict[str, Any]:
        """Build IR usage dict from Anthropic usage."""
        input_tokens = p_usage.get("input_tokens") or 0
        output_tokens = p_usage.get("output_tokens") or 0
        usage_info: dict[str, Any] = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens,
        }
        if "cache_read_input_tokens" in p_usage:
            usage_info["cache_read_tokens"] = p_usage["cache_read_input_tokens"]
        return usage_info

    @staticmethod
    def _build_provider_usage(ir_usage: dict[str, Any]) -> dict[str, Any]:
        """Build Anthropic usage dict from IR usage."""
        usage: dict[str, Any] = {
            "input_tokens": ir_usage.get("prompt_tokens") or 0,
            "output_tokens": ir_usage.get("completion_tokens") or 0,
        }
        if "cache_read_tokens" in ir_usage:
            usage["cache_read_input_tokens"] = ir_usage["cache_read_tokens"]
        return usage

    def _convert_tools_from_p(self, tools: list[Any]) -> list[Any]:
        """Convert provider tool definitions to IR."""
        ir_tools = []
        for t in tools:
            try:
                ir_tools.append(self.tool_ops.p_tool_definition_to_ir(t))
            except Exception as e:
                tool_type = (
                    t.get("type", "unknown")
                    if isinstance(t, dict)
                    else type(t).__name__
                )
                tool_name = t.get("name", "unnamed") if isinstance(t, dict) else str(t)
                raise ValueError(
                    f"Unsupported tool type={tool_type!r} name={tool_name!r}: {e}"
                ) from e
        return ir_tools

    @staticmethod
    def _capture_preserve_metadata(
        provider_response: dict[str, Any],
        ctx: ConversionContext,
    ) -> None:
        """Capture extra fields from provider response for lossless round-trip."""
        p_usage = provider_response.get("usage")
        _ANTHROPIC_CORE_KEYS = {
            "id",
            "type",
            "role",
            "content",
            "model",
            "stop_reason",
            "stop_sequence",
            "usage",
        }
        extras = {
            k: v for k, v in provider_response.items() if k not in _ANTHROPIC_CORE_KEYS
        }
        _USAGE_CORE_KEYS = {
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
        }
        if p_usage:
            usage_extras = {
                k: v for k, v in p_usage.items() if k not in _USAGE_CORE_KEYS
            }
            if usage_extras:
                extras["_usage_extras"] = usage_extras
        if extras:
            ctx.store_response_extras(extras)

        content_blocks = provider_response.get("content", [])
        items_meta: list[dict[str, Any]] = []
        for block in content_blocks:
            if not isinstance(block, dict):
                items_meta.append({})
                continue
            meta: dict[str, Any] = {}
            if "citations" in block:
                meta["citations"] = block["citations"]
            items_meta.append(meta)
        if any(m for m in items_meta):
            ctx.store_output_items_meta(items_meta)

    @staticmethod
    def _apply_preserve_metadata(
        provider_response: dict[str, Any],
        ctx: ConversionContext,
    ) -> None:
        """Re-inject captured metadata fields in preserve mode."""
        echo = ctx.get_echo_fields()
        usage_extras = echo.pop("_usage_extras", None)
        if usage_extras and "usage" in provider_response:
            provider_response["usage"].update(usage_extras)
        _CORE_KEYS = {
            "id",
            "type",
            "role",
            "content",
            "model",
            "stop_reason",
            "stop_sequence",
            "usage",
        }
        for k, v in echo.items():
            if k not in _CORE_KEYS:
                provider_response[k] = v

        items_meta = ctx.get_output_items_meta()
        content = provider_response.get("content", [])
        for i, meta in enumerate(items_meta):
            if i >= len(content):
                break
            for k, v in meta.items():
                content[i][k] = v

    def messages_to_provider(
        self,
        messages: Sequence[Message | ExtensionItem],
        *,
        context: ConversionContext | None = None,
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
        *,
        context: ConversionContext | None = None,
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

    # --- from_provider ---

    def stream_response_from_provider(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None = None,
    ) -> list[IRStreamEvent]:
        """Convert an Anthropic SSE event to IR stream events.

        Args:
            chunk: Anthropic SSE event dict (or SDK object).

        Returns:
            List of IR stream events extracted from the event.
        """
        chunk = self._normalize(chunk)
        events: list[IRStreamEvent] = []

        event_type = chunk.get("type", "")
        handler_name = self._FROM_P_DISPATCH.get(event_type)
        if handler_name is not None:
            getattr(self, handler_name)(chunk, context, events)

        return events

    def _handle_message_start_from_p(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Handle message_start → StreamStartEvent + optional UsageEvent."""
        message = chunk.get("message", {})

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

    def _handle_content_block_start_from_p(
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

            if context is not None:
                context.register_tool_call(tool_call_id, tool_name)

            start_evt = ToolCallStartEvent(
                type="tool_call_start",
                tool_call_id=tool_call_id,
                tool_name=tool_name,
            )
            if context is not None:
                start_evt["tool_call_index"] = len(context._tool_call_order) - 1
            events.append(start_evt)

    def _handle_content_block_delta_from_p(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Handle content_block_delta → TextDeltaEvent, ToolCallDeltaEvent, or ReasoningDeltaEvent."""
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
            tool_call_id = ""
            if context is not None and context.tool_call_id_map:
                tool_call_id = list(context.tool_call_id_map.keys())[-1]

            partial_json = delta.get("partial_json", "")
            delta_evt = ToolCallDeltaEvent(
                type="tool_call_delta",
                tool_call_id=tool_call_id,
                arguments_delta=partial_json,
            )
            if (
                context is not None
                and tool_call_id
                and tool_call_id in context._tool_call_order
            ):
                delta_evt["tool_call_index"] = context._tool_call_order.index(
                    tool_call_id
                )
            events.append(delta_evt)

            if context is not None and tool_call_id:
                context.append_tool_call_args(tool_call_id, partial_json)
        elif delta_type == "thinking_delta":
            events.append(
                ReasoningDeltaEvent(
                    type="reasoning_delta",
                    reasoning=delta.get("thinking", ""),
                )
            )
        elif delta_type == "signature_delta":
            events.append(
                ReasoningDeltaEvent(
                    type="reasoning_delta",
                    reasoning="",
                    signature=delta.get("signature", ""),
                )
            )

    def _handle_content_block_stop_from_p(
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

    def _handle_message_delta_from_p(
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

        if stop_reason:
            events.append(
                FinishEvent(
                    type="finish",
                    finish_reason={
                        "reason": ANTHROPIC_REASON_FROM_PROVIDER.get(
                            stop_reason, "stop"
                        )
                    },
                )
            )

    def _handle_message_stop_from_p(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Handle message_stop → StreamEndEvent."""
        if context is not None:
            context.mark_ended()
            events.append(StreamEndEvent(type="stream_end"))

    _FROM_P_DISPATCH: dict[str, str] = {
        AnthropicEventType.MESSAGE_START: "_handle_message_start_from_p",
        AnthropicEventType.CONTENT_BLOCK_START: "_handle_content_block_start_from_p",
        AnthropicEventType.CONTENT_BLOCK_DELTA: "_handle_content_block_delta_from_p",
        AnthropicEventType.CONTENT_BLOCK_STOP: "_handle_content_block_stop_from_p",
        AnthropicEventType.MESSAGE_DELTA: "_handle_message_delta_from_p",
        AnthropicEventType.MESSAGE_STOP: "_handle_message_stop_from_p",
    }

    # --- to_provider ---

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
        handler_name = self._TO_P_DISPATCH.get(event.get("type", ""))
        if handler_name is None:
            return {}
        return getattr(self, handler_name)(event, context)

    def _handle_stream_start_to_p(
        self, event: StreamStartEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle StreamStartEvent → message_start."""
        if context is not None:
            context.response_id = event["response_id"]
            context.model = event["model"]
            context.mark_started()
        return {
            "type": AnthropicEventType.MESSAGE_START,
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

    def _handle_stream_end_to_p(
        self, event: StreamEndEvent, context: StreamContext | None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Handle StreamEndEvent → message_stop (with optional pending finish flush)."""
        results: list[dict[str, Any]] = []
        if context is not None:
            # Flush any buffered finish that never got a UsageEvent
            if context.pending_finish is not None:
                results.append(
                    {
                        "type": AnthropicEventType.MESSAGE_DELTA,
                        "delta": context.pending_finish,
                        "usage": {"output_tokens": 0},
                    }
                )
                context.pending_finish = None
            context.mark_ended()
        results.append({"type": AnthropicEventType.MESSAGE_STOP})
        return results if len(results) > 1 else results[0]

    def _handle_content_block_start_to_p(
        self, event: ContentBlockStartEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ContentBlockStartEvent → content_block_start."""
        block_index = event["block_index"]
        block_type = event["block_type"]

        if context is not None:
            context.next_block_index()

        if block_type == "text":
            return {
                "type": AnthropicEventType.CONTENT_BLOCK_START,
                "index": block_index,
                "content_block": {"type": "text", "text": ""},
            }
        elif block_type == "thinking":
            return {
                "type": AnthropicEventType.CONTENT_BLOCK_START,
                "index": block_index,
                "content_block": {"type": "thinking", "thinking": ""},
            }
        else:
            return {}

    def _handle_content_block_end_to_p(
        self, event: ContentBlockEndEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ContentBlockEndEvent → content_block_stop."""
        return {
            "type": AnthropicEventType.CONTENT_BLOCK_STOP,
            "index": event["block_index"],
        }

    def _handle_text_delta_to_p(
        self, event: TextDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Handle TextDeltaEvent → content_block_delta (with synthetic start if needed)."""
        result: dict[str, Any] = {
            "type": AnthropicEventType.CONTENT_BLOCK_DELTA,
            "delta": {
                "type": "text_delta",
                "text": event["text"],
            },
        }
        if context is not None:
            if context.current_block_index < 0:
                context.next_block_index()
                result["index"] = context.current_block_index
                return [
                    {
                        "type": AnthropicEventType.CONTENT_BLOCK_START,
                        "index": context.current_block_index,
                        "content_block": {"type": "text", "text": ""},
                    },
                    result,
                ]
            result["index"] = context.current_block_index
        return result

    def _handle_reasoning_delta_to_p(
        self, event: ReasoningDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Handle ReasoningDeltaEvent → content_block_delta (thinking or signature)."""
        signature = event.get("signature")
        rd_result: dict[str, Any]
        if signature is not None:
            rd_result = {
                "type": AnthropicEventType.CONTENT_BLOCK_DELTA,
                "delta": {
                    "type": "signature_delta",
                    "signature": signature,
                },
            }
        else:
            rd_result = {
                "type": AnthropicEventType.CONTENT_BLOCK_DELTA,
                "delta": {
                    "type": "thinking_delta",
                    "thinking": event["reasoning"],
                },
            }
        if context is not None:
            if context.current_block_index < 0:
                context.next_block_index()
                rd_result["index"] = context.current_block_index
                return [
                    {
                        "type": AnthropicEventType.CONTENT_BLOCK_START,
                        "index": context.current_block_index,
                        "content_block": {"type": "thinking", "thinking": ""},
                    },
                    rd_result,
                ]
            rd_result["index"] = context.current_block_index
        return rd_result

    def _handle_tool_call_start_to_p(
        self, event: ToolCallStartEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ToolCallStartEvent → content_block_start for tool_use."""
        result: dict[str, Any] = {
            "type": AnthropicEventType.CONTENT_BLOCK_START,
            "content_block": {
                "type": "tool_use",
                "id": event["tool_call_id"],
                "name": event["tool_name"],
                "input": {},
            },
        }
        if context is not None:
            if context.current_block_index < 0:
                context.next_block_index()
            result["index"] = context.current_block_index
        return result

    def _handle_tool_call_delta_to_p(
        self, event: ToolCallDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ToolCallDeltaEvent → content_block_delta with input_json_delta."""
        result: dict[str, Any] = {
            "type": AnthropicEventType.CONTENT_BLOCK_DELTA,
            "delta": {
                "type": "input_json_delta",
                "partial_json": event["arguments_delta"],
            },
        }
        if context is not None:
            if context.current_block_index < 0:
                context.next_block_index()
            result["index"] = context.current_block_index
        return result

    def _handle_finish_to_p(
        self, event: FinishEvent, context: StreamContext | None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Handle FinishEvent → buffered message_delta + optional content_block_stop."""
        reason = event["finish_reason"]["reason"]
        stop_reason = ANTHROPIC_REASON_TO_PROVIDER.get(reason, "end_turn")
        if context is not None:
            results: list[dict[str, Any]] = []
            if context.current_block_index >= 0:
                results.append(
                    {
                        "type": AnthropicEventType.CONTENT_BLOCK_STOP,
                        "index": context.current_block_index,
                    }
                )
            context.pending_finish = {"stop_reason": stop_reason}
            return results if results else {}
        return {
            "type": AnthropicEventType.MESSAGE_DELTA,
            "delta": {"stop_reason": stop_reason},
            "usage": {"output_tokens": 0},
        }

    def _handle_usage_to_p(
        self, event: UsageEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle UsageEvent → message_delta (merged with pending finish)."""
        usage = event["usage"]
        output_tokens = usage.get("completion_tokens") or 0
        delta: dict[str, Any] = {}
        if context is not None and context.pending_finish is not None:
            delta = context.pending_finish
            context.pending_finish = None
        return {
            "type": AnthropicEventType.MESSAGE_DELTA,
            "delta": delta,
            "usage": {"output_tokens": output_tokens},
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
