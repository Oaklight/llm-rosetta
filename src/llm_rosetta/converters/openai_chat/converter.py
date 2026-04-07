"""
LLM-Rosetta - OpenAI Chat Completions Converter

Top-level converter implementing the 6 explicit interfaces + 2 stream methods.
Composes ContentOps, ToolOps, MessageOps, and ConfigOps for full bidirectional
conversion between IR and OpenAI Chat Completions API format.
"""

from collections.abc import Sequence
from typing import Any

from ...types.ir import (
    ExtensionItem,
    Message,
    is_text_part,
    is_tool_call_part,
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
from ._constants import OPENAI_CHAT_REASON_FROM_PROVIDER, OPENAI_CHAT_REASON_TO_PROVIDER
from .config_ops import OpenAIChatConfigOps
from .content_ops import OpenAIChatContentOps
from .message_ops import OpenAIChatMessageOps
from .tool_ops import OpenAIChatToolOps


class OpenAIChatConverter(BaseConverter):
    """OpenAI Chat Completions API converter.

    Implements the 6 explicit conversion interfaces defined by BaseConverter,
    plus 2 stream methods for SSE chunk-level conversion.

    Uses composition of Ops classes for modular, testable conversion logic.
    """

    content_ops_class = OpenAIChatContentOps
    tool_ops_class = OpenAIChatToolOps
    message_ops_class = OpenAIChatMessageOps
    config_ops_class = OpenAIChatConfigOps

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
        """Convert IRRequest to OpenAI Chat Completions request parameters.

        Orchestrates all Ops classes to build the complete provider request.

        Args:
            ir_request: IR request.

        Returns:
            Tuple of (provider request dict, warnings list).
        """
        ctx = context if context is not None else ConversionContext()
        result: dict[str, Any] = {"model": ir_request["model"]}

        # 1. System instruction → system message
        messages: list[dict[str, Any]] = []
        system_instruction = ir_request.get("system_instruction")
        if system_instruction:
            messages.append({"role": "system", "content": system_instruction})

        # 2. Messages — fix orphaned tool_calls at IR level before conversion.
        # OpenAI Chat API strictly requires every tool_call_id to have a
        # matching role:tool response.  Other providers (Anthropic, Google)
        # are lenient, so cross-format conversions may carry orphaned
        # tool_calls from interrupted sessions.
        ir_messages = fix_orphaned_tool_calls_ir(ir_request.get("messages", []))
        ctx.warnings.extend(strip_orphaned_tool_config(ir_request))
        converted_msgs, msg_warnings = self.message_ops.ir_messages_to_p(ir_messages)
        messages.extend(converted_msgs)
        ctx.warnings.extend(msg_warnings)
        result["messages"] = messages

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
            if "max_calls" in tool_config:
                ctx.warnings.append(
                    "OpenAI Chat does not support max_tool_calls, ignored"
                )

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

        return result, ctx.warnings

    def request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext | None = None,
        **kwargs: Any,
    ) -> IRRequest:
        """Convert OpenAI Chat Completions request to IRRequest.

        Args:
            provider_request: OpenAI request dict (or SDK object).

        Returns:
            IR request.
        """
        provider_request = self._normalize(provider_request)

        ir_request: dict[str, Any] = {
            "model": provider_request.get("model", ""),
            "messages": [],
        }

        # 1. Messages - separate system messages as system_instruction
        messages = provider_request.get("messages", [])
        ir_messages: list[dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    ir_request["system_instruction"] = content
                elif isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append(part["text"])
                    if text_parts:
                        ir_request["system_instruction"] = " ".join(text_parts)
            else:
                converted = self.message_ops._p_message_to_ir(msg)
                if converted:
                    ir_messages.append(converted)

        ir_request["messages"] = ir_messages

        # 2. Tools
        tools = provider_request.get("tools")
        if tools:
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
                    tool_name = (
                        (t.get("function", {}).get("name") or t.get("name", "unnamed"))
                        if isinstance(t, dict)
                        else str(t)
                    )
                    raise ValueError(
                        f"Unsupported tool type={tool_type!r} name={tool_name!r}: {e}"
                    ) from e
            ir_request["tools"] = ir_tools

        # 3. Tool choice
        tool_choice = provider_request.get("tool_choice")
        if tool_choice is not None:
            ir_request["tool_choice"] = self.tool_ops.p_tool_choice_to_ir(tool_choice)

        # 4. Tool config (parallel_tool_calls)
        parallel_tool_calls = provider_request.get("parallel_tool_calls")
        if parallel_tool_calls is not None:
            ir_request["tool_config"] = self.tool_ops.p_tool_config_to_ir(
                {"parallel_tool_calls": parallel_tool_calls}
            )

        # 5. Generation config
        gen_config = self.config_ops.p_generation_config_to_ir(provider_request)
        if gen_config:
            ir_request["generation"] = gen_config

        # 6. Response format
        resp_format = provider_request.get("response_format")
        if resp_format:
            ir_request["response_format"] = self.config_ops.p_response_format_to_ir(
                resp_format
            )

        # 7. Reasoning config
        reasoning_effort = provider_request.get("reasoning_effort")
        if reasoning_effort:
            ir_request["reasoning"] = self.config_ops.p_reasoning_config_to_ir(
                {"reasoning_effort": reasoning_effort}
            )

        # 8. Stream config
        stream = provider_request.get("stream")
        stream_options = provider_request.get("stream_options")
        if stream is not None or stream_options:
            ir_request["stream"] = self.config_ops.p_stream_config_to_ir(
                {"stream": stream, "stream_options": stream_options}
            )

        # 9. Cache config
        cache_fields = {}
        if "prompt_cache_key" in provider_request:
            cache_fields["prompt_cache_key"] = provider_request["prompt_cache_key"]
        if "prompt_cache_retention" in provider_request:
            cache_fields["prompt_cache_retention"] = provider_request[
                "prompt_cache_retention"
            ]
        if cache_fields:
            ir_request["cache"] = self.config_ops.p_cache_config_to_ir(cache_fields)

        return self._validate_ir_request(ir_request)

    def response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext | None = None,
        **kwargs: Any,
    ) -> IRResponse:
        """Convert OpenAI Chat Completions response to IRResponse.

        Args:
            provider_response: OpenAI response dict (or SDK object).

        Returns:
            IR response.
        """
        provider_response = self._normalize(provider_response)

        choices = []
        for p_choice in provider_response.get("choices", []):
            message = self.message_ops._p_message_to_ir(
                p_choice.get("message", p_choice.get("delta", {}))
            )

            finish_reason_val = p_choice.get("finish_reason")

            choice_info: dict[str, Any] = {
                "index": p_choice.get("index", 0),
                "message": message,
                "finish_reason": {
                    "reason": OPENAI_CHAT_REASON_FROM_PROVIDER.get(
                        finish_reason_val, "stop"
                    )
                },
            }

            if "logprobs" in p_choice:
                choice_info["logprobs"] = p_choice["logprobs"]

            choices.append(choice_info)

        ir_response: dict[str, Any] = {
            "id": provider_response.get("id", ""),
            "object": "response",
            "created": provider_response.get("created", 0),
            "model": provider_response.get("model", ""),
            "choices": choices,
        }

        # Usage
        p_usage = provider_response.get("usage")
        if p_usage:
            usage_info: dict[str, Any] = {
                "prompt_tokens": p_usage.get("prompt_tokens") or 0,
                "completion_tokens": p_usage.get("completion_tokens") or 0,
                "total_tokens": p_usage.get("total_tokens") or 0,
            }

            p_prompt_details = p_usage.get("prompt_tokens_details")
            if p_prompt_details:
                usage_info["prompt_tokens_details"] = p_prompt_details
                if "cached_tokens" in p_prompt_details:
                    usage_info["cache_read_tokens"] = p_prompt_details["cached_tokens"]

            p_completion_details = p_usage.get("completion_tokens_details")
            if p_completion_details:
                usage_info["completion_tokens_details"] = p_completion_details
                if "reasoning_tokens" in p_completion_details:
                    usage_info["reasoning_tokens"] = p_completion_details[
                        "reasoning_tokens"
                    ]

            ir_response["usage"] = usage_info

        if provider_response.get("service_tier") is not None:
            ir_response["service_tier"] = provider_response["service_tier"]

        if provider_response.get("system_fingerprint") is not None:
            ir_response["system_fingerprint"] = provider_response["system_fingerprint"]

        return self._validate_ir_response(ir_response)

    def response_to_provider(
        self,
        ir_response: IRResponse,
        *,
        context: ConversionContext | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convert IRResponse to OpenAI Chat Completions response.

        Args:
            ir_response: IR response.

        Returns:
            OpenAI response dict.
        """
        provider_response: dict[str, Any] = {
            "id": ir_response.get("id", ""),
            "object": "chat.completion",
            "created": ir_response.get("created", 0),
            "model": ir_response.get("model", ""),
            "choices": [],
        }

        for choice in ir_response.get("choices", []):
            message = choice.get("message")
            if not message:
                continue

            openai_message: dict[str, Any] = {"role": message.get("role", "assistant")}

            content_parts = message.get("content", [])
            text_parts: list[str] = []
            tool_calls: list[dict[str, Any]] = []

            for part in content_parts:
                if is_text_part(part):
                    text_parts.append(part["text"])
                elif is_tool_call_part(part):
                    tool_calls.append(self.tool_ops.ir_tool_call_to_p(part))

            if text_parts:
                openai_message["content"] = " ".join(text_parts)
            elif not tool_calls:
                openai_message["content"] = ""

            if tool_calls:
                openai_message["tool_calls"] = tool_calls
                if not text_parts:
                    openai_message["content"] = None

            reason = choice.get("finish_reason", {}).get("reason", "stop")
            openai_choice: dict[str, Any] = {
                "index": choice.get("index", 0),
                "message": openai_message,
                "finish_reason": OPENAI_CHAT_REASON_TO_PROVIDER.get(reason, "stop"),
            }

            if "logprobs" in choice:
                openai_choice["logprobs"] = choice["logprobs"]

            provider_response["choices"].append(openai_choice)

        # Usage
        ir_usage = ir_response.get("usage")
        if ir_usage:
            usage: dict[str, Any] = {
                "prompt_tokens": ir_usage.get("prompt_tokens") or 0,
                "completion_tokens": ir_usage.get("completion_tokens") or 0,
                "total_tokens": ir_usage.get("total_tokens") or 0,
            }

            if "prompt_tokens_details" in ir_usage:
                usage["prompt_tokens_details"] = ir_usage["prompt_tokens_details"]

            if "completion_tokens_details" in ir_usage:
                usage["completion_tokens_details"] = ir_usage[
                    "completion_tokens_details"
                ]

            if "cache_read_tokens" in ir_usage:
                if "prompt_tokens_details" not in usage:
                    usage["prompt_tokens_details"] = {}
                usage["prompt_tokens_details"]["cached_tokens"] = ir_usage[
                    "cache_read_tokens"
                ]

            if "reasoning_tokens" in ir_usage:
                if "completion_tokens_details" not in usage:
                    usage["completion_tokens_details"] = {}
                usage["completion_tokens_details"]["reasoning_tokens"] = ir_usage[
                    "reasoning_tokens"
                ]

            provider_response["usage"] = usage

        if "service_tier" in ir_response:
            provider_response["service_tier"] = ir_response["service_tier"]

        if "system_fingerprint" in ir_response:
            provider_response["system_fingerprint"] = ir_response["system_fingerprint"]

        return provider_response

    def messages_to_provider(
        self,
        messages: Sequence[Message | ExtensionItem],
        *,
        context: ConversionContext | None = None,
        **kwargs: Any,
    ) -> tuple[list[Any], list[str]]:
        """Convert IR message list to OpenAI Chat message format.

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
        """Convert OpenAI Chat messages to IR message list.

        Delegates to message_ops.

        Args:
            provider_messages: OpenAI Chat messages.

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
        """Convert an OpenAI SSE chunk to IR stream events.

        A single chunk may produce multiple events (e.g., text delta + finish).

        When a ``context`` is provided, lifecycle events (``StreamStartEvent``,
        ``StreamEndEvent``) are emitted and cross-chunk state is tracked.
        Without a context the behaviour is identical to the previous
        implementation (backward compatible).

        Args:
            chunk: OpenAI SSE chunk dict (or SDK object).
            context: Optional stream context for stateful conversions.

        Returns:
            List of IR stream events extracted from the chunk.
        """
        chunk = self._normalize(chunk)
        events: list[IRStreamEvent] = []

        if context is not None and not context.is_started:
            self._handle_stream_start_from_p(chunk, context, events)

        choices = chunk.get("choices", [])
        for p_choice in choices:
            self._handle_choice_from_p(p_choice, context, events)

        usage = chunk.get("usage")
        if usage:
            self._handle_usage_from_p(usage, events)

        self._handle_stream_end_from_p(choices, events, usage, context)

        return events

    def _handle_stream_start_from_p(
        self,
        chunk: dict[str, Any],
        context: StreamContext,
        events: list[IRStreamEvent],
    ) -> None:
        """Emit StreamStartEvent on the first chunk."""
        response_id = chunk.get("id")
        model = chunk.get("model")
        created = chunk.get("created")
        if response_id and model:
            context.response_id = response_id
            context.model = model
            if created is not None:
                context.created = created
            context.mark_started()
            start_event: StreamStartEvent = {
                "type": "stream_start",
                "response_id": response_id,
                "model": model,
            }
            if created is not None:
                start_event["created"] = created
            events.append(start_event)

    def _handle_choice_from_p(
        self,
        p_choice: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Process a single choice from an OpenAI SSE chunk."""
        choice_index = p_choice.get("index", 0)
        delta = p_choice.get("delta", {})

        # Text delta
        content = delta.get("content")
        if content is not None:
            events.append(
                TextDeltaEvent(
                    type="text_delta",
                    text=content,
                    choice_index=choice_index,
                )
            )

        # Reasoning content delta (OpenAI o1/o3 models)
        reasoning_content = delta.get("reasoning_content")
        if reasoning_content is not None:
            events.append(
                ReasoningDeltaEvent(
                    type="reasoning_delta",
                    reasoning=reasoning_content,
                    choice_index=choice_index,
                )
            )

        # Tool call deltas
        tool_calls = delta.get("tool_calls")
        if tool_calls:
            self._handle_tool_calls_from_p(tool_calls, choice_index, context, events)

        # Finish reason
        finish_reason = p_choice.get("finish_reason")
        if finish_reason:
            self._handle_finish_reason_from_p(
                finish_reason, choice_index, context, events
            )

    def _handle_tool_calls_from_p(
        self,
        tool_calls: list[dict[str, Any]],
        choice_index: int,
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Process tool call deltas from a choice."""
        for tc in tool_calls:
            tc_func = tc.get("function", {})
            tc_id = tc.get("id")
            tc_index = tc.get("index")

            if tc_id:
                start_event_tc = ToolCallStartEvent(
                    type="tool_call_start",
                    tool_call_id=tc_id,
                    tool_name=tc_func.get("name", ""),
                    choice_index=choice_index,
                )
                if tc_index is not None:
                    start_event_tc["tool_call_index"] = tc_index
                events.append(start_event_tc)

                if context is not None:
                    context.register_tool_call(tc_id, tc_func.get("name", ""))

            # Resolve the effective call ID for delta-only chunks
            # (they carry index but no id).
            effective_tc_id = tc_id
            if not effective_tc_id and tc_index is not None and context is not None:
                order = context._tool_call_order
                if 0 <= tc_index < len(order):
                    effective_tc_id = order[tc_index]

            arguments = tc_func.get("arguments")
            if arguments:
                delta_event = ToolCallDeltaEvent(
                    type="tool_call_delta",
                    tool_call_id=effective_tc_id or "",
                    arguments_delta=arguments,
                    choice_index=choice_index,
                )
                if tc_index is not None:
                    delta_event["tool_call_index"] = tc_index
                events.append(delta_event)

                if context is not None and effective_tc_id:
                    context.append_tool_call_args(effective_tc_id, arguments)

    def _handle_finish_reason_from_p(
        self,
        finish_reason: str,
        choice_index: int,
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Emit ContentBlockEndEvent (if needed) and FinishEvent."""
        # Close any open content block before emitting FinishEvent.
        # OpenAI doesn't have an explicit content-block-end concept,
        # but downstream formats (e.g. Anthropic) require it.
        if context is not None and context.current_block_index >= 0:
            events.append(
                ContentBlockEndEvent(
                    type="content_block_end",
                    block_index=context.current_block_index,
                )
            )

        events.append(
            FinishEvent(
                type="finish",
                finish_reason={
                    "reason": OPENAI_CHAT_REASON_FROM_PROVIDER.get(
                        finish_reason, "stop"
                    )
                },
                choice_index=choice_index,
            )
        )

    def _handle_usage_from_p(
        self,
        usage: dict[str, Any],
        events: list[IRStreamEvent],
    ) -> None:
        """Emit UsageEvent from chunk usage data."""
        events.append(
            UsageEvent(
                type="usage",
                usage={
                    "prompt_tokens": usage.get("prompt_tokens") or 0,
                    "completion_tokens": usage.get("completion_tokens") or 0,
                    "total_tokens": usage.get("total_tokens") or 0,
                },
            )
        )

    def _handle_stream_end_from_p(
        self,
        choices: list[dict[str, Any]],
        events: list[IRStreamEvent],
        usage: dict[str, Any] | None,
        context: StreamContext | None,
    ) -> None:
        """Emit StreamEndEvent when the stream is complete."""
        if context is None:
            return

        # Guard: only treat empty choices as stream-end AFTER the stream has
        # actually started.  Some upstreams (e.g. Azure / Argo) send a
        # preflight chunk with ``choices: []`` and empty ``id``/``model``
        # before the real content begins.
        if (
            context.is_started
            and isinstance(choices, list)
            and len(choices) == 0
            and not context.is_ended
        ):
            context.mark_ended()
            events.append(StreamEndEvent(type="stream_end"))

        # Also emit StreamEndEvent when we got a finish_reason but upstream
        # may not send a subsequent empty-choices chunk (e.g. when the
        # upstream ignores stream_options.include_usage).
        if (
            not context.is_ended
            and usage
            and any(e.get("type") == "finish" for e in events if isinstance(e, dict))
        ):
            context.mark_ended()
            events.append(StreamEndEvent(type="stream_end"))

    # --- to_provider ---

    def stream_response_to_provider(
        self,
        event: IRStreamEvent,
        context: StreamContext | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert an IR stream event to an OpenAI SSE chunk.

        When a ``context`` is provided and the stream has been started,
        top-level fields (``id``, ``object``, ``model``, ``created``) are
        populated on every output chunk.

        Args:
            event: IR stream event.
            context: Optional stream context for stateful conversions.

        Returns:
            OpenAI SSE chunk dict, or a list of chunk dicts.
        """
        handler_name = self._TO_P_DISPATCH.get(event.get("type", ""))
        if handler_name is None:
            return {}
        result = getattr(self, handler_name)(event, context)

        # Populate top-level fields when context is available and started
        if (
            context is not None
            and context.is_started
            and isinstance(result, dict)
            and result
        ):
            if "id" not in result:
                result["id"] = context.response_id
            if "object" not in result:
                result["object"] = "chat.completion.chunk"
            if "model" not in result:
                result["model"] = context.model
            if "created" not in result:
                result["created"] = context.created

        return result

    def _handle_stream_start_to_p(
        self, event: StreamStartEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle StreamStartEvent → initial chunk with role delta."""
        if context is not None:
            context.response_id = event["response_id"]
            context.model = event["model"]
            context.created = event.get("created", 0)
            context.mark_started()
        return {
            "id": event["response_id"],
            "object": "chat.completion.chunk",
            "model": event["model"],
            "created": event.get("created", 0),
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant"},
                    "finish_reason": None,
                }
            ],
        }

    def _handle_stream_end_to_p(
        self, event: StreamEndEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle StreamEndEvent → empty choices chunk."""
        if context is not None:
            context.mark_ended()
        return {
            "id": context.response_id if context else "",
            "object": "chat.completion.chunk",
            "model": context.model if context else "",
            "created": context.created if context else 0,
            "choices": [],
        }

    def _handle_content_block_start_to_p(
        self, event: ContentBlockStartEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ContentBlockStartEvent → no-op for OpenAI Chat."""
        return {}

    def _handle_content_block_end_to_p(
        self, event: ContentBlockEndEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ContentBlockEndEvent → no-op for OpenAI Chat."""
        return {}

    def _handle_text_delta_to_p(
        self, event: TextDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle TextDeltaEvent → content delta chunk."""
        choice_index = event.get("choice_index", 0)
        return {
            "choices": [
                {
                    "index": choice_index,
                    "delta": {"content": event["text"]},
                }
            ]
        }

    def _handle_reasoning_delta_to_p(
        self, event: ReasoningDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ReasoningDeltaEvent → reasoning_content delta chunk."""
        choice_index = event.get("choice_index", 0)
        return {
            "choices": [
                {
                    "index": choice_index,
                    "delta": {"reasoning_content": event["reasoning"]},
                }
            ]
        }

    def _handle_tool_call_start_to_p(
        self, event: ToolCallStartEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ToolCallStartEvent → tool_calls delta with id and name."""
        choice_index = event.get("choice_index", 0)
        tc_index = event.get("tool_call_index", 0)
        tc_entry: dict[str, Any] = {
            "index": tc_index,
            "id": event["tool_call_id"],
            "type": "function",
            "function": {
                "name": event["tool_name"],
                "arguments": "",
            },
        }
        return {
            "choices": [
                {
                    "index": choice_index,
                    "delta": {"tool_calls": [tc_entry]},
                }
            ]
        }

    def _handle_tool_call_delta_to_p(
        self, event: ToolCallDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ToolCallDeltaEvent → tool_calls delta with arguments."""
        choice_index = event.get("choice_index", 0)
        tc_index = event.get("tool_call_index", 0)
        tc_delta_entry: dict[str, Any] = {
            "index": tc_index,
            "function": {
                "arguments": event["arguments_delta"],
            },
        }
        return {
            "choices": [
                {
                    "index": choice_index,
                    "delta": {"tool_calls": [tc_delta_entry]},
                }
            ]
        }

    def _handle_finish_to_p(
        self, event: FinishEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle FinishEvent → finish_reason chunk."""
        choice_index = event.get("choice_index", 0)
        reason = event["finish_reason"]["reason"]
        return {
            "choices": [
                {
                    "index": choice_index,
                    "delta": {},
                    "finish_reason": OPENAI_CHAT_REASON_TO_PROVIDER.get(reason, "stop"),
                }
            ]
        }

    def _handle_usage_to_p(
        self, event: UsageEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle UsageEvent → usage chunk with empty choices."""
        usage = event["usage"]
        return {
            "choices": [],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens") or 0,
                "completion_tokens": usage.get("completion_tokens") or 0,
                "total_tokens": usage.get("total_tokens") or 0,
            },
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
