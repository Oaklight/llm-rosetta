"""
LLMIR - OpenAI Chat Completions Converter

Top-level converter implementing the 6 explicit interfaces + 2 stream methods.
Composes ContentOps, ToolOps, MessageOps, and ConfigOps for full bidirectional
conversion between IR and OpenAI Chat Completions API format.
"""

from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

from ...types.ir import (
    ExtensionItem,
    Message,
)
from ...types.ir.request import IRRequest
from ...types.ir.response import IRResponse
from ...types.ir.stream import (
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

    # ==================== Normalization ====================

    @staticmethod
    def _normalize(data: Any) -> dict:
        """Normalize SDK objects to plain dicts.

        Handles Pydantic models (``model_dump()``), dataclasses, and other
        objects with dict-like conversion methods.

        Args:
            data: Input data, possibly an SDK object.

        Returns:
            Plain dict representation.
        """
        if isinstance(data, dict):
            return data
        if hasattr(data, "model_dump"):
            return data.model_dump()
        if hasattr(data, "to_dict"):
            return data.to_dict()
        if hasattr(data, "__dict__"):
            return dict(data.__dict__)
        raise TypeError(f"Cannot normalize {type(data).__name__} to dict")

    # ==================== Top-level Interfaces ====================

    def request_to_provider(
        self,
        ir_request: IRRequest,
        **kwargs: Any,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Convert IRRequest to OpenAI Chat Completions request parameters.

        Orchestrates all Ops classes to build the complete provider request.

        Args:
            ir_request: IR request.

        Returns:
            Tuple of (provider request dict, warnings list).
        """
        warnings: List[str] = []
        result: Dict[str, Any] = {"model": ir_request["model"]}

        # 1. System instruction → system message
        messages: List[Dict[str, Any]] = []
        system_instruction = ir_request.get("system_instruction")
        if system_instruction:
            if isinstance(system_instruction, str):
                messages.append({"role": "system", "content": system_instruction})
            elif isinstance(system_instruction, list):
                text_parts = []
                for part in system_instruction:
                    if isinstance(part, dict) and part.get("type") == "text":
                        text_parts.append(part["text"])
                    elif isinstance(part, str):
                        text_parts.append(part)
                messages.append({"role": "system", "content": " ".join(text_parts)})

        # 2. Messages
        ir_messages = ir_request.get("messages", [])
        converted_msgs, msg_warnings = self.message_ops.ir_messages_to_p(ir_messages)
        messages.extend(converted_msgs)
        warnings.extend(msg_warnings)
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
                warnings.append("OpenAI Chat does not support max_tool_calls, ignored")

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
        provider_request: Dict[str, Any],
        **kwargs: Any,
    ) -> IRRequest:
        """Convert OpenAI Chat Completions request to IRRequest.

        Args:
            provider_request: OpenAI request dict (or SDK object).

        Returns:
            IR request.
        """
        provider_request = self._normalize(provider_request)

        ir_request: Dict[str, Any] = {
            "model": provider_request.get("model", ""),
            "messages": [],
        }

        # 1. Messages - separate system messages as system_instruction
        messages = provider_request.get("messages", [])
        ir_messages: List[Dict[str, Any]] = []

        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "system":
                content = msg.get("content", "")
                if isinstance(content, str):
                    ir_request["system_instruction"] = content
                elif isinstance(content, list):
                    text_parts = []
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "text":
                            text_parts.append({"type": "text", "text": part["text"]})
                    ir_request["system_instruction"] = text_parts
            else:
                converted = self.message_ops._p_message_to_ir(msg)
                if converted:
                    ir_messages.append(converted)

        ir_request["messages"] = ir_messages

        # 2. Tools
        tools = provider_request.get("tools")
        if tools:
            ir_request["tools"] = [
                self.tool_ops.p_tool_definition_to_ir(t) for t in tools
            ]

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

        return ir_request

    def response_from_provider(
        self,
        provider_response: Dict[str, Any],
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
            reason_map = {
                "stop": "stop",
                "length": "length",
                "tool_calls": "tool_calls",
                "content_filter": "content_filter",
                "function_call": "tool_calls",
            }

            choice_info: Dict[str, Any] = {
                "index": p_choice.get("index", 0),
                "message": message,
                "finish_reason": {"reason": reason_map.get(finish_reason_val, "stop")},
            }

            if "logprobs" in p_choice:
                choice_info["logprobs"] = p_choice["logprobs"]

            choices.append(choice_info)

        ir_response: Dict[str, Any] = {
            "id": provider_response.get("id", ""),
            "object": "response",
            "created": provider_response.get("created", 0),
            "model": provider_response.get("model", ""),
            "choices": choices,
        }

        # Usage
        p_usage = provider_response.get("usage")
        if p_usage:
            usage_info: Dict[str, Any] = {
                "prompt_tokens": p_usage.get("prompt_tokens", 0),
                "completion_tokens": p_usage.get("completion_tokens", 0),
                "total_tokens": p_usage.get("total_tokens", 0),
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

        if "service_tier" in provider_response:
            ir_response["service_tier"] = provider_response["service_tier"]

        if "system_fingerprint" in provider_response:
            ir_response["system_fingerprint"] = provider_response["system_fingerprint"]

        return ir_response

    def response_to_provider(
        self,
        ir_response: IRResponse,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Convert IRResponse to OpenAI Chat Completions response.

        Args:
            ir_response: IR response.

        Returns:
            OpenAI response dict.
        """
        provider_response: Dict[str, Any] = {
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

            openai_message: Dict[str, Any] = {"role": message.get("role", "assistant")}

            content_parts = message.get("content", [])
            text_parts: List[str] = []
            tool_calls: List[Dict[str, Any]] = []

            for part in content_parts:
                if part.get("type") == "text":
                    text_parts.append(part["text"])
                elif part.get("type") == "tool_call":
                    tool_calls.append(self.tool_ops.ir_tool_call_to_p(part))

            if text_parts:
                openai_message["content"] = " ".join(text_parts)
            elif not tool_calls:
                openai_message["content"] = ""

            if tool_calls:
                openai_message["tool_calls"] = tool_calls
                if not text_parts:
                    openai_message["content"] = None

            openai_choice: Dict[str, Any] = {
                "index": choice.get("index", 0),
                "message": openai_message,
                "finish_reason": choice.get("finish_reason", {}).get("reason", "stop"),
            }

            if "logprobs" in choice:
                openai_choice["logprobs"] = choice["logprobs"]

            provider_response["choices"].append(openai_choice)

        # Usage
        ir_usage = ir_response.get("usage")
        if ir_usage:
            usage: Dict[str, Any] = {
                "prompt_tokens": ir_usage.get("prompt_tokens", 0),
                "completion_tokens": ir_usage.get("completion_tokens", 0),
                "total_tokens": ir_usage.get("total_tokens", 0),
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
        messages: Iterable[Union[Message, ExtensionItem]],
        **kwargs: Any,
    ) -> Tuple[List[Any], List[str]]:
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
        provider_messages: List[Any],
        **kwargs: Any,
    ) -> List[Union[Message, ExtensionItem]]:
        """Convert OpenAI Chat messages to IR message list.

        Delegates to message_ops.

        Args:
            provider_messages: OpenAI Chat messages.

        Returns:
            IR messages.
        """
        return self.message_ops.p_messages_to_ir(provider_messages, **kwargs)

    # ==================== Stream Support ====================

    def stream_response_from_provider(
        self,
        chunk: Dict[str, Any],
        context: Optional[StreamContext] = None,
    ) -> List[IRStreamEvent]:
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
        events: List[IRStreamEvent] = []

        # --- StreamStartEvent (only with context, on first chunk) ---
        if context is not None and not context.is_started:
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

        choices = chunk.get("choices", [])

        for p_choice in choices:
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
                for tc in tool_calls:
                    tc_func = tc.get("function", {})
                    tc_id = tc.get("id")
                    tc_index = tc.get("index")

                    if tc_id:
                        # New tool call starting
                        start_event_tc = ToolCallStartEvent(
                            type="tool_call_start",
                            tool_call_id=tc_id,
                            tool_name=tc_func.get("name", ""),
                            choice_index=choice_index,
                        )
                        if tc_index is not None:
                            start_event_tc["tool_call_index"] = tc_index
                        events.append(start_event_tc)

                        # Register tool call in context
                        if context is not None:
                            context.register_tool_call(tc_id, tc_func.get("name", ""))

                    arguments = tc_func.get("arguments")
                    if arguments:
                        delta_event = ToolCallDeltaEvent(
                            type="tool_call_delta",
                            tool_call_id=tc_id or "",
                            arguments_delta=arguments,
                            choice_index=choice_index,
                        )
                        if tc_index is not None:
                            delta_event["tool_call_index"] = tc_index
                        events.append(delta_event)

            # Finish reason
            finish_reason = p_choice.get("finish_reason")
            if finish_reason:
                reason_map = {
                    "stop": "stop",
                    "length": "length",
                    "tool_calls": "tool_calls",
                    "content_filter": "content_filter",
                    "function_call": "tool_calls",
                }
                events.append(
                    FinishEvent(
                        type="finish",
                        finish_reason={"reason": reason_map.get(finish_reason, "stop")},
                        choice_index=choice_index,
                    )
                )

        # Usage (typically in the last chunk)
        usage = chunk.get("usage")
        if usage:
            events.append(
                UsageEvent(
                    type="usage",
                    usage={
                        "prompt_tokens": usage.get("prompt_tokens", 0),
                        "completion_tokens": usage.get("completion_tokens", 0),
                        "total_tokens": usage.get("total_tokens", 0),
                    },
                )
            )

        # --- StreamEndEvent (only with context, when choices is empty list) ---
        if context is not None and isinstance(choices, list) and len(choices) == 0:
            context.mark_ended()
            events.append(StreamEndEvent(type="stream_end"))

        return events

    def stream_response_to_provider(
        self,
        ir_event: IRStreamEvent,
        context: Optional[StreamContext] = None,
    ) -> Union[Dict[str, Any], List[Dict[str, Any]]]:
        """Convert an IR stream event to an OpenAI SSE chunk.

        When a ``context`` is provided and the stream has been started,
        top-level fields (``id``, ``object``, ``model``, ``created``) are
        populated on every output chunk.

        Args:
            ir_event: IR stream event.
            context: Optional stream context for stateful conversions.

        Returns:
            OpenAI SSE chunk dict, or a list of chunk dicts.
        """
        event_type = ir_event["type"]
        result: Union[Dict[str, Any], List[Dict[str, Any]]] = {}

        if event_type == "stream_start":
            # Store metadata in context if provided
            if context is not None:
                context.response_id = ir_event["response_id"]
                context.model = ir_event["model"]
                context.created = ir_event.get("created", 0)
                context.mark_started()
            result = {
                "id": ir_event["response_id"],
                "object": "chat.completion.chunk",
                "model": ir_event["model"],
                "created": ir_event.get("created", 0),
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant"},
                        "finish_reason": None,
                    }
                ],
            }
            return result

        elif event_type == "stream_end":
            if context is not None:
                context.mark_ended()
            result = {
                "id": context.response_id if context else "",
                "object": "chat.completion.chunk",
                "model": context.model if context else "",
                "created": context.created if context else 0,
                "choices": [],
            }
            return result

        elif event_type == "content_block_start":
            return {}

        elif event_type == "content_block_end":
            return {}

        elif event_type == "text_delta":
            choice_index = ir_event.get("choice_index", 0)
            result = {
                "choices": [
                    {
                        "index": choice_index,
                        "delta": {"content": ir_event["text"]},
                    }
                ]
            }

        elif event_type == "reasoning_delta":
            choice_index = ir_event.get("choice_index", 0)
            result = {
                "choices": [
                    {
                        "index": choice_index,
                        "delta": {"reasoning_content": ir_event["reasoning"]},
                    }
                ]
            }

        elif event_type == "tool_call_start":
            choice_index = ir_event.get("choice_index", 0)
            tc_entry: Dict[str, Any] = {
                "id": ir_event["tool_call_id"],
                "type": "function",
                "function": {
                    "name": ir_event["tool_name"],
                    "arguments": "",
                },
            }
            tc_index = ir_event.get("tool_call_index")
            if tc_index is not None:
                tc_entry["index"] = tc_index
            result = {
                "choices": [
                    {
                        "index": choice_index,
                        "delta": {"tool_calls": [tc_entry]},
                    }
                ]
            }

        elif event_type == "tool_call_delta":
            choice_index = ir_event.get("choice_index", 0)
            tc_delta_entry: Dict[str, Any] = {
                "function": {
                    "arguments": ir_event["arguments_delta"],
                },
            }
            tc_index = ir_event.get("tool_call_index")
            if tc_index is not None:
                tc_delta_entry["index"] = tc_index
            result = {
                "choices": [
                    {
                        "index": choice_index,
                        "delta": {"tool_calls": [tc_delta_entry]},
                    }
                ]
            }

        elif event_type == "finish":
            choice_index = ir_event.get("choice_index", 0)
            result = {
                "choices": [
                    {
                        "index": choice_index,
                        "delta": {},
                        "finish_reason": ir_event["finish_reason"]["reason"],
                    }
                ]
            }

        elif event_type == "usage":
            usage = ir_event["usage"]
            result = {
                "usage": {
                    "prompt_tokens": usage.get("prompt_tokens", 0),
                    "completion_tokens": usage.get("completion_tokens", 0),
                    "total_tokens": usage.get("total_tokens", 0),
                }
            }

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
