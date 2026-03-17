"""
LLM-Rosetta - Google GenAI Converter

Top-level converter implementing the 6 explicit interfaces.
Composes ContentOps, ToolOps, MessageOps, and ConfigOps for full bidirectional
conversion between IR and Google GenAI API format.

Google-specific:
- System messages → system_instruction (top-level, not in contents)
- Messages → contents (list of Content objects with role + parts)
- Config → GenerateContentConfig (generation params, tools, tool_config)
- Response → candidates (list of Candidate objects)

Also maintains backward compatibility with the old to_provider/from_provider API.
"""

import json
import time
import uuid
from typing import Any, cast
from collections.abc import Iterable


from ...types.ir import (
    ExtensionItem,
    IRInput,
    Message,
    ToolChoice,
    ToolDefinition,
    is_message,
    is_text_part,
    is_tool_call_part,
    is_reasoning_part,
)
from ...types.ir.request import IRRequest
from ...types.ir.response import IRResponse, UsageInfo
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
from .config_ops import GoogleGenAIConfigOps
from .content_ops import GoogleGenAIContentOps
from .message_ops import GoogleGenAIMessageOps
from .tool_ops import GoogleGenAIToolOps


class GoogleGenAIConverter(BaseConverter):
    """Google GenAI API converter.

    Implements the 6 explicit conversion interfaces defined by BaseConverter.

    Uses composition of Ops classes for modular, testable conversion logic.
    """

    content_ops_class = GoogleGenAIContentOps
    tool_ops_class = GoogleGenAIToolOps
    message_ops_class = GoogleGenAIMessageOps
    config_ops_class = GoogleGenAIConfigOps

    def __init__(self):
        self.content_ops = self.content_ops_class()
        self.tool_ops = self.tool_ops_class()
        self.message_ops = self.message_ops_class(self.content_ops, self.tool_ops)
        self.config_ops = self.config_ops_class()

    # ==================== Normalization ====================

    @staticmethod
    def _normalize(data: Any) -> dict:
        """Normalize SDK objects to plain dicts.

        Handles Pydantic models (``model_dump()``), tuples (unwrap first element),
        and other objects with dict-like conversion methods.

        Args:
            data: Input data, possibly an SDK object.

        Returns:
            Plain dict representation.

        Raises:
            TypeError: If data cannot be normalized.
        """
        if isinstance(data, tuple):
            data = data[0]
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
    ) -> tuple[dict[str, Any], list[str]]:
        """Convert IRRequest to Google GenAI request parameters.

        Orchestrates all Ops classes to build the complete provider request.

        Args:
            ir_request: IR request.

        Returns:
            Tuple of (provider request dict, warnings list).
        """
        warnings_list: list[str] = []
        result: dict[str, Any] = {"model": ir_request["model"]}

        # 1. Handle system_instruction
        system_instruction = None

        # From IRRequest.system_instruction field
        ir_system = ir_request.get("system_instruction")
        if ir_system:
            if isinstance(ir_system, str):
                system_instruction = {"role": "user", "parts": [{"text": ir_system}]}
            elif isinstance(ir_system, list):
                parts = []
                for part in ir_system:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append({"text": part["text"]})
                system_instruction = {"role": "user", "parts": parts}

        # 2. Handle messages (extract system messages)
        ir_messages = list(ir_request.get("messages", []))

        # Extract system messages from message list
        for item in ir_messages:
            if is_message(item) and item.get("role") == "system":
                msg_parts = []
                for part in item.get("content", []):
                    if is_text_part(part):
                        msg_parts.append({"text": part["text"]})
                if system_instruction is None:
                    system_instruction = {"role": "user", "parts": msg_parts}
                else:
                    cast(list, system_instruction["parts"]).extend(msg_parts)

        # Convert non-system messages
        contents, msg_warnings = self.message_ops.ir_messages_to_p(ir_messages)
        warnings_list.extend(msg_warnings)
        result["contents"] = contents

        if system_instruction:
            result["system_instruction"] = system_instruction

        # 3. Build config dict
        config: dict[str, Any] = {}

        # Tools
        tools = ir_request.get("tools")
        if tools:
            config["tools"] = [self.tool_ops.ir_tool_definition_to_p(t) for t in tools]

        # Tool choice
        tool_choice = ir_request.get("tool_choice")
        if tool_choice:
            tc_p = self.tool_ops.ir_tool_choice_to_p(tool_choice)
            if tc_p:
                config["tool_config"] = tc_p

        # Generation config
        gen_config = ir_request.get("generation")
        if gen_config:
            gen_fields = self.config_ops.ir_generation_config_to_p(gen_config)
            config.update(gen_fields)

        # Response format
        resp_format = ir_request.get("response_format")
        if resp_format:
            rf_fields = self.config_ops.ir_response_format_to_p(resp_format)
            config.update(rf_fields)

        # Reasoning config
        reasoning = ir_request.get("reasoning")
        if reasoning:
            reasoning_fields = self.config_ops.ir_reasoning_config_to_p(reasoning)
            config.update(reasoning_fields)

        # Stream config
        stream = ir_request.get("stream")
        if stream:
            stream_fields = self.config_ops.ir_stream_config_to_p(stream)
            config.update(stream_fields)

        # Cache config
        cache = ir_request.get("cache")
        if cache:
            cache_fields = self.config_ops.ir_cache_config_to_p(cache)
            config.update(cache_fields)

        # Provider extensions
        extensions = ir_request.get("provider_extensions")
        if extensions:
            config.update(extensions)

        result["config"] = config

        return result, warnings_list

    def request_from_provider(
        self,
        provider_request: dict[str, Any],
        **kwargs: Any,
    ) -> IRRequest:
        """Convert Google GenAI request to IRRequest.

        Args:
            provider_request: Google request dict (or SDK object).

        Returns:
            IR request.
        """
        provider_request = self._normalize(provider_request)

        ir_request: dict[str, Any] = {
            "model": provider_request.get("model", ""),
            "messages": [],
        }

        # 1. System instruction
        system_instruction = provider_request.get("system_instruction")
        if system_instruction:
            if isinstance(system_instruction, str):
                ir_request["system_instruction"] = system_instruction
            elif isinstance(system_instruction, dict):
                parts = system_instruction.get("parts", [])
                text_parts = []
                for part in parts:
                    if isinstance(part, dict) and "text" in part:
                        text_parts.append({"type": "text", "text": part["text"]})
                if text_parts:
                    ir_request["system_instruction"] = text_parts

        # 2. Messages
        contents = provider_request.get("contents", [])
        ir_messages = self.message_ops.p_messages_to_ir(contents)
        ir_request["messages"] = ir_messages

        # 3. Config fields
        config = provider_request.get("config", {})
        if not isinstance(config, dict):
            config = {}

        # Tools
        tools = config.get("tools")
        if tools:
            ir_tools = []
            for t in tools:
                ir_tools.append(self.tool_ops.p_tool_definition_to_ir(t))
            ir_request["tools"] = ir_tools

        # Tool choice
        tool_config = config.get("tool_config")
        if tool_config:
            ir_request["tool_choice"] = self.tool_ops.p_tool_choice_to_ir(tool_config)

        # Generation config
        gen_config = self.config_ops.p_generation_config_to_ir(config)
        if gen_config:
            ir_request["generation"] = gen_config

        # Response format
        if "response_mime_type" in config:
            ir_request["response_format"] = self.config_ops.p_response_format_to_ir(
                config
            )

        # Reasoning config
        if "thinking_config" in config:
            ir_request["reasoning"] = self.config_ops.p_reasoning_config_to_ir(config)

        return cast(IRRequest, ir_request)

    def response_from_provider(
        self,
        provider_response: dict[str, Any],
        **kwargs: Any,
    ) -> IRResponse:
        """Convert Google GenAI response to IRResponse.

        Args:
            provider_response: Google response dict (or SDK object).

        Returns:
            IR response.
        """
        provider_response = self._normalize(provider_response)

        choices = []
        candidates = provider_response.get("candidates", [])

        for p_candidate in candidates:
            content = p_candidate.get("content")
            message = (
                self.message_ops._p_message_to_ir(content)
                if content
                else {"role": "assistant", "content": []}
            )

            finish_reason_val = p_candidate.get("finish_reason") or p_candidate.get(
                "finishReason"
            )
            reason_map = {
                "STOP": "stop",
                "MAX_TOKENS": "length",
                "SAFETY": "content_filter",
                "RECITATION": "content_filter",
                "MALFORMED_FUNCTION_CALL": "error",
                "OTHER": "error",
            }

            choice_info: dict[str, Any] = {
                "index": p_candidate.get("index", 0),
                "message": message,
                "finish_reason": {"reason": reason_map.get(finish_reason_val, "stop")},
            }
            choices.append(choice_info)

        ir_response: dict[str, Any] = {
            "id": provider_response.get("response_id")
            or provider_response.get("responseId")
            or "",
            "object": "response",
            "created": int(time.time()),  # Google doesn't provide timestamp
            "model": provider_response.get("model_version")
            or provider_response.get("modelVersion")
            or "",
            "choices": choices,
        }

        # Handle usage
        p_usage = provider_response.get("usage_metadata") or provider_response.get(
            "usageMetadata"
        )
        if p_usage:
            usage_info: dict[str, Any] = {
                "prompt_tokens": p_usage.get(
                    "prompt_token_count", p_usage.get("promptTokenCount", 0)
                ),
                "completion_tokens": p_usage.get(
                    "candidates_token_count",
                    p_usage.get("candidatesTokenCount", 0),
                ),
                "total_tokens": p_usage.get(
                    "total_token_count", p_usage.get("totalTokenCount", 0)
                ),
            }

            # Reasoning tokens
            thoughts = p_usage.get("thoughts_token_count") or p_usage.get(
                "thoughtsTokenCount"
            )
            if thoughts is not None:
                usage_info["reasoning_tokens"] = thoughts

            ir_response["usage"] = usage_info

        return cast(IRResponse, ir_response)

    def response_to_provider(
        self,
        ir_response: IRResponse,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convert IRResponse to Google GenAI response.

        Args:
            ir_response: IR response.

        Returns:
            Google response dict.
        """
        provider_response: dict[str, Any] = {
            "response_id": ir_response.get("id", ""),
            "model_version": ir_response.get("model", ""),
            "candidates": [],
        }

        reason_map = {
            "stop": "STOP",
            "length": "MAX_TOKENS",
            "content_filter": "SAFETY",
            "tool_calls": "STOP",
            "error": "OTHER",
        }

        for choice in ir_response.get("choices", []):
            message = choice.get("message")
            if not message:
                continue

            # Convert message back to Google Content format
            google_role = "model" if message.get("role") == "assistant" else "user"
            parts: list[dict[str, Any]] = []

            for part in message.get("content", []):
                if is_text_part(part):
                    parts.append(self.content_ops.ir_text_to_p(part))
                elif is_tool_call_part(part):
                    parts.append(self.tool_ops.ir_tool_call_to_p(part))
                elif is_reasoning_part(part):
                    parts.append(self.content_ops.ir_reasoning_to_p(part))

            finish_reason = choice.get("finish_reason", {})
            reason = finish_reason.get("reason", "stop")

            candidate: dict[str, Any] = {
                "index": choice.get("index", 0),
                "content": {"role": google_role, "parts": parts},
                "finish_reason": reason_map.get(reason, "STOP"),
            }
            provider_response["candidates"].append(candidate)

        # Usage
        ir_usage = ir_response.get("usage")
        if ir_usage:
            usage_metadata: dict[str, Any] = {
                "prompt_token_count": ir_usage.get("prompt_tokens") or 0,
                "candidates_token_count": ir_usage.get("completion_tokens") or 0,
                "total_token_count": ir_usage.get("total_tokens") or 0,
            }

            if "reasoning_tokens" in ir_usage:
                usage_metadata["thoughts_token_count"] = ir_usage["reasoning_tokens"]

            provider_response["usage_metadata"] = usage_metadata

        return provider_response

    def messages_to_provider(
        self,
        messages: Iterable[Message | ExtensionItem],
        **kwargs: Any,
    ) -> tuple[list[Any], list[str]]:
        """Convert IR message list to Google GenAI Content format.

        Delegates to message_ops.

        Args:
            messages: IR messages (may contain ExtensionItems).

        Returns:
            Tuple of (converted Content list, warnings).
        """
        return self.message_ops.ir_messages_to_p(messages, **kwargs)

    def messages_from_provider(
        self,
        provider_messages: list[Any],
        **kwargs: Any,
    ) -> list[Message | ExtensionItem]:
        """Convert Google GenAI Content list to IR message list.

        Delegates to message_ops.

        Args:
            provider_messages: Google Content list.

        Returns:
            IR messages.
        """
        return self.message_ops.p_messages_to_ir(provider_messages, **kwargs)

    # ==================== Backward Compatibility ====================

    def build_config(
        self,
        tools: Iterable[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> dict[str, Any] | None:
        """Build Google GenAI config parameters (backward compatibility).

        Args:
            tools: Tool definition list.
            tool_choice: Tool choice configuration.

        Returns:
            Google GenAI config dict, or None if no tool configuration.
        """
        config: dict[str, Any] = {}

        if tools:
            config["tools"] = [self.tool_ops.ir_tool_definition_to_p(t) for t in tools]

        if tool_choice:
            tool_config = self.tool_ops.ir_tool_choice_to_p(tool_choice)
            if tool_config:
                config["tool_config"] = tool_config

        return config if config else None

    def to_provider(
        self,
        ir_input: IRInput | IRRequest,
        tools: Iterable[ToolDefinition] | None = None,
        tool_choice: ToolChoice | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], list[str]]:
        """Convert IR format to Google GenAI format (backward compatibility).

        Supports both IRInput (message list) and IRRequest (full request).

        Args:
            ir_input: IR input list or request object.
            tools: Tool definition list.
            tool_choice: Tool choice configuration.

        Returns:
            (Google GenAI format dict, warning list)
        """
        if isinstance(ir_input, dict) and "messages" in ir_input:
            # Handle IRRequest
            return self.request_to_provider(cast(IRRequest, ir_input))

        # Handle IRInput (message list)
        ir_input_list = list(ir_input)
        warnings_list: list[str] = []

        # Extract system messages
        system_instruction, remaining = self.message_ops.extract_system_instruction(
            ir_input_list
        )

        # Convert non-system messages
        contents, msg_warnings = self.message_ops.ir_messages_to_p(remaining)
        warnings_list.extend(msg_warnings)

        # Build result
        result: dict[str, Any] = {"contents": contents}

        if system_instruction:
            result["system_instruction"] = system_instruction

        # Convert tools
        if tools:
            result["tools"] = [self.tool_ops.ir_tool_definition_to_p(t) for t in tools]

        # Convert tool choice
        if tool_choice:
            tool_config = self.tool_ops.ir_tool_choice_to_p(tool_choice)
            if tool_config:
                result["tool_config"] = tool_config

        return result, warnings_list

    # ==================== Stream Support ====================

    def stream_response_from_provider(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None = None,
    ) -> list[IRStreamEvent]:
        """Convert a Google GenAI stream chunk to IR stream events.

        Google GenAI stream chunks are complete ``GenerateContentResponse``
        objects. Each chunk contains incremental content in
        ``candidates[].content.parts[]``.

        For text parts → ``TextDeltaEvent``
        For thought parts (``thought: true``) → ``ReasoningDeltaEvent``
        For function_call parts → ``ToolCallStartEvent`` + ``ToolCallDeltaEvent``
            (Google sends complete function calls, not incremental deltas)
        For finish_reason → ``FinishEvent``
        For usage_metadata → ``UsageEvent``

        When a ``context`` is provided, lifecycle events (``StreamStartEvent``,
        ``StreamEndEvent``) are emitted and cross-chunk state is tracked.
        Without a context the behaviour is identical to the previous
        implementation (backward compatible).

        Args:
            chunk: Google GenAI stream chunk dict (or SDK object).
            context: Optional stream context for stateful conversions.

        Returns:
            List of IR stream events extracted from the chunk.
        """
        chunk = self._normalize(chunk)
        events: list[IRStreamEvent] = []

        # --- StreamStartEvent (only with context, on first chunk) ---
        if context is not None and not context.is_started:
            response_id = chunk.get("response_id") or chunk.get("responseId") or ""
            model = chunk.get("model_version") or chunk.get("modelVersion") or ""
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

        has_finish_reason = False

        for candidate in chunk.get("candidates", []):
            choice_index = candidate.get("index", 0)
            content = candidate.get("content", {})
            parts = content.get("parts", [])

            for part in parts:
                # Check for thought/reasoning content
                is_thought = part.get("thought", False)

                if "text" in part and part["text"] is not None:
                    if is_thought:
                        events.append(
                            ReasoningDeltaEvent(
                                type="reasoning_delta",
                                reasoning=part["text"],
                                choice_index=choice_index,
                            )
                        )
                    else:
                        events.append(
                            TextDeltaEvent(
                                type="text_delta",
                                text=part["text"],
                                choice_index=choice_index,
                            )
                        )

                # Handle function_call (Google sends complete calls, not deltas)
                func_call = part.get("function_call") or part.get("functionCall")
                if func_call:
                    # Generate a unique tool_call_id since Google doesn't provide one
                    tool_call_id = func_call.get("id") or (
                        f"call_{func_call['name']}_{uuid.uuid4().hex[:8]}"
                    )
                    tool_name = func_call.get("name", "")
                    args = func_call.get("args", {})

                    # Register tool call in context
                    if context is not None:
                        context.register_tool_call(tool_call_id, tool_name)

                    # Build ToolCallStartEvent
                    start_event: dict[str, Any] = {
                        "type": "tool_call_start",
                        "tool_call_id": tool_call_id,
                        "tool_name": tool_name,
                        "choice_index": choice_index,
                    }

                    # Preserve thought_signature in provider_metadata
                    thought_sig = part.get("thoughtSignature") or part.get(
                        "thought_signature"
                    )
                    if thought_sig:
                        start_event["provider_metadata"] = {
                            "google": {"thought_signature": thought_sig}
                        }

                    events.append(cast(ToolCallStartEvent, start_event))

                    # Emit ToolCallDeltaEvent with complete arguments JSON
                    args_json = (
                        json.dumps(args) if isinstance(args, dict) else str(args)
                    )
                    events.append(
                        ToolCallDeltaEvent(
                            type="tool_call_delta",
                            tool_call_id=tool_call_id,
                            arguments_delta=args_json,
                            choice_index=choice_index,
                        )
                    )

            # Finish reason
            finish_reason = candidate.get("finish_reason") or candidate.get(
                "finishReason"
            )
            if finish_reason:
                has_finish_reason = True
                reason_map = {
                    "STOP": "stop",
                    "MAX_TOKENS": "length",
                    "SAFETY": "content_filter",
                    "RECITATION": "content_filter",
                    "MALFORMED_FUNCTION_CALL": "error",
                    "OTHER": "error",
                }
                events.append(
                    FinishEvent(
                        type="finish",
                        finish_reason={"reason": reason_map.get(finish_reason, "stop")},
                        choice_index=choice_index,
                    )
                )

        # Usage metadata (typically in the last chunk)
        usage = chunk.get("usage_metadata") or chunk.get("usageMetadata")
        if usage:
            usage_info: dict[str, Any] = {
                "prompt_tokens": usage.get(
                    "prompt_token_count", usage.get("promptTokenCount", 0)
                ),
                "completion_tokens": usage.get(
                    "candidates_token_count",
                    usage.get("candidatesTokenCount", 0),
                ),
                "total_tokens": usage.get(
                    "total_token_count", usage.get("totalTokenCount", 0)
                ),
            }

            # Reasoning tokens
            thoughts = usage.get("thoughts_token_count") or usage.get(
                "thoughtsTokenCount"
            )
            if thoughts is not None:
                usage_info["reasoning_tokens"] = thoughts

            events.append(
                UsageEvent(
                    type="usage",
                    usage=cast(UsageInfo, usage_info),
                )
            )

        # --- StreamEndEvent (only with context, when finish_reason present) ---
        if context is not None and has_finish_reason:
            context.mark_ended()
            events.append(StreamEndEvent(type="stream_end"))

        return events

    def stream_response_to_provider(
        self,
        event: IRStreamEvent,
        context: StreamContext | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert an IR stream event to a Google GenAI stream chunk.

        Reconstructs a ``GenerateContentResponse``-shaped chunk from an IR
        stream event.

        When a ``context`` is provided, tool call names can be recovered from
        the context for ``tool_call_delta`` events (fixing the P0 issue of
        lost tool names).

        Args:
            event: IR stream event.
            context: Optional stream context for stateful conversions.

        Returns:
            Google GenAI stream chunk dict.
        """
        if is_stream_start_event(event):
            # Store metadata in context if provided
            if context is not None:
                context.response_id = event["response_id"]
                context.model = event["model"]
                context.mark_started()
            return {}

        elif is_stream_end_event(event):
            if context is not None:
                context.mark_ended()
            return {}

        elif is_content_block_start_event(event):
            return {}

        elif is_content_block_end_event(event):
            return {}

        elif is_text_delta_event(event):
            choice_index = event.get("choice_index", 0)
            return {
                "candidates": [
                    {
                        "index": choice_index,
                        "content": {
                            "role": "model",
                            "parts": [{"text": event["text"]}],
                        },
                    }
                ]
            }

        elif is_reasoning_delta_event(event):
            choice_index = event.get("choice_index", 0)
            return {
                "candidates": [
                    {
                        "index": choice_index,
                        "content": {
                            "role": "model",
                            "parts": [{"thought": True, "text": event["reasoning"]}],
                        },
                    }
                ]
            }

        elif is_tool_call_start_event(event):
            # Google sends complete function calls, so tool_call_start
            # creates a function_call part with empty args (args come in delta)
            choice_index = event.get("choice_index", 0)
            return {
                "candidates": [
                    {
                        "index": choice_index,
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "function_call": {
                                        "name": event["tool_name"],
                                        "args": {},
                                    }
                                }
                            ],
                        },
                    }
                ]
            }

        elif is_tool_call_delta_event(event):
            # Google sends complete function calls; reconstruct from delta args
            choice_index = event.get("choice_index", 0)
            try:
                args = json.loads(event["arguments_delta"])
            except (json.JSONDecodeError, TypeError):
                args = {}

            # Recover tool name from context (P0 fix)
            tool_name = ""
            if context is not None:
                tool_name = context.get_tool_name(event["tool_call_id"])

            return {
                "candidates": [
                    {
                        "index": choice_index,
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "function_call": {
                                        "name": tool_name,
                                        "args": args,
                                    }
                                }
                            ],
                        },
                    }
                ]
            }

        elif is_finish_event(event):
            choice_index = event.get("choice_index", 0)
            reason = event["finish_reason"]["reason"]
            reason_map = {
                "stop": "STOP",
                "length": "MAX_TOKENS",
                "content_filter": "SAFETY",
                "tool_calls": "STOP",
                "error": "OTHER",
            }
            return {
                "candidates": [
                    {
                        "index": choice_index,
                        "content": {"role": "model", "parts": []},
                        "finish_reason": reason_map.get(reason, "STOP"),
                    }
                ]
            }

        elif is_usage_event(event):
            usage = event["usage"]
            usage_metadata: dict[str, Any] = {
                "prompt_token_count": usage.get("prompt_tokens") or 0,
                "candidates_token_count": usage.get("completion_tokens") or 0,
                "total_token_count": usage.get("total_tokens") or 0,
            }

            if "reasoning_tokens" in usage:
                usage_metadata["thoughts_token_count"] = usage["reasoning_tokens"]

            return {"usage_metadata": usage_metadata}

        return {}


# Backward compatibility alias
GoogleConverter = GoogleGenAIConverter
