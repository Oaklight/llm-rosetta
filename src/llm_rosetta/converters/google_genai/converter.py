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
from collections.abc import Sequence
from typing import Any, cast


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
    GOOGLE_REASON_FROM_PROVIDER,
    GOOGLE_REASON_TO_PROVIDER,
    generate_tool_call_id,
)
from .config_ops import GoogleGenAIConfigOps
from .content_ops import GoogleGenAIContentOps
from .message_ops import GoogleGenAIMessageOps
from .tool_ops import GoogleGenAIToolOps


def _modality_list_to_dict(modality_list: list[dict]) -> dict[str, int]:
    """Convert Google's ``list[ModalityTokenCount]`` to IR ``dict[str, int]``.

    Example: ``[{"modality": "TEXT", "token_count": 42}]``
    → ``{"text_tokens": 42}``
    """
    result: dict[str, int] = {}
    for item in modality_list:
        modality = (item.get("modality") or "unknown").lower()
        count = item.get("token_count") or item.get("tokenCount") or 0
        result[f"{modality}_tokens"] = count
    return result


def _dict_to_modality_list(details: dict[str, int]) -> list[dict[str, Any]]:
    """Convert IR ``dict[str, int]`` back to Google's ``list[ModalityTokenCount]``.

    Example: ``{"text_tokens": 42}``
    → ``[{"modality": "TEXT", "tokenCount": 42}]``
    """
    result: list[dict[str, Any]] = []
    for key, count in details.items():
        modality = key.removesuffix("_tokens").upper()
        result.append({"modality": modality, "tokenCount": count})
    return result


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

    @staticmethod
    def _to_rest_body(sdk_request: dict[str, Any]) -> dict[str, Any]:
        """Convert SDK-style request dict to Google REST API format.

        The SDK format nests tools, tool_config, and generation parameters
        inside a ``config`` dict.  The REST API expects tools and tool_config
        at the top level, and generation parameters wrapped in a
        ``generationConfig`` object.

        This is a pure dict→dict transform; it does **not** call any
        conversion ops.

        Args:
            sdk_request: SDK-style request dict (as produced by
                ``request_to_provider()`` with the default output format).

        Returns:
            REST API–ready request body.
        """
        body: dict[str, Any] = {"contents": sdk_request["contents"]}
        config = sdk_request.get("config", {})

        # Lift specific keys from config to top level
        for key in ("tools", "tool_config", "response_mime_type", "response_schema"):
            if config.get(key):
                body[key] = config[key]

        # Lift generation config fields into generationConfig
        _GENERATION_KEYS = (
            "temperature",
            "top_p",
            "top_k",
            "max_output_tokens",
            "stop_sequences",
            "candidate_count",
            "seed",
            "presence_penalty",
            "frequency_penalty",
            "logprobs",
            "response_logprobs",
        )
        generation_config: dict[str, Any] = {}
        for key in _GENERATION_KEYS:
            if key in config:
                generation_config[key] = config[key]
        if generation_config:
            body["generationConfig"] = generation_config

        # system_instruction is already at top level from the converter
        if "system_instruction" in sdk_request:
            body["system_instruction"] = sdk_request["system_instruction"]

        return body

    def request_to_provider(
        self,
        ir_request: IRRequest,
        *,
        context: ConversionContext | None = None,
        **kwargs: Any,
    ) -> tuple[dict[str, Any], list[str]]:
        """Convert IRRequest to Google GenAI request parameters.

        Orchestrates all Ops classes to build the complete provider request.

        Args:
            ir_request: IR request.
            **kwargs: Optional keyword arguments.

                - ``output_format``: ``"sdk"`` (default) produces a dict with
                  a nested ``config`` suitable for the Google GenAI Python SDK.
                  ``"rest"`` flattens the config so the result can be sent
                  directly to the Google REST API via ``httpx`` / ``requests``.

        Returns:
            Tuple of (provider request dict, warnings list).
        """
        ctx = context if context is not None else ConversionContext()
        output_format: str = kwargs.pop(
            "output_format",
            ctx.options.get("output_format", "sdk"),
        )
        result: dict[str, Any] = {"model": ir_request["model"]}

        # 1. Handle system_instruction
        system_instruction = None

        # From IRRequest.system_instruction field
        ir_system = ir_request.get("system_instruction")
        if ir_system:
            system_instruction = {"role": "user", "parts": [{"text": ir_system}]}

        # 2. Handle messages — fix orphaned tool_calls/results and strip
        #    orphaned tool_choice/tool_config at IR level before conversion.
        ir_messages = fix_orphaned_tool_calls_ir(ir_request.get("messages", []))
        ctx.warnings.extend(strip_orphaned_tool_config(ir_request))

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
        ctx.warnings.extend(msg_warnings)
        result["contents"] = contents

        if system_instruction:
            result["system_instruction"] = system_instruction

        # 3. Build config dict (tools written by _apply_tool_config)
        self._apply_tool_config(ir_request, result, ctx)
        config = result.setdefault("config", {})

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

        if output_format == "rest":
            return self._to_rest_body(result), ctx.warnings

        return result, ctx.warnings

    def request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext | None = None,
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
            parsed = self._parse_system_instruction(system_instruction)
            if parsed:
                ir_request["system_instruction"] = parsed

        # 2. Messages
        contents = provider_request.get("contents", [])
        ir_messages = self.message_ops.p_messages_to_ir(contents)
        ir_request["messages"] = ir_messages

        # 3. Config fields
        # Support both SDK format (tools/tool_config inside "config" dict)
        # and REST format (tools/tool_config at top level, generation params
        # inside "generationConfig").
        config = provider_request.get("config", {})
        if not isinstance(config, dict):
            config = {}

        # Tools — check SDK config first, then REST top-level
        tools = config.get("tools") or provider_request.get("tools")
        if tools:
            ir_request["tools"] = self._convert_tools_from_p(tools)

        # Tool choice — check SDK/REST snake_case/camelCase
        tool_config = (
            config.get("tool_config")
            or provider_request.get("tool_config")
            or provider_request.get("toolConfig")
        )
        if tool_config:
            ir_request["tool_choice"] = self.tool_ops.p_tool_choice_to_ir(tool_config)

        # Generation config — check SDK config first, then REST generationConfig
        gen_source = config
        rest_gen_config = provider_request.get("generationConfig")
        if rest_gen_config and isinstance(rest_gen_config, dict) and not config:
            gen_source = rest_gen_config
        gen_config = self.config_ops.p_generation_config_to_ir(gen_source)
        if gen_config:
            ir_request["generation"] = gen_config

        # Response format — check both SDK config and REST top-level (snake + camel)
        response_mime_source = None
        if "response_mime_type" in config or "responseMimeType" in config:
            response_mime_source = config
        elif (
            "response_mime_type" in provider_request
            or "responseMimeType" in provider_request
        ):
            response_mime_source = provider_request
        if response_mime_source:
            ir_request["response_format"] = self.config_ops.p_response_format_to_ir(
                response_mime_source
            )

        # Reasoning config (snake + camel)
        if "thinking_config" in config or "thinkingConfig" in config:
            ir_request["reasoning"] = self.config_ops.p_reasoning_config_to_ir(config)

        return self._validate_ir_request(ir_request)

    def response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext | None = None,
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
            message = self.message_ops._p_message_to_ir(content) if content else None
            # Fallback for empty candidates (e.g. thinking consumed all tokens)
            if message is None:
                message = {"role": "assistant", "content": []}

            finish_reason_val = p_candidate.get("finish_reason") or p_candidate.get(
                "finishReason"
            )
            choice_info: dict[str, Any] = {
                "index": p_candidate.get("index", 0),
                "message": message,
                "finish_reason": {
                    "reason": GOOGLE_REASON_FROM_PROVIDER.get(finish_reason_val, "stop")
                },
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
            ir_response["usage"] = self._build_ir_usage(p_usage)

        return self._validate_ir_response(ir_response)

    def response_to_provider(
        self,
        ir_response: IRResponse,
        *,
        context: ConversionContext | None = None,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Convert IRResponse to Google GenAI response.

        Args:
            ir_response: IR response.

        Returns:
            Google response dict.
        """
        provider_response: dict[str, Any] = {
            "responseId": ir_response.get("id", ""),
            "modelVersion": ir_response.get("model", ""),
            "candidates": [],
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
                "finishReason": GOOGLE_REASON_TO_PROVIDER.get(reason, "STOP"),
            }
            provider_response["candidates"].append(candidate)

        # Usage
        ir_usage = ir_response.get("usage")
        if ir_usage:
            provider_response["usageMetadata"] = self._build_provider_usage(ir_usage)  # ty: ignore[invalid-argument-type]

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
        """Apply tools and tool_choice to provider config dict."""
        config = result.setdefault("config", {})
        tools = ir_request.get("tools")
        if tools:
            config["tools"] = [self.tool_ops.ir_tool_definition_to_p(t) for t in tools]

        tool_choice = ir_request.get("tool_choice")
        if tool_choice:
            tc_p = self.tool_ops.ir_tool_choice_to_p(tool_choice)
            if tc_p:
                config["tool_config"] = tc_p

    @staticmethod
    def _build_ir_usage(p_usage: dict[str, Any]) -> dict[str, Any]:
        """Build IR usage dict from Google usage metadata."""
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

        thoughts = p_usage.get("thoughts_token_count") or p_usage.get(
            "thoughtsTokenCount"
        )
        if thoughts is not None:
            usage_info["reasoning_tokens"] = thoughts

        cached = p_usage.get("cached_content_token_count") or p_usage.get(
            "cachedContentTokenCount"
        )
        if cached is not None:
            usage_info["cache_read_tokens"] = cached

        prompt_details = p_usage.get("prompt_tokens_details") or p_usage.get(
            "promptTokensDetails"
        )
        if prompt_details:
            usage_info["prompt_tokens_details"] = (
                _modality_list_to_dict(prompt_details)
                if isinstance(prompt_details, list)
                else prompt_details
            )

        candidates_details = p_usage.get("candidates_tokens_details") or p_usage.get(
            "candidatesTokensDetails"
        )
        if candidates_details:
            usage_info["completion_tokens_details"] = (
                _modality_list_to_dict(candidates_details)
                if isinstance(candidates_details, list)
                else candidates_details
            )

        return usage_info

    @staticmethod
    def _build_provider_usage(ir_usage: dict[str, Any]) -> dict[str, Any]:
        """Build Google usage metadata dict from IR usage."""
        usage_metadata: dict[str, Any] = {
            "promptTokenCount": ir_usage.get("prompt_tokens") or 0,
            "candidatesTokenCount": ir_usage.get("completion_tokens") or 0,
            "totalTokenCount": ir_usage.get("total_tokens") or 0,
        }

        if "reasoning_tokens" in ir_usage:
            usage_metadata["thoughtsTokenCount"] = ir_usage["reasoning_tokens"]

        if "cache_read_tokens" in ir_usage:
            usage_metadata["cachedContentTokenCount"] = ir_usage["cache_read_tokens"]

        if "prompt_tokens_details" in ir_usage:
            details = ir_usage["prompt_tokens_details"]
            usage_metadata["promptTokensDetails"] = (
                _dict_to_modality_list(details)
                if isinstance(details, dict)
                else details
            )

        if "completion_tokens_details" in ir_usage:
            details = ir_usage["completion_tokens_details"]
            usage_metadata["candidatesTokensDetails"] = (
                _dict_to_modality_list(details)
                if isinstance(details, dict)
                else details
            )

        return usage_metadata

    @staticmethod
    def _parse_system_instruction(system_instruction: Any) -> str | None:
        """Parse Google GenAI system_instruction to plain text."""
        if isinstance(system_instruction, str):
            return system_instruction
        if isinstance(system_instruction, dict):
            parts = system_instruction.get("parts", [])
            text_parts = [
                part["text"]
                for part in parts
                if isinstance(part, dict) and "text" in part
            ]
            if text_parts:
                return " ".join(text_parts)
        return None

    def _convert_tools_from_p(self, tools: list[Any]) -> list[Any]:
        """Convert provider tool definitions to IR."""
        ir_tools: list[Any] = []
        for t in tools:
            try:
                result = self.tool_ops.p_tool_definition_to_ir(t)
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
            if result is None:
                continue
            if isinstance(result, list):
                ir_tools.extend(result)
            else:
                ir_tools.append(result)
        return ir_tools

    def messages_to_provider(
        self,
        messages: Sequence[Message | ExtensionItem],
        *,
        context: ConversionContext | None = None,
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
        *,
        context: ConversionContext | None = None,
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
        tools: Sequence[ToolDefinition] | None = None,
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
        tools: Sequence[ToolDefinition] | None = None,
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

    # --- from_provider ---

    def stream_response_from_provider(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None = None,
    ) -> list[IRStreamEvent]:
        """Convert a Google GenAI stream chunk to IR stream events.

        Google GenAI stream chunks are complete ``GenerateContentResponse``
        objects. Each chunk contains incremental content in
        ``candidates[].content.parts[]``.

        When a ``context`` is provided, lifecycle events (``StreamStartEvent``,
        ``StreamEndEvent``) are emitted and cross-chunk state is tracked.

        Args:
            chunk: Google GenAI stream chunk dict (or SDK object).
            context: Optional stream context for stateful conversions.

        Returns:
            List of IR stream events extracted from the chunk.
        """
        chunk = self._normalize(chunk)
        events: list[IRStreamEvent] = []

        if context is not None and not context.is_started:
            self._handle_stream_start_from_p(chunk, context, events)

        has_finish_reason = False
        deferred_finish: FinishEvent | None = None

        for candidate in chunk.get("candidates", []):
            choice_index = candidate.get("index", 0)
            content = candidate.get("content", {})

            for part in content.get("parts", []):
                self._handle_part_from_p(part, choice_index, context, events)

            finish_reason = candidate.get("finish_reason") or candidate.get(
                "finishReason"
            )
            if finish_reason:
                has_finish_reason = True
                deferred_finish = FinishEvent(
                    type="finish",
                    finish_reason={
                        "reason": GOOGLE_REASON_FROM_PROVIDER.get(finish_reason, "stop")  # ty: ignore[invalid-argument-type]
                    },
                    choice_index=choice_index,
                )

        self._handle_usage_from_p(chunk, events)

        if deferred_finish is not None:
            events.append(deferred_finish)

        if context is not None and has_finish_reason:
            context.mark_ended()
            events.append(StreamEndEvent(type="stream_end"))

        return events

    def _handle_stream_start_from_p(
        self,
        chunk: dict[str, Any],
        context: StreamContext,
        events: list[IRStreamEvent],
    ) -> None:
        """Emit StreamStartEvent on the first chunk."""
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

    def _handle_part_from_p(
        self,
        part: dict[str, Any],
        choice_index: int,
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Process a single part from a candidate's content."""
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

        func_call = part.get("function_call") or part.get("functionCall")
        if func_call:
            self._handle_function_call_from_p(
                func_call, part, choice_index, context, events
            )

    def _handle_function_call_from_p(
        self,
        func_call: dict[str, Any],
        part: dict[str, Any],
        choice_index: int,
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Process a function_call part into ToolCallStart + ToolCallDelta events."""
        tool_call_id = func_call.get("id") or generate_tool_call_id()
        tool_name = func_call.get("name", "")
        args = func_call.get("args", {})

        if context is not None:
            context.register_tool_call(tool_call_id, tool_name)

        tc_index = len(context._tool_call_order) - 1 if context is not None else 0
        start_event: dict[str, Any] = {
            "type": "tool_call_start",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "choice_index": choice_index,
            "tool_call_index": tc_index,
        }

        thought_sig = part.get("thoughtSignature") or part.get("thought_signature")
        if thought_sig:
            start_event["provider_metadata"] = {
                "google": {"thought_signature": thought_sig}
            }

        events.append(cast(ToolCallStartEvent, start_event))

        args_json = json.dumps(args) if isinstance(args, dict) else str(args)
        delta_evt = ToolCallDeltaEvent(
            type="tool_call_delta",
            tool_call_id=tool_call_id,
            arguments_delta=args_json,
            choice_index=choice_index,
        )
        delta_evt["tool_call_index"] = tc_index
        events.append(delta_evt)

        if context is not None:
            context.append_tool_call_args(tool_call_id, args_json)

    def _handle_usage_from_p(
        self,
        chunk: dict[str, Any],
        events: list[IRStreamEvent],
    ) -> None:
        """Emit UsageEvent from chunk usage metadata."""
        usage = chunk.get("usage_metadata") or chunk.get("usageMetadata")
        if not usage:
            return

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

        thoughts = usage.get("thoughts_token_count") or usage.get("thoughtsTokenCount")
        if thoughts is not None:
            usage_info["reasoning_tokens"] = thoughts

        events.append(
            UsageEvent(
                type="usage",
                usage=cast(UsageInfo, usage_info),
            )
        )

    # --- to_provider ---

    def stream_response_to_provider(
        self,
        event: IRStreamEvent,
        context: StreamContext | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert an IR stream event to a Google GenAI stream chunk.

        Reconstructs a ``GenerateContentResponse``-shaped chunk from an IR
        stream event.

        Args:
            event: IR stream event.
            context: Optional stream context for stateful conversions.

        Returns:
            Google GenAI stream chunk dict.
        """
        handler_name = self._TO_P_DISPATCH.get(event.get("type", ""))
        if handler_name is None:
            return {}
        return getattr(self, handler_name)(event, context)

    def _handle_stream_start_to_p(
        self, event: StreamStartEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle StreamStartEvent → store metadata, no output."""
        if context is not None:
            context.response_id = event["response_id"]
            context.model = event["model"]
            context.mark_started()
        return {}

    def _handle_stream_end_to_p(
        self, event: StreamEndEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle StreamEndEvent → mark ended, no output."""
        if context is not None:
            context.mark_ended()
        return {}

    def _handle_content_block_start_to_p(
        self, event: ContentBlockStartEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ContentBlockStartEvent → no-op for Google GenAI."""
        return {}

    def _handle_content_block_end_to_p(
        self, event: ContentBlockEndEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ContentBlockEndEvent → no-op for Google GenAI."""
        return {}

    def _handle_text_delta_to_p(
        self, event: TextDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle TextDeltaEvent → text part chunk.

        Returns empty for empty-text deltas (e.g. padding in Google
        finish chunks) to avoid inflating the output event count.
        """
        if not event["text"]:
            return {}
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

    def _handle_reasoning_delta_to_p(
        self, event: ReasoningDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ReasoningDeltaEvent → thought text part chunk."""
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

    def _handle_tool_call_start_to_p(
        self, event: ToolCallStartEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ToolCallStartEvent → register in context, no output."""
        if context is not None:
            context.register_tool_call(event["tool_call_id"], event["tool_name"])
        return {}

    def _handle_tool_call_delta_to_p(
        self, event: ToolCallDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ToolCallDeltaEvent → accumulate args, no output."""
        if context is not None:
            context.append_tool_call_args(
                event["tool_call_id"], event["arguments_delta"]
            )
        return {}

    def _handle_finish_to_p(
        self, event: FinishEvent, context: StreamContext | None
    ) -> list[dict[str, Any]]:
        """Handle FinishEvent → flush tool calls + finish chunk."""
        choice_index = event.get("choice_index", 0)
        reason = event["finish_reason"]["reason"]

        chunks: list[dict[str, Any]] = []

        # Flush accumulated tool calls as complete function_call chunks.
        if context is not None:
            for call_id, tool_name, args_str in context.get_pending_tool_calls():
                try:
                    args = json.loads(args_str) if args_str else {}
                except (json.JSONDecodeError, TypeError):
                    args = {}
                chunks.append(
                    {
                        "candidates": [
                            {
                                "index": choice_index,
                                "content": {
                                    "role": "model",
                                    "parts": [
                                        {
                                            "functionCall": {
                                                "name": tool_name,
                                                "args": args,
                                            }
                                        }
                                    ],
                                },
                            }
                        ]
                    }
                )

        finish_chunk: dict[str, Any] = {
            "candidates": [
                {
                    "index": choice_index,
                    "content": {"role": "model", "parts": []},
                    "finishReason": GOOGLE_REASON_TO_PROVIDER.get(reason, "STOP"),
                }
            ]
        }

        # Merge buffered usage into the finish chunk so that
        # finishReason and usageMetadata stay in a single chunk,
        # matching the original Google format.
        if context is not None and context.pending_usage is not None:
            usage = context.pending_usage
            context.pending_usage = None
            usage_metadata: dict[str, Any] = {
                "promptTokenCount": usage.get("prompt_tokens") or 0,
                "candidatesTokenCount": usage.get("completion_tokens") or 0,
                "totalTokenCount": usage.get("total_tokens") or 0,
            }
            if "reasoning_tokens" in usage:
                usage_metadata["thoughtsTokenCount"] = usage["reasoning_tokens"]
            finish_chunk["usageMetadata"] = usage_metadata

        chunks.append(finish_chunk)

        return chunks

    def _handle_usage_to_p(
        self, event: UsageEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle UsageEvent → buffer for FinishEvent merge.

        When context is provided, buffers usage in pending_usage so
        FinishEvent can emit a single combined chunk with both
        finishReason and usageMetadata, matching the original Google
        format and preventing round-trip inflation.
        """
        usage = event["usage"]
        if context is not None:
            context.pending_usage = dict(usage)
            return {}
        usage_metadata: dict[str, Any] = {
            "promptTokenCount": usage.get("prompt_tokens") or 0,
            "candidatesTokenCount": usage.get("completion_tokens") or 0,
            "totalTokenCount": usage.get("total_tokens") or 0,
        }

        if "reasoning_tokens" in usage:
            usage_metadata["thoughtsTokenCount"] = usage["reasoning_tokens"]

        return {"usageMetadata": usage_metadata}

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


# Backward compatibility alias
GoogleConverter = GoogleGenAIConverter
