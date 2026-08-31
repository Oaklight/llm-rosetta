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

"""

import json
import time
from collections.abc import Mapping
from typing import Any, cast


from ...types.ir import (
    TextPart,
    is_message,
    is_text_part,
    is_tool_call_part,
    is_reasoning_part,
    is_refusal_part,
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
from ..base.helpers import (
    fix_orphaned_tool_calls_ir,
    sanitize_tool_call_id,
    strip_orphaned_tool_config,
)
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


_GOOGLE_MODALITIES = frozenset({"TEXT", "IMAGE", "VIDEO", "AUDIO", "DOCUMENT"})


def _dict_to_modality_list(details: dict[str, int]) -> list[dict[str, Any]]:
    """Convert IR ``dict[str, int]`` back to Google's ``list[ModalityTokenCount]``.

    Only emits entries whose modality is in Google's ``Modality`` enum.
    IR keys like ``cached_tokens`` or ``reasoning_tokens`` are handled by
    dedicated fields (``cachedContentTokenCount``, ``thoughtsTokenCount``)
    and must not appear as ``ModalityTokenCount`` entries.

    Example: ``{"text_tokens": 42}``
    → ``[{"modality": "TEXT", "tokenCount": 42}]``
    """
    result: list[dict[str, Any]] = []
    for key, count in details.items():
        modality = key.removesuffix("_tokens").upper()
        if modality not in _GOOGLE_MODALITIES:
            continue
        result.append({"modality": modality, "tokenCount": count})
    return result


class GoogleGenAIConverter(BaseConverter):
    """Google GenAI API converter.

    Implements the 6 explicit conversion interfaces defined by BaseConverter.

    Uses composition of Ops classes for modular, testable conversion logic.
    """

    _RESPONSE_ID_PREFIX = ""

    content_ops_class = GoogleGenAIContentOps
    tool_ops_class = GoogleGenAIToolOps
    message_ops_class = GoogleGenAIMessageOps
    config_ops_class = GoogleGenAIConfigOps
    _CONVERTER_TAG = "google_genai"
    _PASSTHROUGH_RESTORE_KEY = "candidates"

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
    def _thinking_config_to_rest(tc: dict[str, Any]) -> dict[str, Any]:
        """Convert SDK-style thinking_config to REST camelCase format."""
        _FIELD_MAP = {
            "thinking_level": "thinkingLevel",
            "thinking_budget": "thinkingBudget",
            "include_thoughts": "includeThoughts",
        }
        rest_tc: dict[str, Any] = {}
        for snake, camel in _FIELD_MAP.items():
            if snake in tc:
                rest_tc[camel] = tc[snake]
            elif camel in tc:
                rest_tc[camel] = tc[camel]
        return rest_tc

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

        if "thinking_config" in config and isinstance(config["thinking_config"], dict):
            rest_tc = GoogleGenAIConverter._thinking_config_to_rest(
                config["thinking_config"]
            )
            if rest_tc:
                generation_config["thinkingConfig"] = rest_tc
        elif "thinkingConfig" in config and isinstance(config["thinkingConfig"], dict):
            generation_config["thinkingConfig"] = config["thinkingConfig"]

        if generation_config:
            body["generationConfig"] = generation_config

        # system_instruction is already at top level from the converter
        if "system_instruction" in sdk_request:
            body["system_instruction"] = sdk_request["system_instruction"]

        return body

    def _do_request_to_provider(
        self,
        ir_request: IRRequest,
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        ctx = context
        output_format: str = kwargs.pop(
            "output_format",
            ctx.options.get("output_format", "sdk"),
        )
        result: dict[str, Any] = {"model": ir_request["model"]}

        # 1. Handle system_instruction
        system_instruction = self._build_system_instruction(ir_request)

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
        contents, msg_warnings = self.message_ops.ir_messages_to_p(
            ir_messages, target_provider=self._CONVERTER_TAG
        )
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
            rc_kw = (
                {"reasoning_cap": ctx.options["reasoning_cap"]}
                if ctx and "reasoning_cap" in ctx.options
                else {}
            )
            reasoning_fields = self.config_ops.ir_reasoning_config_to_p(
                reasoning, **rc_kw
            )
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
            return self._to_rest_body(result)

        return result

    def _do_request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        ir_request: dict[str, Any] = {
            "model": provider_request.get("model", ""),
            "messages": [],
        }

        # 1. System instruction
        system_instruction = provider_request.get(
            "system_instruction"
        ) or provider_request.get("systemInstruction")
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

        # Tools — check SDK config first, then REST top-level (with cache)
        tools = config.get("tools") or provider_request.get("tools")
        if tools:
            ir_request["tools"] = self._get_cached_p_tools_to_ir(tools)

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

        # Reasoning config (snake + camel) — check SDK config, then REST generationConfig
        reasoning_source = config
        if not ("thinking_config" in config or "thinkingConfig" in config):
            reasoning_source = gen_source
        if (
            "thinking_config" in reasoning_source
            or "thinkingConfig" in reasoning_source
        ):
            ir_request["reasoning"] = self.config_ops.p_reasoning_config_to_ir(
                reasoning_source
            )

        return ir_request

    def _get_response_id_prefix(
        self, context: ConversionContext | StreamContext | None = None
    ) -> str:
        """Return the response ID prefix from context or class default."""
        if context is not None:
            prefix = context.options.get("response_id_prefix")
            if prefix is not None:
                return prefix
        return self._RESPONSE_ID_PREFIX

    def _do_response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
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

        # Handle prompt-level blocks (no candidates returned)
        if not choices:
            prompt_feedback = (
                provider_response.get("prompt_feedback")
                or provider_response.get("promptFeedback")
                or {}
            )
            block_reason = prompt_feedback.get("block_reason") or prompt_feedback.get(
                "blockReason"
            )
            if block_reason:
                choices.append(
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": []},
                        "finish_reason": {
                            "reason": GOOGLE_REASON_FROM_PROVIDER.get(
                                block_reason, "content_filter"
                            )
                        },
                    }
                )

        ir_response: dict[str, Any] = {
            "id": self.strip_response_id_prefix(
                provider_response.get("response_id")
                or provider_response.get("responseId")
                or "",
                self._get_response_id_prefix(context),
            ),
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
            ir_response["usage"] = self._build_p_usage_to_ir(p_usage)

        return ir_response

    def _do_response_to_provider(
        self,
        ir_response: IRResponse,
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        provider_response: dict[str, Any] = {
            "responseId": self.add_response_id_prefix(
                ir_response.get("id", ""), self._get_response_id_prefix(context)
            ),
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
                elif is_refusal_part(part):
                    parts.append(self.content_ops.ir_refusal_to_p(part))

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
            provider_response["usageMetadata"] = self._build_ir_usage_to_p(ir_usage)

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
            config["tools"] = self._get_cached_ir_tools_to_p(tools)

        tool_choice = ir_request.get("tool_choice")
        if tool_choice:
            tc_p = self.tool_ops.ir_tool_choice_to_p(tool_choice)
            if tc_p:
                config["tool_config"] = tc_p

    @staticmethod
    def _build_p_usage_to_ir(p_usage: dict[str, Any]) -> UsageInfo:
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

        return cast(UsageInfo, usage_info)

    @staticmethod
    def _build_ir_usage_to_p(ir_usage: Mapping[str, Any]) -> dict[str, Any]:
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
    def _build_system_instruction(ir_request: Any) -> dict[str, Any] | None:
        """Build Google system_instruction Content from IR system_instruction.

        Converts IR list[TextPart] to Google's Content format:
        ``{"role": "user", "parts": [{"text": "..."}]}``.
        """
        system_parts = ir_request.get("system_instruction")
        if not system_parts:
            return None
        parts = [{"text": p["text"]} for p in system_parts if is_text_part(p)]
        return {"role": "user", "parts": parts} if parts else None

    @staticmethod
    def _parse_system_instruction(system_instruction: Any) -> list[TextPart] | None:
        """Parse Google GenAI system_instruction to list[TextPart]."""
        if isinstance(system_instruction, str):
            return [TextPart(type="text", text=system_instruction)]
        if isinstance(system_instruction, dict):
            parts = system_instruction.get("parts", [])
            text_parts = [
                TextPart(type="text", text=part["text"])
                for part in parts
                if isinstance(part, dict) and "text" in part
            ]
            return text_parts or None
        return None

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
            self._handle_p_stream_start_to_ir(chunk, context, events)

        has_finish_reason = False
        deferred_finish: FinishEvent | None = None

        for candidate in chunk.get("candidates", []):
            finish = self._process_stream_candidate(candidate, context, events)
            if finish is not None:
                has_finish_reason = True
                deferred_finish = finish

        # Handle prompt-level blocks in streaming (no candidates)
        if not has_finish_reason and not chunk.get("candidates"):
            prompt_feedback = (
                chunk.get("prompt_feedback") or chunk.get("promptFeedback") or {}
            )
            block_reason = prompt_feedback.get("block_reason") or prompt_feedback.get(
                "blockReason"
            )
            if block_reason:
                has_finish_reason = True
                deferred_finish = FinishEvent(
                    type="finish",
                    finish_reason={
                        "reason": GOOGLE_REASON_FROM_PROVIDER.get(
                            block_reason, "content_filter"
                        )  # ty: ignore[invalid-argument-type]
                    },
                )

        self._handle_p_usage_to_ir(chunk, events)

        if deferred_finish is not None:
            events.append(deferred_finish)

        if context is not None and has_finish_reason:
            context.mark_ended()
            events.append(StreamEndEvent(type="stream_end"))

        return events

    def _process_stream_candidate(
        self,
        candidate: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> FinishEvent | None:
        """Process a single candidate from a stream chunk.

        Handles part dispatch, text deferral for compound chunks, and
        finish reason detection.  Returns a ``FinishEvent`` if the
        candidate signals completion, otherwise ``None``.
        """
        choice_index = candidate.get("index", 0)
        cand_content = candidate.get("content", {})
        finish_reason = candidate.get("finish_reason") or candidate.get("finishReason")

        pre_parts_len = len(events)
        for part in cand_content.get("parts", []):
            self._handle_p_part_to_ir(part, choice_index, context, events)

        if finish_reason and context is not None:
            new_events = events[pre_parts_len:]
            deferred_texts: list[str] = []
            kept_new: list[IRStreamEvent] = []
            for ev in new_events:
                if ev["type"] == "text_delta":
                    deferred_texts.append(ev["text"])
                else:
                    kept_new.append(ev)
            if deferred_texts:
                context.pending_text = "".join(deferred_texts)
                events[pre_parts_len:] = kept_new

        if finish_reason:
            return FinishEvent(
                type="finish",
                finish_reason={
                    "reason": GOOGLE_REASON_FROM_PROVIDER.get(finish_reason, "stop")  # ty: ignore[invalid-argument-type]
                },
                choice_index=choice_index,
            )
        return None

    def _handle_p_stream_start_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext,
        events: list[IRStreamEvent],
    ) -> None:
        """Emit StreamStartEvent on the first chunk."""
        response_id = chunk.get("response_id") or chunk.get("responseId") or ""
        model = chunk.get("model_version") or chunk.get("modelVersion") or ""
        prefix = self._get_response_id_prefix(context)
        context.response_id = self.strip_response_id_prefix(response_id, prefix)
        context.model = model
        context.mark_started()
        events.append(
            StreamStartEvent(
                type="stream_start",
                response_id=response_id,
                model=model,
            )
        )

    def _handle_p_part_to_ir(
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
            self._handle_p_function_call_to_ir(
                func_call, part, choice_index, context, events
            )

    def _handle_p_function_call_to_ir(
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

        tc_index = context.tool_call_count - 1 if context is not None else 0
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

    def _handle_p_usage_to_ir(
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

        cached = usage.get("cached_content_token_count") or usage.get(
            "cachedContentTokenCount"
        )
        if cached is not None:
            usage_info["cache_read_tokens"] = cached

        events.append(
            UsageEvent(
                type="usage",
                usage=cast(UsageInfo, usage_info),
            )
        )

    # --- to_provider ---

    @staticmethod
    def _inject_stream_metadata(
        chunk: dict[str, Any], context: StreamContext | None
    ) -> dict[str, Any]:
        """Inject ``responseId`` and ``modelVersion`` into a non-empty chunk.

        Google's streaming API includes these fields on every chunk.
        """
        if not chunk or context is None:
            return chunk
        if context.response_id:
            chunk["responseId"] = context.response_id
        if context.model:
            chunk["modelVersion"] = context.model
        return chunk

    @staticmethod
    def _build_stream_usage_metadata(usage: Mapping[str, Any]) -> dict[str, Any]:
        """Build a Google ``usageMetadata`` dict from IR usage for streaming.

        Lighter than ``_build_ir_usage_to_p`` — omits the details arrays
        which are not typically present in streaming usage events.
        """
        usage_metadata: dict[str, Any] = {
            "promptTokenCount": usage.get("prompt_tokens") or 0,
            "candidatesTokenCount": usage.get("completion_tokens") or 0,
            "totalTokenCount": usage.get("total_tokens") or 0,
        }
        if "reasoning_tokens" in usage:
            usage_metadata["thoughtsTokenCount"] = usage["reasoning_tokens"]
        if "cache_read_tokens" in usage:
            usage_metadata["cachedContentTokenCount"] = usage["cache_read_tokens"]
        return usage_metadata

    def _handle_ir_stream_start_to_p(
        self, event: StreamStartEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle StreamStartEvent → store metadata, no output."""
        if context is not None:
            context.response_id = event["response_id"]
            context.model = event["model"]
            context.mark_started()
        return {}

    def _handle_ir_stream_end_to_p(
        self, event: StreamEndEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle StreamEndEvent → flush any buffered usage, mark ended.

        When usage arrives after the finish chunk (e.g. OpenAI streaming
        sends usage in a separate final chunk), pending_usage won't have
        been merged into the finish chunk. Emit it here so it's not lost.
        """
        if context is not None:
            usage = context.pop_pending_usage()
            context.mark_ended()
            if usage is not None:
                return self._inject_stream_metadata(
                    {"usageMetadata": self._build_stream_usage_metadata(usage)},
                    context,
                )
        return {}

    def _handle_ir_content_block_start_to_p(
        self, event: ContentBlockStartEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ContentBlockStartEvent → no-op for Google GenAI."""
        return {}

    def _handle_ir_content_block_end_to_p(
        self, event: ContentBlockEndEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ContentBlockEndEvent → no-op for Google GenAI."""
        return {}

    def _handle_ir_text_delta_to_p(
        self, event: TextDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle TextDeltaEvent → text part chunk.

        Returns empty for empty-text deltas (e.g. padding in Google
        finish chunks) to avoid inflating the output event count.
        """
        if not event["text"]:
            return {}
        choice_index = event.get("choice_index", 0)
        return self._inject_stream_metadata(
            {
                "candidates": [
                    {
                        "index": choice_index,
                        "content": {
                            "role": "model",
                            "parts": [{"text": event["text"]}],
                        },
                    }
                ]
            },
            context,
        )

    def _handle_ir_reasoning_delta_to_p(
        self, event: ReasoningDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ReasoningDeltaEvent → thought text part chunk."""
        choice_index = event.get("choice_index", 0)
        return self._inject_stream_metadata(
            {
                "candidates": [
                    {
                        "index": choice_index,
                        "content": {
                            "role": "model",
                            "parts": [{"thought": True, "text": event["reasoning"]}],
                        },
                    }
                ]
            },
            context,
        )

    def _handle_ir_tool_call_start_to_p(
        self, event: ToolCallStartEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ToolCallStartEvent → register in context, no output."""
        if context is not None:
            context.register_tool_call(event["tool_call_id"], event["tool_name"])
        return {}

    def _handle_ir_tool_call_delta_to_p(
        self, event: ToolCallDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ToolCallDeltaEvent → accumulate args, no output."""
        if context is not None:
            context.append_tool_call_args(
                event["tool_call_id"], event["arguments_delta"]
            )
        return {}

    def _handle_ir_finish_to_p(
        self, event: FinishEvent, context: StreamContext | None
    ) -> list[dict[str, Any]]:
        """Handle FinishEvent → flush tool calls + finish chunk."""
        choice_index = event.get("choice_index", 0)
        reason = event["finish_reason"]["reason"]

        chunks: list[dict[str, Any]] = []

        # Merge deferred text and tool calls into the finish chunk's
        # parts array, matching Google's native format where a single
        # candidate carries content parts alongside finishReason.
        parts: list[dict[str, Any]] = []
        if context is not None and context.pending_text is not None:
            parts.append({"text": context.pending_text})
            context.pending_text = None

        if context is not None:
            for call_id, tool_name, args_str in context.get_pending_tool_calls():
                try:
                    args = json.loads(args_str) if args_str else {}
                except (json.JSONDecodeError, TypeError):
                    args = {}
                fc: dict[str, Any] = {"name": tool_name, "args": args}
                if call_id:
                    fc["id"] = sanitize_tool_call_id(call_id)
                parts.append({"functionCall": fc})

        finish_chunk: dict[str, Any] = {
            "candidates": [
                {
                    "index": choice_index,
                    "content": {"role": "model", "parts": parts},
                    "finishReason": GOOGLE_REASON_TO_PROVIDER.get(reason, "STOP"),
                }
            ]
        }

        # Merge buffered usage into the finish chunk so that
        # finishReason and usageMetadata stay in a single chunk,
        # matching the original Google format.
        if context is not None:
            usage = context.pop_pending_usage()
        else:
            usage = None
        if usage is not None:
            finish_chunk["usageMetadata"] = self._build_stream_usage_metadata(usage)

        chunks.append(self._inject_stream_metadata(finish_chunk, context))

        return chunks

    def _handle_ir_usage_to_p(
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
            context.buffer_usage(usage)
            return {}
        return {"usageMetadata": self._build_stream_usage_metadata(usage)}


# Backward compatibility alias
GoogleConverter = GoogleGenAIConverter
