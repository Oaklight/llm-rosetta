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

import time
from collections.abc import Mapping
from typing import Any, cast


from ....types.ir import (
    CitationPart,
    TextPart,
    is_citation_part,
    is_text_part,
    is_tool_result_part,
)
from ....types.ir.request import IRRequest
from ....types.ir.response import IRResponse, UsageInfo
from ...base import BaseConverter
from llm_rosetta.conversion_warnings import report_loss
from llm_rosetta.converters.base.helpers.service_tier import google_service_tier
from ...base.context import (
    ConversionContext,
    StreamContext,
)
from ...base.tools.intrinsic import (
    is_intrinsic_part,
)
from ...base.helpers.extensions import wire_extensions
from ...base.helpers.opaque import (
    inject_google_function_call_signature,
)
from llm_rosetta.ir.primitives.native import record_native
from ....transforms.ir import hoist_late_system_messages
from ._constants import (
    GOOGLE_REASON_FROM_PROVIDER,
    validate_google_output_format,
)
from .config_ops import GoogleGenerateConfigOps
from .content_ops import GoogleGenerateContentOps
from .hosted_ops import GoogleGenerateHostedOps
from .message_ops import GoogleGenerateMessageOps
from .response_ops import (
    _modality_list_to_dict,
    _dict_to_modality_list,
    _resolve_system_instruction,
    _hoist_and_extract_system,
    _grounding_metadata_from_ir,
    _google_finish_reason,
    _native_word_lost,
    _native_prompt_feedback,
)
from .stream_from_ops import GoogleGenerateStreamFromOps
from .stream_to_ops import GoogleGenerateStreamToOps
from .tool_ops import GoogleGenerateToolOps


class GoogleGenerateConverter(
    GoogleGenerateStreamFromOps,
    GoogleGenerateStreamToOps,
    GoogleGenerateHostedOps,
    BaseConverter,
):
    """Google GenAI API converter.

    Implements the 6 explicit conversion interfaces defined by BaseConverter.

    Uses composition of Ops classes for modular, testable conversion logic.
    """

    _RESPONSE_ID_PREFIX = ""

    # The mixins annotate ``tool_ops`` with the concrete ops class so their
    # bodies type-check.  Re-declare it as the base contract's ``Any`` here so
    # the converter's public attribute keeps the type it had before the split;
    # otherwise callers see the ops class's ``ToolDefinition | list | None``.
    tool_ops: Any

    content_ops_class = GoogleGenerateContentOps
    tool_ops_class = GoogleGenerateToolOps
    message_ops_class = GoogleGenerateMessageOps
    config_ops_class = GoogleGenerateConfigOps
    _CONVERTER_TAG = "google_generate"
    _PASSTHROUGH_RESTORE_KEY = "candidates"
    _INTRINSIC_IR_TRANSFORMS = (hoist_late_system_messages(),)

    # A server-tool result arrives whole, so the writer needs a handler for
    # the one-shot event the reader emits.  (#986)
    _IR_TO_P_DISPATCH: dict[str, str] = {
        **BaseConverter._IR_TO_P_DISPATCH,
        "tool_result": "_handle_ir_tool_result_to_p",
        "refusal_delta": "_handle_ir_refusal_delta_to_p",
    }

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
        for key in (
            "tools",
            "tool_config",
            "response_mime_type",
            "response_schema",
            "cached_content",
            "service_tier",
        ):
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
            rest_tc = GoogleGenerateConverter._thinking_config_to_rest(
                config["thinking_config"]
            )
            if rest_tc:
                generation_config["thinkingConfig"] = rest_tc
        elif "thinkingConfig" in config and isinstance(config["thinkingConfig"], dict):
            generation_config["thinkingConfig"] = config["thinkingConfig"]

        if generation_config:
            body["generationConfig"] = generation_config

        # system_instruction lives in ``config`` in the SDK shape; the REST wire
        # wants it at the top level.  A top-level key is still honoured so a
        # hand-built dict keeps working.
        system_instruction = _resolve_system_instruction(sdk_request, config)
        if system_instruction:
            body["system_instruction"] = system_instruction

        return body

    def _do_request_to_provider(
        self,
        ir_request: IRRequest,
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        ctx = context
        # State what this request hands upstream before writing it: hosted
        # calls and client function calls share one wire shape, so the response
        # reader can only tell them apart from this record.  See
        # ``p_function_call_intrinsic_kind``.
        self._record_request_tool_vocabulary(ir_request.get("tools"), ctx)
        # The default is the SDK shape because this converter's callers consume
        # the result as ``google.genai`` client arguments (see the converters
        # guide).  ``ConversionPipeline`` defaults to ``"rest"`` instead, since
        # its callers put the body on the wire directly (#882); the divergence
        # is deliberate, not an oversight.
        output_format: str = kwargs.pop(
            "output_format",
            ctx.options.get("output_format", "sdk"),
        )
        validate_google_output_format(output_format)
        result: dict[str, Any] = {"model": ir_request["model"]}

        # 1. Handle system_instruction
        system_instruction = self._build_system_instruction(ir_request)

        # 2. Handle messages — the pairing repair already ran at the
        #    converter boundary; tool_choice/tool_config are stripped here.
        ir_messages = ir_request.get("messages", [])
        # Batch ids and the stripped tool config were applied at the
        # converter boundary, over the same repaired list.

        # Hoist mid-conversation system messages and extract leading ones.
        ir_messages, system_instruction = _hoist_and_extract_system(
            ir_messages, ir_request, system_instruction
        )

        # Convert non-system messages
        contents, msg_warnings = self.message_ops.ir_messages_to_p(
            ir_messages,
            target_provider=self._CONVERTER_TAG,
            image_fetcher=ctx.image_fetcher,
        )
        ctx.warnings.extend(msg_warnings)
        result["contents"] = contents

        # 3. Build config dict (tools written by _apply_tool_config)
        self._apply_tool_config(ir_request, result, ctx)
        config = result.setdefault("config", {})

        # Generation config
        gen_config = ir_request.get("generation")
        if gen_config:
            gen_fields = self.config_ops.ir_generation_config_to_p(gen_config)
            config.update(gen_fields)
            if "truncation" in gen_config:
                report_loss(
                    "Google GenAI has no truncation control; ignored",
                    subject="generation.truncation",
                )

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

        # Service tier -- only the values both sides spell the same way.
        service_tier = ir_request.get("service_tier")
        if service_tier:
            mapped = google_service_tier(service_tier, "Google GenAI")
            if mapped:
                config["service_tier"] = mapped

        # Stream config
        stream = ir_request.get("stream")
        if stream:
            stream_fields = self.config_ops.ir_stream_config_to_p(stream)
            config.update(stream_fields)
            if stream.get("include_usage"):
                report_loss(
                    "Google GenAI reports usage on every response; "
                    "include_usage is not a request field",
                    subject="stream.include_usage",
                )

        # Cache config
        cache = ir_request.get("cache")
        if cache:
            cache_fields = self.config_ops.ir_cache_config_to_p(cache)
            config.update(cache_fields)

        # Provider extensions
        config.update(wire_extensions(ir_request, self._CONVERTER_TAG))

        # System instruction belongs inside ``config``: that is where
        # ``google.genai``'s ``GenerateContentConfig`` declares it, and it makes
        # this dict exactly the ``generate_content(model=…, contents=…,
        # config=…)`` kwargs (see ``types/google/request_types.py``).  Written
        # last so a provider extension cannot clobber the real value.  The REST
        # body lifts it back to the top level in ``_to_rest_body``.
        if system_instruction:
            config["system_instruction"] = system_instruction

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

        # The SDK shape keeps tools, generation params and the system
        # instruction inside ``config``; the REST shape keeps the instruction at
        # the top level instead.  Both are accepted.
        config = provider_request.get("config", {})
        if not isinstance(config, dict):
            config = {}

        # 1. System instruction
        system_instruction = _resolve_system_instruction(provider_request, config)
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
        # inside "generationConfig").  ``config`` was extracted above.

        # Tools — check SDK config first, then REST top-level (uncached, #1410)
        tools = config.get("tools") or provider_request.get("tools")
        if tools:
            ir_request["tools"] = self._convert_p_tools_to_ir(tools)
            # A client replays the model turn it received, where a hosted
            # tool's call is the same ``functionCall`` part a client function
            # uses.  The history has no recorded vocabulary to consult, but
            # the body carries one: classify from the tools it declares, or
            # the pairing pass answers a finished server call with a
            # synthetic result.
            self._classify_hosted_history_calls(ir_messages, tools)

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

        self._apply_cache(provider_request, config, ir_request)

        return ir_request

    def _apply_cache(
        self,
        provider_request: dict[str, Any],
        config: dict[str, Any],
        ir_request: dict[str, Any],
    ) -> None:
        """Read a cached-content resource into *ir_request*, when it names one.

        The SDK shape keeps the resource in ``config``; a REST body carries it
        at the top level, snake or camel.  Without this read the resource is
        write-only: a request that names one loses it on the way back, and a
        cross-format round trip cannot tell that from a request that never had
        one (#1328).
        """
        for source in (config, provider_request):
            if "cached_content" in source or "cachedContent" in source:
                cache = self.config_ops.p_cache_config_to_ir(source)
                if cache:
                    ir_request["cache"] = cache
                    return

    def _get_response_id_prefix(
        self, context: ConversionContext | StreamContext | None = None
    ) -> str:
        """Return the response ID prefix from context or class default."""
        if context is not None:
            prefix = context.options.get("response_id_prefix")
            if prefix is not None:
                return prefix
        return self._RESPONSE_ID_PREFIX

    def _record_native_finish_reason(
        self, message: dict[str, Any], reason: str, native: Any
    ) -> None:
        """Keep Google's own word when the IR would lose it.

        The IR has one reason for what Google distinguishes several ways:
        ``LANGUAGE``, ``MALFORMED_FUNCTION_CALL`` and ``OTHER`` all become
        ``error``, and ``RECITATION``, ``BLOCKLIST``, ``PROHIBITED_CONTENT`` and
        ``SPII`` all become ``content_filter``.  A word the mapping would drop
        is recorded, so a Google target writes back what Google said instead of
        the coarser IR reason; a failure reason is always recorded, because the
        record is also what tells the writer this turn came from Google rather
        than from an upstream that failed.  A blocked prompt's ``blockReason``
        is recorded the same way, because the writer has only a candidate to
        write it on.

        Args:
            message: The IR message the reason belongs to.
            reason: The IR reason the candidate mapped to.
            native: The raw value Google reported, if any.
        """
        if not _native_word_lost(reason, native):
            return
        metadata = message.get("provider_metadata")
        if not isinstance(metadata, dict):
            metadata = {}
            message["provider_metadata"] = metadata
        record_native(metadata, origin=self._CONVERTER_TAG, finish_reason=native)

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
            message = (
                self.message_ops._p_message_to_ir(content, leg="response")
                if content
                else None
            )
            # A model turn keeps the parts it owns (a server call, its inline
            # server result, text).  Only a *client* result splits the turn
            # into a list, which ``choices[0].message`` cannot hold and which a
            # response candidate cannot legitimately carry — keep the assistant
            # turn and drop the client tool turn.  (#965, #986)
            if isinstance(message, list):
                message = next(
                    (m for m in message if m.get("role") == "assistant"),
                    message[0],
                )
            # Fallback for empty candidates (e.g. thinking consumed all tokens)
            if message is None:
                message = {"role": "assistant", "content": []}
            # A hosted tool's call arrives as a plain ``functionCall``, the
            # same part a client function uses; promote the ones this request
            # enabled before the parts decide the turn's finish reason.
            self._classify_hosted_calls(message, context)
            grounding = p_candidate.get("grounding_metadata") or p_candidate.get(
                "groundingMetadata"
            )
            if isinstance(grounding, dict):
                self._apply_candidate_grounding(message, grounding)

            finish_reason_val = p_candidate.get("finish_reason") or p_candidate.get(
                "finishReason"
            )
            reason = GOOGLE_REASON_FROM_PROVIDER.get(finish_reason_val, "stop")
            # A Google word the IR has no room for is recorded where a Google
            # target can write it back; without it ``error`` is ambiguous
            # between Google's own odd stop and a foreign failure.
            self._record_native_finish_reason(
                cast(dict[str, Any], message), reason, finish_reason_val
            )
            choice_info: dict[str, Any] = {
                "index": p_candidate.get("index", 0),
                "message": message,
                "finish_reason": {"reason": reason},
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
                block_ir_reason = GOOGLE_REASON_FROM_PROVIDER.get(
                    block_reason, "content_filter"
                )
                blocked_message: dict[str, Any] = {
                    "role": "assistant",
                    "content": [],
                }
                # A blocked prompt has no candidate at all: the reason is the
                # ``promptFeedback`` field, which the IR has no form for, so
                # keep it verbatim for the writer that can send it back.  The
                # field carries the block reason itself, which is why no
                # separate finish-reason record is needed here.
                record_native(
                    blocked_message.setdefault("provider_metadata", {}),
                    origin=self._CONVERTER_TAG,
                    prompt_feedback=prompt_feedback,
                )
                choices.append(
                    {
                        "index": 0,
                        "message": blocked_message,
                        "finish_reason": {"reason": block_ir_reason},
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

            message_metadata = message.get("provider_metadata") or {}
            # A blocked prompt is the one turn Google answers without a
            # candidate: the reason is the ``promptFeedback`` field, so writing
            # a candidate would hand the client a turn the model never made.
            prompt_feedback = _native_prompt_feedback(
                message_metadata, self._CONVERTER_TAG
            )
            if prompt_feedback is not None:
                provider_response["promptFeedback"] = prompt_feedback
                continue

            # Convert message back to Google Content format
            google_role = "model" if message.get("role") == "assistant" else "user"
            parts: list[dict[str, Any]] = []

            grounding_citations: list[CitationPart] = []
            for part in message.get("content", []):
                if is_citation_part(part):
                    # A citation is not a content part in Gemini; it is rebuilt
                    # into the candidate's ``groundingMetadata`` below.
                    grounding_citations.append(part)
                    continue
                # A client ``tool_result`` belongs to a separate tool turn,
                # which the message path normalizes to a ``role: "tool"``
                # message; a response is a single model turn, so it is dropped
                # here as it was before the intrinsic writer was reused.  An
                # intrinsic (server) result does belong inline next to its
                # call, so only the client shape is skipped.  (#986)
                if is_tool_result_part(part) and not is_intrinsic_part(part):
                    continue
                # Route through the intrinsic-aware part writer: an intrinsic
                # call becomes ``executableCode`` and a server result
                # ``codeExecutionResult`` (or is dropped when Google cannot
                # render it), never a client ``functionCall``.  This is the
                # same writer the message path uses, and it is handed the same
                # warnings sink, so a drop is reported on the conversion rather
                # than only warned about.  (#986)
                converted = self.message_ops._ir_content_part_to_p(
                    part, warnings_list=context.warnings
                )
                if converted:
                    parts.append(converted)

            if google_role == "model":
                inject_google_function_call_signature(parts)

            finish_reason = choice.get("finish_reason", {})
            reason = finish_reason.get("reason", "stop")

            candidate: dict[str, Any] = {
                "index": choice.get("index", 0),
                "content": {"role": google_role, "parts": parts},
                "finishReason": _google_finish_reason(
                    reason,
                    native=message_metadata,
                    base=self._CONVERTER_TAG,
                    response_id=ir_response.get("id", ""),
                    metadata=message_metadata,
                ),
            }
            if grounding_citations:
                grounding = _grounding_metadata_from_ir(
                    grounding_citations,
                    message.get("provider_metadata") or {},
                )
                if grounding:
                    candidate["groundingMetadata"] = grounding
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
        # No sink is handed over: the entry point installed this conversion's
        # sink and its leg, and a site that names its own would keep the old
        # routing rule alive beside it.
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


# Backward compatibility alias
GoogleConverter = GoogleGenerateConverter
