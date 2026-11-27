"""
LLM-Rosetta - Anthropic Messages API Converter

Top-level converter implementing the 6 explicit interfaces + 2 stream methods.
Composes ContentOps, ToolOps, MessageOps, and ConfigOps for full bidirectional
conversion between IR and Anthropic Messages API format.

Key Anthropic differences from OpenAI:
- ``max_tokens`` is required; ``request_constraints`` supplies the value
- System messages via top-level ``system`` parameter
- Single response message (not choices list)
- No ``created`` timestamp (uses ``time.time()``)
- Tool call arguments are Dict (not JSON string)
- Thinking/reasoning with ``signature`` field
"""

import time
from collections.abc import Mapping
from typing import Any, cast

from llm_rosetta.ir.model import (
    TextPart,
    is_text_part,
    is_tool_call_part,
    is_tool_result_part,
    is_reasoning_part,
    is_citation_part,
    is_refusal_part,
)
from llm_rosetta.ir.model.request import IRRequest
from llm_rosetta.ir.model.response import IRResponse, UsageInfo
from llm_rosetta.ir.model.stream import (
    IRStreamEvent,
)
from llm_rosetta.ir.model.constraints import Range, RequestConstraints
from ....ir.passes.tool_names import STRICT_TOOL_NAME_PATTERN
from ...base import BaseConverter
from llm_rosetta.conversion_warnings import report_loss
from llm_rosetta.ir.context import (
    ConversionContext,
    StreamContext,
)
from ...base.helpers.citations import (
    citation_warning_sink,
    report_dropped_citation,
)
from ...base.helpers.extensions import wire_extensions
from llm_rosetta.ir.primitives.system_message_hoist import hoist_system_in_converter
from ....ir.primitives.ir import auto_cache_breakpoints, hoist_late_system_messages
from ._constants import (
    ANTHROPIC_REASON_FROM_PROVIDER,
    AnthropicEventType,
)
from .config_ops import MessagesConfigOps
from .content_ops import MessagesContentOps
from .message_ops import MessagesMessageOps
from .stream_reader import MessagesStreamReader
from .stream_writer import MessagesStreamWriter, _anthropic_stop_reason
from .tool_ops import MessagesToolOps


class MessagesConverter(MessagesStreamReader, MessagesStreamWriter, BaseConverter):
    """Anthropic Messages API converter.

    Implements the 6 explicit conversion interfaces defined by BaseConverter,
    plus 2 stream methods for SSE event-level conversion.

    Uses composition of Ops classes for modular, testable conversion logic.
    """

    _RESPONSE_ID_PREFIX = "msg_"

    content_ops_class = MessagesContentOps
    tool_ops_class = MessagesToolOps
    message_ops_class = MessagesMessageOps
    config_ops_class = MessagesConfigOps
    _CONVERTER_TAG = "messages"
    _PASSTHROUGH_RESTORE_KEY = "content"
    # Anthropic publishes ``^[a-zA-Z0-9_-]{1,64}$`` for a tool name and
    # rejects the request when one does not match.
    _TOOL_NAME_PATTERN = STRICT_TOOL_NAME_PATTERN
    _INTRINSIC_IR_TRANSFORMS = (hoist_late_system_messages(), auto_cache_breakpoints())

    # The Messages API has no default for ``max_tokens``: the field is
    # required, so a request that names no limit needs one invented, and this
    # is the one place that value lives.  ``temperature`` is documented as
    # 0.0-1.0; a client that sends more (OpenAI allows up to 2.0) gets the
    # ceiling, reported as a downgrade rather than silently.
    request_constraints = RequestConstraints(
        defaults={"generation.max_tokens": 4096},
        ranges={"generation.temperature": Range(0.0, 1.0, "clamp")},
    )

    def __init__(self):
        self.content_ops = self.content_ops_class()
        self.tool_ops = self.tool_ops_class()
        self.message_ops = self.message_ops_class(self.content_ops, self.tool_ops)
        self.config_ops = self.config_ops_class()

    # ==================== Top-level Interfaces ====================

    def _do_request_to_provider(
        self,
        ir_request: IRRequest,
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        ctx = context
        result: dict[str, Any] = {"model": ir_request["model"]}

        # 1. System instruction → top-level system parameter
        # system_instruction is list[TextPart]; convert via content_ops
        # so each block's metadata (e.g. cache_hint) is preserved.
        system_parts = ir_request.get("system_instruction")
        if system_parts:
            result["system"] = self.message_ops._ir_system_to_p(system_parts)

        # 2. Messages — the pairing repair already ran at the converter
        #    boundary, so every target sees the same repaired history.
        ir_messages = ir_request.get("messages", [])
        # Batch ids and the stripped tool config were applied at the
        # converter boundary, over the same repaired list.

        # Defense-in-depth: idempotent with pipeline's intrinsic transform.
        ir_messages, hoisted_si = hoist_system_in_converter(ir_messages, ir_request)
        if hoisted_si and "system" not in result:
            result["system"] = self.message_ops._ir_system_to_p(hoisted_si)

        message_kwargs: dict[str, Any] = {}
        if ctx and "reasoning_cap" in ctx.options:
            message_kwargs["reasoning_cap"] = ctx.options["reasoning_cap"]
        message_kwargs["target_provider"] = self._CONVERTER_TAG
        converted_msgs, msg_warnings = self.message_ops.ir_messages_to_p(
            ir_messages, **message_kwargs
        )
        ctx.warnings.extend(msg_warnings)
        result["messages"] = converted_msgs

        # 3. Generation config (must come before tools since max_tokens is required)
        gen_config = ir_request.get("generation")
        if gen_config:
            gen_fields = self.config_ops.ir_generation_config_to_p(gen_config)
            result.update(gen_fields)
            if "truncation" in gen_config:
                report_loss(
                    "Anthropic has no truncation control; ignored",
                    subject="generation.truncation",
                )
        if result.get("max_tokens") is None:
            # The dialect requires a value the request never named.  The base
            # class applies ``request_constraints`` before this writer runs, so
            # this is only reached when that declaration is removed — a
            # converter with no default would write a body the API refuses, so
            # say so here instead (#1460).
            default = self.request_constraints.defaults.get("generation.max_tokens")
            if default is None:
                raise ValueError(
                    "Messages requires generation.max_tokens and this converter "
                    "declares no default; declare it in request_constraints"
                )
            result["max_tokens"] = default
            # The write and the read are separate calls and the body cannot
            # carry a marker, so a per-conversion record is the only place this
            # fact can live; the gateway puts it in the request log (#1328).
            ctx.metadata.setdefault("injected", {})["max_tokens"] = default

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
        self._apply_reasoning_config(ir_request, result, ctx)

        # 10. Cache config (block-level, warning)
        cache = ir_request.get("cache")
        if cache:
            cache_fields = self.config_ops.ir_cache_config_to_p(cache)
            result.update(cache_fields)

        # 11. Service tier.  The Messages API only takes ``auto``; any real
        # tier (priority, flex, ...) is a scheduling preference this upstream
        # does not offer, so it goes unset and is reported.
        service_tier = ir_request.get("service_tier")
        if service_tier == "auto":
            result["service_tier"] = "auto"
        elif service_tier:
            report_loss(
                f"service_tier={service_tier!r} cannot be expressed by the "
                "Messages API; the request is sent without a tier",
                subject="service_tier",
            )

        # 12. Provider extensions (pass-through)
        result.update(wire_extensions(ir_request, self._CONVERTER_TAG))

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

        # 1. System instruction → list[TextPart]
        system_content = provider_request.get("system")
        if system_content:
            if isinstance(system_content, str):
                ir_request["system_instruction"] = [
                    TextPart(type="text", text=system_content)
                ]
            elif isinstance(system_content, list):
                ir_parts = []
                for part in system_content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        ir_parts.append(self.content_ops.p_text_to_ir(part))
                if ir_parts:
                    ir_request["system_instruction"] = ir_parts

        # 2. Messages
        messages = provider_request.get("messages", [])
        ir_messages = self.message_ops.p_messages_to_ir(messages)
        ir_request["messages"] = ir_messages

        # 3. Tools (provider→IR, uncached after #1410)
        tools = provider_request.get("tools")
        if tools:
            ir_request["tools"] = self._convert_p_tools_to_ir(tools)

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

        # 8. Service tier.  Messages models the *capacity policy* (auto,
        # standard_only), not a tier to schedule on, so only ``auto`` -- the
        # absence of a preference -- carries into the IR.  ``standard_only``
        # says which capacity may serve the request, and no other format can
        # express that, so it is reported rather than silently reset.
        service_tier = provider_request.get("service_tier")
        if service_tier == "auto":
            ir_request["service_tier"] = "auto"
        elif service_tier:
            report_loss(
                f"service_tier={service_tier!r} (Messages) has no equivalent in "
                "the IR's tier vocabulary; it was not carried",
                subject="service_tier",
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

        # Extract structured refusal into RefusalPart
        if stop_reason_val == "refusal":
            stop_details = provider_response.get("stop_details") or {}
            # No canned sentence: a refusal that carried no words came back
            # with none, rather than with words nobody said.
            explanation = stop_details.get("explanation", "")
            category = stop_details.get("category")
            refusal_part: dict[str, Any] = {
                "type": "refusal",
                "refusal": explanation,
            }
            if category:
                refusal_part["provider_metadata"] = {
                    "anthropic_refusal_category": category
                }
            msg_content = ir_message.get("content")
            if isinstance(msg_content, list):
                msg_content.append(refusal_part)
            else:
                ir_message["content"] = [refusal_part]

        prefix = self._get_response_id_prefix(context)
        ir_response: dict[str, Any] = {
            "id": self.strip_response_id_prefix(
                provider_response.get("id", ""), prefix
            ),
            "object": "response",
            "created": int(time.time()),  # Anthropic doesn't provide timestamp
            "model": provider_response.get("model", ""),
            "choices": [choice_info],
        }

        # Usage (always present — downstream clients may crash without it)
        p_usage = provider_response.get("usage") or {}
        ir_response["usage"] = self._build_p_usage_to_ir(p_usage)

        return ir_response

    def _ir_part_to_anthropic_block(self, part: Any) -> dict[str, Any] | None:
        """Convert one IR content part to an Anthropic block, or ``None``.

        Reasoning parts return ``None`` when the block was dropped (nothing
        Anthropic could consume, e.g. only a foreign opaque value — #881).

        A server call with no Anthropic server tool for its kind returns no
        block either: this is the response leg, so it is dropped rather than
        handed to the client as a call it would have to run.

        Args:
            part: IR content part.

        Returns:
            The Anthropic content block, or ``None`` when the part maps to no
            block.
        """
        if is_text_part(part):
            return self.content_ops.ir_text_to_p(part)
        if is_tool_call_part(part):
            return self.tool_ops.ir_tool_call_to_p(part, leg="response") or None
        if is_tool_result_part(part):
            # A server result the assistant turn produced stays inline; a
            # client result returns None and is dropped from the response.
            return self.tool_ops.ir_server_tool_result_to_p(part)
        if is_reasoning_part(part):
            return self.content_ops.ir_reasoning_to_p(part)
        return None

    @staticmethod
    def _default_text_citations(anthropic_content: list[dict[str, Any]]) -> None:
        """Every response text block carries the field, null when uncited.

        The spec requires ``citations`` on ``ResponseTextBlock``, so a block that
        has none gets an explicit null rather than the key being absent.
        """
        for block in anthropic_content:
            if block.get("type") == "text":
                block.setdefault("citations", None)

    def _attach_response_citation(
        self,
        part: Any,
        anthropic_content: list[dict[str, Any]],
        warnings: list[str] | None,
    ) -> None:
        """Hang a response citation on the text block it annotates."""
        citation = self.content_ops.ir_citation_to_p(part)
        if citation is None:
            report_dropped_citation(part, warnings=warnings)
        elif anthropic_content and anthropic_content[-1].get("type") == "text":
            anthropic_content[-1].setdefault("citations", []).append(citation)
        else:
            report_dropped_citation(
                part, warnings=warnings, why="no text block to annotate"
            )

    def _do_response_to_provider(
        self,
        ir_response: IRResponse,
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        prefix = self._get_response_id_prefix(context)
        provider_response: dict[str, Any] = {
            "id": self.add_response_id_prefix(ir_response.get("id", ""), prefix),
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

                refusal_part = None
                sink = citation_warning_sink(kwargs, context)
                for part in content_parts:
                    if is_refusal_part(part):
                        refusal_part = part
                        continue
                    if is_citation_part(part):
                        self._attach_response_citation(part, anthropic_content, sink)
                        continue
                    block = self._ir_part_to_anthropic_block(part)
                    if block is not None:
                        anthropic_content.append(block)

                self._default_text_citations(anthropic_content)
                provider_response["content"] = anthropic_content

            # Map finish_reason back to stop_reason
            finish_reason = choice.get("finish_reason", {})
            reason = finish_reason.get("reason", "stop")
            provider_response["stop_reason"] = _anthropic_stop_reason(
                reason,
                ir_response.get("id", ""),
                (message or {}).get("provider_metadata"),
            )

            if "stop_sequence" in finish_reason:
                provider_response["stop_sequence"] = finish_reason["stop_sequence"]

            # Structured refusal: the stop reason says the model declined,
            # ``stop_details`` says why — and only when the turn carries
            # something to say.  The same rule the stream writer's finish
            # delta applies: a part with neither words nor a category is not
            # a structured refusal, and writing one would claim an
            # explanation nobody gave (#1238 review).
            if refusal_part is not None:
                provider_response["stop_reason"] = "refusal"
                pm = refusal_part.get("provider_metadata") or {}
                category = pm.get("anthropic_refusal_category")
                explanation = refusal_part.get("refusal", "")
                if category or explanation:
                    provider_response["stop_details"] = {
                        "type": "refusal",
                        "category": category,
                        "explanation": explanation,
                    }

        # Ensure nullable required fields are always present (spec compliance)
        provider_response.setdefault("stop_sequence", None)
        provider_response.setdefault("stop_details", None)
        if ir_response.get("in_flight"):
            # A message still being produced has no stop reason yet.
            provider_response.setdefault("stop_reason", None)
        provider_response.setdefault("container", None)

        # Usage (always present — Anthropic responses require usage field)
        ir_usage = ir_response.get("usage") or {}
        provider_response["usage"] = self._build_ir_usage_to_p(ir_usage)

        return provider_response

    # ------------------------------------------------------------------
    def _apply_reasoning_config(
        self,
        ir_request: IRRequest,
        result: dict[str, Any],
        ctx: ConversionContext,
    ) -> None:
        """Apply reasoning/thinking config to provider request."""
        reasoning = ir_request.get("reasoning")
        if not reasoning:
            return
        rc_kwargs: dict[str, Any] = {}
        if "reasoning_cap" in ctx.options:
            rc_kwargs["reasoning_cap"] = ctx.options["reasoning_cap"]
        # Pass max_tokens so budget_ratio can derive a
        # default budget when the caller didn't provide one.
        if "max_tokens" in result:
            rc_kwargs["max_tokens"] = result["max_tokens"]
        reasoning_fields = self.config_ops.ir_reasoning_config_to_p(
            reasoning, **rc_kwargs
        )
        result.update(reasoning_fields)

    # Cross-provider consistency helpers
    # ------------------------------------------------------------------

    def _apply_tool_config(
        self,
        ir_request: IRRequest,
        result: dict[str, Any],
        ctx: ConversionContext,
    ) -> None:
        """Apply tools, tool_choice, and tool_config to provider request."""
        # No sink is handed over: the entry point installed this conversion's
        # sink and its leg, and a site that names its own would keep the old
        # routing rule alive beside it.
        tools = ir_request.get("tools")
        if tools:
            result["tools"] = self._get_cached_ir_tools_to_p(tools)

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
                report_loss(
                    "Anthropic does not support max_tool_calls, ignored",
                    subject="tool_config.max_calls",
                )

    @staticmethod
    def _build_p_usage_to_ir(p_usage: dict[str, Any]) -> UsageInfo:
        """Build IR usage dict from Anthropic usage.

        Note: Anthropic's ``cache_read_input_tokens`` and
        ``cache_creation_input_tokens`` are **additive** to
        ``input_tokens`` (not a subset like OpenAI/Google), so the
        total must include them.
        """
        input_tokens = p_usage.get("input_tokens") or 0
        output_tokens = p_usage.get("output_tokens") or 0
        cache_read = p_usage.get("cache_read_input_tokens") or 0
        cache_creation = p_usage.get("cache_creation_input_tokens") or 0
        usage_info: dict[str, Any] = {
            "prompt_tokens": input_tokens,
            "completion_tokens": output_tokens,
            "total_tokens": input_tokens + output_tokens + cache_read + cache_creation,
        }
        if "cache_read_input_tokens" in p_usage:
            usage_info["cache_read_tokens"] = cache_read
        if "cache_creation_input_tokens" in p_usage:
            usage_info["cache_creation_tokens"] = cache_creation
        otd = p_usage.get("output_tokens_details")
        if isinstance(otd, dict):
            thinking = otd.get("thinking_tokens")
            if thinking is not None:
                usage_info["reasoning_tokens"] = thinking
        return cast(UsageInfo, usage_info)

    @staticmethod
    def _build_ir_usage_to_p(ir_usage: Mapping[str, Any]) -> dict[str, Any]:
        """Build Anthropic usage dict from IR usage."""
        reasoning = ir_usage.get("reasoning_tokens")
        usage: dict[str, Any] = {
            "input_tokens": ir_usage.get("prompt_tokens") or 0,
            "output_tokens": ir_usage.get("completion_tokens") or 0,
            "cache_read_input_tokens": ir_usage.get("cache_read_tokens") or 0,
            "cache_creation_input_tokens": ir_usage.get("cache_creation_tokens") or 0,
            "cache_creation": None,
            "output_tokens_details": (
                {"thinking_tokens": reasoning} if reasoning else None
            ),
            "service_tier": None,
            "inference_geo": None,
            "server_tool_use": None,
        }
        return usage

    @staticmethod
    def _build_message_delta_usage(
        ir_usage: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Build Anthropic MessageDeltaUsage dict from IR usage."""
        if ir_usage is None:
            ir_usage = {}
        output_tokens = ir_usage.get("completion_tokens") or 0
        reasoning = ir_usage.get("reasoning_tokens")
        return {
            "output_tokens": output_tokens,
            "input_tokens": ir_usage.get("prompt_tokens") or None,
            "cache_creation_input_tokens": ir_usage.get("cache_creation_tokens")
            or None,
            "cache_read_input_tokens": ir_usage.get("cache_read_tokens") or None,
            "output_tokens_details": (
                {"thinking_tokens": reasoning} if reasoning else None
            ),
            "server_tool_use": None,
        }

    def _capture_preserve_metadata(
        self,
        provider_response: dict[str, Any],
        ir_response: dict[str, Any],
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
            "stop_details",
            "container",
            "usage",
        }
        extras = {
            k: v for k, v in provider_response.items() if k not in _ANTHROPIC_CORE_KEYS
        }
        _USAGE_CORE_KEYS = {
            "input_tokens",
            "output_tokens",
            "cache_read_input_tokens",
            "cache_creation_input_tokens",
            "cache_creation",
            "output_tokens_details",
            "service_tier",
            "inference_geo",
            "server_tool_use",
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

    def _apply_preserve_metadata(
        self,
        provider_response: dict[str, Any],
        ir_response: dict[str, Any],
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
            "stop_details",
            "container",
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

    # ==================== Stream Support ====================

    # --- from_provider ---

    def stream_response_from_provider(
        self,
        chunk: dict[str, Any] | None,
        context: StreamContext | None = None,
    ) -> list[IRStreamEvent]:
        """Convert an Anthropic SSE event to IR stream events.

        Pass ``None`` as the chunk to signal end-of-stream; nothing of this
        dialect is held back across chunks, so it flushes nothing.

        Args:
            chunk: Anthropic SSE event dict (or SDK object), or ``None``
                to signal end-of-stream.

        Returns:
            List of IR stream events extracted from the event.
        """
        if chunk is None:
            # The EOF sentinel: this dialect holds nothing back across chunks
            # (the event that ends its stream carries the usage the Chat
            # reader waits for), so there is nothing to flush.  Callers pass
            # it — ``BaseConverter.iter_stream_from_provider`` and the
            # gateway's per-chunk ``StreamProcessor`` both do — so it must not
            # reach the normalizing below.
            return []

        chunk = self._normalize(chunk)
        events: list[IRStreamEvent] = []

        event_type = chunk.get("type", "")
        handler_name = self._P_TO_IR_DISPATCH.get(event_type)
        if handler_name is not None:
            getattr(self, handler_name)(chunk, context, events)

        return self.finalize_stream_finish(events, context)

    _P_TO_IR_DISPATCH: dict[str, str] = {
        AnthropicEventType.PING: "_handle_p_passthrough_to_ir",
        AnthropicEventType.MESSAGE_START: "_handle_p_message_start_to_ir",
        AnthropicEventType.CONTENT_BLOCK_START: "_handle_p_content_block_start_to_ir",
        AnthropicEventType.CONTENT_BLOCK_DELTA: "_handle_p_content_block_delta_to_ir",
        AnthropicEventType.CONTENT_BLOCK_STOP: "_handle_p_content_block_stop_to_ir",
        AnthropicEventType.MESSAGE_DELTA: "_handle_p_message_delta_to_ir",
        AnthropicEventType.MESSAGE_STOP: "_handle_p_message_stop_to_ir",
    }

    _IR_TO_P_DISPATCH: dict[str, str] = {
        **BaseConverter._IR_TO_P_DISPATCH,
        "tool_result": "_handle_ir_tool_result_to_p",
        "refusal_delta": "_handle_ir_refusal_delta_to_p",
    }
