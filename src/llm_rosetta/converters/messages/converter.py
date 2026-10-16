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

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, cast

from ...types.ir import (
    TextPart,
    is_text_part,
    is_tool_call_part,
    is_tool_result_part,
    is_reasoning_part,
    is_citation_part,
    is_refusal_part,
)
from ...types.ir.request import IRRequest
from ...types.ir.response import IRResponse, UsageInfo
from ...types.ir.stream import (
    CitationDeltaEvent,
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
    ToolResultEvent,
    UsageEvent,
)
from ..base import BaseConverter
from ..base.context import (
    ConversionContext,
    StreamContext,
    carry_registered_call_identity,
)
from ..base.tools import (
    assign_tool_batch_ids,
    fix_orphaned_tool_calls_ir,
    intrinsic_kind_of,
    prefixed_tool_call_id,
    sanitize_tool_call_id,
    strip_orphaned_tool_config,
)
from ..base.helpers.citations import (
    citation_warning_sink,
    report_dropped_citation,
)
from ..base.helpers.extensions import wire_extensions
from ..base.helpers.opaque import (
    get_opaque_origin,
    is_own_origin,
    set_opaque_origin,
)
from ..base.helpers.system_message_hoist import hoist_system_in_converter
from ...transforms.ir import auto_cache_breakpoints, hoist_late_system_messages
from ._constants import (
    ANTHROPIC_REASON_FROM_PROVIDER,
    ANTHROPIC_REASON_TO_PROVIDER,
    AnthropicEventType,
)
from .config_ops import MessagesConfigOps
from .content_ops import MessagesContentOps
from .message_ops import MessagesMessageOps
from .tool_ops import MessagesToolOps, canonical_server_tool_kind

logger = logging.getLogger(__name__)


@dataclass
class _AnthropicBlockState:
    """Track outbound block identity separately from the currently active block."""

    next_index: int = 0
    tool_indices: dict[str, int] = field(default_factory=dict)
    open_indices: set[int] = field(default_factory=set)
    pending_tool_index: int | None = None

    def open(self, context: StreamContext, index: int | None = None) -> int:
        if index is None:
            index = max(self.next_index, context.current_block_index + 1)
        self.next_index = max(self.next_index, index + 1)
        self.open_indices.add(index)
        context.current_block_index = index
        return index


def _stream_blocks(context: StreamContext) -> _AnthropicBlockState:
    """Keep provider-private state on the stream, including generic contexts."""
    state = context.metadata.get("_anthropic_blocks")
    if not isinstance(state, _AnthropicBlockState):
        state = _AnthropicBlockState()
        context.metadata["_anthropic_blocks"] = state
    return state


def _close_content_blocks(context: StreamContext) -> list[dict[str, Any]]:
    """Close every open block once without resetting the allocation watermark."""
    state = _stream_blocks(context)
    if context.current_block_index >= 0:
        state.open(context, context.current_block_index)
    events = [
        {"type": AnthropicEventType.CONTENT_BLOCK_STOP, "index": index}
        for index in sorted(state.open_indices)
    ]
    state.open_indices.clear()
    state.pending_tool_index = None
    context.current_block_index = -1
    context.current_block_type = None
    return events


class MessagesConverter(BaseConverter):
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
    _INTRINSIC_IR_TRANSFORMS = (hoist_late_system_messages(), auto_cache_breakpoints())

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

        # 2. Messages — fix orphaned tool_calls/results at IR level before
        #    conversion.  Anthropic strictly requires bidirectional pairing.
        ir_messages = fix_orphaned_tool_calls_ir(ir_request.get("messages", []))
        # Assign batch_ids over the full list before message_ops runs, so
        # per-format merge_tool_messages can group parallel tool results.
        ir_messages = assign_tool_batch_ids(ir_messages)
        ctx.warnings.extend(strip_orphaned_tool_config(ir_request))

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
        self._apply_reasoning_config(ir_request, result, ctx)

        # 10. Cache config (block-level, warning)
        cache = ir_request.get("cache")
        if cache:
            cache_fields = self.config_ops.ir_cache_config_to_p(cache)
            result.update(cache_fields)

        # 11. Provider extensions (pass-through)
        result.update(wire_extensions(ir_request))

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

        # 3. Tools (with process-level cache)
        tools = provider_request.get("tools")
        if tools:
            ir_request["tools"] = self._get_cached_p_tools_to_ir(tools)

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
            explanation = (
                stop_details.get("explanation", "") or "Request refused by model."
            )
            category = stop_details.get("category")
            refusal_part: dict[str, Any] = {
                "type": "refusal",
                "refusal": explanation or "Request refused by model.",
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
            provider_response["stop_reason"] = ANTHROPIC_REASON_TO_PROVIDER.get(
                reason, "end_turn"
            )

            if "stop_sequence" in finish_reason:
                provider_response["stop_sequence"] = finish_reason["stop_sequence"]

            # Structured refusal: set stop_reason + stop_details
            if refusal_part is not None:
                provider_response["stop_reason"] = "refusal"
                pm = refusal_part.get("provider_metadata") or {}
                category = pm.get("anthropic_refusal_category")
                provider_response["stop_details"] = {
                    "type": "refusal",
                    "category": category,
                    "explanation": refusal_part.get("refusal", ""),
                }

        # Ensure nullable required fields are always present (spec compliance)
        provider_response.setdefault("stop_sequence", None)
        provider_response.setdefault("stop_details", None)
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
        tools = ir_request.get("tools")
        if tools:
            result["tools"] = self._get_cached_ir_tools_to_p(tools)

        tool_choice = ir_request.get("tool_choice")
        if tool_choice:
            result["tool_choice"] = self.tool_ops.ir_tool_choice_to_p(
                tool_choice, warnings=ctx.warnings
            )

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
        handler_name = self._P_TO_IR_DISPATCH.get(event_type)
        if handler_name is not None:
            getattr(self, handler_name)(chunk, context, events)

        return events

    def _handle_p_message_start_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Handle message_start → UsageEvent + StreamStartEvent.

        UsageEvent is emitted first so that downstream ir→p converters
        have ``pending_usage`` populated when they process the
        StreamStartEvent (needed for ``message_start.usage.input_tokens``).
        """
        message = chunk.get("message", {})

        # Emit usage BEFORE stream_start so ir→p can read input_tokens
        # from context.pending_usage when building message_start.
        usage = message.get("usage")
        if usage:
            start_usage = dict(usage)
            start_usage["output_tokens"] = 0
            events.append(
                UsageEvent(
                    type="usage",
                    usage=self._build_p_usage_to_ir(start_usage),
                )
            )

        if context is not None:
            response_id = message.get("id", "")
            model = message.get("model", "")
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

    def _handle_p_content_block_start_to_ir(
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
            if block_type == "server_tool_use":
                # Same fold as the block reader: a newer code-execution version
                # reports itself as ``bash_code_execution``, a kind no shim
                # declares.  Folding only on the non-streaming path left the
                # two disagreeing for the same response.
                tool_name = canonical_server_tool_kind(tool_name)

            start_evt = ToolCallStartEvent(
                type="tool_call_start",
                tool_call_id=tool_call_id,
                tool_name=tool_name,
            )
            if block_type == "server_tool_use":
                # The event has to say which it was, exactly as
                # ``p_tool_call_to_ir`` does for the non-streaming part: mark
                # it intrinsic and carry the canonical kind, so the writer
                # restores ``server_tool_use`` without inferring it from the
                # name.  ``tool_name`` is already folded above.
                start_evt["tool_type"] = "intrinsic"
                start_evt["intrinsic_kind"] = tool_name
            # The same two fields ``p_tool_call_to_ir`` keeps on the
            # non-streaming part, so a streamed call does not lose what the
            # block carries: a cache breakpoint, and a non-``direct`` caller
            # (the default caller is not re-emitted, as on the other path).
            cache_control = content_block.get("cache_control")
            if cache_control is not None:
                start_evt["cache_hint"] = cache_control
            caller = content_block.get("caller")
            if isinstance(caller, dict) and caller.get("type") != "direct":
                start_evt.setdefault("provider_metadata", {})["anthropic_caller"] = (
                    caller
                )
            if context is not None:
                # Register what the block said, not only its name: the deltas
                # that follow are keyed on the call id alone, so the kind (a
                # provider-executed call's whole identity) and the metadata the
                # block carried are read back from here onto each delta.
                context.register_tool_call(
                    tool_call_id,
                    tool_name,
                    tool_type=start_evt.get("tool_type", "function"),
                    provider_metadata=start_evt.get("provider_metadata"),
                    intrinsic_kind=start_evt.get("intrinsic_kind", ""),
                )
                context.block_index_to_tool_call_id[block_index] = tool_call_id
                start_evt["tool_call_index"] = context.tool_call_count - 1
            events.append(start_evt)

        elif isinstance(block_type, str) and block_type.endswith("_tool_result"):
            # Anthropic executes server tools and returns their results whole
            # in a single ``content_block_start`` (no deltas follow).  Build the
            # IR part with the same reader the non-streaming path uses so the
            # payload shape matches, then carry it as a one-shot ToolResultEvent.
            part = MessagesToolOps.p_server_tool_result_to_ir(content_block)
            result_evt = ToolResultEvent(
                type="tool_result",
                tool_call_id=part["tool_call_id"],
                result=part["result"],
                is_error=part.get("is_error", False),
            )
            tool_type = part.get("tool_type")
            if tool_type is not None:
                result_evt["tool_type"] = tool_type
            # Carry the canonical kind on the event's first-class field, the
            # same carrier ``p_server_tool_result_to_ir`` puts it on.
            if "intrinsic_kind" in part:
                result_evt["intrinsic_kind"] = part["intrinsic_kind"]
            if "cache_hint" in part:
                result_evt["cache_hint"] = part["cache_hint"]
            if "provider_metadata" in part:
                result_evt["provider_metadata"] = part["provider_metadata"]
            result_evt["block_index"] = block_index
            events.append(result_evt)

    def _handle_p_content_block_delta_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Handle content_block_delta → TextDeltaEvent, ToolCallDeltaEvent, or ReasoningDeltaEvent."""
        delta = chunk.get("delta", {})
        delta_type = delta.get("type", "")
        # Preserve provider block index on IR events for lossless round-trip (#246)
        chunk_block_index: int | None = chunk.get("index")

        if delta_type == "text_delta":
            evt = TextDeltaEvent(
                type="text_delta",
                text=delta.get("text", ""),
            )
            if chunk_block_index is not None:
                evt["block_index"] = chunk_block_index
            events.append(evt)
        elif delta_type == "input_json_delta":
            tool_call_id = ""
            if context is not None and chunk_block_index is not None:
                tool_call_id = context.block_index_to_tool_call_id.get(
                    chunk_block_index, ""
                )
            if not tool_call_id and context is not None and context.tool_call_id_map:
                tool_call_id = list(context.tool_call_id_map.keys())[-1]

            partial_json = delta.get("partial_json", "")
            delta_evt = ToolCallDeltaEvent(
                type="tool_call_delta",
                tool_call_id=tool_call_id,
                arguments_delta=partial_json,
            )
            if chunk_block_index is not None:
                delta_evt["block_index"] = chunk_block_index
            if context is not None and tool_call_id:
                # The arguments arrive as fragments that carry no identity of
                # their own, so each repeats what the call registered (kind and
                # metadata), as a start event would have carried it.
                carry_registered_call_identity(delta_evt, context, tool_call_id)
            events.append(delta_evt)

            if context is not None and tool_call_id:
                context.append_tool_call_args(tool_call_id, partial_json)
        elif delta_type == "thinking_delta":
            evt_r = ReasoningDeltaEvent(
                type="reasoning_delta",
                reasoning=delta.get("thinking", ""),
            )
            if chunk_block_index is not None:
                evt_r["block_index"] = chunk_block_index
            events.append(evt_r)
        elif delta_type == "citations_delta":
            self._citations_delta_to_ir(delta, chunk_block_index, events)
        elif delta_type == "signature_delta":
            sig_delta = delta.get("signature", "")
            evt_s = ReasoningDeltaEvent(
                type="reasoning_delta",
                reasoning="",
                signature=sig_delta,
            )
            if sig_delta:
                set_opaque_origin(evt_s, signature="messages")
            if chunk_block_index is not None:
                evt_s["block_index"] = chunk_block_index
            events.append(evt_s)

    def _citations_delta_to_ir(
        self,
        delta: dict[str, Any],
        block_index: int | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Anthropic's own citation delta → one IR citation event.

        The citation arrives before the text it supports, and the block it
        annotates opened with an empty ``citations`` array.
        """
        citation = delta.get("citation")
        if not isinstance(citation, dict):
            return
        event = CitationDeltaEvent(
            type="citation_delta",
            citation=self.content_ops.p_citation_to_ir(citation),
        )
        if block_index is not None:
            event["block_index"] = block_index
        events.append(event)

    def _handle_p_content_block_stop_to_ir(
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

    def _handle_p_message_delta_to_ir(
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
            events.append(
                UsageEvent(
                    type="usage",
                    usage=self._build_p_usage_to_ir(usage),
                )
            )

        if stop_reason:
            reason = ANTHROPIC_REASON_FROM_PROVIDER.get(stop_reason, "stop")
            finish: dict[str, Any] = {
                "type": "finish",
                "finish_reason": {"reason": reason},
            }
            # Structured refusal: extract explanation and category
            if stop_reason == "refusal":
                stop_details = delta.get("stop_details") or {}
                explanation = (
                    stop_details.get("explanation", "") or "Request refused by model."
                )
                category = stop_details.get("category")
                pm: dict[str, Any] = {"refusal_text": explanation}
                if category:
                    pm["anthropic_refusal_category"] = category
                finish["provider_metadata"] = pm
            events.append(finish)  # ty: ignore[invalid-argument-type]

    def _handle_p_message_stop_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Handle message_stop → StreamEndEvent."""
        if context is not None:
            context.mark_ended()
            events.append(StreamEndEvent(type="stream_end"))

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
    }

    # --- to_provider ---

    def _handle_ir_stream_start_to_p(
        self, event: StreamStartEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle StreamStartEvent → message_start."""
        ir_usage: Mapping[str, Any] = {}
        if context is not None:
            context.response_id = event["response_id"]
            context.model = event["model"]
            context.mark_started()
            if context.pending_usage is not None:
                ir_usage = context.pending_usage
            elif context.preflight_usage is not None:
                ir_usage = context.preflight_usage

        p_usage = self._build_ir_usage_to_p(ir_usage)

        return {
            "type": AnthropicEventType.MESSAGE_START,
            "message": {
                "id": self.add_response_id_prefix(
                    event["response_id"], self._get_response_id_prefix(context)
                ),
                "type": "message",
                "role": "assistant",
                "model": event["model"],
                "content": [],
                "stop_reason": None,
                "stop_sequence": None,
                "stop_details": None,
                "container": None,
                "usage": p_usage,
            },
        }

    def _handle_ir_stream_end_to_p(
        self, event: StreamEndEvent, context: StreamContext | None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Handle StreamEndEvent → message_stop (with optional pending finish flush)."""
        results: list[dict[str, Any]] = []
        if context is not None:
            results.extend(_close_content_blocks(context))
            # Flush any buffered finish that never got a UsageEvent
            finish = context.pop_pending_finish()
            if finish is not None:
                usage = context.pop_pending_usage()
                results.append(
                    {
                        "type": AnthropicEventType.MESSAGE_DELTA,
                        "delta": finish,
                        "usage": self._build_message_delta_usage(usage),
                    }
                )
            context.mark_ended()
        results.append({"type": AnthropicEventType.MESSAGE_STOP})
        return results if len(results) > 1 else results[0]

    def _handle_ir_content_block_start_to_p(
        self, event: ContentBlockStartEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ContentBlockStartEvent → content_block_start."""
        block_index = event["block_index"]
        block_type = event["block_type"]

        if context is not None:
            # Anchor context to the explicit block_index from the IR event
            # instead of auto-incrementing, so subsequent deltas that read
            # context.current_block_index stay in sync.  (#246)
            state = _stream_blocks(context)
            state.open(context, block_index)
            state.pending_tool_index = (
                block_index if block_type in ("tool_use", "server_tool_use") else None
            )
            context.current_block_type = block_type

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

    def _handle_ir_content_block_end_to_p(
        self, event: ContentBlockEndEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ContentBlockEndEvent → content_block_stop."""
        if context is not None:
            index = event["block_index"]
            state = _stream_blocks(context)
            state.open_indices.discard(index)
            state.next_index = max(state.next_index, index + 1)
            if state.pending_tool_index == index:
                state.pending_tool_index = None
            if context.current_block_index == index:
                context.current_block_index = -1
                context.current_block_type = None
        return {
            "type": AnthropicEventType.CONTENT_BLOCK_STOP,
            "index": event["block_index"],
        }

    def _handle_ir_text_delta_to_p(
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
        # Prefer explicit block_index from the IR event (#246);
        # fall back to context for providers that don't emit block indexes.
        explicit_idx: int | None = event.get("block_index")
        if explicit_idx is not None:
            result["index"] = explicit_idx
            if context is not None:
                _stream_blocks(context).open(context, explicit_idx)
                context.current_block_type = "text"
        elif context is not None:
            needs_new_block = context.current_block_index < 0 or (
                context.current_block_type is not None
                and context.current_block_type != "text"
            )
            if needs_new_block:
                preamble = _close_content_blocks(context)
                _stream_blocks(context).open(context)
                context.current_block_type = "text"
                result["index"] = context.current_block_index
                preamble.append(
                    {
                        "type": AnthropicEventType.CONTENT_BLOCK_START,
                        "index": context.current_block_index,
                        "content_block": {"type": "text", "text": ""},
                    }
                )
                preamble.append(result)
                return preamble
            result["index"] = context.current_block_index
        return result

    def _handle_ir_citation_delta_to_p(
        self, event: CitationDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Handle CitationDeltaEvent → content_block_delta (citations_delta).

        Anthropic's own citations hang off a text block, so the delta is
        addressed to the block the citation annotates: the one the event names,
        or the block currently open.
        """
        citation = self.content_ops.ir_citation_to_p(event["citation"])
        if citation is None:
            report_dropped_citation(
                event["citation"], warnings=getattr(context, "warnings", None)
            )
            return {}
        index = event.get("block_index")
        if index is None and context is not None:
            index = context.current_block_index
        if index is None:
            return {}
        return {
            "type": AnthropicEventType.CONTENT_BLOCK_DELTA,
            "index": index,
            "delta": {"type": "citations_delta", "citation": citation},
        }

    def _handle_ir_reasoning_delta_to_p(
        self, event: ReasoningDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Handle ReasoningDeltaEvent → content_block_delta (thinking and/or signature)."""
        deltas = self._ir_reasoning_deltas(event)
        if not deltas:
            return {}
        rd_results: list[dict[str, Any]] = [
            {"type": AnthropicEventType.CONTENT_BLOCK_DELTA, "delta": delta}
            for delta in deltas
        ]
        # Prefer explicit block_index from the IR event (#246)
        explicit_idx: int | None = event.get("block_index")
        if explicit_idx is not None:
            for result in rd_results:
                result["index"] = explicit_idx
            if context is not None:
                _stream_blocks(context).open(context, explicit_idx)
                context.current_block_type = "thinking"
        elif context is not None:
            needs_new_block = context.current_block_index < 0 or (
                context.current_block_type is not None
                and context.current_block_type != "thinking"
            )
            if needs_new_block:
                preamble = _close_content_blocks(context)
                _stream_blocks(context).open(context)
                context.current_block_type = "thinking"
                index = context.current_block_index
                for result in rd_results:
                    result["index"] = index
                return [
                    *preamble,
                    {
                        "type": AnthropicEventType.CONTENT_BLOCK_START,
                        "index": index,
                        "content_block": {"type": "thinking", "thinking": ""},
                    },
                    *rd_results,
                ]
            for result in rd_results:
                result["index"] = context.current_block_index
        return rd_results[0] if len(rd_results) == 1 else rd_results

    def _ir_reasoning_deltas(self, event: ReasoningDeltaEvent) -> list[dict[str, Any]]:
        """Split an IR reasoning delta event into Anthropic ``delta`` payloads.

        Anthropic carries the visible thinking text and the verification
        signature as separate deltas, so an event holding both yields two
        payloads: the text first, then the signature.  A foreign signature is
        dropped without taking the text along with it.

        Both fields are tested for emptiness rather than presence: an empty
        signature is not a signature, and emitting an empty ``signature_delta``
        would be a delta with no content.  The Interactions writer drops empty
        signatures the same way.

        Dropping an all-empty event keeps the client from ever seeing an empty
        thinking block.  That matters because an Anthropic client replays what
        it receives, and a thinking block cannot be sent back without a
        signature — the API answers ``400
        messages.N.content.0.thinking.signature: Field required`` — so the
        response side must not hand one out.  The request side drops the same
        shape in ``MessagesContentOps.ir_reasoning_to_p``.

        Two things this does not cover.  A non-empty reasoning part with no
        signature still leaves the request path unsigned and is rejected the
        same way; suppressing that is the shim's ``unsigned_blocks: preserve``
        policy, which ``anthropic--messages`` does not declare yet (#954).  And
        the Chat Completions writer keeps the opposite rule on purpose —
        DeepSeek's thinking mode needs ``reasoning_content`` present even when
        it is empty (#948) — so the two are not a divergence to unify.

        Args:
            event: IR reasoning delta event.

        Returns:
            ``content_block_delta`` payloads to emit, in order.
        """
        signature = event.get("signature")
        if signature and not is_own_origin(
            get_opaque_origin(event, "signature"), self._CONVERTER_TAG
        ):
            # Foreign verification signature — not valid for Anthropic.
            signature = None
        deltas: list[dict[str, Any]] = []
        if event["reasoning"]:
            deltas.append({"type": "thinking_delta", "thinking": event["reasoning"]})
        if signature:
            deltas.append({"type": "signature_delta", "signature": signature})
        return deltas

    def _handle_ir_tool_call_start_to_p(
        self, event: ToolCallStartEvent, context: StreamContext | None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Handle ToolCallStartEvent → content_block_start for a tool call.

        Mirrors ``MessagesToolOps.ir_tool_call_to_p``: a call the server
        executed (``tool_type == "intrinsic"``) is re-emitted as
        ``server_tool_use`` with its canonical kind as the block name, and
        everything else as ``tool_use``.  The kind comes from the event's
        first-class ``intrinsic_kind`` field (legacy metadata still accepted),
        never from the tool name.

        A server call Anthropic has no server tool for is dropped — never
        downgraded to a ``tool_use`` the client would be asked to run — and
        the call id is remembered so the fragments that follow it are dropped
        with it rather than attached to whatever block is open.
        """
        block_type = "tool_use"
        tool_name = event["tool_name"]
        if event.get("tool_type", "function") == "intrinsic":
            kind = intrinsic_kind_of(event)
            if not self.tool_ops.intrinsic_renderable(kind, {}, leg="response"):
                logger.warning(
                    "Server tool call of kind %r has no Anthropic server tool; "
                    "dropped from the streamed response",
                    kind,
                )
                if context is not None:
                    dropped = context.metadata.setdefault(
                        "_dropped_intrinsic_calls", set()
                    )
                    dropped.add(event["tool_call_id"])
                return {}
            block_type = "server_tool_use"
            tool_name = kind
        result: dict[str, Any] = {
            "type": AnthropicEventType.CONTENT_BLOCK_START,
            "content_block": {
                "type": block_type,
                "id": (
                    prefixed_tool_call_id(event["tool_call_id"], "srvtoolu_")
                    if block_type == "server_tool_use"
                    else sanitize_tool_call_id(event["tool_call_id"])
                ),
                "name": tool_name,
                "input": {},
            },
        }
        # Mirrors ``ir_tool_call_to_p``'s tail: the cache breakpoint always, and
        # a non-default ``caller`` only on a client ``tool_use`` — the
        # non-streaming writer makes the same restriction.
        block = result["content_block"]
        cache_hint = event.get("cache_hint")
        if cache_hint is not None:
            block["cache_control"] = cache_hint
        caller = (event.get("provider_metadata") or {}).get("anthropic_caller")
        if caller is not None and block_type == "tool_use":
            block["caller"] = caller
        if context is not None:
            state = _stream_blocks(context)
            tool_id = event["tool_call_id"]
            if state.tool_indices.get(tool_id) in state.open_indices:
                return {}
            preamble: list[dict[str, Any]] = []
            if context.current_block_type not in (None, "tool_use", "server_tool_use"):
                preamble = _close_content_blocks(context)
            index = state.pending_tool_index
            state.pending_tool_index = None
            if index is None and context.current_block_type is None:
                # Honor a caller-preallocated generic context index.
                if context.current_block_index >= state.next_index:
                    index = context.current_block_index
            index = state.open(context, index)
            state.tool_indices[tool_id] = index
            context.current_block_type = block_type
            result["index"] = index
            if preamble:
                preamble.append(result)
                return preamble
        return result

    def _handle_ir_tool_call_delta_to_p(
        self, event: ToolCallDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ToolCallDeltaEvent → content_block_delta with input_json_delta.

        A fragment whose call was dropped at its start belongs to no block:
        writing it would open one for a server call the client was never told
        about, so it is dropped too.
        """
        if context is not None:
            dropped = context.metadata.get("_dropped_intrinsic_calls")
            if dropped and event["tool_call_id"] in dropped:
                return {}
        result: dict[str, Any] = {
            "type": AnthropicEventType.CONTENT_BLOCK_DELTA,
            "delta": {
                "type": "input_json_delta",
                "partial_json": event["arguments_delta"],
            },
        }
        # Prefer explicit block_index from the IR event (#246)
        explicit_idx: int | None = event.get("block_index")
        if context is not None:
            state = _stream_blocks(context)
            index = explicit_idx
            if index is None:
                index = state.tool_indices.get(event["tool_call_id"])
            if index is None and context.current_block_index >= 0:
                index = context.current_block_index
            result["index"] = state.open(context, index)
            context.current_block_type = "tool_use"
        elif explicit_idx is not None:
            result["index"] = explicit_idx
        return result

    def _handle_ir_tool_result_to_p(
        self, event: ToolResultEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ToolResultEvent → content_block_start for a server-tool result.

        A server-tool result arrives whole, so the writer emits the matching
        ``*_tool_result`` content block in a single ``content_block_start``.
        When the block was read from this same dialect, the stored ``_native``
        payload is replayed verbatim, preserving fields the IR does not model
        (the original ``content`` shape and ``caller``).  Otherwise the block is
        rebuilt from the IR fields.

        A server result for a kind Anthropic has no server tool for is dropped
        with a warning, the same way its call is: the client can neither run
        the tool nor recognize its result.
        """
        if event.get("tool_type") == "intrinsic" and not (
            self.tool_ops.intrinsic_renderable(
                intrinsic_kind_of(event), None, leg="response"
            )
        ):
            logger.warning(
                "Server tool result of kind %r has no Anthropic server tool; "
                "dropped from the streamed response",
                intrinsic_kind_of(event),
            )
            return {}
        content_block = self._ir_tool_result_block(event)

        result: dict[str, Any] = {
            "type": AnthropicEventType.CONTENT_BLOCK_START,
            "content_block": content_block,
        }
        index = event.get("block_index")
        if context is not None:
            state = _stream_blocks(context)
            index = state.open(context, index)
            context.current_block_type = content_block.get("type")
            result["index"] = index
        elif index is not None:
            result["index"] = index
        return result

    def _ir_tool_result_block(self, event: ToolResultEvent) -> dict[str, Any]:
        """Build the Anthropic ``*_tool_result`` block for a ToolResultEvent.

        Shares the writer with the non-streaming assistant turn
        (``MessagesToolOps.ir_server_tool_result_to_p``), so a same-dialect read
        restores the stored ``_native`` block and a foreign one is rebuilt from
        the IR fields.  A ``ToolResultEvent`` is always a server result, so the
        writer never returns ``None`` here.
        """
        block = MessagesToolOps.ir_server_tool_result_to_p(event)
        if block is None:  # pragma: no cover - defensive; events are intrinsic
            # Generic ``tool_result`` shape: the reader only emits this event for
            # ``*_tool_result`` blocks, but if one ever arrives without an
            # intrinsic kind keep the full block — both fields are valid on the
            # generic block, unlike on a ``*_tool_result`` one (#973).
            block = {
                "type": "tool_result",
                "tool_use_id": sanitize_tool_call_id(event["tool_call_id"]),
                "content": event.get("result", []),
            }
            if event.get("is_error") is not None:
                block["is_error"] = event["is_error"]
            cache_hint = event.get("cache_hint")
            if cache_hint is not None:
                block["cache_control"] = cache_hint
        return block

    def _handle_ir_finish_to_p(
        self, event: FinishEvent, context: StreamContext | None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Handle FinishEvent → buffered message_delta + optional content_block_stop."""
        reason = event["finish_reason"]["reason"]
        stop_reason = ANTHROPIC_REASON_TO_PROVIDER.get(reason, "end_turn")

        # Build stop_details for structured refusal
        stop_details = None
        if reason == "refusal":
            pm = event.get("provider_metadata") or {}
            stop_details = {
                "type": "refusal",
                "category": pm.get("anthropic_refusal_category"),
                "explanation": pm.get("refusal_text", ""),
            }

        if context is not None:
            results = _close_content_blocks(context)
            usage = context.pop_pending_usage()
            if usage is not None:
                delta_payload: dict[str, Any] = {"stop_reason": stop_reason}
                if stop_details:
                    delta_payload["stop_details"] = stop_details
                delta_payload.setdefault("stop_sequence", None)
                delta_payload.setdefault("stop_details", None)
                delta_payload.setdefault("container", None)
                results.append(
                    {
                        "type": AnthropicEventType.MESSAGE_DELTA,
                        "delta": delta_payload,
                        "usage": self._build_message_delta_usage(usage),
                    }
                )
            else:
                # Buffer finish for later UsageEvent or StreamEnd flush.
                finish_delta: dict[str, Any] = {"stop_reason": stop_reason}
                if stop_details:
                    finish_delta["stop_details"] = stop_details
                finish_delta.setdefault("stop_sequence", None)
                finish_delta.setdefault("stop_details", None)
                finish_delta.setdefault("container", None)
                context.buffer_finish(finish_delta)
            return results if results else {}
        return {
            "type": AnthropicEventType.MESSAGE_DELTA,
            "delta": {
                "stop_reason": stop_reason,
                "stop_sequence": None,
                "stop_details": None,
                "container": None,
            },
            "usage": self._build_message_delta_usage(),
        }

    def _handle_ir_usage_to_p(
        self, event: UsageEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle UsageEvent → message_delta (merged with pending finish).

        When a pending_finish exists, merges usage into it and emits a
        message_delta immediately.  Otherwise buffers the usage for a
        later FinishEvent or StreamEndEvent to consume, preventing the
        extra empty message_delta that caused round-trip inflation.
        """
        usage = event["usage"]
        if context is not None:
            delta = context.pop_pending_finish()
            if delta is not None:
                return {
                    "type": AnthropicEventType.MESSAGE_DELTA,
                    "delta": delta,
                    "usage": self._build_message_delta_usage(usage),
                }
            # No pending finish — buffer for later merge.
            context.buffer_usage(usage)
            return {}
        return {
            "type": AnthropicEventType.MESSAGE_DELTA,
            "delta": {},
            "usage": self._build_message_delta_usage(usage),
        }


# Backward-compatible alias (deprecated) — the base type was renamed
# ``anthropic`` → ``messages``.
AnthropicConverter = MessagesConverter
