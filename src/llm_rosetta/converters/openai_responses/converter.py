"""
LLM-Rosetta - OpenAI Responses API Converter

Top-level converter implementing the 6 explicit interfaces.
Composes ContentOps, ToolOps, MessageOps, and ConfigOps for full bidirectional
conversion between IR and OpenAI Responses API format.

Note: Responses API uses a flat list of items (input/output) instead of
nested messages. The converter handles this structural difference.
"""

import logging
import time
from collections import defaultdict
from collections.abc import Mapping
from typing import Any, cast

from ...types.ir import (
    TextPart,
    is_text_part,
    is_tool_call_part,
    is_reasoning_part,
    is_refusal_part,
)
from ...types.ir.passthrough import ProviderPassthroughEvent, ProviderPassthroughItem
from ...types.ir.request import IRRequest
from ...types.ir.response import IRResponse, UsageInfo
from ...types.ir.stream import (
    ContentBlockEndEvent,
    ContentBlockStartEvent,
    FinishEvent,
    IRStreamEvent,
    ReasoningDeltaEvent,
    RefusalDeltaEvent,
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
from .stream_context import OpenAIResponsesStreamContext
from ._constants import (
    RESPONSES_INCOMPLETE_REASON_TO_IR,
    RESPONSES_PRESERVE_FIELDS,
    RESPONSES_REASON_TO_INCOMPLETE_REASON,
    RESPONSES_REASON_TO_STATUS,
    RESPONSES_REQUIRED_DEFAULTS,
    RESPONSES_STATUS_TO_REASON,
    ResponsesEventType,
    generate_message_id,
    generate_reasoning_id,
)
from .config_ops import OpenAIResponsesConfigOps
from .content_ops import OpenAIResponsesContentOps
from .message_ops import OpenAIResponsesMessageOps
from .tool_ops import OpenAIResponsesToolOps
from .utils import build_message_preamble_events, resolve_call_id


def _capture_item_metadata(item: dict[str, Any]) -> dict[str, Any]:
    """Extract per-output-item metadata for lossless round-trip."""
    meta: dict[str, Any] = {}
    if "id" in item:
        meta["id"] = item["id"]
    if "status" in item:
        meta["status"] = item["status"]
    item_content = item.get("content", [])
    if isinstance(item_content, list):
        parts_meta: list[dict[str, Any]] = []
        for cp in item_content:
            if not isinstance(cp, dict):
                continue
            pm: dict[str, Any] = {}
            if "annotations" in cp:
                pm["annotations"] = cp["annotations"]
            if "logprobs" in cp:
                pm["logprobs"] = cp["logprobs"]
            parts_meta.append(pm)
        if parts_meta:
            meta["content_meta"] = parts_meta
    return meta


logger = logging.getLogger(__name__)

_MAX_TOOL_NAME_LEN = 64


def _qualify_tool_name(namespace: str, name: str, used_names: set[str]) -> str | None:
    """Build a qualified ``{namespace}_{name}`` that fits in 64 chars."""
    qualified = f"{namespace}_{name}"
    if len(qualified) > _MAX_TOOL_NAME_LEN:
        max_ns = _MAX_TOOL_NAME_LEN - len(name) - 1
        if max_ns > 0:
            qualified = f"{namespace[:max_ns]}_{name}"
        else:
            logger.warning(
                "Tool name %r too long to qualify with namespace %r",
                name,
                namespace,
            )
            return None
    if qualified in used_names:
        logger.warning("Qualified name %r still collides; keeping %r", qualified, name)
        return None
    return qualified


class OpenAIResponsesConverter(BaseConverter):
    """OpenAI Responses API converter.

    Implements the 6 explicit conversion interfaces defined by BaseConverter.

    Uses composition of Ops classes for modular, testable conversion logic.

    Note: Responses API uses ``input`` for request items and ``output`` for
    response items, with a flat item list structure.
    """

    content_ops_class = OpenAIResponsesContentOps
    tool_ops_class = OpenAIResponsesToolOps
    message_ops_class = OpenAIResponsesMessageOps
    config_ops_class = OpenAIResponsesConfigOps
    _CONVERTER_TAG = "openai_responses"
    _PASSTHROUGH_RESTORE_KEY = "output"

    # Default response ID prefix for the OpenAI Responses format.
    # Used as fallback when no shim-driven prefix is available in the
    # conversion context.
    _RESPONSE_ID_PREFIX = "resp_"

    def _get_response_id_prefix(
        self, context: ConversionContext | StreamContext | None = None
    ) -> str:
        """Return the response ID prefix from context or class default.

        Reads ``response_id_prefix`` from ``context.options`` if set by
        the conversion pipeline (populated from the provider shim's
        ``response_id_prefix`` field).  Falls back to the class constant
        ``_RESPONSE_ID_PREFIX`` for direct converter usage without a
        pipeline.
        """
        if context is not None:
            prefix = context.options.get("response_id_prefix")
            if prefix is not None:
                return prefix
        return self._RESPONSE_ID_PREFIX

    @staticmethod
    def _completion_timestamp() -> int:
        """Current Unix timestamp for completed_at fields."""
        return int(time.time())

    def __init__(self):
        self.content_ops = self.content_ops_class()
        self.tool_ops = self.tool_ops_class()
        self.message_ops = self.message_ops_class(self.content_ops, self.tool_ops)
        self.config_ops = self.config_ops_class()

    @classmethod
    def create_stream_context(cls) -> OpenAIResponsesStreamContext:
        """Create a stream context with Responses API specific state."""
        return OpenAIResponsesStreamContext()

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

        # 1. System instruction → instructions field (plain string)
        system_parts = ir_request.get("system_instruction")
        if system_parts:
            result["instructions"] = " ".join(
                p["text"] for p in system_parts if is_text_part(p)
            )

        # 2. Messages → input items — fix orphaned tool_calls at IR level
        # before conversion.  OpenAI Responses API strictly requires every
        # function_call to have a matching function_call_output.
        ir_messages = fix_orphaned_tool_calls_ir(ir_request.get("messages", []))
        ctx.warnings.extend(strip_orphaned_tool_config(ir_request))
        items, msg_warnings = self.message_ops.ir_messages_to_p(
            ir_messages, target_provider=self._CONVERTER_TAG
        )
        ctx.warnings.extend(msg_warnings)
        result["input"] = items

        # 3-5. Tools + tool_choice + tool_config
        self._apply_tool_config(ir_request, result, ctx)

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
            rc_kwargs: dict[str, Any] = {}
            if ctx and "reasoning_cap" in ctx.options:
                rc_kwargs["reasoning_cap"] = ctx.options["reasoning_cap"]
            reasoning_fields = self.config_ops.ir_reasoning_config_to_p(
                reasoning, **rc_kwargs
            )
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

        # 1. Instructions → system_instruction (list[TextPart])
        instructions = provider_request.get("instructions")
        if instructions:
            ir_request["system_instruction"] = [
                TextPart(type="text", text=instructions)
            ]

        # 2. Input items → messages
        input_items = provider_request.get("input", [])
        if isinstance(input_items, str):
            input_items = [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": input_items}],
                }
            ]
        if isinstance(input_items, list):
            ir_messages = self.message_ops.p_messages_to_ir(input_items)
            ir_request["messages"] = ir_messages

        # 3. Tools (with process-level cache)
        # Filter out disabled tools before caching (external_web_access=False)
        tools = provider_request.get("tools")
        if tools:
            active_tools = [
                t
                for t in tools
                if not (isinstance(t, dict) and t.get("external_web_access") is False)
            ]
            if active_tools:
                ir_tools = self._get_cached_p_tools_to_ir(active_tools)
                if ir_tools:
                    ir_tools = self._dedup_ir_tool_names(ir_tools)
                    ir_request["tools"] = ir_tools

        # 4-5. Tool choice + tool config
        self._convert_p_tool_config_to_ir(provider_request, ir_request)

        # 6. Generation config
        gen_config = self.config_ops.p_generation_config_to_ir(provider_request)
        if gen_config:
            ir_request["generation"] = gen_config

        # 7. Response format (text field in Responses API)
        text_format = provider_request.get("text")
        if text_format:
            ir_request["response_format"] = self.config_ops.p_response_format_to_ir(
                text_format
            )

        # 8. Reasoning config
        reasoning = self.config_ops.p_reasoning_config_to_ir(provider_request)
        if reasoning:
            ir_request["reasoning"] = reasoning

        # 9. Stream config
        stream = provider_request.get("stream")
        stream_options = provider_request.get("stream_options")
        if stream is not None or stream_options:
            ir_request["stream"] = self.config_ops.p_stream_config_to_ir(
                {"stream": stream, "stream_options": stream_options}
            )

        # 10. Cache config
        self._convert_p_cache_to_ir(provider_request, ir_request)

        # 11. Provider extensions (passthrough fields like allowed_tools)
        allowed_tools = provider_request.get("allowed_tools")
        if allowed_tools is not None:
            ir_request.setdefault("provider_extensions", {})["allowed_tools"] = (
                allowed_tools
            )

        # Preserve mode: capture request fields for echo-back in response
        ctx = context if context is not None else ConversionContext()
        if ctx.metadata_mode == "preserve":
            echo = {
                k: v
                for k, v in provider_request.items()
                if k in RESPONSES_PRESERVE_FIELDS
            }
            if echo:
                ctx.store_request_echo(echo)

        return ir_request

    def _do_response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:

        choices = []
        output_items = provider_response.get("output", [])

        # Determine finish reason from status
        status = provider_response.get("status")
        if status == "incomplete":
            incomplete_details = provider_response.get("incomplete_details", {})
            inc_reason = (
                incomplete_details.get("reason", "")
                if isinstance(incomplete_details, dict)
                else ""
            )
            finish_reason_val = RESPONSES_INCOMPLETE_REASON_TO_IR.get(
                inc_reason, "stop"
            )
        else:
            finish_reason_val = RESPONSES_STATUS_TO_REASON.get(status or "", "stop")

        # Convert output items to IR message content
        ir_items = self.message_ops.p_messages_to_ir(output_items)

        # Collect all content parts into a single assistant message
        message_content: list[dict[str, Any]] = []
        message_metadata: dict[str, Any] = {}
        for ir_item in ir_items:
            if isinstance(ir_item, dict) and "role" in ir_item:
                content = ir_item.get("content", [])
                message_content.extend(cast(list, content))
                # Capture provider_metadata from first assistant message
                pm = ir_item.get("provider_metadata")
                if pm and not message_metadata:
                    message_metadata = pm
            elif isinstance(ir_item, dict) and "type" in ir_item:
                pass

        if message_content:
            ir_msg: dict[str, Any] = {
                "role": "assistant",
                "content": message_content,
            }
            if message_metadata:
                ir_msg["provider_metadata"] = message_metadata
            choices.append(
                {
                    "index": 0,
                    "message": ir_msg,
                    "finish_reason": {"reason": finish_reason_val},
                }
            )

        ir_response: dict[str, Any] = {
            "id": self.strip_response_id_prefix(
                provider_response.get("id", ""),
                self._get_response_id_prefix(context),
            ),
            "object": "response",
            "created": int(provider_response.get("created_at", 0)),
            "model": provider_response.get("model", ""),
            "choices": choices,
        }
        passthrough_items = [
            ProviderPassthroughItem(
                type="provider_passthrough_item",
                provider=self._CONVERTER_TAG,
                payload=dict(item),
                position=position,
            )
            for position, item in enumerate(output_items)
            if isinstance(item, dict)
            and not self.message_ops.is_portable_response_output_item(item)
        ]
        if passthrough_items:
            ir_response["provider_passthrough_items"] = passthrough_items

        # Handle usage
        p_usage = provider_response.get("usage")
        if p_usage:
            ir_response["usage"] = self._build_p_usage_to_ir(p_usage)

        if provider_response.get("service_tier") is not None:
            ir_response["service_tier"] = provider_response["service_tier"]

        return ir_response

    def _do_response_to_provider(
        self,
        ir_response: IRResponse,
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        response_id_stem = ir_response.get("id", "")
        provider_response: dict[str, Any] = {
            "id": self.add_response_id_prefix(
                response_id_stem,
                self._get_response_id_prefix(context),
            ),
            "object": "response",
            "created_at": ir_response.get("created", 0),
            "model": ir_response.get("model", ""),
            "output": [],
            "status": "completed",  # may be overwritten by finish_reason below
        }

        msg_item_id = generate_message_id(response_id_stem)

        for choice in ir_response.get("choices", []):
            message = choice.get("message")
            if not message:
                continue

            content_parts = message.get("content", [])
            text_parts: list[dict[str, Any]] = []
            reasoning_index = 0

            for part in content_parts:
                if is_text_part(part):
                    text_parts.append(
                        {
                            "type": "output_text",
                            "text": part["text"],
                            "annotations": [],
                            "logprobs": [],
                        }
                    )
                elif is_tool_call_part(part):
                    provider_response["output"].append(
                        self.tool_ops.ir_tool_call_to_p(part)
                    )
                elif is_reasoning_part(part):
                    provider_response["output"].append(
                        self.content_ops.ir_reasoning_to_p(
                            part,
                            output_item=True,
                            response_id=ir_response.get("id", ""),
                            reasoning_index=reasoning_index,
                        )
                    )
                    reasoning_index += 1
                elif is_refusal_part(part):
                    text_parts.append(self.content_ops.ir_refusal_to_p(part))

            msg_insert_pos = len(provider_response["output"])
            if text_parts:
                msg_item: dict[str, Any] = {
                    "id": msg_item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": text_parts,
                }
                # Restore phase from IR message provider_metadata
                phase = message.get("provider_metadata", {}).get("responses_phase")
                if phase:
                    msg_item["phase"] = phase
                provider_response["output"].insert(msg_insert_pos, msg_item)

            # Set finish reason
            finish_reason = choice.get("finish_reason", {}).get("reason", "stop")
            status = RESPONSES_REASON_TO_STATUS.get(finish_reason, "completed")
            provider_response["status"] = status
            incomplete_reason = RESPONSES_REASON_TO_INCOMPLETE_REASON.get(finish_reason)
            if incomplete_reason:
                provider_response["incomplete_details"] = {"reason": incomplete_reason}

        # Usage
        ir_usage = ir_response.get("usage")
        if ir_usage:
            provider_response["usage"] = self._build_ir_usage_to_p(ir_usage)

        if "service_tier" in ir_response:
            provider_response["service_tier"] = ir_response["service_tier"]

        # Set completed_at for completed responses
        if provider_response.get("status") == "completed":
            provider_response["completed_at"] = self._completion_timestamp()
        else:
            provider_response["completed_at"] = None

        return provider_response

    # ------------------------------------------------------------------
    # Cross-provider consistency helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _dedup_ir_tool_names(ir_tools: list[Any]) -> list[Any]:
        """Rename namespaced IR tools whose names collide.

        Tools originating from a namespace container may share names with
        top-level tools or with tools from a different namespace.  This
        method qualifies colliding names as ``{namespace}_{name}`` while
        leaving non-namespaced (top-level) tools unchanged.

        Returns a new list; renamed entries are shallow-copied to avoid
        mutating cached references.
        """
        name_indices: dict[str, list[int]] = defaultdict(list)
        for i, tool in enumerate(ir_tools):
            name_indices[tool["name"]].append(i)

        result = list(ir_tools)
        used_names: set[str] = {t["name"] for t in ir_tools}

        for name, indices in name_indices.items():
            if len(indices) <= 1:
                continue
            for i in indices:
                ns = (ir_tools[i].get("metadata") or {}).get("namespace")
                if not ns:
                    continue
                qualified = _qualify_tool_name(ns, name, used_names)
                if qualified is None:
                    continue
                copy = dict(ir_tools[i])
                copy["name"] = qualified
                copy["metadata"] = dict(copy.get("metadata", {}))
                result[i] = copy
                used_names.add(qualified)

        return result

    @staticmethod
    def _build_p_usage_to_ir(p_usage: dict[str, Any]) -> UsageInfo:
        """Build IR usage dict from Responses API usage."""
        usage_info: dict[str, Any] = {
            "prompt_tokens": p_usage.get("input_tokens") or 0,
            "completion_tokens": p_usage.get("output_tokens") or 0,
            "total_tokens": p_usage.get("total_tokens") or 0,
        }
        p_input_details = p_usage.get("input_tokens_details")
        if p_input_details:
            if "cached_tokens" in p_input_details:
                usage_info["cache_read_tokens"] = p_input_details["cached_tokens"]
        p_output_details = p_usage.get("output_tokens_details")
        if p_output_details:
            if "reasoning_tokens" in p_output_details:
                usage_info["reasoning_tokens"] = p_output_details["reasoning_tokens"]
        return cast(UsageInfo, usage_info)

    @staticmethod
    def _build_ir_usage_to_p(ir_usage: Mapping[str, Any]) -> dict[str, Any]:
        """Build Responses API usage dict from IR usage."""
        return {
            "input_tokens": ir_usage.get("prompt_tokens") or 0,
            "output_tokens": ir_usage.get("completion_tokens") or 0,
            "total_tokens": ir_usage.get("total_tokens") or 0,
            "input_tokens_details": {
                "cached_tokens": ir_usage.get("cache_read_tokens", 0),
            },
            "output_tokens_details": {
                "reasoning_tokens": ir_usage.get("reasoning_tokens", 0),
            },
        }

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
            result["tool_choice"] = self.tool_ops.ir_tool_choice_to_p(tool_choice)
        tool_config = ir_request.get("tool_config")
        if tool_config:
            tc_fields = self.tool_ops.ir_tool_config_to_p(tool_config)
            result.update(tc_fields)

    def _convert_p_tool_config_to_ir(
        self,
        provider_request: dict[str, Any],
        ir_request: dict[str, Any],
    ) -> None:
        """Extract tool_choice and tool_config from provider request into IR."""
        tool_choice = provider_request.get("tool_choice")
        if tool_choice is not None:
            ir_request["tool_choice"] = self.tool_ops.p_tool_choice_to_ir(tool_choice)
        tool_config_fields: dict[str, Any] = {}
        if "parallel_tool_calls" in provider_request:
            tool_config_fields["parallel_tool_calls"] = provider_request[
                "parallel_tool_calls"
            ]
        if "max_tool_calls" in provider_request:
            tool_config_fields["max_tool_calls"] = provider_request["max_tool_calls"]
        if tool_config_fields:
            ir_request["tool_config"] = self.tool_ops.p_tool_config_to_ir(
                tool_config_fields
            )

    def _convert_p_cache_to_ir(
        self,
        provider_request: dict[str, Any],
        ir_request: dict[str, Any],
    ) -> None:
        """Extract cache config from provider request into IR."""
        cache_fields: dict[str, Any] = {}
        if "prompt_cache_key" in provider_request:
            cache_fields["prompt_cache_key"] = provider_request["prompt_cache_key"]
        if "prompt_cache_retention" in provider_request:
            cache_fields["prompt_cache_retention"] = provider_request[
                "prompt_cache_retention"
            ]
        if cache_fields:
            ir_request["cache"] = self.config_ops.p_cache_config_to_ir(cache_fields)

    def _capture_preserve_metadata(
        self,
        provider_response: dict[str, Any],
        ir_response: dict[str, Any],
        ctx: ConversionContext,
    ) -> None:
        """Capture extra fields from provider response for lossless round-trip."""
        output_items = provider_response.get("output", [])
        extras = {
            k: v
            for k, v in provider_response.items()
            if k in RESPONSES_PRESERVE_FIELDS and v is not None
        }
        if extras:
            ctx.store_response_extras(extras)

        items_meta = [
            _capture_item_metadata(item)
            for item in output_items
            if OpenAIResponsesMessageOps.is_portable_response_output_item(item)
        ]
        if items_meta:
            ctx.store_output_items_meta(items_meta)

    def _apply_preserve_metadata(
        self,
        provider_response: dict[str, Any],
        ir_response: dict[str, Any],
        ctx: ConversionContext,
    ) -> None:
        """Re-inject captured metadata fields in *preserve* mode."""
        echo = ctx.get_echo_fields()
        core_keys = {
            "id",
            "object",
            "created_at",
            "model",
            "output",
            "status",
            "usage",
        }
        # Apply required defaults first, then override with actual echo
        for k, v in RESPONSES_REQUIRED_DEFAULTS.items():
            if k not in core_keys and k not in provider_response:
                provider_response[k] = v
        for k, v in echo.items():
            if k not in core_keys:
                provider_response[k] = v

        # Ensure echoed tools have required 'strict' field
        for tool in provider_response.get("tools", []):
            if isinstance(tool, dict) and tool.get("type") == "function":
                tool.setdefault("strict", None)

        # Restore per-output-item metadata
        items_meta = ctx.get_output_items_meta()
        output = provider_response.get("output", [])
        for i, meta in enumerate(items_meta):
            if i >= len(output):
                break
            item = output[i]
            if "id" in meta:
                item["id"] = meta["id"]
            if "status" in meta:
                item["status"] = meta["status"]
            # Restore per-content-part metadata
            content_meta = meta.get("content_meta", [])
            content = item.get("content", [])
            for j, pm in enumerate(content_meta):
                if j >= len(content):
                    break
                if "annotations" in pm:
                    content[j]["annotations"] = pm["annotations"]
                if "logprobs" in pm:
                    content[j]["logprobs"] = pm["logprobs"]

    # ==================== Stream Support ====================

    def stream_response_from_provider(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None = None,
    ) -> list[IRStreamEvent]:
        """Convert an OpenAI Responses SSE event to IR stream events.

        OpenAI Responses API uses fine-grained SSE events with a ``type`` field
        (e.g. ``response.output_text.delta``) instead of the ``choices[].delta``
        structure used by Chat Completions.

        A single event typically produces zero or one IR events, but
        ``response.completed`` may produce both a ``FinishEvent`` and a
        ``UsageEvent``.

        When a ``context`` is provided, lifecycle events (``StreamStartEvent``,
        ``ContentBlockStartEvent``, ``ContentBlockEndEvent``,
        ``StreamEndEvent``) are emitted and cross-event state is tracked.
        Without a context the behaviour is identical to the previous
        implementation (backward compatible).

        Args:
            chunk: OpenAI Responses SSE event dict (or SDK object).
            context: Optional stream context for stateful conversions.

        Returns:
            List of IR stream events extracted from the event.
        """
        chunk = self._normalize(chunk)
        events: list[IRStreamEvent] = []
        event_type = chunk.get("type", "")

        handler_name = self._P_TO_IR_DISPATCH.get(event_type)
        if handler_name is not None:
            getattr(self, handler_name)(chunk, context, events)

        # All other event types (response.in_progress, response.queued,
        # response.output_text.done, etc.) are dropped — response.in_progress
        # is redundant with response.created and is re-synthesised by
        # _handle_ir_stream_start_to_p on the to_provider side.

        return events

    # --- from_provider handlers ---

    def _handle_p_response_created_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        if context is not None:
            response = chunk.get("response", {})
            response_id = self.strip_response_id_prefix(
                response.get("id", ""),
                self._get_response_id_prefix(context),
            )
            model = response.get("model", "")
            created = int(response.get("created_at", 0))
            context.response_id = response_id
            context.model = model
            context.created = created
            context.mark_started()
            start_event: StreamStartEvent = {
                "type": "stream_start",
                "response_id": response_id,
                "model": model,
            }
            if created:
                start_event["created"] = created
            events.append(start_event)

            # Preserve mode: capture echo fields from the initial response
            if context.metadata_mode == "preserve" and isinstance(response, dict):
                extras = {
                    k: v for k, v in response.items() if k in RESPONSES_PRESERVE_FIELDS
                }
                if extras:
                    context.store_response_extras(extras)

    def _handle_p_output_text_delta_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        events.append(
            TextDeltaEvent(
                type="text_delta",
                text=chunk.get("delta", ""),
            )
        )

    def _handle_p_reasoning_delta_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        event = ReasoningDeltaEvent(
            type="reasoning_delta",
            reasoning=chunk.get("delta", ""),
        )
        if (
            isinstance(context, OpenAIResponsesStreamContext)
            and context._reasoning_item_id
        ):
            event["provider_metadata"] = {
                "responses_reasoning_id": context._reasoning_item_id,
            }
        events.append(event)

    def _handle_p_refusal_delta_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        events.append(
            RefusalDeltaEvent(
                type="refusal_delta",
                refusal=chunk.get("delta", ""),
            )
        )

    def _handle_p_output_item_added_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        item = chunk.get("item", {})
        if isinstance(item, dict):
            item_type = item.get("type", "")

            if item_type in ("function_call", "custom_tool_call"):
                call_id = item.get("call_id", "")
                item_id = item.get("id", "")
                tool_type = "custom" if item_type == "custom_tool_call" else "function"

                # Register tool call in context
                if context is not None:
                    context.register_tool_call(call_id, item.get("name", ""), tool_type)
                    context.register_tool_call_item(call_id, item_id)

                start_event_tc = ToolCallStartEvent(
                    type="tool_call_start",
                    tool_call_id=call_id,
                    tool_name=item.get("name", ""),
                    tool_type=tool_type,
                )
                if item_id and item_id != call_id:
                    start_event_tc["provider_metadata"] = {
                        "responses_item_id": item_id,
                    }
                output_index = chunk.get("output_index")
                if output_index is not None:
                    start_event_tc["tool_call_index"] = output_index
                events.append(start_event_tc)

            elif item_type == "reasoning" and isinstance(
                context, OpenAIResponsesStreamContext
            ):
                # Capture early so deltas can carry the source ID;
                # done event re-confirms authoritatively.
                item_id = item.get("id", "")
                if item_id:
                    context._reasoning_item_id = item_id

            elif item_type == "message":
                # Message-level output item — no IR event needed.
                # The actual content block is signaled by the subsequent
                # ``response.content_part.added`` event which produces its
                # own ``ContentBlockStartEvent``.  Emitting a
                # ``ContentBlockStartEvent`` here would cause duplicate
                # ``response.content_part.added`` events on round-trip.
                #
                # Capture phase for streaming round-trip.
                phase = item.get("phase")
                if phase and context is not None:
                    context.metadata["_responses_phase"] = phase

            elif item_type in ("tool_search_call", "tool_search_output"):
                events.append(
                    ProviderPassthroughEvent(
                        type="provider_passthrough",
                        provider=self._CONVERTER_TAG,
                        payload=dict(chunk),
                    )
                )

    def _handle_p_content_part_added_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        if context is not None:
            part = chunk.get("part", {})
            part_type = part.get("type", "") if isinstance(part, dict) else ""
            block_type = "text"
            if part_type == "output_text":
                block_type = "text"
            elif part_type == "summary_text":
                block_type = "thinking"
            elif part_type == "refusal":
                block_type = "refusal"
            block_index = context.next_block_index()
            events.append(
                ContentBlockStartEvent(
                    type="content_block_start",
                    block_index=block_index,
                    block_type=block_type,
                )
            )

    def _handle_p_content_part_done_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        if context is not None:
            events.append(
                ContentBlockEndEvent(
                    type="content_block_end",
                    block_index=context.current_block_index,
                )
            )

    def _handle_p_output_item_done_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        # For function_call items, store the completed item in context
        # so it can be included in the response.completed output array.
        # Message-level done is a no-op (content_part.done handles it).
        item = chunk.get("item", {})
        if not isinstance(item, dict):
            return
        item_type = item.get("type", "")
        if item_type == "function_call":
            if context is not None:
                call_id = item.get("call_id", "")
                if call_id:
                    context.set_tool_call_args(call_id, item.get("arguments", ""))
        elif item_type == "custom_tool_call":
            if context is not None:
                call_id = item.get("call_id", "")
                if call_id:
                    context.set_tool_call_args(call_id, item.get("input", ""))
        elif item_type == "reasoning" and isinstance(
            context, OpenAIResponsesStreamContext
        ):
            item_id = item.get("id", "")
            if item_id:
                context._reasoning_item_id = item_id
            encrypted_content = item.get("encrypted_content")
            if encrypted_content:
                context._reasoning_encrypted_content = str(encrypted_content)
                # Zero-text delta carries encrypted_content through IR;
                # suppressed on the outbound side when reasoning is empty.
                event = ReasoningDeltaEvent(
                    type="reasoning_delta",
                    reasoning="",
                    encrypted_content=str(encrypted_content),
                )
                if item_id:
                    event["provider_metadata"] = {
                        "responses_reasoning_id": item_id,
                    }
                events.append(event)
        elif item_type in ("tool_search_call", "tool_search_output"):
            events.append(
                ProviderPassthroughEvent(
                    type="provider_passthrough",
                    provider=self._CONVERTER_TAG,
                    payload=dict(chunk),
                )
            )

    def _handle_p_function_call_args_delta_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        delta_text = chunk.get("delta", "")
        call_id = resolve_call_id(chunk, context)
        delta_event = ToolCallDeltaEvent(
            type="tool_call_delta",
            tool_call_id=call_id,
            arguments_delta=delta_text,
        )
        output_index = chunk.get("output_index")
        if output_index is not None:
            delta_event["tool_call_index"] = output_index

        # Accumulate arguments in context
        if context is not None and call_id:
            context.append_tool_call_args(call_id, delta_text)

        events.append(delta_event)

    def _handle_p_function_call_args_done_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        call_id = resolve_call_id(chunk, context)
        arguments = chunk.get("arguments", "")
        # Store final arguments in context
        if context is not None and call_id:
            context.set_tool_call_args(call_id, arguments)

    def _handle_p_custom_tool_call_input_delta_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Handle custom_tool_call_input.delta — same as function_call args delta."""
        delta_text = chunk.get("delta", "")
        call_id = resolve_call_id(chunk, context)
        delta_event = ToolCallDeltaEvent(
            type="tool_call_delta",
            tool_call_id=call_id,
            arguments_delta=delta_text,
        )
        output_index = chunk.get("output_index")
        if output_index is not None:
            delta_event["tool_call_index"] = output_index

        # Accumulate input in context
        if context is not None and call_id:
            context.append_tool_call_args(call_id, delta_text)

        events.append(delta_event)

    def _handle_p_custom_tool_call_input_done_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Handle custom_tool_call_input.done — store final input text."""
        call_id = resolve_call_id(chunk, context)
        input_text = chunk.get("input", "")
        if context is not None and call_id:
            context.set_tool_call_args(call_id, input_text)

    def _handle_p_response_completed_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        response = chunk.get("response", chunk)

        # Determine finish reason from status
        status = response.get("status", "completed")
        if status == "incomplete":
            incomplete_details = response.get("incomplete_details", {})
            inc_reason = (
                incomplete_details.get("reason", "")
                if isinstance(incomplete_details, dict)
                else ""
            )
            reason = RESPONSES_INCOMPLETE_REASON_TO_IR.get(inc_reason, "stop")
        else:
            reason = RESPONSES_STATUS_TO_REASON.get(status, "stop")

        # Check if any output item is a function_call to set tool_calls reason
        output_items = response.get("output", [])
        if isinstance(output_items, list):
            for item in output_items:
                if isinstance(item, dict) and item.get("type") == "function_call":
                    reason = "tool_calls"
                    break

        # Emit UsageEvent before FinishEvent so that downstream
        # converters can store usage in context.pending_usage before
        # FinishEvent builds the terminal response.completed event.
        usage = response.get("usage")
        if isinstance(usage, dict):
            events.append(
                UsageEvent(
                    type="usage",
                    usage=self._build_p_usage_to_ir(usage),
                )
            )

        events.append(
            FinishEvent(
                type="finish",
                finish_reason={"reason": reason},  # ty: ignore[invalid-argument-type]
            )
        )

        # Emit StreamEndEvent after other events
        if context is not None:
            context.mark_ended()
            events.append(StreamEndEvent(type="stream_end"))

    def _handle_p_response_failed_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        events.append(
            FinishEvent(
                type="finish",
                finish_reason={"reason": "error"},
            )
        )

        # Emit StreamEndEvent after FinishEvent
        if context is not None:
            context.mark_ended()
            events.append(StreamEndEvent(type="stream_end"))

    _P_TO_IR_DISPATCH: dict[str, str] = {
        ResponsesEventType.RESPONSE_CREATED: "_handle_p_response_created_to_ir",
        ResponsesEventType.OUTPUT_TEXT_DELTA: "_handle_p_output_text_delta_to_ir",
        ResponsesEventType.REASONING_SUMMARY_TEXT_DELTA: "_handle_p_reasoning_delta_to_ir",
        ResponsesEventType.OUTPUT_ITEM_ADDED: "_handle_p_output_item_added_to_ir",
        ResponsesEventType.CONTENT_PART_ADDED: "_handle_p_content_part_added_to_ir",
        ResponsesEventType.CONTENT_PART_DONE: "_handle_p_content_part_done_to_ir",
        ResponsesEventType.OUTPUT_ITEM_DONE: "_handle_p_output_item_done_to_ir",
        ResponsesEventType.FUNCTION_CALL_ARGS_DELTA: "_handle_p_function_call_args_delta_to_ir",
        ResponsesEventType.FUNCTION_CALL_ARGS_DONE: "_handle_p_function_call_args_done_to_ir",
        ResponsesEventType.CUSTOM_TOOL_CALL_INPUT_DELTA: "_handle_p_custom_tool_call_input_delta_to_ir",
        ResponsesEventType.CUSTOM_TOOL_CALL_INPUT_DONE: "_handle_p_custom_tool_call_input_done_to_ir",
        ResponsesEventType.RESPONSE_COMPLETED: "_handle_p_response_completed_to_ir",
        ResponsesEventType.RESPONSE_FAILED: "_handle_p_response_failed_to_ir",
        ResponsesEventType.REFUSAL_DELTA: "_handle_p_refusal_delta_to_ir",
    }

    _IR_TO_P_DISPATCH: dict[str, str] = {
        **BaseConverter._IR_TO_P_DISPATCH,
        "refusal_delta": "_handle_ir_refusal_delta_to_p",
    }

    def stream_response_to_provider(
        self,
        event: IRStreamEvent,
        context: StreamContext | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert IR stream event with automatic context upgrade.

        If a base ``StreamContext`` is passed, it is automatically upgraded
        to ``OpenAIResponsesStreamContext`` (preserving existing state) so
        that callers do not need to know about the provider-specific subclass.
        """
        # Auto-upgrade base StreamContext to the provider-specific subclass.
        # Cache the upgraded instance in metadata so state persists across calls.
        if context is not None and not isinstance(
            context, OpenAIResponsesStreamContext
        ):
            cached = context.metadata.get("_responses_stream_ctx")
            if cached is None:
                cached = OpenAIResponsesStreamContext.from_base(context)
                context.metadata["_responses_stream_ctx"] = cached
            context = cached

        return super().stream_response_to_provider(event, context)

    def _post_process_ir_to_p(
        self,
        result: dict[str, Any] | list[dict[str, Any]],
        event: IRStreamEvent,
        context: StreamContext | None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Inject sequence_number into emitted Responses events."""
        if isinstance(context, OpenAIResponsesStreamContext):
            if isinstance(result, list):
                for r in result:
                    if isinstance(r, dict) and "type" in r:
                        context._sequence_number += 1
                        r["sequence_number"] = context._sequence_number
            elif isinstance(result, dict) and "type" in result:
                context._sequence_number += 1
                result["sequence_number"] = context._sequence_number
        return result

    # --- to_provider handlers ---

    def _handle_ir_stream_start_to_p(
        self,
        event: StreamStartEvent,
        context: StreamContext | None,
    ) -> list[dict[str, Any]]:
        response_id_stem = event["response_id"]

        # Store metadata in context if provided (stem, without prefix)
        if context is not None:
            context.response_id = response_id_stem
            context.model = event["model"]
            context.created = event.get("created", 0)
            context.mark_started()

        prefixed_id = self.add_response_id_prefix(
            response_id_stem, self._get_response_id_prefix(context)
        )
        response: dict[str, Any] = {
            "id": prefixed_id,
            "object": "response",
            "model": event["model"],
            "status": "in_progress",
            "output": [],
        }
        created = event.get("created", 0)
        response["created_at"] = created or int(time.time())

        # Preserve mode: include echo fields in response.created
        if context is not None and context.metadata_mode == "preserve":
            echo = context.get_echo_fields()
            core_keys = {
                "id",
                "object",
                "created_at",
                "model",
                "output",
                "status",
                "usage",
            }
            for k, v in RESPONSES_REQUIRED_DEFAULTS.items():
                if k not in core_keys and k not in response:
                    response[k] = v
            for k, v in echo.items():
                if k not in core_keys:
                    response[k] = v
            # response.created must include usage: null (not yet available)
            response.setdefault("usage", None)

        # Both events share the same response snapshot — safe because
        # nothing between here and json.dumps serialisation mutates it.
        return [
            {
                "type": ResponsesEventType.RESPONSE_CREATED,
                "response": response,
            },
            {
                "type": ResponsesEventType.RESPONSE_IN_PROGRESS,
                "response": response,
            },
        ]

    def _handle_ir_stream_end_to_p(
        self,
        event: StreamEndEvent,
        context: StreamContext | None,
    ) -> dict[str, Any]:
        if context is not None:
            context.mark_ended()
            # Emit the deferred response.completed if FinishEvent
            # stored one.  This ensures UsageEvents that arrive
            # between FinishEvent and StreamEndEvent (e.g. OpenAI
            # Chat sends usage in a separate chunk after
            # finish_reason) are merged into the response.
            if context.pending_response is not None:
                resp = context.pending_response
                context.pending_response = None
                # Merge any usage that arrived after FinishEvent
                if context.pending_usage is not None and "usage" not in resp:
                    resp["usage"] = self._build_finish_usage(context.pending_usage)
                return {
                    "type": ResponsesEventType.RESPONSE_COMPLETED,
                    "response": resp,
                }
        return {}

    def _handle_ir_content_block_start_to_p(
        self,
        event: ContentBlockStartEvent,
        context: OpenAIResponsesStreamContext | None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        block_type = event["block_type"]
        if context is not None:
            context.current_block_type = block_type
        if block_type in ("text", "refusal"):
            content_part_type = "refusal" if block_type == "refusal" else "output_text"
            if context is not None and not context.output_item_emitted:
                return build_message_preamble_events(context, content_part_type)
            from .utils import _build_content_part

            item_id = context.item_id if context is not None else ""
            msg_idx = context._message_output_index if context is not None else 0
            return {
                "type": ResponsesEventType.CONTENT_PART_ADDED,
                "item_id": item_id,
                "output_index": msg_idx,
                "content_index": 0,
                "part": _build_content_part(content_part_type),
            }
        # Other block types are no-ops for now
        return {}

    def _handle_ir_content_block_end_to_p(
        self,
        event: ContentBlockEndEvent,
        context: OpenAIResponsesStreamContext | None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        if context is not None:
            context.content_part_done_emitted = True
            item_id = context.item_id
            msg_idx = context._message_output_index
            is_refusal = context.current_block_type == "refusal"
            if is_refusal:
                accumulated = context.accumulated_refusal
                return [
                    {
                        "type": ResponsesEventType.REFUSAL_DONE,
                        "item_id": item_id,
                        "output_index": msg_idx,
                        "content_index": 0,
                        "refusal": accumulated,
                    },
                    {
                        "type": ResponsesEventType.CONTENT_PART_DONE,
                        "item_id": item_id,
                        "output_index": msg_idx,
                        "content_index": 0,
                        "part": {
                            "type": "refusal",
                            "refusal": accumulated,
                        },
                    },
                ]
            accumulated = context.accumulated_text
            return [
                {
                    "type": ResponsesEventType.OUTPUT_TEXT_DONE,
                    "item_id": item_id,
                    "output_index": msg_idx,
                    "content_index": 0,
                    "text": accumulated,
                    "logprobs": [],
                },
                {
                    "type": ResponsesEventType.CONTENT_PART_DONE,
                    "item_id": item_id,
                    "output_index": msg_idx,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": accumulated,
                        "annotations": [],
                        "logprobs": [],
                    },
                },
            ]
        return {
            "type": ResponsesEventType.CONTENT_PART_DONE,
            "part": {
                "type": "output_text",
            },
        }

    def _handle_ir_text_delta_to_p(
        self,
        event: TextDeltaEvent,
        context: OpenAIResponsesStreamContext | None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        text = event["text"]

        # Emit output_item.added + content_part.added before the first
        # text delta so clients (e.g. Codex CLI) can register the item.
        preamble: list[dict[str, Any]] = []
        if context is not None and not context.output_item_emitted:
            preamble = build_message_preamble_events(context)

        # Accumulate text in context for response.completed output
        if context is not None:
            context.accumulated_text += text

        msg_idx = context._message_output_index if context is not None else 0
        item_id = context.item_id if context is not None else ""
        delta_event: dict[str, Any] = {
            "type": ResponsesEventType.OUTPUT_TEXT_DELTA,
            "item_id": item_id,
            "output_index": msg_idx,
            "content_index": 0,
            "delta": text,
            "logprobs": [],
        }

        if preamble:
            return preamble + [delta_event]
        return delta_event

    def _handle_ir_refusal_delta_to_p(
        self,
        event: RefusalDeltaEvent,
        context: OpenAIResponsesStreamContext | None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        refusal = event["refusal"]

        preamble: list[dict[str, Any]] = []
        if context is not None and not context.output_item_emitted:
            preamble = build_message_preamble_events(context, "refusal")

        if context is not None:
            context.accumulated_refusal += refusal

        msg_idx = context._message_output_index if context is not None else 0
        item_id = context.item_id if context is not None else ""
        delta_event: dict[str, Any] = {
            "type": ResponsesEventType.REFUSAL_DELTA,
            "item_id": item_id,
            "output_index": msg_idx,
            "content_index": 0,
            "delta": refusal,
        }

        if preamble:
            return preamble + [delta_event]
        return delta_event

    def _handle_ir_reasoning_delta_to_p(
        self,
        event: ReasoningDeltaEvent,
        context: StreamContext | None,
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        ctx = context if isinstance(context, OpenAIResponsesStreamContext) else None
        provider_metadata = event.get("provider_metadata") or {}
        source_item_id = provider_metadata.get("responses_reasoning_id", "")
        if not isinstance(source_item_id, str):
            source_item_id = ""

        if ctx is not None and ctx._reasoning_item_id == "":
            output_index = ctx.next_output_index()
            ctx._reasoning_output_index = output_index
            item_id = source_item_id or generate_reasoning_id(
                ctx.response_id, output_index
            )
            ctx._reasoning_item_id = item_id

            results.append(
                {
                    "type": ResponsesEventType.OUTPUT_ITEM_ADDED,
                    "output_index": output_index,
                    "item": {
                        "id": item_id,
                        "type": "reasoning",
                        "summary": [],
                        "status": "in_progress",
                    },
                }
            )
            results.append(
                {
                    "type": ResponsesEventType.REASONING_SUMMARY_PART_ADDED,
                    "item_id": item_id,
                    "output_index": output_index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": ""},
                }
            )

        enc = event.get("encrypted_content", "")
        if ctx is not None and source_item_id and enc:
            ctx._reasoning_encrypted_content = enc
            if not event["reasoning"]:
                return results

        reasoning_text = event["reasoning"]
        if ctx is not None:
            ctx._reasoning_accumulated_text += reasoning_text

        item_id = ctx._reasoning_item_id if ctx is not None else ""
        r_idx = ctx._reasoning_output_index if ctx is not None else 0
        results.append(
            {
                "type": ResponsesEventType.REASONING_SUMMARY_TEXT_DELTA,
                "item_id": item_id,
                "output_index": r_idx,
                "summary_index": 0,
                "delta": reasoning_text,
            }
        )
        return results

    def _handle_ir_tool_call_start_to_p(
        self,
        event: ToolCallStartEvent,
        context: StreamContext | None,
    ) -> dict[str, Any]:
        call_id = sanitize_tool_call_id(event["tool_call_id"])
        tool_name = event["tool_name"]
        tool_type = event.get("tool_type", "function")
        pm = event.get("provider_metadata") or {}
        item_id = pm.get("responses_item_id") or call_id

        # Register in context for later done events
        if context is not None and call_id:
            context.register_tool_call(call_id, tool_name, tool_type)
            context.register_tool_call_item(call_id, item_id)

        if isinstance(context, OpenAIResponsesStreamContext):
            output_index = context.next_output_index()
            context._tool_call_output_indices[call_id] = output_index
        else:
            tc_index = event.get("tool_call_index")
            output_index = tc_index if tc_index is not None else 0

        if tool_type == "custom":
            item: dict[str, Any] = {
                "id": item_id,
                "type": "custom_tool_call",
                "call_id": call_id,
                "name": tool_name,
                "input": "",
                "status": "in_progress",
            }
        else:
            item = {
                "id": item_id,
                "type": "function_call",
                "call_id": call_id,
                "name": tool_name,
                "arguments": "",
                "status": "in_progress",
            }

        result: dict[str, Any] = {
            "type": ResponsesEventType.OUTPUT_ITEM_ADDED,
            "output_index": output_index,
            "item": item,
        }
        return result

    def _handle_ir_tool_call_delta_to_p(
        self,
        event: ToolCallDeltaEvent,
        context: StreamContext | None,
    ) -> dict[str, Any]:
        call_id = sanitize_tool_call_id(event["tool_call_id"])
        delta = event["arguments_delta"]
        tc_index = event.get("tool_call_index")

        # Defense-in-depth: resolve empty tool_call_id by index.
        # Some upstream providers (e.g. certain Chat Completions
        # implementations) only send tool_call_id on the first chunk.
        if not call_id and context is not None and tc_index is not None:
            if tc_index < context.tool_call_count:
                call_id = context.resolve_tool_call_id_by_index(tc_index)

        # Accumulate arguments in context for done events
        if context is not None and call_id:
            context.append_tool_call_args(call_id, delta)

        # Use item_id per Responses API spec
        item_id = ""
        if context is not None and call_id:
            item_id = context.get_tool_call_item_id(call_id)
        if not item_id and call_id:
            item_id = call_id

        if isinstance(context, OpenAIResponsesStreamContext) and call_id:
            output_index = context._tool_call_output_indices.get(call_id, 0)
        else:
            output_index = tc_index if tc_index is not None else 0

        # Determine event type based on tool type (custom vs function)
        tool_type = (
            context.get_tool_type(call_id)
            if context is not None and call_id
            else "function"
        )
        event_type = (
            ResponsesEventType.CUSTOM_TOOL_CALL_INPUT_DELTA
            if tool_type == "custom"
            else ResponsesEventType.FUNCTION_CALL_ARGS_DELTA
        )

        result: dict[str, Any] = {
            "type": event_type,
            "item_id": item_id,
            "output_index": output_index,
            "delta": delta,
        }
        return result

    def _handle_ir_finish_to_p(
        self,
        event: FinishEvent,
        context: OpenAIResponsesStreamContext | None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        reason = event["finish_reason"]["reason"]
        status = RESPONSES_REASON_TO_STATUS.get(reason, "completed")

        response = self._build_finish_response(status, context, reason)

        # Emit done events before response.completed
        results: list[dict[str, Any]] = []

        if context is not None:
            self._emit_reasoning_done_events(context, results)
            self._emit_text_done_events(context, results)
            self._emit_tool_call_done_events(context, results)

        # With context: defer response.completed to StreamEndEvent
        # so that any UsageEvent arriving after FinishEvent (e.g.
        # OpenAI Chat sends usage in a separate chunk) can still be
        # merged into the response.
        if context is not None:
            context.pending_response = response
            return results

        # Without context: emit immediately (backward compatible)
        results.append(
            {
                "type": ResponsesEventType.RESPONSE_COMPLETED,
                "response": response,
            }
        )
        return results

    def _build_finish_response(
        self,
        status: str,
        context: OpenAIResponsesStreamContext | None,
        finish_reason: str = "length",
    ) -> dict[str, Any]:
        """Build the response dict for a FinishEvent."""
        output = self._collect_finish_output(context)
        response: dict[str, Any] = {"status": status, "output": output}

        # Populate id, object, model, created_at from context so clients can
        # parse the completed response envelope.
        if context is not None:
            response["id"] = self.add_response_id_prefix(
                context.response_id, self._get_response_id_prefix(context)
            )
            response["object"] = "response"
            response["model"] = context.model
            response["created_at"] = context.created or int(time.time())

        # Set completed_at timestamp for completed responses.
        if status == "completed":
            response["completed_at"] = self._completion_timestamp()
        else:
            response["completed_at"] = None

        if status == "incomplete":
            incomplete_reason = RESPONSES_REASON_TO_INCOMPLETE_REASON.get(
                finish_reason, "max_output_tokens"
            )
            response["incomplete_details"] = {"reason": incomplete_reason}

        # Merge pending usage from context if available
        if context is not None and context.pending_usage is not None:
            response["usage"] = self._build_finish_usage(context.pending_usage)

        # Preserve mode: inject echo fields into the response.completed payload
        if context is not None and context.metadata_mode == "preserve":
            self._apply_finish_echo(response, context)

        return response

    def _collect_finish_output(
        self, context: OpenAIResponsesStreamContext | None
    ) -> list[dict[str, Any]]:
        """Collect text and tool call output items from stream context."""
        output: list[dict[str, Any]] = []
        if context is None:
            return output

        # Reasoning item comes first in output array
        if context._reasoning_item_id:
            reasoning_item: dict[str, Any] = {
                "id": context._reasoning_item_id,
                "type": "reasoning",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": context._reasoning_accumulated_text,
                    }
                ],
                "status": "completed",
            }
            if context._reasoning_encrypted_content:
                reasoning_item["encrypted_content"] = (
                    context._reasoning_encrypted_content
                )
            output.append(reasoning_item)

        accumulated = context.accumulated_text
        if accumulated:
            msg_item: dict[str, Any] = {
                "id": context.item_id,
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": accumulated}],
            }
            # Restore phase from context if available
            phase = context.metadata.get("_responses_phase")
            if phase:
                msg_item["phase"] = phase
            if context.metadata_mode == "preserve":
                msg_item["status"] = "completed"
                for part in msg_item.get("content", []):
                    part.setdefault("annotations", [])
                    part.setdefault("logprobs", [])
            output.append(msg_item)

        for call_id in context.tool_call_ids:
            tool_name = context.get_tool_name(call_id)
            arguments = context._tool_call_args.get(call_id, "")
            tc_item_id = context.get_tool_call_item_id(call_id) or call_id
            tool_type = context.get_tool_type(call_id)

            if tool_type == "custom":
                output.append(
                    {
                        "id": tc_item_id,
                        "type": "custom_tool_call",
                        "call_id": call_id,
                        "name": tool_name,
                        "input": arguments,
                        "status": "completed",
                    }
                )
            else:
                output.append(
                    {
                        "id": tc_item_id,
                        "type": "function_call",
                        "call_id": call_id,
                        "name": tool_name,
                        "arguments": arguments,
                        "status": "completed",
                    }
                )
        return output

    @staticmethod
    def _build_finish_usage(pending_usage: dict[str, Any]) -> dict[str, Any]:
        """Build usage dict for the finish response from pending IR usage."""
        usage: dict[str, Any] = {
            "input_tokens": pending_usage.get("prompt_tokens") or 0,
            "output_tokens": pending_usage.get("completion_tokens") or 0,
            "total_tokens": pending_usage.get("total_tokens") or 0,
        }
        cache_read = pending_usage.get("cache_read_tokens")
        usage["input_tokens_details"] = {
            "cached_tokens": cache_read if cache_read is not None else 0
        }
        reasoning = pending_usage.get("reasoning_tokens")
        usage["output_tokens_details"] = {
            "reasoning_tokens": reasoning if reasoning is not None else 0
        }
        return usage

    @staticmethod
    def _apply_finish_echo(
        response: dict[str, Any], context: OpenAIResponsesStreamContext
    ) -> None:
        """Inject preserve-mode echo fields into the finish response."""
        echo = context.get_echo_fields()
        core_keys = {
            "id",
            "object",
            "created_at",
            "model",
            "output",
            "status",
            "usage",
        }
        for k, v in RESPONSES_REQUIRED_DEFAULTS.items():
            if k not in core_keys and k not in response:
                response[k] = v
        for k, v in echo.items():
            if k not in core_keys:
                response[k] = v

    def _emit_reasoning_done_events(
        self,
        context: OpenAIResponsesStreamContext,
        results: list[dict[str, Any]],
    ) -> None:
        """Emit reasoning lifecycle done events."""
        if not context._reasoning_item_id:
            return
        item_id = context._reasoning_item_id
        output_index = context._reasoning_output_index
        accumulated = context._reasoning_accumulated_text

        results.append(
            {
                "type": ResponsesEventType.REASONING_SUMMARY_TEXT_DONE,
                "item_id": item_id,
                "output_index": output_index,
                "summary_index": 0,
                "text": accumulated,
            }
        )
        results.append(
            {
                "type": ResponsesEventType.REASONING_SUMMARY_PART_DONE,
                "item_id": item_id,
                "output_index": output_index,
                "summary_index": 0,
                "part": {"type": "summary_text", "text": accumulated},
            }
        )
        reasoning_item: dict[str, Any] = {
            "id": item_id,
            "type": "reasoning",
            "summary": [{"type": "summary_text", "text": accumulated}],
            "status": "completed",
        }
        if context._reasoning_encrypted_content:
            reasoning_item["encrypted_content"] = context._reasoning_encrypted_content
        results.append(
            {
                "type": ResponsesEventType.OUTPUT_ITEM_DONE,
                "output_index": output_index,
                "item": reasoning_item,
            }
        )

    def _emit_text_done_events(
        self,
        context: OpenAIResponsesStreamContext,
        results: list[dict[str, Any]],
    ) -> None:
        """Emit text done events if we had text output."""
        if not context.output_item_emitted:
            return

        accumulated = context.accumulated_text
        item_id = context.item_id
        msg_idx = context._message_output_index

        # Only emit output_text.done + content_part.done if not
        # already emitted by a prior ContentBlockEndEvent
        if not context.content_part_done_emitted:
            results.append(
                {
                    "type": ResponsesEventType.OUTPUT_TEXT_DONE,
                    "item_id": item_id,
                    "output_index": msg_idx,
                    "content_index": 0,
                    "text": accumulated,
                    "logprobs": [],
                }
            )
            results.append(
                {
                    "type": ResponsesEventType.CONTENT_PART_DONE,
                    "item_id": item_id,
                    "output_index": msg_idx,
                    "content_index": 0,
                    "part": {
                        "type": "output_text",
                        "text": accumulated,
                        "annotations": [],
                        "logprobs": [],
                    },
                }
            )
        results.append(
            {
                "type": ResponsesEventType.OUTPUT_ITEM_DONE,
                "output_index": msg_idx,
                "item": {
                    "id": item_id,
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {
                            "type": "output_text",
                            "text": accumulated,
                            "annotations": [],
                            "logprobs": [],
                        }
                    ],
                },
            }
        )

    def _emit_tool_call_done_events(
        self,
        context: OpenAIResponsesStreamContext,
        results: list[dict[str, Any]],
    ) -> None:
        """Emit done events for each tool call."""
        for tc_idx, call_id in enumerate(context.tool_call_ids):
            tool_name = context.get_tool_name(call_id)
            arguments = context._tool_call_args.get(call_id, "")
            item_id = context.get_tool_call_item_id(call_id) or call_id
            output_index = context._tool_call_output_indices.get(call_id, tc_idx)
            tool_type = context.get_tool_type(call_id)

            if tool_type == "custom":
                # response.custom_tool_call_input.done
                results.append(
                    {
                        "type": ResponsesEventType.CUSTOM_TOOL_CALL_INPUT_DONE,
                        "item_id": item_id,
                        "output_index": output_index,
                        "input": arguments,
                    }
                )

                # response.output_item.done for the custom_tool_call
                results.append(
                    {
                        "type": ResponsesEventType.OUTPUT_ITEM_DONE,
                        "output_index": output_index,
                        "item": {
                            "id": item_id,
                            "type": "custom_tool_call",
                            "call_id": call_id,
                            "name": tool_name,
                            "input": arguments,
                            "status": "completed",
                        },
                    }
                )
            else:
                # response.function_call_arguments.done
                results.append(
                    {
                        "type": ResponsesEventType.FUNCTION_CALL_ARGS_DONE,
                        "item_id": item_id,
                        "output_index": output_index,
                        "arguments": arguments,
                    }
                )

                # response.output_item.done for the function_call
                results.append(
                    {
                        "type": ResponsesEventType.OUTPUT_ITEM_DONE,
                        "output_index": output_index,
                        "item": {
                            "id": item_id,
                            "type": "function_call",
                            "call_id": call_id,
                            "name": tool_name,
                            "arguments": arguments,
                            "status": "completed",
                        },
                    }
                )

    def _handle_ir_usage_to_p(
        self,
        event: UsageEvent,
        context: StreamContext | None,
    ) -> dict[str, Any]:
        usage = event["usage"]

        # With context: store usage for later merging, avoid duplicate
        # response.completed
        if context is not None:
            context.buffer_usage(usage)
            return {}

        # Without context: preserve backward-compatible behavior
        resp: dict[str, Any] = {
            "status": "completed",
            "output": [],
            "usage": {
                "input_tokens": usage.get("prompt_tokens") or 0,
                "output_tokens": usage.get("completion_tokens") or 0,
                "total_tokens": usage.get("total_tokens") or 0,
            },
        }
        return {
            "type": ResponsesEventType.RESPONSE_COMPLETED,
            "response": resp,
        }
