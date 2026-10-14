"""
LLM-Rosetta - OpenAI Responses API Converter

Top-level converter implementing the 6 explicit interfaces.
Composes ContentOps, ToolOps, MessageOps, and ConfigOps for full bidirectional
conversion between IR and OpenAI Responses API format.

Note: Responses API uses a flat list of items (input/output) instead of
nested messages. The converter handles this structural difference.
"""

import json
import time
from collections.abc import Mapping
from typing import Any, ClassVar, cast

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
    ToolResultEvent,
    UsageEvent,
)
from ..base import BaseConverter
from ..base.context import ConversionContext, StreamContext
from ..base.helpers.extensions import wire_extensions
from ..base.helpers.opaque import (
    get_opaque_origin,
    is_own_origin,
    set_opaque_origin,
)
from ..base.helpers.tool_group import (
    GENERATED_GROUP_MARKER,
    ToolGroup,
    group_tools,
)
from ..base.tools import (
    assign_tool_batch_ids,
    fix_orphaned_tool_calls_ir,
    get_intrinsic_kind,
    sanitize_tool_call_id,
    set_intrinsic_kind,
    strip_orphaned_tool_config,
)
from .stream_context import OpenResponsesStreamContext
from ._constants import (
    OPEN_RESPONSES_PRESERVE_FIELDS,
    OPEN_RESPONSES_REQUIRED_DEFAULTS,
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
from .config_ops import OpenResponsesConfigOps
from .content_ops import OpenResponsesContentOps
from .message_ops import OpenResponsesMessageOps
from .tool_ops import (
    _INTRINSIC_RESULT_TYPES,
    _ITEM_TO_INTRINSIC_KIND,
    _server_item_tool_input,
    OpenResponsesToolOps,
    harvest_additional_tools,
)
from .utils import build_message_preamble_events, resolve_call_id


#: Server-tool output item types this format reads (``web_search_call``,
#: ``code_interpreter_call``, …).  A streamed item of one of these types is a
#: provider-executed call; the value is its canonical intrinsic kind.
_SERVER_ITEM_INTRINSIC_KINDS: frozenset[str] = frozenset(_ITEM_TO_INTRINSIC_KIND)

#: Server *result* item types (``code_interpreter_call_output``, …).  A
#: streamed item of one of these types is a provider-executed result, carried
#: as a one-shot ``ToolResultEvent``.
_SERVER_RESULT_ITEM_TYPES: frozenset[str] = frozenset(_INTRINSIC_RESULT_TYPES)


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


_PASSTHROUGH_REQUEST_FIELDS = (
    "store",
    "background",
    "metadata",
    "user",
    "previous_response_id",
    "safety_identifier",
    "prompt",
)


def _forward_passthrough_request_fields(
    provider_request: dict[str, Any], ir_request: dict[str, Any]
) -> None:
    """Forward Responses-specific request fields via ``provider_extensions``.

    These fields have no IR equivalent but must reach the upstream in
    same-format round-trips so the client's values are honoured.
    """
    for field in _PASSTHROUGH_REQUEST_FIELDS:
        val = provider_request.get(field)
        if val is not None:
            ir_request.setdefault("provider_extensions", {})[field] = val
    text_obj = provider_request.get("text")
    if isinstance(text_obj, dict) and "verbosity" in text_obj:
        ir_request.setdefault("provider_extensions", {})["_text_verbosity"] = text_obj[
            "verbosity"
        ]


class OpenResponsesConverter(BaseConverter):
    """Open Responses spec converter (vendor-neutral base).

    Implements the 6 explicit conversion interfaces defined by BaseConverter.

    Uses composition of Ops classes for modular, testable conversion logic.

    Note: the spec uses ``input`` for request items and ``output`` for response
    items, with a flat item list structure.  :class:`OpenAIResponsesConverter`
    derives from this class and layers the OpenAI-specific profile on top.
    """

    content_ops_class = OpenResponsesContentOps
    tool_ops_class = OpenResponsesToolOps
    message_ops_class = OpenResponsesMessageOps
    config_ops_class = OpenResponsesConfigOps
    _CONVERTER_TAG = "open_responses"
    _PASSTHROUGH_RESTORE_KEY = "output"

    # Preserve-mode echo fields and required response defaults.  These are the
    # spec-level defaults (no OpenAI-only lifecycle fields, no forced
    # ``store: true``); the conversion pipeline layers a provider shim's
    # ``preserve_fields`` / ``store_default`` settings on top via
    # ``context.options``, and ``OpenAIResponsesConverter`` overrides the class
    # attributes directly for direct, context-free use.
    _PRESERVE_FIELDS: ClassVar[set[str]] = OPEN_RESPONSES_PRESERVE_FIELDS
    _REQUIRED_DEFAULTS: ClassVar[dict[str, Any]] = OPEN_RESPONSES_REQUIRED_DEFAULTS

    # Default response ID prefix.  The vendor-neutral spec has none; the OpenAI
    # profile sets ``"resp_"``.  Used as a fallback when no shim-driven prefix
    # (``context.options["response_id_prefix"]``) is available.
    _RESPONSE_ID_PREFIX = ""

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

    def _get_preserve_fields(
        self, context: ConversionContext | StreamContext | None = None
    ) -> set[str]:
        """Return the effective preserve/echo field set.

        Starts from the class-level spec set and adds any shim-declared
        ``preserve_fields`` carried in ``context.options`` (populated by the
        pipeline from the provider shim).  Direct converter use without a
        pipeline falls back to the class attribute.
        """
        fields = set(self._PRESERVE_FIELDS)
        if context is not None:
            extra = context.options.get("preserve_fields")
            if extra:
                fields.update(extra)
        return fields

    def _get_required_defaults(
        self, context: ConversionContext | StreamContext | None = None
    ) -> dict[str, Any]:
        """Return the effective response required-defaults map.

        Starts from the class-level spec defaults and, when the provider shim
        declares a ``store_default`` in ``context.options``, sets the ``store``
        field accordingly (the OpenAI profile stores responses server-side).
        """
        defaults = dict(self._REQUIRED_DEFAULTS)
        if context is not None:
            store_default = context.options.get("store_default")
            if store_default is not None:
                defaults["store"] = store_default
        return defaults

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
    def create_stream_context(cls) -> OpenResponsesStreamContext:
        """Create a stream context with Responses API specific state."""
        return OpenResponsesStreamContext()

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
        # Assign batch_ids over the full list before message_ops runs, so
        # per-format merge_tool_messages can group parallel tool results.
        ir_messages = assign_tool_batch_ids(ir_messages)
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
            # Merge _text_verbosity back into the text object
            verbosity = extensions.get("_text_verbosity")
            if verbosity is not None:
                result.setdefault("text", {})["verbosity"] = verbosity
            # ``wire_extensions`` drops internal ``_``-prefixed keys, so the
            # spec-form ``_open_responses_allowed_tools`` (re-emitted inside
            # ``tool_choice`` by ``_apply_tool_config``) never leaks as a
            # top-level field.  A legacy top-level ``allowed_tools`` is a plain
            # key and still passes through.
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
        # Codex transports its tool definitions inside `input` as
        # `additional_tools` items instead of the top-level `tools` field.
        # Harvest them before message conversion so the spent items can be
        # dropped: routed through the passthrough path they would strand a
        # content-less assistant message, which the Chat converter emits as
        # `{"role": "assistant", "content": ""}`.
        top_level_tools = list(provider_request.get("tools") or [])
        nested_tools, input_items = harvest_additional_tools(
            input_items, context.warnings
        )

        if isinstance(input_items, list):
            ir_messages = self.message_ops.p_messages_to_ir(
                input_items, provider=self._CONVERTER_TAG
            )
            ir_request["messages"] = ir_messages

        # 3. Tools (with process-level cache)
        # Filter out disabled tools before caching (external_web_access=False)
        tools = top_level_tools + nested_tools
        if tools:
            active_tools = [
                t
                for t in tools
                if not (isinstance(t, dict) and t.get("external_web_access") is False)
            ]
            if active_tools:
                ir_tools = self._get_cached_p_tools_to_ir(active_tools)
                if ir_tools:
                    # Naming is the target's call and happens there; this
                    # leg only reports the containers that arrived unnamed.
                    self._report_generated_groups(ir_tools, context.warnings)
                    ir_request["tools"] = ir_tools

        # 4-5. Tool choice + tool config
        self._convert_p_tool_config_to_ir(
            provider_request, ir_request, warnings=context.warnings
        )

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

        # 12. Responses-specific request fields not modelled in IR.
        _forward_passthrough_request_fields(provider_request, ir_request)

        # Preserve mode: capture request fields for echo-back in response
        ctx = context if context is not None else ConversionContext()
        if ctx.metadata_mode == "preserve":
            preserve = self._get_preserve_fields(ctx)
            echo = {k: v for k, v in provider_request.items() if k in preserve}
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

        # Convert output items to IR message content.  Unlike the request leg,
        # no ``provider=`` is passed: non-portable response items (e.g.
        # ``compaction``) are captured as ``provider_passthrough_items`` by the
        # caller, so this leg must not also emit ``ProviderPassthroughItem``.
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
    def _report_generated_groups(ir_tools: list[Any], warnings: list[str]) -> None:
        """Report each group whose name had to be invented.

        A container the client left unnamed is malformed input at every
        upstream that implements containers — OpenAI, DeepSeek and Volcengine
        all reject a missing or empty name — so the client should hear that
        their tools are now addressed under a name we chose.

        Emitted here rather than where the id is generated, because the
        flatten is process-cached: the first request carrying a given
        container would hear about it and every later one would not.  Once
        per group, not once per child: the children of one container share the
        name and the cause.
        """
        reported: set[str] = set()
        for tool in ir_tools:
            meta = tool.get("metadata")
            if not isinstance(meta, dict) or not meta.get(GENERATED_GROUP_MARKER):
                continue
            group = tool.get("tool_group")
            if not isinstance(group, str) or group in reported:
                continue
            reported.add(group)
            warnings.append(
                "A tool container arrived with no name of its own, so its "
                f"tools are addressed under the generated group {group!r}.  "
                "Calls route correctly, but the model reads a less meaningful "
                "group than a named container would have given it; name the "
                "container to fix that"
            )

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
            result["tools"] = self._tools_to_p(tools, ctx)
        tool_choice = ir_request.get("tool_choice")
        if tool_choice:
            allowed_tools = (ir_request.get("provider_extensions") or {}).get(
                "_open_responses_allowed_tools"
            )
            result["tool_choice"] = self.tool_ops.ir_tool_choice_to_p(
                tool_choice, allowed_tools=allowed_tools
            )
        tool_config = ir_request.get("tool_config")
        if tool_config:
            tc_fields = self.tool_ops.ir_tool_config_to_p(tool_config)
            result.update(tc_fields)

    def _tools_to_p(self, tools: list[Any], ctx: ConversionContext) -> list[Any]:
        """Render the tool list, groups and all.

        A target that carries tool groups gets the client's grouping back, as a
        container per group; every other target gets the flat list it has
        always had, with the group folded into the tool names.  Whether this
        target is one of the former is the shim's ``tools.tool_groups``,
        passed down on the context, because the converter cannot see a shim.

        The grouping is decided here rather than in the IR: the IR keeps every
        tool flat, with ``tool_group`` recording where it came from, and only a
        writer knows whether its wire format can say that.  Which is also why
        ``group_tools`` is a helper and not a pipeline pass — the ordering and
        the entry shape are the writer's business.

        Children go out under the names the naming pass settled on.  For a
        target that carries groups those are the client's own child names:
        that pass is told this target carries groups, so it qualifies a child
        only against the members of its own container — measured, an upstream
        accepts a child named like a top-level tool and like a child of another
        container.  Nothing is shortened here, because the name map is built
        from the IR, and rewriting a child at this layer would leave a replayed
        history call transliterated to a spelling this function had just
        discarded.
        """
        if not ctx.options.get("tool_groups"):
            return self._get_cached_ir_tools_to_p(tools)

        rendered: list[Any] = []
        for entry in group_tools(tools, carries_groups=True):
            if not isinstance(entry, ToolGroup):
                rendered.extend(self._get_cached_ir_tools_to_p([entry]))
                continue
            children = self._get_cached_ir_tools_to_p(list(entry.tools))
            # A group whose every member this target declined is not a group: a
            # childless container is what every upstream that accepts one
            # rejects.  The list conversion drops each declined entry on its
            # own, so this covers only a writer that declines all of them.
            # Nothing the pipeline produces reaches it — a tool the target
            # cannot express is dropped from the IR and its group never forms —
            # but the IR is a public TypedDict and the converter is public API,
            # so a caller can hand one to ``request_to_provider`` directly, and
            # that is the path the test covers.
            if not children:
                continue
            # ``description`` is required on a container by OpenAI and
            # OpenRouter: probed, an absent one is `400 Missing required
            # parameter: 'tools[0].description'` and `400 Invalid Responses API
            # request` respectively, while an empty string is accepted by both.
            # A container the client gave no description to therefore goes out
            # with "", rather than with text the client never wrote — a longer
            # placeholder is accepted everywhere but costs input tokens on
            # DeepSeek and says nothing the children do not already say.
            container: dict[str, Any] = {
                "type": "namespace",
                "name": entry.name,
                "description": entry.description,
                "tools": children,
            }
            rendered.append(container)
        return rendered

    def _convert_p_tool_config_to_ir(
        self,
        provider_request: dict[str, Any],
        ir_request: dict[str, Any],
        *,
        warnings: list[str] | None = None,
    ) -> None:
        """Extract tool_choice and tool_config from provider request into IR.

        Args:
            provider_request: Responses request body.
            ir_request: IR request under construction.
            warnings: Optional warning sink for lossy conversions.  A namespace
                ``tool_choice`` has no IR equivalent and is reported there.
        """
        tool_choice = provider_request.get("tool_choice")
        if tool_choice is not None:
            # Collect into a scratch dict so an empty ``provider_extensions`` is
            # not created when the tool choice carries no provider-bound object.
            extensions: dict[str, Any] = {}
            ir_request["tool_choice"] = self.tool_ops.p_tool_choice_to_ir(
                tool_choice, extensions=extensions, warnings=warnings
            )
            if extensions:
                ir_request.setdefault("provider_extensions", {}).update(extensions)
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
        preserve = self._get_preserve_fields(ctx)
        extras = {
            k: v
            for k, v in provider_response.items()
            if k in preserve and v is not None
        }
        if extras:
            ctx.store_response_extras(extras)

        items_meta = [
            _capture_item_metadata(item)
            for item in output_items
            if OpenResponsesMessageOps.is_portable_response_output_item(item)
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
        for k, v in self._get_required_defaults(ctx).items():
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
                    k: v
                    for k, v in response.items()
                    if k in self._get_preserve_fields(context)
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
            isinstance(context, OpenResponsesStreamContext)
            and context._inbound_reasoning_item_id
        ):
            event["provider_metadata"] = {
                "responses_reasoning_id": context._inbound_reasoning_item_id,
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

            elif item_type in _SERVER_ITEM_INTRINSIC_KINDS:
                self._emit_server_call_start_ir(item, chunk, context, events)

            elif item_type in _SERVER_RESULT_ITEM_TYPES:
                self._emit_server_result_ir(item, chunk, events)

            elif item_type == "reasoning" and isinstance(
                context, OpenResponsesStreamContext
            ):
                # Capture early so deltas can carry the source ID;
                # done event re-confirms authoritatively.
                item_id = item.get("id", "")
                if item_id:
                    context._inbound_reasoning_item_id = item_id

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

    @staticmethod
    def _emit_server_call_start_ir(
        item: dict[str, Any],
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Emit an intrinsic ToolCallStartEvent for a provider-executed call.

        The kind comes from the item's own type, never the name, so a server
        call never reads as a client ``function_call``.
        """
        intrinsic_kind = _ITEM_TO_INTRINSIC_KIND[item["type"]]
        call_id = item.get("call_id") or item.get("id", "")
        item_id = item.get("id", "")
        if context is not None:
            context.register_tool_call(call_id, intrinsic_kind, "intrinsic")
            context.register_tool_call_item(call_id, item_id)
        start_event = ToolCallStartEvent(
            type="tool_call_start",
            tool_call_id=call_id,
            tool_name=intrinsic_kind,
            tool_type="intrinsic",
        )
        set_intrinsic_kind(start_event, intrinsic_kind)
        if item_id and item_id != call_id:
            start_event.setdefault("provider_metadata", {})["responses_item_id"] = (
                item_id
            )
        output_index = chunk.get("output_index")
        if output_index is not None:
            start_event["tool_call_index"] = output_index
        events.append(start_event)

    def _emit_server_result_ir(
        self,
        item: dict[str, Any],
        chunk: dict[str, Any],
        events: list[IRStreamEvent],
    ) -> None:
        """Emit a one-shot ToolResultEvent for a provider-executed result.

        The item is built with the same reader the non-streaming path uses, so
        the payload shape matches; the provider's whole item is kept on the
        event so a same-dialect writer can replay it verbatim.
        """
        part = OpenResponsesToolOps.p_tool_result_to_ir(item)
        result_event = ToolResultEvent(
            type="tool_result",
            tool_call_id=part["tool_call_id"],
            result=part["result"],
        )
        if part.get("tool_type") is not None:
            result_event["tool_type"] = part["tool_type"]
        if "is_error" in part:
            result_event["is_error"] = part["is_error"]
        pm = dict(part.get("provider_metadata") or {})
        pm["_native"] = {"base": self._CONVERTER_TAG, "item": dict(item)}
        result_event["provider_metadata"] = pm
        output_index = chunk.get("output_index")
        if output_index is not None:
            result_event["block_index"] = output_index
        events.append(result_event)

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
        elif item_type in _SERVER_ITEM_INTRINSIC_KINDS:
            # The full server item only arrives on `done` (its `action` / `code`
            # is absent at `added` time).  Move the payload into the IR as a
            # single residual delta so the write leg rebuilds the same item.
            call_id = item.get("call_id") or item.get("id", "")
            if context is not None and call_id and not context.get_tool_name(call_id):
                context.register_tool_call(
                    call_id, _ITEM_TO_INTRINSIC_KIND[item_type], "intrinsic"
                )
            tool_input = _server_item_tool_input(item, item_type)
            self._emit_residual_tool_call_delta(
                json.dumps(tool_input), call_id, chunk, context, events
            )
        elif item_type == "reasoning" and isinstance(
            context, OpenResponsesStreamContext
        ):
            item_id = item.get("id", "")
            if item_id:
                context._inbound_reasoning_item_id = item_id
            encrypted_content = item.get("encrypted_content")
            if encrypted_content:
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
                set_opaque_origin(event, redacted_data="open_responses")
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

    def _emit_residual_tool_call_delta(
        self,
        final_text: str,
        call_id: str,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        """Emit a ToolCallDeltaEvent for any portion of *final_text* not yet
        delivered by preceding ``.delta`` events, then store *final_text* as
        the authoritative value in *context*.

        Some upstreams deliver the complete payload only on the ``.done``
        event with no preceding ``.delta`` — ARGO's native /v1/responses does
        this for cached turns (#800).  The delta handler is the only path that
        moves arguments into the IR stream, so without this guard the payload
        is silently dropped.  See #800 and #803.
        """
        if context is not None and call_id:
            accumulated = context._tool_call_args.get(call_id, "")
        else:
            accumulated = final_text
        if final_text and final_text != accumulated:
            residual = (
                final_text[len(accumulated) :]
                if final_text.startswith(accumulated)
                else final_text
            )
            if residual:
                delta_event = ToolCallDeltaEvent(
                    type="tool_call_delta",
                    tool_call_id=call_id,
                    arguments_delta=residual,
                )
                output_index = chunk.get("output_index")
                if output_index is not None:
                    delta_event["tool_call_index"] = output_index
                events.append(delta_event)

        if context is not None and call_id:
            context.set_tool_call_args(call_id, final_text)

    def _handle_p_function_call_args_done_to_ir(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None,
        events: list[IRStreamEvent],
    ) -> None:
        call_id = resolve_call_id(chunk, context)
        self._emit_residual_tool_call_delta(
            chunk.get("arguments", ""), call_id, chunk, context, events
        )

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
        self._emit_residual_tool_call_delta(
            chunk.get("input", ""), call_id, chunk, context, events
        )

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
        "tool_result": "_handle_ir_tool_result_to_p",
    }

    def stream_response_to_provider(
        self,
        event: IRStreamEvent,
        context: StreamContext | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Convert IR stream event with automatic context upgrade.

        If a base ``StreamContext`` is passed, it is automatically upgraded
        to ``OpenResponsesStreamContext`` (preserving existing state) so
        that callers do not need to know about the provider-specific subclass.
        """
        # Auto-upgrade base StreamContext to the provider-specific subclass.
        # Cache the upgraded instance in metadata so state persists across calls.
        if context is not None and not isinstance(context, OpenResponsesStreamContext):
            cached = context.metadata.get("_responses_stream_ctx")
            if cached is None:
                cached = OpenResponsesStreamContext.from_base(context)
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
        if isinstance(context, OpenResponsesStreamContext):
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
            for k, v in self._get_required_defaults(context).items():
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
        context: OpenResponsesStreamContext | None,
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
        context: OpenResponsesStreamContext | None,
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
        context: OpenResponsesStreamContext | None,
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
        context: OpenResponsesStreamContext | None,
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
        ctx = context if isinstance(context, OpenResponsesStreamContext) else None
        provider_metadata = event.get("provider_metadata") or {}
        source_item_id = provider_metadata.get("responses_reasoning_id", "")
        if not isinstance(source_item_id, str):
            source_item_id = ""

        item = None
        if ctx is not None:
            # Deltas with no provenance all belong to one item; a new
            # source id starts a new one, so each item keeps the
            # encrypted_content that was minted against its own id.
            item = (
                ctx.reasoning_item(source_item_id)
                if source_item_id
                else ctx.current_reasoning_item
            )

        if ctx is not None and item is None:
            output_index = ctx.next_output_index()
            item = ctx.start_reasoning_item(
                source_item_id or generate_reasoning_id(ctx.response_id, output_index),
                output_index,
            )

            results.append(
                {
                    "type": ResponsesEventType.OUTPUT_ITEM_ADDED,
                    "output_index": item.output_index,
                    "item": {
                        "id": item.item_id,
                        "type": "reasoning",
                        "summary": [],
                        "status": "in_progress",
                    },
                }
            )
            results.append(
                {
                    "type": ResponsesEventType.REASONING_SUMMARY_PART_ADDED,
                    "item_id": item.item_id,
                    "output_index": item.output_index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": ""},
                }
            )

        enc = event.get("encrypted_content", "")
        if enc and not is_own_origin(
            get_opaque_origin(event, "redacted_data"), "open_responses"
        ):
            # Foreign opaque blob — not valid for the Responses API.
            enc = ""
        if item is not None and source_item_id and enc:
            item.encrypted_content = enc
            if not event["reasoning"]:
                return results

        reasoning_text = event["reasoning"]
        if item is not None:
            item.accumulated_text += reasoning_text

        item_id = item.item_id if item is not None else ""
        r_idx = item.output_index if item is not None else 0
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

        # A provider-executed call re-emits the provider's own server item
        # type, never a client ``function_call``.  The kind comes from the
        # event's ``intrinsic_kind`` carrier, never from the tool name; a kind
        # this dialect cannot render is dropped rather than downgraded.
        intrinsic_item: dict[str, Any] | None = None
        if tool_type == "intrinsic":
            intrinsic_item = self.tool_ops.stream_intrinsic_item(
                get_intrinsic_kind(event, tool_name), item_id, status="in_progress"
            )
            if intrinsic_item is None:
                return {}

        # Register in context for later done events, carrying the intrinsic
        # kind through so the done/completed items restore it without
        # re-deriving it from the name.
        if context is not None and call_id:
            context.register_tool_call(
                call_id, tool_name, tool_type, provider_metadata=pm or None
            )
            context.register_tool_call_item(call_id, item_id)

        stream_ctx = (
            context if isinstance(context, OpenResponsesStreamContext) else None
        )

        if stream_ctx is not None:
            output_index = stream_ctx.next_output_index()
            stream_ctx._tool_call_output_indices[call_id] = output_index
            # Record the namespace once, so the later done/completed items
            # can be rebuilt with it straight from the context.
            namespace = pm.get("namespace")
            if namespace and call_id:
                stream_ctx._tool_call_namespaces[call_id] = namespace
        else:
            tc_index = event.get("tool_call_index")
            output_index = tc_index if tc_index is not None else 0

        item: dict[str, Any]
        if tool_type == "intrinsic":
            assert intrinsic_item is not None
            item = intrinsic_item
        elif tool_type == "custom":
            item = self._build_stream_custom_tool_call_item(
                stream_ctx,
                item_id=item_id,
                call_id=call_id,
                tool_name=tool_name,
                input_str="",
                status="in_progress",
            )
        else:
            item = self._build_stream_function_call_item(
                stream_ctx,
                item_id=item_id,
                call_id=call_id,
                tool_name=tool_name,
                arguments="",
                status="in_progress",
            )

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

        if isinstance(context, OpenResponsesStreamContext) and call_id:
            output_index = context._tool_call_output_indices.get(call_id, 0)
        else:
            output_index = tc_index if tc_index is not None else 0

        # Determine event type based on tool type (custom vs function)
        tool_type = (
            context.get_tool_type(call_id)
            if context is not None and call_id
            else "function"
        )
        if tool_type == "intrinsic":
            # A server item carries its payload whole (``action`` / ``code``),
            # not as an arguments delta the way a ``function_call`` does, so
            # the accumulated bytes are kept for the done/completed item and
            # no provider delta event is emitted.
            return {}
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

    def _handle_ir_tool_result_to_p(
        self,
        event: ToolResultEvent,
        context: StreamContext | None,
    ) -> dict[str, Any]:
        """Handle ToolResultEvent → a provider server result item.

        A result read from this same dialect is replayed verbatim from the
        ``_native`` item the reader stored.  Otherwise the item is rebuilt from
        the IR fields for a kind that has its own ``*_call_output`` item; a
        ``web_search`` result (which rides on the call item) and any kind this
        dialect cannot render are dropped silently, matching the non-streaming
        writer.  A server result is never re-encoded as a client
        ``function_call_output``.
        """
        pm = event.get("provider_metadata") or {}
        native = pm.get("_native")
        if isinstance(native, dict) and native.get("base") == self._CONVERTER_TAG:
            native_item = native.get("item")
            if isinstance(native_item, dict):
                return self._stream_result_chunk(dict(native_item), event, context)

        item = self.tool_ops.stream_result_item(
            get_intrinsic_kind(event, ""),
            sanitize_tool_call_id(event["tool_call_id"]),
            event.get("result", ""),
        )
        if item is None:
            return {}
        return self._stream_result_chunk(item, event, context)

    @staticmethod
    def _stream_result_chunk(
        item: dict[str, Any],
        event: ToolResultEvent,
        context: StreamContext | None,
    ) -> dict[str, Any]:
        """Wrap a server result item in an ``output_item.added`` chunk."""
        result: dict[str, Any] = {
            "type": ResponsesEventType.OUTPUT_ITEM_ADDED,
            "item": item,
        }
        output_index = event.get("block_index")
        if isinstance(context, OpenResponsesStreamContext):
            if output_index is None:
                output_index = context.next_output_index()
            result["output_index"] = output_index
        elif output_index is not None:
            result["output_index"] = output_index
        return result

    def _handle_ir_finish_to_p(
        self,
        event: FinishEvent,
        context: OpenResponsesStreamContext | None,
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
        context: OpenResponsesStreamContext | None,
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
        self, context: OpenResponsesStreamContext | None
    ) -> list[dict[str, Any]]:
        """Collect text and tool call output items from stream context."""
        output: list[dict[str, Any]] = []
        if context is None:
            return output

        # (output_index, item) so the array can be restored to emission
        # order below — a reasoning item has to stay immediately before
        # the tool call it produced, which front-loading would break
        # once a turn holds more than one.
        entries: list[tuple[int, dict[str, Any]]] = []

        for item in context._reasoning_items:
            reasoning_item: dict[str, Any] = {
                "id": item.item_id,
                "type": "reasoning",
                "summary": [
                    {
                        "type": "summary_text",
                        "text": item.accumulated_text,
                    }
                ],
                "status": "completed",
            }
            if item.encrypted_content:
                reasoning_item["encrypted_content"] = item.encrypted_content
            entries.append((item.output_index, reasoning_item))

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
            entries.append((context._message_output_index, msg_item))

        for call_id in context.tool_call_ids:
            tool_name = context.get_tool_name(call_id)
            arguments = context._tool_call_args.get(call_id, "")
            tc_item_id = context.get_tool_call_item_id(call_id) or call_id
            tool_type = context.get_tool_type(call_id)
            tc_index = context._tool_call_output_indices.get(call_id, -1)

            if tool_type == "intrinsic":
                intrinsic_item = self._build_stream_intrinsic_item(
                    context,
                    call_id=call_id,
                    item_id=tc_item_id,
                    kind=self._stream_intrinsic_kind(context, call_id, tool_name),
                    arguments=arguments,
                    status="completed",
                )
                if intrinsic_item is not None:
                    entries.append((tc_index, intrinsic_item))
            elif tool_type == "custom":
                entries.append(
                    (
                        tc_index,
                        self._build_stream_custom_tool_call_item(
                            context,
                            item_id=tc_item_id,
                            call_id=call_id,
                            tool_name=tool_name,
                            input_str=arguments,
                            status="completed",
                        ),
                    )
                )
            else:
                entries.append(
                    (
                        tc_index,
                        self._build_stream_function_call_item(
                            context,
                            item_id=tc_item_id,
                            call_id=call_id,
                            tool_name=tool_name,
                            arguments=arguments,
                            status="completed",
                        ),
                    )
                )

        # Only reorder when every item knows its index; otherwise keep the
        # historical reasoning → message → tool-calls order.  sort() is
        # stable, so ties stay as appended.
        if all(index >= 0 for index, _ in entries):
            entries.sort(key=lambda entry: entry[0])
        output.extend(item for _, item in entries)
        return output

    @staticmethod
    def _build_stream_function_call_item(
        context: OpenResponsesStreamContext | None,
        *,
        item_id: str,
        call_id: str,
        tool_name: str,
        arguments: str,
        status: str,
    ) -> dict[str, Any]:
        """Build a streamed ``function_call`` item.

        Shared by the three points that emit one — ``output_item.added``,
        ``output_item.done``, and the ``response.completed`` output list —
        so they cannot drift apart.  Re-attaches the tool's namespace when
        it was flattened out of a namespace container on the request leg.
        """
        item: dict[str, Any] = {
            "id": item_id,
            "type": "function_call",
            "call_id": call_id,
            "name": tool_name,
            "arguments": arguments,
            "status": status,
        }
        if context is not None:
            namespace = context._tool_call_namespaces.get(call_id)
            if namespace:
                item["namespace"] = namespace
        return item

    @staticmethod
    def _build_stream_custom_tool_call_item(
        context: OpenResponsesStreamContext | None,
        *,
        item_id: str,
        call_id: str,
        tool_name: str,
        input_str: str,
        status: str,
    ) -> dict[str, Any]:
        """Build a streamed ``custom_tool_call`` item.

        The custom-tool counterpart of
        :meth:`_build_stream_function_call_item`, shared by the same three
        emit points and carrying the namespace for the same reason.
        """
        item: dict[str, Any] = {
            "id": item_id,
            "type": "custom_tool_call",
            "call_id": call_id,
            "name": tool_name,
            "input": input_str,
            "status": status,
        }
        if context is not None:
            namespace = context._tool_call_namespaces.get(call_id)
            if namespace:
                item["namespace"] = namespace
        return item

    def _build_stream_intrinsic_item(
        self,
        context: OpenResponsesStreamContext | None,
        *,
        call_id: str,
        item_id: str,
        kind: str,
        arguments: str,
        status: str,
    ) -> dict[str, Any] | None:
        """Rebuild a streamed server item from the accumulated payload.

        The counterpart of :meth:`_build_stream_function_call_item` for a
        provider-executed call: it restores the provider's own item type
        (``web_search_call`` / ``code_interpreter_call``, …) with the payload
        the reader moved through a ``ToolCallDeltaEvent``.  ``None`` when the
        kind cannot be rendered as a server item.
        """
        fields: dict[str, Any] = {}
        if arguments:
            try:
                parsed = json.loads(arguments)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                fields = parsed
        return self.tool_ops.stream_intrinsic_item(kind, item_id, fields, status=status)

    @staticmethod
    def _stream_intrinsic_kind(
        context: StreamContext | None, call_id: str, tool_name: str
    ) -> str:
        """Read the intrinsic kind the start event registered, if any.

        The kind is carried in the call's ``provider_metadata`` so the done
        and completed items restore it without re-deriving it from the name;
        the registered name (already the canonical kind for a server call) is
        only a last-resort fallback when no metadata was carried.
        """
        if context is None:
            return tool_name
        pm = context.get_tool_call_provider_metadata(call_id) or {}
        return pm.get("intrinsic_kind") or tool_name

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

    def _apply_finish_echo(
        self, response: dict[str, Any], context: OpenResponsesStreamContext
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
        for k, v in self._get_required_defaults(context).items():
            if k not in core_keys and k not in response:
                response[k] = v
        for k, v in echo.items():
            if k not in core_keys:
                response[k] = v

    def _emit_reasoning_done_events(
        self,
        context: OpenResponsesStreamContext,
        results: list[dict[str, Any]],
    ) -> None:
        """Emit reasoning lifecycle done events, one set per item."""
        for item in context._reasoning_items:
            accumulated = item.accumulated_text
            results.append(
                {
                    "type": ResponsesEventType.REASONING_SUMMARY_TEXT_DONE,
                    "item_id": item.item_id,
                    "output_index": item.output_index,
                    "summary_index": 0,
                    "text": accumulated,
                }
            )
            results.append(
                {
                    "type": ResponsesEventType.REASONING_SUMMARY_PART_DONE,
                    "item_id": item.item_id,
                    "output_index": item.output_index,
                    "summary_index": 0,
                    "part": {"type": "summary_text", "text": accumulated},
                }
            )
            reasoning_item: dict[str, Any] = {
                "id": item.item_id,
                "type": "reasoning",
                "summary": [{"type": "summary_text", "text": accumulated}],
                "status": "completed",
            }
            if item.encrypted_content:
                reasoning_item["encrypted_content"] = item.encrypted_content
            results.append(
                {
                    "type": ResponsesEventType.OUTPUT_ITEM_DONE,
                    "output_index": item.output_index,
                    "item": reasoning_item,
                }
            )

    def _emit_text_done_events(
        self,
        context: OpenResponsesStreamContext,
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
        context: OpenResponsesStreamContext,
        results: list[dict[str, Any]],
    ) -> None:
        """Emit done events for each tool call."""
        for tc_idx, call_id in enumerate(context.tool_call_ids):
            tool_name = context.get_tool_name(call_id)
            arguments = context._tool_call_args.get(call_id, "")
            item_id = context.get_tool_call_item_id(call_id) or call_id
            output_index = context._tool_call_output_indices.get(call_id, tc_idx)
            tool_type = context.get_tool_type(call_id)

            if tool_type == "intrinsic":
                # A server item has no ``*_arguments.done`` event; the item
                # itself carries the payload, so only the output_item.done is
                # emitted.
                intrinsic_item = self._build_stream_intrinsic_item(
                    context,
                    call_id=call_id,
                    item_id=item_id,
                    kind=self._stream_intrinsic_kind(context, call_id, tool_name),
                    arguments=arguments,
                    status="completed",
                )
                if intrinsic_item is None:
                    continue
                results.append(
                    {
                        "type": ResponsesEventType.OUTPUT_ITEM_DONE,
                        "output_index": output_index,
                        "item": intrinsic_item,
                    }
                )
            elif tool_type == "custom":
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
                        "item": self._build_stream_custom_tool_call_item(
                            context,
                            item_id=item_id,
                            call_id=call_id,
                            tool_name=tool_name,
                            input_str=arguments,
                            status="completed",
                        ),
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
                        "item": self._build_stream_function_call_item(
                            context,
                            item_id=item_id,
                            call_id=call_id,
                            tool_name=tool_name,
                            arguments=arguments,
                            status="completed",
                        ),
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


class OpenAIResponsesConverter(OpenResponsesConverter):
    """OpenAI Responses API converter (OpenAI profile preset).

    A thin preset over the vendor-neutral :class:`OpenResponsesConverter` that
    turns the OpenAI profile on for **direct, context-free use**:

    - the ``resp_`` response-id prefix (the spec defines no prefix);
    - OpenAI-only lifecycle echo fields (``billing``, ``moderation``, ...);
    - OpenAI's default of server-side storage (``store: true``).

    The conversion pipeline no longer selects this class: ``openai_responses``
    is a *shim* over the ``open_responses`` base, and the shim's
    ``response_id_prefix`` / ``preserve_fields`` / ``store_default`` settings
    carry the same traits through ``context.options``.  This class is retained
    as an importable preset so existing direct users keep their behavior.
    """

    _CONVERTER_TAG = "openai_responses"
    _RESPONSE_ID_PREFIX = "resp_"
    _PRESERVE_FIELDS = RESPONSES_PRESERVE_FIELDS
    _REQUIRED_DEFAULTS = RESPONSES_REQUIRED_DEFAULTS
