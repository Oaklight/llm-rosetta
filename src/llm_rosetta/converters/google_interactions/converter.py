"""
LLM-Rosetta - Google Interactions Converter

Top-level converter implementing bidirectional conversion between IR
and Google Interactions API format via BaseConverter hooks.
"""

import json
import logging
import time
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any, cast

from ...types.ir.request import IRRequest
from ...types.ir.response import (
    FinishReason,
    IRResponse,
    UsageInfo,
)
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
    ToolResultEvent,
    UsageEvent,
)
from ..base import BaseConverter
from ..base.helpers.opaque import (
    get_opaque_origin,
    is_own_origin,
    set_opaque_origin,
)
from ..base.helpers.system_message_hoist import hoist_system_in_converter
from ..base.tools import is_intrinsic_part
from ..base.tools.intrinsic import get_intrinsic_kind, set_intrinsic_kind
from ...transforms.ir import hoist_late_system_messages
from ..base.context import ConversionContext, StreamContext
from .config_ops import GoogleInteractionsConfigOps
from .content_ops import GoogleInteractionsContentOps
from .message_ops import GoogleInteractionsMessageOps
from .tool_ops import (
    INTRINSIC_KIND_TO_CALL_TYPE,
    INTRINSIC_KIND_TO_RESULT_TYPE,
    SERVER_CALL_TYPES,
    SERVER_RESULT_TYPES,
    GoogleInteractionsToolOps,
)

_STATUS_TO_FINISH_REASON: dict[str, str] = {
    "completed": "stop",
    "requires_action": "tool_calls",
    "incomplete": "length",
    "budget_exceeded": "length",
    "failed": "error",
    "cancelled": "cancelled",
}

_FINISH_REASON_TO_STATUS: dict[str, str] = {
    "stop": "completed",
    "tool_calls": "requires_action",
    "length": "incomplete",
    "error": "failed",
    "cancelled": "cancelled",
}

logger = logging.getLogger(__name__)

_STEP_TYPE_TO_BLOCK_TYPE: dict[str, str] = {
    "model_output": "text",
    "thought": "thinking",
    "function_call": "tool_use",
}


def _parse_iso_to_epoch(iso_str: str | None) -> int:
    if not iso_str:
        return int(time.time())
    try:
        dt = datetime.fromisoformat(iso_str.replace("Z", "+00:00"))
        return int(dt.timestamp())
    except (ValueError, TypeError):
        return int(time.time())


_PROVIDER_EXT_KEYS = (
    "store",
    "previous_interaction_id",
    "background",
    "safety_settings",
)


def _copy_provider_extensions(ir_request: IRRequest, result: dict[str, Any]) -> None:
    exts = ir_request.get("provider_extensions", {})
    for key in _PROVIDER_EXT_KEYS:
        if key in exts:
            result[key] = exts[key]


def _extract_provider_extensions(req: dict[str, Any], ir: dict[str, Any]) -> None:
    exts: dict[str, Any] = {}
    for key in _PROVIDER_EXT_KEYS:
        if key in req:
            exts[key] = req[key]
    if exts:
        ir["provider_extensions"] = exts


class GoogleInteractionsConverter(BaseConverter):
    """Google Interactions API converter."""

    _CONVERTER_TAG = "google_interactions"
    _RESPONSE_ID_PREFIX = ""
    _PASSTHROUGH_RESTORE_KEY = "steps"
    _INTRINSIC_IR_TRANSFORMS = (hoist_late_system_messages(),)

    content_ops_class = GoogleInteractionsContentOps
    tool_ops_class = GoogleInteractionsToolOps
    message_ops_class = GoogleInteractionsMessageOps
    config_ops_class = GoogleInteractionsConfigOps

    def __init__(self):
        self.content_ops = self.content_ops_class()
        self.tool_ops = self.tool_ops_class()
        self.message_ops = self.message_ops_class(self.content_ops, self.tool_ops)
        self.config_ops = self.config_ops_class()

    # ── _do_request_to_provider ────────────────────────────────────

    def _do_request_to_provider(
        self,
        ir_request: IRRequest,
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {"model": ir_request["model"]}

        self._extract_system_instruction(ir_request, result)
        self._convert_messages_to_input(ir_request, result)

        tools = ir_request.get("tools")
        if tools:
            converted = [self.tool_ops.ir_tool_to_p(t) for t in tools]
            result["tools"] = [t for t in converted if t]

        self._build_generation_config(ir_request, result)

        ir_rf = ir_request.get("response_format")
        if ir_rf:
            fmt = self.config_ops.ir_response_format_to_p(ir_rf)
            if fmt:
                result["response_format"] = fmt

        stream = ir_request.get("stream")
        if stream and stream.get("enabled"):
            result["stream"] = True

        _copy_provider_extensions(ir_request, result)
        return result

    def _extract_system_instruction(
        self, ir_request: IRRequest, result: dict[str, Any]
    ) -> None:
        sys_instr = ir_request.get("system_instruction")
        if sys_instr:
            texts = [p["text"] for p in sys_instr if p.get("type") == "text"]
            if texts:
                result["system_instruction"] = " ".join(texts)

    def _convert_messages_to_input(
        self, ir_request: IRRequest, result: dict[str, Any]
    ) -> None:
        messages = list(ir_request.get("messages", []))

        # Defense-in-depth: idempotent with pipeline's intrinsic transform.
        messages, hoisted_si = hoist_system_in_converter(messages, ir_request)
        if hoisted_si and "system_instruction" not in result:
            texts = [
                p["text"]
                for p in hoisted_si
                if isinstance(p, dict) and p.get("type") == "text"
            ]
            if texts:
                result["system_instruction"] = " ".join(texts)

        non_system: list = []
        for msg in messages:
            if hasattr(msg, "get") and msg.get("role") == "system":
                content = msg.get("content", [])
                sys_texts = [
                    cast(dict[str, Any], p).get("text", "")
                    for p in content
                    if isinstance(p, dict) and p.get("type") == "text"
                ]
                if sys_texts and "system_instruction" not in result:
                    result["system_instruction"] = " ".join(sys_texts)
            else:
                non_system.append(msg)

        steps = self.message_ops.ir_messages_to_p_steps(non_system)
        if (
            len(steps) == 1
            and steps[0].get("type") == "user_input"
            and len(steps[0].get("content", [])) == 1
            and steps[0]["content"][0].get("type") == "text"
        ):
            result["input"] = steps[0]["content"][0]["text"]
        else:
            result["input"] = steps

    def _build_generation_config(
        self, ir_request: IRRequest, result: dict[str, Any]
    ) -> None:
        gen_config: dict[str, Any] = {}
        ir_gen = ir_request.get("generation")
        if ir_gen:
            gen_config.update(self.config_ops.ir_generation_config_to_p(ir_gen))
        ir_reasoning = ir_request.get("reasoning")
        if ir_reasoning:
            gen_config.update(self.config_ops.ir_reasoning_to_p(ir_reasoning))
        ir_tc = ir_request.get("tool_choice")
        if ir_tc:
            gen_config["tool_choice"] = self.tool_ops.ir_tool_choice_to_p(ir_tc)
        if gen_config:
            result["generation_config"] = gen_config

    # ── _do_request_from_provider ──────────────────────────────────

    def _do_request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        req = provider_request
        ir: dict[str, Any] = {"model": req.get("model", ""), "messages": []}

        sys_instr = req.get("system_instruction")
        if sys_instr and isinstance(sys_instr, str):
            ir["system_instruction"] = [{"type": "text", "text": sys_instr}]

        self._parse_input_to_messages(req, ir)

        tools = req.get("tools")
        if tools:
            ir["tools"] = [self.tool_ops.p_tool_to_ir(t) for t in tools]

        self._parse_generation_config(req, ir)

        rf = req.get("response_format")
        if rf:
            ir["response_format"] = self.config_ops.p_response_format_to_ir(rf)

        if req.get("stream"):
            ir["stream"] = {"enabled": True}

        _extract_provider_extensions(req, ir)
        return ir

    def _parse_input_to_messages(self, req: dict[str, Any], ir: dict[str, Any]) -> None:
        inp = req.get("input")
        if isinstance(inp, str):
            ir["messages"] = [
                {"role": "user", "content": [{"type": "text", "text": inp}]}
            ]
        elif isinstance(inp, list) and inp:
            first = inp[0]
            if isinstance(first, dict) and first.get("type") in (
                "user_input",
                "model_output",
                "thought",
                "function_call",
                "function_result",
            ):
                ir["messages"] = self.message_ops.p_steps_to_ir_messages(inp)
            else:
                parts = [
                    p
                    for c in inp
                    if (p := self.content_ops.p_content_to_ir(c)) is not None
                ]
                if parts:
                    ir["messages"] = [{"role": "user", "content": parts}]

    def _parse_generation_config(self, req: dict[str, Any], ir: dict[str, Any]) -> None:
        gen_cfg = req.get("generation_config", {})
        if not gen_cfg:
            return
        ir_gen = self.config_ops.p_generation_config_to_ir(gen_cfg)
        if ir_gen:
            ir["generation"] = ir_gen
        ir_reason = self.config_ops.p_reasoning_to_ir(gen_cfg)
        if ir_reason:
            ir["reasoning"] = ir_reason
        tc = gen_cfg.get("tool_choice")
        if tc:
            ir["tool_choice"] = self.tool_ops.p_tool_choice_to_ir(tc)

    # ── _do_response_from_provider ─────────────────────────────────

    def _do_response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        resp = provider_response
        steps = resp.get("steps", [])
        ir_messages = self.message_ops.p_steps_to_ir_messages(steps)

        combined_parts: list = []
        for msg in ir_messages:
            role = msg.get("role")
            if role == "assistant":
                combined_parts.extend(msg.get("content", []))
            elif role == "tool":
                # ``p_steps_to_ir_messages`` emits a ``role: "tool"`` message
                # for a server result step.  A response is a single assistant
                # turn, so its server results ride inline in that turn, in the
                # order the steps arrived (#992).  A client result cannot occur
                # in a response turn, so only the intrinsic ones are kept.
                combined_parts.extend(
                    part for part in msg.get("content", []) if is_intrinsic_part(part)
                )

        assistant_msg: Any = {
            "role": "assistant",
            "content": combined_parts
            if combined_parts
            else [{"type": "text", "text": ""}],
        }

        status = resp.get("status", "completed")
        reason_str = _STATUS_TO_FINISH_REASON.get(status, "stop")
        finish_reason: Any = {"reason": reason_str}

        choice: Any = {
            "index": 0,
            "message": assistant_msg,
            "finish_reason": finish_reason,
        }

        ir_response: dict[str, Any] = {
            "id": resp.get("id", ""),
            "object": "response",
            "created": _parse_iso_to_epoch(resp.get("created")),
            "model": resp.get("model", ""),
            "choices": [choice],
        }

        usage = resp.get("usage")
        if usage:
            ir_response["usage"] = self._build_p_usage_to_ir(usage)

        return ir_response

    # ── _do_response_to_provider ───────────────────────────────────

    def _do_response_to_provider(
        self,
        ir_response: IRResponse,
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        choices = ir_response.get("choices", [])
        finish_reason = "stop"
        assistant_msg = None
        if choices:
            choice = choices[0]
            finish_reason = choice.get("finish_reason", {}).get("reason", "stop")
            assistant_msg = choice.get("message")

        status = _FINISH_REASON_TO_STATUS.get(finish_reason, "completed")

        steps: list[dict] = []
        if assistant_msg:
            steps = self.message_ops.ir_messages_to_p_steps([assistant_msg])

        # Override status when steps contain function_call but finish_reason
        # was "stop" (Google generateContent returns STOP for tool calls)
        if status == "completed" and any(
            s.get("type") == "function_call" for s in steps
        ):
            status = "requires_action"

        created_ts = ir_response.get("created", int(time.time()))
        created_iso = datetime.fromtimestamp(created_ts, tz=timezone.utc).isoformat()

        result: dict[str, Any] = {
            "id": ir_response.get("id", ""),
            "object": "interaction",
            "model": ir_response.get("model", ""),
            "status": status,
            "steps": steps,
            "created": created_iso,
            "updated": created_iso,
        }

        usage = ir_response.get("usage")
        if usage:
            result["usage"] = self._build_ir_usage_to_p(usage)

        return result

    # ── Usage conversion ───────────────────────────────────────────

    @staticmethod
    def _build_p_usage_to_ir(p_usage: dict[str, Any]) -> UsageInfo:
        result: UsageInfo = {
            "prompt_tokens": p_usage.get("total_input_tokens", 0),
            "completion_tokens": p_usage.get("total_output_tokens", 0),
            "total_tokens": p_usage.get("total_tokens", 0),
        }
        if "total_thought_tokens" in p_usage:
            result["reasoning_tokens"] = p_usage["total_thought_tokens"]
        if "total_cached_tokens" in p_usage:
            result["cache_read_tokens"] = p_usage["total_cached_tokens"]
        return result

    @staticmethod
    def _build_ir_usage_to_p(ir_usage: Mapping[str, Any]) -> dict[str, Any]:
        result: dict[str, Any] = {
            "total_input_tokens": ir_usage.get("prompt_tokens", 0),
            "total_output_tokens": ir_usage.get("completion_tokens", 0),
            "total_tokens": ir_usage.get("total_tokens", 0),
        }
        if "reasoning_tokens" in ir_usage:
            result["total_thought_tokens"] = ir_usage["reasoning_tokens"]
        if "cache_read_tokens" in ir_usage:
            result["total_cached_tokens"] = ir_usage["cache_read_tokens"]
        return result

    # ── Tool config ────────────────────────────────────────────────

    def _apply_tool_config(
        self,
        ir_request: IRRequest,
        result: dict[str, Any],
        ctx: ConversionContext,
    ) -> None:
        pass  # Tools are already handled in _do_request_to_provider

    # ── Streaming ──────────────────────────────────────────────────

    def stream_response_from_provider(
        self,
        chunk: dict[str, Any],
        context: StreamContext | None = None,
    ) -> list[IRStreamEvent]:
        data = chunk.get("data", chunk)
        if not isinstance(data, dict):
            return []
        handler = {
            "interaction.created": self._stream_interaction_created,
            "step.start": self._stream_step_start,
            "step.delta": self._stream_step_delta,
            "step.stop": self._stream_step_stop,
            "interaction.completed": self._stream_interaction_completed,
        }.get(data.get("event_type", ""))
        return handler(data, context) if handler else []

    def _stream_interaction_created(
        self, data: dict, context: StreamContext | None = None
    ) -> list[IRStreamEvent]:
        interaction = data.get("interaction", {})
        return [
            StreamStartEvent(
                type="stream_start",
                response_id=interaction.get("id", ""),
                model=interaction.get("model", ""),
            )
        ]

    def _stream_step_start(
        self, data: dict, context: StreamContext | None = None
    ) -> list[IRStreamEvent]:
        index = data.get("index", 0)
        step = data.get("step", {})
        step_type = step.get("type", "")
        block_type = _STEP_TYPE_TO_BLOCK_TYPE.get(step_type, step_type)
        if step_type in SERVER_CALL_TYPES:
            # A server call is a tool block, not a bare step: name it
            # ``server_tool_use`` so the writer defers to the paired
            # ``ToolCallStartEvent`` (which carries the id, kind and
            # signature).  The kind is folded from the step type, never from
            # the step name.
            block_type = "server_tool_use"
        events: list[IRStreamEvent] = [
            ContentBlockStartEvent(
                type="content_block_start",
                block_index=index,
                block_type=block_type,
            )
        ]
        if step_type == "function_call":
            tool_call_id = step.get("id", "")
            if context is not None:
                context.block_index_to_tool_call_id[index] = tool_call_id
            tc_event = ToolCallStartEvent(
                type="tool_call_start",
                tool_call_id=tool_call_id,
                tool_name=step.get("name", ""),
                tool_call_index=index,
            )
            sig = step.get("signature")
            if sig:
                tc_event["provider_metadata"] = {"google": {"thought_signature": sig}}
            events.append(tc_event)
        elif step_type in SERVER_CALL_TYPES:
            kind = SERVER_CALL_TYPES[step_type]
            tool_call_id = step.get("id", "")
            if context is not None:
                context.block_index_to_tool_call_id[index] = tool_call_id
            tc_event = ToolCallStartEvent(
                type="tool_call_start",
                tool_call_id=tool_call_id,
                tool_name=kind,
                tool_call_index=index,
            )
            tc_event["tool_type"] = "intrinsic"
            set_intrinsic_kind(tc_event, kind)
            sig = step.get("signature")
            if sig:
                tc_event["provider_metadata"] = {"google": {"thought_signature": sig}}
            events.append(tc_event)
        elif step_type in SERVER_RESULT_TYPES:
            # The result step's id lives on its ``step.start``; remember it so
            # the payload delta, which may omit it, can still name the call.
            if context is not None:
                context.block_index_to_tool_call_id[index] = step.get("call_id", "")
        # A server result step carries no payload in ``step.start`` — the
        # result rides the following ``step.delta`` — so only the block is
        # opened here; ``_stream_step_delta`` emits the ToolResultEvent.
        return events

    def _stream_step_delta(
        self, data: dict, context: StreamContext | None = None
    ) -> list[IRStreamEvent]:
        index = data.get("index", 0)
        delta = data.get("delta", {})
        delta_type = delta.get("type", "")
        if delta_type == "text":
            return [
                TextDeltaEvent(
                    type="text_delta", text=delta.get("text", ""), block_index=index
                )
            ]
        if delta_type == "thought_summary":
            return [
                ReasoningDeltaEvent(
                    type="reasoning_delta",
                    reasoning=delta.get("text", ""),
                    block_index=index,
                )
            ]
        if delta_type == "thought_signature":
            sig = delta.get("signature", "")
            signature_event = ReasoningDeltaEvent(
                type="reasoning_delta",
                reasoning="",
                signature=sig,
                block_index=index,
            )
            if sig:
                set_opaque_origin(signature_event, signature="google_interactions")
            return [signature_event]
        if delta_type == "arguments":
            return [
                ToolCallDeltaEvent(
                    type="tool_call_delta",
                    tool_call_id=delta.get("call_id", ""),
                    arguments_delta=delta.get("arguments", ""),
                    block_index=index,
                )
            ]
        if delta_type in SERVER_CALL_TYPES:
            # A server call's arguments arrive whole (as a dict) under the
            # step's own type name; serialize them so the IR delta matches the
            # JSON-string shape a client tool call uses.  The step delta
            # carries no call id, so it is read from the id registered for this
            # block at ``step.start``.
            tool_call_id = ""
            if context is not None:
                tool_call_id = context.block_index_to_tool_call_id.get(index, "")
            return [
                ToolCallDeltaEvent(
                    type="tool_call_delta",
                    tool_call_id=tool_call_id,
                    arguments_delta=json.dumps(delta.get("arguments", {})),
                    block_index=index,
                )
            ]
        if delta_type in SERVER_RESULT_TYPES:
            # A server result's payload also rides the step delta.  Emit the
            # one-shot ToolResultEvent here, mirroring the non-streaming
            # reader's payload shape.  The delta may omit the call id, which
            # was registered for this block at ``step.start``.
            tool_call_id = delta.get("call_id", "")
            if not tool_call_id and context is not None:
                tool_call_id = context.block_index_to_tool_call_id.get(index, "")
            result_evt = ToolResultEvent(
                type="tool_result",
                tool_call_id=tool_call_id,
                result=delta.get("result", ""),
                tool_type="intrinsic",
                block_index=index,
            )
            set_intrinsic_kind(result_evt, SERVER_RESULT_TYPES[delta_type])
            if delta.get("is_error"):
                result_evt["is_error"] = True
            return [result_evt]
        return []

    def _stream_step_stop(
        self, data: dict, context: StreamContext | None = None
    ) -> list[IRStreamEvent]:
        index = data.get("index", 0)
        events: list[IRStreamEvent] = [
            ContentBlockEndEvent(type="content_block_end", block_index=index)
        ]
        step_usage = data.get("step_usage") or data.get("usage")
        if step_usage:
            events.append(
                UsageEvent(type="usage", usage=self._build_p_usage_to_ir(step_usage))
            )
        return events

    def _stream_interaction_completed(
        self, data: dict, context: StreamContext | None = None
    ) -> list[IRStreamEvent]:
        interaction = data.get("interaction", {})
        status = interaction.get("status", "completed")
        reason_str = _STATUS_TO_FINISH_REASON.get(status, "stop")
        events: list[IRStreamEvent] = [
            FinishEvent(
                type="finish", finish_reason=cast(FinishReason, {"reason": reason_str})
            )
        ]
        usage = interaction.get("usage")
        if usage:
            events.append(
                UsageEvent(type="usage", usage=self._build_p_usage_to_ir(usage))
            )
        events.append(StreamEndEvent(type="stream_end"))
        return events

    # ── IR → Provider stream dispatch override ──────────────────────

    def stream_response_to_provider(
        self,
        event: IRStreamEvent,
        context: StreamContext | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        ctx = context or StreamContext()
        result = super().stream_response_to_provider(event, ctx)

        # Synthesize step.start before first delta when upstream skipped
        # ContentBlockStartEvent (common for OpenAI Chat upstreams)
        event_type = event.get("type", "")  # type: ignore[union-attr]
        if event_type in ("text_delta", "reasoning_delta", "tool_call_delta"):
            if not ctx.metadata.get("_interactions_step_started"):
                ctx.metadata["_interactions_step_started"] = True
                step_start = {
                    "event_type": "step.start",
                    "index": 0,
                    "step": {"type": "model_output"},
                }
                chunks = [step_start]
                if isinstance(result, list):
                    chunks.extend(result)
                elif result:
                    chunks.append(result)
                return chunks

        # Synthesize step.stop before interaction.completed when missing
        if event_type == "finish" and ctx.metadata.get("_interactions_step_started"):
            if not ctx.metadata.get("_interactions_step_stopped"):
                ctx.metadata["_interactions_step_stopped"] = True
                step_stop = {"event_type": "step.stop", "index": 0}
                chunks = [step_stop]
                if isinstance(result, list):
                    chunks.extend(result)
                elif result:
                    chunks.append(result)
                return chunks

        return result

    # ── IR → Provider stream handlers (for response conversion) ────

    _IR_TO_P_DISPATCH: dict[str, str] = {
        **BaseConverter._IR_TO_P_DISPATCH,
        "tool_result": "_handle_ir_tool_result_to_p",
    }

    def _handle_ir_stream_start_to_p(
        self, event: StreamStartEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        if context is not None:
            context.response_id = event["response_id"]
            context.model = event["model"]
            context.mark_started()
        return {
            "event_type": "interaction.created",
            "interaction": {
                "id": event["response_id"],
                "model": event["model"],
                "status": "in_progress",
            },
        }

    def _handle_ir_stream_end_to_p(
        self, event: StreamEndEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        reason = "stop"
        interaction: dict[str, Any] = {"status": "completed"}
        if context:
            reason = context.metadata.get("_interactions_finish_reason", "stop")
            interaction["id"] = getattr(context, "response_id", "")
            interaction["model"] = getattr(context, "model", "")
            stashed = context.metadata.get("_interactions_usage")
            if stashed:
                interaction["usage"] = stashed
        interaction["status"] = _FINISH_REASON_TO_STATUS.get(reason, "completed")
        return {
            "event_type": "interaction.completed",
            "interaction": interaction,
        }

    def _handle_ir_content_block_start_to_p(
        self, event: ContentBlockStartEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        block_type = event["block_type"]
        # Mark the step as opened so the first delta does not synthesize a
        # second, wrong-index ``step.start`` (see ``stream_response_to_provider``).
        if context is not None:
            context.metadata["_interactions_step_started"] = True
        if block_type in ("tool_use", "server_tool_use"):
            # The enriched ``step.start`` (carrying id/name/signature) is
            # emitted by ``_handle_ir_tool_call_start_to_p``; emitting a bare
            # one here would duplicate it at the same index.  Callers must
            # therefore emit a ``ToolCallStartEvent`` for every tool block —
            # every in-tree converter pairs the two events.
            #
            # Remember the block index so that opener reuses it: the
            # ToolCallStartEvent's own ``tool_call_index`` is a 0-based
            # enumeration of tool calls, not the provider block index, and the
            # deltas carry the block index.
            if context is not None:
                context.metadata["_interactions_pending_tool_index"] = event[
                    "block_index"
                ]
            return {}
        if block_type in SERVER_RESULT_TYPES:
            # The paired ToolResultEvent emits the result step; a bare
            # ``step.start`` here would duplicate it at the same index.
            return {}
        step_type = {v: k for k, v in _STEP_TYPE_TO_BLOCK_TYPE.items()}.get(
            block_type, block_type
        )
        return {
            "event_type": "step.start",
            "index": event["block_index"],
            "step": {"type": step_type},
        }

    def _handle_ir_content_block_end_to_p(
        self, event: ContentBlockEndEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        # Record that a step.stop was emitted so the ``finish`` fallback in
        # ``stream_response_to_provider`` does not synthesize a second one.
        if context is not None:
            context.metadata["_interactions_step_stopped"] = True
        return {
            "event_type": "step.stop",
            "index": event["block_index"],
        }

    def _handle_ir_text_delta_to_p(
        self, event: TextDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        return {
            "event_type": "step.delta",
            "index": event.get("block_index", 0),
            "delta": {"type": "text", "text": event["text"]},
        }

    def _handle_ir_reasoning_delta_to_p(
        self, event: ReasoningDeltaEvent, context: StreamContext | None
    ) -> list[dict[str, Any]]:
        """Handle ReasoningDeltaEvent → thought summary and/or signature deltas.

        The Interactions wire carries the thought summary and its signature as
        separate deltas, so an event holding both yields two.  A foreign
        signature is dropped without taking the summary along with it.
        """
        signature = event.get("signature")
        if signature is not None and not is_own_origin(
            get_opaque_origin(event, "signature"), self._CONVERTER_TAG
        ):
            # Foreign signature — not valid for Google.
            signature = None
        index = event.get("block_index", 0)
        payloads: list[dict[str, Any]] = []
        if event.get("reasoning"):
            payloads.append({"type": "thought_summary", "text": event["reasoning"]})
        if signature:
            payloads.append({"type": "thought_signature", "signature": signature})
        return [
            {"event_type": "step.delta", "index": index, "delta": payload}
            for payload in payloads
        ]

    def _handle_ir_tool_call_start_to_p(
        self, event: ToolCallStartEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        step: dict[str, Any]
        if event.get("tool_type", "function") == "intrinsic":
            # Mirror ``ir_function_call_to_p``: a server call re-emits the
            # provider's own step type and carries no ``name`` (the API rejects
            # one).  The kind comes from the event metadata, never the name.
            kind = get_intrinsic_kind(event, event["tool_name"])
            step_type = INTRINSIC_KIND_TO_CALL_TYPE.get(kind)
            if step_type is None:
                # Unrenderable kind: drop the call rather than fabricate a
                # step type the API does not define (see
                # ``ir_function_call_to_p``).
                return {}
            step = {
                "type": step_type,
                "id": event["tool_call_id"],
            }
        else:
            step = {
                "type": "function_call",
                "id": event["tool_call_id"],
                "name": event["tool_name"],
            }
        pm = event.get("provider_metadata") or {}
        sig = pm.get("google", {}).get("thought_signature")
        if sig:
            step["signature"] = sig
        # Prefer the block index remembered by ``_handle_ir_content_block_start_to_p``
        # so the opener matches the index its deltas carry; fall back to the
        # tool-call enumeration when no block-start preceded it.
        index = event.get("tool_call_index", 0)
        if context is not None:
            # This is a step opener too — record it so the delta fallback in
            # ``stream_response_to_provider`` does not synthesize another one.
            context.metadata["_interactions_step_started"] = True
            pending = context.metadata.pop("_interactions_pending_tool_index", None)
            if pending is not None:
                index = pending
            if event.get("tool_type", "function") == "intrinsic":
                # The step delta of a server call needs the provider step type,
                # which the IR delta event does not carry.
                context.metadata.setdefault("_interactions_intrinsic_steps", {})[
                    index
                ] = step["type"]
        return {
            "event_type": "step.start",
            "index": index,
            "step": step,
        }

    def _handle_ir_tool_call_delta_to_p(
        self, event: ToolCallDeltaEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        index = event.get("block_index", 0)
        step_type = None
        if context is not None:
            step_type = context.metadata.get("_interactions_intrinsic_steps", {}).get(
                index
            )
        if step_type and context is not None:
            # A server call's arguments ride whole, as a dict, under the step's
            # own type name.  The IR carries them as a JSON string, but a
            # cross-format source (e.g. Anthropic ``input_json_delta``) delivers
            # them in fragments, so accumulate until the buffer parses and emit
            # a single delta with an object.  A buffer that never parses is
            # dropped rather than written as a string, so every emitted
            # ``arguments`` is an object.
            buffers = context.metadata.setdefault("_interactions_intrinsic_args", {})
            buffer = buffers.pop(index, "") + (event["arguments_delta"] or "")
            try:
                arguments: Any = json.loads(buffer) if buffer else {}
            except (ValueError, TypeError):
                buffers[index] = buffer
                return {}
            return {
                "event_type": "step.delta",
                "index": index,
                "delta": {"type": step_type, "arguments": arguments},
            }
        return {
            "event_type": "step.delta",
            "index": index,
            "delta": {
                "type": "arguments",
                "call_id": event["tool_call_id"],
                "arguments": event["arguments_delta"],
            },
        }

    def _handle_ir_tool_result_to_p(
        self, event: ToolResultEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        """Handle ToolResultEvent → step.start for a server-tool result.

        A server result is emitted as the provider's own result step, with the
        whole payload on it.  On the wire the provider splits the payload
        across a ``step.start``/``step.delta`` pair, but a consumer accepts the
        completed step in one message, and the paired block-start is dropped by
        ``_handle_ir_content_block_start_to_p`` so nothing is duplicated.
        """
        kind = get_intrinsic_kind(event)
        step_type = INTRINSIC_KIND_TO_RESULT_TYPE.get(kind)
        if step_type is None:
            # Unrenderable kind: drop the result rather than fabricate a step
            # type the API does not define (see ``ir_function_result_to_p``).
            return {}
        step: dict[str, Any] = {
            "type": step_type,
            "call_id": event["tool_call_id"],
            "result": event["result"],
        }
        if event.get("is_error"):
            step["is_error"] = True
        if context is not None:
            context.metadata["_interactions_step_started"] = True
        return {
            "event_type": "step.start",
            "index": event.get("block_index", 0),
            "step": step,
        }

    def _handle_ir_finish_to_p(
        self, event: FinishEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        # Stash finish reason; defer interaction.completed to stream_end
        # so usage (which may arrive after finish) is included
        reason = event["finish_reason"]["reason"]
        if context:
            context.metadata["_interactions_finish_reason"] = reason
        return {}

    def _handle_ir_usage_to_p(
        self, event: UsageEvent, context: StreamContext | None
    ) -> dict[str, Any]:
        usage = self._build_ir_usage_to_p(event["usage"])
        # Stash for injection into interaction.completed
        if context is not None:
            context.metadata["_interactions_usage"] = usage
        return {}
