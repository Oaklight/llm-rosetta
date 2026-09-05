"""
LLM-Rosetta - Google Interactions Converter

Top-level converter implementing bidirectional conversion between IR
and Google Interactions API format via BaseConverter hooks.
"""

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
    UsageEvent,
)
from ..base import BaseConverter
from ..base.context import ConversionContext, StreamContext
from .config_ops import GoogleInteractionsConfigOps
from .content_ops import GoogleInteractionsContentOps
from .message_ops import GoogleInteractionsMessageOps
from .tool_ops import GoogleInteractionsToolOps

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
            result["tools"] = [self.tool_ops.ir_tool_to_p(t) for t in tools]

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
            if msg.get("role") == "assistant":
                combined_parts.extend(msg.get("content", []))

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
        return handler(data) if handler else []

    def _stream_interaction_created(self, data: dict) -> list[IRStreamEvent]:
        interaction = data.get("interaction", {})
        return [
            StreamStartEvent(
                type="stream_start",
                response_id=interaction.get("id", ""),
                model=interaction.get("model", ""),
            )
        ]

    def _stream_step_start(self, data: dict) -> list[IRStreamEvent]:
        index = data.get("index", 0)
        step = data.get("step", {})
        step_type = step.get("type", "")
        block_type = _STEP_TYPE_TO_BLOCK_TYPE.get(step_type, step_type)
        events: list[IRStreamEvent] = [
            ContentBlockStartEvent(
                type="content_block_start",
                block_index=index,
                block_type=block_type,
            )
        ]
        if step_type == "function_call":
            events.append(
                ToolCallStartEvent(
                    type="tool_call_start",
                    tool_call_id=step.get("id", ""),
                    tool_name=step.get("name", ""),
                    tool_call_index=index,
                )
            )
        return events

    def _stream_step_delta(self, data: dict) -> list[IRStreamEvent]:
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
            return [
                ReasoningDeltaEvent(
                    type="reasoning_delta",
                    reasoning="",
                    signature=delta.get("signature", ""),
                    block_index=index,
                )
            ]
        if delta_type == "arguments":
            return [
                ToolCallDeltaEvent(
                    type="tool_call_delta",
                    tool_call_id=delta.get("call_id", ""),
                    arguments_delta=delta.get("arguments", ""),
                    block_index=index,
                )
            ]
        return []

    def _stream_step_stop(self, data: dict) -> list[IRStreamEvent]:
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

    def _stream_interaction_completed(self, data: dict) -> list[IRStreamEvent]:
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
