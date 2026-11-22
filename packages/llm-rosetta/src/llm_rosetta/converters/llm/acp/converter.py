"""The ACP lane: IR ↔ Agent Client Protocol.

ACP is not a model API and not an HTTP body.  A *client* opens a session
with an agent process and sends prompt content blocks; the agent answers with
``session/update`` notifications (message chunks, thought chunks, tool
calls).  This converter owns the IR half of that exchange:

* a request becomes the lane's turn body — the system instruction, the
  conversation as prompt blocks, and the model to ask for;
* the transport (which owns the process) hands back the updates it collected
  and a stop reason, which become an :class:`IRResponse`;
* on the stream leg, one transport chunk is one IR event: text and thought
  deltas, then a finish.

The lane is a **target only**.  No client speaks ACP to the gateway, so
``response_to_provider`` and ``stream_response_to_provider`` are deliberately
unimplemented rather than approximated: a half-written ACP server would look
like a supported source format and silently lose fields.

Tool declarations travel over MCP, not in the turn body: the converter writes
them in ``tools/list`` shape and the transport serves them from a loopback MCP
endpoint.  An agent's call on one of those tools comes back as a tool call,
and the client's result on a later request is handed to the same pending MCP
call.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from typing import Any, cast

from llm_rosetta.conversion_warnings import DROPPED, FAILED, report_loss

from ....types.ir.request import IRRequest
from ....types.ir.response import FinishReason, IRResponse, UsageInfo
from ....types.ir.stream import (
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
from ...base import BaseConverter
from ...base.context import ConversionContext, StreamContext
from .content_ops import AcpContentOps
from .message_ops import AcpMessageOps
from .prompt import render_messages, render_system
from .tool_ops import AcpToolOps

#: ACP stop reasons → IR finish reasons.  The ACP values are the protocol's
#: own words; a value this lane has never seen becomes ``stop``, because the
#: agent ended the turn and the words that arrived are the answer.
_STOP_TO_FINISH: dict[str, str] = {
    "end_turn": "stop",
    "max_tokens": "length",
    "max_turn_requests": "length",
    "refusal": "content_filter",
    "cancelled": "cancelled",
    # The transport's own stop for a turn suspended on a client tool call.
    "tool_calls": "tool_calls",
}


def tool_results_of(messages: Sequence[Any]) -> list[dict[str, Any]]:
    """Collect the tool results an IR history carries, in order.

    The transport hands these to the pending MCP calls of a suspended turn.
    Only the id, result and error bit travel; the rest of the message is
    already in the rendered transcript.
    """
    results: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        for part in message.get("content") or []:
            if not isinstance(part, dict) or part.get("type") != "tool_result":
                continue
            call_id = part.get("tool_call_id")
            if not isinstance(call_id, str) or not call_id:
                continue
            results.append(
                {
                    "tool_call_id": call_id,
                    "result": part.get("result", ""),
                    "is_error": bool(part.get("is_error")),
                }
            )
    return results


def target_finish_reason(stop_reason: str) -> FinishReason:
    """Map an ACP stop reason to the IR's finish reason."""
    reason = _STOP_TO_FINISH.get(stop_reason, "stop")
    return FinishReason(reason=cast(Any, reason))


class AcpConverter(BaseConverter):
    """Intermediate Representation ↔ an ACP turn."""

    content_ops_class = AcpContentOps
    tool_ops_class = AcpToolOps
    message_ops_class = AcpMessageOps

    _CONVERTER_TAG = "acp"
    _PASSTHROUGH_RESTORE_KEY = "acp_passthrough_items"
    # The lane has no syntax for a hosted tool, so a client function whose
    # name matches an intrinsic kind must not be promoted into one.
    _SUPPORTS_NATIVE_SERVER_TOOLS = False
    _SUPPORTS_MULTIMODAL_TOOL_RESULT = False

    def __init__(self) -> None:
        """Build the per-instance operations objects."""
        self.content_ops = self.content_ops_class()
        self.tool_ops = self.tool_ops_class()
        self.message_ops = self.message_ops_class()

    # ==================== Request ====================

    def _do_request_to_provider(
        self,
        ir_request: IRRequest,
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Render the IR request as this lane's turn body.

        Returns:
            ``{"model", "system", "prompt", "tools", "tool_results"}``: the
            model to ask for (empty means the session's own default), the
            system instruction as one string, the conversation as ACP content
            blocks, the declared tools in MCP shape, and the tool results the
            next turn must hand back to a pending MCP call.
        """
        messages = list(ir_request.get("messages") or [])
        result: dict[str, Any] = {
            "model": ir_request.get("model", ""),
            "system": render_system(ir_request.get("system_instruction")),
            "prompt": render_messages(
                cast(list[dict[str, Any]], messages), warnings=context.warnings
            ),
            "tool_results": tool_results_of(messages),
        }
        self._apply_tool_config(ir_request, result, context)
        self._report_uncarried_config(ir_request, context)
        return result

    def _report_uncarried_config(
        self, ir_request: IRRequest, ctx: ConversionContext
    ) -> None:
        """Report the generation settings an ACP turn cannot carry.

        The turn body is a prompt and a model: ACP has no field for sampling
        parameters, and the agent's own session options are not a place to
        write them from here.  Dropping them silently would tell the caller
        nothing, so each one present is reported the way a dropped tool is.
        """
        config = ir_request.get("generation") or {}
        if not isinstance(config, dict):
            return
        for key in sorted(config):
            if config.get(key) is None:
                continue
            report_loss(
                "ACP turns carry no sampling parameters; the setting was dropped.",
                code=DROPPED,
                subject=f"generation.{key}",
                warnings=ctx.warnings,
            )

    def _do_request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Read this lane's turn body back into IR.

        Only the shape :meth:`_do_request_to_provider` writes is understood;
        it exists so a converted body can be inspected and round-tripped.
        """
        ir_request: dict[str, Any] = {
            "model": provider_request.get("model", ""),
            "messages": self.message_ops.p_messages_to_ir(
                list(provider_request.get("prompt") or [])
            ),
        }
        system = provider_request.get("system")
        if isinstance(system, str) and system:
            ir_request["system_instruction"] = [{"type": "text", "text": system}]
        tools = provider_request.get("tools")
        if isinstance(tools, list):
            ir_request["tools"] = [
                tool
                for tool in (
                    self.tool_ops.p_tool_definition_to_ir(item) for item in tools
                )
                if tool is not None
            ]
        return ir_request

    def _apply_tool_config(
        self,
        ir_request: IRRequest,
        result: dict[str, Any],
        ctx: ConversionContext,
    ) -> None:
        """Write the declared tools and report the settings ACP cannot carry."""
        kwargs = {"warnings": ctx.warnings}
        tools: list[dict[str, Any]] = []
        for tool in ir_request.get("tools") or []:
            written = self.tool_ops.ir_tool_definition_to_p(tool, **kwargs)
            if isinstance(written, dict) and written.get("name"):
                tools.append(written)
        result["tools"] = tools
        if ir_request.get("tool_choice") is not None:
            self.tool_ops.ir_tool_choice_to_p(ir_request["tool_choice"], **kwargs)
        if ir_request.get("tool_config") is not None:
            self.tool_ops.ir_tool_config_to_p(ir_request["tool_config"], **kwargs)

    # ==================== Response ====================

    def _do_response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Assemble an IR response from the transport's collected updates."""
        updates = provider_response.get("updates")
        if not isinstance(updates, list):
            updates = []
        text: list[str] = []
        reasoning: list[str] = []
        calls: list[dict[str, Any]] = []
        for update in updates:
            if not isinstance(update, dict):
                continue
            if update.get("type") == "text":
                text.append(str(update.get("text") or ""))
            elif update.get("type") == "thought":
                reasoning.append(str(update.get("text") or ""))
            elif update.get("type") == "tool_call":
                call = self.tool_ops.p_tool_call_to_ir(update)
                if call.get("tool_name"):
                    calls.append(dict(call))
        content: list[dict[str, Any]] = []
        joined_reasoning = "".join(reasoning)
        if joined_reasoning:
            content.append({"type": "reasoning", "reasoning": joined_reasoning})
        joined_text = "".join(text)
        if joined_text or not calls:
            content.append({"type": "text", "text": joined_text})
        content.extend(calls)
        finish = target_finish_reason(
            "tool_calls"
            if calls
            else str(provider_response.get("stop_reason") or "end_turn")
        )
        choice: Any = {
            "index": 0,
            "message": {"role": "assistant", "content": content},
            "finish_reason": finish,
        }
        ir_response: dict[str, Any] = {
            "id": str(provider_response.get("session_id") or ""),
            "object": "response",
            "created": int(provider_response.get("created") or time.time()),
            "model": str(provider_response.get("model") or ""),
            "choices": [choice],
        }
        return ir_response

    def _do_response_to_provider(
        self,
        ir_response: IRResponse,
        *,
        context: ConversionContext,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Refuse: ACP is a target-only lane."""
        raise NotImplementedError(
            "the acp format is an upstream lane; it is not a client source format"
        )

    # ==================== Stream ====================

    def stream_response_from_provider(
        self,
        chunk: dict[str, Any] | None,
        context: StreamContext | None = None,
    ) -> list[IRStreamEvent]:
        """Turn one transport chunk into IR stream events.

        Args:
            chunk: A chunk from the ACP transport: ``session_start``,
                ``text``, ``thought``, ``usage``, ``stop``, ``error``, or
                ``None`` for end-of-stream.
            context: Tracks start/end so exactly one stream end is emitted.

        Returns:
            The IR events this chunk stands for.
        """
        if chunk is None:
            return self._stream_end(context)
        kind = chunk.get("type")
        if kind == "session_start":
            return self._stream_start(chunk, context)
        if kind == "text":
            return self._text_events(chunk)
        if kind == "thought":
            return self._thought_events(chunk)
        if kind == "usage":
            return self._usage_events(chunk)
        if kind == "tool_call":
            return self._tool_call_events(chunk)
        if kind == "stop":
            return [
                FinishEvent(
                    type="finish",
                    finish_reason=target_finish_reason(
                        str(chunk.get("stop_reason") or "end_turn")
                    ),
                )
            ]
        if kind == "error":
            return self._error_events(chunk, context)
        return []

    @staticmethod
    def _stream_end(context: StreamContext | None) -> list[IRStreamEvent]:
        if context is not None:
            if context.is_ended:
                return []
            context.mark_ended()
        return [StreamEndEvent(type="stream_end")]

    @staticmethod
    def _stream_start(
        chunk: dict[str, Any], context: StreamContext | None
    ) -> list[IRStreamEvent]:
        if context is not None and context.is_started:
            # One turn, one session: a repeated start is the transport
            # restating what the stream already announced.
            return []
        response_id = str(chunk.get("session_id") or "")
        model = str(chunk.get("model") or "")
        if context is not None:
            context.response_id = response_id
            context.model = model
            context.mark_started()
        return [
            StreamStartEvent(
                type="stream_start",
                response_id=response_id,
                model=model,
            )
        ]

    @staticmethod
    def _text_events(chunk: dict[str, Any]) -> list[IRStreamEvent]:
        text = str(chunk.get("text") or "")
        if not text:
            return []
        return [TextDeltaEvent(type="text_delta", text=text)]

    @staticmethod
    def _thought_events(chunk: dict[str, Any]) -> list[IRStreamEvent]:
        text = str(chunk.get("text") or "")
        if not text:
            return []
        return [ReasoningDeltaEvent(type="reasoning_delta", reasoning=text)]

    @staticmethod
    def _usage_events(chunk: dict[str, Any]) -> list[IRStreamEvent]:
        # ACP's usage update reports the session's context usage, not the
        # token bill for one turn, so it is carried as an input-side figure
        # and never as an output count.
        used = chunk.get("used")
        if not isinstance(used, int) or used < 0:
            return []
        return [
            UsageEvent(
                type="usage",
                usage=UsageInfo(
                    prompt_tokens=used,
                    completion_tokens=0,
                    total_tokens=used,
                ),
            )
        ]

    def _tool_call_events(self, chunk: dict[str, Any]) -> list[IRStreamEvent]:
        call = self.tool_ops.p_tool_call_to_ir(chunk)
        call_id = str(call.get("tool_call_id") or "")
        name = str(call.get("tool_name") or "")
        if not call_id or not name:
            return []
        return [
            ContentBlockStartEvent(
                type="content_block_start",
                block_index=0,
                block_type="tool_use",
            ),
            ToolCallStartEvent(
                type="tool_call_start",
                tool_call_id=call_id,
                tool_name=name,
                tool_type="function",
            ),
            ToolCallDeltaEvent(
                type="tool_call_delta",
                tool_call_id=call_id,
                arguments_delta=json.dumps(
                    call.get("tool_input") or {},
                    ensure_ascii=False,
                    sort_keys=True,
                ),
            ),
        ]

    @staticmethod
    def _error_events(
        chunk: dict[str, Any], context: StreamContext | None
    ) -> list[IRStreamEvent]:
        report_loss(
            f"ACP agent error: {chunk.get('message') or 'unknown failure'}",
            code=FAILED,
            subject="acp.turn",
            warnings=getattr(context, "warnings", None),
        )
        return [
            FinishEvent(
                type="finish",
                finish_reason=FinishReason(reason="error"),  # type: ignore[typeddict-item]
            )
        ]

    def stream_response_to_provider(
        self,
        event: IRStreamEvent,
        context: StreamContext | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Refuse: ACP is a target-only lane."""
        raise NotImplementedError(
            "the acp format is an upstream lane; it is not a client source format"
        )

    # ==================== Usage ====================

    @staticmethod
    def _build_p_usage_to_ir(p_usage: dict[str, Any]) -> UsageInfo:
        """Read a usage payload in this lane's own naming."""
        prompt = int(p_usage.get("prompt_tokens") or p_usage.get("used") or 0)
        completion = int(p_usage.get("completion_tokens") or 0)
        return UsageInfo(
            prompt_tokens=prompt,
            completion_tokens=completion,
            total_tokens=int(p_usage.get("total_tokens") or prompt + completion),
        )

    @staticmethod
    def _build_ir_usage_to_p(ir_usage: Mapping[str, Any]) -> dict[str, Any]:
        """Write a usage payload in this lane's own naming."""
        prompt = int(ir_usage.get("prompt_tokens") or 0)
        completion = int(ir_usage.get("completion_tokens") or 0)
        return {
            "prompt_tokens": prompt,
            "completion_tokens": completion,
            "total_tokens": int(ir_usage.get("total_tokens") or prompt + completion),
        }


__all__ = ["AcpConverter", "target_finish_reason"]
