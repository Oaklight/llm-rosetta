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

Tool declarations are refused by :class:`AcpToolOps` (no bridge yet), so the
request body never carries a tool list.  That is a visible loss, reported on
the conversion warnings.
"""

from __future__ import annotations

import time
from collections.abc import Mapping
from typing import Any, cast

from llm_rosetta.conversion_warnings import FAILED, report_loss

from ....types.ir.request import IRRequest
from ....types.ir.response import FinishReason, IRResponse, UsageInfo
from ....types.ir.stream import (
    FinishEvent,
    IRStreamEvent,
    ReasoningDeltaEvent,
    StreamEndEvent,
    StreamStartEvent,
    TextDeltaEvent,
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
}


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
            ``{"model", "system", "prompt"}``: the model to ask for (empty
            means the session's own default), the system instruction as one
            string, and the conversation as ACP content blocks.
        """
        self._apply_tool_config(ir_request, {}, context)
        messages = list(ir_request.get("messages") or [])
        return {
            "model": ir_request.get("model", ""),
            "system": render_system(ir_request.get("system_instruction")),
            "prompt": render_messages(cast(list[dict[str, Any]], messages)),
        }

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
        return ir_request

    def _apply_tool_config(
        self,
        ir_request: IRRequest,
        result: dict[str, Any],
        ctx: ConversionContext,
    ) -> None:
        """Report every tool declaration this lane cannot carry."""
        kwargs = {"warnings": ctx.warnings}
        for tool in ir_request.get("tools") or []:
            self.tool_ops.ir_tool_definition_to_p(tool, **kwargs)
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
        for update in updates:
            if not isinstance(update, dict):
                continue
            if update.get("type") == "text":
                text.append(str(update.get("text") or ""))
            elif update.get("type") == "thought":
                reasoning.append(str(update.get("text") or ""))
        content: list[dict[str, Any]] = []
        joined_reasoning = "".join(reasoning)
        if joined_reasoning:
            content.append({"type": "reasoning", "reasoning": joined_reasoning})
        content.append({"type": "text", "text": "".join(text)})
        finish = target_finish_reason(
            str(provider_response.get("stop_reason") or "end_turn")
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
            if context is not None:
                if context.is_ended:
                    return []
                context.mark_ended()
            return [StreamEndEvent(type="stream_end")]

        kind = chunk.get("type")
        events: list[IRStreamEvent] = []
        if kind == "session_start":
            if context is not None and context.is_started:
                # One turn, one session: a repeated start is the transport
                # restating what the stream already announced.
                return []
            if context is not None:
                context.response_id = str(chunk.get("session_id") or "")
                context.model = str(chunk.get("model") or "")
                context.mark_started()
            events.append(
                StreamStartEvent(
                    type="stream_start",
                    response_id=str(chunk.get("session_id") or ""),
                    model=str(chunk.get("model") or ""),
                )
            )
        elif kind == "text":
            text = str(chunk.get("text") or "")
            if text:
                events.append(TextDeltaEvent(type="text_delta", text=text))
        elif kind == "thought":
            text = str(chunk.get("text") or "")
            if text:
                events.append(
                    ReasoningDeltaEvent(type="reasoning_delta", reasoning=text)
                )
        elif kind == "usage":
            # ACP's usage update reports the session's context usage, not the
            # token bill for one turn, so it is carried as an input-side
            # figure and never as an output count.
            used = chunk.get("used")
            if isinstance(used, int) and used >= 0:
                events.append(
                    UsageEvent(
                        type="usage",
                        usage=UsageInfo(
                            prompt_tokens=used,
                            completion_tokens=0,
                            total_tokens=used,
                        ),
                    )
                )
        elif kind == "stop":
            events.append(
                FinishEvent(
                    type="finish",
                    finish_reason=target_finish_reason(
                        str(chunk.get("stop_reason") or "end_turn")
                    ),
                )
            )
        elif kind == "error":
            report_loss(
                f"ACP agent error: {chunk.get('message') or 'unknown failure'}",
                code=FAILED,
                subject="acp.turn",
                warnings=getattr(context, "warnings", None),
            )
            events.append(
                FinishEvent(
                    type="finish",
                    finish_reason=FinishReason(reason="error"),  # type: ignore[typeddict-item]
                )
            )
        return events

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
