"""
LLM-Rosetta - Google Interactions Message Operations

Bidirectional conversion between Interactions API Steps
and IR Messages. Handles step merging (consecutive assistant-role steps).
"""

from collections.abc import Sequence
from typing import Any, cast

from ...types.ir import (
    AssistantMessage,
    Message,
    ToolMessage,
)
from ...types.ir.parts import AssistantContentPart, ContentPart
from ...types.ir.request import IRInputItem
from ..base import BaseMessageOps
from .content_ops import GoogleInteractionsContentOps
from .tool_ops import GoogleInteractionsToolOps


class GoogleInteractionsMessageOps(BaseMessageOps):
    """Google Interactions API message conversion operations."""

    def __init__(self, content_ops=None, tool_ops=None):
        self.content_ops = content_ops or GoogleInteractionsContentOps()
        self.tool_ops = tool_ops or GoogleInteractionsToolOps()

    # ── Steps → IR Messages ────────────────────────────────────────

    def p_steps_to_ir_messages(self, steps: list[dict]) -> list[Message]:
        """Convert Interactions steps to IR messages.

        Consecutive assistant-role steps (thought, model_output, function_call)
        are merged into a single AssistantMessage.
        """
        messages: list[Message] = []
        assistant_parts: list[AssistantContentPart] = []

        def _flush_assistant():
            nonlocal assistant_parts
            if assistant_parts:
                msg: AssistantMessage = {
                    "role": "assistant",
                    "content": list(assistant_parts),
                }
                messages.append(msg)
                assistant_parts = []

        for step in steps:
            step_type = step.get("type")

            if step_type == "user_input":
                _flush_assistant()
                parts = self._p_content_list_to_parts(step.get("content", []))
                if parts:
                    messages.append(cast(Message, {"role": "user", "content": parts}))

            elif step_type == "model_output":
                assistant_parts.extend(
                    self._p_content_list_to_assistant_parts(step.get("content", []))
                )

            elif step_type == "thought":
                reasoning = self.content_ops.p_thought_to_ir(step)
                assistant_parts.append(reasoning)

            elif step_type == "function_call":
                tc = self.tool_ops.p_function_call_to_ir(step)
                assistant_parts.append(tc)

            elif step_type == "function_result":
                _flush_assistant()
                tr = self.tool_ops.p_function_result_to_ir(step)
                msg_t: ToolMessage = {"role": "tool", "content": [tr]}
                messages.append(msg_t)

        _flush_assistant()
        return messages

    def _p_content_list_to_parts(self, content_list: list) -> list[ContentPart]:
        parts: list[ContentPart] = []
        for c in content_list:
            ir_part = self.content_ops.p_content_to_ir(c)
            if isinstance(ir_part, list):
                parts.extend(ir_part)
            elif ir_part:
                parts.append(ir_part)
        return parts

    def _p_content_list_to_assistant_parts(
        self, content_list: list
    ) -> list[AssistantContentPart]:
        parts: list[AssistantContentPart] = []
        for c in content_list:
            ir_part = self.content_ops.p_content_to_ir(c)
            if isinstance(ir_part, list):
                parts.extend(cast(AssistantContentPart, p) for p in ir_part)
            elif ir_part:
                parts.append(cast(AssistantContentPart, ir_part))
        return parts

    # ── IR Messages → Steps ────────────────────────────────────────

    def ir_messages_to_p_steps(self, messages: list | Sequence) -> list[dict]:
        """Convert IR messages to Interactions steps."""
        steps: list[dict] = []
        for msg in messages:
            if not hasattr(msg, "get"):
                continue
            role = msg.get("role")
            content = msg.get("content", [])

            if role == "user":
                steps.append(self._ir_user_to_step(content))
            elif role == "assistant":
                steps.extend(self._ir_assistant_to_steps(content))
            elif role == "tool":
                steps.extend(self._ir_tool_to_steps(content))

        return steps

    def _ir_user_to_step(self, content: list) -> dict:
        p_content = []
        for part in content:
            ptype = part.get("type")
            if ptype == "text":
                p_content.append(self.content_ops.ir_text_to_p(part))
            elif ptype == "image":
                p_content.append(self.content_ops.ir_image_to_p(part))
        return {"type": "user_input", "content": p_content}

    def _ir_assistant_to_steps(self, content: list) -> list[dict]:
        steps: list[dict] = []
        for part in content:
            ptype = part.get("type")
            if ptype == "text":
                steps.append(
                    {
                        "type": "model_output",
                        "content": [self.content_ops.ir_text_to_p(part)],
                    }
                )
            elif ptype == "reasoning":
                steps.append(self.content_ops.ir_reasoning_to_p(part))
            elif ptype == "tool_call":
                steps.append(self.tool_ops.ir_tool_call_to_p(part))
        return steps

    def _ir_tool_to_steps(self, content: list) -> list[dict]:
        return [
            self.tool_ops.ir_tool_result_to_p(part)
            for part in content
            if part.get("type") == "tool_result"
        ]

    # ── Base class abstract method stubs ───────────────────────────

    @staticmethod
    def ir_messages_to_p(
        ir_messages: Sequence[IRInputItem], **kwargs: Any
    ) -> tuple[list[Any], list[str]]:
        ops = GoogleInteractionsMessageOps()
        steps = ops.ir_messages_to_p_steps(list(ir_messages))
        return steps, []

    @staticmethod
    def p_messages_to_ir(
        provider_messages: list[Any], **kwargs: Any
    ) -> list[IRInputItem]:
        ops = GoogleInteractionsMessageOps()
        return cast(list[IRInputItem], ops.p_steps_to_ir_messages(provider_messages))

    def ir_message_to_p(
        self, ir_message: IRInputItem, **kwargs: Any
    ) -> tuple[Any, list[str]]:
        steps = self.ir_messages_to_p_steps([ir_message])
        return steps, []

    def p_message_to_ir(
        self, provider_message: Any, **kwargs: Any
    ) -> IRInputItem | None:
        msgs = self.p_steps_to_ir_messages([provider_message])
        return msgs[0] if msgs else None
