"""Message operations for the ACP lane.

The lane is a target, not a source: a request is rendered into ACP prompt
blocks and a reply is assembled from the agent's update stream.  The reader
here exists so a body produced by :meth:`AcpConverter.request_to_provider`
can be read back — round-trip tests and any future relay use it — and so the
base class's message contract is satisfied without a stub that raises.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, cast

from ....types.ir.messages import (
    AssistantMessage,
    SystemMessage,
    ToolMessage,
    UserMessage,
)
from ....types.ir.request import IRInputItem
from ...base.messages import BaseMessageOps


class AcpMessageOps(BaseMessageOps):
    """IR messages ↔ this lane's prompt-block list."""

    @staticmethod
    def ir_messages_to_p(
        ir_messages: Sequence[IRInputItem], **kwargs: Any
    ) -> tuple[list[Any], list[str]]:
        from .prompt import render_messages

        return render_messages(cast(list[dict[str, Any]], list(ir_messages))), []

    @staticmethod
    def p_messages_to_ir(
        provider_messages: list[Any], **kwargs: Any
    ) -> list[IRInputItem]:
        """Read the labelled text blocks back into IR messages.

        Only the shape this lane writes is understood: a text block whose
        first line is ``[Role]``.  Anything else becomes a user message, so a
        hand-written prompt still round-trips to something usable.
        """
        out: list[IRInputItem] = []
        for block in provider_messages:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "image":
                image: Any = {
                    "type": "image",
                    "image_data": {
                        "data": str(block.get("data") or ""),
                        "media_type": str(block.get("mimeType") or "image/png"),
                    },
                }
                out.append(cast(IRInputItem, {"role": "user", "content": [image]}))
                continue
            if block.get("type") != "text":
                continue
            text = str(block.get("text") or "")
            label, _, rest = text.partition("\n")
            role, content = _role_of(label, rest)
            part: Any = {"type": "text", "text": content}
            if role == "system":
                out.append(SystemMessage(role="system", content=[part]))
            elif role == "assistant":
                out.append(AssistantMessage(role="assistant", content=[part]))
            elif role == "tool":
                out.append(ToolMessage(role="tool", content=[part]))
            else:
                out.append(UserMessage(role="user", content=[part]))
        return out


def _role_of(label: str, rest: str) -> tuple[str, str]:
    """Return ``(role, text)`` for a rendered block's first line."""
    if not label.startswith("[") or not label.endswith("]"):
        return "user", (label + ("\n" + rest if rest else ""))
    word = label[1:-1].strip()
    return (
        {
            "System": "system",
            "Assistant": "assistant",
            "Tool result": "tool",
            "User": "user",
        }.get(word, "user"),
        rest,
    )


__all__ = ["AcpMessageOps"]
