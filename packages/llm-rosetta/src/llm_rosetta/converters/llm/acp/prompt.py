"""Rendering IR turns as ACP prompt blocks.

ACP prompts are a list of content blocks carried on a user turn.  A chat
request, however, is a whole conversation: the client resends its history
every time, and the agent's session may or may not already hold it.  So the
conversion produces the *whole* conversation as blocks and the transport
decides what to send — everything when it opens a session, only the new tail
when the session already holds the rest.

The rendering is deliberately plain: one text block per message, labelled
with its role, images alongside the text of the message they belong to.  Any
formatting this adds is a fact the model sees, so it is kept to what a reader
needs to tell the speakers apart.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .content_ops import AcpContentOps

_LABELS = {
    "system": "System",
    "user": "User",
    "assistant": "Assistant",
    "tool": "Tool result",
}


def _text_blocks(message: dict[str, Any], warnings: Any = None) -> list[dict[str, Any]]:
    """Return the message's own content as ACP blocks, unlabelled.

    An image goes through :meth:`AcpContentOps.ir_image_to_p`, so one that
    cannot ride inline (a URL) is reported to *warnings*, not dropped quietly.
    """
    blocks: list[dict[str, Any]] = []
    for part in message.get("content") or []:
        if not isinstance(part, dict):
            continue
        kind = part.get("type")
        if kind == "text":
            text = part.get("text")
            if isinstance(text, str) and text:
                blocks.append({"type": "text", "text": text})
        elif kind == "image":
            block = AcpContentOps.ir_image_to_p(part, warnings=warnings)  # type: ignore[arg-type]
            if block is not None:
                blocks.append(block)
        elif kind == "tool_call":
            # A replayed call is history the model wrote.  Rendering it as text
            # keeps the transcript readable; the calls themselves are not
            # executable here (no tool bridge yet).
            name = part.get("tool_name") or ""
            blocks.append({"type": "text", "text": f"[called {name}]"})
        elif kind == "tool_result":
            result = part.get("result")
            text = result if isinstance(result, str) else str(result)
            if text:
                blocks.append({"type": "text", "text": text})
    return blocks


def render_messages(
    messages: Sequence[Any], warnings: Any = None
) -> list[dict[str, Any]]:
    """Render IR messages as the ACP content blocks of one prompt turn.

    Args:
        messages: The IR messages, in order.  Typed message shapes are read
            as mappings; anything else is skipped.
        warnings: Sink for the losses the rendering records (a dropped image).

    Returns:
        ACP content blocks.  Adjacent messages of the same role are rendered
        separately, because the label is what tells them apart and merging
        them would not save a block worth the risk of blurring two turns.
    """
    blocks: list[dict[str, Any]] = []
    for message in messages:
        if not isinstance(message, dict):
            continue
        role = str(message.get("role") or "")
        label = _LABELS.get(role, role or "Message")
        own = _text_blocks(message, warnings)
        lead = [b for b in own if b.get("type") == "text"]
        if lead:
            text = "\n".join(b["text"] for b in lead)
            blocks.append({"type": "text", "text": f"[{label}]\n{text}"})
        blocks.extend(b for b in own if b.get("type") == "image")
    return blocks


def render_system(system_instruction: Sequence[Any] | None) -> str:
    """Return the system instruction as one string, or ``""``."""
    if not system_instruction:
        return ""
    return "\n".join(
        part.get("text", "")
        for part in system_instruction
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ).strip()


__all__ = ["render_messages", "render_system"]
