"""Hoist late system messages to preserve prompt cache prefix stability.

When a system/developer message appears mid-conversation (after user or
assistant turns), it breaks the prompt cache prefix for upstreams that
support caching (Anthropic, OpenAI).  The Anthropic converter also drops
non-leading system messages entirely, causing silent data loss.

This helper rewrites the IR request so that:

1. **Leading** system messages (before any non-system message) are moved
   into ``system_instruction``.
2. **Late** system messages (after the first non-system message) are
   rewritten as ``UserMessage`` with a ``[System: ...]`` envelope.

The result is a stable prefix that caching can rely on, and no data loss
regardless of target format.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def _extract_system_text(msg: dict[str, Any]) -> str:
    """Join all text parts of a system message into a single string."""
    content = msg.get("content", [])
    if isinstance(content, str):
        return content
    parts = []
    for part in content:
        if isinstance(part, dict) and part.get("type") == "text":
            parts.append(part.get("text", ""))
    return "\n".join(parts)


def _rewrite_as_user(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert a SystemMessage to a UserMessage with envelope."""
    text = _extract_system_text(msg)
    envelope = f"<system>\n{text}\n</system>" if text.strip() else "<system>\n[system instruction]\n</system>"
    result: dict[str, Any] = {
        "role": "user",
        "content": [{"type": "text", "text": envelope}],
    }
    if "metadata" in msg:
        result["metadata"] = msg["metadata"]
    return result


def _build_new_messages(
    messages: list[dict[str, Any]],
    leading_indices: set[int],
    late_indices: set[int],
) -> list[dict[str, Any]]:
    """Build new message list with late system messages rewritten.

    When a late system message is immediately followed by a user message,
    the envelope is merged into that user message to avoid consecutive
    user roles (which Anthropic rejects).
    """
    skip: set[int] = set(leading_indices)
    new_messages: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if i in skip:
            continue
        if i in late_indices:
            rewritten = _rewrite_as_user(msg)
            next_idx = i + 1
            while next_idx in skip:
                next_idx += 1
            if (
                next_idx < len(messages)
                and isinstance(messages[next_idx], dict)
                and messages[next_idx].get("role") == "user"
            ):
                next_msg = messages[next_idx]
                merged = {
                    **next_msg,
                    "content": rewritten["content"] + list(next_msg.get("content", [])),
                }
                new_messages.append(merged)
                skip.add(next_idx)
            else:
                new_messages.append(rewritten)
        else:
            new_messages.append(msg)
    return new_messages


def hoist_late_system_messages_ir(
    ir_request: dict[str, Any],
    *,
    request_id: str = "-",
) -> dict[str, Any]:
    """Hoist late system messages to preserve prompt cache prefix.

    Args:
        ir_request: The IR request dict (not mutated).
        request_id: For logging.

    Returns:
        A new IR request with system messages hoisted, or the original
        if no changes were needed.
    """
    messages = ir_request.get("messages", [])
    if not messages:
        return ir_request

    first_non_system = len(messages)
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") not in ("system", "developer"):
            first_non_system = i
            break

    leading_indices: set[int] = set()
    late_indices: set[int] = set()
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") not in ("system", "developer"):
            continue
        if i < first_non_system:
            leading_indices.add(i)
        else:
            late_indices.add(i)

    if not leading_indices and not late_indices:
        return ir_request

    existing_si = list(ir_request.get("system_instruction") or [])
    for idx in sorted(leading_indices):
        text = _extract_system_text(messages[idx])
        if text.strip():
            existing_si.append({"type": "text", "text": text})

    new_messages = _build_new_messages(messages, leading_indices, late_indices)

    changed = len(leading_indices) + len(late_indices)
    logger.debug(
        "[%s] hoisted %d system message(s): %d leading, %d late",
        request_id,
        changed,
        len(leading_indices),
        len(late_indices),
    )

    result = {**ir_request, "messages": new_messages}
    if existing_si:
        result["system_instruction"] = existing_si
    return result
