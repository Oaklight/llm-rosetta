"""Hoist late system messages to preserve prompt cache prefix stability.

When a system/developer message appears mid-conversation (after user or
assistant turns), it breaks the prompt cache prefix for upstreams that
support caching (Anthropic, OpenAI).  The Anthropic converter also drops
non-leading system messages entirely, causing silent data loss.

This helper rewrites the IR request so that:

1. **Leading** system messages (before any non-system message) are moved
   into ``system_instruction``.
2. **Late** system messages (after the first non-system message) are
   rewritten as ``UserMessage`` with ``<system>...</system>`` envelope
   tags wrapping the text content while preserving any non-text parts
   (images, files).

The result is a stable prefix that caching can rely on, and no data loss
regardless of target format.

Attribution: envelope approach adapted from codex-rosetta commits
c749003b and e7e7768e (late developer/system message rewriting).
Deviation: operates at IR level (not raw request body), uses no
product-specific metadata checks, and applies to all source→target
conversions (not just Responses→Chat).
"""

from __future__ import annotations

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)

_SYSTEM_OPEN = "<system>"
_SYSTEM_CLOSE = "</system>"


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


def _wrap_content(content: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Wrap system content with ``<system>...</system>`` envelope tags.

    Text parts have tags prepended/appended to the first and last text
    part respectively.  Non-text parts (images, files) are preserved
    as-is.  If there are no text parts, sentinel text parts are inserted
    at the boundaries so the envelope is always present.
    """
    wrapped = copy.deepcopy(content)

    text_indices = [
        i
        for i, part in enumerate(wrapped)
        if isinstance(part, dict) and isinstance(part.get("text"), str)
    ]

    if not text_indices:
        # No text parts — insert boundary sentinels
        return [
            {"type": "text", "text": _SYSTEM_OPEN},
            *wrapped,
            {"type": "text", "text": _SYSTEM_CLOSE},
        ]

    first = text_indices[0]
    last = text_indices[-1]
    wrapped[first]["text"] = f"{_SYSTEM_OPEN}\n{wrapped[first]['text']}"
    wrapped[last]["text"] = f"{wrapped[last]['text']}\n{_SYSTEM_CLOSE}"
    return wrapped


def _rewrite_as_user(msg: dict[str, Any]) -> dict[str, Any]:
    """Convert a SystemMessage to a UserMessage with ``<system>`` envelope.

    Preserves all content parts (text, image, file) and wraps text with
    ``<system>...</system>`` tags.
    """
    content = msg.get("content", [])

    if isinstance(content, str):
        envelope = (
            f"{_SYSTEM_OPEN}\n{content}\n{_SYSTEM_CLOSE}"
            if content.strip()
            else f"{_SYSTEM_OPEN}\n{_SYSTEM_CLOSE}"
        )
        wrapped_content: list[dict[str, Any]] = [{"type": "text", "text": envelope}]
    elif isinstance(content, list):
        wrapped_content = _wrap_content(content)
    else:
        wrapped_content = [
            {"type": "text", "text": f"{_SYSTEM_OPEN}\n{_SYSTEM_CLOSE}"}
        ]

    result: dict[str, Any] = {
        "role": "user",
        "content": wrapped_content,
    }
    if "metadata" in msg:
        result["metadata"] = msg["metadata"]
    return result


def _build_new_messages(
    messages: list[dict[str, Any]],
    leading_indices: set[int],
    late_indices: set[int],
) -> list[dict[str, Any]]:
    """Build new message list with late system messages rewritten."""
    skip: set[int] = set(leading_indices)
    new_messages: list[dict[str, Any]] = []
    for i, msg in enumerate(messages):
        if i in skip:
            continue
        if i in late_indices:
            new_messages.append(_rewrite_as_user(msg))
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
        if not isinstance(msg, dict) or msg.get("role") != "system":
            first_non_system = i
            break

    leading_indices: set[int] = set()
    late_indices: set[int] = set()
    for i, msg in enumerate(messages):
        if not isinstance(msg, dict) or msg.get("role") != "system":
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
