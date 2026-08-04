"""Multimodal tool result dual-encoding helpers.

When a provider format does not support multimodal content in tool result
messages (e.g. OpenAI Chat Completions ``role: "tool"`` only accepts text),
these helpers implement a dual-encoding strategy:

**Packing (IR → provider)**: multimodal blocks (images, files) are extracted
from the tool result and stored in a ``multimodal_packs`` dict keyed by
``call_id``.  After all messages are converted, ``inject_packed_tool_content``
inserts a synthetic ``role: "user"`` message containing the visual content
wrapped in ``<tool-content call-id="...">`` XML tags.  The original tool
message keeps ``json.dumps(result)`` as a fallback.

**Unpacking (provider → IR)**: ``unpack_tool_content`` detects synthetic user
messages, parses the XML tags, and extracts ``call_id → content blocks``
mappings so the consuming converter can reconstruct the multimodal IR result.

Originally implemented in ``openai_chat/message_ops.py`` (PR #108).  Extracted
here so any text-only converter can reuse the protocol.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..content import BaseContentOps

TOOL_CONTENT_OPEN_TAG_RE = re.compile(r'^<tool-content\s+call-id="([^"]+)">$')
TOOL_CONTENT_CLOSE_TAG = "</tool-content>"


def has_multimodal_content(result: Any) -> bool:
    """Check if a tool result contains non-text content blocks."""
    if not isinstance(result, list):
        return False
    return any(
        isinstance(block, dict) and block.get("type") not in ("text", None)
        for block in result
    )


def pack_multimodal_tool_result(
    result: list[Any],
    content_ops: BaseContentOps,
    multimodal_packs: dict[str, list[dict[str, Any]]],
    call_id: str,
    warnings: list[str],
) -> None:
    """Extract visual content blocks from a multimodal tool result.

    Converted blocks are stored in ``multimodal_packs[call_id]`` for
    later injection by ``inject_packed_tool_content``.

    Args:
        result: The IR tool result content (list of content blocks).
        content_ops: Provider content ops for converting blocks.
        multimodal_packs: Accumulator mapping call_id → provider content blocks.
        call_id: The tool call ID.
        warnings: Warning accumulator.
    """
    packed_parts: list[dict[str, Any]] = []

    for block in result:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            packed_parts.append(content_ops.ir_text_to_p(block))
        elif block_type == "image":
            try:
                packed_parts.append(content_ops.ir_image_to_p(block))
            except ValueError as e:
                warnings.append(f"Skipped image in tool result packing: {e}")
        elif block_type == "file":
            warnings.append(
                "File content not supported in tool result packing, skipped"
            )
        else:
            warnings.append(
                f"Unsupported block type in tool result packing: {block_type}"
            )

    if packed_parts:
        multimodal_packs[call_id] = packed_parts


def inject_packed_tool_content(
    messages: list[dict[str, Any]],
    multimodal_packs: dict[str, list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Insert synthetic user messages with packed multimodal tool content.

    After each group of consecutive ``role: "tool"`` messages, inserts a
    synthetic ``role: "user"`` message containing ``<tool-content>`` tagged
    blocks for any tool results that have multimodal content.
    """
    if not multimodal_packs:
        return messages

    result: list[dict[str, Any]] = []
    i = 0

    while i < len(messages):
        msg = messages[i]
        result.append(msg)
        i += 1

        if msg.get("role") != "tool":
            continue

        tool_call_ids = [msg.get("tool_call_id")]
        while i < len(messages) and messages[i].get("role") == "tool":
            result.append(messages[i])
            tool_call_ids.append(messages[i].get("tool_call_id"))
            i += 1

        synthetic_parts: list[dict[str, Any]] = []
        for tcid in tool_call_ids:
            if tcid and tcid in multimodal_packs:
                synthetic_parts.append(
                    {"type": "text", "text": f'<tool-content call-id="{tcid}">'}
                )
                synthetic_parts.extend(multimodal_packs[tcid])
                synthetic_parts.append({"type": "text", "text": TOOL_CONTENT_CLOSE_TAG})

        if synthetic_parts:
            result.append({"role": "user", "content": synthetic_parts})

    return result


def is_synthetic_tool_content_msg(msg: dict[str, Any]) -> bool:
    """Check if a user message is a synthetic tool content message."""
    if msg.get("role") != "user":
        return False
    content = msg.get("content")
    if not isinstance(content, list) or not content:
        return False
    first = content[0]
    if isinstance(first, dict) and first.get("type") == "text":
        return bool(TOOL_CONTENT_OPEN_TAG_RE.match(first.get("text", "")))
    return False


def unpack_tool_content(
    messages: list[dict[str, Any]],
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Extract multimodal content from synthetic user messages.

    Returns:
        Tuple of (unpacked_content mapping, clean message list).
    """
    unpacked: dict[str, list[dict[str, Any]]] = {}
    clean: list[dict[str, Any]] = []

    for msg in messages:
        if not is_synthetic_tool_content_msg(msg):
            clean.append(msg)
            continue

        content = msg.get("content", [])
        current_call_id: str | None = None
        current_blocks: list[dict[str, Any]] = []

        for part in content:
            if not isinstance(part, dict):
                continue

            if part.get("type") == "text":
                text = part.get("text", "")

                open_match = TOOL_CONTENT_OPEN_TAG_RE.match(text)
                if open_match:
                    if current_call_id and current_blocks:
                        unpacked[current_call_id] = current_blocks
                    current_call_id = open_match.group(1)
                    current_blocks = []
                    continue

                if text == TOOL_CONTENT_CLOSE_TAG:
                    if current_call_id and current_blocks:
                        unpacked[current_call_id] = current_blocks
                    current_call_id = None
                    current_blocks = []
                    continue

            if current_call_id is not None:
                current_blocks.append(part)

        if current_call_id and current_blocks:
            unpacked[current_call_id] = current_blocks

    return unpacked, clean


__all__ = [
    "TOOL_CONTENT_CLOSE_TAG",
    "TOOL_CONTENT_OPEN_TAG_RE",
    "has_multimodal_content",
    "inject_packed_tool_content",
    "is_synthetic_tool_content_msg",
    "pack_multimodal_tool_result",
    "unpack_tool_content",
]
