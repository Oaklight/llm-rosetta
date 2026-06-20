"""Image truncation for providers with max_images limits."""

from __future__ import annotations

import copy
import logging
from typing import Any

from llm_rosetta.types.ir.type_guards import is_image_part, is_tool_result_part

logger = logging.getLogger(__name__)

# Placeholder text for replaced images.
_PLACEHOLDER = "[image omitted: provider limit of {limit} images per request]"

# Position types for image locations in the IR.
# Direct: (msg_idx, part_idx, None)  — image in message content
# Nested: (msg_idx, part_idx, block_idx)  — image inside tool_result.result list
_ImagePos = tuple[int, int, int | None]


def _collect_image_positions(messages: list[Any]) -> list[_ImagePos]:
    """Find all image positions in messages, including inside tool results.

    Returns positions in conversation order (oldest first).
    """
    positions: list[_ImagePos] = []
    for msg_idx, msg in enumerate(messages):
        for part_idx, part in enumerate(msg.get("content", [])):
            if is_image_part(part):
                positions.append((msg_idx, part_idx, None))
            elif is_tool_result_part(part):
                result = part.get("result")
                if isinstance(result, list):
                    for block_idx, block in enumerate(result):
                        if isinstance(block, dict) and is_image_part(block):
                            positions.append((msg_idx, part_idx, block_idx))
    return positions


def truncate_images(
    ir_request: dict[str, Any],
    max_images: int,
    *,
    request_id: str = "-",
) -> dict[str, Any]:
    """Return a (possibly new) IR request with at most *max_images* images.

    Strategy: keep the MOST RECENT images; replace earlier ones with a
    text placeholder so the conversation context is preserved.

    Scans both direct message content and images embedded inside
    ``tool_result.result`` lists (which the OpenAI Chat converter
    unpacks into synthetic user messages with ``image_url`` parts).

    Args:
        ir_request: The IR request dict to inspect/modify.
        max_images: Maximum number of image parts allowed.
        request_id: Used in log messages for traceability.

    Returns:
        The original dict if no truncation needed, otherwise a shallow copy
        with ``messages`` replaced by a new list with excess images replaced.
    """
    messages = ir_request.get("messages", [])
    image_positions = _collect_image_positions(messages)

    total = len(image_positions)
    if total <= max_images:
        return ir_request

    # Keep the last max_images, truncate the rest (oldest first)
    to_replace = image_positions[: total - max_images]
    logger.warning(
        "[%s] truncated %d images to %d (provider limit of %d)",
        request_id,
        total,
        max_images,
        max_images,
    )

    placeholder = {"type": "text", "text": _PLACEHOLDER.format(limit=max_images)}

    new_messages: list[Any] = copy.deepcopy(messages)
    for msg_idx, part_idx, block_idx in to_replace:
        if block_idx is None:
            # Direct image in message content
            new_messages[msg_idx]["content"][part_idx] = placeholder.copy()
        else:
            # Image inside tool_result.result list
            new_messages[msg_idx]["content"][part_idx]["result"][block_idx] = (
                placeholder.copy()
            )

    return {**ir_request, "messages": new_messages}
