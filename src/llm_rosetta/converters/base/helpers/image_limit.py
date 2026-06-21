"""Image handling for providers with max_images limits or no vision support."""

from __future__ import annotations

import copy
import logging
from typing import Any, cast

from llm_rosetta.types.ir.type_guards import is_image_part, is_tool_result_part

logger = logging.getLogger(__name__)

# Placeholders for different image removal scenarios.
_PLACEHOLDER_NO_VISION = {"type": "text", "text": "[image not available]"}
_PLACEHOLDER_LIMIT = {"type": "text", "text": "[image omitted due to limit]"}

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
                        if isinstance(block, dict) and is_image_part(cast(Any, block)):
                            positions.append((msg_idx, part_idx, block_idx))
    return positions


def _replace_images(
    ir_request: dict[str, Any],
    positions: list[_ImagePos],
    placeholder: dict[str, str],
) -> dict[str, Any]:
    """Replace image parts at the given positions with placeholders.

    Only deep-copies affected messages to avoid copying hundreds of MB
    of base64 image data in large conversations.

    Args:
        ir_request: The IR request dict.
        positions: List of (msg_idx, part_idx, block_idx) to replace.
        placeholder: The text part dict to use as replacement.

    Returns:
        A new IR request dict with affected messages replaced.
    """
    messages = ir_request.get("messages", [])

    # Group by message index to minimize copies
    affected_msgs: dict[int, list[tuple[int, int | None]]] = {}
    for msg_idx, part_idx, block_idx in positions:
        affected_msgs.setdefault(msg_idx, []).append((part_idx, block_idx))

    new_messages: list[Any] = list(messages)  # shallow copy of list
    for msg_idx, replacements in affected_msgs.items():
        msg_copy = copy.deepcopy(messages[msg_idx])
        for part_idx, block_idx in replacements:
            if block_idx is None:
                msg_copy["content"][part_idx] = placeholder.copy()
            else:
                msg_copy["content"][part_idx]["result"][block_idx] = placeholder.copy()
        new_messages[msg_idx] = msg_copy

    return {**ir_request, "messages": new_messages}


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

    return _replace_images(ir_request, to_replace, _PLACEHOLDER_LIMIT)


def strip_images_for_non_vision(
    ir_request: dict[str, Any],
    *,
    model: str = "",
    request_id: str = "-",
) -> dict[str, Any]:
    """Strip ALL images from an IR request for a non-vision model.

    Replaces every image part (direct and inside tool results) with a
    ``[image not available]`` text placeholder.

    Args:
        ir_request: The IR request dict to inspect/modify.
        model: Model name for log messages.
        request_id: Used in log messages for traceability.

    Returns:
        The original dict if no images found, otherwise a new dict
        with all images replaced by text placeholders.
    """
    messages = ir_request.get("messages", [])
    image_positions = _collect_image_positions(messages)

    if not image_positions:
        return ir_request

    logger.warning(
        "[%s] stripped %d images: model %s does not have vision capability",
        request_id,
        len(image_positions),
        model,
    )

    return _replace_images(ir_request, image_positions, _PLACEHOLDER_NO_VISION)
