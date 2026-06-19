"""Image truncation for providers with max_images limits."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from llm_rosetta.types.ir.request import IRRequest

from llm_rosetta.types.ir.type_guards import is_image_part

logger = logging.getLogger(__name__)


def truncate_images(
    ir_request: IRRequest,
    max_images: int,
    *,
    request_id: str = "-",
) -> IRRequest:
    """Return a (possibly new) IR request with at most *max_images* images.

    Strategy: keep the MOST RECENT images; replace earlier ones with a
    text placeholder so the conversation context is preserved.

    Args:
        ir_request: The IR request dict to inspect/modify.
        max_images: Maximum number of image parts allowed.
        request_id: Used in log messages for traceability.

    Returns:
        The original dict if no truncation needed, otherwise a shallow copy
        with ``messages`` replaced by a new list with excess images replaced.
    """
    # Collect all (msg_idx, part_idx) for image parts, in order
    messages = ir_request.get("messages", [])
    image_positions: list[tuple[int, int]] = []
    for msg_idx, msg in enumerate(messages):
        for part_idx, part in enumerate(msg.get("content", [])):
            if is_image_part(part):
                image_positions.append((msg_idx, part_idx))

    total = len(image_positions)
    if total <= max_images:
        return ir_request

    # Keep the last max_images, truncate the rest
    to_replace = image_positions[: total - max_images]
    logger.warning(
        "[%s] truncated %d images to %d (provider limit of %d)",
        request_id,
        total,
        max_images,
        max_images,
    )

    # Build new messages list with placeholders
    import copy

    new_messages = copy.deepcopy(messages)
    for msg_idx, part_idx in to_replace:
        new_messages[msg_idx]["content"][part_idx] = {
            "type": "text",
            "text": f"[image omitted: provider limit of {max_images} images per request]",
        }

    return {**ir_request, "messages": new_messages}
