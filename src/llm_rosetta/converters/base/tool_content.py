"""
LLM-Rosetta - Tool Content Conversion Helpers

Shared helpers for converting multimodal content blocks inside tool results.
Used by each provider's tool_ops to normalize content through content_ops
during p_tool_result_to_ir() and ir_tool_result_to_p() conversions.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .content import BaseContentOps

logger = logging.getLogger(__name__)

# IR content block type names
_IR_TEXT_TYPES = {"text"}
_IR_IMAGE_TYPES = {"image"}
_IR_FILE_TYPES = {"file"}

# Provider-specific type names that map to IR types
_PROVIDER_TEXT_TYPES = {"text", "input_text", "output_text"}
_PROVIDER_IMAGE_TYPES = {"image", "input_image", "image_url"}
_PROVIDER_FILE_TYPES = {"file", "input_file", "document"}


def convert_content_blocks_to_ir(
    blocks: list[Any],
    content_ops_class: type[BaseContentOps],
) -> list[dict[str, Any]]:
    """Convert a list of provider-specific content blocks to IR format.

    Dispatches each block by its ``type`` field to the appropriate
    ``content_ops_class.p_*_to_ir()`` method. Blocks without a
    recognized type are passed through unchanged.

    Args:
        blocks: List of provider content block dicts.
        content_ops_class: The provider's ContentOps class for conversion.

    Returns:
        List of IR content part dicts.
    """
    result: list[dict[str, Any]] = []
    for block in blocks:
        converted = _p_block_to_ir(block, content_ops_class)
        if converted is not None:
            result.append(converted)
    return result


def convert_ir_content_blocks_to_p(
    blocks: list[Any],
    content_ops_class: type[BaseContentOps],
) -> list[dict[str, Any]]:
    """Convert a list of IR content blocks to provider format.

    Dispatches each block by its ``type`` field to the appropriate
    ``content_ops_class.ir_*_to_p()`` method. Blocks without a
    recognized IR type are passed through unchanged.

    Args:
        blocks: List of IR content part dicts.
        content_ops_class: The provider's ContentOps class for conversion.

    Returns:
        List of provider content block dicts.
    """
    result: list[dict[str, Any]] = []
    for block in blocks:
        converted = _ir_block_to_p(block, content_ops_class)
        if converted is not None:
            result.append(converted)
    return result


def _p_block_to_ir(
    block: Any,
    content_ops_class: type[BaseContentOps],
) -> dict[str, Any] | None:
    """Convert a single provider content block to IR format."""
    if isinstance(block, str):
        return {**content_ops_class.p_text_to_ir(block)}

    if not isinstance(block, dict):
        return block

    block_type = block.get("type", "")

    # Text
    if block_type in _PROVIDER_TEXT_TYPES:
        return {**content_ops_class.p_text_to_ir(block)}

    # Image
    if block_type in _PROVIDER_IMAGE_TYPES:
        return {**content_ops_class.p_image_to_ir(block)}

    # File / Document
    if block_type in _PROVIDER_FILE_TYPES:
        return {**content_ops_class.p_file_to_ir(block)}

    # Google-style inline data (no "type" field, uses "inlineData" key)
    if "inlineData" in block or "inline_data" in block:
        return {**content_ops_class.p_image_to_ir(block)}

    # Unknown — pass through as-is
    logger.debug("Unknown content block type in tool result: %s", block_type)
    return block


def _ir_block_to_p(
    block: Any,
    content_ops_class: type[BaseContentOps],
) -> dict[str, Any] | None:
    """Convert a single IR content block to provider format."""
    if isinstance(block, str):
        return {**content_ops_class.ir_text_to_p({"type": "text", "text": block})}

    if not isinstance(block, dict):
        return block

    block_type = block.get("type", "")

    # IR text
    if block_type in _IR_TEXT_TYPES:
        result = content_ops_class.ir_text_to_p(block)
        return {**result} if result is not None else None

    # IR image
    if block_type in _IR_IMAGE_TYPES:
        try:
            result = content_ops_class.ir_image_to_p(block)
            return {**result} if result is not None else None
        except (ValueError, KeyError):
            logger.warning("Failed to convert IR image block: %s", block)
            return None

    # IR file
    if block_type in _IR_FILE_TYPES:
        try:
            result = content_ops_class.ir_file_to_p(block)
            return {**result} if result is not None else None
        except (ValueError, KeyError):
            logger.warning("Failed to convert IR file block: %s", block)
            return None

    # Unknown — pass through as-is
    logger.debug("Unknown IR content block type in tool result: %s", block_type)
    return block
