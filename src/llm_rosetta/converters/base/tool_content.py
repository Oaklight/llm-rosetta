"""Backward-compatibility shim for ``llm_rosetta.converters.base.tool_content``.

This module moved to ``llm_rosetta.converters.base.helpers.tool_content`` in
v0.6.11.  It is re-exported here so the old import path keeps working.

New code should import from ``llm_rosetta.converters.base.helpers`` instead.
"""

from .helpers.tool_content import (
    convert_content_blocks_to_ir,
    convert_ir_content_blocks_to_p,
)

__all__ = [
    "convert_content_blocks_to_ir",
    "convert_ir_content_blocks_to_p",
]
