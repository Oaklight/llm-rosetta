"""Converter helper utilities — IR-level pre/post-processing.

This subpackage contains utility functions that support the conversion
pipeline but are not part of the abstract Ops interface hierarchy.
All helpers operate on IR-level data structures and are provider-agnostic.

Modules:
    cache           — LRU cache for tool definition conversion results.
    image_limit     — Truncate images to provider-declared limits.
    orphan_fix      — Fix mismatched tool_call / tool_result pairing.
    schema          — JSON Schema sanitization for provider compatibility.
    tool_call_unwind — Unwind parallel tool calls into sequential pairs.
    tool_content    — Multimodal content block conversion inside tool results.
"""

# Re-export public functions so call sites can use:
#   from ..base.helpers import fix_orphaned_tool_calls_ir
# without knowing internal file layout.

from .orphan_fix import (
    extract_part_ids,
    fix_orphaned_tool_calls_ir,
    log_orphan_warnings,
    strip_orphaned_tool_config,
)
from .schema import sanitize_schema
from .tool_call_unwind import unwind_parallel_tool_calls_ir
from .tool_content import convert_content_blocks_to_ir, convert_ir_content_blocks_to_p

__all__ = [
    # orphan_fix
    "extract_part_ids",
    "fix_orphaned_tool_calls_ir",
    "log_orphan_warnings",
    "strip_orphaned_tool_config",
    # schema
    "sanitize_schema",
    # tool_call_unwind
    "unwind_parallel_tool_calls_ir",
    # tool_content
    "convert_content_blocks_to_ir",
    "convert_ir_content_blocks_to_p",
]
