"""Converter helper utilities — IR-level pre/post-processing.

This subpackage contains non-tool utility functions that support the
conversion pipeline.  All helpers operate on IR-level data structures
and are provider-agnostic.

Tool-related helpers have moved to :mod:`converters.base.tools` as of
PR #871.  Importing them from here will raise :class:`ImportError` with
a migration hint.

Modules:
    cache              — LRU cache for tool definition conversion results.
    cache_breakpoints  — Cache breakpoint detection.
    image_limit        — Truncate images to provider-declared limits.
    reasoning          — Shim-driven reasoning configuration helpers.
    system_message_hoist — Hoist late system messages to the front.
    truncate           — Fit an identifier into a length budget uniquely.
"""

from .reasoning import DEFAULT_REASONING_CAPS, apply_reasoning_config
from .truncate import truncate_with_digest
from .system_message_hoist import hoist_late_system_messages_ir

# ── Moved names: clear error instead of silent breakage ──

_MOVED_TO_TOOLS: set[str] = {
    "extract_part_ids",
    "fix_orphaned_tool_calls_ir",
    "log_orphan_warnings",
    "strip_orphaned_tool_config",
    "convert_nullable_to_type_array",
    "sanitize_schema",
    "sanitize_tool_call_id",
    "assign_tool_batch_ids",
    "merge_tool_messages",
    "unwind_parallel_tool_calls_ir",
    "convert_content_blocks_to_ir",
    "convert_ir_content_blocks_to_p",
    "is_intrinsic_part",
    "get_intrinsic_kind",
    "set_intrinsic_kind",
    "make_intrinsic_tool_call",
    "make_intrinsic_tool_result",
}


def __getattr__(name: str) -> object:
    if name in _MOVED_TO_TOOLS:
        raise ImportError(
            f"{name!r} has moved from 'converters.base.helpers' to "
            f"'converters.base.tools'.  Update your import to:\n"
            f"  from llm_rosetta.converters.base.tools import {name}\n"
            f"See: https://github.com/Oaklight/llm-rosetta/pull/871"
        )
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "DEFAULT_REASONING_CAPS",
    "apply_reasoning_config",
    "truncate_with_digest",
    "hoist_late_system_messages_ir",
]
