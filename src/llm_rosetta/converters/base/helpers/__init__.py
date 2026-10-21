"""Converter helper utilities — IR-level pre/post-processing.

This subpackage contains non-tool utility functions that support the
conversion pipeline.  All helpers operate on IR-level data structures
and are provider-agnostic.

Tool-related helpers have moved to :mod:`converters.base.tools` as of
PR #871.  Importing them from here will raise :class:`ImportError` with
a migration hint.

Modules:
    cache              — LRU cache for tool definition conversion results.
    citations          — Citation-block provenance and filtering.
    extensions         — Wire-extension passthrough.
    opaque             — Provenance for opaque provider-specific reasoning values.
    reasoning          — Shim-driven reasoning configuration helpers.

The stateless IR helpers — cache breakpoints, image limits, system-message
hoisting, tool groups and truncation — live in
:mod:`llm_rosetta.ir.primitives`; this package re-exports them.
"""

from .opaque import (
    CARRIER_KEY,
    DUMMY_THOUGHT_SIGNATURE,
    OPAQUE_SAME_ORIGIN,
    carrier_metadata,
    get_opaque_origin,
    google_thought_signature,
    inject_google_function_call_signature,
    inject_interactions_function_call_signature,
    is_own_origin,
    set_google_thought_signature,
    set_opaque_origin,
    strip_internal_metadata,
)
from .reasoning import DEFAULT_REASONING_CAPS, apply_reasoning_config
from llm_rosetta.ir.primitives.truncate import truncate_with_digest
from llm_rosetta.ir.primitives.system_message_hoist import hoist_late_system_messages_ir

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
    "CARRIER_KEY",
    "DEFAULT_REASONING_CAPS",
    "DUMMY_THOUGHT_SIGNATURE",
    "OPAQUE_SAME_ORIGIN",
    "carrier_metadata",
    "apply_reasoning_config",
    "get_opaque_origin",
    "google_thought_signature",
    "hoist_late_system_messages_ir",
    "inject_google_function_call_signature",
    "inject_interactions_function_call_signature",
    "is_own_origin",
    "set_google_thought_signature",
    "set_opaque_origin",
    "strip_internal_metadata",
    "truncate_with_digest",
]
