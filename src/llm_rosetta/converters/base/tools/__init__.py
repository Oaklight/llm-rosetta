"""Base tool layer — abstract ops class and stateless tool helpers.

This package consolidates all tool-related code that was previously
split between ``converters/base/tools.py`` (the abstract base class)
and ``converters/base/helpers/tool_*.py`` (stateless helpers).

Public API contract — the following import paths are stable and used
by external callers (e.g. argo-proxy)::

    from llm_rosetta.converters.base.tools import BaseToolOps
    from llm_rosetta.converters.base.tools import sanitize_schema

Submodules:
    ops             — :class:`BaseToolOps` abstract base class
    schema          — JSON Schema sanitization for provider compatibility
    call_id         — Tool call ID sanitization
    content         — Multimodal content block conversion in tool results
    orphan_fix      — Orphaned tool call/result repair
    call_unwind     — Parallel tool call unwinding
    intrinsic       — Intrinsic (server-side) tool helpers
    batch           — Tool result batch tracking
    multimodal_patch — Tool result dual-encoding for non-multimodal providers
"""

# -- Abstract base class --
from .ops import BaseToolOps

# -- Schema --
from .schema import convert_nullable_to_type_array, sanitize_schema

# -- Call ID --
from .call_id import sanitize_tool_call_id

# -- Orphan fix --
from .orphan_fix import (
    extract_part_ids,
    fix_orphaned_tool_calls_ir,
    log_orphan_warnings,
    strip_orphaned_tool_config,
)

# -- Call unwind --
from .call_unwind import unwind_parallel_tool_calls_ir

# -- Content --
from .content import convert_content_blocks_to_ir, convert_ir_content_blocks_to_p

# -- Batch --
from .batch import assign_tool_batch_ids, merge_tool_messages

# -- Intrinsic --
from .intrinsic import (
    get_intrinsic_kind,
    is_intrinsic_part,
    make_intrinsic_tool_call,
    make_intrinsic_tool_result,
    set_intrinsic_kind,
)

# -- Multimodal patch --
from .multimodal_patch import (
    TOOL_CONTENT_CLOSE_TAG,
    TOOL_CONTENT_OPEN_TAG_RE,
    has_multimodal_content,
    inject_packed_tool_content,
    is_synthetic_tool_content_msg,
    pack_multimodal_tool_result,
    unpack_tool_content,
)

__all__ = [
    # ops
    "BaseToolOps",
    # schema
    "convert_nullable_to_type_array",
    "sanitize_schema",
    # call_id
    "sanitize_tool_call_id",
    # orphan_fix
    "extract_part_ids",
    "fix_orphaned_tool_calls_ir",
    "log_orphan_warnings",
    "strip_orphaned_tool_config",
    # call_unwind
    "unwind_parallel_tool_calls_ir",
    # content
    "convert_content_blocks_to_ir",
    "convert_ir_content_blocks_to_p",
    # batch
    "assign_tool_batch_ids",
    "merge_tool_messages",
    # intrinsic
    "is_intrinsic_part",
    "get_intrinsic_kind",
    "set_intrinsic_kind",
    "make_intrinsic_tool_call",
    "make_intrinsic_tool_result",
    # multimodal_patch
    "TOOL_CONTENT_CLOSE_TAG",
    "TOOL_CONTENT_OPEN_TAG_RE",
    "has_multimodal_content",
    "inject_packed_tool_content",
    "is_synthetic_tool_content_msg",
    "pack_multimodal_tool_result",
    "unpack_tool_content",
]
