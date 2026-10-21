"""Base tool layer — abstract ops class and stateless tool helpers.

This package consolidates all tool-related code that was previously
split between ``converters/base/tools.py`` (the abstract base class)
and ``converters/base/helpers/tool_*.py`` (stateless helpers).

Public API contract — the following import paths are stable and used
by external callers (e.g. argo-proxy)::

    from llm_rosetta.converters.base.tools import BaseToolOps
    from llm_rosetta.converters.base.tools import sanitize_schema

The stateless helpers live in :mod:`llm_rosetta.ir.tools` (and
``ir.primitives.call_unwind``); this package re-exports them so the documented
``converters.base.tools`` names keep working.  What stays here is the part
that needs converter internals:

    ops             — :class:`BaseToolOps` abstract base class
    schema          — JSON Schema sanitization (uses the converter cache)
    content         — Multimodal content block conversion (``BaseContentOps``)
    multimodal_patch — Tool result dual-encoding (``BaseContentOps``)
"""

# -- Abstract base class --
from .ops import BaseToolOps, IntrinsicLeg

# -- Schema --
from .schema import convert_nullable_to_type_array, sanitize_schema

# -- Call ID --
from llm_rosetta.ir.tools.call_id import prefixed_tool_call_id, sanitize_tool_call_id

# -- Orphan fix --
from llm_rosetta.ir.tools.orphan_fix import (
    extract_part_ids,
    fix_orphaned_tool_calls_ir,
    log_orphan_warnings,
    strip_orphaned_tool_config,
)

# -- Call unwind --
from llm_rosetta.ir.primitives.call_unwind import unwind_parallel_tool_calls_ir

# -- Content --
from .content import convert_content_blocks_to_ir, convert_ir_content_blocks_to_p

# -- Batch --
from llm_rosetta.ir.tools.batch import assign_tool_batch_ids, merge_tool_messages

# -- Intrinsic --
from llm_rosetta.ir.tools.intrinsic import (
    get_definition_kind,
    get_intrinsic_kind,
    get_native_definition,
    intrinsic_kind_of,
    is_intrinsic,
    is_intrinsic_part,
    make_intrinsic_tool_call,
    make_intrinsic_tool_definition,
    make_intrinsic_tool_result,
)

# -- Error marker --
from llm_rosetta.ir.tools.error_marker import (
    TOOL_ERROR_MARKER,
    mark_tool_error,
    mark_tool_error_text,
    warn_tool_error,
    warn_tool_error_dropped,
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

# -- Strict mode --
from llm_rosetta.ir.tools.strict_mode import (
    ir_strict,
    write_ir_strict,
)

# -- Allow-list --
from llm_rosetta.ir.tools.allowed_tools import (
    allowed_tool_names,
    allowed_tool_refs,
    provider_allowed_tools_to_ir,
)

__all__ = [
    # ops
    "BaseToolOps",
    "IntrinsicLeg",
    # schema
    "convert_nullable_to_type_array",
    "sanitize_schema",
    # call_id
    "prefixed_tool_call_id",
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
    "get_definition_kind",
    "get_native_definition",
    "intrinsic_kind_of",
    "is_intrinsic",
    "make_intrinsic_tool_definition",
    "is_intrinsic_part",
    "get_intrinsic_kind",
    "make_intrinsic_tool_call",
    "make_intrinsic_tool_result",
    # error_marker
    "TOOL_ERROR_MARKER",
    "mark_tool_error",
    "mark_tool_error_text",
    "warn_tool_error",
    "warn_tool_error_dropped",
    # multimodal_patch
    "TOOL_CONTENT_CLOSE_TAG",
    "TOOL_CONTENT_OPEN_TAG_RE",
    "has_multimodal_content",
    "inject_packed_tool_content",
    "is_synthetic_tool_content_msg",
    "pack_multimodal_tool_result",
    "unpack_tool_content",
    # strict_mode
    "ir_strict",
    "write_ir_strict",
    # allowed_tools
    "allowed_tool_names",
    "allowed_tool_refs",
    "provider_allowed_tools_to_ir",
]
