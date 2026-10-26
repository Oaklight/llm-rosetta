"""Stateless tool helpers — the IR tool layer.

These modules depend on the standard library and ``llm_rosetta.types.ir``
only.  The converter-facing ``BaseToolOps`` abstract class, the schema
sanitizer that uses the converter cache, and the content/packing helpers that
need ``BaseContentOps`` stay in :mod:`llm_rosetta.converters.base.tools`;
this package is the part of that tree with no converter dependency.
"""

from __future__ import annotations

from .allowed_tools import (  # noqa: F401
    allowed_tool_names,
    allowed_tool_refs,
    provider_allowed_tools_to_ir,
)
from .batch import assign_tool_batch_ids, merge_tool_messages  # noqa: F401
from .call_id import prefixed_tool_call_id, sanitize_tool_call_id  # noqa: F401
from .error_marker import (  # noqa: F401
    TOOL_ERROR_MARKER,
    mark_tool_error,
    mark_tool_error_text,
    warn_tool_error,
    warn_tool_error_dropped,
)
from .intrinsic import (  # noqa: F401
    get_definition_kind,
    get_intrinsic_kind,
    get_native_definition,
    history_native_calls,
    intrinsic_kind_of,
    is_intrinsic,
    is_intrinsic_part,
    make_intrinsic_tool_call,
    make_intrinsic_tool_definition,
    make_intrinsic_tool_result,
)
from .orphan_fix import (  # noqa: F401
    UNANSWERED_TOOL_CALL,
    extract_part_ids,
    fix_orphaned_tool_calls_ir,
    log_orphan_warnings,
    strip_orphaned_tool_config,
)
from .strict_mode import ir_strict, write_ir_strict  # noqa: F401

__all__ = [
    "TOOL_ERROR_MARKER",
    "UNANSWERED_TOOL_CALL",
    "allowed_tool_names",
    "allowed_tool_refs",
    "assign_tool_batch_ids",
    "extract_part_ids",
    "fix_orphaned_tool_calls_ir",
    "get_definition_kind",
    "get_intrinsic_kind",
    "get_native_definition",
    "history_native_calls",
    "intrinsic_kind_of",
    "ir_strict",
    "is_intrinsic",
    "is_intrinsic_part",
    "log_orphan_warnings",
    "make_intrinsic_tool_call",
    "make_intrinsic_tool_definition",
    "make_intrinsic_tool_result",
    "mark_tool_error",
    "mark_tool_error_text",
    "merge_tool_messages",
    "prefixed_tool_call_id",
    "provider_allowed_tools_to_ir",
    "sanitize_tool_call_id",
    "strip_orphaned_tool_config",
    "warn_tool_error",
    "warn_tool_error_dropped",
    "write_ir_strict",
]
