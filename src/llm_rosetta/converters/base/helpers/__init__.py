"""Converter helper utilities — IR-level pre/post-processing.

This subpackage contains utility functions that support the conversion
pipeline but are not part of the abstract Ops interface hierarchy.
All helpers operate on IR-level data structures and are provider-agnostic.

Tool-related helpers have moved to :mod:`converters.base.tools`.
The ``__getattr__`` fallback below keeps the old import paths working
(e.g. ``from ..base.helpers import sanitize_schema``).

Modules (remaining in helpers):
    cache              — LRU cache for tool definition conversion results.
    cache_breakpoints  — Cache breakpoint detection.
    image_limit        — Truncate images to provider-declared limits.
    reasoning          — Shim-driven reasoning configuration helpers.
    system_message_hoist — Hoist late system messages to the front.
    truncate           — Fit an identifier into a length budget uniquely.
"""

from __future__ import annotations

# ── Non-tool helpers (canonical home) ──

from .reasoning import DEFAULT_REASONING_CAPS, apply_reasoning_config
from .truncate import truncate_with_digest
from .system_message_hoist import hoist_late_system_messages_ir

# ── Lazy backward-compat re-exports from tools/ ──
# Using __getattr__ avoids circular imports: tools/call_id.py imports
# helpers/truncate.py, which would trigger helpers/__init__.py, which
# would try to import tools/call_id.py again.

_TOOLS_REEXPORTS: dict[str, tuple[str, str]] = {
    # name → (submodule of ..tools, attribute name)
    "extract_part_ids": ("orphan_fix", "extract_part_ids"),
    "fix_orphaned_tool_calls_ir": ("orphan_fix", "fix_orphaned_tool_calls_ir"),
    "log_orphan_warnings": ("orphan_fix", "log_orphan_warnings"),
    "strip_orphaned_tool_config": ("orphan_fix", "strip_orphaned_tool_config"),
    "convert_nullable_to_type_array": ("schema", "convert_nullable_to_type_array"),
    "sanitize_schema": ("schema", "sanitize_schema"),
    "sanitize_tool_call_id": ("call_id", "sanitize_tool_call_id"),
    "assign_tool_batch_ids": ("batch", "assign_tool_batch_ids"),
    "merge_tool_messages": ("batch", "merge_tool_messages"),
    "unwind_parallel_tool_calls_ir": ("call_unwind", "unwind_parallel_tool_calls_ir"),
    "convert_content_blocks_to_ir": ("content", "convert_content_blocks_to_ir"),
    "convert_ir_content_blocks_to_p": ("content", "convert_ir_content_blocks_to_p"),
    "is_intrinsic_part": ("intrinsic", "is_intrinsic_part"),
    "get_intrinsic_kind": ("intrinsic", "get_intrinsic_kind"),
    "set_intrinsic_kind": ("intrinsic", "set_intrinsic_kind"),
    "make_intrinsic_tool_call": ("intrinsic", "make_intrinsic_tool_call"),
    "make_intrinsic_tool_result": ("intrinsic", "make_intrinsic_tool_result"),
}


def __getattr__(name: str) -> object:
    if name in _TOOLS_REEXPORTS:
        submod, attr = _TOOLS_REEXPORTS[name]
        import importlib

        mod = importlib.import_module(f"..tools.{submod}", __name__)
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # reasoning (canonical)
    "DEFAULT_REASONING_CAPS",
    "apply_reasoning_config",
    # truncate (canonical)
    "truncate_with_digest",
    # system_message_hoist (canonical)
    "hoist_late_system_messages_ir",
    # ── re-exports from tools/ (backward compat) ──
    *_TOOLS_REEXPORTS,
]
