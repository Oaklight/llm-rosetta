"""Capability/adaptation passes over IR input.

A pass adapts an IR value to a model's declared capabilities.  Passes may use
``ir.primitives``, ``ir.tools`` and ``ir.context``; they must not import a
converter family or the shim contract.  The passes that read a shim's
declarations (reasoning, vision, custom and intrinsic tools) are
:mod:`llm_rosetta.pipeline.enforcement`.
"""

from __future__ import annotations

from .description_limit import relocate_oversized_tool_descriptions  # noqa: F401
from .intrinsic_history import degrade_intrinsic_history  # noqa: F401
from .request_constraints import (  # noqa: F401
    RequestConstraintViolation,
    apply_request_constraints,
)
from .tool_names import (  # noqa: F401
    MAX_TOOL_NAME_LEN,
    STRICT_TOOL_NAME_PATTERN,
    ToolNameMap,
    apply_upstream_tool_names,
    build_tool_name_map,
    dedupe_tool_definitions,
    qualify_tool_names,
    restore_client_tool_name_events,
    restore_client_tool_names,
    sanitize_tool_names,
)

__all__ = [
    "MAX_TOOL_NAME_LEN",
    "STRICT_TOOL_NAME_PATTERN",
    "CustomToolRestorer",
    "RequestConstraintViolation",
    "ToolNameMap",
    "apply_request_constraints",
    "apply_upstream_tool_names",
    "backfill_reasoning_replay",
    "build_tool_name_map",
    "dedupe_tool_definitions",
    "degrade_intrinsic_history",
    "enforce_custom_tools",
    "enforce_reasoning",
    "reasoning_capability",
    "drop_nameless_tools",
    "enforce_vision",
    "get_custom_tool_names",
    "qualify_tool_names",
    "relocate_oversized_tool_descriptions",
    "resolve_intrinsic_tools",
    "restore_client_tool_name_events",
    "restore_client_tool_names",
    "sanitize_tool_names",
    "restore_custom_tool_calls",
    "strip_reasoning_for_non_reasoning",
    "unwrap_custom_tool_input",
]
