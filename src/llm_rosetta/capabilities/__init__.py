"""Capability enforcement — adapt IR input to model and target capabilities.

Functions follow the ``enforce_*`` naming convention for the enforcement passes:

- :func:`enforce_reasoning` — configure reasoning output mode (pre-IR)
- :func:`strip_reasoning_for_non_reasoning` — strip reasoning for non-reasoning models
- :func:`enforce_vision` — strip images for non-vision models
- :func:`enforce_custom_tools` — downgrade custom tools for non-supporting providers
- :func:`dedupe_tool_definitions` — collapse duplicate tool definitions (post-IR)

Split by concern: :mod:`.enforcement` for the capability passes, :mod:`.tool_names`
for tool identity and naming, :mod:`.description_limit` for oversized descriptions.
"""

from .description_limit import relocate_oversized_tool_descriptions
from .enforcement import (
    enforce_reasoning,
    strip_reasoning_for_non_reasoning,
    enforce_vision,
    strip_intrinsic_tools,
    resolve_intrinsic_tools,
    enforce_custom_tools,
    get_custom_tool_names,
    restore_custom_tool_calls,
    unwrap_custom_tool_input,
)
from .tool_names import (
    MAX_TOOL_NAME_LEN,
    qualify_tool_names,
    build_tool_name_map,
    apply_upstream_tool_names,
    restore_client_tool_names,
    restore_client_tool_name_events,
    dedupe_tool_definitions,
    ToolNameMap,
)

__all__ = [
    "MAX_TOOL_NAME_LEN",
    "enforce_reasoning",
    "strip_reasoning_for_non_reasoning",
    "enforce_vision",
    "strip_intrinsic_tools",
    "resolve_intrinsic_tools",
    "enforce_custom_tools",
    "get_custom_tool_names",
    "restore_custom_tool_calls",
    "unwrap_custom_tool_input",
    "qualify_tool_names",
    "build_tool_name_map",
    "apply_upstream_tool_names",
    "restore_client_tool_names",
    "restore_client_tool_name_events",
    "dedupe_tool_definitions",
    "relocate_oversized_tool_descriptions",
    "ToolNameMap",
]
