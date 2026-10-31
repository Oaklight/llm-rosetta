"""IR primitives: body/IR transforms and the stateless helpers they need.

Everything here depends only on the standard library and
``llm_rosetta.types.ir``.  The transform factories are re-exported for
``llm_rosetta.transforms``; the helper implementations live in their own
submodules so a caller can reach them without the transform surface.
"""

from __future__ import annotations

from .body import (  # noqa: F401
    Transform,
    _NamedTransform,
    apply_transforms,
    backfill_reasoning_content,
    default_message_field,
    default_tool_description,
    flatten_system_content,
    rename_field,
    replace_message_field,
    rewrite_harmony_tool_calls,
    force_fields,
    set_defaults,
    strip_fields,
    strip_fields_for_model,
)
from .cache_breakpoints import inject_cache_breakpoints  # noqa: F401
from .call_unwind import unwind_parallel_tool_calls_ir  # noqa: F401
from .native import (  # noqa: F401
    LEGACY_FINISH_REASON_KEY,
    LEGACY_PROMPT_FEEDBACK_KEY,
    NATIVE_RESPONSE_KEY,
    native_record,
    record_native,
)
from .image_limit import (  # noqa: F401
    strip_images_for_non_vision,
    truncate_images as truncate_images_impl,
)
from .ir import (  # noqa: F401
    IRTransform,
    TransformContext,
    _NamedIRTransform,
    apply_ir_transforms,
    auto_cache_breakpoints,
    hoist_late_system_messages,
    truncate_images,
    unwind_parallel_tool_calls,
)
from .system_message_hoist import (  # noqa: F401
    hoist_late_system_messages_ir,
    hoist_system_in_converter,
)
from .tool_group import (  # noqa: F401
    GENERATED_GROUP_MARKER,
    ToolGroup,
    generated_group_id,
    group_tools,
    tool_call_group,
)
from .truncate import truncate_with_digest  # noqa: F401

__all__ = [
    "GENERATED_GROUP_MARKER",
    "LEGACY_FINISH_REASON_KEY",
    "LEGACY_PROMPT_FEEDBACK_KEY",
    "NATIVE_RESPONSE_KEY",
    "native_record",
    "record_native",
    "IRTransform",
    "ToolGroup",
    "Transform",
    "TransformContext",
    "apply_ir_transforms",
    "apply_transforms",
    "auto_cache_breakpoints",
    "backfill_reasoning_content",
    "default_message_field",
    "default_tool_description",
    "flatten_system_content",
    "generated_group_id",
    "group_tools",
    "hoist_late_system_messages",
    "hoist_late_system_messages_ir",
    "hoist_system_in_converter",
    "inject_cache_breakpoints",
    "rename_field",
    "replace_message_field",
    "rewrite_harmony_tool_calls",
    "force_fields",
    "set_defaults",
    "strip_fields",
    "strip_fields_for_model",
    "strip_images_for_non_vision",
    "tool_call_group",
    "truncate_images",
    "truncate_images_impl",
    "truncate_with_digest",
    "unwind_parallel_tool_calls",
    "unwind_parallel_tool_calls_ir",
]
