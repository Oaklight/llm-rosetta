"""The transform contract: hook types and the primitives a shim builds from.

A shim's ``transforms.py`` exports tuples under the names in
:mod:`llm_rosetta.contracts` (``pre_ir_transforms``, ``post_ir_transforms``,
``ir_transforms``, ``response_body_transforms``).  Two levels exist:

* **Body-level** — :data:`Transform`, ``(body) -> body``: provider-specific
  field operations on a raw request or response body.
* **IR-level** — :data:`IRTransform`, ``(ir_request, context) -> ir_request``:
  context-aware operations on the format-neutral request, with a
  :class:`TransformContext`.

Wrap a hand-written transform in :class:`NamedTransform` (body) or
:class:`NamedIRTransform` (IR) so it has a readable ``repr``.  The factories
below already return named transforms.

The implementations live in :mod:`llm_rosetta.ir.primitives`, which is
internal; this module is the stable name.  ``apply_transforms`` and
``apply_ir_transforms`` run a tuple the way the pipeline does, which is what a
shim's own tests need.

Typical use::

    from llm_rosetta.contracts.transform import hoist_late_system_messages, strip_fields

    pre_ir_transforms = (strip_fields("store"),)
    ir_transforms = (hoist_late_system_messages(),)
"""

from __future__ import annotations

from llm_rosetta.ir.primitives.body import (
    NamedTransform,
    Transform,
    apply_transforms,
    backfill_reasoning_content,
    default_message_field,
    default_tool_description,
    flatten_system_content,
    force_fields,
    rename_field,
    replace_message_field,
    rewrite_harmony_tool_calls,
    set_defaults,
    strip_fields,
    strip_fields_for_model,
)
from llm_rosetta.ir.primitives.ir import (
    IRTransform,
    NamedIRTransform,
    TransformContext,
    apply_ir_transforms,
    auto_cache_breakpoints,
    hoist_late_system_messages,
    truncate_images,
    unwind_parallel_tool_calls,
)

__all__ = [
    # Hook types
    "IRTransform",
    "NamedIRTransform",
    "NamedTransform",
    "Transform",
    "TransformContext",
    # Running a tuple of transforms
    "apply_ir_transforms",
    "apply_transforms",
    # Body-level primitives
    "backfill_reasoning_content",
    "default_message_field",
    "default_tool_description",
    "flatten_system_content",
    "force_fields",
    "rename_field",
    "replace_message_field",
    "rewrite_harmony_tool_calls",
    "set_defaults",
    "strip_fields",
    "strip_fields_for_model",
    # IR-level primitives
    "auto_cache_breakpoints",
    "hoist_late_system_messages",
    "truncate_images",
    "unwind_parallel_tool_calls",
]
