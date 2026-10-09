"""Transform primitives for the LLM-Rosetta conversion pipeline.

Two transform levels:

* **Body-level** (:mod:`.body`) — ``dict → dict``.  Provider-specific
  field operations (rename, strip, inject defaults) on raw request/response
  bodies.
* **IR-level** (:mod:`.ir`) — ``(dict, TransformContext) → dict``.
  Context-aware transforms on the format-neutral intermediate
  representation.

Most callers can import directly from this package::

    from llm_rosetta.transforms import hoist_late_system_messages, strip_fields
"""

from llm_rosetta.transforms.body import (  # noqa: F401
    Transform,
    _NamedTransform,
    apply_transforms,
    default_message_field,
    default_tool_description,
    flatten_system_content,
    rename_field,
    replace_message_field,
    rewrite_harmony_tool_calls,
    set_defaults,
    strip_fields,
    strip_fields_for_model,
)
from llm_rosetta.transforms.ir import (  # noqa: F401
    IRTransform,
    TransformContext,
    _NamedIRTransform,
    apply_ir_transforms,
    auto_cache_breakpoints,
    hoist_late_system_messages,
    strip_non_vision_images,
    truncate_images,
    unwind_parallel_tool_calls,
)
