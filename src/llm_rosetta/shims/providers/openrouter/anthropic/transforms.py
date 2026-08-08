"""OpenRouter Anthropic shim transforms.

OpenRouter's Anthropic-compatible endpoint (`/api`) is a faithful proxy
of the Anthropic Messages API — no request- or response-side transforms
are needed.  The IR-level transforms hoist late system messages (to
preserve prompt cache prefix) and inject cache breakpoints for
cross-format requests.
"""

from llm_rosetta.shims.transforms import (
    auto_cache_breakpoints,
    hoist_late_system_messages,
)

post_ir_transforms = ()
pre_ir_transforms = ()
ir_transforms = (hoist_late_system_messages(), auto_cache_breakpoints())
