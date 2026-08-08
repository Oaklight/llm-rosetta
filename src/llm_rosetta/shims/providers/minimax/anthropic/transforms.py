"""MiniMax Anthropic schema transforms.

IR-level: hoist late system messages and inject cache breakpoints for
prompt cache prefix stability.
"""

from llm_rosetta.shims.transforms import (
    auto_cache_breakpoints,
    hoist_late_system_messages,
)

post_ir_transforms = ()
pre_ir_transforms = ()
ir_transforms = (hoist_late_system_messages(), auto_cache_breakpoints())
