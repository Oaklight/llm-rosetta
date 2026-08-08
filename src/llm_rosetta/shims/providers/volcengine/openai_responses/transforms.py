"""Volcengine OpenAI Responses schema transforms.

IR-level: hoist late system messages for prompt cache prefix stability.
"""

from llm_rosetta.shims.transforms import hoist_late_system_messages

post_ir_transforms = ()
pre_ir_transforms = ()
ir_transforms = (hoist_late_system_messages(),)
