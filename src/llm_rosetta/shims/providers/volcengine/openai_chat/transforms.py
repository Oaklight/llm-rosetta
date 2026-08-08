"""Volcengine (Doubao) schema transforms.

Volcengine does not support the ``logprobs`` / ``top_logprobs`` fields.
Strip them before sending upstream to avoid 400 errors.
"""

from llm_rosetta.shims.transforms import hoist_late_system_messages, strip_fields

post_ir_transforms = (strip_fields("logprobs", "top_logprobs"),)
pre_ir_transforms = ()
ir_transforms = (hoist_late_system_messages(),)
