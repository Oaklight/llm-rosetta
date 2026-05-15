"""Volcengine (Doubao) schema transforms.

Volcengine does not support the ``logprobs`` / ``top_logprobs`` fields.
Strip them before sending upstream to avoid 400 errors.
"""

from llm_rosetta.shims.transforms import strip_fields

to_transforms = (strip_fields("logprobs", "top_logprobs"),)
from_transforms = ()
