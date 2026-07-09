"""Moonshot (Kimi) schema transforms.

Moonshot does not support ``logprobs``, ``top_logprobs``, ``logit_bias``,
or ``seed`` in its chat completions API.  Strip them before sending
upstream to avoid errors.

References:
    https://platform.moonshot.cn/docs/api/chat
"""

from llm_rosetta.shims.transforms import strip_fields

post_ir_transforms = (strip_fields("logprobs", "top_logprobs", "logit_bias", "seed"),)
pre_ir_transforms = ()
