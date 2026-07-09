"""Qwen (DashScope) schema transforms.

Qwen's OpenAI-compatible endpoint does not support ``frequency_penalty``
or ``logit_bias``.  Qwen uses its own ``repetition_penalty`` (via
``extra_body``) instead of ``frequency_penalty``.

References:
    https://help.aliyun.com/zh/model-studio/compatibility-of-openai-with-dashscope
"""

from llm_rosetta.shims.transforms import strip_fields

post_ir_transforms = (strip_fields("frequency_penalty", "logit_bias"),)
pre_ir_transforms = ()
