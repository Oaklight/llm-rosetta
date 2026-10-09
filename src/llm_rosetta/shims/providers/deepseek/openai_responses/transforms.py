"""DeepSeek Responses API schema transforms.

DeepSeek's Responses API does not support ``n``, ``logit_bias``, or ``seed``.
"""

from llm_rosetta.transforms import strip_fields

post_ir_transforms = (strip_fields("n", "logit_bias", "seed"),)
pre_ir_transforms = ()
ir_transforms = ()
