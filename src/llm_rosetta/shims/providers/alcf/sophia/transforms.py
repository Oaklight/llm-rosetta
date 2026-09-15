"""ALCF Sophia (vLLM on NVIDIA A100) schema transforms.

vLLM may not support ``logprobs`` / ``top_logprobs``.  Strip the
``developer`` role (not supported by open-source models) and replace
``content: null`` with an empty string to avoid vLLM crashes.
"""

from ..model_utils import make_alcf_model_list_transform
from llm_rosetta.shims.transforms import (
    default_tool_description,
    default_message_field,
    replace_message_field,
    strip_fields,
)

post_ir_transforms = (
    strip_fields("logprobs", "top_logprobs"),
    replace_message_field("role", "developer", "system"),
    default_message_field("content", ""),
    default_tool_description(),
)
pre_ir_transforms = ()
ir_transforms = ()

model_list_transform = make_alcf_model_list_transform("vllm")
