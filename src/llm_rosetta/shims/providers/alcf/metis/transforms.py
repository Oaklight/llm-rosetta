"""ALCF Metis (SambaNova SN40L) schema transforms.

SambaNova has known tool-call sanitization issues and does not support
several OpenAI-specific fields.  Strip ``logprobs``, ``top_logprobs``,
and ``parallel_tool_calls`` to avoid 400 errors.  Downgrade the
``developer`` role and fill ``content: null`` for safety.
"""

from ..model_utils import make_alcf_model_list_transform
from llm_rosetta.shims.transforms import (
    default_message_field,
    replace_message_field,
    strip_fields,
)

post_ir_transforms = (
    strip_fields("logprobs", "top_logprobs", "parallel_tool_calls"),
    replace_message_field("role", "developer", "system"),
    default_message_field("content", ""),
)
pre_ir_transforms = ()
ir_transforms = ()

model_list_transform = make_alcf_model_list_transform("api")
