"""ALCF Minerva (NVIDIA B200) schema transforms.

Minerva hosts nemotron-3-ultra and inkling-bf16.  Strip unsupported
fields and downgrade the ``developer`` role.

Note: inkling-bf16 has an intermittent (~35%) tool-call bug where it
emits ``<|content_invoke_tool_json|>`` as plain text instead of a
structured ``tool_calls`` field.  This is a response-side issue that
cannot be addressed by request transforms; see the ATPI toolfix proxy
or the planned gateway response-transform pipeline for a fix.
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

model_list_transform = make_alcf_model_list_transform("api")
