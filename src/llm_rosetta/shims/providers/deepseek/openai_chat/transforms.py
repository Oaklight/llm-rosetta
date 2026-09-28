"""DeepSeek schema transforms.

DeepSeek does not support the ``n``, ``logit_bias``, or ``seed`` fields
in chat completions.  ``frequency_penalty`` and ``presence_penalty`` are
deprecated and silently ignored.  Strip them all before sending upstream.

References:
    https://api-docs.deepseek.com/api/create-chat-completion
"""

from llm_rosetta.shims.transforms import hoist_late_system_messages, strip_fields

post_ir_transforms = (
    strip_fields("n", "logit_bias", "seed", "frequency_penalty", "presence_penalty"),
)
pre_ir_transforms = ()
ir_transforms = (hoist_late_system_messages(),)
