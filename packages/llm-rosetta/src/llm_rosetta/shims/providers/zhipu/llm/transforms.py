"""Zhipu (GLM) schema transforms.

Zhipu does not support ``n``, ``presence_penalty``, ``frequency_penalty``,
``logprobs``, ``top_logprobs``, ``logit_bias``, or ``seed``.
Strip them before sending upstream to avoid errors.

The value ranges this lane accepts are not a transform concern: they are
declared in ``provider.yaml`` under ``request.ranges`` (``temperature`` is
[0, 1] versus OpenAI's [0, 2], and ``top_p`` has a 0.01 minimum), and the
conversion applies them with the change reported under the IR path.

References:
    https://docs.bigmodel.cn
    https://docs.z.ai/api-reference/llm/chat-completion
"""

from llm_rosetta.transforms import hoist_late_system_messages, strip_fields

post_ir_transforms = (
    strip_fields(
        "n",
        "presence_penalty",
        "frequency_penalty",
        "logprobs",
        "top_logprobs",
        "logit_bias",
        "seed",
    ),
)
pre_ir_transforms = ()
ir_transforms = (hoist_late_system_messages(),)
