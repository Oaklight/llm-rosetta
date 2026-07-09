"""Zhipu (GLM) schema transforms.

Zhipu does not support ``n``, ``presence_penalty``, ``frequency_penalty``,
``logprobs``, ``top_logprobs``, ``logit_bias``, or ``seed``.
Strip them before sending upstream to avoid errors.

Additionally, ``temperature`` range is [0, 1] (vs OpenAI's [0, 2]) and
``top_p`` minimum is 0.01.  These value-range adjustments are not handled
by simple field stripping and may need future transform support.

References:
    https://docs.bigmodel.cn
    https://docs.z.ai/api-reference/llm/chat-completion
"""

from llm_rosetta.shims.transforms import strip_fields

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
