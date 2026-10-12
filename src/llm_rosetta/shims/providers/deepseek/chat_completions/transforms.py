"""DeepSeek schema transforms.

DeepSeek does not support the ``n``, ``logit_bias``, or ``seed`` fields
in chat completions.  ``frequency_penalty`` and ``presence_penalty`` are
deprecated and silently ignored.  Strip them all before sending upstream.

References:
    https://api-docs.deepseek.com/api/create-chat-completion
"""

from llm_rosetta.transforms import (
    backfill_reasoning_content,
    hoist_late_system_messages,
    strip_fields,
)

post_ir_transforms = (
    strip_fields("n", "logit_bias", "seed", "frequency_penalty", "presence_penalty"),
    # Thinking mode requires the reasoning to be replayed once tools are in
    # play: any request whose message list does not end on a ``user`` turn must
    # carry ``reasoning_content`` on every assistant message, tool call or not.
    # Clients that drop the reasoning when replaying history trip this; an
    # empty string is accepted.  See ``backfill_reasoning_content`` and #948.
    backfill_reasoning_content(),
)
pre_ir_transforms = ()
ir_transforms = (hoist_late_system_messages(),)
