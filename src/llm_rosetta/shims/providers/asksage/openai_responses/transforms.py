"""AskSage OpenAI Responses schema transforms.

Request-side:
- Rename ``max_tokens`` to ``max_completion_tokens`` (AskSage rejects
  ``max_tokens`` on its OpenAI-compatible endpoint).

References:
    https://docs.asksage.ai/api-documentation/api-endpoints/
"""

from llm_rosetta.transforms import rename_field

post_ir_transforms = (rename_field("max_tokens", "max_completion_tokens"),)
pre_ir_transforms = ()
ir_transforms = ()
