"""Argo OpenAI Responses schema transforms.

Request-side (post_ir_transforms) — body-level
-----------------------------------------------
- ``strip_fields_for_model(..., "temperature", "top_p", "top_k")``: strips
  sampling params for models that reject them on the Responses endpoint.
  Covers o-series and GPT-5 base/5.5/5.6 (same models as Chat, plus o-series
  which accepts sampling on Chat but rejects it on Responses).

The Responses path is otherwise simpler than Chat — role downgrade, content
defaulting, and system flattening do not apply.
"""

from llm_rosetta.transforms import strip_fields_for_model

from ..model_utils import model_list_transform  # noqa: F401

post_ir_transforms = (
    strip_fields_for_model(
        r"^(gpto|gpt5(nano|mini)?$|gpt5[56])",
        "temperature",
        "top_p",
        "top_k",
    ),
)
pre_ir_transforms = ()
ir_transforms = ()
