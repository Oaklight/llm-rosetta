"""Argo OpenAI Chat schema transforms.

Request-side (to_transforms) — body-level
-------------------------------------------
- ``rename_field("max_tokens", "max_completion_tokens")``: converts the
  deprecated ``max_tokens`` parameter for newer OpenAI models.
- ``replace_message_field("role", "developer", "system")``: downgrades
  the ``developer`` role (OpenAI 2024-12-17+) to ``system`` for upstream
  gateways that don't support it.
- ``default_message_field("content", "")``: replaces ``content: null``
  with an empty string — upstream gateways (e.g. Argo Gemini) crash on
  null content when iterating message bodies.
- ``strip_fields_for_model(r"^claudeopus47", "temperature")``: strips
  ``temperature`` for reasoning models that reject it (e.g. Claude Opus 4.7).

Request-side (ir_transforms) — IR-level
-----------------------------------------
- ``truncate_images(50, pattern=r"^(gpt|o\\d)")``: enforce 50-image limit
  for GPT/o* models; Gemini and Claude pass through untouched.
- ``unwind_parallel_tool_calls(pattern=r"^gemini")``: split parallel tool
  calls into sequential pairs for Gemini models through Argo.
"""

from llm_rosetta.shims.transforms import (
    default_message_field,
    rename_field,
    replace_message_field,
    strip_fields_for_model,
    truncate_images,
    unwind_parallel_tool_calls,
)

to_transforms = (
    rename_field("max_tokens", "max_completion_tokens"),
    replace_message_field("role", "developer", "system"),
    default_message_field("content", ""),
    strip_fields_for_model(r"^claudeopus47", "temperature"),
)
from_transforms = ()
ir_transforms = (
    truncate_images(50, pattern=r"^(gpt|o\d)"),
    unwind_parallel_tool_calls(pattern=r"^gemini"),
)
