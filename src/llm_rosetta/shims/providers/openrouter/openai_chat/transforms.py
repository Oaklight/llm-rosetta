"""OpenRouter schema transforms.

Response-side: OpenRouter returns reasoning content in ``message.reasoning``
(string) instead of ``message.reasoning_content``. The converter expects
``reasoning_content``, so we rename the field before parsing.

Request-side: no transforms needed (OpenRouter accepts standard OpenAI Chat
fields plus ``reasoning_effort``).
"""

from __future__ import annotations

from typing import Any

from llm_rosetta.shims.transforms import _NamedTransform


def _rename_reasoning_field(body: dict[str, Any]) -> dict[str, Any]:
    """Rename ``message.reasoning`` → ``message.reasoning_content``.

    OpenRouter uses ``reasoning`` (not ``reasoning_content``) for the
    reasoning text field on chat completion response messages.  The
    OpenAI Chat converter expects ``reasoning_content``.

    Also copies ``reasoning_details`` as-is (the converter ignores it,
    but preserving it avoids silent data loss for downstream consumers).
    """
    choices = body.get("choices")
    if not isinstance(choices, list):
        return body
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        message = choice.get("message")
        if not isinstance(message, dict):
            continue
        # Rename reasoning → reasoning_content
        if "reasoning" in message and "reasoning_content" not in message:
            message["reasoning_content"] = message.pop("reasoning")
    return body


to_transforms = ()
from_transforms = (_NamedTransform(_rename_reasoning_field, "rename_reasoning()"),)
