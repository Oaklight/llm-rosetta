"""OpenRouter schema transforms.

Response-side: OpenRouter returns reasoning content in ``message.reasoning``
(string) instead of ``message.reasoning_content``. The converter expects
``reasoning_content``, so we rename the field before parsing.

Request-side: no transforms needed (OpenRouter accepts standard OpenAI Chat
fields plus ``reasoning_effort``).
"""

from __future__ import annotations

from typing import Any

from llm_rosetta.transforms import _NamedTransform, hoist_late_system_messages


def _rename_reasoning_field(body: dict[str, Any]) -> dict[str, Any]:
    """Rename ``reasoning`` → ``reasoning_content`` on messages and deltas.

    OpenRouter uses ``reasoning`` (not ``reasoning_content``) for the
    reasoning text field.  The OpenAI Chat converter expects
    ``reasoning_content``.

    Both holders matter: the streaming path routes chunks through this
    helper, and a chunk carries the text on ``delta`` rather than on
    ``message`` — renaming only the message half dropped streamed reasoning
    silently (found in review of #969; ``alcf--sophia`` has the same shape).

    ``reasoning_details`` is untouched here: the converter preserves it
    through ``provider_metadata`` (see ``chat_completions/message_ops.py``).
    """
    choices = body.get("choices")
    if not isinstance(choices, list):
        return body
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        for holder_key in ("message", "delta"):
            holder = choice.get(holder_key)
            if not isinstance(holder, dict):
                continue
            # Rename reasoning → reasoning_content
            if "reasoning" in holder and "reasoning_content" not in holder:
                holder["reasoning_content"] = holder.pop("reasoning")
    return body


post_ir_transforms = ()
pre_ir_transforms = (_NamedTransform(_rename_reasoning_field, "rename_reasoning()"),)
ir_transforms = (hoist_late_system_messages(),)
