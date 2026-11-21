"""Groq schema transforms.

Groq serves its reasoning text on ``message.reasoning`` / ``delta.reasoning``
for the ``openai/gpt-oss-*`` models, while the Chat Completions converter
reads ``reasoning_content``.  The rename is a no-op on a response that
carries no reasoning at all, so it is safe for the non-reasoning models this
lane also serves.

The field name follows Groq's documented gpt-oss response shape; it has not
been probed live because no key was available when the shim landed.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta.contracts.transform import NamedTransform, hoist_late_system_messages


def _rename_reasoning_field(body: dict[str, Any]) -> dict[str, Any]:
    """Rename ``reasoning`` → ``reasoning_content`` on messages and deltas.

    Both holders matter: the streaming path routes chunks through this
    helper, and a chunk carries the text on ``delta`` rather than on
    ``message``.
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
            if "reasoning" in holder and "reasoning_content" not in holder:
                holder["reasoning_content"] = holder.pop("reasoning")
    return body


post_ir_transforms = ()
pre_ir_transforms = (NamedTransform(_rename_reasoning_field, "rename_reasoning()"),)
ir_transforms = (hoist_late_system_messages(),)
