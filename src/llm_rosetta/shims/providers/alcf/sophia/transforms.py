"""ALCF Sophia (vLLM on NVIDIA A100) schema transforms.

vLLM may not support ``logprobs`` / ``top_logprobs``.  Strip the
``developer`` role (not supported by open-source models) and replace
``content: null`` with an empty string to avoid vLLM crashes.

vLLM also names the reasoning text ``reasoning`` — it was renamed from
``reasoning_content`` — while the Chat Completions converter reads only
``reasoning_content``, so the field is renamed on the way in.
"""

from __future__ import annotations

from typing import Any

from ..model_utils import make_alcf_model_list_transform
from llm_rosetta.transforms import (
    _NamedTransform,
    default_tool_description,
    default_message_field,
    hoist_late_system_messages,
    replace_message_field,
    strip_fields,
)


def _rename_reasoning_field(body: dict[str, Any]) -> dict[str, Any]:
    """Rename ``reasoning`` → ``reasoning_content`` on messages and deltas.

    vLLM's reasoning field is ``reasoning``; the converter expects
    ``reasoning_content``, so without this the lane's reasoning is dropped,
    non-streaming and streaming alike (#961).

    Args:
        body: A chat completion response body or one streaming chunk.

    Returns:
        The same body, with the field renamed where present.
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


post_ir_transforms = (
    strip_fields("logprobs", "top_logprobs"),
    replace_message_field("role", "developer", "system"),
    default_message_field("content", ""),
    default_tool_description(),
)
pre_ir_transforms = (_NamedTransform(_rename_reasoning_field, "rename_reasoning()"),)
ir_transforms = (hoist_late_system_messages(),)

model_list_transform = make_alcf_model_list_transform("vllm")
