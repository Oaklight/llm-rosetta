"""MiniMax schema transforms.

Request-side:
- Strip unsupported fields (logprobs, top_logprobs, seed, stop).
- Inject ``reasoning_split: true`` when a ``thinking`` object is present,
  so MiniMax returns ``reasoning_content`` as a separate field instead of
  embedding ``<think>`` tags in ``content``.

Response-side:
- Parse ``<think>...</think>`` tags from ``content`` into
  ``reasoning_content`` when ``reasoning_split`` was not set (fallback).

References:
    https://platform.minimaxi.com/document/ChatCompletion%20v2
"""

from __future__ import annotations

import re
from typing import Any

from llm_rosetta.shims.transforms import _NamedTransform, strip_fields


def _inject_reasoning_split(body: dict[str, Any]) -> dict[str, Any]:
    """Inject ``reasoning_split: true`` when thinking is requested.

    MiniMax's OpenAI Chat endpoint defaults to embedding thinking in
    ``<think>`` tags within ``content``.  Setting ``reasoning_split: true``
    makes it return ``reasoning_content`` as a separate field, which the
    converter can parse directly.
    """
    if "thinking" in body and "reasoning_split" not in body:
        body["reasoning_split"] = True
    return body


_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def _parse_think_tags(body: dict[str, Any]) -> dict[str, Any]:
    """Extract ``<think>`` tags from content into ``reasoning_content``.

    Fallback for responses where ``reasoning_split`` was not set.
    If ``reasoning_content`` already exists, this is a no-op.
    Handles multiple ``<think>`` blocks by joining them with newlines.
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
        if "reasoning_content" in message:
            continue  # already split
        content = message.get("content")
        if not isinstance(content, str):
            continue
        matches = _THINK_RE.findall(content)
        if matches:
            message["reasoning_content"] = "\n".join(m.strip() for m in matches)
            message["content"] = _THINK_RE.sub("", content).strip()
    return body


to_transforms = (
    strip_fields("logprobs", "top_logprobs", "seed", "stop"),
    _NamedTransform(_inject_reasoning_split, "inject_reasoning_split()"),
)
from_transforms = (_NamedTransform(_parse_think_tags, "parse_think_tags()"),)
