"""Argo Anthropic schema transforms.

Argo does not support ``thinking.type = "adaptive"`` — only
``"enabled"`` (with ``budget_tokens``) and ``"disabled"`` are accepted.
Convert ``"adaptive"`` to ``"enabled"`` with budget_tokens derived from
max_tokens so the constraint ``max_tokens > budget_tokens`` is always met.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta.shims.transforms import _NamedTransform

# Fraction of max_tokens to allocate as budget_tokens when converting
# "adaptive" → "enabled".  80% leaves 20% for the actual response.
_BUDGET_RATIO = 0.8


def _normalize_thinking(body: dict[str, Any]) -> dict[str, Any]:
    """Normalize thinking block for Argo compatibility.

    - ``"adaptive"`` → ``"enabled"`` with budget_tokens = 80% of
      max_tokens (Argo only accepts ``"enabled"`` or ``"disabled"``,
      and requires ``max_tokens > budget_tokens``)
    """
    thinking = body.get("thinking")
    if not isinstance(thinking, dict):
        return body
    if thinking.get("type") == "adaptive":
        thinking["type"] = "enabled"
        if "budget_tokens" not in thinking:
            max_tokens = body.get("max_tokens", 16384)
            thinking["budget_tokens"] = int(max_tokens * _BUDGET_RATIO)
    return body


to_transforms = (_NamedTransform(_normalize_thinking, "normalize_thinking()"),)
from_transforms = ()
