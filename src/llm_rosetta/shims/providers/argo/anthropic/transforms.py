"""Argo Anthropic schema transforms.

Response-side (from_transforms)
--------------------------------
``_normalize_openai_response`` rewrites OpenAI Chat Completions format responses
to Anthropic Messages format.  Argo's ``/v1/messages`` endpoint inconsistently
returns ``choices[0].message`` for some Claude model versions; this transform
normalises those responses before the Anthropic converter sees them.

Request-side thinking normalization (``enabled`` ↔ ``adaptive`` per model) is
handled declaratively via ``reasoning.model_overrides`` in ``provider.yaml``
and the generic ``reasoning_helpers.py`` machinery.
"""

from __future__ import annotations

import copy
import json
from typing import Any

from llm_rosetta.shims.transforms import _NamedTransform

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Finish-reason mapping: OpenAI → Anthropic stop_reason.
_FINISH_REASON_MAP: dict[str, str] = {
    "stop": "end_turn",
    "length": "max_tokens",
    "tool_calls": "tool_use",
    "content_filter": "stop_sequence",
}

# ---------------------------------------------------------------------------
# Response-side transform
# ---------------------------------------------------------------------------


def _normalize_openai_response(body: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAI Chat Completions response to Anthropic Messages format.

    Argo's ``/v1/messages`` endpoint returns standard Anthropic format for most
    Claude models but falls back to OpenAI Chat format (``choices[0].message``)
    for some model versions.  This transform detects the OpenAI layout and
    rewrites it to Anthropic format so the Anthropic converter can parse it.

    Pass-through: responses that already have a top-level ``"content"`` list or
    ``"type": "message"`` field are returned unchanged.

    Args:
        body: Raw upstream JSON response dict.

    Returns:
        Anthropic-format response dict (possibly the same object if no
        conversion was needed).
    """
    # Already Anthropic format — nothing to do.
    if "content" in body or body.get("type") == "message":
        return body

    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        return body

    choice = choices[0]
    if not isinstance(choice, dict):
        return body

    message = choice.get("message")
    if not isinstance(message, dict):
        return body

    # Build Anthropic-format content from the OpenAI message.
    raw_content = message.get("content") or ""
    tool_calls = message.get("tool_calls") or []

    anthropic_content: list[dict[str, Any]] = []

    if raw_content:
        anthropic_content.append({"type": "text", "text": raw_content})

    for tc in tool_calls:
        if not isinstance(tc, dict):
            continue
        fn = tc.get("function", {})
        try:
            input_data = json.loads(fn.get("arguments", "{}"))
        except (ValueError, TypeError):
            input_data = {}
        anthropic_content.append(
            {
                "type": "tool_use",
                "id": tc.get("id", ""),
                "name": fn.get("name", ""),
                "input": input_data,
            }
        )

    # Map finish_reason → stop_reason.
    finish_reason = choice.get("finish_reason") or "stop"
    stop_reason = _FINISH_REASON_MAP.get(finish_reason, "end_turn")

    # Build usage block.
    oai_usage = body.get("usage") or {}
    usage: dict[str, Any] = {
        "input_tokens": oai_usage.get("prompt_tokens", 0),
        "output_tokens": oai_usage.get("completion_tokens", 0),
    }

    result = copy.copy(body)
    result.pop("choices", None)
    result.pop("object", None)
    result["type"] = "message"
    result["role"] = message.get("role", "assistant")
    result["content"] = anthropic_content
    result["stop_reason"] = stop_reason
    result["usage"] = usage
    return result


# ---------------------------------------------------------------------------
# Transform tuples (consumed by the shim loader)
# ---------------------------------------------------------------------------

to_transforms = ()
from_transforms = (
    _NamedTransform(_normalize_openai_response, "normalize_openai_response()"),
)
