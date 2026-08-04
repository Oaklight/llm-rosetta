"""OpenAI Chat converter constants — reason mappings and tool content packing."""

from ..base.helpers.multimodal_tool_patch import (  # noqa: F401  # re-export
    TOOL_CONTENT_CLOSE_TAG,
    TOOL_CONTENT_OPEN_TAG_RE,
)

# --- Reason mappings ---

OPENAI_CHAT_REASON_FROM_PROVIDER: dict[str, str] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "content_filter": "content_filter",
    "function_call": "tool_calls",
}

# Identity mapping — IR reasons are OpenAI Chat reasons.
# Kept for structural symmetry with other providers and to document
# the valid set of OpenAI Chat finish_reason values.
OPENAI_CHAT_REASON_TO_PROVIDER: dict[str, str] = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "content_filter": "content_filter",
    "refusal": "stop",
}
