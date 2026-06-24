"""Downstream SSE formatters — IR/provider chunks → SSE text for clients.

The gateway always speaks HTTP/SSE to downstream clients, regardless of
the upstream transport protocol.  These formatters produce the SSE text
lines that are written to the client response stream.
"""

from __future__ import annotations

import json
from typing import Any


# ---------------------------------------------------------------------------
# Per-provider SSE formatters
# ---------------------------------------------------------------------------


def _format_sse_openai_chat(chunk: dict[str, Any]) -> str:
    """Format a chunk as OpenAI Chat SSE line."""
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def format_sse_done() -> str:
    """Emit the OpenAI Chat ``[DONE]`` marker."""
    return "data: [DONE]\n\n"


def _format_sse_anthropic(chunk: dict[str, Any]) -> str:
    """Format a chunk as Anthropic SSE (``event: type\\ndata: json``)."""
    event_type = chunk.get("type", "unknown")
    return f"event: {event_type}\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _format_sse_openai_responses(chunk: dict[str, Any]) -> str:
    """Format a chunk as OpenAI Responses SSE (``event: type\\ndata: json``)."""
    event_type = chunk.get("type", "unknown")
    return f"event: {event_type}\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _format_sse_google(chunk: dict[str, Any]) -> str:
    """Format a chunk as Google SSE line."""
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


SSE_FORMATTERS: dict[str, Any] = {
    "openai_chat": _format_sse_openai_chat,
    "openai_responses": _format_sse_openai_responses,
    "open_responses": _format_sse_openai_responses,
    "anthropic": _format_sse_anthropic,
    "google": _format_sse_google,
}
