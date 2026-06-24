"""SSE (Server-Sent Events) parsing and formatting.

Handles two directions:

* **Parsing** — upstream SSE lines → structured data (``parse_sse_line``,
  ``parse_sse_data``, ``is_openai_done``).
* **Formatting** — IR/provider chunks → downstream SSE text, keyed by
  provider type in ``SSE_FORMATTERS``.
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger("llm-rosetta-gateway")

# Sentinel returned by :func:`parse_sse_data` when the stream signals
# completion (e.g. OpenAI ``[DONE]``).
SENTINEL_DONE = object()


# ---------------------------------------------------------------------------
# SSE parsing (upstream → structured data)
# ---------------------------------------------------------------------------


def parse_sse_line(line: str) -> tuple[str, str] | None:
    """Parse a single SSE line into ``(field, value)`` or ``None``.

    Returns:
        ``("data", <value>)`` for data lines, ``("event", <value>)`` for
        event lines, or ``None`` for empty / irrelevant lines.
    """
    if not line:
        return None
    if line.startswith("data: "):
        return ("data", line[6:])
    if line.startswith("event: "):
        return ("event", line[7:])
    return None


def is_openai_done(data: str) -> bool:
    """Check if the SSE data payload signals end-of-stream (OpenAI ``[DONE]``)."""
    return data.strip() == "[DONE]"


def parse_sse_data(line: str) -> Any:
    """Parse a single SSE line and return the JSON chunk, or ``None`` to skip.

    Returns :data:`SENTINEL_DONE` when the stream signals completion.
    """
    parsed = parse_sse_line(line)
    if parsed is None:
        return None
    field, value = parsed
    if field == "event" or field != "data" or value is None:
        return None
    if is_openai_done(value):
        return SENTINEL_DONE
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        logger.warning("Skipping malformed SSE data: %s", value[:200])
        return None


# ---------------------------------------------------------------------------
# SSE formatting (IR events → source-format SSE text)
# ---------------------------------------------------------------------------


def _format_sse_openai_chat(chunk: dict[str, Any]) -> str:
    """Format a chunk as OpenAI Chat SSE line."""
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _format_sse_openai_chat_done() -> str:
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
