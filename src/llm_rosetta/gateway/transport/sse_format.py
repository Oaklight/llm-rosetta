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


# ---------------------------------------------------------------------------
# Terminal error events
# ---------------------------------------------------------------------------

_TERMINAL_ERROR_MESSAGE = "Upstream stream ended before completion: {reason}"


def build_stream_error_events(
    source_provider: str,
    reason: str,
    *,
    response_id: str = "",
    sequence_number: int | None = None,
) -> list[dict[str, Any]]:
    """Build terminal events announcing that a stream failed mid-flight.

    Clients that wait for a format-specific terminal event (Codex waits for
    ``response.completed``) otherwise see only an abrupt socket close, which
    is indistinguishable from a network fault and hides the upstream reason.

    Args:
        source_provider: Client-facing format.
        reason: Short description of the upstream failure.
        response_id: Response ID already advertised to the client, if any.
        sequence_number: Next sequence number for Responses events.

    Returns:
        Event dicts to format and yield. Empty for formats with no terminal
        event convention.
    """
    message = _TERMINAL_ERROR_MESSAGE.format(reason=reason)

    if source_provider in ("openai_responses", "open_responses"):
        response: dict[str, Any] = {
            "id": response_id,
            "object": "response",
            "status": "failed",
            "error": {"code": "server_error", "message": message},
        }
        event: dict[str, Any] = {"type": "response.failed", "response": response}
        if sequence_number is not None:
            event["sequence_number"] = sequence_number
        return [event]

    if source_provider == "openai_chat":
        return [{"error": {"message": message, "type": "server_error", "code": None}}]

    if source_provider == "anthropic":
        return [{"type": "error", "error": {"type": "api_error", "message": message}}]

    if source_provider == "google":
        return [{"error": {"code": 500, "message": message, "status": "INTERNAL"}}]

    return []


# ---------------------------------------------------------------------------
# In-band upstream error detection
# ---------------------------------------------------------------------------


def is_upstream_error_chunk(chunk: Any) -> bool:
    """True if *chunk* is an in-band upstream error rather than content.

    Some upstreams report request errors inside a 200 SSE stream rather than
    by HTTP status.  Such a chunk converts to zero source events, leaving the
    client a successful but empty response.

    Only treats a chunk as an error when it carries no payload of its own, so
    a provider that legitimately ships an ``error`` field alongside content is
    left alone.
    """
    if not isinstance(chunk, dict) or not chunk.get("error"):
        return False
    return not any(k in chunk for k in ("choices", "delta", "candidates", "type"))


def extract_upstream_error_message(chunk: dict[str, Any]) -> str:
    """Pull a human-readable message out of an upstream error envelope."""
    err = chunk.get("error")
    if isinstance(err, dict):
        return str(err.get("message") or err)
    return str(err)
