"""Downstream SSE formatters — IR/provider chunks → SSE text for clients.

The gateway always speaks HTTP/SSE to downstream clients, regardless of
the upstream transport protocol.  These formatters produce the SSE text
lines that are written to the client response stream.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from llm_rosetta.formats import FORMAT_SPECS


# ---------------------------------------------------------------------------
# Per-provider SSE formatters
# ---------------------------------------------------------------------------


def _format_sse_chat_completions(chunk: dict[str, Any]) -> str:
    """Format a chunk as OpenAI Chat SSE line."""
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def format_sse_done() -> str:
    """Emit the OpenAI Chat ``[DONE]`` marker."""
    return "data: [DONE]\n\n"


def _format_sse_messages(chunk: dict[str, Any]) -> str:
    """Format a chunk as Anthropic SSE (``event: type\\ndata: json``)."""
    event_type = chunk.get("type", "unknown")
    return f"event: {event_type}\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _format_sse_open_responses(chunk: dict[str, Any]) -> str:
    """Format a chunk as OpenAI Responses SSE (``event: type\\ndata: json``)."""
    event_type = chunk.get("type", "unknown")
    return f"event: {event_type}\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _format_sse_google(chunk: dict[str, Any]) -> str:
    """Format a chunk as Google SSE line."""
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _format_sse_google_interactions(chunk: dict[str, Any]) -> str:
    """Format a chunk as Google Interactions SSE (``event: type\\ndata: json``)."""
    event_type = chunk.get("event_type", "unknown")
    return f"event: {event_type}\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"


_SSE_BY_KEY: dict[str, Any] = {
    "chat_completions": _format_sse_chat_completions,
    "open_responses": _format_sse_open_responses,
    "messages": _format_sse_messages,
    "google": _format_sse_google,
    "google_interactions": _format_sse_google_interactions,
}

# Derived from the format catalog.  The two legacy keys are historical route
# spellings kept until #919 removes them; they are not separate formats.
SSE_FORMATTERS: dict[str, Any] = {
    spec.format: _SSE_BY_KEY[spec.sse]
    for spec in FORMAT_SPECS
    if spec.model_type == "llm" and spec.sse is not None
}
SSE_FORMATTERS["openai_responses"] = _SSE_BY_KEY["open_responses"]
SSE_FORMATTERS["google"] = _SSE_BY_KEY["google"]


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
    code: str | None = None,
    request_id: str = "",
    status: int = 500,
    google_status: str = "INTERNAL",
    message: str | None = None,
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
        code: The upstream's machine-readable code, when it named one.  It
            replaces the generic ``server_error`` so the client can branch on
            it (a plan-usage 429 and a transient 503 recover differently).
        request_id: The upstream request id, used when no response id was
            advertised yet.
        status: The HTTP status this failure would have carried as a JSON
            response, for dialects whose in-band envelope repeats it.  A
            proxy's ``502`` ("the upstream produced something I cannot carry")
            and a gateway fault's ``500`` read differently to a client's retry
            logic, and before this parameter the Google envelope always said
            ``500 INTERNAL`` whatever the JSON leg said.
        google_status: The Google ``error.status`` that goes with *status*.
        message: The client-facing sentence.  Defaults to "the upstream stream
            ended before completion", which is right for a stream that died
            mid-flight; a converter that *refused* a complete turn is a
            different diagnosis and passes its own.

    Returns:
        Event dicts to format and yield. Empty for formats with no terminal
        event convention.
    """
    message = message or _TERMINAL_ERROR_MESSAGE.format(reason=reason)

    if source_provider in ("openai_responses", "open_responses"):
        response: dict[str, Any] = {
            "id": response_id or request_id,
            "object": "response",
            "status": "failed",
            "error": {"code": code or "server_error", "message": message},
        }
        event: dict[str, Any] = {"type": "response.failed", "response": response}
        if sequence_number is not None:
            event["sequence_number"] = sequence_number
        return [event]

    if source_provider == "chat_completions":
        # The in-band type mirrors the JSON envelope's classification rather
        # than a single generic word: a 502 says the upstream produced what the
        # proxy could not carry (``api_error``, as `error_response_for_source`
        # writes it), while a gateway fault keeps OpenAI's stream default.  A
        # client that retries on one and surfaces the other should not get a
        # different answer for asking with `stream: true`.
        return [
            {
                "error": {
                    "message": message,
                    "type": code or ("api_error" if status != 500 else "server_error"),
                    "code": code,
                }
            }
        ]

    if source_provider == "messages":
        return [
            {
                "type": "error",
                "error": {"type": code or "api_error", "message": message},
            }
        ]

    if source_provider in ("google", "google_generate"):
        return [
            {"error": {"code": status, "message": message, "status": google_status}}
        ]

    if source_provider == "google_interactions":
        return [
            {
                "event_type": "interaction.completed",
                "interaction": {
                    "id": response_id,
                    "status": "failed",
                    "errors": [{"message": message}],
                },
            }
        ]

    return []


# ---------------------------------------------------------------------------
# In-band upstream error detection
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UpstreamStreamError:
    """An error an upstream reported *inside* a 200 SSE stream.

    ``code`` is the provider's machine-readable code when it named one; two
    failures with the same HTTP status can need opposite recovery (a plan
    usage limit must not be retried, a transient 503 should be), so the code
    is carried to the client rather than flattened into a generic message.
    """

    message: str
    code: str | None = None
    request_id: str | None = None
    status: int | None = None


def classify_upstream_error(chunk: Any) -> UpstreamStreamError | None:
    """Return the error *chunk* carries, or ``None`` when it is content.

    Recognised shapes:

    * ``{"type": "response.failed", "response": {"error": {...}}}`` — the
      Responses terminal failure, which carries the code the client needs;
    * ``{"type": "error", "code": ..., "message": ...}`` — the Responses
      in-stream error event;
    * a top-level ``error`` envelope with no payload of its own, which is how
      most Chat-compatible upstreams report a rejected request.

    A chunk that carries content *and* an ``error`` field is left alone: that
    is a provider echoing something, not a failure.
    """
    if not isinstance(chunk, dict):
        return None

    chunk_type = chunk.get("type")

    if chunk_type == "response.failed":
        response = chunk.get("response")
        response = response if isinstance(response, dict) else {}
        err = response.get("error")
        err = err if isinstance(err, dict) else {}
        message = str(err.get("message") or err or "response failed")
        code = err.get("code")
        return UpstreamStreamError(
            message=message,
            code=str(code) if code else None,
            request_id=str(response.get("id") or chunk.get("response_id") or "")
            or None,
        )

    if chunk_type == "error":
        code = chunk.get("code")
        message = str(chunk.get("message") or code or "stream error")
        return UpstreamStreamError(
            message=message,
            code=str(code) if code else None,
            request_id=str(chunk.get("request_id") or "") or None,
        )

    err = chunk.get("error")
    if not err:
        return None
    if any(k in chunk for k in ("choices", "delta", "candidates", "type")):
        return None
    if isinstance(err, dict):
        code = err.get("code")
        return UpstreamStreamError(
            message=str(err.get("message") or err),
            code=str(code) if code else None,
        )
    return UpstreamStreamError(message=str(err))


def is_upstream_error_chunk(chunk: Any) -> bool:
    """True if *chunk* is an in-band upstream error rather than content."""
    return classify_upstream_error(chunk) is not None


def extract_upstream_error_message(chunk: dict[str, Any]) -> str:
    """Pull a human-readable message out of an upstream error envelope."""
    classified = classify_upstream_error(chunk)
    return classified.message if classified is not None else ""
