"""Proxy engine — upstream request building, SSE handling, and response conversion.

This module contains the core proxy logic extracted from ``app.py``:
- Upstream request preparation (including Google body fixups)
- SSE parsing and formatting
- Provider metadata caching (e.g. Google ``thought_signature``)
- HTTP client pool management
- Non-streaming and streaming request handlers
- Error response helpers
- Request body helpers
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from typing import Any

import httpx
from starlette.responses import JSONResponse, Response, StreamingResponse

from llm_rosetta import get_converter_for_provider
from llm_rosetta.auto_detect import ProviderType
from llm_rosetta.converters.base.context import ConversionContext


from .logging import (
    get_logger,
    log_converted_request,
    log_original_request,
    log_response,
    log_stream_summary,
    log_upstream_error,
)
from .providers import ProviderInfo

logger = get_logger()

# ---------------------------------------------------------------------------
# Upstream request building
# ---------------------------------------------------------------------------


def prepare_upstream(
    target_provider: ProviderType,
    provider_info: ProviderInfo,
    provider_request: dict[str, Any],
    model: str,
    *,
    stream: bool,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Return (url, headers, body) ready for the upstream HTTP call."""
    url = provider_info.upstream_url(model, stream=stream)
    headers = {
        "Content-Type": "application/json",
        **provider_info.auth_headers(),
    }

    body = dict(provider_request)

    # Inject stream flag into the body for providers that use it
    if stream:
        if target_provider in ("openai_chat",):
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        elif target_provider in ("openai_responses", "anthropic"):
            body["stream"] = True
        # Google streaming is signaled via URL, not body

    return url, headers, body


# ---------------------------------------------------------------------------
# SSE parsing (upstream → IR events)
# ---------------------------------------------------------------------------


def _iter_sse_lines(line: str) -> tuple[str | None, str | None] | None:
    """Parse a single SSE line into (field, value) or None if not relevant.

    Returns:
        ("data", <value>)  for data lines
        ("event", <value>) for event lines
        None               for empty/irrelevant lines
    """
    if not line:
        return None
    if line.startswith("data: "):
        return ("data", line[6:])
    if line.startswith("event: "):
        return ("event", line[7:])
    return None


def _is_openai_done(data: str) -> bool:
    """Check if the SSE data payload signals end-of-stream (OpenAI [DONE])."""
    return data.strip() == "[DONE]"


# ---------------------------------------------------------------------------
# SSE emission (IR events → source-format SSE text)
# ---------------------------------------------------------------------------


def _format_sse_openai_chat(chunk: dict[str, Any]) -> str:
    """Format a chunk as OpenAI Chat SSE line."""
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _format_sse_openai_chat_done() -> str:
    return "data: [DONE]\n\n"


def _format_sse_anthropic(chunk: dict[str, Any]) -> str:
    """Format a chunk as Anthropic SSE (event: type\\ndata: json)."""
    event_type = chunk.get("type", "unknown")
    return f"event: {event_type}\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _format_sse_openai_responses(chunk: dict[str, Any]) -> str:
    """Format a chunk as OpenAI Responses SSE (event: type\\ndata: json)."""
    event_type = chunk.get("type", "unknown")
    return f"event: {event_type}\ndata: {json.dumps(chunk, ensure_ascii=False)}\n\n"


def _format_sse_google(chunk: dict[str, Any]) -> str:
    """Format a chunk as Google SSE line."""
    return f"data: {json.dumps(chunk, ensure_ascii=False)}\n\n"


SSE_FORMATTERS: dict[str, Any] = {
    "openai_chat": _format_sse_openai_chat,
    "openai_responses": _format_sse_openai_responses,
    "anthropic": _format_sse_anthropic,
    "google": _format_sse_google,
}


# ---------------------------------------------------------------------------
# Error helpers
# ---------------------------------------------------------------------------


def error_response_for_source(
    source_provider: ProviderType, status_code: int, message: str
) -> Response:
    """Return an error response formatted for the source provider's envelope."""
    if source_provider == "openai_chat":
        body = {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": None,
            }
        }
    elif source_provider == "openai_responses":
        body = {
            "error": {
                "message": message,
                "type": "invalid_request_error",
                "code": None,
            }
        }
    elif source_provider == "anthropic":
        body = {
            "type": "error",
            "error": {"type": "invalid_request_error", "message": message},
        }
    elif source_provider == "google":
        body = {
            "error": {
                "code": status_code,
                "message": message,
                "status": "INVALID_ARGUMENT",
            }
        }
    else:
        body = {"error": {"message": message}}

    return JSONResponse(body, status_code=status_code)


# ---------------------------------------------------------------------------
# Request body helpers
# ---------------------------------------------------------------------------


def detect_stream_request(source_provider: ProviderType, body: dict[str, Any]) -> bool:
    """Detect if the incoming request asks for streaming."""
    if source_provider in ("openai_chat", "openai_responses", "anthropic"):
        return bool(body.get("stream", False))
    # Google streaming is determined by the endpoint path, not the body
    return False


def extract_model(source_provider: ProviderType, body: dict[str, Any]) -> str | None:
    """Extract the model name from a source-format request body."""
    return body.get("model")


# ---------------------------------------------------------------------------
# HTTP client pool
# ---------------------------------------------------------------------------

# Shared httpx clients keyed by proxy URL (None = direct connection)
_http_clients: dict[str | None, httpx.AsyncClient] = {}


def get_client(proxy_url: str | None = None) -> httpx.AsyncClient:
    """Get or create an ``httpx.AsyncClient`` for the given proxy URL."""
    if proxy_url not in _http_clients:
        _http_clients[proxy_url] = httpx.AsyncClient(
            timeout=300.0,
            proxy=proxy_url,
        )
    return _http_clients[proxy_url]


async def close_clients() -> None:
    """Close all pooled HTTP clients (called on app shutdown)."""
    for client in _http_clients.values():
        await client.aclose()
    _http_clients.clear()


# ---------------------------------------------------------------------------
# Provider metadata cache (e.g. Google thought_signature)
# ---------------------------------------------------------------------------
# Maps tool_call_id → provider_metadata dict.  Populated when we receive
# tool-call responses from upstream; consumed when the follow-up request
# contains the corresponding tool results.  Short-lived: entries are
# deleted once consumed.

_provider_metadata_cache: dict[str, dict[str, Any]] = {}


def _cache_provider_metadata(ir_response: dict[str, Any]) -> None:
    """Extract provider_metadata from tool calls in an IR response and cache it."""
    for choice in ir_response.get("choices", []):
        msg = choice.get("message", {})
        for part in msg.get("content", []):
            if part.get("type") == "tool_call" and "provider_metadata" in part:
                tool_call_id = part.get("tool_call_id")
                if tool_call_id:
                    _provider_metadata_cache[tool_call_id] = part["provider_metadata"]
                    logger.debug(
                        "Cached provider_metadata for tool_call %s", tool_call_id
                    )


def _inject_provider_metadata(ir_request: dict[str, Any]) -> None:
    """Inject cached provider_metadata into tool call parts in an IR request.

    Clients send the full conversation history on every request, so the same
    tool_call_id may appear in multiple requests.  We use ``get()`` instead of
    ``pop()`` to keep entries alive for subsequent turns.
    """
    logger.debug(
        "_inject: cache has %d entries: %s",
        len(_provider_metadata_cache),
        list(_provider_metadata_cache.keys()),
    )
    for msg in ir_request.get("messages", []):
        for part in msg.get("content", []):
            if part.get("type") == "tool_call":
                tool_call_id = part.get("tool_call_id")
                if tool_call_id and tool_call_id in _provider_metadata_cache:
                    part["provider_metadata"] = _provider_metadata_cache[tool_call_id]


# ---------------------------------------------------------------------------
# Core proxy handlers
# ---------------------------------------------------------------------------


async def handle_non_streaming(
    source_provider: ProviderType,
    target_provider: ProviderType,
    provider_info: ProviderInfo,
    body: dict[str, Any],
    model: str,
) -> Response:
    """Non-streaming proxy: convert -> forward -> convert back -> respond."""
    source_converter = get_converter_for_provider(source_provider)
    target_converter = get_converter_for_provider(target_provider)

    # Shared context for the conversion pipeline
    ctx = ConversionContext()
    if target_provider == "google":
        ctx.options["output_format"] = "rest"

    # 1. Source -> IR
    try:
        ir_request = source_converter.request_from_provider(body, context=ctx)
    except Exception as exc:
        return error_response_for_source(
            source_provider, 400, f"Failed to parse request: {exc}"
        )

    # 1b. Restore cached provider_metadata (e.g. Google thought_signature)
    _inject_provider_metadata(ir_request)

    # -- body log: IR request (after source -> IR) --
    log_original_request(ir_request)

    # 2. IR -> Target
    try:
        target_body, warnings = target_converter.request_to_provider(
            ir_request, context=ctx
        )
    except Exception as exc:
        return error_response_for_source(
            source_provider, 400, f"Conversion error: {exc}"
        )
    if warnings:
        logger.warning("Conversion warnings: %s", warnings)

    # 3. Build upstream request
    url, headers, upstream_body = prepare_upstream(
        target_provider, provider_info, target_body, model, stream=False
    )

    # -- body log: target request body --
    log_converted_request(upstream_body)

    # 4. Forward to upstream
    client = get_client(provider_info.proxy_url)
    try:
        upstream_resp = await client.post(url, json=upstream_body, headers=headers)
    except httpx.HTTPError as exc:
        return error_response_for_source(
            source_provider, 502, f"Upstream request failed: {exc}"
        )

    # 5. Pass through upstream errors
    if upstream_resp.status_code >= 400:
        log_upstream_error(
            upstream_resp.status_code,
            upstream_resp.text,
            endpoint=str(target_provider),
        )
        return Response(
            content=upstream_resp.content,
            status_code=upstream_resp.status_code,
            media_type="application/json",
        )

    # 6. Target response -> IR
    try:
        upstream_json = upstream_resp.json()
        ir_response = target_converter.response_from_provider(
            upstream_json, context=ctx
        )
    except Exception as exc:
        return error_response_for_source(
            source_provider, 502, f"Failed to parse upstream response: {exc}"
        )

    # -- body log: upstream response --
    log_response(upstream_json, label="UPSTREAM RESPONSE")

    # 6b. Cache provider_metadata from tool calls for follow-up requests
    _cache_provider_metadata(ir_response)

    # 7. IR -> Source response
    try:
        source_response = source_converter.response_to_provider(
            ir_response, context=ctx
        )
    except Exception as exc:
        return error_response_for_source(
            source_provider, 500, f"Failed to convert response: {exc}"
        )

    return JSONResponse(source_response)


async def handle_streaming(
    source_provider: ProviderType,
    target_provider: ProviderType,
    provider_info: ProviderInfo,
    body: dict[str, Any],
    model: str,
) -> Response:
    """Streaming proxy: convert -> forward -> stream-convert back -> SSE."""
    source_converter = get_converter_for_provider(source_provider)
    target_converter = get_converter_for_provider(target_provider)

    # Shared context for the request conversion phase
    ctx = ConversionContext()
    if target_provider == "google":
        ctx.options["output_format"] = "rest"

    # 1. Source -> IR
    try:
        ir_request = source_converter.request_from_provider(body, context=ctx)
    except Exception as exc:
        return error_response_for_source(
            source_provider, 400, f"Failed to parse request: {exc}"
        )

    # 1b. Inject cached provider_metadata (e.g. Google thought_signature)
    _inject_provider_metadata(ir_request)

    # -- body log: IR request (after source -> IR) --
    log_original_request(ir_request)

    # 2. IR -> Target
    try:
        target_body, warnings = target_converter.request_to_provider(
            ir_request, context=ctx
        )
    except Exception as exc:
        return error_response_for_source(
            source_provider, 400, f"Conversion error: {exc}"
        )
    if warnings:
        logger.warning("Conversion warnings: %s", warnings)

    # 3. Build upstream request (with stream=True)
    url, headers, upstream_body = prepare_upstream(
        target_provider, provider_info, target_body, model, stream=True
    )

    # -- body log: target request body --
    log_converted_request(upstream_body)

    format_sse = SSE_FORMATTERS[source_provider]

    async def event_generator() -> AsyncIterator[str]:
        """Stream SSE events from upstream, converting each chunk."""
        from_ctx = target_converter.create_stream_context()  # upstream -> IR
        to_ctx = source_converter.create_stream_context()  # IR -> source
        chunk_count = 0
        t0 = time.monotonic()

        client = get_client(provider_info.proxy_url)
        async with client.stream(
            "POST", url, json=upstream_body, headers=headers
        ) as upstream_resp:
            if upstream_resp.status_code >= 400:
                # Read error body and yield as error
                await upstream_resp.aread()
                error_text = upstream_resp.text
                log_upstream_error(
                    upstream_resp.status_code,
                    error_text,
                    endpoint=str(target_provider),
                    is_streaming=True,
                )
                try:
                    error_body = json.loads(error_text)
                    error_msg = json.dumps(error_body)
                except json.JSONDecodeError:
                    error_msg = error_text
                yield f"data: {error_msg}\n\n"
                return

            async for line in upstream_resp.aiter_lines():
                parsed = _iter_sse_lines(line)
                if parsed is None:
                    continue
                field, value = parsed

                # Skip event-type lines (type info is inside the data JSON)
                if field == "event":
                    continue

                if field != "data" or value is None:
                    continue

                # OpenAI [DONE] signal
                if _is_openai_done(value):
                    break

                try:
                    chunk = json.loads(value)
                except json.JSONDecodeError:
                    logger.warning("Skipping malformed SSE data: %s", value[:200])
                    continue

                chunk_count += 1

                # Upstream chunk -> IR events
                ir_events = target_converter.stream_response_from_provider(
                    chunk, context=from_ctx
                )

                # IR events -> source-format chunks
                for ir_event in ir_events:
                    # Cache provider_metadata from tool_call_start events
                    if (
                        ir_event.get("type") == "tool_call_start"
                        and "provider_metadata" in ir_event
                    ):
                        _provider_metadata_cache[ir_event["tool_call_id"]] = ir_event[
                            "provider_metadata"
                        ]

                    source_chunks = source_converter.stream_response_to_provider(
                        ir_event, context=to_ctx
                    )
                    if isinstance(source_chunks, list):
                        for sc in source_chunks:
                            if sc:
                                yield format_sse(sc)
                    elif source_chunks:
                        yield format_sse(source_chunks)

        # Emit end-of-stream marker for OpenAI Chat format
        if source_provider == "openai_chat":
            yield _format_sse_openai_chat_done()

        # -- stream summary (no per-chunk spam) --
        log_stream_summary(
            model=model,
            duration_s=time.monotonic() - t0,
            chunk_count=chunk_count,
        )

    return StreamingResponse(event_generator(), media_type="text/event-stream")
