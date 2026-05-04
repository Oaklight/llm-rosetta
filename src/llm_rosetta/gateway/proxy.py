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
from dataclasses import dataclass, field
from typing import Any

from llm_rosetta._vendor.httpclient import (
    AsyncClient,
    HttpClientError,
    Response as HttpResponse,
    StreamingResponse as HttpStreamingResponse,
)
from llm_rosetta._vendor.httpserver import JSONResponse, Response, StreamingResponse

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
    extra_headers: dict[str, str] | None = None,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Return (url, headers, body) ready for the upstream HTTP call."""
    url = provider_info.upstream_url(model, stream=stream)
    headers = {
        "Content-Type": "application/json",
        **provider_info.auth_headers(),
    }
    if extra_headers:
        headers.update(extra_headers)

    body = dict(provider_request)

    # Inject stream flag into the body for providers that use it
    if stream:
        if target_provider in ("openai_chat",):
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        elif target_provider in ("openai_responses", "open_responses", "anthropic"):
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
    "open_responses": _format_sse_openai_responses,
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
    elif source_provider in ("openai_responses", "open_responses"):
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
    if source_provider in (
        "openai_chat",
        "openai_responses",
        "open_responses",
        "anthropic",
    ):
        return bool(body.get("stream", False))
    # Google streaming is determined by the endpoint path, not the body
    return False


def extract_model(source_provider: ProviderType, body: dict[str, Any]) -> str | None:
    """Extract the model name from a source-format request body."""
    return body.get("model")


# ---------------------------------------------------------------------------
# HTTP client pool
# ---------------------------------------------------------------------------

# Shared HTTP clients keyed by proxy URL (None = direct connection)
_http_clients: dict[str | None, AsyncClient] = {}


def get_client(proxy_url: str | None = None) -> AsyncClient:
    """Get or create an ``AsyncClient`` for the given proxy URL."""
    if proxy_url not in _http_clients:
        _http_clients[proxy_url] = AsyncClient(
            timeout=300.0,
            proxy=proxy_url,
        )
    return _http_clients[proxy_url]


async def close_resources(
    *, metadata_store: ProviderMetadataStore | None = None
) -> None:
    """Close all pooled HTTP clients and clear metadata store (called on app shutdown)."""
    for client in _http_clients.values():
        await client.aclose()
    _http_clients.clear()
    store = metadata_store or _default_metadata_store
    store.clear()


# ---------------------------------------------------------------------------
# Provider metadata store (e.g. Google thought_signature)
# ---------------------------------------------------------------------------
# Bridges provider_metadata across HTTP request boundaries.  Request 1's
# response may contain a ``thought_signature`` that must be injected into
# Request 2's tool result.  Entries are keyed by ``tool_call_id`` and are
# kept alive (``get``, not ``pop``) because clients resend the full
# conversation history on every request.


@dataclass
class _CacheEntry:
    """A single cached provider_metadata entry with creation timestamp."""

    data: dict[str, Any]
    created: float = field(default_factory=time.monotonic)


class ProviderMetadataStore:
    """Stores provider_metadata across request boundaries with TTL and bounds.

    Args:
        ttl: Time-to-live in seconds for each entry.  Defaults to 30 minutes.
        max_size: Maximum number of entries.  Oldest is evicted on overflow.
    """

    def __init__(self, *, ttl: float = 1800.0, max_size: int = 10_000) -> None:
        self._store: dict[str, _CacheEntry] = {}
        self._ttl = ttl
        self._max_size = max_size

    def _evict_expired(self) -> None:
        now = time.monotonic()
        expired = [k for k, e in self._store.items() if now - e.created > self._ttl]
        for k in expired:
            del self._store[k]

    def _evict_oldest(self) -> None:
        if len(self._store) >= self._max_size:
            oldest_key = min(self._store, key=lambda k: self._store[k].created)
            del self._store[oldest_key]

    def cache_from_response(self, ir_response: dict[str, Any]) -> None:
        """Extract and cache provider_metadata from tool calls in an IR response."""
        self._evict_expired()
        for choice in ir_response.get("choices", []):
            msg = choice.get("message", {})
            for part in msg.get("content", []):
                if part.get("type") == "tool_call" and "provider_metadata" in part:
                    tool_call_id = part.get("tool_call_id")
                    if tool_call_id:
                        self._evict_oldest()
                        self._store[tool_call_id] = _CacheEntry(
                            data=part["provider_metadata"],
                        )
                        logger.debug(
                            "Cached provider_metadata for tool_call %s", tool_call_id
                        )

    def cache_from_stream_event(self, ir_event: dict[str, Any]) -> None:
        """Cache provider_metadata from a tool_call_start stream event."""
        if (
            ir_event.get("type") == "tool_call_start"
            and "provider_metadata" in ir_event
        ):
            self._evict_expired()
            self._evict_oldest()
            self._store[ir_event["tool_call_id"]] = _CacheEntry(
                data=ir_event["provider_metadata"],
            )

    def inject_into_request(self, ir_request: dict[str, Any]) -> None:
        """Inject cached provider_metadata into tool call parts in an IR request.

        Clients send the full conversation history on every request, so the
        same tool_call_id may appear in multiple requests.  Entries are kept
        alive (not popped) for subsequent turns.
        """
        self._evict_expired()
        logger.debug(
            "inject: store has %d entries: %s",
            len(self._store),
            list(self._store.keys()),
        )
        for msg in ir_request.get("messages", []):
            for part in msg.get("content", []):
                if part.get("type") == "tool_call":
                    tool_call_id = part.get("tool_call_id")
                    if tool_call_id and tool_call_id in self._store:
                        part["provider_metadata"] = self._store[tool_call_id].data

    def clear(self) -> None:
        """Remove all entries."""
        self._store.clear()

    def __len__(self) -> int:
        return len(self._store)


_default_metadata_store = ProviderMetadataStore()


# ---------------------------------------------------------------------------
# Core proxy handlers
# ---------------------------------------------------------------------------


async def handle_non_streaming(
    source_provider: ProviderType,
    target_provider: ProviderType,
    provider_info: ProviderInfo,
    body: dict[str, Any],
    model: str,
    *,
    metadata_store: ProviderMetadataStore | None = None,
    extra_headers: dict[str, str] | None = None,
) -> Response:
    """Non-streaming proxy: convert -> forward -> convert back -> respond."""
    store = metadata_store or _default_metadata_store
    source_converter = get_converter_for_provider(source_provider)
    target_converter = get_converter_for_provider(target_provider)

    # Shared context for the conversion pipeline
    ctx = ConversionContext()
    ctx.options["metadata_mode"] = "preserve"
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
    store.inject_into_request(ir_request)

    # -- body log: IR request (after source -> IR) --
    log_original_request(ir_request)

    # 2. IR -> Target
    try:
        target_body, _ = target_converter.request_to_provider(ir_request, context=ctx)
    except Exception as exc:
        return error_response_for_source(
            source_provider, 400, f"Conversion error: {exc}"
        )
    if ctx.warnings:
        logger.warning("Conversion warnings: %s", ctx.warnings)

    # 3. Build upstream request
    url, headers, upstream_body = prepare_upstream(
        target_provider,
        provider_info,
        target_body,
        model,
        stream=False,
        extra_headers=extra_headers,
    )

    # -- body log: target request body --
    log_converted_request(upstream_body)

    # 4. Forward to upstream
    client = get_client(provider_info.proxy_url)
    try:
        upstream_resp = await client.post(url, json=upstream_body, headers=headers)
    except HttpClientError as exc:
        return error_response_for_source(
            source_provider, 502, f"Upstream request failed: {exc}"
        )
    assert isinstance(upstream_resp, HttpResponse)

    # 5. Pass through upstream errors
    if upstream_resp.status_code >= 400:
        log_upstream_error(
            upstream_resp.status_code,
            upstream_resp.text,
            endpoint=str(target_provider),
        )
        return Response(
            body=upstream_resp.content,
            status_code=upstream_resp.status_code,
            content_type="application/json",
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
    store.cache_from_response(ir_response)

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


_SENTINEL_DONE = object()


def _parse_sse_data(line: str) -> Any:
    """Parse a single SSE line and return the JSON chunk, or None to skip.

    Returns ``_SENTINEL_DONE`` when the stream signals completion.
    """
    parsed = _iter_sse_lines(line)
    if parsed is None:
        return None
    field, value = parsed
    if field == "event" or field != "data" or value is None:
        return None
    if _is_openai_done(value):
        return _SENTINEL_DONE
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        logger.warning("Skipping malformed SSE data: %s", value[:200])
        return None


async def _format_upstream_error(upstream_resp: Any, endpoint: str) -> str:
    """Read an error response from upstream and format it as an SSE data line."""
    raw = await upstream_resp.aread()
    error_text = (
        raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw
    )
    log_upstream_error(
        upstream_resp.status_code,
        error_text,
        endpoint=endpoint,
        is_streaming=True,
    )
    try:
        error_body = json.loads(error_text)
        error_msg = json.dumps(error_body)
    except json.JSONDecodeError:
        error_msg = error_text
    return f"data: {error_msg}\n\n"


async def _stream_event_generator(
    *,
    source_provider: ProviderType,
    target_provider: ProviderType,
    source_converter: Any,
    target_converter: Any,
    ctx: ConversionContext,
    provider_info: ProviderInfo,
    url: str,
    upstream_body: dict[str, Any],
    headers: dict[str, str],
    format_sse: Any,
    store: ProviderMetadataStore,
    model: str,
) -> AsyncIterator[str]:
    """Stream SSE events from upstream, converting each chunk."""
    from_ctx = target_converter.create_stream_context()  # upstream -> IR
    to_ctx = source_converter.create_stream_context()  # IR -> source

    # Bridge preserve-mode metadata from request phase to streaming context
    to_ctx.options["metadata_mode"] = "preserve"
    from_ctx.options["metadata_mode"] = "preserve"
    if "_request_echo" in ctx.metadata:
        to_ctx.metadata["_request_echo"] = ctx.metadata["_request_echo"]

    chunk_count = 0
    t0 = time.monotonic()

    client = get_client(provider_info.proxy_url)
    upstream_resp = await client.post(
        url, json=upstream_body, headers=headers, stream=True
    )
    assert isinstance(upstream_resp, HttpStreamingResponse)
    async with upstream_resp:
        if upstream_resp.status_code >= 400:
            error_sse = await _format_upstream_error(
                upstream_resp, str(target_provider)
            )
            yield error_sse
            return

        async for line in upstream_resp.aiter_lines():
            chunk = _parse_sse_data(line)
            if chunk is _SENTINEL_DONE:
                break
            if chunk is None:
                continue

            chunk_count += 1

            ir_events = target_converter.stream_response_from_provider(
                chunk, context=from_ctx
            )

            if "_response_extras" in from_ctx.metadata:
                to_ctx.metadata["_response_extras"] = from_ctx.metadata[
                    "_response_extras"
                ]

            for ir_event in ir_events:
                store.cache_from_stream_event(ir_event)
                source_chunks = source_converter.stream_response_to_provider(
                    ir_event, context=to_ctx
                )
                if isinstance(source_chunks, list):
                    for sc in source_chunks:
                        if sc:
                            yield format_sse(sc)
                elif source_chunks:
                    yield format_sse(source_chunks)

    if source_provider == "openai_chat":
        yield _format_sse_openai_chat_done()

    log_stream_summary(
        model=model,
        duration_s=time.monotonic() - t0,
        chunk_count=chunk_count,
    )


async def handle_streaming(
    source_provider: ProviderType,
    target_provider: ProviderType,
    provider_info: ProviderInfo,
    body: dict[str, Any],
    model: str,
    *,
    metadata_store: ProviderMetadataStore | None = None,
    extra_headers: dict[str, str] | None = None,
) -> Response | StreamingResponse:
    """Streaming proxy: convert -> forward -> stream-convert back -> SSE."""
    store = metadata_store or _default_metadata_store
    source_converter = get_converter_for_provider(source_provider)
    target_converter = get_converter_for_provider(target_provider)

    # Shared context for the request conversion phase
    ctx = ConversionContext()
    ctx.options["metadata_mode"] = "preserve"
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
    store.inject_into_request(ir_request)

    # -- body log: IR request (after source -> IR) --
    log_original_request(ir_request)

    # 2. IR -> Target
    try:
        target_body, _ = target_converter.request_to_provider(ir_request, context=ctx)
    except Exception as exc:
        return error_response_for_source(
            source_provider, 400, f"Conversion error: {exc}"
        )
    if ctx.warnings:
        logger.warning("Conversion warnings: %s", ctx.warnings)

    # 3. Build upstream request (with stream=True)
    url, headers, upstream_body = prepare_upstream(
        target_provider,
        provider_info,
        target_body,
        model,
        stream=True,
        extra_headers=extra_headers,
    )

    # -- body log: target request body --
    log_converted_request(upstream_body)

    format_sse = SSE_FORMATTERS[source_provider]

    return StreamingResponse(
        _stream_event_generator(
            source_provider=source_provider,
            target_provider=target_provider,
            source_converter=source_converter,
            target_converter=target_converter,
            ctx=ctx,
            provider_info=provider_info,
            url=url,
            upstream_body=upstream_body,
            headers=headers,
            format_sse=format_sse,
            store=store,
            model=model,
        ),
        content_type="text/event-stream",
    )
