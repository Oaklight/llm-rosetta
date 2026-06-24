"""Proxy engine — response conversion, metadata caching, and pipeline handlers.

This module contains the core proxy logic:
- Provider metadata caching (e.g. Google ``thought_signature``)
- Shim transform resolution
- Non-streaming and streaming request handlers
- Error response helpers
- Request body helpers

Transport-level concerns (HTTP client, SSE parsing, upstream request assembly)
are delegated to the :class:`~transport.UpstreamTransport` interface.
Downstream SSE formatting lives in :mod:`transport.sse_format`.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response, StreamingResponse

from llm_rosetta.auto_detect import ProviderType
from llm_rosetta.pipeline import ConversionError, ConversionPipeline

from .logging import (
    get_logger,
    log_converted_request,
    log_original_request,
    log_response,
    log_stream_summary,
    log_upstream_error,
)
from .transport import (
    ProviderInfo,
    UpstreamConnectionError,
    UpstreamTransport,
)
from .transport.sse_format import SSE_FORMATTERS, format_sse_done

logger = get_logger()


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
# Resource cleanup
# ---------------------------------------------------------------------------


async def close_resources(
    *,
    transport: UpstreamTransport | None = None,
    metadata_store: ProviderMetadataStore | None = None,
) -> None:
    """Close transport and clear metadata store (called on app shutdown)."""
    if transport is not None:
        await transport.close()
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
    transport: UpstreamTransport,
    metadata_store: ProviderMetadataStore | None = None,
    extra_headers: dict[str, str] | None = None,
    target_shim_name: str | None = None,
    reasoning_config_override: dict[str, Any] | None = None,
    model_capabilities: list[str] | None = None,
) -> Response:
    """Non-streaming proxy: convert -> forward -> convert back -> respond."""
    store = metadata_store or _default_metadata_store

    pipeline = ConversionPipeline(
        source_provider,
        target_provider,
        target_shim_name,
        upstream_model=body.get("model"),
        model_capabilities=model_capabilities,
        reasoning_config_override=reasoning_config_override,
    )

    # Phase 1+2: Source → IR → Target
    try:
        target_body = pipeline.convert_request(
            body, on_ir_ready=store.inject_into_request
        )
    except ConversionError as exc:
        return error_response_for_source(source_provider, 400, str(exc))

    log_original_request(pipeline.ir_request)
    if pipeline.warnings:
        logger.warning("Conversion warnings: %s", pipeline.warnings)
    log_converted_request(target_body)

    # Phase 3: Forward to upstream via transport
    try:
        resp = await transport.send_request(
            provider_info,
            target_provider,
            target_body,
            model,
            extra_headers=extra_headers,
        )
    except UpstreamConnectionError as exc:
        return error_response_for_source(
            source_provider, 502, f"Upstream request failed: {exc}"
        )

    if resp.is_error:
        log_upstream_error(
            resp.status_code,
            resp.error_text,
            endpoint=str(target_provider),
        )
        return Response(
            body=resp.raw_content,
            status_code=resp.status_code,
            content_type="application/json",
        )

    # Phase 4: Target response → Source response
    assert resp.body is not None
    log_response(resp.body, label="UPSTREAM RESPONSE")
    try:
        source_response = pipeline.convert_response(
            resp.body, on_ir_ready=store.cache_from_response
        )
    except ConversionError as exc:
        return error_response_for_source(source_provider, 502, str(exc))

    return JSONResponse(source_response)


async def _stream_event_generator(
    *,
    source_provider: ProviderType,
    target_provider: ProviderType,
    processor: Any,
    transport: UpstreamTransport,
    provider_info: ProviderInfo,
    target_body: dict[str, Any],
    model: str,
    format_sse: Any,
    extra_headers: dict[str, str] | None = None,
) -> AsyncIterator[str]:
    """Stream SSE events from upstream, converting each chunk via Pipeline."""
    chunk_count = 0
    t0 = time.monotonic()

    try:
        stream = await transport.send_streaming(
            provider_info,
            target_provider,
            target_body,
            model,
            extra_headers=extra_headers,
        )
    except UpstreamConnectionError as exc:
        yield f"data: {json.dumps({'error': {'message': str(exc)}})}\n\n"
        return

    async with stream:
        if stream.is_error:
            error_text = await stream.read_error()
            log_upstream_error(
                stream.status_code,
                error_text,
                endpoint=str(target_provider),
                is_streaming=True,
            )
            try:
                error_msg = json.dumps(json.loads(error_text))
            except json.JSONDecodeError:
                error_msg = error_text
            yield f"data: {error_msg}\n\n"
            return

        async for chunk in stream:
            chunk_count += 1
            for source_event in processor.process_chunk(chunk):
                yield format_sse(source_event)

    if source_provider == "openai_chat":
        yield format_sse_done()

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
    transport: UpstreamTransport,
    metadata_store: ProviderMetadataStore | None = None,
    extra_headers: dict[str, str] | None = None,
    target_shim_name: str | None = None,
    reasoning_config_override: dict[str, Any] | None = None,
    model_capabilities: list[str] | None = None,
) -> Response | StreamingResponse:
    """Streaming proxy: convert -> forward -> stream-convert back -> SSE."""
    store = metadata_store or _default_metadata_store

    pipeline = ConversionPipeline(
        source_provider,
        target_provider,
        target_shim_name,
        upstream_model=body.get("model"),
        model_capabilities=model_capabilities,
        reasoning_config_override=reasoning_config_override,
    )

    # Phase 1+2: Source → IR → Target
    try:
        target_body = pipeline.convert_request(
            body, on_ir_ready=store.inject_into_request
        )
    except ConversionError as exc:
        return error_response_for_source(source_provider, 400, str(exc))

    log_original_request(pipeline.ir_request)
    if pipeline.warnings:
        logger.warning("Conversion warnings: %s", pipeline.warnings)

    log_converted_request(target_body)

    # Create stream processor for per-chunk conversion
    processor = pipeline.create_stream_processor(
        on_ir_event=store.cache_from_stream_event,
    )
    format_sse = SSE_FORMATTERS[source_provider]

    return StreamingResponse(
        _stream_event_generator(
            source_provider=source_provider,
            target_provider=target_provider,
            processor=processor,
            transport=transport,
            provider_info=provider_info,
            target_body=target_body,
            model=model,
            format_sse=format_sse,
            extra_headers=extra_headers,
        ),
        content_type="text/event-stream",
    )
