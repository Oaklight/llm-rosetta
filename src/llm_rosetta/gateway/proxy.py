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

from llm_rosetta import get_converter_for_provider
from llm_rosetta.auto_detect import ProviderType
from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.shims import get_shim
from llm_rosetta.pipeline import apply_shim_to_ir, setup_shim_context
from llm_rosetta.shims.transforms import Transform, apply_transforms

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
# Shim transform resolution
# ---------------------------------------------------------------------------


_EMPTY_TRANSFORMS: tuple[Transform, ...] = ()


def _resolve_target_transforms(
    shim_name: str | None,
    model: str | None = None,
) -> tuple[tuple[Transform, ...], tuple[Transform, ...]]:
    """Look up target-side transforms from the shim registry.

    Args:
        shim_name: Registered shim name (e.g. ``"volcengine"``), or ``None``.
        model: Unused, kept for API compatibility.

    Returns:
        ``(from_transforms, to_transforms)`` ready for ``apply_transforms``.
        Both are empty tuples when no shim is found.
    """
    if shim_name is None:
        return _EMPTY_TRANSFORMS, _EMPTY_TRANSFORMS
    shim = get_shim(shim_name)
    if shim is None:
        return _EMPTY_TRANSFORMS, _EMPTY_TRANSFORMS
    return shim.from_transforms, shim.to_transforms


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
    source_converter = get_converter_for_provider(source_provider)
    target_converter = get_converter_for_provider(target_provider)

    # Resolve target-side transforms from shim registry
    target_from_t, target_to_t = _resolve_target_transforms(target_shim_name, model)

    # Shared context for the conversion pipeline
    ctx = ConversionContext()
    ctx.options["metadata_mode"] = "preserve"
    if target_provider == "google":
        ctx.options["output_format"] = "rest"

    # Inject shim reasoning capability so converters use it.
    setup_shim_context(
        ctx,
        target_shim_name,
        model=body.get("model"),
        config_override=reasoning_config_override,
    )

    # 1. Source -> IR
    try:
        ir_request = source_converter.request_from_provider(body, context=ctx)
    except Exception as exc:
        return error_response_for_source(
            source_provider, 400, f"Failed to parse request: {exc}"
        )

    # 1b. Restore cached provider_metadata (e.g. Google thought_signature)
    store.inject_into_request(ir_request)

    request_id = ctx.options.get("request_id", "-")

    # 1c–1e. Shim-driven IR transforms (non-vision strip, image limit, unwind)
    ir_request = apply_shim_to_ir(
        ir_request,
        target_shim_name,
        upstream_model=body.get("model"),
        model_capabilities=model_capabilities,
        request_id=request_id,
    )

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

    # 2b. Apply target shim to_transforms (e.g. strip unsupported fields)
    if target_to_t:
        target_body = apply_transforms(target_to_t, target_body)

    # -- body log: target request body --
    log_converted_request(target_body)

    # 3. Forward to upstream via transport
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

    # 4. Pass through upstream errors
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

    # 5. Target response -> IR
    upstream_json = resp.body
    assert upstream_json is not None
    try:
        # 5a. Apply target shim from_transforms (normalise response dialect)
        if target_from_t:
            upstream_json = apply_transforms(target_from_t, upstream_json)
        ir_response = target_converter.response_from_provider(
            upstream_json, context=ctx
        )
    except Exception as exc:
        return error_response_for_source(
            source_provider, 502, f"Failed to parse upstream response: {exc}"
        )

    # -- body log: upstream response --
    log_response(upstream_json, label="UPSTREAM RESPONSE")

    # 5b. Cache provider_metadata from tool calls for follow-up requests
    store.cache_from_response(ir_response)

    # 6. IR -> Source response
    try:
        source_response = source_converter.response_to_provider(
            ir_response, context=ctx
        )
    except Exception as exc:
        return error_response_for_source(
            source_provider, 500, f"Failed to convert response: {exc}"
        )

    return JSONResponse(source_response)


def process_stream_chunk(
    chunk: dict[str, Any],
    *,
    target_converter: Any,
    source_converter: Any,
    from_ctx: Any,
    to_ctx: Any,
    store: ProviderMetadataStore,
    format_sse: Any,
    target_from_transforms: tuple[Transform, ...],
) -> list[str]:
    """Convert one upstream chunk through the full pipeline to source SSE strings.

    Handles: shim transforms → upstream→IR conversion → metadata bridging
    → IR→source conversion → SSE formatting.
    """
    if target_from_transforms:
        chunk = apply_transforms(target_from_transforms, chunk)

    ir_events = target_converter.stream_response_from_provider(chunk, context=from_ctx)

    if "_response_extras" in from_ctx.metadata:
        to_ctx.metadata["_response_extras"] = from_ctx.metadata["_response_extras"]

    result: list[str] = []
    for ir_event in ir_events:
        store.cache_from_stream_event(ir_event)
        source_chunks = source_converter.stream_response_to_provider(
            ir_event, context=to_ctx
        )
        if isinstance(source_chunks, list):
            result.extend(format_sse(sc) for sc in source_chunks if sc)
        elif source_chunks:
            result.append(format_sse(source_chunks))
    return result


async def _stream_event_generator(
    *,
    source_provider: ProviderType,
    target_provider: ProviderType,
    source_converter: Any,
    target_converter: Any,
    ctx: ConversionContext,
    transport: UpstreamTransport,
    provider_info: ProviderInfo,
    target_body: dict[str, Any],
    model: str,
    format_sse: Any,
    store: ProviderMetadataStore,
    target_from_transforms: tuple[Transform, ...] = (),
    extra_headers: dict[str, str] | None = None,
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
            for sse_line in process_stream_chunk(
                chunk,
                target_converter=target_converter,
                source_converter=source_converter,
                from_ctx=from_ctx,
                to_ctx=to_ctx,
                store=store,
                format_sse=format_sse,
                target_from_transforms=target_from_transforms,
            ):
                yield sse_line

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
    source_converter = get_converter_for_provider(source_provider)
    target_converter = get_converter_for_provider(target_provider)

    # Resolve target-side transforms from shim registry
    target_from_t, target_to_t = _resolve_target_transforms(target_shim_name, model)

    # Shared context for the request conversion phase
    ctx = ConversionContext()
    ctx.options["metadata_mode"] = "preserve"
    if target_provider == "google":
        ctx.options["output_format"] = "rest"

    # Inject shim reasoning capability so converters use it.
    setup_shim_context(
        ctx,
        target_shim_name,
        model=body.get("model"),
        config_override=reasoning_config_override,
    )

    # 1. Source -> IR
    try:
        ir_request = source_converter.request_from_provider(body, context=ctx)
    except Exception as exc:
        return error_response_for_source(
            source_provider, 400, f"Failed to parse request: {exc}"
        )

    # 1b. Inject cached provider_metadata (e.g. Google thought_signature)
    store.inject_into_request(ir_request)

    request_id = ctx.options.get("request_id", "-")

    # 1c–1e. Shim-driven IR transforms (non-vision strip, image limit, unwind)
    ir_request = apply_shim_to_ir(
        ir_request,
        target_shim_name,
        upstream_model=body.get("model"),
        model_capabilities=model_capabilities,
        request_id=request_id,
    )

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

    # 2b. Apply target shim to_transforms (e.g. strip unsupported fields)
    if target_to_t:
        target_body = apply_transforms(target_to_t, target_body)

    # -- body log: target request body --
    log_converted_request(target_body)

    format_sse = SSE_FORMATTERS[source_provider]

    return StreamingResponse(
        _stream_event_generator(
            source_provider=source_provider,
            target_provider=target_provider,
            source_converter=source_converter,
            target_converter=target_converter,
            ctx=ctx,
            transport=transport,
            provider_info=provider_info,
            target_body=target_body,
            model=model,
            format_sse=format_sse,
            store=store,
            target_from_transforms=target_from_t,
            extra_headers=extra_headers,
        ),
        content_type="text/event-stream",
    )
