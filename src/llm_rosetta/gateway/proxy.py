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

import asyncio
import json
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response, StreamingResponse

from llm_rosetta.auto_detect import ProviderType
from llm_rosetta.pipeline import ConversionError, ConversionPipeline
from llm_rosetta.routing import ResolvedRoute

from llm_rosetta.observability.capture import CapturedRequest, CaptureState
from llm_rosetta.observability.error_dump import dump_error

from .logging import (
    get_logger,
    log_converted_request,
    log_original_request,
    log_response,
    log_stream_summary,
    log_upstream_error,
)
from .sanitize import sanitize_upstream_error

from .transport import (
    ProviderInfo,
    UpstreamConnectionError,
    UpstreamTransport,
)
from .transport.sse_format import (
    SSE_FORMATTERS,
    build_stream_error_events,
    extract_upstream_error_message,
    is_upstream_error_chunk,
    format_sse_done,
)

logger = get_logger()


@dataclass
class DumpContext:
    """Bundles error-dump parameters passed through streaming pipelines."""

    persistence: Any | None = None
    request_body: dict | None = None
    converted_body: dict | None = None
    source_provider: str | None = None
    target_provider: str | None = None
    provider_name: str | None = None
    upstream_url: str | None = None


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


def _inject_stream_flags(
    target_provider: ProviderType, body: dict[str, Any], *, stream: bool
) -> dict[str, Any]:
    """Inject provider-specific stream flags into an outbound request body.

    Returns the body unchanged when *stream* is False. When True, returns
    a shallow copy with stream flags set for the target provider.
    """
    if not stream:
        return body
    body = dict(body)
    if target_provider == "openai_chat":
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    elif target_provider in ("openai_responses", "open_responses", "anthropic"):
        body["stream"] = True
    # Google streaming is signaled via URL, not body
    return body


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


def _strip_internal_metadata(obj: dict | list) -> None:
    """Recursively remove ``_provider_metadata`` from an outbound request body.

    Converters attach ``_provider_metadata`` to provider-format objects
    for cross-converter round-trip fidelity.  This must be stripped
    before sending to upstream providers, which do not recognise it.

    Operates in-place for zero-copy overhead.  Accepts both dict and
    list inputs for natural recursion through nested structures.
    """
    if isinstance(obj, dict):
        for key in list(obj.keys()):
            if key == "_provider_metadata":
                del obj[key]
            else:
                _strip_internal_metadata(obj[key])
    elif isinstance(obj, list):
        for item in obj:
            _strip_internal_metadata(item)


async def handle_non_streaming(
    route: ResolvedRoute,
    provider_info: ProviderInfo,
    body: dict[str, Any],
    *,
    transport: UpstreamTransport,
    metadata_store: ProviderMetadataStore | None = None,
    extra_headers: dict[str, str] | None = None,
    persistence: Any | None = None,
    capture_state: CaptureState | None = None,
) -> tuple[Response, dict[str, Any]]:
    """Non-streaming proxy: convert -> forward -> convert back -> respond.

    Returns:
        A ``(response, profile)`` tuple.  The profile dict contains
        per-phase timing data merged from the conversion pipeline and
        gateway-level measurements (upstream latency).
    """
    store = metadata_store or _default_metadata_store
    profile: dict[str, Any] = {}
    # model was already injected into body by app.py
    model = body.get("model", "")

    pipeline = ConversionPipeline(
        route.source_provider,
        route.target_provider,
        target_shim=route.shim_name,
        upstream_model=model,
        model_capabilities=route.model_capabilities,
        reasoning_config_override=route.reasoning_override,
        supports_custom_tools=route.supports_custom_tools,
        hoist_system_messages=route.hoist_system_messages,
    )

    # Phase 1+2: Source → IR → Target
    try:
        target_body = pipeline.convert_request(
            body, on_ir_ready=store.inject_into_request
        )
    except ConversionError as exc:
        dump_error(
            persistence,
            request_body=body,
            response_text=str(exc),
            model=model,
            source_provider=route.source_provider,
            target_provider=route.target_provider,
            provider_name=route.provider_name,
            status_code=400,
            error_phase="conversion",
            upstream_url=str(provider_info.base_url),
        )
        return error_response_for_source(route.source_provider, 400, str(exc)), profile

    profile.update(pipeline.profile)

    log_original_request(pipeline.ir_request)
    if pipeline.warnings:
        logger.warning("Conversion warnings: %s", pipeline.warnings)

    # Per-model system message flattening (gateway config override)
    if route.flatten_system:
        from llm_rosetta.shims.transforms import flatten_system_content

        target_body = flatten_system_content()(target_body)

    log_converted_request(target_body)
    _strip_internal_metadata(target_body)

    # Phase 3: Forward to upstream via transport
    upstream_url = provider_info.upstream_url(model)
    t_upstream = time.perf_counter()
    try:
        resp = await transport.send(
            provider_info,
            upstream_url,
            target_body,
            extra_headers=extra_headers,
        )
    except UpstreamConnectionError as exc:
        profile["upstream_ms"] = round((time.perf_counter() - t_upstream) * 1000, 2)
        dump_error(
            persistence,
            request_body=body,
            response_text=str(exc),
            converted_body=target_body,
            model=model,
            source_provider=route.source_provider,
            target_provider=route.target_provider,
            provider_name=route.provider_name,
            status_code=502,
            error_phase="upstream",
            upstream_url=str(provider_info.base_url),
        )
        if capture_state is not None and capture_state.should_capture():
            capture_state.record(
                CapturedRequest(
                    original_request=body,
                    converted_body=target_body,
                    request_id=(
                        extra_headers.get("x-request-id", "") if extra_headers else ""
                    ),
                    model=model,
                    source_provider=route.source_provider,
                    target_provider=route.target_provider,
                    is_stream=False,
                    status_code=502,
                    upstream_response={"error": str(exc)},
                )
            )
        return (
            error_response_for_source(
                route.source_provider, 502, f"Upstream request failed: {exc}"
            ),
            profile,
        )
    profile["upstream_ms"] = round((time.perf_counter() - t_upstream) * 1000, 2)

    if resp.is_error:
        log_upstream_error(
            resp.status_code,
            resp.error_text,
            endpoint=str(route.target_provider),
        )
        dump_error(
            persistence,
            request_body=body,
            response_text=resp.error_text,
            converted_body=target_body,
            model=model,
            source_provider=route.source_provider,
            target_provider=route.target_provider,
            provider_name=route.provider_name,
            status_code=resp.status_code,
            error_phase="upstream",
            upstream_url=str(provider_info.base_url),
        )
        if capture_state is not None and capture_state.should_capture():
            capture_state.record(
                CapturedRequest(
                    original_request=body,
                    converted_body=target_body,
                    upstream_response=resp.body,
                    request_id=(
                        extra_headers.get("x-request-id", "") if extra_headers else ""
                    ),
                    model=model,
                    source_provider=route.source_provider,
                    target_provider=route.target_provider,
                    is_stream=False,
                    status_code=resp.status_code,
                )
            )
        return (
            Response(
                body=sanitize_upstream_error(resp.raw_content),
                status_code=resp.status_code,
                content_type="application/json",
            ),
            profile,
        )

    # Phase 4: Target response → Source response
    assert resp.body is not None
    log_response(resp.body, label="UPSTREAM RESPONSE")
    try:
        source_response = pipeline.convert_response(
            resp.body, on_ir_ready=store.cache_from_response
        )
    except ConversionError as exc:
        profile.update(pipeline.profile)
        dump_error(
            persistence,
            request_body=body,
            response_text=str(exc),
            converted_body=target_body,
            model=model,
            source_provider=route.source_provider,
            target_provider=route.target_provider,
            provider_name=route.provider_name,
            status_code=502,
            error_phase="response",
            upstream_url=str(provider_info.base_url),
        )
        return error_response_for_source(route.source_provider, 502, str(exc)), profile

    # Merge response-phase timings from pipeline
    profile.update(pipeline.profile)

    # Content capture (non-streaming): record all three stages
    if capture_state is not None and capture_state.should_capture():
        capture_state.record(
            CapturedRequest(
                original_request=body,
                converted_body=target_body,
                upstream_response=resp.body,
                request_id=(
                    extra_headers.get("x-request-id", "") if extra_headers else ""
                ),
                model=model,
                source_provider=route.source_provider,
                target_provider=route.target_provider,
                is_stream=False,
                status_code=resp.status_code,
            )
        )

    _strip_internal_metadata(source_response)
    return JSONResponse(source_response), profile


def _terminal_error_sse(
    source_provider: ProviderType,
    processor: Any,
    format_sse: Any,
    exc: BaseException,
) -> list[str]:
    """Build SSE text announcing that the upstream stream died mid-flight.

    Without this the socket simply closes and clients that wait for a
    terminal event (Codex waits for ``response.completed``) report a bare
    "stream closed before response.completed" with no upstream reason.

    Never raises: a failure to synthesize the notice must not replace the
    original exception, which carries the real diagnosis.

    Returns:
        SSE strings to yield, or an empty list if no notice applies.
    """
    # Unreachable while the caller catches Exception, since CancelledError
    # derives from BaseException. Kept so widening that clause later cannot
    # silently start writing to a client that has already hung up.
    if isinstance(exc, asyncio.CancelledError):
        return []

    try:
        ctx = getattr(processor, "source_context", None)
        if ctx is not None and ctx.is_ended:
            # Stream already reached its terminal event — the failure
            # happened during teardown, so a second one would corrupt it.
            return []

        events = build_stream_error_events(
            source_provider,
            str(exc) or exc.__class__.__name__,
            response_id=getattr(ctx, "outbound_response_id", "") or "",
            sequence_number=getattr(ctx, "next_sequence_number", None),
        )
        out = [format_sse(event) for event in events]
        if source_provider in ("openai_chat", "openai_responses", "open_responses"):
            out.append(format_sse_done())
        return out
    except Exception:
        logger.debug("Failed to build terminal error event", exc_info=True)
        return []


async def _stream_event_generator(
    *,
    source_provider: ProviderType,
    stream: Any,
    processor: Any,
    model: str,
    format_sse: Any,
    entry_id: str | None = None,
    request_log: Any | None = None,
    capture_record: CapturedRequest | None = None,
    capture_state: CaptureState | None = None,
    dump_ctx: DumpContext | None = None,
) -> AsyncIterator[str]:
    """Stream SSE events from an already-opened upstream stream.

    The caller (``handle_streaming``) is responsible for opening the upstream
    connection and checking for immediate errors *before* constructing the
    ``StreamingResponse``.  This ensures the HTTP status code sent to the
    client reflects the upstream status (e.g. 400 for token-limit errors)
    rather than always being 200.
    """
    chunk_count = 0
    t0 = time.monotonic()
    stream_error: str | None = None
    ttfb_ms: float | None = None
    t_stream_open = time.perf_counter()
    upstream_chunks: list[dict[str, Any]] | None = (
        [] if capture_record is not None else None
    )

    try:
        async with stream:
            async for chunk in stream:
                if chunk_count == 0:
                    ttfb_ms = round((time.perf_counter() - t_stream_open) * 1000, 2)
                chunk_count += 1
                if upstream_chunks is not None:
                    upstream_chunks.append(chunk)

                if is_upstream_error_chunk(chunk):
                    stream_error = extract_upstream_error_message(chunk)
                    log_upstream_error(
                        getattr(stream, "status_code", 200),
                        json.dumps(chunk)[:2000],
                        endpoint=model,
                        is_streaming=True,
                    )
                    _dc = dump_ctx or DumpContext()
                    dump_error(
                        _dc.persistence,
                        request_body=_dc.request_body,
                        response_text=json.dumps(chunk)[:2000],
                        converted_body=_dc.converted_body,
                        model=model,
                        source_provider=_dc.source_provider,
                        target_provider=_dc.target_provider,
                        provider_name=_dc.provider_name,
                        status_code=getattr(stream, "status_code", 0),
                        error_phase="stream_chunk",
                        upstream_url=_dc.upstream_url,
                        request_log_id=entry_id,
                    )
                    for event in build_stream_error_events(
                        source_provider, stream_error
                    ):
                        yield format_sse(event)
                    break

                for source_event in processor.process_chunk(chunk):
                    yield format_sse(source_event)

        if source_provider in ("openai_chat", "openai_responses", "open_responses"):
            yield format_sse_done()

        log_stream_summary(
            model=model,
            duration_s=time.monotonic() - t0,
            chunk_count=chunk_count,
        )
    except Exception as exc:
        stream_error = str(exc)
        for sse_text in _terminal_error_sse(
            source_provider, processor, format_sse, exc
        ):
            yield sse_text
        raise
    finally:
        # Write back stream profile to request log entry
        if entry_id and request_log is not None:
            stream_profile: dict[str, Any] = {
                "stream_duration_ms": round((time.monotonic() - t0) * 1000, 2),
                "stream_chunks": chunk_count,
                "stream_complete": stream_error is None,
            }
            if ttfb_ms is not None:
                stream_profile["stream_ttfb_ms"] = ttfb_ms
            if stream_error is not None:
                stream_profile["stream_error"] = stream_error[:500]
            try:
                request_log.update_profile(entry_id, stream_profile)
            except Exception:
                logger.debug("Failed to write stream profile for %s", entry_id)

        # Content capture (streaming): record accumulated upstream chunks
        if capture_record is not None and capture_state is not None:
            capture_record.upstream_response = upstream_chunks
            capture_record.status_code = getattr(stream, "status_code", 200)
            capture_state.record(capture_record)


# ---------------------------------------------------------------------------
# Preflight token count helpers
# ---------------------------------------------------------------------------

_PREFLIGHT_TIMEOUT = 5.0  # seconds — bound worst-case TTFT impact


def _build_preflight_body(
    target_body: dict[str, Any], target_provider: ProviderType
) -> dict[str, Any]:
    """Build a minimal non-streaming request body for token counting.

    Uses a shallow copy of *target_body* (messages/tools are read-only)
    and only deep-copies nested dicts that are mutated, to avoid
    duplicating large conversation histories.
    """
    body = dict(target_body)

    # Disable streaming
    body.pop("stream", None)
    body.pop("stream_options", None)

    # Set minimal output
    if target_provider == "google":
        gc = dict(body.get("generationConfig", {}))
        gc["maxOutputTokens"] = 1
        gc.pop("responseModalities", None)
        body["generationConfig"] = gc
    elif target_provider in ("openai_responses", "open_responses"):
        body["max_output_tokens"] = 1
    else:
        body["max_tokens"] = 1

    # Strip thinking / reasoning to minimise cost
    body.pop("thinking", None)
    body.pop("reasoning", None)
    if isinstance(body.get("metadata"), dict):
        body["metadata"] = {
            k: v for k, v in body["metadata"].items() if k != "thinking"
        }

    return body


def _extract_preflight_input_tokens(
    response_body: dict[str, Any], target_provider: ProviderType
) -> int | None:
    """Extract input/prompt token count from a non-streaming response."""
    try:
        if target_provider == "google":
            return int(response_body["usageMetadata"]["promptTokenCount"])
        usage = response_body.get("usage", {})
        if target_provider in ("openai_chat",):
            return int(usage["prompt_tokens"])
        # Anthropic, OpenAI Responses, Open Responses
        return int(usage["input_tokens"])
    except (KeyError, TypeError, ValueError):
        return None


async def _run_preflight(
    transport: UpstreamTransport,
    provider_info: ProviderInfo,
    target_body: dict[str, Any],
    target_provider: ProviderType,
    model: str,
    *,
    extra_headers: dict[str, str] | None = None,
) -> int | None:
    """Send a preflight request and return input token count.

    Never raises — returns ``None`` on any failure so the real streaming
    request can proceed with ``input_tokens=0`` as before.
    """
    try:
        preflight_body = _build_preflight_body(target_body, target_provider)
        upstream_url = provider_info.upstream_url(model)
        pinfo = provider_info.with_timeout(_PREFLIGHT_TIMEOUT)
        resp = await transport.send(
            pinfo,
            upstream_url,
            preflight_body,
            extra_headers=extra_headers,
        )
        if resp.is_error:
            logger.debug("Preflight request failed with status %d", resp.status_code)
            return None
        if resp.body is None:
            return None
        count = _extract_preflight_input_tokens(resp.body, target_provider)
        logger.debug("Preflight input_tokens: %s", count)
        return count
    except Exception:
        logger.debug("Preflight request error", exc_info=True)
        return None


async def handle_streaming(
    route: ResolvedRoute,
    provider_info: ProviderInfo,
    body: dict[str, Any],
    *,
    transport: UpstreamTransport,
    metadata_store: ProviderMetadataStore | None = None,
    extra_headers: dict[str, str] | None = None,
    entry_id: str | None = None,
    request_log: Any | None = None,
    persistence: Any | None = None,
    capture_state: CaptureState | None = None,
    preflight_token_count: bool = False,
) -> tuple[Response | StreamingResponse, dict[str, Any]]:
    """Streaming proxy: convert -> forward -> stream-convert back -> SSE.

    Opens the upstream connection *before* constructing the
    ``StreamingResponse`` so that immediate errors (4xx/5xx from the
    upstream) are returned with the correct HTTP status code instead of
    being buried inside an SSE event on an HTTP 200 response.

    Returns:
        A ``(response, profile)`` tuple.  The profile dict contains
        request-phase timing data.  Stream-phase metrics (TTFB,
        duration, chunks) are written back to the request log entry
        after the stream completes.
    """
    store = metadata_store or _default_metadata_store
    profile: dict[str, Any] = {}
    # model was already injected into body by app.py
    model = body.get("model", "")

    pipeline = ConversionPipeline(
        route.source_provider,
        route.target_provider,
        target_shim=route.shim_name,
        upstream_model=model,
        model_capabilities=route.model_capabilities,
        reasoning_config_override=route.reasoning_override,
        supports_custom_tools=route.supports_custom_tools,
        hoist_system_messages=route.hoist_system_messages,
    )

    # Phase 1+2: Source → IR → Target
    try:
        target_body = pipeline.convert_request(
            body, on_ir_ready=store.inject_into_request
        )
    except ConversionError as exc:
        dump_error(
            persistence,
            request_body=body,
            response_text=str(exc),
            model=model,
            source_provider=route.source_provider,
            target_provider=route.target_provider,
            provider_name=route.provider_name,
            status_code=400,
            error_phase="conversion",
            upstream_url=str(provider_info.base_url),
        )
        return error_response_for_source(route.source_provider, 400, str(exc)), profile

    profile.update(pipeline.profile)

    log_original_request(pipeline.ir_request)
    if pipeline.warnings:
        logger.warning("Conversion warnings: %s", pipeline.warnings)

    # Per-model system message flattening (gateway config override)
    if route.flatten_system:
        from llm_rosetta.shims.transforms import flatten_system_content

        target_body = flatten_system_content()(target_body)

    log_converted_request(target_body)
    _strip_internal_metadata(target_body)

    # Preflight: get exact input_tokens before streaming (opt-in)
    _preflight_input_tokens: int | None = None
    if preflight_token_count and route.source_provider != route.target_provider:
        _preflight_input_tokens = await _run_preflight(
            transport,
            provider_info,
            target_body,
            route.target_provider,
            model,
            extra_headers=extra_headers,
        )

    # Phase 3: Open upstream connection and check for immediate errors
    # *before* committing to a 200 StreamingResponse.
    upstream_url = provider_info.upstream_url(model, stream=True)
    target_body = _inject_stream_flags(route.target_provider, target_body, stream=True)
    t_connect = time.perf_counter()
    try:
        stream = await transport.send_streaming(
            provider_info,
            upstream_url,
            target_body,
            extra_headers=extra_headers,
        )
    except UpstreamConnectionError as exc:
        profile["stream_connect_ms"] = round(
            (time.perf_counter() - t_connect) * 1000, 2
        )
        # Connection-level failure — no upstream HTTP response exists, so
        # the gateway synthesizes an error message and returns 502.
        error_msg = str(exc)
        dump_error(
            persistence,
            request_body=body,
            response_text=error_msg,
            converted_body=target_body,
            model=model,
            source_provider=route.source_provider,
            target_provider=route.target_provider,
            provider_name=route.provider_name,
            status_code=502,
            error_phase="stream_header",
            upstream_url=str(provider_info.base_url),
            request_log_id=entry_id,
        )
        return (
            error_response_for_source(
                route.source_provider, 502, f"Upstream request failed: {exc}"
            ),
            profile,
        )

    profile["stream_connect_ms"] = round((time.perf_counter() - t_connect) * 1000, 2)

    # Application-level error — upstream returned a valid HTTP response with
    # a 4xx/5xx status.  Pass the original body through as-is so the client
    # SDK can parse the real error (e.g. "context_length_exceeded").
    if stream.is_error:
        error_text = await stream.read_error()
        await stream.close()
        log_upstream_error(
            stream.status_code,
            error_text,
            endpoint=str(route.target_provider),
            is_streaming=True,
        )
        dump_error(
            persistence,
            request_body=body,
            response_text=error_text,
            converted_body=target_body,
            model=model,
            source_provider=route.source_provider,
            target_provider=route.target_provider,
            provider_name=route.provider_name,
            status_code=stream.status_code,
            error_phase="stream_header",
            upstream_url=str(provider_info.base_url),
            request_log_id=entry_id,
        )
        return (
            Response(
                body=sanitize_upstream_error(error_text),
                status_code=stream.status_code,
                content_type="application/json",
            ),
            profile,
        )

    # Phase 4: No error — create stream processor and return SSE response
    processor = pipeline.create_stream_processor(
        on_ir_event=store.cache_from_stream_event,
    )

    if _preflight_input_tokens is not None:
        ctx = getattr(processor, "source_context", None)
        if ctx is not None:
            ctx.preflight_usage = {"prompt_tokens": _preflight_input_tokens}

    format_sse = SSE_FORMATTERS[route.source_provider]

    # Build a partial capture record if content capture is active.
    # The generator will fill in upstream_response and status_code
    # after the stream completes.
    _capture_record: CapturedRequest | None = None
    if capture_state is not None and capture_state.should_capture():
        _capture_record = CapturedRequest(
            original_request=body,
            converted_body=target_body,
            request_id=(extra_headers.get("x-request-id", "") if extra_headers else ""),
            model=model,
            source_provider=route.source_provider,
            target_provider=route.target_provider,
            is_stream=True,
        )

    return (
        StreamingResponse(
            _stream_event_generator(
                source_provider=route.source_provider,
                stream=stream,
                processor=processor,
                model=model,
                format_sse=format_sse,
                entry_id=entry_id,
                request_log=request_log,
                capture_record=_capture_record,
                capture_state=capture_state,
                dump_ctx=DumpContext(
                    persistence=persistence,
                    request_body=body,
                    converted_body=target_body,
                    source_provider=route.source_provider,
                    target_provider=route.target_provider,
                    provider_name=route.provider_name,
                    upstream_url=str(provider_info.base_url),
                ),
            ),
            content_type="text/event-stream",
        ),
        profile,
    )
