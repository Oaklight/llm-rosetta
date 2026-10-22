"""The streaming path: event generator, terminal errors, usage write-back."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import Response, StreamingResponse

from llm_rosetta.auto_detect import ProviderType
from llm_rosetta.pipeline import ConversionError, ConversionPipeline
from llm_rosetta.routing import ResolvedRoute
from llm_rosetta.shims.provider_shim import SoftErrorPattern

from llm_rosetta_gateway.observability.capture import CapturedRequest, CaptureState
from llm_rosetta_gateway.observability.error_dump import dump_error
from llm_rosetta.observability.usage import usage_columns

from ..logging import (
    get_logger,
    log_converted_request,
    log_original_request,
    log_stream_summary,
)
from ..routing import (
    SelectionContext,
)
from ..transport import (
    ProviderInfo,
    UpstreamConnectionError,
    UpstreamTransport,
)
from ..transport.sse_format import (
    SSE_FORMATTERS,
    UpstreamStreamError,
    build_stream_error_events,
    format_sse_done,
)
from .image_fetch import fetch_image
from .errors import (
    QuotaRefusal,
    _connection_error_response,
    _detect_stream_chunk_error,
    _handle_401,
    _quota_refusal,
    _upstream_error_passthrough,
    DumpContext,
    error_response_for_source,
)
from .metadata import (
    ProviderMetadataStore,
    _default_metadata_store,
    _strip_internal_metadata,
)
from .fidelity import (
    _fidelity_check_and_dump,
)
from .engine import (
    _apply_request_stages,
    _inject_stream_flags,
    _log_response_warnings,
)
from .preflight import (
    _run_preflight,
)

logger = get_logger()


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
        if source_provider in (
            "chat_completions",
            "openai_responses",
            "open_responses",
        ):
            out.append(format_sse_done())
        return out
    except Exception:
        logger.debug("Failed to build terminal error event", exc_info=True)
        return []


async def _write_back_stream_usage(
    processor: Any,
    entry_id: str | None,
    request_log: Any | None,
    metrics: Any | None,
    model: str,
    provider_name: str | None = None,
) -> None:
    """Extract accumulated token usage from the stream processor and persist it."""
    usage = getattr(processor, "get_accumulated_usage", lambda: None)()
    if not usage or not entry_id or request_log is None:
        return
    # The same normalization the non-streaming leg writes, so both legs fill a
    # row identically — the total included.
    columns = usage_columns(usage)
    if "input_tokens" not in columns and "output_tokens" not in columns:
        return
    try:
        await request_log.update_usage(entry_id, **columns)
    except Exception:
        logger.debug("Failed to write usage for %s", entry_id)
    if metrics is not None:
        metrics.record_usage(
            model=model,
            provider_name=provider_name,
            **columns,
        )


def _flush_and_finalize(
    processor: Any,
    format_sse: Any,
    source_provider: str,
    format_sse_done: Any,
) -> list[str]:
    """Flush deferred terminal events and append the SSE done sentinel."""
    tail = [format_sse(e) for e in processor.process_chunk(None)]
    if source_provider in ("chat_completions", "openai_responses", "open_responses"):
        tail.append(format_sse_done())
    return tail


async def _stream_event_generator(
    *,
    source_provider: ProviderType,
    stream: Any,
    processor: Any,
    model: str,
    format_sse: Any,
    reported_model: str | None = None,
    entry_id: str | None = None,
    request_log: Any | None = None,
    metrics: Any | None = None,
    capture_record: CapturedRequest | None = None,
    capture_state: CaptureState | None = None,
    dump_ctx: DumpContext | None = None,
    pipeline: Any | None = None,
    warnings_before: int = 0,
    soft_error_patterns: tuple[SoftErrorPattern, ...] = (),
) -> AsyncIterator[str]:
    """Stream SSE events from an already-opened upstream stream.

    The caller (``handle_streaming``) is responsible for opening the upstream
    connection and checking for immediate errors *before* constructing the
    ``StreamingResponse``.  This ensures the HTTP status code sent to the
    client reflects the upstream status (e.g. 400 for token-limit errors)
    rather than always being 200.
    """
    reported_model = reported_model or model
    chunk_count = 0
    t0 = time.monotonic()
    stream_error: UpstreamStreamError | None = None
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

                stream_error = await _detect_stream_chunk_error(
                    chunk,
                    soft_error_patterns=soft_error_patterns,
                    stream_status=getattr(stream, "status_code", 200),
                    model=model,
                    dump_ctx=dump_ctx,
                    entry_id=entry_id,
                )
                if stream_error is not None:
                    for event in build_stream_error_events(
                        source_provider,
                        stream_error.message,
                        code=stream_error.code,
                        request_id=stream_error.request_id or "",
                    ):
                        yield format_sse(event)
                    break

                for source_event in processor.process_chunk(chunk):
                    yield format_sse(source_event)

        # Flush any deferred terminal events and append the SSE [DONE]
        # sentinel for OpenAI-family formats.
        for sse_text in _flush_and_finalize(
            processor, format_sse, source_provider, format_sse_done
        ):
            yield sse_text

        log_stream_summary(
            model=model,
            duration_s=time.monotonic() - t0,
            chunk_count=chunk_count,
        )
    except Exception as exc:
        stream_error = UpstreamStreamError(message=str(exc))
        for sse_text in _terminal_error_sse(
            source_provider, processor, format_sse, exc
        ):
            yield sse_text
        raise
    finally:
        # The caller logged the request leg's warnings before the stream was
        # opened; anything the chunks added lands here.  In the finally so a
        # stream that died partway still reports what it managed to find.
        _log_response_warnings(pipeline, warnings_before)

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
                stream_profile["stream_error"] = stream_error.message[:500]
            try:
                await request_log.update_profile(entry_id, stream_profile)
            except Exception:
                logger.debug("Failed to write stream profile for %s", entry_id)

            await _write_back_stream_usage(
                processor,
                entry_id,
                request_log,
                metrics,
                reported_model,
                provider_name=dump_ctx.provider_name if dump_ctx else None,
            )

        # Content capture (streaming): record accumulated upstream chunks
        if capture_record is not None and capture_state is not None:
            capture_record.upstream_response = upstream_chunks
            capture_record.status_code = getattr(stream, "status_code", 200)
            capture_state.record(capture_record)


async def handle_streaming(
    route: ResolvedRoute,
    provider_info: ProviderInfo,
    body: dict[str, Any],
    *,
    reported_model: str | None = None,
    transport: UpstreamTransport,
    metadata_store: ProviderMetadataStore | None = None,
    extra_headers: dict[str, str] | None = None,
    entry_id: str | None = None,
    request_log: Any | None = None,
    metrics: Any | None = None,
    persistence: Any | None = None,
    capture_state: CaptureState | None = None,
    preflight_token_count: bool = False,
    selection: SelectionContext,
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
    # model was already injected into body by app.py, and app.py rewrites it to
    # the upstream id before calling us.  ``reported_model`` carries the name
    # the client asked for, so usage is grouped under the model the user named
    # rather than under whatever the upstream calls it.
    model = body.get("model", "")
    reported_model = reported_model or model

    pipeline = ConversionPipeline(
        route.source_provider,
        route.target_provider,
        target_shim=route.shim_name,
        upstream_model=model,
        model_capabilities=route.model_capabilities,
        reasoning_config_override=route.reasoning_override,
        supports_custom_tools=route.supports_custom_tools,
        max_tool_description_length=route.max_tool_description_length,
        hoist_system_messages=route.hoist_system_messages,
        image_fetcher=fetch_image,
    )

    # Phase 1+2: Source → IR → Target
    try:
        target_body = pipeline.convert_request(
            body, on_ir_ready=store.inject_into_request
        )
    except ConversionError as exc:
        await dump_error(
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
            request_log_id=entry_id,
        )
        return error_response_for_source(route.source_provider, 400, str(exc)), profile

    profile.update(pipeline.profile)

    await _fidelity_check_and_dump(
        route, body, target_body, "request", profile, model, persistence, entry_id
    )
    log_original_request(pipeline.ir_request)
    if pipeline.warnings:
        logger.warning("Conversion warnings: %s", pipeline.warnings)

    # Per-model system message flattening (gateway config override)
    if route.flatten_system:
        from llm_rosetta.transforms import flatten_system_content

        target_body = flatten_system_content()(target_body)

    log_converted_request(target_body)
    _strip_internal_metadata(target_body)

    selection.with_provider_info(provider_info)
    selection.with_ir_request(pipeline.ir_request)
    provider_info = _apply_request_stages(selection)

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
    # Wrapped in a retry loop: on 401 from a token_command provider,
    # reactively refresh the token and retry once.
    upstream_url = provider_info.upstream_url(model, stream=True)
    target_body = _inject_stream_flags(route.target_provider, target_body, stream=True)
    stream = None
    stream_error_text: str | None = None
    stream_error_code: int = 0
    for _attempt in range(2):
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
            return (
                await _connection_error_response(
                    exc,
                    body=body,
                    target_body=target_body,
                    route=route,
                    provider_info=provider_info,
                    persistence=persistence,
                    entry_id=entry_id,
                    extra_headers=extra_headers,
                    capture_state=capture_state,
                    error_phase="stream_header",
                ),
                profile,
            )

        profile["stream_connect_ms"] = round(
            (time.perf_counter() - t_connect) * 1000, 2
        )

        if not stream.is_error:
            stream_error_text = None
            break

        # Read and close the error stream before deciding on retry
        stream_error_text = await stream.read_error()
        stream_error_code = stream.status_code
        await stream.close()

        retry, err = await _handle_401(
            provider_info,
            stream_error_code,
            stream_error_text,
            route.source_provider,
        )
        if err is not None:
            return err, profile
        if not retry:
            # Same-provider retry only.  Failover for a stream belongs to the
            # window before the first event reaches the client; once bytes are
            # on the wire the response cannot be taken back.
            refusal, _signal = _quota_refusal(
                provider_info, stream_error_code, stream_error_text
            )
            retry = refusal is QuotaRefusal.RETRY
        if not retry:
            break

    # Application-level error — upstream returned a valid HTTP response with
    # a 4xx/5xx status.  Pass the original body through as-is so the client
    # SDK can parse the real error (e.g. "context_length_exceeded").
    # Note: streaming errors only provide the error as a decoded string
    # (via stream.read_error()), not separate raw bytes + parsed dict like
    # the non-streaming path.  We pass the string for both raw_content and
    # error_text; sanitize_upstream_error() accepts str | bytes.
    if stream_error_text is not None:
        return (
            await _upstream_error_passthrough(
                stream_error_code,
                stream_error_text,
                stream_error_text,
                None,
                body=body,
                target_body=target_body,
                route=route,
                provider_info=provider_info,
                persistence=persistence,
                entry_id=entry_id,
                extra_headers=extra_headers,
                capture_state=capture_state,
                error_phase="stream_header",
                is_streaming=True,
                tool_name_collisions=pipeline.tool_name_collisions,
            ),
            profile,
        )

    assert stream is not None

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
    if capture_state is not None and capture_state.enabled:
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
                reported_model=reported_model,
                format_sse=format_sse,
                entry_id=entry_id,
                request_log=request_log,
                metrics=metrics,
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
                    tool_name_collisions=pipeline.tool_name_collisions,
                ),
                pipeline=pipeline,
                warnings_before=len(pipeline.warnings),
                soft_error_patterns=route.soft_error_patterns,
            ),
            content_type="text/event-stream",
        ),
        profile,
    )
