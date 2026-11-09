"""The streaming path: event generator, terminal errors, usage write-back."""

from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import Response, StreamingResponse

from llm_rosetta.auto_detect import ProviderType
from llm_rosetta.routing import ResolvedRoute
from llm_rosetta.shims.provider_shim import SoftErrorPattern

from llm_rosetta_gateway.observability.capture import CapturedRequest, CaptureState
from llm_rosetta.observability.usage import usage_columns

from ..logging import (
    get_logger,
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
from .errors import (
    RefusalAction,
    RefusalDecision,
    RefusalScope,
    _connection_error_response,
    _first_token_timeout_response,
    _detect_stream_chunk_error,
    _upstream_error_passthrough,
    DumpContext,
    decide_on_refusal,
    note_upstream_success,
)
from .failover import (
    FailoverBudget,
    attempted_lane,
    prepare_failover,
    record_failover,
)
from .metadata import (
    ProviderMetadataStore,
    _default_metadata_store,
)
from .engine import (
    RequestRefused,
    _inject_stream_flags,
    prepare_request,
)
from .preflight import (
    _run_preflight,
)

logger = get_logger()

#: What an in-band notice says when the converter refused a complete turn
#: (a terminal state the client's dialect cannot express): the upstream
#: finished, and blaming the stream would misdiagnose it.
_REFUSAL_MESSAGE = (
    "The upstream response ended in a state this endpoint cannot express: {reason}"
)


async def _write_back_stream_meta(
    request_log: Any,
    entry_id: str,
    profile: dict[str, Any],
    ttfb_ms: float | None,
) -> None:
    """Record what only the stream body knows, after the row already exists.

    Both writes are best-effort: a log that cannot take them must not turn a
    served response into a failed one.

    Args:
        request_log: The log the row went into.
        entry_id: The row to update.
        profile: Stream metrics to merge into the profile JSON.
        ttfb_ms: Milliseconds to the first chunk, or ``None`` if none came.
    """
    try:
        await request_log.update_profile(entry_id, profile)
    except Exception:
        logger.debug("Failed to write stream profile for %s", entry_id)
    # The column, not just the profile: the rate view reads it per row, and a
    # JSON lookup on every row would be a scan.
    try:
        await request_log.update_ttft(entry_id, ttfb_ms)
    except Exception:
        logger.debug("Failed to write ttft for %s", entry_id)


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

        response_id = getattr(ctx, "outbound_response_id", "") or ""
        status = 500
        message: str | None = None
        if getattr(exc, "reason", ""):
            # Not a gateway fault: the upstream produced a terminal state the
            # client's dialect has no finish reason for, so the JSON leg would
            # have answered 502.  Say the same in-band (#1237), carrying the id
            # the converter knew rather than the one the stream never learned.
            # The words for the status come from the shared classification
            # table, so this leg cannot drift from the JSON envelope.
            status = 502
            response_id = getattr(exc, "response_id", "") or response_id
            # The stream did not die: the turn arrived complete and the
            # gateway has no word for how it ended.  Saying "the upstream
            # stream ended before completion" would blame the wrong party.
            message = _REFUSAL_MESSAGE.format(reason=str(exc))

        events = build_stream_error_events(
            source_provider,
            str(exc) or exc.__class__.__name__,
            response_id=response_id,
            sequence_number=getattr(ctx, "next_sequence_number", None),
            status=status,
            message=message,
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


async def _close_quietly(stream: Any) -> None:
    """Close an abandoned upstream without letting the close replace the answer.

    A socket that went quiet is the one most likely to refuse a clean close;
    the 504 or the failover is the answer the client is owed either way.

    Args:
        stream: The upstream stream to release.
    """
    try:
        await stream.close()
    except Exception:  # noqa: BLE001 - the deadline's answer outranks the close
        logger.warning(
            "Closing the upstream after a deadline miss failed", exc_info=True
        )


async def _first_chunk_within_budget(
    *,
    budget_ms: int | None,
    stream: Any,
    sent_at: float,
) -> tuple[dict[str, Any] | None, AsyncIterator[dict[str, Any]], bool]:
    """Pull the upstream's first chunk, under the model's deadline if any.

    Args:
        budget_ms: The model's first-token budget, or ``None`` when it sets
            none.
        stream: The upstream stream the transport just opened.
        sent_at: ``time.perf_counter()`` taken before the request was sent; the
            budget is measured from there.

    Returns:
        ``(first_chunk, iterator, timed_out)``.  The iterator is the one the
        chunk was read from, so the caller continues where this stopped; an
        exhausted stream hands back ``None`` and is not a timeout.
    """
    iterator = stream.__aiter__()
    if budget_ms is None:
        return None, iterator, False
    remaining = budget_ms / 1000 - (time.perf_counter() - sent_at)
    try:
        chunk = await asyncio.wait_for(iterator.__anext__(), max(0.0, remaining))
    except asyncio.TimeoutError:
        return None, iterator, True
    except StopAsyncIteration:
        return None, iterator, False
    return chunk, iterator, False


@dataclass(frozen=True, slots=True)
class FirstTokenDeadline:
    """A lane that accepted the stream and sent nothing within the budget.

    Shaped like the refusal signals the failover path already takes: it only
    has to say, in words a request log can carry, why the request moved.
    """

    provider: str
    budget_ms: int

    @property
    def reason(self) -> str:
        """The words the profile records for this failure."""
        return f"no first token within {self.budget_ms} ms"


def _first_token_budget(selection: SelectionContext) -> int | None:
    """The model's first-token deadline in milliseconds, when it sets one.

    Args:
        selection: This request's routing context, which carries the model's
            block.

    Returns:
        The budget, or ``None`` when the model sets none.  A value the model
        does not set, or one the parser would have refused, never applies.
    """
    route = getattr(selection, "model_route", None)
    routing = getattr(route, "routing", None)
    budget = getattr(routing, "first_token_ms", None)
    return budget if isinstance(budget, int) and budget > 0 else None


async def _stream_chunks(
    stream: Any,
    chunk_iterator: AsyncIterator[dict[str, Any]] | None,
    first_chunk: dict[str, Any] | None,
) -> AsyncIterator[dict[str, Any]]:
    """Yield the upstream's chunks, starting with the one the deadline pulled.

    The deadline path consumes the first chunk before the response exists;
    replaying it here keeps one place that sees every upstream event, so error
    detection, capture and the TTFB reading all still start at chunk one.

    Args:
        stream: The upstream stream, used when no iterator was handed over.
        chunk_iterator: The iterator the deadline already advanced, if any.
        first_chunk: The chunk that iterator produced, if any.

    Yields:
        The upstream's JSON chunks, in order.
    """
    if first_chunk is not None:
        yield first_chunk
    source = chunk_iterator if chunk_iterator is not None else stream
    async for item in source:
        yield item


async def _stream_event_generator(
    *,
    source_provider: ProviderType,
    stream: Any,
    processor: Any,
    model: str,
    format_sse: Any,
    chunk_iterator: AsyncIterator[dict[str, Any]] | None = None,
    first_chunk: dict[str, Any] | None = None,
    started_at: float | None = None,
    reported_model: str | None = None,
    entry_id: str | None = None,
    request_log: Any | None = None,
    metrics: Any | None = None,
    capture_record: CapturedRequest | None = None,
    capture_state: CaptureState | None = None,
    dump_ctx: DumpContext | None = None,
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
    # The clock the first chunk is measured against: the moment the upstream
    # request was sent, so the reading spans connect, headers and the chunk,
    # the same way the deadline does.
    t_stream_open = started_at if started_at is not None else time.perf_counter()
    upstream_chunks: list[dict[str, Any]] | None = (
        [] if capture_record is not None else None
    )

    try:
        async with stream:
            async for chunk in _stream_chunks(stream, chunk_iterator, first_chunk):
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
            await _write_back_stream_meta(
                request_log, entry_id, stream_profile, ttfb_ms
            )

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


async def _failover_stream(
    config: Any | None,
    selection: SelectionContext,
    route: ResolvedRoute,
    body: dict[str, Any],
    *,
    reported_model: str | None,
    decision: Any,
    transport: UpstreamTransport,
    metadata_store: ProviderMetadataStore | None,
    extra_headers: dict[str, str] | None,
    entry_id: str | None,
    request_log: Any | None,
    metrics: Any | None,
    persistence: Any | None,
    capture_state: CaptureState | None,
    preflight_token_count: bool,
    profile: dict[str, Any],
    budget: FailoverBudget,
) -> tuple[Response | StreamingResponse, dict[str, Any]] | None:
    """Serve this stream from another lane or provider, before the first event.

    Reached only while the upstream has refused and nothing has been written
    to the client, which is what makes the second answer honest.

    Returns:
        The attempt's response and this request's merged profile, or ``None``
        when there will not be one.
    """
    prepared = prepare_failover(
        config, selection, route, body, decision=decision, budget=budget
    )
    if prepared is None:
        return None
    new_route, new_pinfo, retry_body, next_budget = prepared
    sub_profile: dict[str, Any] = {}
    try:
        response, sub_profile = await handle_streaming(
            new_route,
            new_pinfo,
            retry_body,
            reported_model=reported_model,
            transport=transport,
            metadata_store=metadata_store,
            extra_headers=extra_headers,
            entry_id=entry_id,
            request_log=request_log,
            metrics=metrics,
            persistence=persistence,
            capture_state=capture_state,
            preflight_token_count=preflight_token_count,
            selection=selection,
            config=config,
            _failover_budget=next_budget,
        )
    finally:
        # Same rule as the non-streaming path: whichever provider ran owns
        # the outcome, including a failure.
        record_failover(
            profile,
            route=route,
            new_route=new_route,
            decision=decision,
            sub_profile=sub_profile,
        )
    return response, profile


async def handle_streaming(  # noqa: C901
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
    config: Any | None = None,
    _failover_budget: FailoverBudget | None = None,
) -> tuple[Response | StreamingResponse, dict[str, Any]]:
    """Streaming proxy: convert -> forward -> stream-convert back -> SSE.

    ``config`` enables cross-provider failover on a quota refusal.  It can only
    happen here, before the ``StreamingResponse`` exists: once the first event
    is on the wire the answer cannot be retracted, so a mid-stream failure ends
    the client's stream with its code (see #1060).

    The model's ``first_token_ms`` is enforced at the same point: the first
    upstream chunk is pulled under that deadline, and a lane that misses it is
    closed before anything reaches the client, so the request can still move
    to another provider or answer ``504``.

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
    budget = _failover_budget or FailoverBudget()
    profile: dict[str, Any] = {}
    # model was already injected into body by app.py, and app.py rewrites it to
    # the upstream id before calling us.  ``reported_model`` carries the name
    # the client asked for, so usage is grouped under the model the user named
    # rather than under whatever the upstream calls it.
    try:
        prepared = await prepare_request(
            route,
            provider_info,
            body,
            metadata_store=store,
            persistence=persistence,
            entry_id=entry_id,
            selection=selection,
            profile=profile,
        )
    except RequestRefused as refused:
        return refused.response, refused.profile
    pipeline = prepared.pipeline
    target_body = prepared.target_body
    provider_info = prepared.provider_info
    model = prepared.model
    reported_model = reported_model or model

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
            from ..quota.service import clear_upstream_rate_limit

            clear_upstream_rate_limit(provider_info)
            note_upstream_success(
                provider_info, lane=attempted_lane(selection, route, selection.model)
            )
            stream_error_text = None
            break

        # Read and close the error stream before deciding on retry
        stream_error_text = await stream.read_error()
        stream_error_code = stream.status_code
        await stream.close()

        decision = await decide_on_refusal(
            provider_info,
            stream_error_code,
            stream_error_text,
            route.source_provider,
            rate_limit=getattr(config, "routing_rate_limit", None),
            lane=attempted_lane(selection, route, selection.model),
        )
        if decision.action is RefusalAction.RETURN:
            assert decision.response is not None
            return decision.response, profile
        if decision.action is RefusalAction.RETRY:
            continue
        # Nothing was written to the client yet, so another provider can still
        # answer — if this model opted in and has one.
        if decision.action is RefusalAction.FAILOVER:
            failed_over = await _failover_stream(
                config,
                selection,
                route,
                body,
                reported_model=reported_model,
                decision=decision,
                transport=transport,
                metadata_store=store,
                extra_headers=extra_headers,
                entry_id=entry_id,
                request_log=request_log,
                metrics=metrics,
                persistence=persistence,
                capture_state=capture_state,
                preflight_token_count=preflight_token_count,
                profile=profile,
                budget=budget,
            )
            if failed_over is not None:
                return failed_over
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

    # A lane that returns headers and then goes quiet is as unusable as one
    # that refused.  The first chunk is pulled here, while nothing has been
    # written to the client, which is the only point at which another lane
    # can still answer.
    budget_ms = _first_token_budget(selection)
    first_chunk, chunk_iterator, timed_out = await _first_chunk_within_budget(
        budget_ms=budget_ms, stream=stream, sent_at=t_connect
    )
    if timed_out:
        assert budget_ms is not None  # only a budget can time out
        profile["first_token_timeout_ms"] = budget_ms
        await _close_quietly(stream)
        logger.warning(
            "First-token deadline of %d ms missed on %s for model %s",
            budget_ms,
            route.provider_name,
            selection.model,
        )
        failed_over = await _failover_stream(
            config,
            selection,
            route,
            body,
            reported_model=reported_model,
            # A lane that accepted the stream and then went quiet is a
            # content-level failure: its siblings may still answer.
            decision=RefusalDecision(
                RefusalAction.FAILOVER,
                signal=FirstTokenDeadline(route.provider_name, budget_ms),
                scope=RefusalScope.LANE,
            ),
            transport=transport,
            metadata_store=store,
            extra_headers=extra_headers,
            entry_id=entry_id,
            request_log=request_log,
            metrics=metrics,
            persistence=persistence,
            capture_state=capture_state,
            preflight_token_count=preflight_token_count,
            profile=profile,
            budget=budget,
        )
        if failed_over is not None:
            return failed_over
        return (
            await _first_token_timeout_response(
                body=body,
                target_body=target_body,
                route=route,
                provider_info=provider_info,
                persistence=persistence,
                entry_id=entry_id,
                capture_state=capture_state,
                budget_ms=budget_ms,
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
                chunk_iterator=chunk_iterator,
                first_chunk=first_chunk,
                started_at=t_connect,
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
                soft_error_patterns=route.soft_error_patterns,
            ),
            content_type="text/event-stream",
        ),
        profile,
    )
