"""The non-streaming request path and the request stages it shares with streaming."""

from __future__ import annotations

import time
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response

from llm_rosetta.auto_detect import ProviderType
from llm_rosetta.pipeline import ConversionError, ConversionPipeline
from llm_rosetta.routing import ResolvedRoute

from llm_rosetta_gateway.observability.capture import CapturedRequest, CaptureState
from llm_rosetta_gateway.observability.error_dump import dump_error

from ..logging import (
    get_logger,
    log_converted_request,
    log_original_request,
    log_response,
)
from ..routing import (
    ProviderKeyAffinity,
    RoutingChain,
    SelectionContext,
    SessionHeaderAffinity,
)
from ..transport import (
    ProviderInfo,
    UpstreamConnectionError,
    UpstreamTransport,
)
from .image_fetch import fetch_image
from .errors import (
    QuotaRefusal,
    _connection_error_response,
    _detect_stream_chunk_error,
    _handle_401,
    _quota_refusal,
    _soft_error_response,
    _upstream_error_passthrough,
    error_response_for_source,
)
from .failover import prepare_failover, record_failover
from .metadata import (
    ProviderMetadataStore,
    _default_metadata_store,
    _strip_internal_metadata,
)
from .fidelity import (
    _fidelity_check_and_dump,
)
from .stream_merge import (
    merge_chat_completion_events,
)

logger = get_logger()


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
    if target_provider == "chat_completions":
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    elif target_provider in ("openai_responses", "open_responses", "messages"):
        body["stream"] = True
    elif target_provider == "google_interactions":
        body["stream"] = True
    # Google (generateContent) streaming is signaled via URL, not body
    return body


def detect_stream_request(source_provider: ProviderType, body: dict[str, Any]) -> bool:
    """Detect if the incoming request asks for streaming."""
    if source_provider in (
        "chat_completions",
        "openai_responses",
        "open_responses",
        "messages",
        "google_interactions",
    ):
        return bool(body.get("stream", False))
    # Google (generateContent) streaming is determined by the endpoint path, not the body
    return False


def extract_model(source_provider: ProviderType, body: dict[str, Any]) -> str | None:
    """Extract the model name from a source-format request body."""
    return body.get("model")


_REQUEST_STAGES = RoutingChain([ProviderKeyAffinity(), SessionHeaderAffinity()])


def _apply_request_stages(ctx: SelectionContext) -> ProviderInfo:
    """Fill in the stage-2 decisions and return the transport config to use.

    The caller sets ``ctx.provider_info`` and ``ctx.ir_request`` first; what
    comes back is the transport config to send with.
    """
    _REQUEST_STAGES.run(ctx)
    provider_info = ctx.provider_info
    assert provider_info is not None  # the caller sets it before running these
    return provider_info


async def _failover_response(
    config: Any | None,
    selection: SelectionContext,
    route: ResolvedRoute,
    body: dict[str, Any],
    *,
    signal: Any,
    transport: UpstreamTransport,
    metadata_store: ProviderMetadataStore | None,
    extra_headers: dict[str, str] | None,
    persistence: Any | None,
    capture_state: CaptureState | None,
    entry_id: str | None,
    profile: dict[str, Any],
    attempted: str | None,
) -> tuple[Response, dict[str, Any]] | None:
    """Serve the request from another provider after a quota refusal.

    Returns:
        The second attempt's response and this request's merged profile, or
        ``None`` when there will not be a second attempt — failover is off for
        this model, has already happened, or nothing else can serve.  The
        caller then reports the upstream's own refusal.
    """
    prepared = prepare_failover(
        config, selection, route, body, signal=signal, attempted=attempted
    )
    if prepared is None:
        return None
    new_route, new_pinfo, retry_body = prepared
    sub_profile: dict[str, Any] = {}
    try:
        response, sub_profile = await handle_non_streaming(
            new_route,
            new_pinfo,
            retry_body,
            transport=transport,
            metadata_store=metadata_store,
            extra_headers=extra_headers,
            persistence=persistence,
            capture_state=capture_state,
            entry_id=entry_id,
            selection=selection,
            config=config,
            _failover_from=route.provider_name,
        )
    finally:
        # Recorded even when the second attempt raises: a failure produced by
        # the provider that actually ran must not be attributed to the one
        # that refused.
        record_failover(
            profile,
            route=route,
            new_route=new_route,
            signal=signal,
            sub_profile=sub_profile,
        )
    return response, profile


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


async def _non_streaming_from_stream(
    *,
    route: ResolvedRoute,
    provider_info: ProviderInfo,
    pipeline: ConversionPipeline,
    target_body: dict[str, Any],
    body: dict[str, Any],
    model: str,
    transport: UpstreamTransport,
    metadata_store: ProviderMetadataStore | None,
    extra_headers: dict[str, str] | None,
    persistence: Any | None,
    capture_state: CaptureState | None,
    entry_id: str | None,
    profile: dict[str, Any],
) -> tuple[Response, dict[str, Any]]:
    """Serve a non-streaming client from an upstream that only streams.

    A shim declares ``connection.stream_required`` when the vendor refuses
    ``stream: false``.  The request is sent as a stream, the source-format
    events are reassembled into one body, and the result is returned as
    ordinary JSON.  Only Chat Completions clients are supported: every other
    source format has a different terminal-event shape, and synthesizing one
    without a measured sample would be guesswork.
    """
    if route.source_provider != "chat_completions":
        return (
            error_response_for_source(
                route.source_provider,
                400,
                "This upstream only answers streaming requests, and the gateway "
                "can only reassemble a non-streaming reply for the Chat "
                "Completions format. Retry with stream=true.",
            ),
            profile,
        )

    target_body = _inject_stream_flags(route.target_provider, target_body, stream=True)
    upstream_url = provider_info.upstream_url(model, stream=True)
    stream = None
    error_text = ""
    for _attempt in range(2):
        stream = await transport.send_streaming(
            provider_info, upstream_url, target_body, extra_headers=extra_headers
        )
        if not stream.is_error:
            break
        error_text = await stream.read_error()
        status = stream.status_code
        await stream.close()
        stream = None
        retry, err = await _handle_401(
            provider_info, status, error_text, route.source_provider
        )
        if err is not None:
            return err, profile
        if not retry:
            break
    if stream is None:
        return (
            await _upstream_error_passthrough(
                status,
                error_text,
                error_text,
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

    store = metadata_store or _default_metadata_store
    processor = pipeline.create_stream_processor(
        on_ir_event=store.cache_from_stream_event,
    )
    events: list[dict[str, Any]] = []
    async with stream:
        async for chunk in stream:
            stream_error = await _detect_stream_chunk_error(
                chunk,
                soft_error_patterns=route.soft_error_patterns,
                stream_status=getattr(stream, "status_code", 200),
                model=model,
                dump_ctx=None,
                entry_id=entry_id,
            )
            if stream_error is not None:
                return (
                    error_response_for_source(
                        route.source_provider, 502, stream_error.message
                    ),
                    profile,
                )
            events.extend(processor.process_chunk(chunk))
    try:
        events.extend(processor.process_chunk(None))
    except Exception:  # noqa: BLE001 - a flush failure is not a lost reply
        logger.debug("Stream flush failed during reassembly", exc_info=True)

    merged = merge_chat_completion_events(events)
    usage = getattr(processor, "get_accumulated_usage", lambda: None)()
    if usage:
        profile["usage"] = usage
    if capture_state is not None:
        capture_state.record(
            CapturedRequest(
                original_request=body,
                converted_body=target_body,
                upstream_response=events,
                request_id=(
                    extra_headers.get("x-request-id", "") if extra_headers else ""
                ),
                model=model,
                source_provider=route.source_provider,
                target_provider=route.target_provider,
                is_stream=True,
                status_code=200,
            )
        )
    return JSONResponse(merged), profile


async def handle_non_streaming(  # noqa: C901
    route: ResolvedRoute,
    provider_info: ProviderInfo,
    body: dict[str, Any],
    *,
    transport: UpstreamTransport,
    metadata_store: ProviderMetadataStore | None = None,
    extra_headers: dict[str, str] | None = None,
    persistence: Any | None = None,
    capture_state: CaptureState | None = None,
    entry_id: str | None = None,
    selection: SelectionContext,
    config: Any | None = None,
    _failover_from: str | None = None,
) -> tuple[Response, dict[str, Any]]:
    """Non-streaming proxy: convert -> forward -> convert back -> respond.

    Args:
        config: The live config, when the caller has one.  ``None`` disables
            cross-provider failover, which is what a bare handler call (a
            test, an embedder) gets.
        _failover_from: The provider a failover is already coming from.  Set
            on the second attempt so the retry budget cannot be exceeded.

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

    # Per-model system message flattening (gateway config override)
    if route.flatten_system:
        from llm_rosetta.transforms import flatten_system_content

        target_body = flatten_system_content()(target_body)

    log_converted_request(target_body)
    _strip_internal_metadata(target_body)

    selection.with_provider_info(provider_info)
    selection.with_ir_request(pipeline.ir_request)
    provider_info = _apply_request_stages(selection)

    # ``is True`` rather than truthiness: a mock provider in a test would
    # otherwise present a truthy attribute and divert the request.
    if getattr(provider_info, "stream_required", False) is True:
        return await _non_streaming_from_stream(
            route=route,
            provider_info=provider_info,
            pipeline=pipeline,
            target_body=target_body,
            body=body,
            model=model,
            transport=transport,
            metadata_store=store,
            extra_headers=extra_headers,
            persistence=persistence,
            capture_state=capture_state,
            entry_id=entry_id,
            profile=profile,
        )

    # Phase 3: Forward to upstream via transport
    # Wrapped in a retry loop: on 401 from a token_command provider,
    # reactively refresh the token and retry once.
    upstream_url = provider_info.upstream_url(model)
    resp = None
    for _attempt in range(2):
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
                ),
                profile,
            )
        profile["upstream_ms"] = round((time.perf_counter() - t_upstream) * 1000, 2)

        if not resp.is_error:
            break

        # Reactive 401 retry — _handle_401 is a no-op for non-401 errors
        # and for providers without token_command.
        retry, err = await _handle_401(
            provider_info,
            resp.status_code,
            resp.error_text,
            route.source_provider,
        )
        if err is not None:
            return err, profile
        refusal, signal = (
            (None, None)
            if retry
            else _quota_refusal(provider_info, resp.status_code, resp.error_text)
        )
        if retry or refusal is QuotaRefusal.RETRY:
            continue
        # No credential of this provider can serve the request any more.  A
        # second provider is a different prompt cache, so trying one is the
        # model's opt-in, and the budget is one.
        if refusal is QuotaRefusal.EXHAUSTED:
            failed_over = await _failover_response(
                config,
                selection,
                route,
                body,
                signal=signal,
                transport=transport,
                metadata_store=store,
                extra_headers=extra_headers,
                persistence=persistence,
                capture_state=capture_state,
                entry_id=entry_id,
                profile=profile,
                attempted=_failover_from,
            )
            if failed_over is not None:
                return failed_over
        break

    assert resp is not None

    # Check for upstream errors (HTTP 4xx/5xx) or soft-errors (HTTP 200
    # with an error embedded in the body, e.g. Argo auth warnings).
    _err = (
        await _upstream_error_passthrough(
            resp.status_code,
            resp.error_text,
            resp.raw_content,
            resp.body,
            body=body,
            target_body=target_body,
            route=route,
            provider_info=provider_info,
            persistence=persistence,
            entry_id=entry_id,
            extra_headers=extra_headers,
            capture_state=capture_state,
            tool_name_collisions=pipeline.tool_name_collisions,
        )
        if resp.is_error
        else await _soft_error_response(
            route,
            resp.body,
            body,
            target_body,
            provider_info=provider_info,
            persistence=persistence,
            entry_id=entry_id,
            extra_headers=extra_headers,
            capture_state=capture_state,
            tool_name_collisions=pipeline.tool_name_collisions,
        )
    )
    if _err is not None:
        return _err, profile

    # Phase 4: Target response → Source response
    assert resp.body is not None
    log_response(resp.body, label="UPSTREAM RESPONSE")
    ir_usage: dict[str, Any] = {}

    def _capture_ir(ir_response: dict[str, Any]) -> None:
        store.cache_from_response(ir_response)
        u = ir_response.get("usage")
        if u:
            ir_usage.update(u)

    try:
        source_response = pipeline.convert_response(resp.body, on_ir_ready=_capture_ir)
    except ConversionError as exc:
        profile.update(pipeline.profile)
        await dump_error(
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
            request_log_id=entry_id,
        )
        return error_response_for_source(route.source_provider, 502, str(exc)), profile

    # Merge response-phase timings from pipeline
    profile.update(pipeline.profile)

    await _fidelity_check_and_dump(
        route,
        resp.body,
        source_response,
        "response",
        profile,
        model,
        persistence,
        entry_id,
    )

    # Content capture (non-streaming): record all three stages
    if capture_state is not None:
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
    if ir_usage:
        profile["usage"] = ir_usage
    return JSONResponse(source_response), profile
