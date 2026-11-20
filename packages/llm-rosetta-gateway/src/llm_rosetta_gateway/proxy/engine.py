"""The non-streaming request path and the request stages it shares with streaming."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response

from llm_rosetta.api import (
    ConversionError,
    ConversionPipeline,
    ProviderType,
    UnrepresentableFinishReason,
)
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
    ConversationIdentity,
    ProviderKeyAffinity,
    RoutingChain,
    SelectionContext,
    SessionHeaderAffinity,
)
from ..routing.conversation import note_conversation
from ..transport import (
    ProviderInfo,
    UpstreamConnectionError,
    UpstreamTransport,
)
from .image_fetch import fetch_image
from .errors import (
    RefusalAction,
    _connection_error_response,
    connection_refusal_decision,
    _detect_stream_chunk_error,
    _soft_error_response,
    _upstream_error_passthrough,
    decide_on_refusal,
    error_response_for_source,
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
from .fidelity import (
    _fidelity_check_and_dump,
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
    elif target_provider in ("open_responses", "messages"):
        body["stream"] = True
    elif target_provider == "google_interactions":
        body["stream"] = True
    # Google (generateContent) streaming is signaled via URL, not body
    return body


def detect_stream_request(source_provider: ProviderType, body: dict[str, Any]) -> bool:
    """Detect if the incoming request asks for streaming."""
    if source_provider in (
        "chat_completions",
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


_REQUEST_STAGES = RoutingChain(
    [ConversationIdentity(), ProviderKeyAffinity(), SessionHeaderAffinity()]
)


def _apply_request_stages(ctx: SelectionContext) -> ProviderInfo:
    """Fill in the stage-2 decisions and return the transport config to use.

    The caller sets ``ctx.provider_info`` and ``ctx.ir_request`` first; what
    comes back is the transport config to send with.  The conversation id the
    first stage derived is published for the request log, which is written
    after the handler returns and has no other way to reach it.
    """
    _REQUEST_STAGES.run(ctx)
    note_conversation(ctx.identity.conversation)
    provider_info = ctx.provider_info
    assert provider_info is not None  # the caller sets it before running these
    return provider_info


@dataclass
class PreparedRequest:
    """What both request handlers need before their transports diverge.

    Built by :func:`prepare_request`; the two handlers read it and then take
    their own path — one opens a JSON request, the other an SSE stream.
    """

    pipeline: ConversionPipeline
    target_body: dict[str, Any]
    provider_info: ProviderInfo
    model: str


class RequestRefused(Exception):
    """The prelude answered the request itself; the caller returns the response.

    A conversion failure is the source dialect's own ``400``, and it is
    decided in one place instead of once per handler.
    """

    def __init__(self, response: Response, profile: dict[str, Any]) -> None:
        super().__init__("the request was refused before it reached an upstream")
        self.response = response
        self.profile = profile


async def prepare_request(
    route: ResolvedRoute,
    provider_info: ProviderInfo,
    body: dict[str, Any],
    *,
    metadata_store: ProviderMetadataStore,
    persistence: Any | None,
    entry_id: str | None,
    selection: SelectionContext,
    profile: dict[str, Any],
) -> PreparedRequest:
    """Build the pipeline and the target body both handlers send.

    The steps here are the same whatever the response mode: build the
    pipeline, convert the request to the target dialect, report a conversion
    failure as the source's ``400``, record the request's fidelity, flatten
    the system message when the model asks for it, and run the request
    stages that pick the transport config.

    The body is already wire-ready when the conversion returns: every
    ``*_to_provider`` strips the internal ``_llm_rosetta_provider_metadata``
    carrier at the converter boundary (#941).  The gateway used to repeat
    that walk here; on a production-sized body it was a second full pass
    over the whole history that always found nothing (#1410).

    Args:
        route: The resolved route the request will take.
        provider_info: The provider's transport config, before the stages.
        body: The source-format request body.
        metadata_store: The provider metadata store the conversion writes to.
        persistence: Request-log persistence, for the failure dump.
        entry_id: The request-log entry this request is recorded under.
        selection: The routing context the request stages read and fill.
        profile: The per-request profile dict, updated in place.

    Returns:
        The pipeline, the converted body, the provider info the stages chose,
        and the model the client named.

    Raises:
        RequestRefused: The conversion failed; ``response`` is the answer and
            ``profile`` is the dict to return with it.
    """
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
            body, on_ir_ready=metadata_store.inject_into_request
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
        raise RequestRefused(
            error_response_for_source(route.source_provider, 400, str(exc)), profile
        ) from exc

    profile.update(pipeline.profile)
    # Fields the target required and the request did not name.  The converter
    # records them on its context; the request log carries them on the row, so
    # "the gateway added this" is visible after the fact (#1328).
    injected = pipeline.context.metadata.get("injected")
    if injected:
        profile["injected"] = dict(injected)

    await _fidelity_check_and_dump(
        route, body, target_body, "request", profile, model, persistence, entry_id
    )

    log_original_request(pipeline.ir_request)

    # Per-model system message flattening (gateway config override)
    if route.flatten_system:
        from llm_rosetta.api import flatten_system_content

        target_body = flatten_system_content()(target_body)

    log_converted_request(target_body)

    selection.with_provider_info(provider_info)
    selection.with_ir_request(pipeline.ir_request)
    provider_info = _apply_request_stages(selection)

    return PreparedRequest(
        pipeline=pipeline,
        target_body=target_body,
        provider_info=provider_info,
        model=model,
    )


async def _connection_failure(
    config: Any | None,
    selection: SelectionContext,
    route: ResolvedRoute,
    body: dict[str, Any],
    *,
    provider_info: ProviderInfo,
    exc: UpstreamConnectionError,
    target_body: dict[str, Any],
    transport: UpstreamTransport,
    metadata_store: ProviderMetadataStore | None,
    extra_headers: dict[str, str] | None,
    persistence: Any | None,
    capture_state: CaptureState | None,
    entry_id: str | None,
    profile: dict[str, Any],
    budget: FailoverBudget,
) -> tuple[Response, dict[str, Any]]:
    """Answer a request whose upstream host could not be reached.

    Another lane is tried first when the model opted in and has one left; the
    502 is what the client gets otherwise.  Split from the handler because it
    is one decision in a row of them, and the handler is the row.
    """
    failed_over = await _connection_failover(
        config,
        selection,
        route,
        body,
        provider_info=provider_info,
        exc=exc,
        transport=transport,
        metadata_store=metadata_store,
        extra_headers=extra_headers,
        persistence=persistence,
        capture_state=capture_state,
        entry_id=entry_id,
        profile=profile,
        budget=budget,
    )
    if failed_over is not None:
        return failed_over
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


async def _connection_failover(
    config: Any | None,
    selection: SelectionContext,
    route: ResolvedRoute,
    body: dict[str, Any],
    *,
    provider_info: ProviderInfo,
    exc: UpstreamConnectionError,
    transport: UpstreamTransport,
    metadata_store: ProviderMetadataStore | None,
    extra_headers: dict[str, str] | None,
    persistence: Any | None,
    capture_state: CaptureState | None,
    entry_id: str | None,
    profile: dict[str, Any],
    budget: FailoverBudget,
) -> tuple[Response, dict[str, Any]] | None:
    """Try a request the upstream host never answered on another lane.

    A transport failure carries no upstream status, so it earns the same
    treatment as any other capacity refusal: it is counted against the lane and
    offered to failover.  Split out of the handler because it is one decision,
    and the handler is where a dozen of them meet.

    Returns:
        The attempt's response and merged profile, or ``None`` when the model
        has not opted in or nothing else can serve.
    """
    return await _failover_response(
        config,
        selection,
        route,
        body,
        decision=connection_refusal_decision(
            provider_info,
            exc,
            rate_limit=getattr(config, "routing_rate_limit", None),
            lane=attempted_lane(selection, route, selection.model),
        ),
        transport=transport,
        metadata_store=metadata_store,
        extra_headers=extra_headers,
        persistence=persistence,
        capture_state=capture_state,
        entry_id=entry_id,
        profile=profile,
        budget=budget,
    )


async def _failover_response(
    config: Any | None,
    selection: SelectionContext,
    route: ResolvedRoute,
    body: dict[str, Any],
    *,
    decision: Any,
    transport: UpstreamTransport,
    metadata_store: ProviderMetadataStore | None,
    extra_headers: dict[str, str] | None,
    persistence: Any | None,
    capture_state: CaptureState | None,
    entry_id: str | None,
    profile: dict[str, Any],
    budget: FailoverBudget,
) -> tuple[Response, dict[str, Any]] | None:
    """Serve the request from another lane or provider after a refusal.

    Returns:
        The attempt's response and this request's merged profile, or ``None``
        when there will not be one — failover is off for this model, the
        budget is spent, or nothing else can serve.  The caller then reports
        the upstream's own refusal.
    """
    prepared = prepare_failover(
        config, selection, route, body, decision=decision, budget=budget
    )
    if prepared is None:
        return None
    new_route, new_pinfo, retry_body, next_budget = prepared
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
            _failover_budget=next_budget,
        )
    finally:
        # Recorded even when the second attempt raises: a failure produced by
        # the provider that actually ran must not be attributed to the one
        # that refused.
        record_failover(
            profile,
            route=route,
            new_route=new_route,
            decision=decision,
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


def _stream_envelope(route: ResolvedRoute) -> dict[str, Any]:
    """The echo fields a source dialect's stream will supply, as a body keeps them.

    The catalog names them per format (``FormatSpec.stream_echo_fields``), so
    the gateway does not have to know which dialect repeats which field: a
    Responses stream declares none and the envelope stays empty.  Only chat
    routes reach the stream-required bridge, so the model type is ``llm``.
    """
    from llm_rosetta.api import spec_for

    spec = spec_for("llm", route.source_provider)
    fields = spec.stream_echo_fields if spec is not None else ()
    return dict.fromkeys(fields)


def _gather_echo_fields(chunk: dict[str, Any], envelope: dict[str, Any]) -> None:
    """Keep the echo fields *chunk* states, once each.

    A Chat-compatible stream repeats them on every chunk, so the first chunk
    that states one settles it.  A field the stream never states stays absent
    rather than becoming a guess.
    """
    for key in envelope:
        if envelope[key] is None and chunk.get(key) is not None:
            envelope[key] = chunk[key]


async def _folded_response(
    pipeline: ConversionPipeline,
    ir_events: list[dict[str, Any]],
    *,
    route: ResolvedRoute,
    provider_info: ProviderInfo,
    body: dict[str, Any],
    target_body: dict[str, Any],
    model: str,
    persistence: Any | None,
    entry_id: str | None,
    profile: dict[str, Any],
    envelope: dict[str, Any] | None = None,
) -> dict[str, Any] | Response:
    """Fold a stream-required upstream's events into the source client's body.

    Args:
        envelope: Response-level facts the source stream's chunks carried and
            no IR event does (``service_tier``, ``system_fingerprint``).

    Returns:
        The source-format response body, or the error ``Response`` to send
        when the source writer refuses the folded turn (a terminal state that
        dialect cannot express).  The dump is written here so the JSON bridge
        and the ordinary JSON path report a refusal the same way.
    """
    try:
        return pipeline.convert_stream_events(ir_events, response=envelope)
    except (ConversionError, UnrepresentableFinishReason) as exc:
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
        return error_response_for_source(route.source_provider, 502, str(exc))


async def _non_streaming_from_stream(  # noqa: C901 - one branch per refusal kind
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
    selection: SelectionContext | None = None,
    config: Any | None = None,
    _failover_budget: FailoverBudget | None = None,
) -> tuple[Response, dict[str, Any]]:
    """Serve a non-streaming client from an upstream that only streams.

    A shim declares ``connection.stream_required`` when the vendor refuses
    ``stream: false``.  The request is sent as a stream, the IR events the
    processor produces are folded back into the response they describe, and
    that response is written in the client's own dialect and returned as
    ordinary JSON.

    The fold is format-agnostic, so one implementation serves every source
    format — Chat Completions, Messages, Responses, Google and Interactions
    alike.  It is the same response leg a JSON upstream's answer takes
    (:meth:`ConversionPipeline.convert_stream_events`), so a synthesized reply
    cannot disagree with a real one about the turn's content or its terminal
    reason.

    A refused attempt is classified exactly as on the other two paths: a 401
    a refresh can fix is retried, an in-band quota refusal demotes the
    credential, and a provider with nothing left hands the request to another
    one when the model opted in.  Nothing has reached the client here, so
    failover is always still possible.

    Args:
        selection: This request's routing state.  ``None`` disables failover,
            which is what a bare handler call (a test) gets.
        config: The live config; with *selection*, enables failover.
        _failover_budget: What a failover has already spent.  Set on a retry,
            so the next refusal only gets what is left of the budget.
    """
    budget = _failover_budget or FailoverBudget()
    target_body = _inject_stream_flags(route.target_provider, target_body, stream=True)
    upstream_url = provider_info.upstream_url(model, stream=True)
    stream = None
    error_text = ""
    for _attempt in range(2):
        stream = await transport.send_streaming(
            provider_info, upstream_url, target_body, extra_headers=extra_headers
        )
        if not stream.is_error:
            from ..quota.service import clear_upstream_rate_limit

            clear_upstream_rate_limit(provider_info)
            note_upstream_success(
                provider_info,
                lane=attempted_lane(
                    selection, route, selection.model if selection else model
                ),
            )
            break
        error_text = await stream.read_error()
        status = stream.status_code
        await stream.close()
        stream = None
        # The same decision the JSON and SSE paths take, so a 429 here is not
        # a dead end just because the client asked for one body.
        decision = await decide_on_refusal(
            provider_info,
            status,
            error_text,
            route.source_provider,
            rate_limit=getattr(config, "routing_rate_limit", None),
            lane=attempted_lane(
                selection, route, selection.model if selection else model
            ),
        )
        if decision.action is RefusalAction.RETURN:
            assert decision.response is not None
            return decision.response, profile
        if decision.action is RefusalAction.RETRY:
            continue
        if (
            decision.action is RefusalAction.FAILOVER
            and config is not None
            and selection is not None
        ):
            failed_over = await _failover_response(
                config,
                selection,
                route,
                body,
                decision=decision,
                transport=transport,
                metadata_store=metadata_store,
                extra_headers=extra_headers,
                persistence=persistence,
                capture_state=capture_state,
                entry_id=entry_id,
                profile=profile,
                budget=budget,
            )
            if failed_over is not None:
                return failed_over
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
                tool_name_map=pipeline.tool_name_map,
            ),
            profile,
        )

    store = metadata_store or _default_metadata_store
    ir_events: list[dict[str, Any]] = []

    def _observe(ir_event: dict[str, Any]) -> None:
        """Cache the metadata every event may carry, and keep the IR stream."""
        store.cache_from_stream_event(ir_event)
        ir_events.append(ir_event)

    processor = pipeline.create_stream_processor(on_ir_event=_observe)
    events: list[dict[str, Any]] = []
    envelope = _stream_envelope(route)
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
            _gather_echo_fields(chunk, envelope)
            events.extend(processor.process_chunk(chunk))
    try:
        events.extend(processor.process_chunk(None))
    except Exception:  # noqa: BLE001 - a flush failure is not a lost reply
        logger.debug("Stream flush failed during reassembly", exc_info=True)

    # One reassembly for every source dialect: the IR events the processor
    # produced are folded back into the response they describe, and the
    # ordinary response leg writes that in the client's own format — the same
    # code a JSON upstream's answer takes, so a synthesized reply cannot
    # disagree with a real one.
    merged = await _folded_response(
        pipeline,
        ir_events,
        route=route,
        provider_info=provider_info,
        body=body,
        target_body=target_body,
        model=model,
        persistence=persistence,
        entry_id=entry_id,
        profile=profile,
        envelope=envelope,
    )
    if isinstance(merged, Response):
        return merged, profile
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
    _failover_budget: FailoverBudget | None = None,
) -> tuple[Response, dict[str, Any]]:
    """Non-streaming proxy: convert -> forward -> convert back -> respond.

    Args:
        config: The live config, when the caller has one.  ``None`` disables
            cross-provider failover, which is what a bare handler call (a
            test, an embedder) gets.
        _failover_budget: What a failover has already spent.  Set on a retry
            so the retry budget cannot be exceeded.

    Returns:
        A ``(response, profile)`` tuple.  The profile dict contains
        per-phase timing data merged from the conversion pipeline and
        gateway-level measurements (upstream latency).
    """
    store = metadata_store or _default_metadata_store
    budget = _failover_budget or FailoverBudget()
    profile: dict[str, Any] = {}
    # model was already injected into body by app.py
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
            selection=selection,
            config=config,
            _failover_budget=_failover_budget,
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
            return await _connection_failure(
                config,
                selection,
                route,
                body,
                provider_info=provider_info,
                exc=exc,
                target_body=target_body,
                transport=transport,
                metadata_store=store,
                extra_headers=extra_headers,
                persistence=persistence,
                capture_state=capture_state,
                entry_id=entry_id,
                profile=profile,
                budget=budget,
            )
        profile["upstream_ms"] = round((time.perf_counter() - t_upstream) * 1000, 2)

        if not resp.is_error:
            break

        # One decision for a refused attempt: answer the client, retry the
        # same upstream, or hand the request to another provider.
        decision = await decide_on_refusal(
            provider_info,
            resp.status_code,
            resp.error_text,
            route.source_provider,
            rate_limit=getattr(config, "routing_rate_limit", None),
            lane=attempted_lane(selection, route, selection.model),
        )
        if decision.action is RefusalAction.RETURN:
            assert decision.response is not None
            return decision.response, profile
        if decision.action is RefusalAction.RETRY:
            continue
        # Whether another lane or another provider may serve is the
        # decision's scope; whether this request may move at all is the
        # model's opt-in, and how often is the budget's.
        if decision.action is RefusalAction.FAILOVER:
            failed_over = await _failover_response(
                config,
                selection,
                route,
                body,
                decision=decision,
                transport=transport,
                metadata_store=store,
                extra_headers=extra_headers,
                persistence=persistence,
                capture_state=capture_state,
                entry_id=entry_id,
                profile=profile,
                budget=budget,
            )
            if failed_over is not None:
                return failed_over
        break

    assert resp is not None
    if not resp.is_error:
        # A success clears the lane's 429 count and strike ladder, so one
        # transient refusal does not accumulate across hours.
        from ..quota.service import clear_upstream_rate_limit

        clear_upstream_rate_limit(provider_info)
        note_upstream_success(
            provider_info, lane=attempted_lane(selection, route, selection.model)
        )

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
            tool_name_map=pipeline.tool_name_map,
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
            tool_name_map=pipeline.tool_name_map,
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
    except (ConversionError, UnrepresentableFinishReason) as exc:
        profile.update(pipeline.profile)
        # The turn ended in a terminal state the client's dialect has no
        # finish reason for.  Dump the payload the exception carries so an
        # operator can see *which* upstream response it was, and answer 502:
        # the upstream produced something this gateway cannot represent.
        detail = str(exc)
        reason = getattr(exc, "reason", "")
        if reason:
            detail = (
                f"{detail} (reason={reason!r}, "
                f"provider_metadata={getattr(exc, 'provider_metadata', {})!r})"
            )
        await dump_error(
            persistence,
            request_body=body,
            response_text=detail,
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

    if ir_usage:
        profile["usage"] = ir_usage
    return JSONResponse(source_response), profile
