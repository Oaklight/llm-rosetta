"""The chat routes: the shared proxy handler and the five format handlers."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, cast


if TYPE_CHECKING:
    pass

from llm_rosetta_gateway._vendor.httpserver import (
    App,
    Response,
    StreamingResponse,
)
from llm_rosetta.auto_detect import ProviderType
from llm_rosetta.provider_names import canonical_provider_name

from ..middleware.auth import (
    api_key_context_var,
)
from llm_rosetta.routing import ResolvedRoute

from ..config import GatewayConfig
from ..routing import SelectionContext
from ..lifecycle import current_lifecycle, instrumented_handler
from ..transport import ProviderInfo
from ..middleware.headers import (
    build_upstream_extra_headers,
    get_preflight_tokens_override,
)
from ..logging import get_logger

from ..proxy import (
    ProviderMetadataStore,
    detect_stream_request,
    error_response_for_source,
    extract_model,
    handle_non_streaming,
    handle_streaming,
)
from ..maintenance.startup import ProviderNotReady

from . import _state

logger = get_logger()


def _resolve_or_error(
    config: GatewayConfig,
    selection: SelectionContext,
    request_id: str,
) -> tuple[ResolvedRoute, ProviderInfo] | Response:
    """Resolve the selection's model to route+provider, or a Response on failure."""
    source_provider = cast(ProviderType, selection.source_format)
    model = selection.model
    try:
        return config.resolve(selection)
    except ProviderNotReady as exc:
        status = 502 if exc.failed else 503
        resp = error_response_for_source(source_provider, status, str(exc))
        resp.headers["x-request-id"] = request_id
        if not exc.failed:
            resp.headers["Retry-After"] = "5"
        return resp
    except KeyError:
        configured = ", ".join(sorted(config.models.keys()))
        resp = error_response_for_source(
            source_provider,
            404,
            f"Unknown model: '{model}'. Configured models: {configured}",
        )
        resp.headers["x-request-id"] = request_id
        return resp


async def _proxy_handler(
    request: Any,
    source_provider: ProviderType,
    model_override: str | None = None,
    force_stream: bool = False,
) -> Response | StreamingResponse:
    """Shared handler for all proxy endpoints."""
    assert _state._config is not None

    # Legacy provider spellings (e.g. ``openai_chat``) migrate to the
    # canonical format so downstream route/table lookups match.
    source_provider = cast(ProviderType, canonical_provider_name(str(source_provider)))

    async def _handle(
        request: Any, config: GatewayConfig
    ) -> Response | StreamingResponse:
        return await _handle_llm(
            request, config, source_provider, model_override, force_stream
        )

    return await instrumented_handler(
        request,
        _state._config,
        _handle,
        source_format=source_provider,
        error_phase="conversion",
        error_format=source_provider,
    )


async def _handle_llm(
    request: Any,
    config: GatewayConfig,
    source_provider: ProviderType,
    model_override: str | None,
    force_stream: bool,
) -> Response | StreamingResponse:
    """The chat route's own work, run inside the shared lifecycle."""
    ctx = current_lifecycle()
    assert ctx is not None

    body = ctx.body
    if body is None:
        resp = error_response_for_source(source_provider, 400, "Invalid JSON body")
        resp.headers["x-request-id"] = ctx.request_id
        return resp

    # Determine model
    model = model_override or extract_model(source_provider, body)
    if not model:
        resp = error_response_for_source(
            source_provider, 400, "Missing 'model' in request body"
        )
        resp.headers["x-request-id"] = ctx.request_id
        return resp

    # If model came from URL (Google), inject it into body for the converter
    if model_override and "model" not in body:
        body["model"] = model_override

    # Who the request is, read once and carried on the selection context:
    # provider selection needs it now, and the upstream credential and session
    # identity will need it again once the request has been converted.
    _kctx = api_key_context_var.get()
    selection = config.selection_context(model, source_provider)
    selection.with_client(_kctx.key_hash if _kctx else None)

    # Resolve target provider via unified routing
    result = _resolve_or_error(config, selection, ctx.request_id)
    if isinstance(result, Response):
        return result
    route, provider_info = result

    # Model alias: replace the model name in the request body with the
    # actual upstream identifier so the converter and upstream provider
    # both see the correct name.
    if route.upstream_model:
        body["model"] = route.upstream_model

    ctx.note_route(
        model=model,
        target_format=route.target_provider,
        provider_name=route.provider_name,
    )

    # --- Circuit breaker check ---
    cb_reject = ctx.check_circuit_breaker(request)
    if cb_reject is not None:
        return cb_reject

    # Determine streaming
    is_stream = force_stream or detect_stream_request(source_provider, body)
    ctx.note_route(is_stream=is_stream)

    model_label = (
        f"{model} (upstream={route.upstream_model})" if route.upstream_model else model
    )
    logger.info(
        "[%s] %s -> %s | model=%s stream=%s",
        ctx.request_id,
        source_provider,
        route.target_provider,
        model_label,
        is_stream,
    )

    store: ProviderMetadataStore = request.app.metadata_store

    # Forward only explicitly supported client headers to upstream. For
    # providers that declare a per-conversation session header, a client-sent
    # value is preserved (docs recommend forwarding it verbatim); it overrides
    # the gateway-derived value in the transport.
    extra_headers = build_upstream_extra_headers(
        request, ctx.request_id, session_header=provider_info.session_header
    )

    # --- Metrics instrumentation ---
    if is_stream:
        metrics = getattr(request.app, "metrics", None)
        if metrics:
            metrics.active_streams += 1

    selection.with_key_affinity(
        config.provider_key_affinity.get(route.provider_name, True)
    )
    capture_state = getattr(request.app, "capture_state", None)

    if is_stream:
        preflight_override = get_preflight_tokens_override(request)
        preflight = (
            preflight_override
            if preflight_override is not None
            else route.preflight_token_count
        )

        response, profile = await handle_streaming(
            route,
            provider_info,
            body,
            reported_model=model,
            transport=request.app.transport,
            metadata_store=store,
            extra_headers=extra_headers,
            entry_id=ctx.entry_id,
            request_log=getattr(request.app, "request_log", None),
            metrics=getattr(request.app, "metrics", None),
            persistence=ctx.dump_store,
            preflight_token_count=preflight,
            selection=selection,
            capture_state=capture_state,
        )
    else:
        response, profile = await handle_non_streaming(
            route,
            provider_info,
            body,
            transport=request.app.transport,
            metadata_store=store,
            extra_headers=extra_headers,
            persistence=ctx.dump_store,
            entry_id=ctx.entry_id,
            selection=selection,
            capture_state=capture_state,
        )
    ctx.profile = profile
    return response


async def handle_chat_completion(request: Any) -> Response | StreamingResponse:
    return await _proxy_handler(request, source_provider="chat_completions")


async def handle_messages(request: Any) -> Response | StreamingResponse:
    return await _proxy_handler(request, source_provider="messages")


async def handle_open_responses(request: Any) -> Response | StreamingResponse:
    return await _proxy_handler(request, source_provider="open_responses")


async def handle_google_generate(
    request: Any, model_path: str = ""
) -> Response | StreamingResponse:
    if model_path.endswith(":streamGenerateContent"):
        model = model_path.removesuffix(":streamGenerateContent")
        return await _proxy_handler(
            request,
            source_provider="google_generate",
            model_override=model,
            force_stream=True,
        )
    elif model_path.endswith(":generateContent"):
        model = model_path.removesuffix(":generateContent")
        return await _proxy_handler(
            request, source_provider="google_generate", model_override=model
        )
    else:
        return Response(
            body='{"error": "Unknown Google GenAI method"}',
            status_code=404,
            content_type="application/json",
        )


async def handle_google_interactions(
    request: Any,
) -> Response | StreamingResponse:
    return await _proxy_handler(request, source_provider="google_interactions")


_LLM_ROUTE_HANDLERS: dict[str, Callable[..., Any]] = {
    "chat_completions": handle_chat_completion,
    "messages": handle_messages,
    "open_responses": handle_open_responses,
    "google_generate": handle_google_generate,
    "google_interactions": handle_google_interactions,
}


def _register_llm_routes(app: App) -> None:
    """Register the chat routes declared by the ``llm`` model type."""
    from ..model_types import get_model_type

    descriptor = get_model_type("llm")
    if descriptor is None:
        raise RuntimeError("the 'llm' model type is not registered")

    for spec in descriptor.routes:
        handler = _LLM_ROUTE_HANDLERS.get(spec.source_format or "")
        if handler is None:
            raise RuntimeError(
                f"llm route {spec.path!r} names unknown source format "
                f"{spec.source_format!r}"
            )
        app.route(spec.path, methods=spec.methods)(_llm_route_handler(handler))


def _llm_route_handler(handler: Callable[..., Any]) -> Callable[..., Any]:
    """Adapt a chat handler to the router's ``(request, **path_params)`` call."""

    async def _route(request: Any, **path_params: Any) -> Response:
        return await handler(request, **path_params)

    return _route
