"""llm-rosetta Gateway — HTTP application and route handlers."""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, cast

from llm_rosetta._vendor.httpserver import (
    App,
    JSONResponse,
    Response,
    StreamingResponse,
)
from llm_rosetta.auto_detect import ProviderType

from .middleware.auth import (
    AuthState,
    api_key_context_var,
    create_auth_hook,
)
from .config import GatewayConfig, ResolvedRoute
from .middleware.error_format import (
    apply_cors_headers,
    detect_api_format,
    format_error_response,
    is_admin_path as _is_admin_path,
)
from .keystore import KeyStore
from .middleware.circuit_breaker import CircuitBreaker
from .middleware.request_context import request_context_var, setup_request_context
from .transport import ProviderInfo
from .middleware.headers import (
    build_upstream_extra_headers,
    get_preflight_tokens_override,
    get_request_id,
)
from .logging import get_logger
from llm_rosetta.observability.error_dump import dump_error

from .proxy import (
    ProviderMetadataStore,
    close_resources,
    detect_stream_request,
    error_response_for_source,
    extract_model,
    handle_non_streaming,
    handle_streaming,
)
from .deferred_startup import ProviderNotReady

logger = get_logger()


def _record_telemetry(
    request: Any,
    *,
    model: str,
    source_provider: ProviderType,
    target_provider: ProviderType,
    provider_name: str,
    is_stream: bool,
    status_code: int,
    duration_ms: float,
    error_detail: str | None,
    profile: dict[str, Any] | None = None,
    entry_id_override: str | None = None,
) -> str | None:
    """Record metrics and request log entry after a proxy call completes.

    Args:
        entry_id_override: Pre-generated entry ID for streaming requests.
            When provided, the entry is created with this ID so the
            stream generator can write back profile data by ID.

    Returns:
        The request log entry ID, or ``None`` if no request log is
        configured.
    """
    metrics = getattr(request.app, "metrics", None)
    if is_stream and metrics:
        metrics.active_streams -= 1
    # Extract usage from profile (non-streaming only; streaming writes
    # back usage separately after the stream completes)
    _usage = (profile or {}).get("usage") if not is_stream else None
    _input_tokens = _usage.get("prompt_tokens") if _usage else None
    _output_tokens = _usage.get("completion_tokens") if _usage else None
    _total_tokens = _usage.get("total_tokens") if _usage else None
    _cache_read_tokens = _usage.get("cache_read_tokens") if _usage else None
    _cache_creation_tokens = _usage.get("cache_creation_tokens") if _usage else None
    _reasoning_tokens = _usage.get("reasoning_tokens") if _usage else None

    if metrics:
        metrics.record_request(
            model=model,
            source=source_provider,
            target=target_provider,
            status_code=status_code,
            duration_ms=duration_ms,
            is_stream=is_stream,
            provider_name=provider_name,
            error_detail=error_detail,
            input_tokens=_input_tokens,
            output_tokens=_output_tokens,
            cache_read_tokens=_cache_read_tokens,
            cache_creation_tokens=_cache_creation_tokens,
            reasoning_tokens=_reasoning_tokens,
        )

    request_log = getattr(request.app, "request_log", None)
    if request_log is not None:
        from dataclasses import replace as _dc_replace

        from llm_rosetta.observability import RequestLogEntry

        entry = RequestLogEntry.create(
            model=model,
            source_provider=source_provider,
            target_provider=target_provider,
            target_provider_name=provider_name,
            is_stream=is_stream,
            status_code=status_code,
            duration_ms=duration_ms,
            error_detail=error_detail,
            api_key_label=(
                _kctx.label if (_kctx := api_key_context_var.get()) else None
            ),
            client_ip=(
                _rctx.client_ip if (_rctx := request_context_var.get()) else None
            ),
            profile=profile,
            input_tokens=_input_tokens,
            output_tokens=_output_tokens,
            total_tokens=_total_tokens,
            cache_read_tokens=_cache_read_tokens,
            cache_creation_tokens=_cache_creation_tokens,
            reasoning_tokens=_reasoning_tokens,
        )
        # For streaming, use the pre-generated ID so the stream
        # generator can write back profile data by this ID.
        if entry_id_override:
            entry = _dc_replace(entry, id=entry_id_override)
        request_log.add(entry)
        return entry.id
    return None


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


def _try_start_profiler(app: Any) -> Any | None:
    """Start a per-request deep profiler if profiling is enabled.

    Returns a started DeepProfiler instance, or ``None`` if profiling
    is disabled or pyinstrument is not installed.
    """
    state = getattr(app, "profiler_state", None)
    if state is None or not state.should_profile():
        return None
    try:
        profiler = state.create_profiler()
        profiler.start()
        return profiler
    except RuntimeError:
        # pyinstrument not installed — restore the consumed slot
        state.remaining += 1
        if not state.enabled:
            state.enabled = True
        return None


def _try_stop_profiler(
    profiler: Any,
    app: Any,
    *,
    request_id: str,
    model: str,
    source: str,
    target: str,
    is_stream: bool,
    duration_ms: float,
) -> None:
    """Stop a running deep profiler and store the result."""
    if profiler is None:
        return
    try:
        profiler.stop()
        state = getattr(app, "profiler_state", None)
        if state is not None:
            state.store_result(
                profiler,
                request_id=request_id,
                model=model,
                source=source,
                target=target,
                is_stream=is_stream,
                duration_ms=duration_ms,
            )
    except Exception:
        logger.debug("Failed to store profiling result")


# Global config — set at startup
_config: GatewayConfig | None = None


def _resolve_or_error(
    source_provider: ProviderType,
    model: str,
    request_id: str,
) -> tuple[ResolvedRoute, ProviderInfo] | Response:
    """Resolve model to route+provider, returning a Response on failure."""
    assert _config is not None
    try:
        return _config.resolve(source_provider, model)
    except ProviderNotReady as exc:
        status = 502 if exc.failed else 503
        resp = error_response_for_source(source_provider, status, str(exc))
        resp.headers["x-request-id"] = request_id
        if not exc.failed:
            resp.headers["Retry-After"] = "5"
        return resp
    except KeyError:
        configured = ", ".join(sorted(_config.models.keys()))
        resp = error_response_for_source(
            source_provider,
            404,
            f"Unknown model: '{model}'. Configured models: {configured}",
        )
        resp.headers["x-request-id"] = request_id
        return resp


def _record_circuit_breaker_outcome(cb: CircuitBreaker, status_code: int) -> None:
    """Record a circuit breaker success or failure based on HTTP status.

    5xx responses are treated as upstream failures.  Everything else
    (including 4xx client errors) counts as a success because the
    upstream is reachable and functioning.
    """
    if status_code >= 500:
        cb.record_failure()
    else:
        cb.record_success()


def _check_circuit_breaker(
    cb: CircuitBreaker,
    provider_name: str,
    request: Any,
    request_id: str,
) -> Response | None:
    """Return a 503 response if the circuit breaker is open, else None."""
    if cb.allow_request():
        return None
    remaining = cb.cooldown_remaining()
    rctx = request_context_var.get()
    api_format = (
        rctx.api_format if rctx and rctx.api_format else detect_api_format(request.path)
    )
    resp = format_error_response(
        api_format,
        503,
        (
            f"Provider '{provider_name}' is temporarily unavailable "
            f"(circuit breaker open, cooldown {remaining:.0f}s remaining)"
        ),
        error_type="service_unavailable",
        google_status="UNAVAILABLE",
        cors=True,
    )
    resp.headers["x-request-id"] = request_id
    resp.headers["Retry-After"] = str(max(1, int(remaining)))
    return resp


async def _proxy_handler(
    request: Any,
    source_provider: ProviderType,
    model_override: str | None = None,
    force_stream: bool = False,
) -> Response | StreamingResponse:
    """Shared handler for all proxy endpoints."""
    assert _config is not None

    # Read request ID from context (populated by the early middleware
    # hook) and fall back to header extraction for safety.
    rctx = request_context_var.get()
    request_id = rctx.request_id if rctx else get_request_id(request)

    try:
        body: dict[str, Any] = request.json()
    except Exception:
        resp = error_response_for_source(source_provider, 400, "Invalid JSON body")
        resp.headers["x-request-id"] = request_id
        return resp

    # Determine model
    model = model_override or extract_model(source_provider, body)
    if not model:
        resp = error_response_for_source(
            source_provider, 400, "Missing 'model' in request body"
        )
        resp.headers["x-request-id"] = request_id
        return resp

    # If model came from URL (Google), inject it into body for the converter
    if model_override and "model" not in body:
        body["model"] = model_override

    # Resolve target provider via unified routing
    result = _resolve_or_error(source_provider, model, request_id)
    if isinstance(result, Response):
        return result
    route, provider_info = result

    # Model alias: replace the model name in the request body with the
    # actual upstream identifier so the converter and upstream provider
    # both see the correct name.
    if route.upstream_model:
        body["model"] = route.upstream_model

    # --- Circuit breaker check ---
    cb = _config.circuit_breaker_registry.get_or_create(route.provider_name)
    cb_reject = _check_circuit_breaker(cb, route.provider_name, request, request_id)
    if cb_reject is not None:
        return cb_reject

    # Determine streaming
    is_stream = force_stream or detect_stream_request(source_provider, body)

    model_label = (
        f"{model} (upstream={route.upstream_model})" if route.upstream_model else model
    )
    logger.info(
        "[%s] %s -> %s | model=%s stream=%s",
        request_id,
        source_provider,
        route.target_provider,
        model_label,
        is_stream,
    )

    store: ProviderMetadataStore = request.app.metadata_store

    # Forward only explicitly supported client headers to upstream.
    extra_headers = build_upstream_extra_headers(request, request_id)

    # --- Metrics instrumentation ---
    if is_stream:
        metrics = getattr(request.app, "metrics", None)
        if metrics:
            metrics.active_streams += 1

    t0 = time.monotonic()
    status_code = 500
    error_detail: str | None = None
    profile: dict[str, Any] | None = None
    request_log = getattr(request.app, "request_log", None)

    # Error dump persistence — pass None to disable dump_error in handlers
    _raw_persistence = getattr(request.app, "persistence", None)
    persistence = _raw_persistence if _config.error_dumps_enabled else None
    deep_profiler = _try_start_profiler(request.app)

    # Shared across streaming / non-streaming paths — also stored on
    # request.state so lifecycle hooks can write back to the log entry.
    pre_entry_id = uuid.uuid4().hex
    request.state.log_entry_id = pre_entry_id
    _kctx = api_key_context_var.get()
    _client_key_hash = _kctx.key_hash if _kctx else ""
    _key_affinity = _config.provider_key_affinity.get(route.provider_name, True)

    try:
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
                transport=request.app.transport,
                metadata_store=store,
                extra_headers=extra_headers,
                entry_id=pre_entry_id,
                request_log=request_log,
                metrics=getattr(request.app, "metrics", None),
                persistence=persistence,
                preflight_token_count=preflight,
                client_key_hash=_client_key_hash,
                key_affinity=_key_affinity,
            )
        else:
            response, profile = await handle_non_streaming(
                route,
                provider_info,
                body,
                transport=request.app.transport,
                metadata_store=store,
                extra_headers=extra_headers,
                persistence=persistence,
                entry_id=pre_entry_id,
                client_key_hash=_client_key_hash,
                key_affinity=_key_affinity,
            )
        status_code = response.status_code
        if status_code >= 400 and hasattr(response, "body"):
            body_bytes = response.body
            if isinstance(body_bytes, bytes):
                error_detail = body_bytes.decode("utf-8", errors="replace")
        response.headers["x-request-id"] = request_id
        logger.info("[%s] response status=%s", request_id, status_code)

        _record_circuit_breaker_outcome(cb, status_code)

        # For streaming responses, defer profiler stop to after the
        # generator is fully consumed via StreamingResponse.background.
        if deep_profiler is not None and isinstance(response, StreamingResponse):
            _dp = deep_profiler
            _app = request.app
            _t0 = t0
            _rid = request_id
            _model = model
            _src = source_provider
            _tgt = route.target_provider

            def _stop_profiler_background() -> None:
                _try_stop_profiler(
                    _dp,
                    _app,
                    request_id=_rid,
                    model=_model,
                    source=_src,
                    target=_tgt,
                    is_stream=True,
                    duration_ms=(time.monotonic() - _t0) * 1000,
                )

            response.background = _stop_profiler_background
            deep_profiler = None  # prevent finally from double-stopping

        return response
    except Exception as exc:
        error_detail = str(exc)
        logger.exception("[%s] unhandled error in proxy handler", request_id)
        status_code = 500
        cb.record_failure()
        dump_error(
            persistence,
            request_body=body,
            response_text=error_detail,
            model=model,
            source_provider=source_provider,
            target_provider=route.target_provider,
            provider_name=route.provider_name,
            status_code=500,
            error_phase="conversion",
            request_log_id=pre_entry_id,
        )
        resp = error_response_for_source(
            source_provider, 500, f"Internal server error: {exc}"
        )
        resp.headers["x-request-id"] = request_id
        return resp
    finally:
        duration_ms = (time.monotonic() - t0) * 1000

        _try_stop_profiler(
            deep_profiler,
            request.app,
            request_id=request_id,
            model=model,
            source=source_provider,
            target=route.target_provider,
            is_stream=is_stream,
            duration_ms=duration_ms,
        )

        _record_telemetry(
            request,
            model=model,
            source_provider=source_provider,
            target_provider=route.target_provider,
            provider_name=route.provider_name,
            is_stream=is_stream,
            status_code=status_code,
            duration_ms=duration_ms,
            error_detail=error_detail,
            profile=profile,
            entry_id_override=pre_entry_id,
        )


# --- Endpoint handlers ---


async def handle_openai_chat(request: Any) -> Response | StreamingResponse:
    return await _proxy_handler(request, source_provider="openai_chat")


async def handle_anthropic(request: Any) -> Response | StreamingResponse:
    return await _proxy_handler(request, source_provider="anthropic")


async def handle_openai_responses(request: Any) -> Response | StreamingResponse:
    return await _proxy_handler(request, source_provider="openai_responses")


async def handle_google_generate(
    request: Any, model_path: str = ""
) -> Response | StreamingResponse:
    if model_path.endswith(":streamGenerateContent"):
        model = model_path.removesuffix(":streamGenerateContent")
        return await _proxy_handler(
            request,
            source_provider="google",
            model_override=model,
            force_stream=True,
        )
    elif model_path.endswith(":generateContent"):
        model = model_path.removesuffix(":generateContent")
        return await _proxy_handler(
            request, source_provider="google", model_override=model
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


async def handle_list_models(request: Any) -> Response:
    """List configured models in a format compatible with OpenAI and Anthropic SDKs.

    Supports an optional ``?type=`` query parameter to filter by model type:

    * ``type=llm`` — only LLM models (default pool)
    * ``type=embedding`` — only embedding models
    * ``type=rerank`` — only rerank models
    * ``type=all`` or omitted — all model types
    """
    assert _config is not None

    # Parse optional type filter from query params
    type_filter = None
    qp = getattr(request, "query_params", {})
    type_vals = qp.get("type")
    if type_vals:
        type_filter = type_vals[0] if isinstance(type_vals, list) else type_vals
        if type_filter == "all":
            type_filter = None

    data = []

    # LLM models (main pool)
    if type_filter is None or type_filter == "llm":
        for name in sorted(_config.models.keys()):
            model_route = _config.models[name]
            provider_name = model_route.providers[0].name
            api_standard = _config.provider_types.get(provider_name, "unknown")
            capabilities = _config.model_capabilities.get(name, ["text"])
            entry: dict[str, Any] = {
                "id": name,
                "object": "model",
                "created": 0,
                "owned_by": provider_name,
                "api_standard": api_standard,
                "capabilities": capabilities,
                "type": "llm",
                "display_name": name,
                "created_at": "1970-01-01T00:00:00Z",
            }
            if model_route.is_multi:
                entry["providers"] = [
                    {"name": p.name, "weight": p.weight} for p in model_route.providers
                ]
                entry["routing_strategy"] = "weighted_round_robin"
            data.append(entry)

    # Embedding models
    if type_filter is None or type_filter == "embedding":
        for name in sorted(_config.embedding_models.keys()):
            provider_name = _config.embedding_models[name]
            api_standard = _config.provider_types.get(provider_name, "unknown")
            data.append(
                {
                    "id": name,
                    "object": "model",
                    "created": 0,
                    "owned_by": provider_name,
                    "api_standard": api_standard,
                    "capabilities": ["embedding"],
                    "type": "embedding",
                    "display_name": name,
                    "created_at": "1970-01-01T00:00:00Z",
                }
            )

    # Rerank models
    if type_filter is None or type_filter == "rerank":
        for name in sorted(_config.rerank_models.keys()):
            provider_name = _config.rerank_models[name]
            api_standard = _config.provider_types.get(provider_name, "unknown")
            data.append(
                {
                    "id": name,
                    "object": "model",
                    "created": 0,
                    "owned_by": provider_name,
                    "api_standard": api_standard,
                    "capabilities": ["rerank"],
                    "type": "rerank",
                    "display_name": name,
                    "created_at": "1970-01-01T00:00:00Z",
                }
            )

    all_ids = [d["id"] for d in data]
    return JSONResponse(
        {
            "object": "list",
            "data": data,
            "has_more": False,
            "first_id": all_ids[0] if all_ids else None,
            "last_id": all_ids[-1] if all_ids else None,
        }
    )


async def handle_list_models_google(request: Any) -> Response:
    """List configured models in Google GenAI SDK format.

    Supports the same ``?type=`` filter as :func:`handle_list_models`.
    """
    assert _config is not None

    # Parse optional type filter from query params
    type_filter = None
    qp = getattr(request, "query_params", {})
    type_vals = qp.get("type")
    if type_vals:
        type_filter = type_vals[0] if isinstance(type_vals, list) else type_vals
        if type_filter == "all":
            type_filter = None

    models_list: list[dict[str, Any]] = []

    if type_filter is None or type_filter == "llm":
        for name in sorted(_config.models.keys()):
            models_list.append(
                {
                    "name": f"models/{name}",
                    "displayName": name,
                    "supportedGenerationMethods": [
                        "generateContent",
                        "streamGenerateContent",
                    ],
                }
            )

    if type_filter is None or type_filter == "embedding":
        for name in sorted(_config.embedding_models.keys()):
            models_list.append(
                {
                    "name": f"models/{name}",
                    "displayName": name,
                    "supportedGenerationMethods": ["embedContent"],
                }
            )

    if type_filter is None or type_filter == "rerank":
        for name in sorted(_config.rerank_models.keys()):
            models_list.append(
                {
                    "name": f"models/{name}",
                    "displayName": name,
                    "supportedGenerationMethods": ["rerank"],
                }
            )

    return JSONResponse({"models": models_list})


async def handle_health(request: Any) -> Response:
    """Return aggregate health status without exposing provider details.

    Always returns HTTP 200. Use ``status: "degraded"`` in the payload
    to signal provider issues without breaking existing monitors.
    For a 503-on-unhealthy probe use ``/health/ready``.

    Per-provider health details are available via the authenticated
    admin metrics endpoint (``/admin/api/metrics``).
    """
    metrics = getattr(request.app, "metrics", None)
    if metrics is None:
        return JSONResponse({"status": "ok"})

    snap = metrics.snapshot(series_seconds=3600)
    errors_last_hour = sum(
        pt["errors"] for pt in snap.get("series", []) if pt.get("errors", 0)
    )

    critical = metrics.any_critical_provider()
    overall_status = "degraded" if critical else "ok"

    deferred = getattr(request.app, "deferred_startup", None)
    payload = {
        "status": overall_status,
        "ready": deferred.is_fully_ready() if deferred else True,
        "uptime_seconds": snap["uptime_seconds"],
        "requests_total": snap["total_requests"],
        "errors_last_hour": errors_last_hour,
        "detail_url": "/admin/api/metrics",
    }
    return JSONResponse(payload, status_code=200)


async def handle_health_live(request: Any) -> Response:
    """Kubernetes liveness probe — always 200 while the process is up."""
    return JSONResponse({"status": "ok"})


async def handle_health_ready(request: Any) -> Response:
    """Kubernetes readiness probe — 200 if fully operational, 503 if not.

    Returns 503 when deferred startup tasks are still running (token
    seeding, counter rebuild, backfills) or when provider health has
    degraded critically.
    """
    deferred = getattr(request.app, "deferred_startup", None)
    if deferred is not None and not deferred.is_fully_ready():
        return JSONResponse(
            {"status": "not_ready", **deferred.status()},
            status_code=503,
        )

    metrics = getattr(request.app, "metrics", None)
    if metrics is None:
        return JSONResponse({"status": "ok"})

    health = metrics.provider_health_snapshot()
    critical_count = sum(1 for v in health.values() if v.get("status") == "critical")
    if critical_count:
        return JSONResponse(
            {"status": "not_ready", "critical_providers": critical_count},
            status_code=503,
        )
    return JSONResponse({"status": "ready"})


# ---------------------------------------------------------------------------
# Registry-driven route registration for non-LLM types
# ---------------------------------------------------------------------------


def _register_non_llm_routes(app: App, config: GatewayConfig) -> None:
    """Register HTTP routes for all non-LLM model types from the registry.

    For each non-LLM descriptor whose ``pipeline`` is set, resolves the
    handler via ``desc.pipeline()`` (a lazy import wrapper) and registers
    its routes.  Descriptors with ``pipeline=None`` are skipped with a
    warning — this is expected only for the ``llm`` type, which uses
    ``_proxy_handler`` with per-format route registration.

    The wrapper adds the same cross-cutting concerns that
    ``_proxy_handler`` provides for LLM routes:

    * Telemetry recording (metrics + request log)
    * Deep profiling (``_try_start_profiler`` / ``_try_stop_profiler``)
    * Error dumps (``dump_error``) on unhandled exceptions
    """
    from .model_types import all_model_types

    for desc in all_model_types():
        if desc.name == "llm":
            continue

        if desc.pipeline is None:
            logger.warning(
                "Model type %r has no pipeline — routes skipped",
                desc.name,
            )
            continue

        # Resolve the handler via the lazy import wrapper stored in
        # the descriptor.  The wrapper defers imports to avoid circular
        # dependencies between model_types and handler modules.
        raw_handler = desc.pipeline()

        # Wrap the raw handler to inject the gateway config AND add
        # telemetry / profiling / error-dump instrumentation, matching
        # the cross-cutting concerns that ``_proxy_handler`` provides.
        def _make_handler(h: Callable, type_name: str) -> Callable:
            async def _handler(request: Any) -> Response:
                assert _config is not None

                rctx = request_context_var.get()
                request_id = rctx.request_id if rctx else get_request_id(request)

                # Best-effort model extraction from JSON body
                model = ""
                try:
                    body: dict[str, Any] = request.json()
                    model = body.get("model", "") or ""
                except Exception:
                    pass

                t0 = time.monotonic()
                status_code = 500
                error_detail: str | None = None
                deep_profiler = _try_start_profiler(request.app)

                _raw_persistence = getattr(request.app, "persistence", None)
                persistence = _raw_persistence if _config.error_dumps_enabled else None

                try:
                    response = await h(request, _config)
                    status_code = response.status_code
                    if status_code >= 400 and hasattr(response, "body"):
                        body_bytes = response.body
                        if isinstance(body_bytes, bytes):
                            error_detail = body_bytes.decode("utf-8", errors="replace")
                    return response
                except Exception as exc:
                    error_detail = str(exc)
                    logger.exception(
                        "[%s] unhandled error in %s handler",
                        request_id,
                        type_name,
                    )
                    status_code = 500
                    dump_error(
                        persistence,
                        request_body=None,
                        response_text=error_detail,
                        model=model or None,
                        source_provider=type_name,
                        target_provider=type_name,
                        status_code=500,
                        error_phase="handler",
                    )
                    from .proxy import error_response_for_source

                    resp = error_response_for_source(
                        "openai_chat",
                        500,
                        f"Internal server error: {exc}",
                    )
                    resp.headers["x-request-id"] = request_id
                    return resp
                finally:
                    duration_ms = (time.monotonic() - t0) * 1000

                    _try_stop_profiler(
                        deep_profiler,
                        request.app,
                        request_id=request_id,
                        model=model,
                        source=type_name,
                        target=type_name,
                        is_stream=False,
                        duration_ms=duration_ms,
                    )

                    _record_telemetry(
                        request,
                        model=model,
                        source_provider=cast(ProviderType, type_name),
                        target_provider=cast(ProviderType, type_name),
                        provider_name=type_name,
                        is_stream=False,
                        status_code=status_code,
                        duration_ms=duration_ms,
                        error_detail=error_detail,
                    )

            return _handler

        handler = _make_handler(raw_handler, desc.name)

        for route_spec in desc.routes:
            app.route(route_spec.path, methods=route_spec.methods)(handler)


# ---------------------------------------------------------------------------
# Persistence flush helpers
# ---------------------------------------------------------------------------

_FLUSH_METRICS_INTERVAL = 30  # seconds


async def _periodic_flush(app: App) -> None:
    """Periodically flush metrics counters to disk."""
    while True:
        await asyncio.sleep(_FLUSH_METRICS_INTERVAL)
        persistence = getattr(app, "persistence", None)
        if persistence is None:
            continue
        metrics = getattr(app, "metrics", None)
        if metrics is not None:
            try:
                if persistence.check_and_clear_rebuild_flag():
                    logger.info(
                        "Rebuild flag detected (external cleanup), rebuilding counters"
                    )
                    metrics.rebuild_counters(persistence.iter_log_rows_for_rebuild())
                persistence.save_metrics(metrics.export_counters())
            except Exception as exc:
                logger.warning("Failed to flush metrics: %s", exc)


def _flush_now(app: App) -> None:
    """Final synchronous flush on shutdown."""
    persistence = getattr(app, "persistence", None)
    if persistence is None:
        return

    metrics = getattr(app, "metrics", None)
    if metrics is not None:
        try:
            persistence.save_metrics(metrics.export_counters())
        except Exception as exc:
            logger.warning("Shutdown: failed to flush metrics: %s", exc)

    # Record shutdown event (skip pruning to avoid latency before close)
    ops_log = getattr(app, "ops_log", None)
    if ops_log is not None:
        from llm_rosetta.observability.ops_log import (
            EVENT_SHUTDOWN,
            OpsLogEntry,
            SEVERITY_INFO,
            SOURCE_GATEWAY,
        )

        ops_log.add(
            OpsLogEntry.create(
                event_type=EVENT_SHUTDOWN,
                severity=SEVERITY_INFO,
                message="Gateway shutting down",
                source=SOURCE_GATEWAY,
            ),
            _skip_prune=True,
        )

    persistence.close()

    keystore = getattr(app, "keystore", None)
    if keystore is not None:
        keystore.close()

    logger.info("Persistence flushed and closed on shutdown")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _resolve_data_dir_for_app(
    config: GatewayConfig, config_path: str | None
) -> str | None:
    """Resolve data directory from config, anchoring relative paths to config dir."""
    import os

    if config.data_dir:
        if config_path and not os.path.isabs(config.data_dir):
            return os.path.join(os.path.dirname(config_path), config.data_dir)
        return config.data_dir
    if config_path:
        return os.path.join(os.path.dirname(config_path), "data")
    return None


def _setup_auth(
    config: GatewayConfig,
    config_path: str | None,
    data_dir: str | None = None,
) -> tuple[str, KeyStore, AuthState]:
    """Create KeyStore, import config keys, build AuthState."""
    import os
    import secrets

    internal_token = f"rsk-internal-{secrets.token_hex(16)}"

    if config.api_keys_db:
        keys_db_path = config.api_keys_db
    elif data_dir:
        new_path = os.path.join(data_dir, "keys.db")
        old_path = (
            os.path.join(os.path.dirname(os.path.abspath(config_path)), "keys.db")
            if config_path
            else None
        )
        if os.path.exists(new_path):
            keys_db_path = new_path
        elif old_path and os.path.exists(old_path):
            keys_db_path = old_path
            logger.warning(
                "Using keys.db from legacy location %s — "
                "move it to %s to silence this warning.",
                old_path,
                new_path,
            )
        else:
            keys_db_path = new_path
    elif config_path:
        keys_db_path = os.path.join(
            os.path.dirname(os.path.abspath(config_path)), "keys.db"
        )
    else:
        keys_db_path = "keys.db"

    keystore = KeyStore(keys_db_path)

    auth_state = AuthState(
        keystore=keystore,
        internal_token=internal_token,
        admin_password=config.admin_password,
        open_on_no_keys=config.open_on_no_keys,
    )
    if not auth_state._has_keys():
        _no_key_msg = (
            "No API keys configured — all /v1/* endpoints are OPEN "
            "(server.open_on_no_keys=true). Set server.api_keys in your "
            "config or generate keys in the admin panel to restrict access."
            if config.open_on_no_keys
            else "No API keys configured — all /v1/* endpoints are BLOCKED. "
            "Generate keys in the admin panel, set server.api_keys, or "
            "set server.open_on_no_keys=true to allow anonymous access."
        )
        logger.warning(_no_key_msg)

    return internal_token, keystore, auth_state


def _install_root_redirect(app: App, target: str | None) -> None:
    """Register a GET / → *target* redirect when *target* is set."""
    if not target:
        return

    @app.route("/", methods=["GET"])
    async def root_redirect(request: Any) -> Response:
        return Response(body=b"", status_code=307, headers={"Location": target})


@dataclass
class GatewayExtensions:
    """Extension points for downstream projects that wrap the gateway.

    Pass an instance to :func:`create_app` to customise transport, auth,
    middleware, routes, and admin panel without duplicating the factory.
    All fields are optional — defaults preserve the standard gateway behaviour.
    """

    transport: Any | None = None
    """Pre-built transport instance.  When *None*, an :class:`HttpTransport`
    is created from *config.upstream_timeout*."""

    before_hooks: list[Callable] = field(default_factory=list)
    """Extra ``before_request`` hooks registered **after** built-in auth."""

    after_hooks: list[Callable] = field(default_factory=list)
    """Extra ``after_request`` hooks registered **after** built-in CORS."""

    extra_routes: list[tuple[str, list[str], Callable]] = field(default_factory=list)
    """Additional routes as ``(path, methods, handler)`` tuples."""

    branding: dict | None = None
    """Admin panel branding dict forwarded to :func:`setup_admin`."""

    custom_head: str | None = None
    """Extra ``<head>`` HTML injected into the admin page."""

    disabled_tabs: list[str] | None = None
    """Admin tabs to hide (e.g. ``["keys"]``)."""

    config_io: Any | None = None
    """Custom :class:`ConfigIO` for admin config persistence."""

    max_body_size: int | None = None
    """Override the default 50 MB max request body size."""

    enable_rate_limiting: bool = True
    """Set to *False* to skip built-in rate-limiting middleware."""

    skip_default_routes: bool = False
    """When *True*, the standard proxy routes are **not** registered.
    Use this when the downstream project provides its own handlers."""

    skip_builtin_auth: bool = False
    """When *True*, the API-key auth hook is **not** registered.
    ``_setup_auth`` still runs (admin needs *internal_token*)."""

    skip_admin_setup: bool = False
    """When *True*, :func:`setup_admin` is **not** called.
    The downstream project is responsible for calling it later
    (e.g. after async initialisation)."""


def _install_cors(app: App, admin_cors_origins: list[str]) -> None:
    """Register CORS after-request hook and OPTIONS preflight handler."""

    def _apply_cors(response: Any, origin: str | None) -> None:
        if admin_cors_origins and origin and origin in admin_cors_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Methods"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "*"

    @app.after_request
    async def add_cors_headers(request: Any, response: Any) -> Any:
        if _is_admin_path(request.path):
            _apply_cors(response, request.headers.get("origin"))
            if request.path.startswith("/admin/api/"):
                response.headers.setdefault(
                    "Cache-Control", "no-cache, no-store, must-revalidate"
                )
        else:
            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "*"
        return response

    @app.route("/<path:_path>", methods=["OPTIONS"])
    async def cors_preflight(request: Any, _path: str = "") -> Response:
        resp = Response(body=b"", status_code=204)
        if not _is_admin_path(request.path):
            resp.headers["Access-Control-Allow-Origin"] = "*"
            resp.headers["Access-Control-Allow-Methods"] = "*"
            resp.headers["Access-Control-Allow-Headers"] = "*"
        else:
            _apply_cors(resp, request.headers.get("origin"))
        return resp


def _install_rate_limiting(app: App, config: GatewayConfig) -> None:
    """Set up rate-limiting before/after hooks and attach state to *app*."""
    from .middleware.ratelimit import (
        RateLimitState,
        create_rate_limit_after_hook,
        create_rate_limit_hook,
    )

    rate_limit_state = RateLimitState()
    rate_limit_state.rebuild(config)
    app.before_request(create_rate_limit_hook(rate_limit_state))
    app.rate_limit_state = rate_limit_state  # type: ignore
    app.after_request(create_rate_limit_after_hook())


def _install_lifecycle_hooks(app: App) -> None:
    """Register httpserver response lifecycle hooks for TTFB and disconnect tracking."""

    @app.on_response_started
    async def _on_response_started(request: Any, response: Any) -> None:
        ctx = request_context_var.get()
        if ctx is None or ctx.request_start is None:
            return
        ttfb_ms = round((time.monotonic() - ctx.request_start) * 1000, 2)
        request.state.ttfb_ms = ttfb_ms

    @app.on_response_completed
    async def _on_response_completed(request: Any, response: Any) -> None:
        ttfb_ms = getattr(request.state, "ttfb_ms", None)
        entry_id = getattr(request.state, "log_entry_id", None)
        if ttfb_ms is not None and entry_id is not None:
            request_log = getattr(request.app, "request_log", None)
            if request_log is not None:
                request_log.update_profile(entry_id, {"ttfb_ms": ttfb_ms})

    @app.on_client_disconnect
    async def _on_client_disconnect(request: Any) -> None:
        metrics = getattr(request.app, "metrics", None)
        if metrics is not None:
            metrics.record_disconnect()

        entry_id = getattr(request.state, "log_entry_id", None)
        if entry_id is not None:
            request_log = getattr(request.app, "request_log", None)
            if request_log is not None:
                request_log.update_profile(entry_id, {"client_disconnected": True})

        ctx = request_context_var.get()
        logger.debug(
            "Client disconnected mid-response: client=%s path=%s request_id=%s",
            ctx.client_ip if ctx else "unknown",
            request.path,
            ctx.request_id if ctx else "unknown",
        )


def create_app(
    config: GatewayConfig,
    config_path: str | None = None,
    extensions: GatewayExtensions | None = None,
) -> App:
    """Create the httpserver application."""
    global _config
    _config = config
    ext = extensions or GatewayExtensions()

    # Expose global proxy as env vars so downstream code (e.g. image
    # downloads in converters) can use it without threading config through.
    import os

    if config.proxy:
        os.environ.setdefault("HTTP_PROXY", config.proxy)
        os.environ.setdefault("HTTPS_PROXY", config.proxy)

    from .transport import HttpTransport

    metadata_store = ProviderMetadataStore()
    transport = (
        ext.transport
        if ext.transport is not None
        else HttpTransport(timeout=config.upstream_timeout)
    )

    app = App(
        max_body_size=ext.max_body_size
        if ext.max_body_size is not None
        else 50_000_000,
        read_timeout=config.read_timeout,
    )

    # --- Routes ---
    if not ext.skip_default_routes:
        # LLM routes: explicit registration (complex per-format handler logic)
        app.route("/v1/chat/completions", methods=["POST"])(handle_openai_chat)
        app.route("/v1/messages", methods=["POST"])(handle_anthropic)
        app.route("/v1/responses", methods=["POST"])(handle_openai_responses)
        app.route("/v1beta/models/<path:model_path>", methods=["POST"])(
            handle_google_generate
        )
        app.route("/v1beta/interactions", methods=["POST"])(handle_google_interactions)

        # Non-LLM routes: registered from the model type registry
        _register_non_llm_routes(app, config)

        # Utility / listing / health routes
        app.route("/v1/models", methods=["GET"])(handle_list_models)
        app.route("/v1beta/models", methods=["GET"])(handle_list_models_google)
        app.route("/health", methods=["GET"])(handle_health)
        app.route("/health/live", methods=["GET"])(handle_health_live)
        app.route("/health/ready", methods=["GET"])(handle_health_ready)

    _install_root_redirect(app, config.root_redirect)

    # --- Resolve data directory (shared by keystore + persistence) ---
    resolved_data_dir = _resolve_data_dir_for_app(config, config_path)

    # --- Request context (earliest hook — before auth and rate limiting) ---
    app.before_request(setup_request_context(trust_proxy=config.rate_limit_trust_proxy))

    # --- Auth (SQLite keystore + config fallback) ---
    internal_token, keystore, auth_state = _setup_auth(
        config, config_path, data_dir=resolved_data_dir
    )
    if not ext.skip_builtin_auth:
        app.before_request(create_auth_hook(auth_state))

    # --- Rate Limiting (after auth so api_key_context_var is set) ---
    if ext.enable_rate_limiting:
        _install_rate_limiting(app, config)

    # --- Extension hooks (after built-in auth/rate-limit) ---
    for hook in ext.before_hooks:
        app.before_request(hook)

    _install_cors(app, config.admin_cors_origins)

    for hook in ext.after_hooks:
        app.after_request(hook)

    # --- Extension routes ---
    for path, methods, handler in ext.extra_routes:
        app.route(path, methods=methods)(handler)

    # --- .well-known (always registered, not a proxy route) ---
    @app.route("/.well-known/change-password", methods=["GET"])
    async def well_known_change_password(request: Any) -> Response:
        return Response(
            body=b"", status_code=302, headers={"Location": "/admin#change-password"}
        )

    @app.errorhandler(404)
    async def handle_404(request: Any, exc: Any) -> Response:
        resp = JSONResponse({"error": "Not Found"}, status_code=404)
        if not _is_admin_path(request.path):
            apply_cors_headers(resp)
        return resp

    @app.errorhandler(405)
    async def handle_405(request: Any, exc: Any) -> Response:
        resp = JSONResponse({"error": "Method Not Allowed"}, status_code=405)
        if not _is_admin_path(request.path):
            apply_cors_headers(resp)
        return resp

    # --- Admin routes ---
    from .admin import setup_admin
    from .admin.routes import register_admin_routes

    register_admin_routes(app)

    # --- App-level state ---
    app.transport = transport  # type: ignore
    app.metadata_store = metadata_store  # type: ignore
    app.internal_token = internal_token  # type: ignore
    app.auth_state = auth_state  # type: ignore
    app.keystore = keystore  # type: ignore

    if not ext.skip_admin_setup:
        setup_admin(
            app,
            config,
            config_path,
            config_io=ext.config_io,
            disabled_tabs=ext.disabled_tabs,
            custom_head=ext.custom_head,
            branding=ext.branding,
            data_dir=resolved_data_dir,
        )

    # --- Response lifecycle hooks (httpserver 0.5.0+) ---
    _install_lifecycle_hooks(app)

    return app


async def run_gateway(
    app: App,
    host: str,
    port: int,
    *,
    socket: str | None = None,
    ssl_context: Any | None = None,
) -> None:
    """Start the gateway with lifecycle management."""
    # Expose bind address so admin test tasks can self-call.
    setattr(app, "_bind_host", host)
    setattr(app, "_bind_port", port)
    # Record startup event
    ops_log = getattr(app, "ops_log", None)
    if ops_log is not None:
        from llm_rosetta.observability.ops_log import (
            EVENT_STARTUP,
            OpsLogEntry,
            SEVERITY_INFO,
            SOURCE_GATEWAY,
        )

        config = getattr(app, "gateway_config", None)
        ops_log.add(
            OpsLogEntry.create(
                event_type=EVENT_STARTUP,
                severity=SEVERITY_INFO,
                message=f"Gateway starting on {host}:{port}",
                details={
                    "host": host,
                    "port": port,
                    "provider_count": len(config.providers) if config else 0,
                    "model_count": len(config.models) if config else 0,
                },
                source=SOURCE_GATEWAY,
            )
        )

    flush_task = asyncio.create_task(_periodic_flush(app))

    # Deferred startup: token seeding, counter rebuild, backfills
    from .deferred_startup import DeferredStartup

    config = getattr(app, "gateway_config", None)
    deferred = DeferredStartup(config, app) if config else None
    if deferred is not None:
        setattr(app, "deferred_startup", deferred)
        # Expose on config so resolve() can check provider states
        setattr(config, "_deferred_startup", deferred)
        await deferred.start()

    # Start background refresh loops for providers WITHOUT token_command
    # (those with token_command are handled by DeferredStartup.seed_tokens)
    from .transport.token_refresh import start_token_refreshers

    static_providers = {
        name: pinfo
        for name, pinfo in (config.providers.items() if config else {})
        if pinfo.token_command is None
    }
    refresh_tasks = await start_token_refreshers(static_providers)

    try:
        await app._serve(host, port, socket=socket, ssl_context=ssl_context)
    finally:
        if deferred is not None:
            await deferred.shutdown()
        # Cancel token refresh tasks (both static and deferred-startup ones)
        all_refresh = refresh_tasks + getattr(app, "_token_refresh_tasks", [])
        for task in all_refresh:
            task.cancel()
        flush_task.cancel()
        for task in all_refresh:
            try:
                await task
            except asyncio.CancelledError:
                pass
        try:
            await flush_task
        except asyncio.CancelledError:
            pass
        _flush_now(app)
        await close_resources(
            transport=app.transport,  # type: ignore
            metadata_store=app.metadata_store,  # type: ignore
        )
