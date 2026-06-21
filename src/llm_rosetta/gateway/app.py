"""llm-rosetta Gateway — HTTP application and route handlers."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, cast

from llm_rosetta._vendor.httpserver import (
    App,
    JSONResponse,
    Response,
    StreamingResponse,
)
from llm_rosetta.auto_detect import ProviderType

from .auth import AuthState, api_key_label_var, create_auth_hook
from .config import GatewayConfig
from .embeddings import handle_embeddings as _handle_embeddings
from .logging import get_logger
from .proxy import (
    ProviderMetadataStore,
    close_resources,
    detect_stream_request,
    error_response_for_source,
    extract_model,
    handle_non_streaming,
    handle_streaming,
)

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
) -> None:
    """Record metrics and request log entry after a proxy call completes."""
    metrics = getattr(request.app, "metrics", None)
    if is_stream and metrics:
        metrics.active_streams -= 1
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
        )

    request_log = getattr(request.app, "request_log", None)
    if request_log is not None:
        from .admin.request_log import RequestLogEntry

        request_log.add(
            RequestLogEntry.create(
                model=model,
                source_provider=source_provider,
                target_provider=target_provider,
                target_provider_name=provider_name,
                is_stream=is_stream,
                status_code=status_code,
                duration_ms=duration_ms,
                error_detail=error_detail,
                api_key_label=api_key_label_var.get(),
                client_ip=_extract_client_ip(request),
            )
        )


def _extract_client_ip(request: Any) -> str | None:
    """Extract the client IP from the request.

    Checks ``X-Forwarded-For`` and ``X-Real-IP`` headers first (set by
    reverse proxies), then falls back to the TCP peer address.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        # X-Forwarded-For may contain a chain: "client, proxy1, proxy2"
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    addr = getattr(request, "client_addr", None)
    if addr and isinstance(addr, (tuple, list)) and addr[0]:
        return str(addr[0])
    return None


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------

# Global config — set at startup
_config: GatewayConfig | None = None


async def _proxy_handler(
    request: Any,
    source_provider: ProviderType,
    model_override: str | None = None,
    force_stream: bool = False,
) -> Response | StreamingResponse:
    """Shared handler for all proxy endpoints."""
    assert _config is not None

    # Generate or honour a request ID for end-to-end traceability.
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())

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

    # Resolve target provider
    try:
        (
            target_provider_str,
            provider_info,
            target_shim_name,
            upstream_model,
            provider_name,
        ) = _config.resolve_model(model)
        target_provider: ProviderType = cast(ProviderType, target_provider_str)
    except KeyError:
        configured = ", ".join(sorted(_config.models.keys()))
        resp = error_response_for_source(
            source_provider,
            404,
            f"Unknown model: '{model}'. Configured models: {configured}",
        )
        resp.headers["x-request-id"] = request_id
        return resp

    # Model alias: replace the model name in the request body with the
    # actual upstream identifier so the converter and upstream provider
    # both see the correct name.
    if upstream_model:
        body["model"] = upstream_model

    # Determine streaming
    is_stream = force_stream or detect_stream_request(source_provider, body)

    model_label = f"{model} (upstream={upstream_model})" if upstream_model else model
    logger.info(
        "[%s] %s -> %s | model=%s stream=%s",
        request_id,
        source_provider,
        target_provider,
        model_label,
        is_stream,
    )

    store: ProviderMetadataStore = request.app.metadata_store

    # Forward OpenResponses-Version header and request ID to upstream if present
    extra_headers: dict[str, str] = {"x-request-id": request_id}
    or_version = request.headers.get("openresponses-version")
    if or_version:
        extra_headers["OpenResponses-Version"] = or_version

    # --- Metrics instrumentation ---
    if is_stream:
        metrics = getattr(request.app, "metrics", None)
        if metrics:
            metrics.active_streams += 1

    t0 = time.monotonic()
    status_code = 500
    error_detail: str | None = None

    try:
        handler = handle_streaming if is_stream else handle_non_streaming
        # Resolve config-level reasoning override (keyed by gateway model name)
        reasoning_override = _config.model_reasoning_overrides.get(model)
        model_caps = _config.model_capabilities.get(model, ["text"])
        response = await handler(
            source_provider,
            target_provider,
            provider_info,
            body,
            model,
            metadata_store=store,
            extra_headers=extra_headers,
            target_shim_name=target_shim_name,
            reasoning_config_override=reasoning_override,
            model_capabilities=model_caps,
        )
        status_code = response.status_code
        if status_code >= 400 and hasattr(response, "body"):
            body_bytes = response.body
            if isinstance(body_bytes, bytes):
                error_detail = body_bytes.decode("utf-8", errors="replace")
        response.headers["x-request-id"] = request_id
        logger.info("[%s] response status=%s", request_id, status_code)
        return response
    except Exception as exc:
        error_detail = str(exc)
        logger.exception("[%s] unhandled error in proxy handler", request_id)
        status_code = 500
        resp = error_response_for_source(
            source_provider, 500, f"Internal server error: {exc}"
        )
        resp.headers["x-request-id"] = request_id
        return resp
    finally:
        _record_telemetry(
            request,
            model=model,
            source_provider=source_provider,
            target_provider=target_provider,
            provider_name=provider_name,
            is_stream=is_stream,
            status_code=status_code,
            duration_ms=(time.monotonic() - t0) * 1000,
            error_detail=error_detail,
        )


# --- Endpoint handlers ---


async def handle_openai_chat(request: Any) -> Response | StreamingResponse:
    return await _proxy_handler(request, source_provider="openai_chat")


async def handle_embeddings(request: Any) -> Response:
    assert _config is not None
    return await _handle_embeddings(request, _config)


async def handle_anthropic(request: Any) -> Response | StreamingResponse:
    return await _proxy_handler(request, source_provider="anthropic")


async def handle_openai_responses(request: Any) -> Response | StreamingResponse:
    return await _proxy_handler(request, source_provider="openai_responses")


async def handle_google_genai(
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


async def handle_list_models(request: Any) -> Response:
    """List configured models in a format compatible with OpenAI and Anthropic SDKs."""
    assert _config is not None
    models = sorted(_config.models.keys())
    data = []
    for name in models:
        provider_name = _config.models[name]
        api_standard = _config.provider_types.get(provider_name, "unknown")
        capabilities = _config.model_capabilities.get(name, ["text"])
        data.append(
            {
                "id": name,
                "object": "model",
                "created": 0,
                "owned_by": provider_name,
                "api_standard": api_standard,
                "capabilities": capabilities,
                "type": "model",
                "display_name": name,
                "created_at": "1970-01-01T00:00:00Z",
            }
        )
    return JSONResponse(
        {
            "object": "list",
            "data": data,
            "has_more": False,
            "first_id": models[0] if models else None,
            "last_id": models[-1] if models else None,
        }
    )


async def handle_list_models_google(request: Any) -> Response:
    """List configured models in Google GenAI SDK format."""
    assert _config is not None
    models_list = [
        {
            "name": f"models/{name}",
            "displayName": name,
            "supportedGenerationMethods": [
                "generateContent",
                "streamGenerateContent",
            ],
        }
        for name in sorted(_config.models.keys())
    ]
    return JSONResponse({"models": models_list})


async def handle_health(request: Any) -> Response:
    """Return operational metrics and per-provider health status.

    Always returns HTTP 200. Use ``status: "degraded"`` in the payload
    to signal provider issues without breaking existing monitors.
    For a 503-on-unhealthy probe use ``/health/ready``.
    """
    metrics = getattr(request.app, "metrics", None)
    if metrics is None:
        return JSONResponse({"status": "ok"})

    snap = metrics.snapshot(series_seconds=3600)  # 1-hour window for errors_last_hour
    errors_last_hour = sum(
        pt["errors"] for pt in snap.get("series", []) if pt.get("errors", 0)
    )

    provider_health = metrics.provider_health_snapshot()
    critical = metrics.any_critical_provider()
    overall_status = "degraded" if critical else "ok"

    payload = {
        "status": overall_status,
        "uptime_seconds": snap["uptime_seconds"],
        "requests_total": snap["total_requests"],
        "errors_last_hour": errors_last_hour,
        "providers": provider_health,
    }
    return JSONResponse(payload, status_code=200)


async def handle_health_live(request: Any) -> Response:
    """Kubernetes liveness probe — always 200 while the process is up."""
    return JSONResponse({"status": "ok"})


async def handle_health_ready(request: Any) -> Response:
    """Kubernetes readiness probe — 200 if all providers are operational, 503 if not."""
    metrics = getattr(request.app, "metrics", None)
    if metrics is None:
        return JSONResponse({"status": "ok"})

    critical = metrics.any_critical_provider()
    if critical:
        provider_health = metrics.provider_health_snapshot()
        return JSONResponse(
            {"status": "not_ready", "providers": provider_health},
            status_code=503,
        )
    return JSONResponse({"status": "ready"})


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

    persistence.close()
    logger.info("Persistence flushed and closed on shutdown")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(config: GatewayConfig, config_path: str | None = None) -> App:
    """Create the httpserver application."""
    global _config
    _config = config

    # Expose global proxy as env vars so downstream code (e.g. image
    # downloads in converters) can use it without threading config through.
    import os

    if config.proxy:
        os.environ.setdefault("HTTP_PROXY", config.proxy)
        os.environ.setdefault("HTTPS_PROXY", config.proxy)

    metadata_store = ProviderMetadataStore()

    app = App(max_body_size=50_000_000, read_timeout=300.0)

    # --- Routes ---
    app.route("/v1/chat/completions", methods=["POST"])(handle_openai_chat)
    app.route("/v1/embeddings", methods=["POST"])(handle_embeddings)
    app.route("/v1/messages", methods=["POST"])(handle_anthropic)
    app.route("/v1/responses", methods=["POST"])(handle_openai_responses)
    app.route("/v1/models", methods=["GET"])(handle_list_models)
    app.route("/v1beta/models", methods=["GET"])(handle_list_models_google)
    app.route("/v1beta/models/<path:model_path>", methods=["POST"])(handle_google_genai)
    app.route("/health", methods=["GET"])(handle_health)
    app.route("/health/live", methods=["GET"])(handle_health_live)
    app.route("/health/ready", methods=["GET"])(handle_health_ready)

    # --- Auth ---
    import secrets

    internal_token = f"rsk-internal-{secrets.token_hex(16)}"
    auth_state = AuthState(
        config.api_key_set,
        config.api_key_labels,
        internal_token,
        admin_password=config.admin_password,
    )
    app.before_request(create_auth_hook(auth_state))

    # --- CORS ---
    # Admin API endpoints are restricted to same-origin by default.
    # /v1/* proxy endpoints remain open (Access-Control-Allow-Origin: *).
    # The list of allowed origins for admin can be overridden via
    # server.admin_cors_origins in config (default [] = same-origin only).
    _admin_cors_origins: list[str] = config.admin_cors_origins

    def _is_admin_path(path: str) -> bool:
        return path.startswith("/admin/") or path == "/admin"

    def _apply_cors(response: Any, origin: str | None) -> None:
        """Set CORS headers on *response* for admin requests.

        When *_admin_cors_origins* is non-empty the request Origin is reflected
        only if it matches the allow-list; otherwise no CORS header is emitted
        so browsers fall back to same-origin behaviour.
        """
        if _admin_cors_origins and origin and origin in _admin_cors_origins:
            response.headers["Access-Control-Allow-Origin"] = origin
            response.headers["Vary"] = "Origin"
            response.headers["Access-Control-Allow-Methods"] = "*"
            response.headers["Access-Control-Allow-Headers"] = "*"
        # Default: no header -> same-origin only (browser blocks cross-origin).

    @app.after_request
    async def add_cors_headers(request: Any, response: Any) -> Any:
        if _is_admin_path(request.path):
            # Restricted CORS for admin endpoints: same-origin only by default,
            # or explicit allow-list via server.admin_cors_origins.
            _apply_cors(response, request.headers.get("origin"))
            # Prevent reverse-proxy caching of admin API responses (e.g. Caddy/Souin).
            # Uses the full directive set that Souin recognises as NO-STORE-DIRECTIVE.
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

    @app.errorhandler(404)
    async def handle_404(request: Any, exc: Any) -> Response:
        resp = JSONResponse({"error": "Not Found"}, status_code=404)
        if not _is_admin_path(request.path):
            resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    @app.errorhandler(405)
    async def handle_405(request: Any, exc: Any) -> Response:
        resp = JSONResponse({"error": "Method Not Allowed"}, status_code=405)
        if not _is_admin_path(request.path):
            resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    # --- Admin routes ---
    from .admin import setup_admin
    from .admin.routes import register_admin_routes

    register_admin_routes(app)

    # --- App-level state ---
    app.metadata_store = metadata_store  # type: ignore
    app.internal_token = internal_token  # type: ignore
    app.auth_state = auth_state  # type: ignore

    setup_admin(app, config, config_path)

    return app


async def run_gateway(app: App, host: str, port: int) -> None:
    """Start the gateway with lifecycle management."""
    # Expose bind address so admin test tasks can self-call.
    setattr(app, "_bind_host", host)
    setattr(app, "_bind_port", port)
    flush_task = asyncio.create_task(_periodic_flush(app))
    try:
        await app._serve(host, port)
    finally:
        flush_task.cancel()
        try:
            await flush_task
        except asyncio.CancelledError:
            pass
        _flush_now(app)
        await close_resources(metadata_store=app.metadata_store)  # type: ignore
