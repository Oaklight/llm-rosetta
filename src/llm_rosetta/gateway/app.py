"""llm-rosetta Gateway — HTTP application and route handlers."""

from __future__ import annotations

import asyncio
import time
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

    try:
        body: dict[str, Any] = request.json()
    except Exception:
        return error_response_for_source(source_provider, 400, "Invalid JSON body")

    # Determine model
    model = model_override or extract_model(source_provider, body)
    if not model:
        return error_response_for_source(
            source_provider, 400, "Missing 'model' in request body"
        )

    # If model came from URL (Google), inject it into body for the converter
    if model_override and "model" not in body:
        body["model"] = model_override

    # Resolve target provider
    try:
        target_provider_str, provider_info = _config.resolve_model(model)
        target_provider = cast(ProviderType, target_provider_str)
    except KeyError:
        configured = ", ".join(sorted(_config.models.keys()))
        return error_response_for_source(
            source_provider,
            404,
            f"Unknown model: '{model}'. Configured models: {configured}",
        )

    # Determine streaming
    is_stream = force_stream or detect_stream_request(source_provider, body)

    logger.info(
        "%s -> %s | model=%s stream=%s",
        source_provider,
        target_provider,
        model,
        is_stream,
    )

    store: ProviderMetadataStore = request.app.metadata_store

    # Forward OpenResponses-Version header to upstream if present
    extra_headers: dict[str, str] | None = None
    or_version = request.headers.get("openresponses-version")
    if or_version:
        extra_headers = {"OpenResponses-Version": or_version}

    # --- Metrics instrumentation ---
    metrics = getattr(request.app, "metrics", None)
    request_log = getattr(request.app, "request_log", None)
    t0 = time.monotonic()
    status_code = 500
    error_detail: str | None = None

    try:
        if is_stream:
            if metrics:
                metrics.active_streams += 1
            response = await handle_streaming(
                source_provider,
                target_provider,
                provider_info,
                body,
                model,
                metadata_store=store,
                extra_headers=extra_headers,
            )
        else:
            response = await handle_non_streaming(
                source_provider,
                target_provider,
                provider_info,
                body,
                model,
                metadata_store=store,
                extra_headers=extra_headers,
            )
        status_code = response.status_code
        if status_code >= 400 and hasattr(response, "body"):
            body_bytes = response.body
            if isinstance(body_bytes, bytes):
                error_detail = body_bytes.decode("utf-8", errors="replace")
        return response
    except Exception as exc:
        error_detail = str(exc)
        raise
    finally:
        duration_ms = (time.monotonic() - t0) * 1000
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
            )
        if request_log is not None:
            from .admin.request_log import RequestLogEntry

            api_key_label = api_key_label_var.get()
            request_log.add(
                RequestLogEntry.create(
                    model=model,
                    source_provider=source_provider,
                    target_provider=target_provider,
                    is_stream=is_stream,
                    status_code=status_code,
                    duration_ms=duration_ms,
                    error_detail=error_detail,
                    api_key_label=api_key_label,
                )
            )


# --- Endpoint handlers ---


async def handle_openai_chat(request: Any) -> Response | StreamingResponse:
    return await _proxy_handler(request, source_provider="openai_chat")


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
    data = [
        {
            "id": name,
            "object": "model",
            "created": 0,
            "owned_by": _config.models[name],
            "type": "model",
            "display_name": name,
            "created_at": "1970-01-01T00:00:00Z",
        }
        for name in models
    ]
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
    assert _config is not None
    return JSONResponse(
        {
            "status": "ok",
            "providers": list(_config.providers.keys()),
            "models": list(_config.models.keys()),
        }
    )


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
    app.route("/v1/messages", methods=["POST"])(handle_anthropic)
    app.route("/v1/responses", methods=["POST"])(handle_openai_responses)
    app.route("/v1/models", methods=["GET"])(handle_list_models)
    app.route("/v1beta/models", methods=["GET"])(handle_list_models_google)
    app.route("/v1beta/models/<path:model_path>", methods=["POST"])(handle_google_genai)
    app.route("/health", methods=["GET"])(handle_health)

    # --- Auth ---
    import secrets

    internal_token = f"rsk-internal-{secrets.token_hex(16)}"
    auth_state = AuthState(config.api_key_set, config.api_key_labels, internal_token)
    app.before_request(create_auth_hook(auth_state))

    # --- CORS ---
    @app.after_request
    async def add_cors_headers(request: Any, response: Any) -> Any:
        response.headers["Access-Control-Allow-Origin"] = "*"
        response.headers["Access-Control-Allow-Methods"] = "*"
        response.headers["Access-Control-Allow-Headers"] = "*"
        return response

    @app.route("/<path:_path>", methods=["OPTIONS"])
    async def cors_preflight(request: Any, _path: str = "") -> Response:
        return Response(body=b"", status_code=204)

    @app.errorhandler(404)
    async def handle_404(request: Any, exc: Any) -> Response:
        resp = JSONResponse({"error": "Not Found"}, status_code=404)
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    @app.errorhandler(405)
    async def handle_405(request: Any, exc: Any) -> Response:
        resp = JSONResponse({"error": "Method Not Allowed"}, status_code=405)
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
