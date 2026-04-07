"""llm-rosetta Gateway — ASGI application and route handlers."""

from __future__ import annotations

from contextlib import asynccontextmanager
from collections.abc import AsyncGenerator
from typing import Any

from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Route

from llm_rosetta.auto_detect import ProviderType

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
    request: Request,
    source_provider: ProviderType,
    model_override: str | None = None,
    force_stream: bool = False,
) -> Response:
    """Shared handler for all proxy endpoints."""
    assert _config is not None

    try:
        body: dict[str, Any] = await request.json()
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
        target_provider, provider_info = _config.resolve_model(model)
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

    store: ProviderMetadataStore = request.app.state.metadata_store

    if is_stream:
        return await handle_streaming(
            source_provider,
            target_provider,
            provider_info,
            body,
            model,
            metadata_store=store,
        )
    else:
        return await handle_non_streaming(
            source_provider,
            target_provider,
            provider_info,
            body,
            model,
            metadata_store=store,
        )


# --- Endpoint handlers ---


async def handle_openai_chat(request: Request) -> Response:
    return await _proxy_handler(request, source_provider="openai_chat")


async def handle_anthropic(request: Request) -> Response:
    return await _proxy_handler(request, source_provider="anthropic")


async def handle_openai_responses(request: Request) -> Response:
    return await _proxy_handler(request, source_provider="openai_responses")


async def handle_google_genai(request: Request) -> Response:
    model_path = request.path_params["model_path"]
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
            status_code=404,
            content='{"error": "Unknown Google GenAI method"}',
            media_type="application/json",
        )


async def handle_list_models(request: Request) -> Response:
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


async def handle_list_models_google(request: Request) -> Response:
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


async def handle_health(request: Request) -> Response:
    assert _config is not None
    return JSONResponse(
        {
            "status": "ok",
            "providers": list(_config.providers.keys()),
            "models": list(_config.models.keys()),
        }
    )


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app(config: GatewayConfig) -> Starlette:
    """Create the Starlette ASGI application."""
    global _config
    _config = config

    metadata_store = ProviderMetadataStore()

    routes = [
        Route("/v1/chat/completions", handle_openai_chat, methods=["POST"]),
        Route("/v1/messages", handle_anthropic, methods=["POST"]),
        Route("/v1/responses", handle_openai_responses, methods=["POST"]),
        Route("/v1/models", handle_list_models, methods=["GET"]),
        Route("/v1beta/models", handle_list_models_google, methods=["GET"]),
        Route(
            "/v1beta/models/{model_path:path}",
            handle_google_genai,
            methods=["POST"],
        ),
        Route("/health", handle_health, methods=["GET"]),
    ]

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncGenerator[None, None]:
        yield
        await close_resources(metadata_store=metadata_store)

    app = Starlette(routes=routes, lifespan=lifespan)
    app.state.metadata_store = metadata_store
    return app
