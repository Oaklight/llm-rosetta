"""Embeddings proxy handler with optional cross-format conversion.

Proxies ``/v1/embeddings`` requests to upstream embedding providers.
When ``embedding_providers`` / ``embedding_models`` are configured, uses
IR-based conversion between OpenAI, Cohere, Jina, and Voyage formats.
Otherwise falls back to passthrough via the chat provider routing
(backward compatible with existing configs).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, cast

from llm_rosetta._vendor.httpserver import JSONResponse, Response

from llm_rosetta.auto_detect import ProviderType

from .config import GatewayConfig
from .embedding_pipeline import EmbeddingConversionPipeline
from .headers import build_upstream_extra_headers, get_request_id
from .logging import get_logger
from .transport import (
    ProviderInfo,
    UpstreamConnectionError,
    UpstreamTimeoutError,
    UpstreamTransport,
)

logger = get_logger()


def _detect_embedding_source(body: dict[str, Any], config: GatewayConfig) -> str:
    """Infer the source embedding format from the request body.

    Detection order:
    1. ``texts`` field (without ``input``) → Cohere
    2. ``task`` field → Jina
    3. ``output_dtype`` field → Voyage
    4. Fall back to ``config.default_embedding_format``
    """
    if "texts" in body and "input" not in body:
        return "cohere"
    if "task" in body:
        return "jina"
    if "output_dtype" in body:
        return "voyage"
    return config.default_embedding_format


@dataclass
class _ResolvedEmbedding:
    provider_info: ProviderInfo
    upstream_url: str
    provider_name: str
    target_format: str = "openai_chat"
    source_format: str = "openai_chat"
    pipeline: EmbeddingConversionPipeline | None = field(default=None)


def _resolve_embedding_provider(
    config: GatewayConfig, model: str, body: dict[str, Any]
) -> _ResolvedEmbedding | None:
    """Resolve embedding provider; returns None if model not found anywhere."""
    # Try embedding-specific routing first
    try:
        route = config.resolve_embedding(model)
        source_format = _detect_embedding_source(body, config)
        logger.debug("embedding: detected source format: %s", source_format)
        pipeline = (
            EmbeddingConversionPipeline(source_format, route.format)
            if source_format != route.format
            else None
        )
        return _ResolvedEmbedding(
            provider_info=route.provider_info,
            upstream_url=f"{route.provider_info.base_url}{route.embedding_path}",
            provider_name=route.provider_name,
            target_format=route.format,
            source_format=source_format,
            pipeline=pipeline,
        )
    except KeyError:
        pass

    # Fall back to chat provider routing (backward compat)
    try:
        chat_route, provider_info = config.resolve("openai_chat", model)
    except KeyError:
        return None

    if chat_route.upstream_model:
        body["model"] = chat_route.upstream_model

    return _ResolvedEmbedding(
        provider_info=provider_info,
        upstream_url=f"{provider_info.base_url}/embeddings",
        provider_name=chat_route.provider_name,
    )


async def handle_embeddings(
    request: Any,
    config: GatewayConfig,
) -> Response:
    """Proxy an embedding request with optional cross-format conversion."""
    request_id = get_request_id(request)

    def with_request_id(response: Response) -> Response:
        response.headers["x-request-id"] = request_id
        return response

    # --- Parse request ---
    try:
        body: dict[str, Any] = request.json()
    except Exception:
        return with_request_id(
            JSONResponse(
                {
                    "error": {
                        "message": "Invalid JSON body",
                        "type": "invalid_request_error",
                    }
                },
                status_code=400,
            )
        )

    model = body.get("model")
    if not model:
        return with_request_id(
            JSONResponse(
                {
                    "error": {
                        "message": "Missing 'model' in request body",
                        "type": "invalid_request_error",
                    }
                },
                status_code=400,
            )
        )

    # --- Resolve provider ---
    resolved = _resolve_embedding_provider(config, model, body)
    if resolved is None:
        all_models = sorted(
            set(config.models.keys()) | set(config.embedding_models.keys())
        )
        return with_request_id(
            JSONResponse(
                {
                    "error": {
                        "message": (
                            f"Unknown model: '{model}'. "
                            f"Configured: {', '.join(all_models)}"
                        ),
                        "type": "model_not_found",
                    }
                },
                status_code=404,
            )
        )

    # --- Convert request (if cross-format) ---
    if resolved.pipeline:
        try:
            body = resolved.pipeline.convert_request(body)
        except Exception as exc:
            logger.warning("embedding: request conversion failed: %s", exc)
            return with_request_id(
                JSONResponse(
                    {
                        "error": {
                            "message": f"Request conversion failed: {exc}",
                            "type": "conversion_error",
                        }
                    },
                    status_code=400,
                )
            )

    # --- Forward via transport ---
    transport: UpstreamTransport = request.app.transport
    extra_headers = build_upstream_extra_headers(request, request_id)

    t0 = time.monotonic()
    status_code = 500
    error_detail: str | None = None

    try:
        resp = await transport.send(
            resolved.provider_info,
            resolved.upstream_url,
            body,
            extra_headers=extra_headers,
        )
        status_code = resp.status_code

        if resp.is_error:
            error_detail = resp.error_text
            return with_request_id(
                Response(
                    body=resp.raw_content,
                    status_code=resp.status_code,
                    content_type="application/json",
                )
            )

        # --- Convert response (if cross-format) ---
        if resolved.pipeline and resp.body is not None:
            try:
                source_body = resolved.pipeline.convert_response(resp.body)
                if not source_body.get("model"):
                    source_body["model"] = model
                return with_request_id(JSONResponse(source_body, status_code=200))
            except Exception as exc:
                logger.warning("embedding: response conversion failed: %s", exc)
                fallback = with_request_id(
                    Response(
                        body=resp.raw_content,
                        status_code=200,
                        content_type="application/json",
                    )
                )
                fallback.headers["x-rosetta-conversion"] = "passthrough"
                return fallback

        return with_request_id(
            Response(
                body=resp.raw_content,
                status_code=200,
                content_type="application/json",
            )
        )
    except UpstreamTimeoutError as exc:
        error_detail = str(exc)
        status_code = 504
        return with_request_id(
            JSONResponse(
                {
                    "error": {
                        "message": f"Upstream timeout: {exc}",
                        "type": "upstream_error",
                    }
                },
                status_code=504,
            )
        )
    except UpstreamConnectionError as exc:
        error_detail = str(exc)
        status_code = 502
        return with_request_id(
            JSONResponse(
                {
                    "error": {
                        "message": f"Upstream request failed: {exc}",
                        "type": "upstream_error",
                    }
                },
                status_code=502,
            )
        )
    except Exception as exc:
        error_detail = str(exc)
        raise
    finally:
        from .app import _record_telemetry

        _record_telemetry(
            request,
            model=model,
            source_provider=cast(ProviderType, resolved.source_format),
            target_provider=cast(ProviderType, resolved.target_format),
            provider_name=resolved.provider_name,
            is_stream=False,
            status_code=status_code,
            duration_ms=(time.monotonic() - t0) * 1000,
            error_detail=error_detail,
        )
