"""Embeddings proxy handler with optional cross-format conversion.

Proxies ``/v1/embeddings`` requests to upstream embedding providers.
When the model's provider declares an embedding endpoint (``embedding_format``
/ ``embedding_shim``, or the legacy ``embedding_providers`` section), uses
IR-based conversion between OpenAI, Cohere, Jina, Voyage and Google GenAI
formats.  There is no chat-provider passthrough: an unconfigured model gets a
404 rather than a chat-shaped request forwarded to a provider that never
declared an embedding endpoint.

Every request that reaches an upstream is recorded in the usage ledger with
``model_type="embedding"``, through the same
:class:`~llm_rosetta.gateway.ops.proxy.OpsProxyRequest` the chat routes use.
Requests rejected before the upstream call — bad JSON, an unknown model, a
failed request conversion — are not recorded: nothing was billed.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response

from ..config import GatewayConfig
from .embedding_pipeline import EmbeddingConversionPipeline, response_usage
from ..middleware.headers import build_upstream_extra_headers, get_request_id
from ..logging import get_logger
from ..ops.proxy import record_proxy_request
from ..transport import (
    ProviderInfo,
    UpstreamConnectionError,
    UpstreamTimeoutError,
    UpstreamTransport,
)

logger = get_logger()


def _detect_embedding_source(
    request: Any, body: dict[str, Any], config: GatewayConfig
) -> str:
    """Infer the source embedding format through the type descriptor.

    The detector lives in :mod:`gateway.source_formats` and is attached to the
    ``embedding`` descriptor, so it is the same object the descriptor exposes
    and stays testable without a handler.  The fallback here only covers a
    registry that has no detector (a test double or a future type).
    """
    from ..model_types import get_model_type

    descriptor = get_model_type("embedding")
    if descriptor is not None and descriptor.detect_source_format is not None:
        return descriptor.detect_source_format(request, body, config)
    return config.default_embedding_format


@dataclass
class _ResolvedEmbedding:
    provider_info: ProviderInfo
    upstream_url: str
    provider_name: str
    target_format: str = "openai"
    source_format: str = "openai"
    #: The configured endpoint path.  It may carry ``{model}`` (Google puts
    #: the model in the URL) and is used to build the batch variant after the
    #: request has been converted.
    path: str = ""
    pipeline: EmbeddingConversionPipeline | None = field(default=None)


def _resolve_embedding_provider(
    config: GatewayConfig,
    model: str,
    body: dict[str, Any],
    request: Any = None,
) -> _ResolvedEmbedding | None:
    """Resolve the embedding provider, or ``None`` when the model is unknown.

    Embedding traffic is routed only through the embedding pool.  There is no
    chat-provider fallback: an unconfigured model returns 404 instead of
    forwarding a chat-shaped request to a provider that never declared an
    embedding endpoint.
    """
    try:
        route = config.resolve_embedding(model)
    except KeyError:
        return None

    source_format = _detect_embedding_source(request, body, config)
    logger.debug("embedding: detected source format: %s", source_format)
    pipeline = (
        EmbeddingConversionPipeline(source_format, route.format)
        if source_format != route.format
        else None
    )
    upstream_model = config.model_upstream_names.get(model)
    if upstream_model:
        body["model"] = upstream_model
    return _ResolvedEmbedding(
        provider_info=route.provider_info,
        upstream_url=route.provider_info.upstream_url(upstream_model or model),
        provider_name=route.provider_name,
        target_format=route.format,
        source_format=source_format,
        path=route.embedding_path,
        pipeline=pipeline,
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
    resolved = _resolve_embedding_provider(config, model, body, request)
    if resolved is None:
        all_models = sorted(config.embedding_models.keys())
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

    # The model to place in the upstream URL; conversion may consume it.
    upstream_model = body.get("model") or model

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

    # Google carries the model in the path and has a distinct batch suffix.
    # Both are decided after conversion: the converted body is what says
    # single (``content``) or batch (``requests``).
    upstream_url = resolved.upstream_url
    if resolved.target_format == "google_generate":
        path = resolved.path.replace("{model}", upstream_model)
        if "requests" in body and path.endswith(":embedContent"):
            path = path[: -len(":embedContent")] + ":batchEmbedContents"
        upstream_url = resolved.provider_info.base_url + path

    # --- Forward via transport ---
    transport: UpstreamTransport = request.app.transport
    extra_headers = build_upstream_extra_headers(request, request_id)

    t0 = time.monotonic()
    status_code = 500
    error_detail: str | None = None
    usage: dict[str, Any] = {}

    try:
        resp = await transport.send(
            resolved.provider_info,
            upstream_url,
            body,
            extra_headers=extra_headers,
        )
        status_code = resp.status_code
        # Read even on an error status: a 4xx can still carry the tokens the
        # upstream counted before it refused.
        usage = response_usage(resolved.target_format, resp.body)

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
        status_code = 504
        error_detail = f"Upstream timeout: {exc}"
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
        status_code = 502
        error_detail = f"Upstream request failed: {exc}"
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
    finally:
        await record_proxy_request(
            request.app,
            model=model,
            source_provider=resolved.source_format,
            target_provider=resolved.target_format,
            provider_name=resolved.provider_name,
            is_stream=False,
            status_code=status_code,
            duration_ms=(time.monotonic() - t0) * 1000,
            error_detail=error_detail,
            model_type="embedding",
            usage=usage,
        )
