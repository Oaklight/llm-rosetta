"""Rerank proxy handler with cross-format conversion.

Proxies ``/v1/rerank`` and ``/v2/rerank`` requests to upstream rerank
providers, converting between Jina, Cohere, and Voyage formats via IR.
Uses the shared :class:`~transport.UpstreamTransport` interface,
consistent with the embeddings and chat handlers.

Every request that reaches an upstream is recorded in the usage ledger with
``model_type="rerank"``, through the same
:class:`~llm_rosetta.gateway.ops.proxy.OpsProxyRequest` the chat routes use.
Requests rejected before the upstream call — bad JSON, an unknown model, a
failed request conversion — are not recorded: nothing was billed.
"""

from __future__ import annotations

import time
from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response

from ..config import GatewayConfig
from ..middleware.headers import build_upstream_extra_headers, get_request_id
from ..logging import get_logger
from ..ops.proxy import record_proxy_request
from .rerank_pipeline import RerankConversionPipeline, response_usage
from ..transport import UpstreamConnectionError, UpstreamTimeoutError, UpstreamTransport

logger = get_logger()


def _detect_source_format(
    request: Any, body: dict[str, Any], config: GatewayConfig
) -> str:
    """Infer the source rerank format from the request path and body.

    Detection order:
    1. ``/v2/rerank`` path → Cohere (only Cohere uses v2)
    2. ``top_k`` in body (without ``top_n``) → Voyage
    3. Fall back to ``config.default_rerank_format``
    """
    path: str = getattr(request, "path", "")
    if path.startswith("/v2/"):
        return "cohere"
    if "top_k" in body and "top_n" not in body:
        return "voyage"
    return config.default_rerank_format


async def handle_rerank(
    request: Any,
    config: GatewayConfig,
) -> Response:
    """Proxy a rerank request with cross-format conversion."""
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

    # --- Resolve rerank provider ---
    try:
        route = config.resolve_rerank(model)
    except KeyError:
        configured = ", ".join(sorted(config.rerank_models.keys()))
        return with_request_id(
            JSONResponse(
                {
                    "error": {
                        "message": (
                            f"Unknown rerank model: '{model}'. "
                            f"Configured: {configured or '(none)'}"
                        ),
                        "type": "model_not_found",
                    }
                },
                status_code=404,
            )
        )

    source_format = _detect_source_format(request, body, config)

    logger.debug("rerank: detected source format: %s", source_format)
    # --- Convert request ---
    pipeline = RerankConversionPipeline(source_format, route.format)
    try:
        target_body = pipeline.convert_request(body)
    except Exception as exc:
        logger.warning("rerank: request conversion failed: %s", exc)
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
    upstream_url = route.provider_info.upstream_url("")
    transport: UpstreamTransport = request.app.transport
    extra_headers = build_upstream_extra_headers(request, request_id)

    t0 = time.monotonic()
    status_code = 500
    error_detail: str | None = None
    usage: dict[str, Any] = {}

    try:
        resp = await transport.send(
            route.provider_info,
            upstream_url,
            target_body,
            extra_headers=extra_headers,
        )
        status_code = resp.status_code
        # Read even on an error status: a 4xx can still carry the tokens the
        # upstream counted before it refused.
        usage = response_usage(route.format, resp.body)

        if resp.is_error:
            error_detail = resp.error_text
            return with_request_id(
                Response(
                    body=resp.raw_content,
                    status_code=resp.status_code,
                    content_type="application/json",
                )
            )

        # --- Convert response ---
        try:
            upstream_body = resp.body
            if upstream_body is None:
                raise ValueError("Empty response body")
        except Exception:
            fallback = with_request_id(
                Response(
                    body=resp.raw_content,
                    status_code=200,
                    content_type="application/json",
                )
            )
            fallback.headers["x-rosetta-conversion"] = "passthrough"
            return fallback

        try:
            source_body = pipeline.convert_response(upstream_body)
        except Exception as exc:
            logger.warning("rerank: response conversion failed: %s", exc)
            fallback = with_request_id(
                Response(
                    body=resp.raw_content,
                    status_code=200,
                    content_type="application/json",
                )
            )
            fallback.headers["x-rosetta-conversion"] = "passthrough"
            return fallback

        return with_request_id(JSONResponse(source_body, status_code=200))

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
        error_detail = f"Upstream connection failed: {exc}"
        return with_request_id(
            JSONResponse(
                {
                    "error": {
                        "message": f"Upstream connection failed: {exc}",
                        "type": "upstream_error",
                    }
                },
                status_code=502,
            )
        )
    except Exception:
        raise
    finally:
        duration_ms = (time.monotonic() - t0) * 1000
        if pipeline.warnings:
            logger.info(
                "rerank: conversion warnings for %s: %s",
                model,
                "; ".join(pipeline.warnings),
            )
        logger.info(
            "rerank: %s → %s (%s→%s) %dms status=%d",
            model,
            route.provider_name,
            source_format,
            route.format,
            duration_ms,
            status_code,
        )
        await record_proxy_request(
            request.app,
            model=model,
            source_provider=source_format,
            target_provider=route.format,
            provider_name=route.provider_name,
            is_stream=False,
            status_code=status_code,
            duration_ms=duration_ms,
            error_detail=error_detail,
            model_type="rerank",
            usage=usage,
        )
