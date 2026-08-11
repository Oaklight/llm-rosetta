"""Rerank proxy handler with cross-format conversion.

Proxies ``/v1/rerank`` and ``/v2/rerank`` requests to upstream rerank
providers, converting between Jina, Cohere, and Voyage formats via IR.
"""

from __future__ import annotations

import time
from typing import Any

from llm_rosetta._vendor.httpclient import (
    AsyncClient,
    HttpConnectionError,
    HttpTimeoutError,
)
from llm_rosetta._vendor.httpclient import Response as _HCResponse
from llm_rosetta._vendor.httpserver import JSONResponse, Response

from .config import GatewayConfig
from .headers import build_upstream_extra_headers, get_request_id
from .logging import get_logger
from .rerank_pipeline import RerankConversionPipeline

logger = get_logger()


def _detect_source_format(request: Any, config: GatewayConfig) -> str:
    """Infer the source rerank format from the request path.

    ``/v2/rerank`` implies Cohere format (only Cohere uses v2).
    ``/v1/rerank`` falls back to ``config.default_rerank_format``.
    """
    path: str = getattr(request, "path", "")
    if path.startswith("/v2/"):
        return "cohere"
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

    source_format = _detect_source_format(request, config)

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

    # --- Forward to upstream ---
    upstream_url = f"{route.base_url}{route.rerank_path}"
    extra_headers = build_upstream_extra_headers(request, request_id)
    headers = {
        "Content-Type": "application/json",
        **route.auth_headers,
        **extra_headers,
    }

    t0 = time.monotonic()
    status_code = 500

    try:
        async with AsyncClient(timeout=config.upstream_timeout) as client:
            _resp = await client.post(
                upstream_url,
                json=target_body,
                headers=headers,
            )
        assert isinstance(_resp, _HCResponse)  # rerank never streams
        resp = _resp
        status_code = resp.status_code

        if resp.status_code >= 400:
            return with_request_id(
                Response(
                    body=resp.content,
                    status_code=resp.status_code,
                    content_type="application/json",
                )
            )

        # --- Convert response ---
        try:
            upstream_body = resp.json()
        except Exception:
            fallback = with_request_id(
                Response(
                    body=resp.content,
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
                    body=resp.content,
                    status_code=200,
                    content_type="application/json",
                )
            )
            fallback.headers["x-rosetta-conversion"] = "passthrough"
            return fallback

        return with_request_id(JSONResponse(source_body, status_code=200))

    except HttpTimeoutError as exc:
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
    except HttpConnectionError as exc:
        status_code = 502
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
