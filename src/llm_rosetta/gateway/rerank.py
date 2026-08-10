"""Rerank proxy handler with cross-format conversion.

Proxies ``/v1/rerank`` requests to upstream rerank providers,
converting between Jina, Cohere, and Voyage formats via IR.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from llm_rosetta._vendor.httpserver import JSONResponse, Response

from .config import GatewayConfig
from .headers import build_upstream_extra_headers, get_request_id
from .logging import get_logger
from .rerank_pipeline import RerankConversionPipeline

logger = get_logger()


async def handle_rerank(
    request: Any,
    config: GatewayConfig,
) -> Response:
    """Proxy a rerank request with cross-format conversion.

    Incoming requests can be in any supported rerank format (Jina,
    Cohere, Voyage).  The pipeline converts to the upstream provider's
    native format, forwards the request, and converts the response back.
    """
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
        provider_name, target_format, base_url, rerank_path, auth_headers, _ = (
            config.resolve_rerank(model)
        )
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

    source_format = config.default_rerank_format

    # --- Convert request ---
    pipeline = RerankConversionPipeline(source_format, target_format)
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
    upstream_url = f"{base_url}{rerank_path}"
    extra_headers = build_upstream_extra_headers(request, request_id)
    headers = {
        "Content-Type": "application/json",
        **auth_headers,
        **extra_headers,
    }

    t0 = time.monotonic()
    status_code = 500

    try:
        async with httpx.AsyncClient(timeout=config.upstream_timeout) as client:
            resp = await client.post(
                upstream_url,
                json=target_body,
                headers=headers,
            )
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
            return with_request_id(
                Response(
                    body=resp.content,
                    status_code=200,
                    content_type="application/json",
                )
            )

        try:
            source_body = pipeline.convert_response(upstream_body)
        except Exception as exc:
            logger.warning("rerank: response conversion failed: %s", exc)
            return with_request_id(
                Response(
                    body=resp.content,
                    status_code=200,
                    content_type="application/json",
                )
            )

        return with_request_id(JSONResponse(source_body, status_code=200))

    except httpx.TimeoutException as exc:
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
    except httpx.ConnectError as exc:
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
            provider_name,
            source_format,
            target_format,
            duration_ms,
            status_code,
        )
