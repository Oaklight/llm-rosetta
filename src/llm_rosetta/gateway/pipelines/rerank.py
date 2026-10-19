"""Rerank proxy handler with IR-based conversion.

Proxies ``/v1/rerank`` and ``/v2/rerank`` requests to upstream rerank
providers, converting every request and response through the IR — between
Jina, Cohere, and Voyage formats, and even when the source and target format
are the same.  Uses the shared :class:`~transport.UpstreamTransport`
interface, consistent with the embeddings and chat handlers.

Every request that reaches an upstream is recorded in the usage ledger with
``model_type="rerank"``, through the same
:class:`~llm_rosetta.gateway.ops.proxy.OpsProxyRequest` the chat routes use.
Requests rejected before the upstream call — bad JSON, an unknown model, a
failed request conversion — are not recorded: nothing was billed.
"""

from __future__ import annotations

import copy
import time
from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response

from ..config import GatewayConfig
from ..middleware.headers import build_upstream_extra_headers, get_request_id
from ..logging import get_logger
from ..ops.proxy import record_proxy_request
from ._fidelity import check_same_format_fidelity
from llm_rosetta.converters.rerank.pipeline import (
    RerankConversionPipeline,
    response_usage,
)
from ..transport import UpstreamConnectionError, UpstreamTimeoutError, UpstreamTransport

logger = get_logger()


def _detect_source_format(
    request: Any, body: dict[str, Any], config: GatewayConfig
) -> str:
    """Infer the source rerank format through the type descriptor.

    The detector lives in :mod:`gateway.source_formats` and is attached to the
    ``rerank`` descriptor; this wrapper keeps the handler-local name used by
    the tests and covers a registry without one.
    """
    from ..model_types import get_model_type

    descriptor = get_model_type("rerank")
    if descriptor is not None and descriptor.detect_source_format is not None:
        return descriptor.detect_source_format(request, body, config)
    return config.default_rerank_format


async def handle_rerank(
    request: Any,
    config: GatewayConfig,
) -> Response:
    """Proxy a rerank request with IR-based conversion."""
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
    same_format = source_format == route.format
    original_body = copy.deepcopy(body) if same_format else None
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
    check_same_format_fidelity(
        "rerank",
        source_format,
        route.format,
        original_body,
        target_body,
        "request",
        model=model,
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
            original_response = copy.deepcopy(upstream_body) if same_format else None
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
        check_same_format_fidelity(
            "rerank",
            source_format,
            route.format,
            original_response,
            source_body,
            "response",
            model=model,
        )

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
