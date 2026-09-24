"""Decision proxy handler with optional cross-format conversion.

Proxies ``/v1/decision`` and ``/v1/systemone`` requests to upstream
decision providers (currently TypeSafe System One).  No streaming —
decision requests return a single JSON response.

When the source format (inferred from ``default_decision_format``)
differs from the target format (declared on the provider config), uses
IR-based conversion via
:class:`~llm_rosetta.converters.decision.pipeline.DecisionConversionPipeline`.

Registered via the model type registry in ``model_types.py``.
"""

from __future__ import annotations

import time
from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response

from ..config import GatewayConfig
from ..middleware.headers import build_upstream_extra_headers, get_request_id
from ..logging import get_logger
from .decision_pipeline import DecisionConversionPipeline
from ..transport import UpstreamConnectionError, UpstreamTimeoutError, UpstreamTransport

logger = get_logger()


async def handle_decision(
    request: Any,
    config: GatewayConfig,
) -> Response:
    """Proxy a decision request with optional cross-format conversion."""
    request_id = get_request_id(request)

    def with_request_id(response: Response) -> Response:
        response.headers["x-request-id"] = request_id
        return response

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

    try:
        route = config.resolve_decision(model)
    except KeyError:
        configured = ", ".join(sorted(config.decision_models.keys()))
        return with_request_id(
            JSONResponse(
                {
                    "error": {
                        "message": (
                            f"Unknown decision model: '{model}'. "
                            f"Configured: {configured or '(none)'}"
                        ),
                        "type": "model_not_found",
                    }
                },
                status_code=404,
            )
        )

    # --- Build pipeline for format conversion ---
    source_format = config.default_decision_format
    target_format = route.format
    pipeline = (
        DecisionConversionPipeline(source_format, target_format)
        if source_format != target_format
        else None
    )

    # --- Convert request (if cross-format) ---
    if pipeline:
        try:
            body = pipeline.convert_request(body)
        except Exception as exc:
            logger.warning("decision: request conversion failed: %s", exc)
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

    upstream_url = route.provider_info.upstream_url("")
    transport: UpstreamTransport = request.app.transport
    extra_headers = build_upstream_extra_headers(request, request_id)

    t0 = time.monotonic()
    status_code = 500

    try:
        resp = await transport.send(
            route.provider_info,
            upstream_url,
            body,
            extra_headers=extra_headers,
        )
        status_code = resp.status_code

        if resp.is_error:
            return with_request_id(
                Response(
                    body=resp.raw_content,
                    status_code=resp.status_code,
                    content_type="application/json",
                )
            )

        # --- Convert response (if cross-format) ---
        if pipeline and resp.body is not None:
            try:
                source_body = pipeline.convert_response(resp.body)
                if not source_body.get("model"):
                    source_body["model"] = model
                return with_request_id(JSONResponse(source_body, status_code=200))
            except Exception as exc:
                logger.warning("decision: response conversion failed: %s", exc)
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
        if pipeline and pipeline.warnings:
            logger.info(
                "decision: conversion warnings for %s: %s",
                model,
                "; ".join(pipeline.warnings),
            )
        logger.info(
            "decision: %s → %s (%s→%s) %dms status=%d",
            model,
            route.provider_name,
            source_format,
            target_format,
            duration_ms,
            status_code,
        )
