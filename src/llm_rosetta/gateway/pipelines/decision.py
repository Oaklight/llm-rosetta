"""Decision proxy handler with optional cross-format conversion.

Proxies ``/v1/decision`` and ``/v1/systemone`` requests to upstream
decision providers (currently TypeSafe System One).  No streaming —
decision requests return a single JSON response.

When the source format (inferred from ``default_decision_format``)
differs from the target format (declared on the provider config), uses
IR-based conversion via
:class:`~llm_rosetta.converters.decision.pipeline.DecisionConversionPipeline`.

Registered via the model type registry in ``model_types.py``.

Every request that reaches an upstream is recorded in the usage ledger with
``model_type="decision"``, through the same
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
from .decision_pipeline import DecisionConversionPipeline, response_usage
from ..transport import UpstreamConnectionError, UpstreamTimeoutError, UpstreamTransport

logger = get_logger()


def _detect_decision_source_format(
    request: Any, body: dict[str, Any], config: GatewayConfig
) -> str:
    """Infer the source decision format from the request path and body.

    Detection order:
    1. ``/v1/decisions`` path → OpenAI Decisions
    2. ``/v1/systemone`` path → System One
    3. OpenAI-shaped body (``input`` + list ``questions``) → OpenAI Decisions
    4. System One-shaped body (``state`` + map ``questions``) → System One
    5. Fall back to ``config.default_decision_format``

    The canonical ``/v1/decision`` route predates both conventions, so it falls
    through to body detection (then the configured default).
    """
    path: str = (getattr(request, "path", "") or "").rstrip("/")
    if path == "/v1/decisions":
        return "openai_decisions"
    if path == "/v1/systemone":
        return "systemone"
    questions = body.get("questions")
    if "input" in body and isinstance(questions, list):
        return "openai_decisions"
    if "state" in body and isinstance(questions, dict):
        return "systemone"
    return config.default_decision_format


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
    source_format = _detect_decision_source_format(request, body, config)
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
    error_detail: str | None = None
    usage: dict[str, Any] = {}

    try:
        resp = await transport.send(
            route.provider_info,
            upstream_url,
            body,
            extra_headers=extra_headers,
        )
        status_code = resp.status_code
        # Read even on an error status: a 4xx can still carry the tokens the
        # upstream counted before it refused.
        usage = response_usage(target_format, resp.body)

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
        await record_proxy_request(
            request.app,
            model=model,
            source_provider=source_format,
            target_provider=target_format,
            provider_name=route.provider_name,
            is_stream=False,
            status_code=status_code,
            duration_ms=duration_ms,
            error_detail=error_detail,
            model_type="decision",
            usage=usage,
        )
