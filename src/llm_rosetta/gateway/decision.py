"""Decision proxy handler.

Proxies ``/v1/decision`` and ``/v1/systemone`` requests to upstream decision
providers (currently TypeSafe System One).  No streaming — decision
requests return a single JSON response.
"""

from __future__ import annotations

import time
from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response

from .config import GatewayConfig
from .headers import build_upstream_extra_headers, get_request_id
from .logging import get_logger
from .transport import UpstreamConnectionError, UpstreamTimeoutError, UpstreamTransport

logger = get_logger()


async def handle_decision(
    request: Any,
    config: GatewayConfig,
) -> Response:
    """Proxy a decision request to an upstream decision provider."""
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

    # --- Resolve eval provider ---
    # resolve_decision / decision_models will be added to GatewayConfig when
    # eval provider routing is implemented.  Until then, guard with
    # getattr to keep ty happy.
    resolve_decision = getattr(config, "resolve_decision", None)
    if resolve_decision is None:
        return with_request_id(
            JSONResponse(
                {
                    "error": {
                        "message": "Decision provider routing is not configured",
                        "type": "not_implemented",
                    }
                },
                status_code=501,
            )
        )
    try:
        route = resolve_decision(model)
    except (KeyError, AttributeError):
        decision_models = getattr(config, "decision_models", {})
        configured = (
            ", ".join(sorted(decision_models.keys())) if decision_models else ""
        )
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

    # --- Forward via transport ---
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
        logger.info(
            "decision: %s → %s %dms status=%d",
            model,
            route.provider_name,
            duration_ms,
            status_code,
        )
