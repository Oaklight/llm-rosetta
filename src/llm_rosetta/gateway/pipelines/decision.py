"""Decision proxy handler.

Proxies ``/v1/decision`` and ``/v1/systemone`` requests to upstream
decision providers (currently TypeSafe System One).  No streaming —
decision requests return a single JSON response.

Registered via the model type registry in ``model_types.py``.
"""

from __future__ import annotations

import time
from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response

from ..config import GatewayConfig
from ..middleware.headers import build_upstream_extra_headers, get_request_id
from ..logging import get_logger
from ..transport import UpstreamConnectionError, UpstreamTimeoutError, UpstreamTransport

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
