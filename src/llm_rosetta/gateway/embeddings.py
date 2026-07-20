"""Embeddings passthrough handler.

Proxies ``/v1/embeddings`` requests to the upstream provider without
format conversion — the OpenAI embeddings API format is universal
across providers that support it.

Uses the :class:`~transport.UpstreamTransport` interface so the handler
is transport-agnostic (works with HTTP, and future gRPC/WebSocket).
"""

from __future__ import annotations

import time
from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response

from .config import GatewayConfig
from .headers import build_upstream_extra_headers
from .logging import get_logger
from .transport import UpstreamConnectionError, UpstreamTransport

logger = get_logger()


async def handle_embeddings(
    request: Any,
    config: GatewayConfig,
) -> Response:
    """Proxy an embeddings request to the upstream provider.

    This is a thin passthrough — no IR conversion is performed.
    The request body is forwarded as-is after model resolution.

    Args:
        request: The incoming HTTP request.
        config: The live gateway configuration.

    Returns:
        The upstream response, forwarded to the client.
    """
    # --- Parse request ---
    try:
        body: dict[str, Any] = request.json()
    except Exception:
        return JSONResponse(
            {
                "error": {
                    "message": "Invalid JSON body",
                    "type": "invalid_request_error",
                }
            },
            status_code=400,
        )

    model = body.get("model")
    if not model:
        return JSONResponse(
            {
                "error": {
                    "message": "Missing 'model' in request body",
                    "type": "invalid_request_error",
                }
            },
            status_code=400,
        )

    # --- Resolve provider via unified routing ---
    try:
        route, provider_info = config.resolve("openai_chat", model)
    except KeyError:
        configured = ", ".join(sorted(config.models.keys()))
        return JSONResponse(
            {
                "error": {
                    "message": f"Unknown model: '{model}'. Configured models: {configured}",
                    "type": "model_not_found",
                }
            },
            status_code=404,
        )

    # Model alias: replace the model name in the request body with the
    # actual upstream identifier so the upstream provider sees the correct name.
    if route.upstream_model:
        body["model"] = route.upstream_model

    # --- Forward via transport ---
    upstream_url = f"{provider_info.base_url}/embeddings"
    transport: UpstreamTransport = request.app.transport
    request_id = request.headers.get("x-request-id", "")
    extra_headers = build_upstream_extra_headers(request, request_id)

    t0 = time.monotonic()
    status_code = 500
    error_detail: str | None = None

    try:
        resp = await transport.send_passthrough(
            provider_info,
            upstream_url,
            body,
            extra_headers=extra_headers,
        )
        status_code = resp.status_code

        if resp.is_error:
            error_detail = resp.error_text
            return Response(
                body=resp.raw_content,
                status_code=resp.status_code,
                content_type="application/json",
            )

        return Response(
            body=resp.raw_content,
            status_code=200,
            content_type="application/json",
        )
    except UpstreamConnectionError as exc:
        error_detail = str(exc)
        status_code = 502
        return JSONResponse(
            {
                "error": {
                    "message": f"Upstream request failed: {exc}",
                    "type": "upstream_error",
                }
            },
            status_code=502,
        )
    except Exception as exc:
        error_detail = str(exc)
        raise
    finally:
        from .app import _record_telemetry

        _record_telemetry(
            request,
            model=model,
            source_provider="openai_chat",
            target_provider=route.target_provider,
            provider_name=route.provider_name,
            is_stream=False,
            status_code=status_code,
            duration_ms=(time.monotonic() - t0) * 1000,
            error_detail=error_detail,
        )
