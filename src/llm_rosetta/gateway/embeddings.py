"""Embeddings passthrough handler.

Proxies ``/v1/embeddings`` requests to the upstream provider without
format conversion — the OpenAI embeddings API format is universal
across providers that support it.
"""

from __future__ import annotations

import time
from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response

from .auth import api_key_label_var
from .config import GatewayConfig
from .logging import get_logger
from .transport import UpstreamConnectionError

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

    # --- Resolve provider ---
    try:
        _, provider_info, _, upstream_model, provider_name = config.resolve_model(model)
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
    if upstream_model:
        body["model"] = upstream_model

    # --- Build upstream URL ---
    base_url = provider_info.base_url
    upstream_url = f"{base_url}/embeddings"

    # --- Forward request ---
    headers = provider_info.auth_headers()
    headers["Content-Type"] = "application/json"

    metrics = getattr(request.app, "metrics", None)
    request_log = getattr(request.app, "request_log", None)
    t0 = time.monotonic()
    status_code = 500
    error_detail: str | None = None

    # Use the transport's client pool for connection reuse
    from .transport.http import HttpTransport

    transport = request.app.transport
    assert isinstance(transport, HttpTransport)
    client = transport._pool.get(provider_info.proxy_url)

    try:
        from llm_rosetta._vendor.httpclient import (
            HttpClientError,
            Response as HttpResponse,
        )

        upstream_resp = await client.post(upstream_url, json=body, headers=headers)
        assert isinstance(upstream_resp, HttpResponse)

        status_code = upstream_resp.status_code

        if upstream_resp.status_code >= 400:
            error_detail = upstream_resp.text
            return Response(
                body=upstream_resp.content,
                status_code=upstream_resp.status_code,
                content_type="application/json",
            )

        return Response(
            body=upstream_resp.content,
            status_code=200,
            content_type="application/json",
        )
    except (HttpClientError, UpstreamConnectionError) as exc:
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
        duration_ms = (time.monotonic() - t0) * 1000
        provider_type = config.provider_types.get(
            config.models.get(model, ""), "unknown"
        )
        if metrics:
            metrics.record_request(
                model=model,
                source="openai_chat",
                target=provider_type,
                status_code=status_code,
                duration_ms=duration_ms,
                is_stream=False,
            )
        if request_log is not None:
            from .admin.request_log import RequestLogEntry
            from .app import _extract_client_ip

            api_key_label = api_key_label_var.get()
            client_ip = _extract_client_ip(request)
            request_log.add(
                RequestLogEntry.create(
                    model=model,
                    source_provider="openai_chat",
                    target_provider=provider_type,
                    target_provider_name=provider_name,
                    is_stream=False,
                    status_code=status_code,
                    duration_ms=duration_ms,
                    error_detail=error_detail,
                    api_key_label=api_key_label,
                    client_ip=client_ip,
                )
            )
