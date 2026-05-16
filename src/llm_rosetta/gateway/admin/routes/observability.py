"""Observability route handlers: metrics, request log, network diagnostics."""

from __future__ import annotations

from typing import Any

from llm_rosetta._vendor.httpclient import AsyncClient, Response as HttpResponse
from llm_rosetta._vendor.httpserver import JSONResponse, Response

from ...config import GatewayConfig
from ._shared import _qp


async def get_metrics(request: Any) -> Response:
    """Return a full metrics snapshot."""
    metrics = request.app.metrics
    seconds = int(_qp(request, "seconds", "60"))
    seconds = max(1, min(seconds, 300))
    return JSONResponse(metrics.snapshot(series_seconds=seconds))


async def get_requests(request: Any) -> Response:
    """Return paginated, filtered request log entries."""
    log = request.app.request_log
    limit = int(_qp(request, "limit", "50"))
    offset = int(_qp(request, "offset", "0"))
    model = _qp(request, "model")
    provider = _qp(request, "provider")
    status = _qp(request, "status")

    entries, total = log.get_entries(
        limit=limit, offset=offset, model=model, provider=provider, status=status
    )
    return JSONResponse({"entries": entries, "total": total})


async def clear_requests(request: Any) -> Response:
    """Clear the request log."""
    log = request.app.request_log
    log.clear()
    return JSONResponse({"ok": True})


async def get_provider_key(request: Any, **kwargs: Any) -> Response:
    """Return the raw (unmasked) API key for a single provider."""
    config: GatewayConfig = request.app.gateway_config
    if not config.credential_visible:
        return JSONResponse(
            {"error": "Credential visibility is disabled"}, status_code=403
        )
    from ._shared import _get_config_path
    from ...config import load_config_raw

    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    name = request.path_params["name"]

    try:
        data = load_config_raw(config_path)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read config: {exc}"}, status_code=500)

    provider = data.get("providers", {}).get(name)
    if not provider:
        return JSONResponse({"error": f"Provider '{name}' not found"}, status_code=404)

    return JSONResponse({"api_key": provider.get("api_key", "")})


async def network_diagnostics(request: Any) -> Response:
    """Run basic network diagnostics: IP geolocation and Google connectivity.

    Uses the gateway's configured global proxy (if any) so the diagnostics
    reflect the actual outbound path of API requests.
    """
    # Resolve the global proxy from current gateway config
    gw_config: GatewayConfig | None = getattr(request.app, "gateway_config", None)
    proxy_url = gw_config.proxy if gw_config else None

    client_kwargs: dict[str, Any] = {"timeout": 15.0}
    if proxy_url:
        client_kwargs["proxy"] = proxy_url

    results: dict[str, Any] = {}
    if proxy_url:
        results["proxy"] = proxy_url

    # IP geolocation via ip-api.com (no key required, JSON by default)
    try:
        async with AsyncClient(**client_kwargs) as client:
            resp = await client.get(
                "http://ip-api.com/json/?fields=query,country,city,isp"
            )
            assert isinstance(resp, HttpResponse)
            if resp.status_code == 200:
                data = resp.json()
                results["ip"] = {
                    "ok": True,
                    "ip": data.get("query", ""),
                    "country": data.get("country", ""),
                    "city": data.get("city", ""),
                    "isp": data.get("isp", ""),
                }
            else:
                results["ip"] = {"ok": False, "error": f"HTTP {resp.status_code}"}
    except Exception as exc:
        results["ip"] = {"ok": False, "error": str(exc)}

    # Google connectivity
    try:
        async with AsyncClient(**client_kwargs) as client:
            resp = await client.get("https://www.google.com/generate_204")
            results["google"] = {
                "ok": resp.status_code == 204,
                "status": resp.status_code,
            }
    except Exception as exc:
        results["google"] = {"ok": False, "error": str(exc)}

    return JSONResponse(results)
