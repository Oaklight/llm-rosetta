"""Observability route handlers: metrics, request log, network diagnostics."""

from __future__ import annotations

from typing import Any

from llm_rosetta._vendor.httpclient import AsyncClient, Response as HttpResponse
from llm_rosetta._vendor.httpserver import JSONResponse, Response

from ...config import GatewayConfig
from ._shared import _qp


def _detect_host_ip() -> dict[str, Any]:
    """Detect the Docker host IP from the default gateway in /proc/net/route.

    This is a synchronous, microsecond-level operation that reads a
    single procfs file — safe to call on every page load.

    Returns:
        Dict with ``ok``, ``ip`` (on success) or ``error`` (on failure).
    """
    try:
        with open("/proc/net/route") as f:
            for line in f:
                fields = line.strip().split()
                if fields[1] == "00000000":  # default route
                    gw = int(fields[2], 16)
                    host_ip = (
                        f"{gw & 0xFF}.{(gw >> 8) & 0xFF}"
                        f".{(gw >> 16) & 0xFF}.{(gw >> 24) & 0xFF}"
                    )
                    return {"ok": True, "ip": host_ip}
        return {"ok": False, "error": "No default route"}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def _persistence_snapshot(persistence: Any) -> dict[str, Any] | None:
    """Build the persistence sub-block for the metrics snapshot.

    Returns a dict with on-disk byte sizes and per-class entry counts
    plus retention caps, or ``None`` when persistence is not configured
    (e.g. in tests without a config file).
    """
    if persistence is None:
        return None
    try:
        sizes = persistence.db_file_sizes()
        return {
            **sizes,
            "log_entries": persistence.count_log_entries(),
            "log_success_entries": persistence.count_success_entries(),
            "log_error_entries": persistence.count_error_entries(),
            "log_max_success": persistence.success_max,
            "log_max_error": persistence.error_max,
        }
    except Exception:
        # Persistence introspection is purely informational; never let
        # it block the metrics endpoint.
        return None


async def get_metrics(request: Any) -> Response:
    """Return a full metrics snapshot."""
    metrics = request.app.metrics
    seconds = int(_qp(request, "seconds", "60"))
    seconds = max(1, min(seconds, 300))
    snap = metrics.snapshot(series_seconds=seconds)

    persistence_snap = _persistence_snapshot(getattr(request.app, "persistence", None))
    if persistence_snap is not None:
        snap["persistence"] = persistence_snap

    return JSONResponse(snap)


async def get_request_key_labels(request: Any) -> Response:
    """Return API key labels seen in request logs."""
    log = request.app.request_log
    return JSONResponse({"labels": log.get_api_key_labels()})


async def get_requests(request: Any) -> Response:
    """Return paginated, filtered request log entries."""
    log = request.app.request_log
    limit = int(_qp(request, "limit", "50"))
    offset = int(_qp(request, "offset", "0"))
    model = _qp(request, "model")
    provider = _qp(request, "provider")
    status = _qp(request, "status")
    api_key_label = _qp(request, "api_key_label")

    # Resolve provider name → provider type so the query can match old
    # entries that only have target_provider (the API type) without a
    # target_provider_name backfill.  Use the raw config so that even
    # disabled providers (missing API key, etc.) are resolvable.
    provider_type: str | None = None
    if provider:
        config: GatewayConfig | None = getattr(request.app, "gateway_config", None)
        if config is not None:
            provider_type = config.provider_types.get(provider)
        if provider_type is None:
            # Fallback: read from raw config file (covers disabled providers)
            from ._shared import _get_config_path
            from ...config import load_config_raw

            config_path = _get_config_path(request)
            if config_path:
                try:
                    raw = load_config_raw(config_path)
                    raw_prov = raw.get("providers", {}).get(provider, {})
                    raw_type = raw_prov.get("type") or raw_prov.get("shim")
                    if raw_type:
                        from llm_rosetta.shims import resolve_base

                        provider_type = resolve_base(raw_type)
                except Exception:
                    pass

    entries, total = log.get_entries(
        limit=limit,
        offset=offset,
        model=model,
        provider=provider,
        provider_type=provider_type,
        status=status,
        api_key_label=api_key_label,
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

    results["host"] = _detect_host_ip()

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


async def get_host_ip(request: Any) -> Response:
    """Return the detected Docker host IP (lightweight, no network calls)."""
    return JSONResponse(_detect_host_ip())
