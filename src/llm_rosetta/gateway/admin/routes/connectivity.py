"""Provider connectivity test handler."""

from __future__ import annotations

from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response
from llm_rosetta.gateway.transport.provider_info import _VERSION_SUFFIXES

from ._shared import _resolve_models_path


def _resolve_base_url(provider_cfg: dict, config: Any, name: str) -> str:
    """Return the base_url from raw config, falling back to shim defaults."""
    url = provider_cfg.get("base_url", "").rstrip("/")
    if not url:
        pinfo = config.providers.get(name) if hasattr(config, "providers") else None
        if pinfo and pinfo.base_url:
            url = pinfo.base_url.rstrip("/")
    return url


async def test_provider_connectivity(request: Any, name: str) -> Response:
    """Probe a provider's base_url and endpoint paths for reachability.

    Checks:
    1. Base URL reachability (GET with short timeout)
    2. Each configured endpoint (models list, embedding, rerank)
    3. Double version prefix detection
    """
    config = getattr(request.app, "config", None) or getattr(
        request.app, "gateway_config", None
    )
    if config is None:
        return JSONResponse({"error": "No config loaded"}, status_code=500)

    raw_providers = getattr(config, "_raw_providers", {})
    provider_cfg = raw_providers.get(name)
    if provider_cfg is None:
        return JSONResponse({"error": f"Provider '{name}' not found"}, status_code=404)

    base_url = _resolve_base_url(provider_cfg, config, name)
    if not base_url:
        return JSONResponse(
            {"error": "Provider has no base_url configured"}, status_code=400
        )

    from llm_rosetta._vendor.httpclient import AsyncClient

    proxy = provider_cfg.get("proxy") or getattr(config, "proxy", None)
    timeout = float(provider_cfg.get("timeout", 10))
    client = AsyncClient(timeout=min(timeout, 10), proxy=proxy)

    # Build auth headers from provider config (same as fetch-models)
    pinfo = config.providers.get(name) if hasattr(config, "providers") else None
    auth_headers = pinfo.auth_headers() if pinfo else {}

    results: dict[str, Any] = {
        "provider": name,
        "base_url": base_url,
        "endpoints": {},
        "warnings": [],
    }

    # 1. Probe base URL — deliberately unauthenticated. This only answers
    # "can we reach the host at all", so an auth rejection would be noise;
    # any HTTP response, including 401/404, proves reachability.
    try:
        resp = await client.get(base_url)
        results["reachable"] = True
        results["base_status"] = resp.status_code
    except Exception as exc:
        results["reachable"] = False
        results["base_status"] = None
        results["base_error"] = str(exc)

    # 2. Check models endpoint (type-aware URL, same as fetch_upstream_models)
    ptype = (
        config.provider_types.get(name, "unknown")
        if hasattr(config, "provider_types")
        else "unknown"
    )
    explicit_path = _resolve_models_path(provider_cfg, config, name)
    if explicit_path:
        if explicit_path.startswith(("https://", "http://")):
            models_url = explicit_path
        else:
            models_url = f"{base_url}{explicit_path}"
    elif ptype == "google":
        models_url = f"{base_url}/v1beta/models"
    elif ptype == "anthropic":
        models_url = f"{base_url}/v1/models"
    else:
        models_url = f"{base_url}/models"
    try:
        resp = await client.get(models_url, headers=auth_headers)
        results["endpoints"]["models"] = {
            "url": models_url,
            "status": resp.status_code,
            "ok": resp.status_code < 400,
        }
    except Exception as exc:
        results["endpoints"]["models"] = {
            "url": models_url,
            "status": None,
            "ok": False,
            "error": str(exc),
        }

    # 3–5. Check optional POST endpoints (embedding, rerank, decision)
    for ep_name, fmt_key, path_key, default_path in (
        ("embedding", "embedding_format", "embedding_path", "/v1/embeddings"),
        ("rerank", "rerank_format", "rerank_path", "/v1/rerank"),
        ("decision", "decision_format", "decision_path", "/v1/systemone"),
    ):
        if not provider_cfg.get(fmt_key):
            continue
        await _check_post_endpoint(
            client,
            base_url,
            auth_headers,
            provider_cfg.get(path_key, default_path),
            ep_name,
            results,
        )

    return JSONResponse(results)


async def _check_post_endpoint(
    client: Any,
    base_url: str,
    auth_headers: dict[str, str],
    path: str,
    ep_name: str,
    results: dict[str, Any],
) -> None:
    """Probe a POST endpoint for reachability."""
    url = f"{base_url}{path}"
    _check_double_prefix(base_url, path, ep_name, results)
    try:
        resp = await client.post(url, headers=auth_headers, json={})
        results["endpoints"][ep_name] = {
            "url": url,
            "status": resp.status_code,
            "ok": resp.status_code != 404,
        }
    except Exception as exc:
        results["endpoints"][ep_name] = {
            "url": url,
            "status": None,
            "ok": False,
            "error": str(exc),
        }


def _check_double_prefix(
    base_url: str, path: str, endpoint_name: str, results: dict
) -> None:
    """Warn if base_url and path share a version prefix."""
    m = _VERSION_SUFFIXES.search(base_url)
    if m and path.startswith(m.group()):
        results["warnings"].append(
            f"{endpoint_name}: base_url ends with '{m.group()}' and "
            f"{endpoint_name}_path starts with '{m.group()}' — "
            f"this would cause a double prefix but is auto-corrected by "
            f"the gateway's base_url normalization."
        )
