"""Provider connectivity test handler."""

from __future__ import annotations

from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response
from llm_rosetta.gateway.transport.provider_info import _VERSION_SUFFIXES


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

    base_url = provider_cfg.get("base_url", "").rstrip("/")
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

    # 1. Probe base URL
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
    if ptype == "google":
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

    # 3. Check embedding endpoint if configured (POST-only, so just verify reachable)
    if provider_cfg.get("embedding_format"):
        embedding_path = provider_cfg.get("embedding_path", "/v1/embeddings")
        embed_url = f"{base_url}{embedding_path}"
        _check_double_prefix(base_url, embedding_path, "embedding", results)
        try:
            resp = await client.post(embed_url, headers=auth_headers, json={})
            results["endpoints"]["embedding"] = {
                "url": embed_url,
                "status": resp.status_code,
                "ok": resp.status_code != 404,
            }
        except Exception as exc:
            results["endpoints"]["embedding"] = {
                "url": embed_url,
                "status": None,
                "ok": False,
                "error": str(exc),
            }

    # 4. Check rerank endpoint if configured (POST-only, so just verify reachable)
    if provider_cfg.get("rerank_format"):
        rerank_path = provider_cfg.get("rerank_path", "/v1/rerank")
        rerank_url = f"{base_url}{rerank_path}"
        _check_double_prefix(base_url, rerank_path, "rerank", results)
        try:
            resp = await client.post(rerank_url, headers=auth_headers, json={})
            results["endpoints"]["rerank"] = {
                "url": rerank_url,
                "status": resp.status_code,
                "ok": resp.status_code != 404,
            }
        except Exception as exc:
            results["endpoints"]["rerank"] = {
                "url": rerank_url,
                "status": None,
                "ok": False,
                "error": str(exc),
            }

    return JSONResponse(results)


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
