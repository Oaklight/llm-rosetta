"""Provider connectivity test handler."""

from __future__ import annotations

import re
from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response

_VERSION_PREFIX = re.compile(r"/v\d+(?:beta\d?)?(?=/|$)", re.IGNORECASE)


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

    # 2. Check for models endpoint (OpenAI convention)
    models_url = f"{base_url}/models"
    try:
        resp = await client.get(models_url)
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

    # 3. Check embedding endpoint if configured
    embedding_path = provider_cfg.get("embedding_path", "/v1/embeddings")
    if provider_cfg.get("embedding_format"):
        embed_url = f"{base_url}{embedding_path}"
        from llm_rosetta.gateway.transport.provider_info import _normalize_base_url

        normalized_base = _normalize_base_url(base_url, "{base_url}" + embedding_path)
        normalized_url = f"{normalized_base}{embedding_path}"
        _check_double_prefix(base_url, embedding_path, "embedding", results)
        try:
            resp = await client.get(embed_url)
            results["endpoints"]["embedding"] = {
                "url": embed_url,
                "normalized_url": normalized_url,
                "status": resp.status_code,
                "ok": resp.status_code != 404,
            }
        except Exception as exc:
            results["endpoints"]["embedding"] = {
                "url": embed_url,
                "normalized_url": normalized_url,
                "status": None,
                "ok": False,
                "error": str(exc),
            }

    # 4. Check rerank endpoint if configured
    rerank_path = provider_cfg.get("rerank_path", "/v1/rerank")
    if provider_cfg.get("rerank_format"):
        rerank_url = f"{base_url}{rerank_path}"
        from llm_rosetta.gateway.transport.provider_info import _normalize_base_url

        normalized_base = _normalize_base_url(base_url, "{base_url}" + rerank_path)
        normalized_rerank_url = f"{normalized_base}{rerank_path}"
        _check_double_prefix(base_url, rerank_path, "rerank", results)
        try:
            resp = await client.get(rerank_url)
            results["endpoints"]["rerank"] = {
                "url": rerank_url,
                "normalized_url": normalized_rerank_url,
                "status": resp.status_code,
                "ok": resp.status_code != 404,
            }
        except Exception as exc:
            results["endpoints"]["rerank"] = {
                "url": rerank_url,
                "normalized_url": normalized_rerank_url,
                "status": None,
                "ok": False,
                "error": str(exc),
            }

    return JSONResponse(results)


def _check_double_prefix(
    base_url: str, path: str, endpoint_name: str, results: dict
) -> None:
    """Warn if base_url and path share a version prefix."""
    m = _VERSION_PREFIX.search(base_url)
    if m and path.startswith(m.group()):
        results["warnings"].append(
            f"{endpoint_name}: base_url ends with '{m.group()}' and "
            f"{endpoint_name}_path starts with '{m.group()}' — "
            f"this would cause a double prefix but is auto-corrected by "
            f"the gateway's base_url normalization."
        )
