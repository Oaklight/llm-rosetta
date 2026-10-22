"""Upstream model discovery and bulk add, plus routing-loop checks."""

from __future__ import annotations

import logging
from typing import Any

from llm_rosetta_gateway._vendor.httpclient import AsyncClient, Response as HttpResponse
from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response
from llm_rosetta.shims import get_shim

from ....providers import default_models_path
from .._shared import (
    _resolve_models_path,
)

from .payload import _get_gateway_config

logger = logging.getLogger("llm-rosetta-gateway")


def _format_connection_error(exc: Exception, url: str) -> str:
    """Return a user-friendly message for common upstream connection errors."""
    err_str = str(exc)
    if "Connection refused" in err_str or "Errno 111" in err_str:
        return (
            f"Connection refused at {url}. "
            "Check that the service is running and the port is correct. "
            "If running in Docker, ensure the host firewall (e.g. ufw) "
            "allows connections from the Docker bridge network."
        )
    if "timed out" in err_str.lower():
        return (
            f"Connection to {url} timed out. "
            "Check that the host/port is reachable from this container."
        )
    if "Name or service not known" in err_str or "getaddrinfo" in err_str:
        return f"Cannot resolve hostname in {url}. Check the Base URL."
    return f"Failed to connect to upstream: {err_str}"


def _extract_model_ids(
    body: dict[str, Any] | list[dict[str, Any]],
    ptype: str,
    shim_name: str | None,
    id_field: str | None,
) -> tuple[list[str], dict[str, str]]:
    """Extract model IDs from an upstream ``/models`` response body.

    Returns ``(model_ids, upstream_map)``.  Handles three response shapes:

    * OpenAI-style ``{data: [{id: ...}, ...]}``
    * Google-style ``{models: [{name: ...}, ...]}``
    * Bare JSON array ``[{id: ...}, ...]`` (e.g. ALCF) — requires a
      ``model_list_transform``; without one the array is silently ignored.

    When a shim provides a ``model_list_transform`` hook, it is used to
    derive human-readable slugs and the mapping back to internal upstream
    IDs.
    """
    from llm_rosetta.shims.providers import get_model_list_transform

    mlt = get_model_list_transform(shim_name) if shim_name else None

    upstream_map: dict[str, str] = {}
    model_ids: list[str] = []

    if mlt is not None:
        if isinstance(body, list):
            raw_entries: list[dict[str, Any]] = body
        else:
            raw_entries = (
                body.get("data", []) if "data" in body else body.get("models", [])
            )
        model_ids, upstream_map = mlt(raw_entries)
    elif isinstance(body, list):
        logger.debug(
            "Upstream returned bare array but no model_list_transform is registered "
            "for shim %r — cannot extract model IDs",
            shim_name,
        )
        return [], {}
    elif ptype in ("google", "google_generate"):
        for m in body.get("models", []):
            name = m.get("name", "")
            if name.startswith("models/"):
                name = name[len("models/") :]
            model_ids.append(m.get(id_field, name) if id_field else name)
    else:
        for m in body.get("data", []):
            model_ids.append(
                m.get(id_field, m.get("id", "")) if id_field else m.get("id", "")
            )

    model_ids = [m for m in model_ids if m]
    model_ids.sort()
    return model_ids, upstream_map


async def fetch_upstream_models(request: Any, **kwargs: Any) -> Response:
    """Fetch the model list from an upstream provider's /v1/models endpoint."""

    provider_name = request.path_params["name"]
    config = _get_gateway_config(request)
    if config is None:
        return JSONResponse({"error": "Gateway config not loaded"}, status_code=500)

    if provider_name not in config.providers:
        return JSONResponse(
            {"error": f"Provider '{provider_name}' not found"}, status_code=404
        )

    pinfo = config.providers[provider_name]
    ptype = config.provider_types.get(provider_name, "unknown")

    # A token_command provider whose initial fetch has not completed still
    # holds the pending sentinel.  Sending it would come back as an upstream
    # 401 that reads as "this provider may not support model listing", so
    # report the real state instead.
    if not pinfo.ready:
        return JSONResponse(
            {
                "error": (
                    "Provider is still starting up — its API key has not been "
                    "fetched yet. Retry in a moment."
                )
            }
        )

    # Build the models listing URL.  Explicit ``models_path`` in the
    # provider config takes precedence (e.g. ``/v1/models`` for Jina).
    raw_cfg = getattr(config, "_raw_providers", {}).get(provider_name, {})
    explicit_path = _resolve_models_path(raw_cfg, config, provider_name)
    if explicit_path:
        if explicit_path.startswith(("https://", "http://")):
            models_url = explicit_path
        else:
            models_url = f"{pinfo.base_url}{explicit_path}"
    else:
        models_url = f"{pinfo.base_url}{default_models_path(ptype)}"

    headers = {**pinfo.extra_headers, **pinfo.auth_headers()}

    try:
        client = AsyncClient(timeout=10.0, proxy=pinfo.proxy_url)
        raw_resp = await client.get(models_url, headers=headers)
        assert isinstance(raw_resp, HttpResponse), "Expected non-streaming response"
        resp: HttpResponse = raw_resp
        await client.aclose()
    except Exception as exc:
        logger.warning("Failed to fetch models from %s: %s", provider_name, exc)
        msg = _format_connection_error(exc, models_url)
        return JSONResponse({"error": msg})  # 200 so reverse proxies don't intercept

    if resp.status_code >= 400:
        logger.warning(
            "Upstream %s returned %d for model listing", provider_name, resp.status_code
        )
        return JSONResponse(
            {
                "error": (
                    f"Upstream returned HTTP {resp.status_code}. "
                    "This provider may not support model listing."
                ),
            },
        )

    try:
        body = resp.json()
    except Exception:
        return JSONResponse(
            {"error": "Upstream returned non-JSON response"},
        )

    # Resolve model_id_field from shim (e.g. "internal_id" instead of "model")
    shim_name = config.provider_shim_names.get(provider_name)
    shim = get_shim(shim_name) if shim_name else None
    id_field = (
        shim.connection.model_id_field
        if shim and shim.connection.model_id_field
        else None
    )

    model_ids, upstream_map = _extract_model_ids(body, ptype, shim_name, id_field)

    result: dict[str, Any] = {
        "provider": provider_name,
        "api_standard": ptype,
        "models": model_ids,
    }
    if upstream_map:
        result["upstream_map"] = upstream_map
    return JSONResponse(result)


async def check_routing_loops(request: Any) -> Response:
    """Return any detected routing loops in the current config."""
    from ....config import detect_routing_loops

    config = getattr(request.app, "config", None)
    if config is None:
        return JSONResponse({"warnings": []})

    warnings = detect_routing_loops(config.models, config.model_upstream_names)
    return JSONResponse({"warnings": warnings})
