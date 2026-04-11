"""Starlette route handlers for the admin panel API."""

from __future__ import annotations

import re
from typing import Any

from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from ..config import GatewayConfig, load_config, load_config_raw, write_config
from ..providers import known_provider_types
from .static import load_admin_html

# Cached HTML — loaded once on first request.
_admin_html: str | None = None

_ENV_VAR_RE = re.compile(r"^\$\{.+\}$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _mask_api_key(value: str) -> str:
    """Mask a literal API key, leaving env-var placeholders intact."""
    if _ENV_VAR_RE.match(value):
        return value
    if len(value) <= 8:
        return "***"
    return value[:4] + "***" + value[-4:]


def _get_config_path(request: Request) -> str | None:
    return getattr(request.app.state, "config_path", None)


def _reload_gateway_config(request: Request, config_path: str) -> GatewayConfig:
    """Re-read config from disk, rebuild GatewayConfig, swap into app state.

    The import of ``app._config`` is deferred to avoid circular imports.
    """
    import llm_rosetta.gateway.app as _app_mod

    raw = load_config(config_path)
    new_config = GatewayConfig(raw)
    _app_mod._config = new_config
    request.app.state.gateway_config = new_config
    return new_config


# ---------------------------------------------------------------------------
# HTML handler
# ---------------------------------------------------------------------------


async def serve_admin_html(request: Request) -> Response:
    """Serve the admin panel SPA."""
    global _admin_html
    if _admin_html is None:
        _admin_html = load_admin_html()
    return HTMLResponse(_admin_html)


# ---------------------------------------------------------------------------
# Config API
# ---------------------------------------------------------------------------


async def get_config(request: Request) -> Response:
    """Return the current (raw) gateway configuration."""
    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    try:
        raw = load_config_raw(config_path)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read config: {exc}"}, status_code=500)

    # Mask API keys in the response
    providers = raw.get("providers", {})
    masked_providers: dict[str, Any] = {}
    for name, cfg in providers.items():
        masked = dict(cfg)
        if "api_key" in masked:
            masked["api_key"] = _mask_api_key(masked["api_key"])
        masked_providers[name] = masked

    # Normalize models to dict format for consistent admin UI
    raw_models = raw.get("models", {})
    models_normalized: dict[str, Any] = {}
    for name, value in raw_models.items():
        if isinstance(value, str):
            models_normalized[name] = {"provider": value, "capabilities": ["text"]}
        elif isinstance(value, dict):
            models_normalized[name] = {
                "provider": value.get("provider", ""),
                "capabilities": value.get("capabilities", ["text"]),
            }

    return JSONResponse(
        {
            "config_path": config_path,
            "providers": masked_providers,
            "models": models_normalized,
            "server": raw.get("server", {}),
            "known_provider_types": known_provider_types(),
        }
    )


async def put_provider(request: Request) -> Response:
    """Add or update a provider entry."""
    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    name = request.path_params["name"]

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    api_key = body.get("api_key")
    base_url = body.get("base_url")
    if not api_key or not base_url:
        return JSONResponse(
            {"error": "Both 'api_key' and 'base_url' are required"}, status_code=400
        )

    try:
        data = load_config_raw(config_path)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read config: {exc}"}, status_code=500)

    provider_entry: dict[str, Any] = {"api_key": api_key, "base_url": base_url}

    # Include proxy if provided, empty string removes it
    if "proxy" in body:
        proxy = body["proxy"]
        if proxy:
            provider_entry["proxy"] = proxy

    data.setdefault("providers", {})[name] = provider_entry

    try:
        write_config(config_path, data)
    except Exception as exc:
        return JSONResponse(
            {"error": f"Failed to write config: {exc}"}, status_code=500
        )

    try:
        new_config = _reload_gateway_config(request, config_path)
    except Exception as exc:
        return JSONResponse(
            {
                "error": f"Config saved but reload failed: {exc}",
                "saved": True,
                "reloaded": False,
            },
            status_code=500,
        )

    return JSONResponse(
        {
            "ok": True,
            "provider": name,
            "providers": list(new_config.providers.keys()),
        }
    )


async def delete_provider(request: Request) -> Response:
    """Remove a provider entry."""
    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    name = request.path_params["name"]

    try:
        data = load_config_raw(config_path)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read config: {exc}"}, status_code=500)

    providers = data.get("providers", {})
    if name not in providers:
        return JSONResponse({"error": f"Provider '{name}' not found"}, status_code=404)

    # Check if any model still references this provider
    models = data.get("models", {})
    referencing = [m for m, p in models.items() if p == name]
    if referencing:
        return JSONResponse(
            {
                "error": f"Cannot delete provider '{name}': referenced by models: {referencing}"
            },
            status_code=409,
        )

    del providers[name]

    try:
        write_config(config_path, data)
    except Exception as exc:
        return JSONResponse(
            {"error": f"Failed to write config: {exc}"}, status_code=500
        )

    try:
        new_config = _reload_gateway_config(request, config_path)
    except Exception as exc:
        return JSONResponse(
            {
                "error": f"Config saved but reload failed: {exc}",
                "saved": True,
                "reloaded": False,
            },
            status_code=500,
        )

    return JSONResponse(
        {
            "ok": True,
            "deleted": name,
            "providers": list(new_config.providers.keys()),
        }
    )


async def put_model(request: Request) -> Response:
    """Add or update a model routing entry."""
    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    name = request.path_params["name"]

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    provider = body.get("provider")
    if not provider:
        return JSONResponse({"error": "'provider' is required"}, status_code=400)

    try:
        data = load_config_raw(config_path)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read config: {exc}"}, status_code=500)

    # Validate that the provider exists
    providers = data.get("providers", {})
    if provider not in providers:
        return JSONResponse(
            {"error": f"Provider '{provider}' not found in config"}, status_code=400
        )

    capabilities = body.get("capabilities", ["text"])
    data.setdefault("models", {})[name] = {
        "provider": provider,
        "capabilities": capabilities,
    }

    try:
        write_config(config_path, data)
    except Exception as exc:
        return JSONResponse(
            {"error": f"Failed to write config: {exc}"}, status_code=500
        )

    try:
        new_config = _reload_gateway_config(request, config_path)
    except Exception as exc:
        return JSONResponse(
            {
                "error": f"Config saved but reload failed: {exc}",
                "saved": True,
                "reloaded": False,
            },
            status_code=500,
        )

    return JSONResponse(
        {
            "ok": True,
            "model": name,
            "provider": provider,
            "capabilities": capabilities,
            "models": dict(new_config.models),
        }
    )


async def delete_model(request: Request) -> Response:
    """Remove a model routing entry."""
    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    name = request.path_params["name"]

    try:
        data = load_config_raw(config_path)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read config: {exc}"}, status_code=500)

    models = data.get("models", {})
    if name not in models:
        return JSONResponse({"error": f"Model '{name}' not found"}, status_code=404)

    del models[name]

    try:
        write_config(config_path, data)
    except Exception as exc:
        return JSONResponse(
            {"error": f"Failed to write config: {exc}"}, status_code=500
        )

    try:
        new_config = _reload_gateway_config(request, config_path)
    except Exception as exc:
        return JSONResponse(
            {
                "error": f"Config saved but reload failed: {exc}",
                "saved": True,
                "reloaded": False,
            },
            status_code=500,
        )

    return JSONResponse(
        {
            "ok": True,
            "deleted": name,
            "models": dict(new_config.models),
        }
    )


async def put_server_settings(request: Request) -> Response:
    """Update server settings (e.g. global proxy)."""
    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    try:
        data = load_config_raw(config_path)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read config: {exc}"}, status_code=500)

    server = data.setdefault("server", {})

    # Update proxy — empty string removes it
    if "proxy" in body:
        proxy = body["proxy"]
        if proxy:
            server["proxy"] = proxy
        else:
            server.pop("proxy", None)

    try:
        write_config(config_path, data)
    except Exception as exc:
        return JSONResponse(
            {"error": f"Failed to write config: {exc}"}, status_code=500
        )

    try:
        _reload_gateway_config(request, config_path)
    except Exception as exc:
        return JSONResponse(
            {
                "error": f"Config saved but reload failed: {exc}",
                "saved": True,
                "reloaded": False,
            },
            status_code=500,
        )

    return JSONResponse({"ok": True, "server": data.get("server", {})})


async def reload_config(request: Request) -> Response:
    """Force hot-reload of the config from disk."""
    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    try:
        new_config = _reload_gateway_config(request, config_path)
    except Exception as exc:
        return JSONResponse({"error": f"Reload failed: {exc}"}, status_code=500)

    return JSONResponse(
        {
            "ok": True,
            "providers": list(new_config.providers.keys()),
            "models": dict(new_config.models),
        }
    )


# ---------------------------------------------------------------------------
# Metrics API
# ---------------------------------------------------------------------------


async def get_metrics(request: Request) -> Response:
    """Return a full metrics snapshot."""
    metrics = request.app.state.metrics
    seconds = int(request.query_params.get("seconds", "60"))
    seconds = max(1, min(seconds, 300))
    return JSONResponse(metrics.snapshot(series_seconds=seconds))


# ---------------------------------------------------------------------------
# Request log API
# ---------------------------------------------------------------------------


async def get_requests(request: Request) -> Response:
    """Return paginated, filtered request log entries."""
    log = request.app.state.request_log
    limit = int(request.query_params.get("limit", "50"))
    offset = int(request.query_params.get("offset", "0"))
    model = request.query_params.get("model")
    provider = request.query_params.get("provider")
    status = request.query_params.get("status")

    entries, total = log.get_entries(
        limit=limit, offset=offset, model=model, provider=provider, status=status
    )
    return JSONResponse({"entries": entries, "total": total})


async def clear_requests(request: Request) -> Response:
    """Clear the request log."""
    log = request.app.state.request_log
    log.clear()
    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# Route table
# ---------------------------------------------------------------------------

admin_routes: list[Route] = [
    # HTML
    Route("/admin", serve_admin_html, methods=["GET"]),
    Route("/admin/", serve_admin_html, methods=["GET"]),
    # Config CRUD
    Route("/admin/api/config", get_config, methods=["GET"]),
    Route(
        "/admin/api/config/providers/{name}",
        put_provider,
        methods=["PUT"],
    ),
    Route(
        "/admin/api/config/providers/{name}",
        delete_provider,
        methods=["DELETE"],
    ),
    Route(
        "/admin/api/config/models/{name:path}",
        put_model,
        methods=["PUT"],
    ),
    Route(
        "/admin/api/config/models/{name:path}",
        delete_model,
        methods=["DELETE"],
    ),
    Route("/admin/api/config/server", put_server_settings, methods=["PUT"]),
    Route("/admin/api/config/reload", reload_config, methods=["POST"]),
    # Metrics
    Route("/admin/api/metrics", get_metrics, methods=["GET"]),
    # Request log
    Route("/admin/api/requests", get_requests, methods=["GET"]),
    Route("/admin/api/requests", clear_requests, methods=["DELETE"]),
]
