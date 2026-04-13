"""Starlette route handlers for the admin panel API."""

from __future__ import annotations

import re
import secrets
import uuid
from datetime import datetime, timezone
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

    Also refreshes the auth middleware's key set so new/deleted keys take
    effect immediately.  The import of ``app._config`` is deferred to
    avoid circular imports.
    """
    import llm_rosetta.gateway.app as _app_mod

    raw = load_config(config_path)
    new_config = GatewayConfig(raw)
    _app_mod._config = new_config
    request.app.state.gateway_config = new_config

    _sync_auth_middleware(request.app, new_config)

    return new_config


def _sync_auth_middleware(app: Any, config: GatewayConfig) -> None:
    """Walk the ASGI middleware chain and update the auth middleware keys."""
    from ..auth import GatewayAuthMiddleware

    layer: Any = app.middleware_stack
    while layer is not None:
        if isinstance(layer, GatewayAuthMiddleware):
            layer._key_set = config.api_key_set
            layer._labels = dict(config.api_key_labels)
            break
        layer = getattr(layer, "app", None)


# ---------------------------------------------------------------------------
# HTML handler
# ---------------------------------------------------------------------------


async def serve_admin_html(request: Request) -> Response:
    """Serve the admin panel SPA."""
    global _admin_html
    if _admin_html is None:
        _admin_html = load_admin_html()
    return HTMLResponse(
        _admin_html,
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


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

    # Mask API keys and ensure each provider has a "type" field
    providers = raw.get("providers", {})
    masked_providers: dict[str, Any] = {}
    for name, cfg in providers.items():
        masked = dict(cfg)
        if "api_key" in masked:
            masked["api_key"] = _mask_api_key(masked["api_key"])
        # Ensure explicit type — fall back to provider name for legacy configs
        if "type" not in masked:
            masked["type"] = name
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

    # Mask api_keys in server section for the response
    server = dict(raw.get("server", {}))
    if "api_key" in server:
        server["api_key"] = _mask_api_key(server["api_key"])
    if "api_keys" in server:
        server["api_keys"] = [
            {**entry, "key": _mask_api_key(entry.get("key", ""))}
            for entry in server["api_keys"]
        ]

    return JSONResponse(
        {
            "config_path": config_path,
            "providers": masked_providers,
            "models": models_normalized,
            "server": server,
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

    # If the submitted api_key is a masked value (contains ***), preserve the
    # original key from the config file to prevent accidental overwrite.
    existing_providers = data.get("providers", {})
    resolve_name = body.get("rename_from", name) or name
    if "***" in api_key and resolve_name in existing_providers:
        api_key = existing_providers[resolve_name].get("api_key", api_key)

    provider_entry: dict[str, Any] = {"api_key": api_key, "base_url": base_url}

    # Include API standard type if provided
    provider_type = body.get("type")
    if provider_type:
        provider_entry["type"] = provider_type

    # Include proxy if provided, empty string removes it
    if "proxy" in body:
        proxy = body["proxy"]
        if proxy:
            provider_entry["proxy"] = proxy

    # Preserve enabled state from existing entry (toggle is separate endpoint)
    if resolve_name in existing_providers:
        existing_enabled = existing_providers[resolve_name].get("enabled")
        if existing_enabled is not None:
            provider_entry["enabled"] = existing_enabled

    # Handle rename: remove old entry and update model references
    rename_from = body.get("rename_from")
    if rename_from and rename_from != name:
        providers = data.get("providers", {})
        if rename_from not in providers:
            return JSONResponse(
                {"error": f"Original provider '{rename_from}' not found"},
                status_code=404,
            )
        if name in providers:
            return JSONResponse(
                {"error": f"Provider '{name}' already exists"},
                status_code=409,
            )
        del providers[rename_from]
        # Update all model references from old name to new name
        models = data.get("models", {})
        for model_name, model_val in models.items():
            if isinstance(model_val, str) and model_val == rename_from:
                models[model_name] = name
            elif (
                isinstance(model_val, dict) and model_val.get("provider") == rename_from
            ):
                model_val["provider"] = name

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
    referencing = [
        m
        for m, p in models.items()
        if (p["provider"] if isinstance(p, dict) else p) == name
    ]
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


async def toggle_provider(request: Request) -> Response:
    """Toggle a provider's enabled/disabled state."""
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

    # Toggle: if currently enabled (or unset → default True), disable; otherwise enable
    currently_enabled = providers[name].get("enabled", True)
    new_enabled = not currently_enabled

    if new_enabled:
        # Remove the key entirely when re-enabling (True is the default)
        providers[name].pop("enabled", None)
    else:
        providers[name]["enabled"] = False

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

    return JSONResponse({"ok": True, "provider": name, "enabled": new_enabled})


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

    # Handle rename: remove old entry
    rename_from = body.get("rename_from")
    if rename_from and rename_from != name:
        models = data.get("models", {})
        if rename_from not in models:
            return JSONResponse(
                {"error": f"Original model '{rename_from}' not found"},
                status_code=404,
            )
        if name in models:
            return JSONResponse(
                {"error": f"Model '{name}' already exists"},
                status_code=409,
            )
        del models[rename_from]

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


async def get_provider_key(request: Request) -> Response:
    """Return the raw (unmasked) API key for a single provider."""
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


# ---------------------------------------------------------------------------
# Network diagnostics
# ---------------------------------------------------------------------------


async def network_diagnostics(request: Request) -> Response:
    """Run basic network diagnostics: IP geolocation and Google connectivity.

    Uses the gateway's configured global proxy (if any) so the diagnostics
    reflect the actual outbound path of API requests.
    """
    import httpx

    # Resolve the global proxy from current gateway config
    gw_config: GatewayConfig | None = getattr(request.app.state, "gateway_config", None)
    proxy_url = gw_config.proxy if gw_config else None

    client_kwargs: dict[str, Any] = {"timeout": 15}
    if proxy_url:
        client_kwargs["proxy"] = proxy_url

    results: dict[str, Any] = {}
    if proxy_url:
        results["proxy"] = proxy_url

    # IP geolocation via ip-api.com (no key required, JSON by default)
    try:
        async with httpx.AsyncClient(**client_kwargs) as client:
            resp = await client.get(
                "http://ip-api.com/json/?fields=query,country,city,isp"
            )
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
        async with httpx.AsyncClient(**client_kwargs) as client:
            resp = await client.get("https://www.google.com/generate_204")
            results["google"] = {
                "ok": resp.status_code == 204,
                "status": resp.status_code,
            }
    except Exception as exc:
        results["google"] = {"ok": False, "error": str(exc)}

    return JSONResponse(results)


# ---------------------------------------------------------------------------
# API Key management
# ---------------------------------------------------------------------------


async def get_api_keys(request: Request) -> Response:
    """List all gateway API keys (values masked)."""
    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    try:
        data = load_config_raw(config_path)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read config: {exc}"}, status_code=500)

    server = data.get("server", {})
    keys = list(server.get("api_keys", []))
    # Backward compat: expose legacy single key as a synthetic entry
    if not keys and server.get("api_key"):
        keys = [
            {
                "id": "default",
                "key": server["api_key"],
                "label": "default",
                "created": "",
            }
        ]

    masked = [{**entry, "key": _mask_api_key(entry.get("key", ""))} for entry in keys]
    return JSONResponse({"keys": masked})


async def create_api_key(request: Request) -> Response:
    """Create a new gateway API key."""
    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    try:
        body = await request.json()
    except Exception:
        body = {}

    label = body.get("label", "")
    manual_key = body.get("key")
    key_value = manual_key if manual_key else f"rsk-{secrets.token_hex(16)}"

    entry = {
        "id": uuid.uuid4().hex[:8],
        "key": key_value,
        "label": label,
        "created": datetime.now(timezone.utc).isoformat(),
    }

    try:
        data = load_config_raw(config_path)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read config: {exc}"}, status_code=500)

    server = data.setdefault("server", {})

    # Migrate legacy single key → api_keys array
    if "api_key" in server and "api_keys" not in server:
        old_key = server.pop("api_key")
        server["api_keys"] = [
            {"id": "default", "key": old_key, "label": "default", "created": ""}
        ]

    server.setdefault("api_keys", []).append(entry)

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

    # Return the full key exactly once so the user can copy it
    return JSONResponse({"ok": True, "key": entry})


async def update_api_key(request: Request) -> Response:
    """Update an API key's label."""
    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    key_id = request.path_params["key_id"]

    try:
        body = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    try:
        data = load_config_raw(config_path)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read config: {exc}"}, status_code=500)

    keys = data.get("server", {}).get("api_keys", [])
    target = None
    for entry in keys:
        if entry.get("id") == key_id:
            target = entry
            break

    if target is None:
        return JSONResponse({"error": f"Key '{key_id}' not found"}, status_code=404)

    if "label" in body:
        target["label"] = body["label"]

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

    return JSONResponse({"ok": True, "id": key_id, "label": target["label"]})


async def delete_api_key(request: Request) -> Response:
    """Delete a gateway API key."""
    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    key_id = request.path_params["key_id"]

    try:
        data = load_config_raw(config_path)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read config: {exc}"}, status_code=500)

    keys = data.get("server", {}).get("api_keys", [])
    original_len = len(keys)
    keys[:] = [e for e in keys if e.get("id") != key_id]

    if len(keys) == original_len:
        return JSONResponse({"error": f"Key '{key_id}' not found"}, status_code=404)

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

    return JSONResponse({"ok": True, "deleted": key_id})


async def reveal_api_key(request: Request) -> Response:
    """Return the raw (unmasked) API key value."""
    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    key_id = request.path_params["key_id"]

    try:
        data = load_config_raw(config_path)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read config: {exc}"}, status_code=500)

    keys = data.get("server", {}).get("api_keys", [])
    for entry in keys:
        if entry.get("id") == key_id:
            return JSONResponse({"key": entry.get("key", "")})

    return JSONResponse({"error": f"Key '{key_id}' not found"}, status_code=404)


async def get_internal_token(request: Request) -> Response:
    """Return the ephemeral internal token for admin panel test requests."""
    token = getattr(request.app.state, "internal_token", None)
    if not token:
        return JSONResponse({"error": "No internal token available"}, status_code=500)
    return JSONResponse({"token": token})


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
        "/admin/api/config/providers/{name}/toggle",
        toggle_provider,
        methods=["POST"],
    ),
    Route(
        "/admin/api/config/providers/{name}/key",
        get_provider_key,
        methods=["GET"],
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
    # Network diagnostics
    Route("/admin/api/diagnostics/network", network_diagnostics, methods=["GET"]),
    # API key management
    Route("/admin/api/keys", get_api_keys, methods=["GET"]),
    Route("/admin/api/keys", create_api_key, methods=["POST"]),
    Route("/admin/api/keys/{key_id}", update_api_key, methods=["PUT"]),
    Route("/admin/api/keys/{key_id}", delete_api_key, methods=["DELETE"]),
    Route("/admin/api/keys/{key_id}/reveal", reveal_api_key, methods=["GET"]),
    Route("/admin/api/internal-token", get_internal_token, methods=["GET"]),
]
