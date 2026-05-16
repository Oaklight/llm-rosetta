"""Route handlers for the admin panel API."""

from __future__ import annotations

import asyncio
import logging
import re
import secrets
import time
import uuid
from datetime import datetime, timezone
from typing import Any, overload

from llm_rosetta._vendor.httpclient import AsyncClient, Response as HttpResponse
from llm_rosetta._vendor.httpserver import JSONResponse, Response

from llm_rosetta.shims import list_shims

from ..config import GatewayConfig, load_config, load_config_raw, write_config
from ..providers import known_provider_types
from .static import load_admin_html

logger = logging.getLogger("llm-rosetta-gateway")

# Cached HTML — loaded once on first request.
_admin_html: str | None = None

_ENV_VAR_RE = re.compile(r"^\$\{.+\}$")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@overload
def _qp(request: Any, key: str) -> str | None: ...


@overload
def _qp(request: Any, key: str, default: str) -> str: ...


def _qp(request: Any, key: str, default: str | None = None) -> str | None:
    """Extract a single query param value (httpserver convenience)."""
    vals = request.query_params.get(key)
    if vals:
        return vals[0]
    return default


def _mask_api_key(value: str) -> str:
    """Mask a literal API key, leaving env-var placeholders intact."""
    if _ENV_VAR_RE.match(value):
        return value
    if len(value) <= 8:
        return "***"
    return value[:4] + "***" + value[-4:]


def _get_config_path(request: Any) -> str | None:
    return getattr(request.app, "config_path", None)


def _reload_gateway_config(request: Any, config_path: str) -> GatewayConfig:
    """Re-read config from disk, rebuild GatewayConfig, swap into app state."""
    import llm_rosetta.gateway.app as _app_mod

    raw = load_config(config_path)
    new_config = GatewayConfig(raw)
    _app_mod._config = new_config
    request.app.gateway_config = new_config

    _sync_auth_middleware(request.app, new_config)

    return new_config


def _sync_auth_middleware(app: Any, config: GatewayConfig) -> None:
    """Update the auth hook's state for hot-reload."""
    auth_state = getattr(app, "auth_state", None)
    if auth_state is not None:
        auth_state.key_set = config.api_key_set
        auth_state.labels = dict(config.api_key_labels)


def _build_provider_entry(
    body: dict[str, Any],
    api_key: str,
    base_url: str,
    existing_providers: dict[str, Any],
    resolve_name: str,
) -> dict[str, Any]:
    """Build a provider entry dict from request body, resolving masked keys."""
    if "***" in api_key and resolve_name in existing_providers:
        api_key = existing_providers[resolve_name].get("api_key", api_key)

    entry: dict[str, Any] = {"api_key": api_key, "base_url": base_url}

    provider_type = body.get("type")
    if provider_type:
        entry["type"] = provider_type

    if "proxy" in body:
        proxy = body["proxy"]
        if proxy:
            entry["proxy"] = proxy

    if resolve_name in existing_providers:
        existing_enabled = existing_providers[resolve_name].get("enabled")
        if existing_enabled is not None:
            entry["enabled"] = existing_enabled

    return entry


def _handle_provider_rename(
    data: dict[str, Any], rename_from: str, name: str
) -> Response | None:
    """Handle provider rename: remove old entry, update model refs."""
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
    models = data.get("models", {})
    for model_name, model_val in models.items():
        if isinstance(model_val, str) and model_val == rename_from:
            models[model_name] = name
        elif isinstance(model_val, dict) and model_val.get("provider") == rename_from:
            model_val["provider"] = name
    return None


# ---------------------------------------------------------------------------
# HTML handler
# ---------------------------------------------------------------------------


async def serve_admin_html(request: Any) -> Response:
    """Serve the admin panel SPA."""
    global _admin_html
    if _admin_html is None:
        _admin_html = load_admin_html()
    return Response(
        body=_admin_html,
        status_code=200,
        content_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# ---------------------------------------------------------------------------
# Admin authentication — login rate limiter
# ---------------------------------------------------------------------------

# Per-IP failure tracking: {ip: {"count": int, "locked_until": float}}
_login_failures: dict[str, dict[str, Any]] = {}
_LOGIN_MAX_ATTEMPTS = 5  # failures before lockout
_LOGIN_LOCKOUT_SECONDS = 300  # 5-minute lockout window


def _get_client_ip(request: Any) -> str:
    """Extract client IP, honouring X-Forwarded-For when present."""
    xff = request.headers.get("x-forwarded-for", "")
    if xff:
        return xff.split(",")[0].strip()
    addr = getattr(request, "client_addr", None)
    if addr:
        return str(addr[0])
    return "unknown"


def _check_login_rate_limit(ip: str) -> tuple[bool, float]:
    """Return (is_blocked, retry_after_seconds).

    An IP is blocked for ``_LOGIN_LOCKOUT_SECONDS`` after
    ``_LOGIN_MAX_ATTEMPTS`` consecutive failures.
    """
    rec = _login_failures.get(ip)
    if not rec:
        return False, 0.0
    locked_until = rec.get("locked_until", 0.0)
    if locked_until and time.monotonic() < locked_until:
        return True, locked_until - time.monotonic()
    return False, 0.0


def _record_login_failure(ip: str) -> None:
    """Increment failure counter; lock out the IP after max attempts."""
    rec = _login_failures.setdefault(ip, {"count": 0, "locked_until": 0.0})
    # Reset counter if a previous lockout has expired
    if rec["locked_until"] and time.monotonic() >= rec["locked_until"]:
        rec["count"] = 0
        rec["locked_until"] = 0.0
    rec["count"] += 1
    if rec["count"] >= _LOGIN_MAX_ATTEMPTS:
        rec["locked_until"] = time.monotonic() + _LOGIN_LOCKOUT_SECONDS


def _clear_login_failures(ip: str) -> None:
    """Reset failure counter on successful login."""
    _login_failures.pop(ip, None)


async def admin_login(request: Any) -> Response:
    """Validate admin password and return a session token."""
    auth_state = request.app.auth_state
    if not auth_state.admin_password:
        return JSONResponse({"error": "Admin password not configured"}, status_code=400)

    ip = _get_client_ip(request)
    blocked, retry_after = _check_login_rate_limit(ip)
    if blocked:
        return JSONResponse(
            {
                "error": f"Too many failed attempts. Try again in {int(retry_after) + 1}s."
            },
            status_code=429,
        )

    try:
        body = request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    password = body.get("password", "")
    if password != auth_state.admin_password:
        _record_login_failure(ip)
        blocked, retry_after = _check_login_rate_limit(ip)
        resp: dict[str, Any] = {"error": "Invalid password"}
        if blocked:
            resp["error"] = (
                f"Too many failed attempts. Locked for {int(retry_after) + 1}s."
            )
        return JSONResponse(resp, status_code=401)

    _clear_login_failures(ip)
    return JSONResponse({"ok": True, "token": auth_state.admin_token})


async def admin_check(request: Any) -> Response:
    """Check whether admin auth is required (before loading config)."""
    auth_state = request.app.auth_state
    requires_auth = bool(auth_state.admin_password)
    return JSONResponse({"requires_auth": requires_auth})


# ---------------------------------------------------------------------------
# Config API
# ---------------------------------------------------------------------------


async def get_config(request: Any) -> Response:
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
            entry = {
                "provider": value.get("provider", ""),
                "capabilities": value.get("capabilities", ["text"]),
            }
            if value.get("upstream_model"):
                entry["upstream_model"] = value["upstream_model"]
            models_normalized[name] = entry

    # Mask api_keys in server section for the response
    server = dict(raw.get("server", {}))
    if "api_key" in server:
        server["api_key"] = _mask_api_key(server["api_key"])
    if "api_keys" in server:
        server["api_keys"] = [
            {**entry, "key": _mask_api_key(entry.get("key", ""))}
            for entry in server["api_keys"]
        ]

    config: GatewayConfig = request.app.gateway_config
    return JSONResponse(
        {
            "config_path": config_path,
            "providers": masked_providers,
            "models": models_normalized,
            "server": server,
            "credential_visible": config.credential_visible,
            "known_provider_types": known_provider_types(),
            "registered_shims": [
                {
                    "name": s.name,
                    "base": s.base,
                    "logo": s.logo,
                    "default_base_url": s.default_base_url,
                    "default_api_key_env": s.default_api_key_env,
                }
                for s in list_shims()
            ],
        }
    )


async def put_provider(request: Any, **kwargs: Any) -> Response:
    """Add or update a provider entry."""
    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    name = request.path_params["name"]

    try:
        body = request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    api_key = body.get("api_key", "")
    base_url = body.get("base_url", "")

    try:
        data = load_config_raw(config_path)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read config: {exc}"}, status_code=500)

    existing_providers = data.get("providers", {})
    resolve_name = body.get("rename_from", name) or name

    # When api_key is omitted/empty and we're editing, keep the existing key
    if not api_key and resolve_name in existing_providers:
        api_key = existing_providers[resolve_name].get("api_key", "")

    if not api_key or not base_url:
        return JSONResponse(
            {"error": "Both 'api_key' and 'base_url' are required"}, status_code=400
        )

    provider_entry = _build_provider_entry(
        body, api_key, base_url, existing_providers, resolve_name
    )

    # Handle rename: remove old entry and update model references
    rename_from = body.get("rename_from")
    if rename_from and rename_from != name:
        rename_err = _handle_provider_rename(data, rename_from, name)
        if rename_err is not None:
            return rename_err

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


async def delete_provider(request: Any, **kwargs: Any) -> Response:
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

    cascade = _qp(request, "cascade") in ("true", "1")
    if referencing and not cascade:
        return JSONResponse(
            {
                "error": f"Cannot delete provider '{name}': referenced by models: {referencing}"
            },
            status_code=409,
        )

    # Cascade: remove referencing models first
    cascade_deleted: list[str] = []
    if referencing and cascade:
        for model_name in referencing:
            del models[model_name]
            cascade_deleted.append(model_name)

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

    result: dict[str, Any] = {
        "ok": True,
        "deleted": name,
        "providers": list(new_config.providers.keys()),
    }
    if cascade_deleted:
        result["cascade_deleted_models"] = cascade_deleted
    return JSONResponse(result)


async def toggle_provider(request: Any, **kwargs: Any) -> Response:
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


async def put_model(request: Any, **kwargs: Any) -> Response:
    """Add or update a model routing entry."""
    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    name = request.path_params["name"]

    try:
        body = request.json()
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

    model_entry: dict[str, Any] = {
        "provider": provider,
        "capabilities": capabilities,
    }
    upstream_model = body.get("upstream_model")
    if upstream_model:
        model_entry["upstream_model"] = upstream_model
    data.setdefault("models", {})[name] = model_entry

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


async def delete_model(request: Any, **kwargs: Any) -> Response:
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


async def put_server_settings(request: Any) -> Response:
    """Update server settings (e.g. global proxy)."""
    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    try:
        body = request.json()
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


async def reload_config(request: Any) -> Response:
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


async def get_metrics(request: Any) -> Response:
    """Return a full metrics snapshot."""
    metrics = request.app.metrics
    seconds = int(_qp(request, "seconds", "60"))
    seconds = max(1, min(seconds, 300))
    return JSONResponse(metrics.snapshot(series_seconds=seconds))


# ---------------------------------------------------------------------------
# Request log API
# ---------------------------------------------------------------------------


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


async def network_diagnostics(request: Any) -> Response:
    """Run basic network diagnostics: IP geolocation and Google connectivity.

    Uses the gateway's configured global proxy (if any) so the diagnostics
    reflect the actual outbound path of API requests.
    """
    from llm_rosetta._vendor.httpclient import (
        AsyncClient as HttpClient,
        Response as HttpResponse,
    )

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
        async with HttpClient(**client_kwargs) as client:
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
        async with HttpClient(**client_kwargs) as client:
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


async def get_api_keys(request: Any) -> Response:
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


async def create_api_key(request: Any) -> Response:
    """Create a new gateway API key."""
    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    try:
        body = request.json()
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


async def update_api_key(request: Any, **kwargs: Any) -> Response:
    """Update an API key's label."""
    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    key_id = request.path_params["key_id"]

    try:
        body = request.json()
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


async def delete_api_key(request: Any, **kwargs: Any) -> Response:
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


async def reveal_api_key(request: Any, **kwargs: Any) -> Response:
    """Return the raw (unmasked) API key value."""
    config: GatewayConfig = request.app.gateway_config
    if not config.credential_visible:
        return JSONResponse(
            {"error": "Credential visibility is disabled"}, status_code=403
        )
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


async def get_internal_token(request: Any) -> Response:
    """Return the ephemeral internal token for admin panel test requests."""
    token = getattr(request.app, "internal_token", None)
    if not token:
        return JSONResponse({"error": "No internal token available"}, status_code=500)
    return JSONResponse({"token": token})


# ---------------------------------------------------------------------------
# Async model test tasks
# ---------------------------------------------------------------------------

# In-memory store: task_id → {status, result, asyncio_task, started, ...}
_test_tasks: dict[str, dict[str, Any]] = {}
_MAX_TASK_AGE = 300  # seconds — auto-cleanup threshold


def _cleanup_stale_tasks() -> None:
    """Remove tasks older than _MAX_TASK_AGE."""
    now = time.monotonic()
    expired = [
        tid
        for tid, t in _test_tasks.items()
        if now - t.get("started", 0) > _MAX_TASK_AGE
    ]
    for tid in expired:
        task = _test_tasks.pop(tid, None)
        if task and not task.get("_task_obj", asyncio.Future()).done():
            task["_task_obj"].cancel()


async def _run_test_task(
    task_id: str,
    endpoint: str,
    payload: dict[str, Any],
    internal_token: str,
    proxy_url: str | None,  # noqa: ARG001 — reserved for future use
) -> None:
    """Execute an upstream test request via a self-call to the gateway.

    Results are stored in ``_test_tasks[task_id]``.  This runs as an
    ``asyncio.Task`` so the admin API handler can return immediately.

    Important: we create a **dedicated** ``AsyncClient`` per task instead
    of reusing the shared pool from ``proxy.get_client()``.  The shared
    client serialises non-streaming requests with an ``asyncio.Lock``,
    which would deadlock when the self-call triggers the proxy handler
    that itself needs the same lock to call the upstream provider.
    """
    try:
        # Use a per-task client to avoid lock contention / deadlock
        # with the shared proxy client.
        client = AsyncClient(timeout=300.0)
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {internal_token}",
            }

            base_url = _test_tasks[task_id].get("_base_url", "http://127.0.0.1:28765")
            url = f"{base_url}{endpoint}"

            resp = await client.post(url, json=payload, headers=headers)
            assert isinstance(resp, HttpResponse)

            # Try to parse JSON body
            try:
                body = resp.json()
            except Exception:
                body = resp.text

            _test_tasks[task_id].update(
                {
                    "status": "done",
                    "status_code": resp.status_code,
                    "body": body,
                }
            )
        finally:
            await client.aclose()
    except asyncio.CancelledError:
        _test_tasks[task_id]["status"] = "cancelled"
    except Exception as exc:
        _test_tasks[task_id].update(
            {
                "status": "error",
                "error": str(exc),
            }
        )


async def start_test(request: Any) -> Response:
    """Start an async model test.  Returns a task_id immediately.

    POST /admin/api/test
    Body: {endpoint: "/v1/...", payload: {...}}
    """
    _cleanup_stale_tasks()

    try:
        body = request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    endpoint = body.get("endpoint")
    payload = body.get("payload")
    if not endpoint or not isinstance(payload, dict):
        return JSONResponse({"error": "Missing endpoint or payload"}, status_code=400)

    internal_token = getattr(request.app, "internal_token", "")
    if not internal_token:
        return JSONResponse({"error": "No internal token available"}, status_code=500)

    # Determine the base URL the gateway is listening on so the test
    # task can POST back to ourselves.
    host = getattr(request.app, "_bind_host", "127.0.0.1")
    port = getattr(request.app, "_bind_port", 28765)
    # Always use 127.0.0.1 for self-calls — even if bound to 0.0.0.0
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    base_url = f"http://{host}:{port}"

    task_id = uuid.uuid4().hex[:12]
    _test_tasks[task_id] = {
        "status": "pending",
        "started": time.monotonic(),
        "_base_url": base_url,
    }

    # Determine proxy from gateway config
    gw_config = _get_gateway_config(request)
    proxy_url = gw_config.proxy if gw_config else None

    asyncio_task = asyncio.create_task(
        _run_test_task(task_id, endpoint, payload, internal_token, proxy_url)
    )
    _test_tasks[task_id]["_task_obj"] = asyncio_task

    return JSONResponse({"task_id": task_id})


async def get_test_result(request: Any, task_id: str = "") -> Response:
    """Poll for a test task result.

    GET /admin/api/test/<task_id>
    """
    task = _test_tasks.get(task_id)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)

    # Return only serialisable fields (skip _task_obj, _base_url)
    result: dict[str, Any] = {k: v for k, v in task.items() if not k.startswith("_")}
    return JSONResponse(result)


async def cancel_test(request: Any, task_id: str = "") -> Response:
    """Cancel a running test task.

    DELETE /admin/api/test/<task_id>
    """
    task = _test_tasks.get(task_id)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)

    task_obj = task.get("_task_obj")
    if task_obj and not task_obj.done():
        task_obj.cancel()
        task["status"] = "cancelled"

    return JSONResponse({"ok": True})


# ---------------------------------------------------------------------------
# Fetch upstream models
# ---------------------------------------------------------------------------


def _get_gateway_config(request: Any) -> GatewayConfig | None:
    """Return the live GatewayConfig from the app module."""
    import llm_rosetta.gateway.app as _app_mod

    return _app_mod._config


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

    # Build the models listing URL based on provider type
    if ptype == "google":
        models_url = f"{pinfo.base_url}/v1beta/models"
    elif ptype == "anthropic":
        models_url = f"{pinfo.base_url}/v1/models"
    else:
        # OpenAI-compatible (openai_chat, openai_responses, etc.)
        models_url = f"{pinfo.base_url}/models"

    headers = pinfo.auth_headers()

    try:
        client = AsyncClient(timeout=30.0, proxy=pinfo.proxy_url)
        raw_resp = await client.get(models_url, headers=headers)
        assert isinstance(raw_resp, HttpResponse), "Expected non-streaming response"
        resp: HttpResponse = raw_resp
        await client.aclose()
    except Exception as exc:
        logger.warning("Failed to fetch models from %s: %s", provider_name, exc)
        return JSONResponse(
            {"error": f"Failed to connect to upstream: {exc}"}, status_code=502
        )

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
            status_code=502,
        )

    try:
        body = resp.json()
    except Exception:
        return JSONResponse(
            {"error": "Upstream returned non-JSON response"},
            status_code=502,
        )

    # Resolve model_id_field from shim (e.g. Argo uses "internal_id")
    from llm_rosetta.shims import get_shim

    shim_name = config.provider_shim_names.get(provider_name)
    shim = get_shim(shim_name) if shim_name else None
    id_field = shim.model_id_field if shim and shim.model_id_field else None

    # Normalize response — different providers return different formats
    model_ids: list[str] = []
    if ptype == "google":
        # Google: {"models": [{"name": "models/gemini-...", ...}]}
        for m in body.get("models", []):
            name = m.get("name", "")
            if name.startswith("models/"):
                name = name[len("models/") :]
            if id_field:
                name = m.get(id_field, name)
            model_ids.append(name)
    elif ptype == "anthropic":
        # Anthropic: {"data": [{"id": "claude-...", ...}]}
        for m in body.get("data", []):
            model_ids.append(
                m.get(id_field, m.get("id", "")) if id_field else m.get("id", "")
            )
    else:
        # OpenAI-compatible: {"data": [{"id": "gpt-...", ...}]}
        for m in body.get("data", []):
            model_ids.append(
                m.get(id_field, m.get("id", "")) if id_field else m.get("id", "")
            )

    model_ids = [m for m in model_ids if m]
    model_ids.sort()

    return JSONResponse(
        {
            "provider": provider_name,
            "api_standard": ptype,
            "models": model_ids,
        }
    )


async def bulk_add_models(request: Any) -> Response:
    """Bulk-add multiple models for a given provider."""
    config_path = _get_config_path(request)
    if not config_path:
        return JSONResponse({"error": "No config file path available"}, status_code=500)

    try:
        body = request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    provider = body.get("provider")
    models_to_add: list[str] = body.get("models", [])
    prefix = body.get("prefix", "")
    capabilities = body.get("capabilities", ["text", "vision", "tools"])

    if not provider or not models_to_add:
        return JSONResponse(
            {"error": "'provider' and 'models' are required"}, status_code=400
        )

    try:
        data = load_config_raw(config_path)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read config: {exc}"}, status_code=500)

    # Validate provider exists
    providers = data.get("providers", {})
    if provider not in providers:
        return JSONResponse(
            {"error": f"Provider '{provider}' not found"}, status_code=400
        )

    models_section = data.setdefault("models", {})
    added: list[str] = []
    skipped: list[str] = []

    for model_id in models_to_add:
        display_name = f"{prefix}{model_id}" if prefix else model_id
        if display_name in models_section:
            skipped.append(display_name)
            continue
        entry: dict[str, Any] = {
            "provider": provider,
            "capabilities": capabilities,
        }
        # When a prefix is used, the gateway name differs from the
        # upstream model id — store the original as upstream_model so the
        # proxy handler can substitute it before forwarding.
        if prefix:
            entry["upstream_model"] = model_id
        models_section[display_name] = entry
        added.append(display_name)

    if not added:
        return JSONResponse(
            {
                "ok": True,
                "added": [],
                "skipped": skipped,
                "message": "All models already exist",
            }
        )

    try:
        write_config(config_path, data)
    except Exception as exc:
        return JSONResponse(
            {"error": f"Failed to write config: {exc}"}, status_code=500
        )

    new_config = _reload_gateway_config(request, config_path)

    return JSONResponse(
        {
            "ok": True,
            "added": added,
            "skipped": skipped,
            "models": dict(new_config.models),
        }
    )


# ---------------------------------------------------------------------------
# Route registration
# ---------------------------------------------------------------------------


def register_admin_routes(app: Any) -> None:
    """Register all admin panel routes on the httpserver App."""
    # HTML
    app.route("/admin", methods=["GET"])(serve_admin_html)
    app.route("/admin/", methods=["GET"])(serve_admin_html)
    # Admin auth
    app.route("/admin/api/login", methods=["POST"])(admin_login)
    app.route("/admin/api/auth-check", methods=["GET"])(admin_check)
    # Config CRUD
    app.route("/admin/api/config", methods=["GET"])(get_config)
    app.route("/admin/api/config/providers/<name>", methods=["PUT"])(put_provider)
    app.route("/admin/api/config/providers/<name>", methods=["DELETE"])(delete_provider)
    app.route("/admin/api/config/providers/<name>/toggle", methods=["POST"])(
        toggle_provider
    )
    app.route("/admin/api/config/providers/<name>/key", methods=["GET"])(
        get_provider_key
    )
    app.route("/admin/api/config/models/<path:name>", methods=["PUT"])(put_model)
    app.route("/admin/api/config/models/<path:name>", methods=["DELETE"])(delete_model)
    app.route("/admin/api/config/providers/<name>/models", methods=["GET"])(
        fetch_upstream_models
    )
    app.route("/admin/api/config/models", methods=["POST"])(bulk_add_models)
    app.route("/admin/api/config/server", methods=["PUT"])(put_server_settings)
    app.route("/admin/api/config/reload", methods=["POST"])(reload_config)
    # Metrics
    app.route("/admin/api/metrics", methods=["GET"])(get_metrics)
    # Request log
    app.route("/admin/api/requests", methods=["GET"])(get_requests)
    app.route("/admin/api/requests", methods=["DELETE"])(clear_requests)
    # Network diagnostics
    app.route("/admin/api/diagnostics/network", methods=["GET"])(network_diagnostics)
    # API key management
    app.route("/admin/api/keys", methods=["GET"])(get_api_keys)
    app.route("/admin/api/keys", methods=["POST"])(create_api_key)
    app.route("/admin/api/keys/<key_id>", methods=["PUT"])(update_api_key)
    app.route("/admin/api/keys/<key_id>", methods=["DELETE"])(delete_api_key)
    app.route("/admin/api/keys/<key_id>/reveal", methods=["GET"])(reveal_api_key)
    app.route("/admin/api/internal-token", methods=["GET"])(get_internal_token)
    # Async model test
    app.route("/admin/api/test", methods=["POST"])(start_test)
    app.route("/admin/api/test/<task_id>", methods=["GET"])(get_test_result)
    app.route("/admin/api/test/<task_id>", methods=["DELETE"])(cancel_test)
