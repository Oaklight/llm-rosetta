"""Config CRUD and upstream model fetch route handlers."""

from __future__ import annotations

from typing import Any

from llm_rosetta._vendor.httpclient import AsyncClient, Response as HttpResponse
from llm_rosetta._vendor.httpserver import JSONResponse, Response
from llm_rosetta.shims import list_shims

from ...config import GatewayConfig, load_config_raw, write_config
from ...providers import known_provider_types
from ._shared import (
    _build_provider_entry,
    _get_config_path,
    _handle_provider_rename,
    _mask_api_key,
    _reload_gateway_config,
)

import logging

logger = logging.getLogger("llm-rosetta-gateway")


def _get_gateway_config(request: Any) -> GatewayConfig | None:
    """Return the live GatewayConfig from the app module."""
    import llm_rosetta.gateway.app as _app_mod

    return _app_mod._config


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

    from ._shared import _qp

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


async def fetch_upstream_models(request: Any, **kwargs: Any) -> Response:
    """Fetch the model list from an upstream provider's /v1/models endpoint."""
    from llm_rosetta.shims import get_shim

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
