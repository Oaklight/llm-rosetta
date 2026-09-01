"""Shared helpers used by multiple admin route modules."""

from __future__ import annotations

import re
from typing import Any, overload

from llm_rosetta._vendor.httpserver import JSONResponse, Response

from ...config import ConfigIO, GatewayConfig

_ENV_VAR_RE = re.compile(r"^\$\{.+\}$")


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


def _get_config_path(request: Any) -> str:
    """Return the config file path stored on the app object."""
    path = getattr(request.app, "config_path", None)
    if path is None:
        raise RuntimeError("No config file path available")
    return path


def _get_config_io(request: Any) -> ConfigIO:
    """Return the :class:`ConfigIO` adapter stored on the app object."""
    io = getattr(request.app, "config_io", None)
    if io is None:
        raise RuntimeError("No ConfigIO adapter configured on this application")
    return io


def _reload_gateway_config(request: Any, config_path: str) -> GatewayConfig:
    """Re-read config from disk, rebuild GatewayConfig, swap into app state."""
    import llm_rosetta.gateway.app as _app_mod

    config_io = _get_config_io(request)
    raw = config_io.load(config_path)
    new_config = GatewayConfig(raw)
    _app_mod._config = new_config
    request.app.gateway_config = new_config

    _sync_auth_middleware(request.app, new_config)

    # Hot-reload log level
    from llm_rosetta.gateway.logging import setup_logging as _setup_logging

    _setup_logging(
        verbose=new_config.verbose,
        log_bodies=new_config.log_bodies,
        log_format=new_config.log_format,
    )

    # Hot-reload rate limiting (only rebuild if config actually changed)
    rate_limit_state = getattr(request.app, "rate_limit_state", None)
    if rate_limit_state is not None:
        old_config = getattr(request.app, "_prev_gateway_config", None)
        rl_changed = old_config is None or any(
            getattr(new_config, a, None) != getattr(old_config, a, None)
            for a in (
                "rate_limit_enabled",
                "rate_limit_algorithm",
                "rate_limit_global",
                "rate_limit_per_ip",
                "rate_limit_per_key",
                "rate_limit_per_model",
                "rate_limit_exclude",
                "rate_limit_trust_proxy",
            )
        )
        if rl_changed:
            rate_limit_state.rebuild(new_config)
    request.app._prev_gateway_config = new_config

    # Hot-reload log retention caps
    persistence = getattr(request.app, "persistence", None)
    if persistence is not None:
        rl_cfg = new_config.request_log or {}
        if "success_max" in rl_cfg:
            persistence.success_max = int(rl_cfg["success_max"])
        if "error_max" in rl_cfg:
            persistence.error_max = int(rl_cfg["error_max"])

    return new_config


def _sync_auth_middleware(app: Any, config: GatewayConfig) -> None:
    """Update the auth hook's state for hot-reload."""
    auth_state = getattr(app, "auth_state", None)
    if auth_state is not None:
        from ...auth import _build_config_fallback

        auth_state.config_fallback = _build_config_fallback(config.api_keys)
        # Sync admin password (e.g. changed via CLI or config edit)
        if config.admin_password != auth_state.admin_password:
            auth_state.change_password(config.admin_password or "")


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

    for tpl_key in ("url_template", "stream_url_template"):
        tpl_val = body.get(tpl_key, "")
        if tpl_val:
            entry[tpl_key] = tpl_val

    if resolve_name in existing_providers:
        existing_enabled = existing_providers[resolve_name].get("enabled")
        if existing_enabled is not None:
            entry["enabled"] = existing_enabled

    for flag in (
        "supports_custom_tools",
        "hoist_system_messages",
        "preflight_token_count",
    ):
        if flag in body:
            entry[flag] = bool(body[flag])
    if "timeout" in body and body["timeout"] not in (None, ""):
        entry["timeout"] = float(body["timeout"])

    # Optional provider-level fields: set when truthy, clear when
    # explicitly sent as empty (so the admin UI can remove them).
    for opt_key in (
        "models_path",
        "logo",
        "embedding_format",
        "embedding_path",
        "rerank_format",
        "rerank_path",
    ):
        val = body.get(opt_key)
        if val:
            entry[opt_key] = val
        elif opt_key in body:
            entry.pop(opt_key, None)

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
