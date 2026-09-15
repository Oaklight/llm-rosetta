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


def _mask_proxy_url(url: str | None) -> str | None:
    """Mask userinfo credentials in a proxy URL, leaving the rest intact."""
    if not url:
        return url
    if _ENV_VAR_RE.match(url):
        return url
    # Match scheme://user:pass@host...
    m = re.match(r"^(https?://)([^:]+):([^@]+)@(.+)$", url)
    if m:
        return f"{m.group(1)}{m.group(2)}:***@{m.group(4)}"
    return url


def _sanitize_server_section(server: dict[str, Any]) -> dict[str, Any]:
    """Strip or mask sensitive fields from a server config dict."""
    safe = dict(server)
    safe.pop("admin_password", None)
    if "api_key" in safe:
        safe["api_key"] = _mask_api_key(safe["api_key"])
    if "api_keys" in safe:
        safe["api_keys"] = [
            {**entry, "key": _mask_api_key(entry.get("key", ""))}
            for entry in safe["api_keys"]
        ]
    if "proxy" in safe:
        safe["proxy"] = _mask_proxy_url(safe["proxy"])
    return safe


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
        ol_cfg = new_config.ops_log or {}
        if "info_max" in ol_cfg:
            persistence._ops_info_max = max(100, int(ol_cfg["info_max"]))
        if "warn_max" in ol_cfg:
            persistence._ops_warn_max = max(100, int(ol_cfg["warn_max"]))

    # Record config reload event
    ops_log = getattr(request.app, "ops_log", None)
    if ops_log is not None:
        from llm_rosetta.observability.ops_log import (
            EVENT_CONFIG_RELOAD,
            OpsLogEntry,
            SEVERITY_INFO,
            SOURCE_CONFIG,
        )

        ops_log.add(
            OpsLogEntry.create(
                event_type=EVENT_CONFIG_RELOAD,
                severity=SEVERITY_INFO,
                message=(
                    f"Config reloaded ({len(new_config.providers)} providers, "
                    f"{len(new_config.models)} models)"
                ),
                details={
                    "provider_count": len(new_config.providers),
                    "model_count": len(new_config.models),
                },
                source=SOURCE_CONFIG,
            )
        )

    return new_config


def _sync_auth_middleware(app: Any, config: GatewayConfig) -> None:
    """Update the auth hook's state for hot-reload."""
    auth_state = getattr(app, "auth_state", None)
    if auth_state is not None:
        # Sync admin password (e.g. changed via CLI or config edit)
        if config.admin_password != auth_state.admin_password:
            auth_state.change_password(config.admin_password or "")


def _apply_optional_int(entry: dict[str, Any], body: dict[str, Any], key: str) -> None:
    """Set or clear an optional integer field on a provider entry."""
    if key in body:
        val = body[key]
        if val not in (None, ""):
            entry[key] = int(val)
        else:
            entry.pop(key, None)


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
    _apply_optional_int(entry, body, "max_tool_description_length")

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


def _resolve_models_path(provider_cfg: dict, config: Any, name: str) -> str | None:
    """Return explicit models_path from provider config or shim, if any."""
    path = provider_cfg.get("models_path")
    if path:
        return path
    from llm_rosetta.shims import get_shim

    shim_name = (
        config.provider_shim_names.get(name)
        if hasattr(config, "provider_shim_names")
        else None
    )
    shim = get_shim(shim_name) if shim_name else None
    return shim.models_path if shim else None


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
