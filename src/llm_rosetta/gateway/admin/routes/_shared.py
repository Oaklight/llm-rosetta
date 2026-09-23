"""Shared helpers used by multiple admin route modules."""

from __future__ import annotations

import logging
import re
from contextlib import contextmanager
from typing import Any, overload
from collections.abc import Generator

from llm_rosetta._vendor.httpserver import JSONResponse, Response

from ...config import ConfigIO, GatewayConfig, config_lock

logger = logging.getLogger("llm-rosetta-gateway")

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


def parse_json_body(request: Any) -> tuple[dict[str, Any], Response | None]:
    """Parse the JSON body from a request.

    Returns:
        A ``(body, None)`` tuple on success, or ``({}, error_response)``
        on failure (invalid JSON).
    """
    try:
        body = request.json()
    except Exception:
        return {}, JSONResponse({"error": "Invalid JSON body"}, status_code=400)
    return body, None


class ConfigMutationContext:
    """Holds state for a :func:`config_mutate` session.

    Attributes:
        data: The raw config dict loaded from disk.  Mutate in place.
        error: Set by the context manager if load/save/reload fails.
            Callers should check this after the ``with`` block.
        new_config: The reloaded :class:`GatewayConfig` after a
            successful commit.  ``None`` until commit + reload completes.
    """

    __slots__ = ("data", "error", "new_config", "_committed")

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.error: Response | None = None
        self.new_config: GatewayConfig | None = None
        self._committed: bool = False

    def commit(self) -> None:
        """Mark the mutation for save + reload on context exit."""
        self._committed = True


@contextmanager
def config_mutate(
    request: Any,
) -> Generator[ConfigMutationContext, None, None]:
    """Context manager for the lock → load → mutate → save → reload cycle.

    Usage::

        with config_mutate(request) as ctx:
            if ctx.error:
                return ctx.error
            # ... validate and mutate ctx.data ...
            if bad:
                return JSONResponse(...)  # exits without saving
            ctx.commit()
        if ctx.error:
            return ctx.error
        # ctx.new_config is now available

    The save and reload only happen when :meth:`ConfigMutationContext.commit`
    has been called **and** no exception was raised.  Early returns from the
    handler (e.g. validation errors) skip the save automatically.
    """
    ctx = ConfigMutationContext()
    config_path = _get_config_path(request)

    with config_lock(config_path):
        try:
            ctx.data = _get_config_io(request).load_raw(config_path)
        except Exception as exc:
            ctx.error = JSONResponse(
                {"error": f"Failed to read config: {exc}"}, status_code=500
            )
            yield ctx
            return

        yield ctx

        if not ctx._committed:
            return

        try:
            _get_config_io(request).save(config_path, ctx.data)
        except Exception as exc:
            ctx.error = JSONResponse(
                {"error": f"Failed to write config: {exc}"}, status_code=500
            )
            return

    # Reload happens outside the lock
    try:
        ctx.new_config = _reload_gateway_config(request, config_path)
    except Exception as exc:
        ctx.error = JSONResponse(
            {
                "error": f"Config saved but reload failed: {exc}",
                "saved": True,
                "reloaded": False,
            },
            status_code=500,
        )


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


def _optional_provider_keys() -> list[str]:
    """Return provider config keys that should be set-or-cleared.

    Derives format/path keys from the model type registry so that new
    types are picked up automatically without hardcoding.
    """
    from ...model_types import all_model_types

    keys: list[str] = ["models_path", "logo"]
    for desc in all_model_types():
        if desc.is_llm:
            continue
        if desc.config_format_key:
            keys.append(desc.config_format_key)
        if desc.config_path_key:
            keys.append(desc.config_path_key)
    return keys


def _apply_optional_provider_fields(
    entry: dict[str, Any], body: dict[str, Any]
) -> None:
    """Set or clear optional provider-level fields on *entry*.

    Truthy values in *body* are stored; keys explicitly sent as empty
    are removed so the admin UI can clear them.
    """
    for opt_key in _optional_provider_keys():
        val = body.get(opt_key)
        if val:
            entry[opt_key] = val
        elif opt_key in body:
            entry.pop(opt_key, None)


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
    _apply_optional_provider_fields(entry, body)

    return entry


def _resolve_models_path(provider_cfg: dict, config: Any, name: str) -> str | None:
    """Return explicit models_path from provider config or shim, if any."""
    path = provider_cfg.get("models_path")
    if not path:
        from llm_rosetta.shims import get_shim

        shim_name = (
            config.provider_shim_names.get(name)
            if hasattr(config, "provider_shim_names")
            else None
        )
        shim = get_shim(shim_name) if shim_name else None
        path = shim.connection.models_path if shim else None
    if path and path.startswith("http://"):
        logger.warning(
            "models_path for %s uses plain HTTP — auth headers will be sent "
            "over an unencrypted connection",
            name,
        )
    return path


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
