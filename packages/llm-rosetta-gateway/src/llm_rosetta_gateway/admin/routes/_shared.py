"""Shared helpers used by multiple admin route modules."""

from __future__ import annotations

import functools
import logging
import re
from contextlib import asynccontextmanager
from typing import Any, overload
from collections.abc import AsyncGenerator

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response

from ...config import ConfigIO, GatewayConfig, async_config_lock

logger = logging.getLogger("llm-rosetta-gateway")

_ENV_VAR_RE = re.compile(r"^\$\{.+\}$")


@overload
def _qp(request: Any, key: str) -> str | None: ...


@overload
def _qp(request: Any, key: str, default: str) -> str: ...


def _guard(tab_id: str, handler: Any) -> Any:
    """Wrap a route handler to return 404 when its tab is disabled.

    Read-only endpoints (``get_config``, ``get_metrics``,
    ``get_request_key_labels``) are intentionally left unguarded —
    disabled tabs are a UI convenience, not security isolation, and
    other components (e.g. argo-proxy's ``ArgoConfigIO``) may depend
    on them regardless of tab visibility.
    """

    @functools.wraps(handler)
    async def _guarded(request: Any, **kw: Any) -> Any:
        if tab_id in getattr(request.app, "disabled_tabs", ()):
            return JSONResponse(
                {"error": "Tab disabled", "disabled": True}, status_code=404
            )
        return await handler(request, **kw)

    return _guarded


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

    __slots__ = ("data", "error", "new_config", "warnings", "_committed")

    def __init__(self) -> None:
        self.data: dict[str, Any] = {}
        self.error: Response | None = None
        self.new_config: GatewayConfig | None = None
        self.warnings: list[dict[str, Any]] = []
        self._committed: bool = False

    def commit(self) -> None:
        """Mark the mutation for save + reload on context exit."""
        self._committed = True


@asynccontextmanager
async def config_mutate(
    request: Any,
) -> AsyncGenerator[ConfigMutationContext, None]:
    """Async context manager for the lock -> load -> mutate -> save -> reload cycle.

    Usage::

        async with config_mutate(request) as ctx:
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

    async with async_config_lock(config_path):
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
        ctx.new_config = await _reload_gateway_config(request, config_path)
        ctx.warnings = list(getattr(request.app, "provider_validation_warnings", []))
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


async def _reload_gateway_config(
    request: Any, config_path: str, *, force_shims: bool = False
) -> GatewayConfig:
    """Re-read config from disk, rebuild GatewayConfig, swap into app state.

    Declarative shim sources are (re)applied before the config is rebuilt, but
    only when the ``shims`` section changed (or ``force_shims`` is set) so an
    unrelated admin edit never triggers a remote refetch.  Applying them drops
    shims whose source was removed.
    """
    import asyncio

    from llm_rosetta_gateway.app import _state as _app_state
    from ...shim_sources import apply_shim_sources, shims_signature

    config_io = _get_config_io(request)
    raw = config_io.load(config_path)
    # Taken in the same turn as the read, before anything can await: the reader
    # records the registry for the document it just read, and this is the config
    # object built from that document, so it is the one that reports it.  A
    # reader that runs no load-time steps has no such attribute, and reports
    # nothing — the same answer as a config built from a literal.
    maintenance = getattr(config_io, "last_load_registry", None)

    report = getattr(request.app, "shim_source_report", None)
    signature = shims_signature(raw)
    if force_shims or signature != getattr(request.app, "shim_source_signature", None):
        loop = asyncio.get_running_loop()
        report = await loop.run_in_executor(None, apply_shim_sources, raw, config_path)
        request.app.shim_source_report = report
        request.app.shim_source_signature = signature

    from ...ops import OpsContext, OpsProviderValidation

    recovery = OpsProviderValidation(
        getattr(request.app, "ops_ctx", None) or OpsContext(),
        raw,
        report=report if "report" in locals() else None,
        persist=True,
    )
    recovery_result = await recovery.execute()
    request.app.provider_validation_warnings = recovery_result.get("disabled", [])
    new_config = GatewayConfig(raw, maintenance=maintenance, config_path=config_path)
    _app_state._config = new_config
    request.app.gateway_config = new_config

    # A theme saved from the admin panel must reach the next page load.
    from .. import apply_admin_theme

    apply_admin_theme(request.app, getattr(new_config, "admin_custom_css", ""))

    _sync_auth_middleware(request.app, new_config)

    # Hot-reload log level
    from llm_rosetta_gateway.logging import setup_logging as _setup_logging

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
        if "error_dump_max" in rl_cfg:
            persistence.dump_max = int(rl_cfg["error_dump_max"])
        ol_cfg = new_config.ops_log or {}
        if "info_max" in ol_cfg:
            persistence.retention.ops_info_max = max(100, int(ol_cfg["info_max"]))
        if "warn_max" in ol_cfg:
            persistence.retention.ops_warn_max = max(100, int(ol_cfg["warn_max"]))

    # Record config reload event
    ops_log = getattr(request.app, "ops_log", None)
    if ops_log is not None:
        from llm_rosetta_gateway.observability.ops_log import (
            EVENT_CONFIG_RELOAD,
            OpsLogEntry,
            SEVERITY_INFO,
            SOURCE_CONFIG,
        )

        await ops_log.add(
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

    # Re-seed credential-minting providers.  The rebuild above replaced every
    # ProviderInfo, so each of them is back on the pending sentinel with no
    # refresh loop.  Seeding otherwise runs only at process start, so without
    # this the provider never obtains a credential and every request to it is
    # rejected until the process restarts.
    deferred = getattr(request.app, "deferred_startup", None)
    if deferred is not None:
        await deferred.apply_config_reload(new_config)

    return new_config


def _sync_auth_middleware(app: Any, config: GatewayConfig) -> None:
    """Update the auth hook's state for hot-reload."""
    auth_state = getattr(app, "auth_state", None)
    if auth_state is not None:
        # Sync admin password (e.g. changed via CLI or config edit)
        if config.admin_password != auth_state.admin_password:
            auth_state.change_password(config.admin_password or "")
        auth_state.configure_sessions(
            idle_minutes=config.admin_session_idle_minutes,
            remember_days=config.admin_session_remember_days,
        )


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

    keys: list[str] = [
        "models_path",
        "logo",
        "quota",
        # The credential source, for providers that mint it from a command
        # (ALCF, AskSage).  Without these a save rewrote the entry without
        # them and the provider quietly fell back to a static key.
        "token_command",
        "token_refresh_interval",
        # Names the variable the command reads the operator's key from; part of
        # the same credential-source block, and just as invisible to the editor.
        "token_command_env",
    ]
    for desc in all_model_types():
        if desc.is_llm:
            continue
        if desc.config_format_key:
            keys.append(desc.config_format_key)
        if desc.config_path_key:
            keys.append(desc.config_path_key)
        if desc.config_shim_key:
            keys.append(desc.config_shim_key)
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


def _shim_ships_credential_exchange(shim: Any) -> bool:
    """Whether *shim* mints the upstream credential itself in ``auth.py``.

    Such a provider has no key to type into the editor, exactly like a
    ``token_command`` one, so the save path must not demand one.
    """
    return shim is not None and getattr(shim, "auth", None) is not None


def _credential_command(body: dict[str, Any], existing: dict[str, Any] | None) -> Any:
    """Return the command that owns the credential after this save, if any.

    The body wins; an absent key means "keep what is stored", which is how the
    provider editor behaves (it has no field for the command).
    """
    command = body.get("token_command")
    if command is None and existing:
        command = existing.get("token_command")
    return command


def _credential_fields(
    body: dict[str, Any], api_key: str, existing: dict[str, Any] | None
) -> dict[str, Any]:
    """The credential slots of a provider entry.

    A command owns the credential when there is one, so an empty key is not a
    key: storing ``"api_key": ""`` beside ``token_command`` gives the loader two
    credential sources where it accepts one, and the provider comes back
    unusable.  The command itself is not something the editor can express, so
    an absent key in *body* means "keep what is stored"; a caller that wants to
    clear it sends the key with an empty value.
    """
    fields: dict[str, Any] = {}
    if api_key or not _credential_command(body, existing):
        fields["api_key"] = api_key
    if existing is not None:
        for carry in ("token_command", "token_refresh_interval", "token_command_env"):
            if carry not in body and carry in existing:
                fields[carry] = existing[carry]
    return fields


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

    entry: dict[str, Any] = {"base_url": base_url}
    entry.update(
        _credential_fields(body, api_key, existing_providers.get(resolve_name))
    )

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
