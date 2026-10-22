"""The config payload: the document the admin panel reads and writes."""

from __future__ import annotations

import logging
import os
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response
from llm_rosetta.shims import get_shim, list_shims
from llm_rosetta.shims.provider_shim import match_model_reasoning

from ....config import GatewayConfig
from ....providers import (
    BASE_FORMATS,
    get_base_type_hints,
    get_default_supports_custom_tools,
    known_provider_types,
)
from ....shims import SHIM_SOURCE_PACKAGES
from .._shared import (
    _get_config_io,
    _get_config_path,
    _mask_api_key,
    _mask_proxy_url,
    _sanitize_server_section,
)

from .shims import _shim_removal_advisories, _shim_status_payload

logger = logging.getLogger("llm-rosetta-gateway")


def _shim_source_presets_payload() -> list[dict[str, Any]]:
    """Payload for the Settings → Shim Sources preset picker.

    The catalogue is server-supplied so the client never hardcodes package
    names; adding a standalone package is a change to
    :data:`llm_rosetta.shims.SHIM_SOURCE_PACKAGES` only.
    """
    return [entry.to_dict() for entry in SHIM_SOURCE_PACKAGES]


def _base_formats_payload() -> list[dict[str, Any]]:
    """Payload for the base-format options in the provider-type picker.

    The set of formats follows ``BASE_FORMATS`` (next to ``_PROVIDER_REGISTRY``)
    so a new base type is added in one place.  ``supports_custom_tools`` carries
    the format's own default so a bare format matches its vendor shim instead of
    silently dropping the toggle.
    ``placeholder_base_url`` / ``placeholder_api_key_env`` carry a vendor
    suggestion for the input's ``placeholder`` attribute only: a bare base
    format is vendor-neutral, so these are hints, never pre-filled values or
    runtime fallbacks.
    ``hoist_system_messages`` is not carried: every format defaults it to true
    (the JS reads ``!== false``), so there is nothing to override.
    """
    out: list[dict[str, Any]] = []
    for name in BASE_FORMATS:
        hints = get_base_type_hints(name)
        out.append(
            {
                "name": name,
                "placeholder_base_url": hints.get("base_url", ""),
                "placeholder_api_key_env": hints.get("api_key_env", ""),
                "supports_custom_tools": get_default_supports_custom_tools(name),
            }
        )
    return out


def _get_model_type_metadata() -> list[dict[str, Any]]:
    """Return metadata for all registered model types.

    Exposed via ``/admin/api/config`` so the admin UI can render type
    badges, seg-control options, and filters from server-provided data
    instead of hardcoding type names.
    """
    from ....model_types import all_model_types

    return [
        {
            "name": desc.name,
            "formats": desc.formats,
            "supports_streaming": desc.supports_streaming,
            "badge_class": desc.badge_class,
            "routes": [{"path": r.path, "methods": r.methods} for r in desc.routes],
            "config_format_key": desc.config_format_key,
            "config_path_key": desc.config_path_key,
            "config_shim_key": desc.config_shim_key,
            "default_format": desc.formats[0] if desc.formats else None,
            "default_path": desc.default_path,
            "default_paths": dict(desc.default_paths),
            "icon_svg": desc.icon_svg,
            "color": desc.color,
            "is_llm": desc.is_llm,
        }
        for desc in all_model_types()
    ]


def _get_gateway_config(request: Any) -> GatewayConfig | None:
    """Return the live GatewayConfig from the app or module."""
    cfg = getattr(request.app, "gateway_config", None)
    if cfg is not None:
        return cfg
    from llm_rosetta_gateway.app import _state as _app_state

    return _app_state._config


def _get_version() -> str:
    """Return the gateway package version (the panel's own)."""
    try:
        from llm_rosetta_gateway import __version__

        return __version__
    except Exception:
        return "unknown"


def _get_core_version() -> str:
    """Return the conversion core's version, which the gateway depends on."""
    try:
        from llm_rosetta import __version__

        return __version__
    except Exception:
        return "unknown"


def _resolve_model_reasoning(
    model_name: str,
    entry: dict[str, Any],
    raw_models: dict[str, Any],
    providers: dict[str, Any],
) -> dict[str, Any]:
    """Resolve the effective reasoning config for a model.

    Resolution priority:
    1. config.jsonc model-level ``reasoning_override``
    2. Shim ``model_reasoning`` (exact upstream ID, then glob patterns)
    3. Shim provider-level ``reasoning``

    Returns a dict with ``source`` indicating where the config came from,
    plus the effective field values.
    """
    provider_name = entry.get("provider", "")
    provider_cfg = providers.get(provider_name, {})
    shim_name = provider_cfg.get("type") or provider_name
    upstream = entry.get("upstream_model") or model_name

    shim = get_shim(shim_name)
    if not shim or not shim.reasoning:
        return {"source": "none"}

    cap = shim.reasoning
    source = "provider"
    override = match_model_reasoning(shim, upstream)
    if override is not None:
        cap = override
        source = "model_override"

    # Check config-level override
    raw_model = raw_models.get(model_name, {})
    config_override = (
        raw_model.get("reasoning_override") if isinstance(raw_model, dict) else None
    )
    if config_override:
        source = "config"

    return {
        "source": source,
        "thinking_modes": cap.thinking_modes,
        "effort_field": cap.effort_field,
        "effort_range": list(cap.effort_range) if cap.effort_range else None,
        "budget_ratio": cap.budget_ratio,
        "visibility_modes": cap.visibility_modes,
        "replay_missing": cap.replay_missing,
        "unsigned_blocks": cap.unsigned_blocks,
        "config_override": config_override,
    }


def _normalize_model_entry(value: Any) -> dict[str, Any]:
    """Normalize a single model config value to a dict for the admin UI."""
    if isinstance(value, str):
        return {"provider": value, "capabilities": ["text"]}
    entry: dict[str, Any] = {
        "capabilities": value.get("capabilities", ["text"]),
    }
    if "providers" in value:
        entry["providers"] = value["providers"]
        if "strategy" in value:
            entry["strategy"] = value["strategy"]
        # For backward compat, set "provider" to the first entry
        first = value["providers"][0] if value["providers"] else ""
        entry["provider"] = first if isinstance(first, str) else first.get("name", "")
    else:
        entry["provider"] = value.get("provider", "")
    if value.get("type"):
        entry["type"] = value["type"]
    for key in (
        "upstream_model",
        "reasoning_override",
        "flatten_system",
        "url_template",
        "stream_url_template",
    ):
        if value.get(key):
            entry[key] = value[key]
    if value.get("enabled") is not None:
        entry["enabled"] = value["enabled"]
    return entry


def _merge_typed_models(raw: dict[str, Any], models: dict[str, Any]) -> None:
    """Merge embedding/rerank/decision models into the main models dict."""
    for model_type in ("embedding", "rerank", "decision"):
        for model_name, provider_name in raw.get(f"{model_type}_models", {}).items():
            if model_name not in models:
                prov = (
                    provider_name
                    if isinstance(provider_name, str)
                    else provider_name.get("provider", "")
                )
                models[model_name] = {
                    "provider": prov,
                    "type": model_type,
                    "capabilities": [model_type],
                }


def _mask_endpoint_markers(endpoints: Any) -> dict[str, dict[str, str]]:
    """Sanitise per-endpoint degradation markers, keyed by model type.

    Same contract as the whole-provider marker: the reason and the category
    are configuration text and travel to the panel, nothing else does.
    """
    if not isinstance(endpoints, dict):
        return {}
    return {
        str(type_name): {
            key: str(marker[key])
            for key in ("reason", "category")
            if marker.get(key) is not None
        }
        for type_name, marker in endpoints.items()
        if isinstance(marker, dict)
    }


def _with_live_endpoint_markers(
    masked: dict[str, Any], name: str, gateway_config: Any
) -> None:
    """Overlay the endpoint markers the running parse recorded (#1106).

    ``config/typed.py`` writes ``_auto_disabled_endpoints`` onto the caller's
    provider mapping while the config is parsed; this payload reads the
    *document on disk*, so a degraded endpoint would never reach the panel.
    The whole-provider marker does not have that problem because the
    validation step persists it — this one is deliberately not persisted (a
    parse runs on every load, and a stale marker on disk would outlive the
    shim being fixed), so it is merged in from the live config instead.
    """
    if gateway_config is None:
        return
    live_providers = getattr(gateway_config, "raw_providers", None)
    if not isinstance(live_providers, dict):
        return
    live = live_providers.get(name)
    if not isinstance(live, dict):
        return
    markers = _mask_endpoint_markers(live.get("_auto_disabled_endpoints"))
    if markers:
        masked["_auto_disabled_endpoints"] = markers


def _mask_provider_entry(name: str, cfg: Any) -> dict[str, Any]:
    """Mask one provider's credentials and keep its diagnostics.

    The admin UI needs the validation reasons but never the API key, the
    proxy userinfo, or any other field a diagnostic mapping might carry.
    """
    masked = dict(cfg)
    if "api_key" in masked:
        masked["api_key"] = _mask_api_key(masked["api_key"])
    if "proxy" in masked:
        masked["proxy"] = _mask_proxy_url(masked["proxy"])
    # Keep only the diagnostic fields needed by the admin UI.  Provider
    # validation reasons are configuration text, never credentials.
    marker = masked.get("_auto_disabled")
    if isinstance(marker, dict):
        masked["_auto_disabled"] = {
            key: str(marker[key])
            for key in ("reason", "category", "shim")
            if marker.get(key) is not None
        }
    else:
        masked.pop("_auto_disabled", None)
    # A provider can serve several model types; one endpoint having a bad
    # shim degrades that endpoint only.  Keep the same diagnostic shape,
    # keyed by model type.
    markers = _mask_endpoint_markers(masked.get("_auto_disabled_endpoints"))
    if markers:
        masked["_auto_disabled_endpoints"] = markers
    else:
        masked.pop("_auto_disabled_endpoints", None)
    # Ensure explicit type — fall back to provider name for legacy configs
    if "type" not in masked:
        masked["type"] = name
    return masked


async def get_config(request: Any) -> Response:
    """Return the current (raw) gateway configuration."""
    config_path = _get_config_path(request)

    try:
        raw = _get_config_io(request).load_raw(config_path)
    except Exception as exc:
        return JSONResponse({"error": f"Failed to read config: {exc}"}, status_code=500)

    # Mask API keys and ensure each provider has a "type" field
    providers = raw.get("providers", {})
    masked_providers: dict[str, Any] = {}
    gateway_config: GatewayConfig | None = getattr(request.app, "gateway_config", None)
    for name, cfg in providers.items():
        masked = _mask_provider_entry(name, cfg)
        _with_live_endpoint_markers(masked, name, gateway_config)
        # Inject token_status from the live ProviderInfo if available
        if gateway_config is not None:
            pinfo = gateway_config.providers.get(name)
            if pinfo is not None and pinfo.token_status is not None:
                masked["token_status"] = pinfo.token_status
            elif pinfo is not None and pinfo.mints_credential:
                masked["token_status"] = {
                    "enabled": True,
                    "last_refresh": None,
                    "consecutive_failures": 0,
                    "last_error": None,
                }
            if pinfo is not None:
                masked["keyless"] = pinfo.keyless
        masked_providers[name] = masked

    # Normalize models to dict format for consistent admin UI
    raw_models = raw.get("models", {})
    models_normalized = {
        name: _normalize_model_entry(value) for name, value in raw_models.items()
    }

    # Merge typed models (embedding, rerank, decision) so the admin UI
    # shows all model types in one unified list.
    _merge_typed_models(raw, models_normalized)

    # Resolve effective reasoning config per model
    for model_name, entry in models_normalized.items():
        entry["reasoning"] = _resolve_model_reasoning(
            model_name, entry, raw_models, providers
        )

    server = _sanitize_server_section(raw.get("server", {}))

    config: GatewayConfig = request.app.gateway_config
    return JSONResponse(
        {
            "config_path": os.path.basename(config_path),
            "providers": masked_providers,
            "models": models_normalized,
            "server": server,
            "shims": raw.get("shims", {}),
            "shim_sources": _shim_status_payload(request),
            "shim_removals": _shim_removal_advisories(request, raw),
            "shim_source_presets": _shim_source_presets_payload(),
            "debug": raw.get("debug", {}),
            "credential_visible": config.credential_visible,
            "has_admin_password": bool(request.app.auth_state.admin_password),
            "api_keys_db": config.api_keys_db,
            "version": _get_version(),
            "core_version": _get_core_version(),
            "known_provider_types": known_provider_types(),
            "base_formats": _base_formats_payload(),
            "registered_shims": [
                {
                    "name": s.name,
                    "base": s.base,  # deprecated alias for `format`
                    "format": s.format,
                    "model_type": s.model_type,
                    "logo": s.logo,
                    "display_name": s.display_name,
                    "group": s.group,
                    "group_label": s.group_label,
                    "free_source": s.free_source,
                    "keyless": s.connection.keyless,
                    "default_base_url": s.connection.base_url,
                    "default_api_key_env": s.connection.api_key_env,
                    "default_path": s.connection.default_path,
                    "supports_custom_tools": s.tools.custom_tools,
                    "hoist_system_messages": s.hoist_system_messages,
                    "max_tool_description_length": s.tools.max_description_length,
                }
                for s in list_shims()
            ],
            "embedding_formats": GatewayConfig.embedding_formats(),
            "rerank_formats": GatewayConfig.rerank_formats(),
            "model_types": _get_model_type_metadata(),
        }
    )
