"""Model CRUD: create, update, delete, toggle, bulk edit."""

from __future__ import annotations

import logging
from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response

from .._shared import (
    config_mutate,
    parse_json_body,
)

from .providers import _handle_rename_merge

logger = logging.getLogger("llm-rosetta-gateway")


async def toggle_model(request: Any, **kwargs: Any) -> Response:
    """Toggle a model's enabled/disabled state."""
    name = request.path_params["name"]

    async with config_mutate(request) as ctx:
        if ctx.error:
            return ctx.error

        models = ctx.data.get("models", {})
        if name not in models:
            return JSONResponse({"error": f"Model '{name}' not found"}, status_code=404)

        # Normalize string shorthand to dict
        if isinstance(models[name], str):
            models[name] = {"provider": models[name]}

        currently_enabled = models[name].get("enabled", True)
        new_enabled = not currently_enabled

        if new_enabled:
            models[name].pop("enabled", None)
        else:
            models[name]["enabled"] = False

        ctx.commit()

    if ctx.error:
        return ctx.error

    return JSONResponse({"ok": True, "model": name, "enabled": new_enabled})


async def bulk_update_models(request: Any) -> Response:
    """Bulk enable, disable, or delete models."""
    body, err = parse_json_body(request)
    if err:
        return err

    action = body.get("action")
    names = body.get("models", [])
    if action not in ("enable", "disable", "delete") or not names:
        return JSONResponse(
            {"error": "'action' (enable/disable/delete) and 'models' are required"},
            status_code=400,
        )

    async with config_mutate(request) as ctx:
        if ctx.error:
            return ctx.error

        models = ctx.data.get("models", {})
        affected: list[str] = []

        for name in names:
            if name not in models:
                continue
            if action == "delete":
                del models[name]
                affected.append(name)
            else:
                if isinstance(models[name], str):
                    models[name] = {"provider": models[name]}
                if action == "enable":
                    models[name].pop("enabled", None)
                else:
                    models[name]["enabled"] = False
                affected.append(name)

        ctx.commit()

    if ctx.error:
        return ctx.error
    assert ctx.new_config is not None

    return JSONResponse(
        {
            "ok": True,
            "action": action,
            "affected": affected,
            "models": list(ctx.new_config.models),
        }
    )


def _build_model_entry(body: dict[str, Any], provider: str) -> dict[str, Any]:
    """Build a model config entry from request body.

    The model type is validated against the model type registry so that
    new types registered via :func:`~model_types.register_model_type`
    are automatically accepted without code changes here.
    """
    from ....model_types import registered_type_names

    model_type = body.get("type", "llm")
    valid_types = registered_type_names()
    if model_type not in valid_types:
        # Fall back to "llm" for unrecognised types — maintains backward
        # compat with configs that omit the type field entirely.
        model_type = "llm"
    entry: dict[str, Any] = {
        "provider": provider,
        "capabilities": body.get("capabilities", ["text"]),
    }
    if model_type != "llm":
        entry["type"] = model_type
    for opt_key in ("upstream_model", "url_template", "stream_url_template"):
        if body.get(opt_key):
            entry[opt_key] = body[opt_key]
    if body.get("flatten_system") is not None:
        entry["flatten_system"] = bool(body["flatten_system"])
    if body.get("timeout") not in (None, ""):
        entry["timeout"] = float(body["timeout"])
    reasoning_override = body.get("reasoning_override")
    if isinstance(reasoning_override, dict):
        cleaned = {k: v for k, v in reasoning_override.items() if v is not None}
        if cleaned:
            entry["reasoning_override"] = cleaned
    return entry


def _update_multi_provider_entry(
    existing: dict[str, Any], body: dict[str, Any]
) -> None:
    """Update a multi-provider model entry without destroying the providers list."""
    if "capabilities" in body:
        existing["capabilities"] = body["capabilities"]
    model_type = body.get("type")
    if model_type and model_type != "llm":
        existing["type"] = model_type
    elif "type" in existing and (not model_type or model_type == "llm"):
        del existing["type"]
    for key in ("upstream_model", "url_template", "stream_url_template"):
        if body.get(key):
            existing[key] = body[key]
        elif key in existing and key in body:
            del existing[key]
    if body.get("flatten_system") is not None:
        existing["flatten_system"] = bool(body["flatten_system"])
    if body.get("timeout") not in (None, ""):
        existing["timeout"] = float(body["timeout"])
    elif "timeout" in existing and "timeout" in body:
        del existing["timeout"]
    reasoning_override = body.get("reasoning_override")
    if isinstance(reasoning_override, dict):
        cleaned = {k: v for k, v in reasoning_override.items() if v is not None}
        if cleaned:
            existing["reasoning_override"] = cleaned
        elif "reasoning_override" in existing:
            del existing["reasoning_override"]
    if body.get("enabled") is not None:
        existing["enabled"] = body["enabled"]


async def put_model(request: Any, **kwargs: Any) -> Response:
    """Add or update a model routing entry."""
    name = request.path_params["name"]

    body, err = parse_json_body(request)
    if err:
        return err

    provider = body.get("provider")
    if not provider:
        return JSONResponse({"error": "'provider' is required"}, status_code=400)

    merged = False

    async with config_mutate(request) as ctx:
        if ctx.error:
            return ctx.error

        # Validate that the provider exists
        providers = ctx.data.get("providers", {})
        if provider not in providers:
            return JSONResponse(
                {"error": f"Provider '{provider}' not found in config"},
                status_code=400,
            )

        # Handle rename: remove old entry, or merge into existing model
        rename_from = body.get("rename_from")
        if rename_from and rename_from != name:
            models = ctx.data.get("models", {})
            if rename_from not in models:
                return JSONResponse(
                    {"error": f"Original model '{rename_from}' not found"},
                    status_code=404,
                )
            if name in models:
                merge_err = _handle_rename_merge(
                    models, name, rename_from, body, provider
                )
                if merge_err is not None:
                    return merge_err
                # Merge succeeded — commit and let response be built after
                merged = True
                ctx.commit()
            else:
                del models[rename_from]

        if not merged:
            models_section = ctx.data.setdefault("models", {})
            existing = models_section.get(name)
            if (
                isinstance(existing, dict)
                and "providers" in existing
                and not body.get("_force_single")
            ):
                _update_multi_provider_entry(existing, body)
            else:
                models_section[name] = _build_model_entry(body, provider)
            ctx.commit()

    if ctx.error:
        return ctx.error
    assert ctx.new_config is not None

    result: dict[str, Any] = {
        "ok": True,
        "model": name,
        "provider": provider,
        "capabilities": body.get("capabilities", ["text"]),
        "models": list(ctx.new_config.models),
    }
    if merged:
        result["merged"] = True
    return JSONResponse(result)


async def delete_model(request: Any, **kwargs: Any) -> Response:
    """Remove a model routing entry."""
    name = request.path_params["name"]

    async with config_mutate(request) as ctx:
        if ctx.error:
            return ctx.error

        models = ctx.data.get("models", {})
        if name not in models:
            return JSONResponse({"error": f"Model '{name}' not found"}, status_code=404)

        del models[name]
        ctx.commit()

    if ctx.error:
        return ctx.error
    assert ctx.new_config is not None

    return JSONResponse(
        {
            "ok": True,
            "deleted": name,
            "models": list(ctx.new_config.models),
        }
    )
