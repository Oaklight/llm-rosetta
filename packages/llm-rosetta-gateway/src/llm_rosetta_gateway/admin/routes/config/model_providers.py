"""Model-to-provider links: weights and removal."""

from __future__ import annotations

import logging
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response

from .._shared import (
    config_mutate,
    parse_json_body,
)

logger = logging.getLogger("llm-rosetta-gateway")


def _remove_provider_from_entry(
    models: dict[str, Any], name: str, provider: str
) -> Response | None:
    """Remove *provider* from the model entry *name* inside *models*.

    Returns a :class:`JSONResponse` error on failure, or ``None`` on
    success (caller should commit).
    """
    existing = models[name]

    # Single-provider shorthand (string or dict without "providers")
    if isinstance(existing, str):
        if existing != provider:
            return JSONResponse(
                {"error": f"Provider '{provider}' not found on model '{name}'"},
                status_code=404,
            )
        del models[name]
        return None

    if not isinstance(existing, dict):
        return JSONResponse({"error": "Invalid model entry"}, status_code=400)

    if "providers" not in existing:
        if existing.get("provider") != provider:
            return JSONResponse(
                {"error": f"Provider '{provider}' not found on model '{name}'"},
                status_code=404,
            )
        del models[name]
        return None

    # Multi-provider list
    plist = existing["providers"]
    new_list = [
        p for p in plist if (p if isinstance(p, str) else p.get("name", "")) != provider
    ]
    if len(new_list) == len(plist):
        return JSONResponse(
            {"error": f"Provider '{provider}' not found on model '{name}'"},
            status_code=404,
        )

    if not new_list:
        del models[name]
    elif len(new_list) == 1:
        _collapse_to_single_provider(models, name, existing, new_list[0])
    else:
        existing["providers"] = new_list

    return None


def _collapse_to_single_provider(
    models: dict[str, Any],
    name: str,
    existing: dict[str, Any],
    sole: Any,
) -> None:
    """Collapse a multi-provider model to single-provider format."""
    sole_name = sole if isinstance(sole, str) else sole.get("name", "")
    entry: dict[str, Any] = {"provider": sole_name}
    if isinstance(sole, dict) and sole.get("upstream_model"):
        entry["upstream_model"] = sole["upstream_model"]
    for key in (
        "capabilities",
        "type",
        "enabled",
        "flatten_system",
        "timeout",
        "reasoning_override",
        "url_template",
        "stream_url_template",
    ):
        if key in existing:
            entry[key] = existing[key]
    models[name] = entry


async def remove_model_provider(request: Any, **kwargs: Any) -> Response:
    """Remove a single provider from a multi-provider model entry."""
    name = request.path_params["name"]

    body, err = parse_json_body(request)
    if err:
        return err

    provider = body.get("provider")
    if not provider:
        return JSONResponse({"error": "'provider' is required"}, status_code=400)

    async with config_mutate(request) as ctx:
        if ctx.error:
            return ctx.error

        models = ctx.data.get("models", {})
        if name not in models:
            return JSONResponse({"error": f"Model '{name}' not found"}, status_code=404)

        err_resp = _remove_provider_from_entry(models, name, provider)
        if err_resp is not None:
            return err_resp

        ctx.commit()

    if ctx.error:
        return ctx.error
    assert ctx.new_config is not None

    deleted = name not in ctx.new_config.models and name not in models
    result: dict[str, Any] = {"ok": True, "model": name, "removed_provider": provider}
    if deleted:
        result["deleted"] = name
    result["models"] = list(ctx.new_config.models)
    return JSONResponse(result)


def _validate_weight(raw: Any) -> tuple[int | None, Response | None]:
    """Parse and validate a weight value from a request body.

    Returns ``(weight, None)`` on success or ``(None, error_response)``
    on failure.
    """
    if raw is None:
        return None, None
    try:
        w = int(raw)
    except (TypeError, ValueError):
        return None, JSONResponse(
            {"error": "'weight' must be an integer"}, status_code=400
        )
    if w < 1:
        return None, JSONResponse({"error": "'weight' must be >= 1"}, status_code=400)
    if w > 100:
        return None, JSONResponse({"error": "'weight' must be <= 100"}, status_code=400)
    return w, None


def _set_provider_attrs(
    plist: list[Any],
    provider: str,
    weight: int | None,
    enabled: bool | None,
) -> bool:
    """Find *provider* in *plist* and apply the given attributes.

    A provider entry turned off keeps its place in the list with
    ``enabled: false``, so the operator can turn it back on instead of
    re-adding it.  ``enabled`` true removes the key, which is the default.

    Returns True if *provider* was found.
    """
    for i, entry in enumerate(plist):
        pname = entry if isinstance(entry, str) else entry.get("name", "")
        if pname != provider:
            continue
        if isinstance(entry, str):
            new_entry: dict[str, Any] = {"name": entry}
            plist[i] = new_entry
            entry = new_entry
        if weight is not None:
            if weight == 1:
                entry.pop("weight", None)
            else:
                entry["weight"] = weight
        if enabled is not None:
            if enabled:
                entry.pop("enabled", None)
            else:
                entry["enabled"] = False
        return True
    return False


async def update_model_provider(request: Any, **kwargs: Any) -> Response:
    """Update a provider's attributes (e.g. weight) on a multi-provider model."""
    name = request.path_params["name"]

    body, err = parse_json_body(request)
    if err:
        return err

    provider = body.get("provider")
    if not provider:
        return JSONResponse({"error": "'provider' is required"}, status_code=400)

    weight, weight_err = _validate_weight(body.get("weight"))
    if weight_err:
        return weight_err

    enabled = body.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        return JSONResponse({"error": "'enabled' must be a boolean"}, status_code=400)
    if weight is None and enabled is None:
        return JSONResponse(
            {"error": "One of 'weight' or 'enabled' is required"}, status_code=400
        )

    async with config_mutate(request) as ctx:
        if ctx.error:
            return ctx.error

        models = ctx.data.get("models", {})
        if name not in models:
            return JSONResponse({"error": f"Model '{name}' not found"}, status_code=404)

        existing = models[name]
        if not isinstance(existing, dict) or "providers" not in existing:
            return JSONResponse(
                {"error": "Weights are only supported for multi-provider models"},
                status_code=400,
            )

        if not _set_provider_attrs(existing["providers"], provider, weight, enabled):
            return JSONResponse(
                {"error": f"Provider '{provider}' not found on model '{name}'"},
                status_code=404,
            )

        ctx.commit()

    if ctx.error:
        return ctx.error

    return JSONResponse(
        {
            "ok": True,
            "model": name,
            "provider": provider,
            "weight": weight,
            "enabled": enabled,
        }
    )
