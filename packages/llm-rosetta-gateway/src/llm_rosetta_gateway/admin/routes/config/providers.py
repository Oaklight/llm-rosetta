"""Provider CRUD: create, update, delete, toggle, rename-merge."""

from __future__ import annotations

import logging
from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response
from llm_rosetta.shims import get_shim

from .._shared import (
    _build_provider_entry,
    _handle_provider_rename,
    config_mutate,
    parse_json_body,
)

from .bulk import _merge_provider_into_model

logger = logging.getLogger("llm-rosetta-gateway")


async def put_provider(request: Any, **kwargs: Any) -> Response:
    """Add or update a provider entry."""
    name = request.path_params["name"]

    body, err = parse_json_body(request)
    if err:
        return err

    api_key = body.get("api_key", "")
    base_url = body.get("base_url", "")

    async with config_mutate(request) as ctx:
        if ctx.error:
            return ctx.error

        existing_providers = ctx.data.get("providers", {})
        resolve_name = body.get("rename_from", name) or name

        # When api_key is omitted/empty and we're editing, keep the existing key
        if not api_key and resolve_name in existing_providers:
            api_key = existing_providers[resolve_name].get("api_key", "")

        # A shim can supply the connection defaults, so neither field is
        # mandatory when the selected type is one: the shim's base_url, and —
        # for a keyless shim (free resource) — no credential at all.  A
        # non-LLM endpoint shim (``embedding_shim``, …) counts too, so a
        # provider defined only by that shim needs no duplicated credentials.
        from ....model_types import all_model_types

        shims = [get_shim(body.get("type") or resolve_name)]
        for desc in all_model_types():
            if desc.is_llm or not desc.config_shim_key:
                continue
            shims.append(get_shim(body.get(desc.config_shim_key) or ""))
        shim_base = any(s is not None and s.connection.base_url for s in shims)
        keyless_ok = any(s is not None and s.connection.keyless for s in shims)
        missing = []
        if not base_url and not shim_base:
            missing.append("base_url")
        if not api_key and not keyless_ok:
            missing.append("api_key")
        if missing:
            return JSONResponse(
                {"error": f"Missing required field(s): {', '.join(missing)}"},
                status_code=400,
            )

        provider_entry = _build_provider_entry(
            body, api_key, base_url, existing_providers, resolve_name
        )

        # Handle rename: remove old entry and update model references
        rename_from = body.get("rename_from")
        if rename_from and rename_from != name:
            rename_err = _handle_provider_rename(ctx.data, rename_from, name)
            if rename_err is not None:
                return rename_err

        ctx.data.setdefault("providers", {})[name] = provider_entry
        ctx.commit()

    if ctx.error:
        return ctx.error
    assert ctx.new_config is not None

    return JSONResponse(
        {
            "ok": True,
            "provider": name,
            "providers": list(ctx.new_config.providers.keys()),
            "provider_warnings": ctx.warnings,
        }
    )


async def delete_provider(request: Any, **kwargs: Any) -> Response:
    """Remove a provider entry."""
    name = request.path_params["name"]

    async with config_mutate(request) as ctx:
        if ctx.error:
            return ctx.error

        providers = ctx.data.get("providers", {})
        if name not in providers:
            return JSONResponse(
                {"error": f"Provider '{name}' not found"}, status_code=404
            )

        # Check if any model still references this provider
        models = ctx.data.get("models", {})

        def _model_references_provider(model_cfg: Any, provider_name: str) -> bool:
            if isinstance(model_cfg, str):
                return model_cfg == provider_name
            if isinstance(model_cfg, dict):
                if "providers" in model_cfg:
                    for entry in model_cfg["providers"]:
                        pn = entry if isinstance(entry, str) else entry.get("name", "")
                        if pn == provider_name:
                            return True
                    return False
                return model_cfg.get("provider", "") == provider_name
            return False

        referencing = [
            m for m, p in models.items() if _model_references_provider(p, name)
        ]

        from .._shared import _qp

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
        ctx.commit()

    if ctx.error:
        return ctx.error
    assert ctx.new_config is not None

    result: dict[str, Any] = {
        "ok": True,
        "deleted": name,
        "providers": list(ctx.new_config.providers.keys()),
    }
    if cascade_deleted:
        result["cascade_deleted_models"] = cascade_deleted
    return JSONResponse(result)


async def toggle_provider(request: Any, **kwargs: Any) -> Response:
    """Toggle a provider's enabled/disabled state."""
    name = request.path_params["name"]

    async with config_mutate(request) as ctx:
        if ctx.error:
            return ctx.error

        providers = ctx.data.get("providers", {})
        if name not in providers:
            return JSONResponse(
                {"error": f"Provider '{name}' not found"}, status_code=404
            )

        # Toggle: if currently enabled (or unset → default True), disable; otherwise enable
        currently_enabled = providers[name].get("enabled", True)
        new_enabled = not currently_enabled

        if new_enabled:
            # Remove the key entirely when re-enabling (True is the default).
            # A prior automatic disable is no longer authoritative after an
            # explicit operator action; the next reload will revalidate it.
            providers[name].pop("enabled", None)
            providers[name].pop("_auto_disabled", None)
        else:
            providers[name]["enabled"] = False

        ctx.commit()

    if ctx.error:
        return ctx.error

    warnings = ctx.warnings
    effective_enabled = (
        name in ctx.new_config.providers if ctx.new_config is not None else new_enabled
    )
    response: dict[str, Any] = {
        "ok": True,
        "provider": name,
        "enabled": effective_enabled,
    }
    if warnings:
        response["provider_warnings"] = warnings
    return JSONResponse(response)


def _get_existing_provider_label(existing: Any) -> str:
    """Return a human-readable provider label for an existing model entry."""
    if isinstance(existing, str):
        return existing
    if isinstance(existing, dict) and "providers" in existing:
        names = [
            p if isinstance(p, str) else p.get("name", "")
            for p in existing["providers"]
        ]
        return ", ".join(filter(None, names))
    return existing.get("provider", "") if isinstance(existing, dict) else ""


def _handle_rename_merge(
    models: dict[str, Any],
    name: str,
    rename_from: str,
    body: dict[str, Any],
    provider: str,
) -> JSONResponse | None:
    """Handle the merge branch of a rename collision.

    Returns a 409 JSONResponse if merge was not requested, or ``None``
    after performing the merge (caller should save and reload).
    """
    existing = models[name]
    existing_provider = _get_existing_provider_label(existing)
    if not body.get("merge"):
        return JSONResponse(
            {
                "error": f"Model '{name}' already exists",
                "merge_possible": True,
                "existing_provider": existing_provider,
            },
            status_code=409,
        )
    raw_upstream = body.get("upstream_model")
    upstream = raw_upstream if raw_upstream is not None else rename_from
    new_p: dict[str, Any] = {"name": provider, "weight": 1}
    if upstream:
        new_p["upstream_model"] = upstream
    _merge_provider_into_model(models, name, new_p, body.get("capabilities", ["text"]))
    del models[rename_from]
    return None
