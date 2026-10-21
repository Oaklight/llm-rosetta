"""Bulk add: merge providers into model entries, then create them."""

from __future__ import annotations

from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response

from .._shared import (
    config_mutate,
    parse_json_body,
)


def _add_new_models(
    models_section: dict[str, Any],
    models_to_add: list[str],
    provider: str,
    prefix: str,
    upstream_map: dict[str, str],
    body: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Add new model entries to the models section.

    Returns (added, skipped) display name lists.
    """
    added: list[str] = []
    skipped: list[str] = []
    for model_id in models_to_add:
        display_name = f"{prefix}{model_id}" if prefix else model_id
        if display_name in models_section:
            skipped.append(display_name)
            continue
        model_type = body.get("type", "llm")
        entry: dict[str, Any] = {
            "provider": provider,
            "capabilities": body.get("capabilities", ["text", "vision", "tools"]),
        }
        if model_type != "llm":
            entry["type"] = model_type
        mapped_upstream = upstream_map.get(model_id)
        if mapped_upstream:
            entry["upstream_model"] = mapped_upstream
        elif prefix:
            entry["upstream_model"] = model_id
        models_section[display_name] = entry
        added.append(display_name)
    return added, skipped


def _make_provider_entry(
    provider: str,
    model_id: str,
    prefix: str,
    upstream_map: dict[str, str],
) -> dict[str, Any]:
    """Build a provider entry dict for multi-provider configs."""
    entry: dict[str, Any] = {"name": provider, "weight": 1}
    mapped = upstream_map.get(model_id)
    if mapped:
        entry["upstream_model"] = mapped
    elif prefix:
        entry["upstream_model"] = model_id
    return entry


def _merge_provider_into_model(
    models_section: dict[str, Any],
    display_name: str,
    new_provider_entry: dict[str, Any],
    capabilities: list[str],
) -> None:
    """Merge a provider entry into an existing model, converting to multi-provider."""
    existing = models_section[display_name]
    provider_name = new_provider_entry["name"]

    if isinstance(existing, str):
        if existing == provider_name:
            return
        old_entry: dict[str, Any] = {"name": existing, "weight": 1}
        models_section[display_name] = {
            "providers": [old_entry, new_provider_entry],
            "capabilities": capabilities,
        }
        return

    if not isinstance(existing, dict):
        return

    if "providers" in existing:
        plist = existing["providers"]
        names = [(p if isinstance(p, str) else p.get("name", "")) for p in plist]
        if provider_name not in names:
            plist.append(new_provider_entry)
        return

    if "provider" in existing:
        if existing["provider"] == provider_name:
            return
        old_provider = existing.pop("provider")
        old_upstream = existing.pop("upstream_model", None)
        old_p: dict[str, Any] = {"name": old_provider, "weight": 1}
        if old_upstream:
            old_p["upstream_model"] = old_upstream
        existing["providers"] = [old_p, new_provider_entry]
        return


def _append_provider_to_model(
    models_section: dict[str, Any],
    display_name: str,
    provider: str,
    model_id: str,
    prefix: str,
    upstream_map: dict[str, str],
    capabilities: list[str],
) -> str:
    """Append a provider to an existing model entry.

    Returns "appended", "skipped", or "ignored".
    """
    existing = models_section[display_name]
    new_p = _make_provider_entry(provider, model_id, prefix, upstream_map)

    if isinstance(existing, str):
        # String entries don't store capabilities; use the caller's value.
        old_entry: dict[str, Any] = {"name": existing, "weight": 1}
        models_section[display_name] = {
            "providers": [old_entry, new_p],
            "capabilities": capabilities,
        }
        return "appended"

    if not isinstance(existing, dict):
        return "ignored"

    if "providers" in existing:
        plist = existing["providers"]
        names = [(p if isinstance(p, str) else p.get("name", "")) for p in plist]
        if provider in names:
            return "skipped"
        plist.append(new_p)
        return "appended"

    if "provider" in existing:
        old_provider = existing.pop("provider")
        old_upstream = existing.pop("upstream_model", None)
        old_p: dict[str, Any] = {"name": old_provider, "weight": 1}
        if old_upstream:
            old_p["upstream_model"] = old_upstream
        existing["providers"] = [old_p, new_p]
        return "appended"

    return "ignored"


async def bulk_add_models(request: Any) -> Response:
    """Bulk-add multiple models for a given provider."""
    body, err = parse_json_body(request)
    if err:
        return err

    provider = body.get("provider")
    models_to_add: list[str] = body.get("models", [])
    models_to_append: list[str] = body.get("models_append", [])
    prefix = body.get("prefix", "")
    upstream_map: dict[str, str] = body.get("upstream_map", {})

    if not provider or (not models_to_add and not models_to_append):
        return JSONResponse(
            {
                "error": (
                    "'provider' and at least one of 'models'/'models_append' required"
                )
            },
            status_code=400,
        )

    async with config_mutate(request) as ctx:
        if ctx.error:
            return ctx.error

        # Validate provider exists
        providers = ctx.data.get("providers", {})
        if provider not in providers:
            return JSONResponse(
                {"error": f"Provider '{provider}' not found"}, status_code=400
            )

        models_section = ctx.data.setdefault("models", {})
        added, skipped = _add_new_models(
            models_section,
            models_to_add,
            provider,
            prefix,
            upstream_map,
            body,
        )

        # Append provider to existing models (multi-provider support)
        appended: list[str] = []
        for model_id in models_to_append:
            display_name = f"{prefix}{model_id}" if prefix else model_id
            if display_name not in models_section:
                continue
            result = _append_provider_to_model(
                models_section,
                display_name,
                provider,
                model_id,
                prefix,
                upstream_map,
                body.get("capabilities", ["text", "vision", "tools"]),
            )
            if result == "appended":
                appended.append(display_name)
            elif result == "skipped":
                skipped.append(display_name)

        if not added and not appended:
            return JSONResponse(
                {
                    "ok": True,
                    "added": [],
                    "appended": [],
                    "skipped": skipped,
                    "message": "All models already exist",
                }
            )

        ctx.commit()

    if ctx.error:
        return ctx.error
    assert ctx.new_config is not None

    return JSONResponse(
        {
            "ok": True,
            "added": added,
            "appended": appended,
            "skipped": skipped,
            "models": list(ctx.new_config.models),
        }
    )
