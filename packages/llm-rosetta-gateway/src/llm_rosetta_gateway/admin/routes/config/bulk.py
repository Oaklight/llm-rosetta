"""Bulk add: merge providers into model entries, then create them."""

from __future__ import annotations

from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response

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
        from llm_rosetta.formats import canonical_model_type

        model_type = canonical_model_type(body.get("type", "llm"))
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
    upstream: str | None = None,
) -> dict[str, Any]:
    """Build a provider entry dict for multi-provider configs.

    ``upstream`` names the lane explicitly — a second lane of a provider is
    only a second lane because it asks for a different upstream model.
    """
    entry: dict[str, Any] = {"name": provider, "weight": 1}
    if upstream:
        entry["upstream_model"] = upstream
        return entry
    mapped = upstream_map.get(model_id)
    if mapped:
        entry["upstream_model"] = mapped
    elif prefix:
        entry["upstream_model"] = model_id
    return entry


#: What identifies one candidate lane: the provider and the upstream model it
#: asks for.  Two lanes of one provider are two candidates; the same pair is
#: one candidate written twice.  Mirrors
#: :attr:`~llm_rosetta_gateway.routing.strategy.ProviderEntry.key`.
Lane = tuple[str, str | None]


def lane_key(name: str, upstream: Any) -> Lane:
    """The identity of one lane.  An absent and an empty upstream are one."""
    return (name, upstream or None)


def _provider_names(entry: Any) -> list[str]:
    """Return the provider names a model entry already routes through."""
    return [lane[0] for lane in _lane_keys(entry)]


def _lane_keys(entry: Any) -> list[Lane]:
    """The lanes a model entry already lists, in declared order."""
    if isinstance(entry, str):
        return [lane_key(entry, None)]
    if not isinstance(entry, dict):
        return []
    if "providers" in entry:
        return [
            lane_key(p, None)
            if isinstance(p, str)
            else lane_key(p.get("name", ""), p.get("upstream_model"))
            for p in entry["providers"]
        ]
    if "provider" in entry:
        return [lane_key(entry["provider"], entry.get("upstream_model"))]
    return []


def _merge_outcome(existing: Any, lane: Lane) -> str:
    """Say what adding *lane* to *existing* would do.

    A merge that adds nothing is not a no-op the caller may ignore: in the
    rename flow the caller deletes the source model afterwards, so a silent
    skip loses a model.  Deciding here, without mutating, lets the handler
    offer the merge only when it can happen and refuse it otherwise.

    The decision is per **lane**, not per provider: one provider may serve
    several upstream models, and only the same ``(provider, upstream_model)``
    pair twice is a duplicate.

    Returns:
        ``"merged"`` when the entry can take the lane (a new provider, or a
        second lane of one already listed), ``"skipped"`` when that exact lane
        is already there, or ``"ignored"`` when the entry is not a shape that
        can hold a provider list.
    """
    if isinstance(existing, str):
        return "skipped" if lane_key(existing, None) == lane else "merged"
    if not isinstance(existing, dict):
        return "ignored"
    if "providers" in existing:
        return "skipped" if lane in _lane_keys(existing) else "merged"
    if "provider" in existing:
        existing_lane = lane_key(existing["provider"], existing.get("upstream_model"))
        return "skipped" if existing_lane == lane else "merged"
    return "ignored"


def _merge_provider_into_model(
    models_section: dict[str, Any],
    display_name: str,
    new_provider_entry: dict[str, Any],
    capabilities: list[str],
) -> str:
    """Merge a provider entry into an existing model, converting to multi-provider.

    Returns:
        The :func:`_merge_outcome` verdict: ``"merged"`` when the provider was
        added, ``"skipped"`` when the entry already routes through it (nothing
        changed), or ``"ignored"`` when the entry cannot hold a provider list.
    """
    existing = models_section[display_name]
    lane = lane_key(
        new_provider_entry["name"], new_provider_entry.get("upstream_model")
    )
    outcome = _merge_outcome(existing, lane)
    if outcome != "merged":
        return outcome

    if isinstance(existing, str):
        old_entry: dict[str, Any] = {"name": existing, "weight": 1}
        models_section[display_name] = {
            "providers": [old_entry, new_provider_entry],
            "capabilities": capabilities,
        }
        return "merged"

    if "providers" in existing:
        existing["providers"].append(new_provider_entry)
        return "merged"

    # Single-provider dict: lift the old pair into the list, keeping its
    # upstream model with it.
    old_provider = existing.pop("provider")
    old_upstream = existing.pop("upstream_model", None)
    old_p: dict[str, Any] = {"name": old_provider, "weight": 1}
    if old_upstream:
        old_p["upstream_model"] = old_upstream
    existing["providers"] = [old_p, new_provider_entry]
    return "merged"


def _append_provider_to_model(
    models_section: dict[str, Any],
    display_name: str,
    provider: str,
    model_id: str,
    prefix: str,
    upstream_map: dict[str, str],
    capabilities: list[str],
    upstream: str | None = None,
) -> str:
    """Append a lane to an existing model entry.

    Returns "appended", "skipped", or "ignored".  A provider may appear more
    than once; only the same ``(provider, upstream_model)`` pair is a skip.
    """
    existing = models_section[display_name]
    new_p = _make_provider_entry(provider, model_id, prefix, upstream_map, upstream)
    # ``_merge_outcome`` says "merged" for a lane that can be added; this
    # helper's vocabulary calls that "appended".
    outcome = _merge_outcome(existing, lane_key(provider, new_p.get("upstream_model")))
    if outcome != "merged":
        return outcome

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
        existing["providers"].append(new_p)
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
        ignored: list[str] = []
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
            elif result == "ignored":
                # The entry is neither a provider name nor an object with one,
                # so there is no lane to add.  Say so instead of reporting a
                # batch that silently did nothing for this name.
                ignored.append(display_name)

        if not added and not appended:
            response: dict[str, Any] = {
                "ok": True,
                "added": [],
                "appended": [],
                "skipped": skipped,
                "ignored": ignored,
            }
            if not ignored:
                response["message"] = "All models already exist"
            return JSONResponse(response)

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
            "ignored": ignored,
            "models": list(ctx.new_config.models),
        }
    )
