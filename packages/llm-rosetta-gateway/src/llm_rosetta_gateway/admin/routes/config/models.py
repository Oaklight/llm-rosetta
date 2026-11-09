"""Model CRUD: create, update, delete, toggle, bulk edit."""

from __future__ import annotations

import logging
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response

from .._shared import (
    config_mutate,
    parse_json_body,
)

from .bulk import Lane, _append_provider_to_model, lane_key
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


def _build_model_entry(
    body: dict[str, Any],
    provider: str,
    existing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a model config entry from request body.

    The model type is validated against the model type registry so that
    new types registered via :func:`~model_types.register_model_type`
    are automatically accepted without code changes here.

    ``existing`` is the entry being replaced: an option the editor does not
    render (yet) is carried over rather than silently dropped by a save that
    never knew about it.
    """
    from llm_rosetta.formats import canonical_model_type

    from ....model_types import registered_type_names

    model_type = canonical_model_type(body.get("type", "llm"))
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
    failover = body.get("failover_on_error", body.get("failover_on_quota"))
    if failover is not None:
        # On is the default, so only an opt-out needs the key.
        if not failover:
            entry["failover_on_error"] = False
    elif existing and (
        existing.get("failover_on_error") is False
        or existing.get("failover_on_quota") is False
    ):
        entry["failover_on_error"] = False
    return entry


def _merge_failover_flag(existing: dict[str, Any], body: dict[str, Any]) -> None:
    """Merge the cross-provider failover flag, accepting the older spelling.

    Failover is on by default, so only an explicit ``false`` writes a key; a
    body that turns it on removes the key, and an explicit ``None`` restores
    the default.  ``failover_on_quota`` is the older spelling, and its
    explicit ``false`` is an opt-out too.  The old key is removed whenever the
    entry is written.
    """
    value = body.get("failover_on_error", body.get("failover_on_quota"))
    if value is not None:
        if value:
            existing.pop("failover_on_error", None)
        else:
            existing["failover_on_error"] = False
    elif "failover_on_error" in body or "failover_on_quota" in body:
        existing.pop("failover_on_error", None)
    elif existing.get("failover_on_quota") is False:
        existing["failover_on_error"] = False
    existing.pop("failover_on_quota", None)


#: Placeholder key for the one-entry mapping the merge helper works on.
_MERGE_KEY = "__merged__"


def _declared_providers(entry: dict[str, Any]) -> list[str]:
    """The provider names an entry already lists, in order."""
    return [lane[0] for lane in _declared_lanes(entry)]


def _declared_lanes(entry: dict[str, Any]) -> list[Lane]:
    """The lanes an entry already lists, in declared order.

    One provider may appear several times, each lane asking for its own
    upstream model; the pair is what makes a lane distinct.
    """
    raw = entry.get("providers")
    if isinstance(raw, list):
        return [
            lane_key(p, None)
            if isinstance(p, str)
            else lane_key(p.get("name", ""), p.get("upstream_model"))
            for p in raw
        ]
    provider = entry.get("provider")
    if isinstance(provider, str) and provider:
        return [lane_key(provider, entry.get("upstream_model"))]
    return []


def _to_multi_provider_entry(
    existing: dict[str, Any], new_provider: str, body: dict[str, Any]
) -> dict[str, Any]:
    """Merge a second provider into a single-provider model entry.

    Reuses the bulk-add conversion, which already knows how to move a
    single-provider entry's ``upstream_model`` onto its own provider entry.
    The fields the editor just saved then apply to the model as a whole.
    """
    holder: dict[str, Any] = {_MERGE_KEY: dict(existing)}
    _append_provider_to_model(
        holder,
        _MERGE_KEY,
        new_provider,
        new_provider,
        "",
        {},
        body.get("capabilities") or ["text"],
        upstream=body.get("upstream_model"),
    )
    entry = holder[_MERGE_KEY]
    _update_multi_provider_entry(entry, body, new_provider)
    if "enabled" in existing:
        entry["enabled"] = existing["enabled"]
    return entry


def _carry_routing(existing: dict[str, Any], body: dict[str, Any]) -> None:
    """Carry the model's ``routing`` block, which the editor writes whole.

    The block is stored verbatim; the config rebuild that follows the save is
    what validates it, so a bad value comes back as the parser's own message
    instead of a second, weaker check here.  A body that omits the key leaves
    a block already stored in place.
    """
    if "routing" not in body:
        return
    routing = body["routing"]
    if routing in (None, {}):
        existing.pop("routing", None)
    else:
        existing["routing"] = routing


def _append_lane(existing: dict[str, Any], provider: str, body: dict[str, Any]) -> None:
    """Add the lane *body* asks for, unless the entry already has it.

    A provider may join several times; the lane — provider and upstream model
    — is what is checked, and the lane carries the upstream the caller named.
    """
    lane_entry: dict[str, Any] = {"name": provider}
    if body.get("upstream_model"):
        lane_entry["upstream_model"] = body["upstream_model"]
    if lane_key(provider, body.get("upstream_model")) not in _declared_lanes(existing):
        existing.setdefault("providers", []).append(lane_entry)


def _update_multi_provider_entry(
    existing: dict[str, Any], body: dict[str, Any], provider: str = ""
) -> None:
    """Update a multi-provider model entry without destroying the providers list.

    ``provider`` joins the list when the caller asked to merge and the entry
    does not name it yet.
    """
    if provider and body.get("merge"):
        _append_lane(existing, provider, body)
    if "capabilities" in body:
        existing["capabilities"] = body["capabilities"]
    from llm_rosetta.formats import canonical_model_type

    model_type = canonical_model_type(body["type"]) if body.get("type") else None
    if model_type and model_type != "llm":
        existing["type"] = model_type
    elif "type" in existing and (not model_type or model_type == "llm"):
        del existing["type"]
    # ``upstream_model`` is historical at this level: once an entry lists
    # ``providers[]``, every reader — the loader, the routing route, the panel —
    # takes the upstream from the lane, and a top-level value is ignored.  It
    # is still written here because the editor's single-provider form posts the
    # field at the entry level; a multi-provider entry keeping both is known and
    # pinned by a test.
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
    # ``failover_on_error`` is canonical; ``failover_on_quota`` is the older
    # spelling an existing entry may still carry.  A body that mentions
    # neither keeps the current behaviour.
    _merge_failover_flag(existing, body)
    _carry_routing(existing, body)
    if body.get("enabled") is not None:
        existing["enabled"] = body["enabled"]


def _second_lane_confirmation(
    name: str, provider: str, existing: dict[str, Any]
) -> JSONResponse:
    """409 offering to add a second lane of *provider* instead of replacing one.

    The upstream named in the message comes from the lanes themselves: a
    ``providers[]`` entry keeps each lane's upstream on that lane, so reading
    the entry-level field there would report the model name as an upstream.
    """
    upstreams = [
        lane[1] or name for lane in _declared_lanes(existing) if lane[0] == provider
    ]
    shown = ", ".join(f"'{upstream}'" for upstream in upstreams) or f"'{name}'"
    return JSONResponse(
        {
            "error": f"Model '{name}' already routes '{provider}' to {shown}",
            "merge_possible": True,
            "existing_provider": provider,
        },
        status_code=409,
    )


def _write_model_entry(
    models_section: dict[str, Any],
    name: str,
    existing: Any,
    provider: str,
    body: dict[str, Any],
) -> tuple[Response | None, bool]:
    """Store one model entry, merging a second provider when asked.

    Returns ``(error, merged)``.  The error is a 409 when *name* is already
    routed through a different provider and the caller did not ask to merge:
    silently replacing the entry would drop the first provider's model.

    Args:
        models_section: The config's ``models`` mapping, changed in place.
        name: Model name being written.
        existing: The entry already stored under *name*, if any.
        provider: Provider the request names.
        body: The request body.

    Returns:
        ``(None, merged)`` on success, or ``(error_response, False)``.
    """
    if isinstance(existing, str):
        existing = {"provider": existing}
    if (
        isinstance(existing, dict)
        and "providers" in existing
        and not body.get("_force_single")
    ):
        # Naming a lane the entry does not list is a request to add it; that
        # still needs the caller to confirm, exactly as a second provider does.
        body_upstream = body.get("upstream_model")
        lane = lane_key(provider, body_upstream)
        if (
            body_upstream is not None
            and lane not in _declared_lanes(existing)
            and not body.get("merge")
        ):
            return _second_lane_confirmation(name, provider, existing), False
        _update_multi_provider_entry(existing, body, provider)
        return None, lane in _declared_lanes(existing)
    if isinstance(existing, dict) and existing.get("provider") != provider:
        if not body.get("merge"):
            return (
                JSONResponse(
                    {
                        "error": f"Model '{name}' is already served by "
                        f"'{existing.get('provider')}'",
                        "merge_possible": True,
                        "existing_provider": existing.get("provider"),
                    },
                    status_code=409,
                ),
                False,
            )
        models_section[name] = _to_multi_provider_entry(existing, provider, body)
        return None, True
    if (
        isinstance(existing, dict)
        and body.get("upstream_model") is not None
        and lane_key(provider, body.get("upstream_model"))
        != lane_key(existing.get("provider", ""), existing.get("upstream_model"))
    ):
        # The same provider, a different upstream model: that is a second lane,
        # not an edit of the first one.  Silently replacing the entry would
        # drop the lane the model already serves.
        if not body.get("merge"):
            return _second_lane_confirmation(name, provider, existing), False
        models_section[name] = _to_multi_provider_entry(existing, provider, body)
        return None, True
    models_section[name] = _build_model_entry(
        body, provider, existing if isinstance(existing, dict) else None
    )
    return None, False


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
            err, merged = _write_model_entry(
                models_section, name, models_section.get(name), provider, body
            )
            if err is not None:
                return err
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
