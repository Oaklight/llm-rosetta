"""Provider CRUD: create, update, delete, toggle, rename-merge."""

from __future__ import annotations

import logging
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response
from llm_rosetta.shims import get_shim

from .._shared import (
    _build_provider_entry,
    _config_preflight_error,
    _handle_provider_rename,
    _shim_ships_credential_command,
    _shim_ships_credential_exchange,
    config_mutate,
    parse_json_body,
)

from .bulk import _merge_outcome, _merge_provider_into_model

logger = logging.getLogger("llm-rosetta-gateway")


def _missing_provider_fields_error(
    body: dict[str, Any],
    api_key: str,
    base_url: str,
    resolve_name: str,
    existing_providers: dict[str, Any],
) -> Response | None:
    """Refuse a body that names no usable connection, if it does not.

    A shim can supply the connection defaults, so neither field is mandatory
    when the selected type is one: the shim's base_url, and — for a keyless
    shim (free resource) — no credential at all.  A non-LLM endpoint shim
    (``embeddings_shim``, …) counts too, so a provider defined only by that
    shim needs no duplicated credentials.
    """
    from ....model_types import all_model_types

    shims = [get_shim(body.get("type") or resolve_name)]
    for desc in all_model_types():
        if desc.is_llm or not desc.config_shim_key:
            continue
        shims.append(get_shim(body.get(desc.config_shim_key) or ""))
    shim_base = any(s is not None and s.connection.base_url for s in shims)
    keyless_ok = any(s is not None and s.connection.keyless for s in shims)
    # A shim that ships ``auth.py`` mints the credential itself, so the
    # operator has no key to type here either.
    shim_exchange_ok = any(_shim_ships_credential_exchange(s) for s in shims)
    # The argv form of the same declaration: the shim's own command is the
    # credential source, so an empty key is the correct body.
    shim_command_ok = any(_shim_ships_credential_command(s) for s in shims)
    # A provider that mints its credential from a command stores no key — the
    # two sources are mutually exclusive, so demanding one here made every
    # ``token_command`` provider unsaveable from the panel: the editor
    # legitimately sends an empty key and the save came back 400.
    token_command = body.get("token_command")
    if token_command is None and resolve_name in existing_providers:
        token_command = existing_providers[resolve_name].get("token_command")
    missing = []
    if not base_url and not shim_base:
        missing.append("base_url")
    if (
        not api_key
        and not keyless_ok
        and not token_command
        and not shim_exchange_ok
        and not shim_command_ok
    ):
        missing.append("api_key")
    if not missing:
        return None
    # The editor has no key box for a command provider, so "missing api_key"
    # names a field the operator cannot see.  When the save just cleared the
    # command, say what the save would leave behind — and still name any other
    # missing field, so clearing the command and the base_url together does
    # not hide the second problem.
    clearing = "token_command" in body and not body.get("token_command")
    if clearing and "api_key" in missing:
        message = (
            "This provider would have no credential: the token command is "
            "empty and no API key is set."
        )
        others = [field for field in missing if field != "api_key"]
        if others:
            message += f" Missing required field(s): {', '.join(others)}."
        return JSONResponse({"error": message}, status_code=400)
    return JSONResponse(
        {"error": f"Missing required field(s): {', '.join(missing)}"},
        status_code=400,
    )


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
        # ...unless the editor switched away from the static key, which asks
        # for the stored one to be removed.  The save then has to be judged on
        # the credential it will actually leave behind, not on the old key.
        if body.get("clear_api_key") is True:
            api_key = ""

        missing_err = _missing_provider_fields_error(
            body, api_key, base_url, resolve_name, existing_providers
        )
        if missing_err is not None:
            return missing_err

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

        # Write only what the loader will accept.  Saving first and reloading
        # second left a rejected document on disk, and the next process start
        # failed on it; this turns that 500 into a 400 that writes nothing.
        load_error = _config_preflight_error(ctx.data)
        if load_error is not None:
            return JSONResponse({"error": load_error}, status_code=400)

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
            # Enabling is the moment a broken entry becomes loadable — or
            # fatal.  Validate before the switch is written, not after.
            load_error = _config_preflight_error(ctx.data)
            if load_error is not None:
                return JSONResponse({"error": load_error}, status_code=400)
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


def _one_provider(entry: Any) -> str:
    """Name one provider leg, with the upstream model it asks for."""
    if isinstance(entry, str):
        return entry
    name = entry.get("name", "")
    upstream = entry.get("upstream_model")
    return f"{name} -> {upstream}" if upstream else name


def _source_upstream(entry: Any, provider: str) -> str | None:
    """The upstream model the source entry asks *provider* for, if it names one.

    The source of a rename-merge may be a bare name, a single-provider entry or
    a provider list; only the leg naming the provider being merged says which
    upstream model the new entry should keep.
    """
    if not isinstance(entry, dict):
        return None
    if "providers" in entry:
        # The same provider may appear on several lanes; the source's upstream
        # is the first *declared* one among them, not the first lane that
        # happens to name the provider and declares nothing.
        for lane in entry["providers"]:
            name = lane if isinstance(lane, str) else lane.get("name", "")
            if name != provider:
                continue
            upstream = None if isinstance(lane, str) else lane.get("upstream_model")
            if upstream:
                return upstream
        return None
    if entry.get("provider") == provider:
        upstream = entry.get("upstream_model")
        return upstream or None
    return None


def _describe_routing(entry: Any) -> str:
    """Name each provider a model entry routes through, with its upstream."""
    if isinstance(entry, str):
        return entry
    if isinstance(entry, dict) and "providers" in entry:
        return ", ".join(_one_provider(p) for p in entry["providers"])
    if isinstance(entry, dict) and "provider" in entry:
        return _one_provider(
            {"name": entry["provider"], "upstream_model": entry.get("upstream_model")}
        )
    return ""


def _rename_merge_conflict(
    name: str,
    rename_from: str,
    provider: str,
    existing: Any,
    outcome: str,
) -> JSONResponse:
    """409 for a rename-merge that could not add the provider.

    The caller deletes the source model once the merge reports success, so a
    merge that adds nothing must refuse instead of reporting while losing a
    model (#1429).
    """
    if outcome == "skipped":
        reason = (
            f"Model '{name}' already routes through provider '{provider}'"
            f" ({_describe_routing(existing)}); the same provider cannot appear"
            " twice under one model name."
        )
    else:
        routing = _describe_routing(existing) or "unrecognised entry"
        reason = f"Model '{name}' cannot take another provider ({routing})."
    return JSONResponse(
        {
            "error": f"{reason} '{rename_from}' was not merged; both models are unchanged."
        },
        status_code=409,
    )


def _handle_rename_merge(
    models: dict[str, Any],
    name: str,
    rename_from: str,
    body: dict[str, Any],
    provider: str,
) -> JSONResponse | None:
    """Handle the merge branch of a rename collision.

    Returns a 409 JSONResponse if the merge cannot happen or was not
    requested, or ``None`` after performing the merge (caller should save and
    reload).  A 409 leaves both models untouched.
    """
    existing = models[name]
    outcome = _merge_outcome(existing, provider)
    if outcome != "merged":
        return _rename_merge_conflict(name, rename_from, provider, existing, outcome)
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
    if raw_upstream is None:
        # Omitting the field means "keep what the source lane asked for".  The
        # source model's name is only the last resort, for a source that names
        # no upstream at all — it is usually not a model the provider serves.
        upstream = _source_upstream(models[rename_from], provider) or rename_from
    else:
        upstream = raw_upstream
    new_p: dict[str, Any] = {"name": provider, "weight": 1}
    if upstream:
        new_p["upstream_model"] = upstream
    merged = _merge_provider_into_model(
        models, name, new_p, body.get("capabilities", ["text"])
    )
    if merged != "merged":  # defensive: the entry changed shape under us
        return _rename_merge_conflict(name, rename_from, provider, existing, merged)
    del models[rename_from]
    return None
