"""Provider CRUD: create, update, delete, toggle, rename-merge."""

from __future__ import annotations

import copy
import logging
import os
import shutil
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response
from llm_rosetta.api import get_shim

from .._shared import (
    _build_provider_entry,
    _config_preflight_error,
    _handle_provider_rename,
    _shim_ships_credential_command,
    _shim_ships_credential_exchange,
    config_mutate,
    parse_json_body,
)
from .payload import _get_gateway_config

from .bulk import Lane, _merge_outcome, _merge_provider_into_model, lane_key
from .model_providers import _lanes_named, _remove_all_lanes_of_provider

logger = logging.getLogger("llm-rosetta-gateway")


def _owned_profile_paths(
    name: str, entry: dict[str, Any], data_dir: str | None, transport: str
) -> dict[str, str]:
    """The profile directories the gateway owns for one entry.

    A path named by the operator (``acp.home``, ``--gemini-dir``) is never
    included: lifecycle actions must not move or delete it.
    """
    from ....config.providers import managed_acp_state_dir, managed_auth_state_dir

    paths: dict[str, str] = {}
    if transport == "acp":
        home = managed_acp_state_dir(name, entry, data_dir)
        if home:
            paths["acp"] = home
    auth = managed_auth_state_dir(name, data_dir)
    if auth:
        paths["auth"] = auth
    return paths


def _remove_profile(path: str, report: dict[str, Any]) -> None:
    if not os.path.isdir(path):
        return
    try:
        shutil.rmtree(path)
    except OSError as exc:
        report["warnings"].append(f"could not remove the old profile {path!r}: {exc}")
        return
    report["removed"].append(path)


def _relocate_profiles(
    *,
    old_name: str,
    old_entry: dict[str, Any],
    old_transport: str,
    new_name: str | None,
    new_entry: dict[str, Any] | None,
    new_transport: str,
    data_dir: str | None,
) -> dict[str, Any]:
    """Move a renamed entry's profile, or remove a deleted entry's.

    Only gateway-managed directories are touched.  A rename keeps the login
    when the new entry still uses an ACP lane and names no profile of its
    own; otherwise the old profile is removed, because nothing may keep using
    a login the entry no longer owns.
    """
    report: dict[str, Any] = {"moved": [], "removed": [], "warnings": []}
    old_paths = _owned_profile_paths(old_name, old_entry, data_dir, old_transport)
    if new_name is not None and new_entry is not None:
        new_paths = _owned_profile_paths(new_name, new_entry, data_dir, new_transport)
        for kind, old_path in old_paths.items():
            new_path = new_paths.get(kind)
            if not os.path.isdir(old_path):
                continue
            if not new_path:
                _remove_profile(old_path, report)
                continue
            if os.path.abspath(old_path) == os.path.abspath(new_path):
                continue
            if os.path.exists(new_path):
                report["warnings"].append(
                    f"the new profile {new_path!r} already exists; the old "
                    f"profile {old_path!r} was left in place"
                )
                continue
            try:
                os.makedirs(os.path.dirname(new_path), exist_ok=True)
                shutil.move(old_path, new_path)
            except OSError as exc:
                report["warnings"].append(
                    f"could not move the profile {old_path!r} to {new_path!r}: {exc}"
                )
                continue
            report["moved"].append({"from": old_path, "to": new_path})
    else:
        for old_path in old_paths.values():
            _remove_profile(old_path, report)
    return report


async def _close_lane_processes(request: Any, names: list[str]) -> None:
    """Drop the agent processes bound to a renamed or deleted entry."""
    transports = list((getattr(request.app, "transports", {}) or {}).values())
    seen: set[int] = set()
    for transport in transports:
        if id(transport) in seen:
            continue
        seen.add(id(transport))
        close = getattr(transport, "aclose_provider", None)
        if not callable(close):
            continue
        for name in names:
            try:
                await close(name)
            except Exception:  # noqa: BLE001 - the config change already won
                logger.debug("could not close the lane for %r", name, exc_info=True)


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
    from ....config.typed import face_value
    from ....model_types import all_model_types

    shims = [get_shim(body.get("type") or resolve_name)]
    for desc in all_model_types():
        if desc.is_llm or not desc.config_shim_key:
            continue
        # The endpoint block first, the flat key after it — the runtime's own
        # order.  Reading only the flat key made an entry the v5 fold had
        # already moved unsaveable: the shim that supplies its connection was
        # no longer where this looked, and the save answered 400 asking for a
        # base_url the operator had never typed (#1595 review).
        named = face_value(body, desc.config_shim_key, desc.name)
        shims.append(get_shim(named or ""))
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

    current_config = _get_gateway_config(request)
    resolved_data_dir = getattr(current_config, "data_dir", None)
    current_pinfo = (
        current_config.providers.get(body.get("rename_from") or name)
        if current_config is not None
        else None
    )

    async with config_mutate(request) as ctx:
        if ctx.error:
            return ctx.error

        existing_providers = ctx.data.get("providers", {})
        resolve_name = body.get("rename_from", name) or name
        old_entry = copy.deepcopy(existing_providers.get(resolve_name) or {})
        old_transport = (
            getattr(current_pinfo, "transport", None)
            or old_entry.get("transport")
            or "http"
        )

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

        try:
            provider_entry = _build_provider_entry(
                body, api_key, base_url, existing_providers, resolve_name
            )
        except ValueError as exc:
            # A declared setup input the entry cannot hold: refused before it
            # reaches the config file, with the field named.
            return JSONResponse({"error": str(exc)}, status_code=400)

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
        load_error = _config_preflight_error(ctx.data, resolved_data_dir)
        if load_error is not None:
            return JSONResponse({"error": load_error}, status_code=400)

        ctx.commit()

    if ctx.error:
        return ctx.error
    assert ctx.new_config is not None

    renamed = bool(body.get("rename_from") and body.get("rename_from") != name)
    profile_report: dict[str, Any] | None = None
    if renamed:
        data_dir = getattr(ctx.new_config, "data_dir", None) or resolved_data_dir
        new_entry = copy.deepcopy(ctx.data.get("providers", {}).get(name) or {})
        new_pinfo = ctx.new_config.providers.get(name)
        profile_report = _relocate_profiles(
            old_name=resolve_name,
            old_entry=old_entry,
            old_transport=old_transport,
            new_name=name,
            new_entry=new_entry,
            new_transport=getattr(new_pinfo, "transport", "http") or "http",
            data_dir=data_dir,
        )
        await _close_lane_processes(request, [resolve_name, name])

    result: dict[str, Any] = {
        "ok": True,
        "provider": name,
        "providers": list(ctx.new_config.providers.keys()),
        "provider_warnings": ctx.warnings,
    }
    if profile_report is not None:
        result["profile"] = profile_report
    return JSONResponse(result)


def _models_naming(models: dict[str, Any], provider: str) -> list[str]:
    """The model entries that route through *provider* on any lane.

    Reads the same lane rule the cascade removes with, so the refusal and the
    cascade can never disagree about which entries the delete touches.
    """
    return [
        model_name for model_name, cfg in models.items() if _lanes_named(cfg, provider)
    ]


async def delete_provider(request: Any, **kwargs: Any) -> Response:
    """Remove a provider entry."""
    name = request.path_params["name"]

    current_config = _get_gateway_config(request)
    resolved_data_dir = getattr(current_config, "data_dir", None)
    current_pinfo = (
        current_config.providers.get(name) if current_config is not None else None
    )

    async with config_mutate(request) as ctx:
        if ctx.error:
            return ctx.error

        providers = ctx.data.get("providers", {})
        if name not in providers:
            return JSONResponse(
                {"error": f"Provider '{name}' not found"}, status_code=404
            )

        old_entry = copy.deepcopy(providers[name])
        old_transport = (
            getattr(current_pinfo, "transport", None)
            or old_entry.get("transport")
            or "http"
        )

        # Check if any model still references this provider
        models = ctx.data.get("models", {})
        referencing = _models_naming(models, name)

        from .._shared import _qp

        cascade = _qp(request, "cascade") in ("true", "1")
        if referencing and not cascade:
            return JSONResponse(
                {
                    "error": f"Cannot delete provider '{name}': referenced by models: {referencing}"
                },
                status_code=409,
            )

        # Cascade: drop this provider's lanes, keep the models that others serve
        cascade_deleted: list[str] = []
        cascade_trimmed: list[str] = []
        if referencing and cascade:
            for model_name in referencing:
                outcome = _remove_all_lanes_of_provider(models, model_name, name)
                if outcome == "deleted":
                    cascade_deleted.append(model_name)
                elif outcome == "trimmed":
                    cascade_trimmed.append(model_name)

        del providers[name]
        ctx.commit()

    if ctx.error:
        return ctx.error
    assert ctx.new_config is not None

    data_dir = getattr(ctx.new_config, "data_dir", None) or resolved_data_dir
    profile_report = _relocate_profiles(
        old_name=name,
        old_entry=old_entry,
        old_transport=old_transport,
        new_name=None,
        new_entry=None,
        new_transport="http",
        data_dir=data_dir,
    )
    await _close_lane_processes(request, [name])

    result: dict[str, Any] = {
        "ok": True,
        "deleted": name,
        "providers": list(ctx.new_config.providers.keys()),
        "profile": profile_report,
    }
    if cascade_deleted:
        result["cascade_deleted_models"] = cascade_deleted
    if cascade_trimmed:
        result["cascade_trimmed_models"] = cascade_trimmed
    return JSONResponse(result)


async def toggle_provider(request: Any, **kwargs: Any) -> Response:
    """Toggle a provider's enabled/disabled state."""
    name = request.path_params["name"]
    current_config = _get_gateway_config(request)
    resolved_data_dir = getattr(current_config, "data_dir", None)

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
            load_error = _config_preflight_error(ctx.data, resolved_data_dir)
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
    lane: Lane,
) -> JSONResponse:
    """409 for a rename-merge that could not add the lane.

    The caller deletes the source model once the merge reports success, so a
    merge that adds nothing must refuse instead of reporting while losing a
    model (#1429).  A provider may serve several upstream models, so only the
    same ``(provider, upstream_model)`` pair twice is refused.
    """
    if outcome == "skipped":
        upstream = lane[1]
        target = (
            f"provider '{provider}' to upstream '{upstream}'"
            if upstream
            else f"provider '{provider}'"
        )
        reason = (
            f"Model '{name}' already routes {target}"
            f" ({_describe_routing(existing)}); that lane cannot appear twice"
            " under one model name."
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
    # The lane this merge would add is decided once, here, and the same value
    # is judged and then written — the pre-check and the mutation cannot
    # disagree about which upstream model the new lane asks for.
    raw_upstream = body.get("upstream_model")
    if raw_upstream is None:
        # Omitting the field means "keep what the source lane asked for".  The
        # source model's name is only the last resort, for a source that names
        # no upstream at all — it is usually not a model the provider serves.
        upstream = _source_upstream(models[rename_from], provider) or rename_from
    else:
        upstream = raw_upstream
    lane = lane_key(provider, upstream)
    outcome = _merge_outcome(existing, lane)
    if outcome != "merged":
        return _rename_merge_conflict(
            name, rename_from, provider, existing, outcome, lane
        )
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
    new_p: dict[str, Any] = {"name": provider, "weight": 1}
    if upstream:
        new_p["upstream_model"] = upstream
    merged = _merge_provider_into_model(
        models, name, new_p, body.get("capabilities", ["text"])
    )
    if merged != "merged":  # defensive: the entry changed shape under us
        return _rename_merge_conflict(
            name, rename_from, provider, existing, merged, lane
        )
    del models[rename_from]
    return None
