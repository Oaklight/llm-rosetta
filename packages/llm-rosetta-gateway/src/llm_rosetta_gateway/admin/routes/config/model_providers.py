"""Model-to-provider links: weights and removal."""

from __future__ import annotations

import logging
from collections.abc import Iterator
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response
from llm_rosetta_gateway.config.parse import ENTRY_CAPABILITIES

from .._shared import (
    config_mutate,
    parse_json_body,
)
from .bulk import Lane, lane_key

logger = logging.getLogger("llm-rosetta-gateway")


def _lane_of(item: Any) -> Lane | None:
    """The lane one provider item describes (a bare name has no upstream)."""
    if isinstance(item, str):
        return lane_key(item, None)
    if isinstance(item, dict) and isinstance(item.get("name"), str):
        return lane_key(item["name"], item.get("upstream_model"))
    return None


def _lanes_named(existing: Any, provider: str) -> list[Lane]:
    """Every lane of *provider* the entry lists, in order."""
    if isinstance(existing, str):
        return [lane_key(existing, None)] if existing == provider else []
    if not isinstance(existing, dict):
        return []
    if "providers" in existing and isinstance(existing["providers"], list):
        lanes = [_lane_of(item) for item in existing["providers"]]
        return [lane for lane in lanes if lane is not None and lane[0] == provider]
    if isinstance(existing.get("provider"), str):
        lane = lane_key(existing["provider"], existing.get("upstream_model"))
        return [lane] if lane[0] == provider else []
    return []


def _target_lane(
    name: str, existing: Any, provider: str, upstream: str | None
) -> tuple[Lane | None, Response | None]:
    """The lane a request addresses, or a 409 when the name is not enough.

    A provider may appear on several lanes.  Naming only the provider is
    unambiguous while it appears once; with several lanes the caller has to
    send ``upstream_model`` too, because guessing would edit or remove a lane
    the operator did not mean.
    """
    lanes = _lanes_named(existing, provider)
    if upstream is not None:
        lane = lane_key(provider, upstream)
        return (lane, None) if lane in lanes else (None, None)
    if len(lanes) > 1:
        listed = ", ".join(
            repr(lane[1]) if lane[1] else "no upstream" for lane in lanes
        )
        return None, JSONResponse(
            {
                "error": f"Provider '{provider}' appears on {len(lanes)} lanes of "
                f"model '{name}' ({listed}); send 'upstream_model' to name one"
            },
            status_code=409,
        )
    return (lanes[0], None) if lanes else (None, None)


def _remove_provider_from_entry(
    models: dict[str, Any],
    name: str,
    provider: str,
    upstream: str | None = None,
) -> Response | None:
    """Remove one lane of *provider* from the model entry *name*.

    Returns a :class:`JSONResponse` error on failure, or ``None`` on
    success (caller should commit).  ``upstream`` names the lane when the
    provider appears more than once; without it a multi-lane provider is a
    409 rather than a guess.
    """
    existing = models[name]
    lane, ambiguity = _target_lane(name, existing, provider, upstream)
    if ambiguity is not None:
        return ambiguity

    # Single-provider shorthand (string or dict without "providers")
    if isinstance(existing, str) or (
        isinstance(existing, dict) and "providers" not in existing
    ):
        if lane is None:
            return JSONResponse(
                {"error": f"Provider '{provider}' not found on model '{name}'"},
                status_code=404,
            )
        del models[name]
        return None

    if not isinstance(existing, dict):
        return JSONResponse({"error": "Invalid model entry"}, status_code=400)

    # Multi-provider list: drop that one lane, keep its siblings.
    plist = existing["providers"]
    new_list = [item for item in plist if _lane_of(item) != lane]
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
    # A lane that declares no upstream of its own is served by the entry-level
    # fallback.  The collapsed entry has to carry that value: without it, the
    # surviving lane would quietly start asking for the model's own name, which
    # is the implicit edit this step exists to remove.
    declared = sole.get("upstream_model") if isinstance(sole, dict) else None
    upstream = declared or existing.get("upstream_model")
    if upstream:
        entry["upstream_model"] = upstream
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


def _entry_name(entry: Any) -> str:
    """The provider an entry names, whether it is a string or an object."""
    return entry if isinstance(entry, str) else entry.get("name", "")


def _lane_hidden(entry: Any, raw_providers: dict[str, Any] | None) -> bool:
    """Whether the selector skips this entry, so the board does not show it.

    A lane is off the route when the operator turned the entry off, or when
    the provider behind it is not in the document at all — a disabled provider
    is dropped before the route is built, and the model entry it leaves behind
    is inert.
    """
    if isinstance(entry, dict) and entry.get("enabled") is False:
        return True
    if raw_providers is None:
        return False
    provider = raw_providers.get(_entry_name(entry))
    if provider is None:
        return True
    return isinstance(provider, dict) and provider.get("enabled") is False


def _reorder_request_error(name: str, existing: Any, providers: Any) -> Response | None:
    """The shape checks: the model has a list, and so does the request."""
    if not isinstance(existing, dict) or "providers" not in existing:
        return JSONResponse(
            {"error": "Order is only supported for multi-provider models"},
            status_code=400,
        )
    if not isinstance(providers, list) or not all(
        isinstance(item, str) for item in providers
    ):
        return JSONResponse(
            {"error": "'providers' must be a list of provider names"},
            status_code=400,
        )
    return None


def _permutation_error(
    name: str, providers: list[str], visible_names: set[str]
) -> Response:
    """Refuse a list that is not a permutation of the lanes the board shows."""
    missing = sorted(visible_names - set(providers))
    detail = (
        f"missing {', '.join(repr(item) for item in missing)}"
        if missing
        else "a name appears twice"
    )
    return JSONResponse(
        {
            "error": f"The new order must name every provider of model "
            f"'{name}' once: {detail}"
        },
        status_code=400,
    )


def _reorder_visible(
    existing: dict[str, Any],
    name: str,
    providers: list[str],
    raw: list[Any],
    raw_providers: dict[str, Any] | None,
    by_name: dict[str, Any],
) -> Response | None:
    """Reorder the lanes the board shows, leaving the hidden ones in place."""
    visible = [entry for entry in raw if not _lane_hidden(entry, raw_providers)]
    visible_names = {_entry_name(entry) for entry in visible}
    unknown = [entry for entry in providers if entry not in visible_names]
    if unknown:
        return JSONResponse(
            {"error": f"Provider '{unknown[0]}' is not on model '{name}'"},
            status_code=400,
        )
    # A permutation, not a list: a dropped name would silently remove a lane
    # from the route, which is what the remove-provider route is for.
    if len(providers) != len(visible) or len(set(providers)) != len(providers):
        return _permutation_error(name, providers, visible_names)
    wanted: Iterator[str] = iter(providers)
    existing["providers"] = [
        by_name[next(wanted)] if not _lane_hidden(entry, raw_providers) else entry
        for entry in raw
    ]
    return None


def _reorder_providers(
    models: dict[str, Any],
    name: str,
    providers: Any,
    raw_providers: dict[str, Any] | None = None,
) -> Response | None:
    """Put a model's providers in the order given, or say why not.

    The order is the route the selector reads — the first eligible provider of
    an ``ordered`` model, the tie-break of every other strategy — so it is
    written where the route is declared, not kept in the panel.

    Returns a :class:`JSONResponse` error, or ``None`` on success (the caller
    commits).
    """
    existing = models[name]
    error = _reorder_request_error(name, existing, providers)
    if error is not None:
        return error
    raw = existing["providers"]
    by_name = {_entry_name(entry): entry for entry in raw}
    # Two lists are legal.  The full one names every entry — the Models page's
    # provider menu shows the switched-off ones too — and is written as given.
    # The short one names only the lanes the selector reads, because the
    # Routing board does not show a lane the operator turned off; those keep
    # the slots they already hold.
    if len(providers) == len(by_name) and set(providers) == set(by_name):
        existing["providers"] = [by_name[entry] for entry in providers]
        return None
    return _reorder_visible(existing, name, providers, raw, raw_providers, by_name)


async def reorder_model_providers(request: Any, **kwargs: Any) -> Response:
    """Write a multi-provider model's provider order."""
    name = request.path_params["name"]

    body, err = parse_json_body(request)
    if err:
        return err

    async with config_mutate(request) as ctx:
        if ctx.error:
            return ctx.error

        models = ctx.data.get("models", {})
        if name not in models:
            return JSONResponse({"error": f"Model '{name}' not found"}, status_code=404)

        err_resp = _reorder_providers(
            models, name, body.get("providers"), ctx.data.get("providers", {})
        )
        if err_resp is not None:
            return err_resp

        ctx.commit()

    if ctx.error:
        return ctx.error
    assert ctx.new_config is not None

    return JSONResponse({"ok": True, "model": name, "providers": body.get("providers")})


def _lane_request_fields(
    body: dict[str, Any],
) -> tuple[str, str | None, Response | None]:
    """Read the lane a request addresses: provider, upstream model, error."""
    provider = body.get("provider")
    if not provider:
        return (
            "",
            None,
            JSONResponse({"error": "'provider' is required"}, status_code=400),
        )
    upstream = body.get("upstream_model")
    if upstream is not None and not isinstance(upstream, str):
        return (
            "",
            None,
            JSONResponse(
                {"error": "'upstream_model' must be a string"}, status_code=400
            ),
        )
    return provider, upstream, None


async def remove_model_provider(request: Any, **kwargs: Any) -> Response:
    """Remove a single provider from a multi-provider model entry."""
    name = request.path_params["name"]

    body, err = parse_json_body(request)
    if err:
        return err

    provider, upstream, fields_err = _lane_request_fields(body)
    if fields_err is not None:
        return fields_err

    async with config_mutate(request) as ctx:
        if ctx.error:
            return ctx.error

        models = ctx.data.get("models", {})
        if name not in models:
            return JSONResponse({"error": f"Model '{name}' not found"}, status_code=404)

        err_resp = _remove_provider_from_entry(models, name, provider, upstream)
        if err_resp is not None:
            return err_resp

        ctx.commit()

    if ctx.error:
        return ctx.error
    assert ctx.new_config is not None

    deleted = name not in ctx.new_config.models and name not in models
    result: dict[str, Any] = {
        "ok": True,
        "model": name,
        "removed_provider": provider,
        "removed_upstream_model": upstream,
    }
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


#: Marks "the caller did not send this field", which is not the same as an
#: explicit ``None`` that clears it.
_UNSET: Any = object()


def _set_provider_attrs(
    plist: list[Any],
    provider: str,
    weight: int | None,
    enabled: bool | None,
    capabilities: Any = _UNSET,
    upstream: str | None = None,
) -> bool:
    """Find the addressed lane in *plist* and apply the given attributes.

    A provider may appear on several lanes; ``upstream`` picks one.  When it
    is ``None`` the caller has already established that the provider appears
    once, so the name is enough.

    A provider entry turned off keeps its place in the list with
    ``enabled: false``, so the operator can turn it back on instead of
    re-adding it.  ``enabled`` true removes the key, which is the default.

    Returns True if the lane was found.
    """
    target = lane_key(provider, upstream)
    for i, entry in enumerate(plist):
        lane = _lane_of(entry)
        if lane != target:
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
        if capabilities is not _UNSET:
            # ``None`` or an empty list clears the declaration, which puts the
            # entry back to inheriting the provider's own capabilities.
            if capabilities:
                entry["capabilities"] = list(capabilities)
            else:
                entry.pop("capabilities", None)
        return True
    return False


async def update_model_provider(request: Any, **kwargs: Any) -> Response:
    """Update a provider's attributes (e.g. weight) on a multi-provider model."""
    name = request.path_params["name"]

    body, err = parse_json_body(request)
    if err:
        return err

    provider, upstream, fields_err = _lane_request_fields(body)
    if fields_err is not None:
        return fields_err

    weight, weight_err = _validate_weight(body.get("weight"))
    if weight_err:
        return weight_err

    enabled = body.get("enabled")
    if enabled is not None and not isinstance(enabled, bool):
        return JSONResponse({"error": "'enabled' must be a boolean"}, status_code=400)

    capabilities = body.get("capabilities", _UNSET)
    if capabilities is not _UNSET and not (
        capabilities is None
        or (
            isinstance(capabilities, list)
            and all(isinstance(c, str) for c in capabilities)
        )
    ):
        return JSONResponse(
            {"error": "'capabilities' must be a list of names"}, status_code=400
        )
    # The same vocabulary and the same normalisation the file parser applies,
    # so a value saved here cannot come back as an unknown name later.
    if isinstance(capabilities, list):
        names = [item.strip().lower() for item in capabilities]
        unknown = sorted(set(names) - ENTRY_CAPABILITIES)
        if unknown:
            return JSONResponse(
                {
                    "error": f"Unknown capabilities: {', '.join(unknown)}; "
                    f"known: {', '.join(sorted(ENTRY_CAPABILITIES))}"
                },
                status_code=400,
            )
        capabilities = names

    if weight is None and enabled is None and capabilities is _UNSET:
        return JSONResponse(
            {"error": "One of 'weight', 'enabled' or 'capabilities' is required"},
            status_code=400,
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

        lane, ambiguity = _target_lane(name, existing, provider, upstream)
        if ambiguity is not None:
            return ambiguity
        if lane is None or not _set_provider_attrs(
            existing["providers"],
            provider,
            weight,
            enabled,
            capabilities,
            upstream=lane[1],
        ):
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
            "upstream_model": upstream,
            "weight": weight,
            "enabled": enabled,
            "capabilities": (None if capabilities is _UNSET else capabilities),
        }
    )
