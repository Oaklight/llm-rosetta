"""Bulk add: merge providers into model entries, then create them."""

from __future__ import annotations

import logging
from typing import Any

from llm_rosetta._vendor.httpclient import AsyncClient, Response as HttpResponse
from llm_rosetta._vendor.httpserver import JSONResponse, Response
from llm_rosetta.shims import get_shim

from .._shared import (
    _resolve_models_path,
    config_mutate,
    parse_json_body,
)

from .payload import _get_gateway_config

logger = logging.getLogger("llm-rosetta-gateway")


def _format_connection_error(exc: Exception, url: str) -> str:
    """Return a user-friendly message for common upstream connection errors."""
    err_str = str(exc)
    if "Connection refused" in err_str or "Errno 111" in err_str:
        return (
            f"Connection refused at {url}. "
            "Check that the service is running and the port is correct. "
            "If running in Docker, ensure the host firewall (e.g. ufw) "
            "allows connections from the Docker bridge network."
        )
    if "timed out" in err_str.lower():
        return (
            f"Connection to {url} timed out. "
            "Check that the host/port is reachable from this container."
        )
    if "Name or service not known" in err_str or "getaddrinfo" in err_str:
        return f"Cannot resolve hostname in {url}. Check the Base URL."
    return f"Failed to connect to upstream: {err_str}"


def _extract_model_ids(
    body: dict[str, Any] | list[dict[str, Any]],
    ptype: str,
    shim_name: str | None,
    id_field: str | None,
) -> tuple[list[str], dict[str, str]]:
    """Extract model IDs from an upstream ``/models`` response body.

    Returns ``(model_ids, upstream_map)``.  Handles three response shapes:

    * OpenAI-style ``{data: [{id: ...}, ...]}``
    * Google-style ``{models: [{name: ...}, ...]}``
    * Bare JSON array ``[{id: ...}, ...]`` (e.g. ALCF) — requires a
      ``model_list_transform``; without one the array is silently ignored.

    When a shim provides a ``model_list_transform`` hook, it is used to
    derive human-readable slugs and the mapping back to internal upstream
    IDs.
    """
    from llm_rosetta.shims.providers import get_model_list_transform

    mlt = get_model_list_transform(shim_name) if shim_name else None

    upstream_map: dict[str, str] = {}
    model_ids: list[str] = []

    if mlt is not None:
        if isinstance(body, list):
            raw_entries: list[dict[str, Any]] = body
        else:
            raw_entries = (
                body.get("data", []) if "data" in body else body.get("models", [])
            )
        model_ids, upstream_map = mlt(raw_entries)
    elif isinstance(body, list):
        logger.debug(
            "Upstream returned bare array but no model_list_transform is registered "
            "for shim %r — cannot extract model IDs",
            shim_name,
        )
        return [], {}
    elif ptype in ("google", "google_generate"):
        for m in body.get("models", []):
            name = m.get("name", "")
            if name.startswith("models/"):
                name = name[len("models/") :]
            model_ids.append(m.get(id_field, name) if id_field else name)
    else:
        for m in body.get("data", []):
            model_ids.append(
                m.get(id_field, m.get("id", "")) if id_field else m.get("id", "")
            )

    model_ids = [m for m in model_ids if m]
    model_ids.sort()
    return model_ids, upstream_map


async def fetch_upstream_models(request: Any, **kwargs: Any) -> Response:
    """Fetch the model list from an upstream provider's /v1/models endpoint."""

    provider_name = request.path_params["name"]
    config = _get_gateway_config(request)
    if config is None:
        return JSONResponse({"error": "Gateway config not loaded"}, status_code=500)

    if provider_name not in config.providers:
        return JSONResponse(
            {"error": f"Provider '{provider_name}' not found"}, status_code=404
        )

    pinfo = config.providers[provider_name]
    ptype = config.provider_types.get(provider_name, "unknown")

    # A token_command provider whose initial fetch has not completed still
    # holds the pending sentinel.  Sending it would come back as an upstream
    # 401 that reads as "this provider may not support model listing", so
    # report the real state instead.
    if not pinfo.ready:
        return JSONResponse(
            {
                "error": (
                    "Provider is still starting up — its API key has not been "
                    "fetched yet. Retry in a moment."
                )
            }
        )

    # Build the models listing URL.  Explicit ``models_path`` in the
    # provider config takes precedence (e.g. ``/v1/models`` for Jina).
    raw_cfg = getattr(config, "_raw_providers", {}).get(provider_name, {})
    explicit_path = _resolve_models_path(raw_cfg, config, provider_name)
    if explicit_path:
        if explicit_path.startswith(("https://", "http://")):
            models_url = explicit_path
        else:
            models_url = f"{pinfo.base_url}{explicit_path}"
    elif ptype in ("google", "google_generate"):
        models_url = f"{pinfo.base_url}/v1beta/models"
    elif ptype == "messages":
        models_url = f"{pinfo.base_url}/v1/models"
    else:
        models_url = f"{pinfo.base_url}/models"

    headers = {**pinfo.extra_headers, **pinfo.auth_headers()}

    try:
        client = AsyncClient(timeout=10.0, proxy=pinfo.proxy_url)
        raw_resp = await client.get(models_url, headers=headers)
        assert isinstance(raw_resp, HttpResponse), "Expected non-streaming response"
        resp: HttpResponse = raw_resp
        await client.aclose()
    except Exception as exc:
        logger.warning("Failed to fetch models from %s: %s", provider_name, exc)
        msg = _format_connection_error(exc, models_url)
        return JSONResponse({"error": msg})  # 200 so reverse proxies don't intercept

    if resp.status_code >= 400:
        logger.warning(
            "Upstream %s returned %d for model listing", provider_name, resp.status_code
        )
        return JSONResponse(
            {
                "error": (
                    f"Upstream returned HTTP {resp.status_code}. "
                    "This provider may not support model listing."
                ),
            },
        )

    try:
        body = resp.json()
    except Exception:
        return JSONResponse(
            {"error": "Upstream returned non-JSON response"},
        )

    # Resolve model_id_field from shim (e.g. "internal_id" instead of "model")
    shim_name = config.provider_shim_names.get(provider_name)
    shim = get_shim(shim_name) if shim_name else None
    id_field = (
        shim.connection.model_id_field
        if shim and shim.connection.model_id_field
        else None
    )

    model_ids, upstream_map = _extract_model_ids(body, ptype, shim_name, id_field)

    result: dict[str, Any] = {
        "provider": provider_name,
        "api_standard": ptype,
        "models": model_ids,
    }
    if upstream_map:
        result["upstream_map"] = upstream_map
    return JSONResponse(result)


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
