"""Shim sources: declare, reload, and report removal advisories."""

from __future__ import annotations

import logging
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response

from ....shim_sources import apply_shim_sources, resolve_source_config, shims_signature
from .._shared import (
    _get_config_io,
    _get_config_path,
    _reload_gateway_config,
    config_mutate,
    parse_json_body,
)

logger = logging.getLogger("llm-rosetta-gateway")


def _shim_removal_advisories(request: Any, raw: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the removal advisories for the config as it stands right now.

    Derived per request from the freshly loaded ``raw`` rather than cached on the
    app, so a saved shim source clears the advisory on the very next read
    instead of lingering until a restart.
    """
    from llm_rosetta_gateway.shim_sources import shim_removal_advisories

    report = getattr(request.app, "shim_source_report", None)
    return [
        advisory.to_dict()
        for advisory in shim_removal_advisories(raw.get("providers"), report)
    ]


def _shim_status_payload(request: Any) -> dict[str, Any]:
    """Return the last shim-source report as a JSON-safe dict."""
    report = getattr(request.app, "shim_source_report", None)
    if report is None:
        return {
            "sources": [],
            "total_shims": 0,
            "rebuilt_at": 0.0,
            "failure_count": 0,
        }
    return report.to_dict()


def _validate_shim_sources_body(
    body: dict[str, Any], config_path: str
) -> Response | None:
    """Validate a shim-sources PUT body; return an error response, or ``None``.

    Runs the same resolution the reload path will run, so a body that would
    make ``apply_shim_sources`` raise is rejected *before* it is persisted —
    otherwise the bad section lands on disk and the next startup fails.
    """
    paths = body.get("paths")
    remotes = body.get("remotes")
    presets = body.get("presets")
    disabled = body.get("disabled")
    if paths is not None and not isinstance(paths, list):
        return JSONResponse({"error": "'paths' must be an array"}, status_code=400)
    if remotes is not None and not isinstance(remotes, list):
        return JSONResponse({"error": "'remotes' must be an array"}, status_code=400)
    if presets is not None and not isinstance(presets, list):
        return JSONResponse({"error": "'presets' must be an array"}, status_code=400)
    if disabled is not None and (
        not isinstance(disabled, list) or any(not isinstance(n, str) for n in disabled)
    ):
        return JSONResponse(
            {"error": "'disabled' must be an array of shim names"}, status_code=400
        )
    for entry in remotes or []:
        if not isinstance(entry, dict) or not entry.get("url"):
            return JSONResponse(
                {"error": "each remote source needs a non-empty 'url'"},
                status_code=400,
            )
    try:
        resolve_source_config(body, config_path)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=400)
    return None


def _merge_shim_sources_body(
    section: dict[str, Any], body: dict[str, Any]
) -> dict[str, Any]:
    """Merge a validated PUT body into the ``shims`` section.

    A key absent from *body* is left alone; a present ``null`` for a list key
    also leaves it alone (the panel sends the full state it wants).  Optional
    knobs clear on a falsy value, and empty sub-keys are dropped so the
    section stays tidy.
    """
    for key in ("paths", "remotes", "presets", "disabled"):
        if body.get(key) is not None:
            section[key] = body[key]
    for key, empty in (
        ("cache_dir", ""),
        ("remote_ttl", None),
        ("strict", False),
    ):
        if key not in body:
            continue
        value = body[key]
        if value is None or value == empty:
            section.pop(key, None)
        else:
            section[key] = value
    for key in ("paths", "remotes", "presets", "disabled", "cache_dir"):
        if key in section and not section[key]:
            section.pop(key)
    return section


async def put_shim_sources(request: Any) -> Response:
    """Update the external shim sources (the top-level ``shims`` section)."""
    body, err = parse_json_body(request)
    if err:
        return err
    if not isinstance(body, dict):
        return JSONResponse({"error": "Invalid JSON body"}, status_code=400)

    validation_error = _validate_shim_sources_body(body, _get_config_path(request))
    if validation_error is not None:
        return validation_error

    async with config_mutate(request) as ctx:
        if ctx.error:
            return ctx.error

        section = ctx.data.get("shims")
        if not isinstance(section, dict):
            section = {}
        section = _merge_shim_sources_body(section, body)
        if section:
            ctx.data["shims"] = section
        else:
            ctx.data.pop("shims", None)

        ctx.commit()

    if ctx.error:
        return ctx.error

    return JSONResponse(
        {
            "ok": True,
            "shims": ctx.data.get("shims", {}),
            "status": _shim_status_payload(request),
        }
    )


async def post_shims_reload(request: Any) -> Response:
    """Re-apply the declared shim sources.

    An optional ``{"refresh": "always"}`` body forces every remote to be
    re-fetched, bypassing its cache and refresh mode.
    """
    body, _err = parse_json_body(request)
    force_refresh = isinstance(body, dict) and body.get("refresh") == "always"
    config_path = _get_config_path(request)

    if force_refresh:
        # Apply with a forced refetch, then rebuild the config so provider
        # types pick up any newly registered shims.  A forced refetch is real
        # network I/O, so run it off the event loop (as the normal reload path
        # does) instead of blocking every in-flight request.
        import asyncio
        import functools

        raw = _get_config_io(request).load_raw(config_path)
        try:
            loop = asyncio.get_running_loop()
            report = await loop.run_in_executor(
                None,
                functools.partial(
                    apply_shim_sources, raw, config_path, force_refresh=True
                ),
            )
        except Exception as exc:
            return JSONResponse(
                {"error": f"Shim reload failed: {exc}"}, status_code=500
            )
        request.app.shim_source_report = report
        request.app.shim_source_signature = shims_signature(raw)

    try:
        new_config = await _reload_gateway_config(
            request, config_path, force_shims=True
        )
    except Exception as exc:
        return JSONResponse({"error": f"Shim reload failed: {exc}"}, status_code=500)

    return JSONResponse(
        {
            "ok": True,
            "status": _shim_status_payload(request),
            "provider_warnings": getattr(
                request.app, "provider_validation_warnings", []
            ),
            "providers": list(new_config.providers.keys()),
        }
    )


async def reload_config(request: Any) -> Response:
    """Force hot-reload of the config from disk."""
    config_path = _get_config_path(request)

    try:
        new_config = await _reload_gateway_config(request, config_path)
    except Exception as exc:
        return JSONResponse({"error": f"Reload failed: {exc}"}, status_code=500)

    return JSONResponse(
        {
            "ok": True,
            "providers": list(new_config.providers.keys()),
            "models": list(new_config.models),
            "provider_warnings": getattr(
                request.app, "provider_validation_warnings", []
            ),
        }
    )
