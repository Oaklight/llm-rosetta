"""Shim sources: declare, reload, and report removal advisories."""

from __future__ import annotations

import logging
import time
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response

from ....shim_sources import apply_shim_sources, resolve_source_config, shims_signature
from .._shared import (
    _get_config_io,
    _install_shim_report,
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
    for key in ("sources", "install"):
        if body.get(key) is not None and not isinstance(body.get(key), list):
            return JSONResponse({"error": f"'{key}' must be an array"}, status_code=400)
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
    for key in (
        "paths",
        "remotes",
        "presets",
        "sources",
        "install",
        "disabled",
        "include",
        "exclude",
    ):
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
    for key in (
        "paths",
        "remotes",
        "presets",
        "sources",
        "install",
        "disabled",
        "include",
        "exclude",
        "cache_dir",
    ):
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


async def _refetch_and_reload(
    request: Any, config_path: str
) -> tuple[Response | None, Any]:
    """Re-fetch every remote source, then rebuild the config.

    A forced refetch is real network I/O, so it runs off the event loop (as
    the normal reload path does) instead of blocking every in-flight request.

    Args:
        request: The admin request.
        config_path: The config file the sources are declared in.

    Returns:
        ``(error_response, None)`` when a step failed, else ``(None, config)``
        with the rebuilt gateway config.
    """
    import asyncio
    import functools

    raw = _get_config_io(request).load_raw(config_path)
    try:
        loop = asyncio.get_running_loop()
        report = await loop.run_in_executor(
            None,
            functools.partial(apply_shim_sources, raw, config_path, force_refresh=True),
        )
    except Exception as exc:
        return JSONResponse(
            {"error": f"Shim reload failed: {exc}"}, status_code=500
        ), None
    _install_shim_report(request, report, shims_signature(raw))

    try:
        new_config = await _reload_gateway_config(
            request, config_path, force_shims=True
        )
    except Exception as exc:
        return JSONResponse(
            {"error": f"Shim reload failed: {exc}"}, status_code=500
        ), None
    return None, new_config


async def post_shims_reload(request: Any) -> Response:
    """Re-apply the declared shim sources.

    An optional ``{"refresh": "always"}`` body forces every remote to be
    re-fetched, bypassing its cache and refresh mode.
    """
    body, _err = parse_json_body(request)
    force_refresh = isinstance(body, dict) and body.get("refresh") == "always"
    config_path = _get_config_path(request)

    if force_refresh:
        failure, new_config = await _refetch_and_reload(request, config_path)
        if failure is not None:
            return failure
    else:
        try:
            new_config = await _reload_gateway_config(
                request, config_path, force_shims=True
            )
        except Exception as exc:
            return JSONResponse(
                {"error": f"Shim reload failed: {exc}"}, status_code=500
            )

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


#: How long a check-updates answer is reused.  One GitHub API call per remote
#: source is cheap but not free: unauthenticated callers get 60 per hour per
#: address, and the settings pane asks on every open.
UPDATE_CHECK_TTL = 300.0


async def post_shims_check_updates(request: Any) -> Response:
    """Report whether each remote source's repository has moved past its pin.

    A remote is pinned to the commit it was fetched at, so this is one HEAD
    lookup per source.  The answer is cached on the app for
    :data:`UPDATE_CHECK_TTL`; ``{"refresh": true}`` forces a new round of
    lookups (the panel's *Check now*).  A source whose head cannot be read
    carries the error and no verdict, so a rate limit reads as "unknown"
    rather than "up to date".
    """
    report = getattr(request.app, "shim_source_report", None)
    if report is None:
        return JSONResponse({"sources": [], "checked_at": 0.0})

    body, _err = parse_json_body(request)
    force = isinstance(body, dict) and bool(body.get("refresh"))
    cached = getattr(request.app, "shim_update_check", None)
    now = time.time()
    if not force and isinstance(cached, dict):
        checked_at = cached.get("checked_at") or 0.0
        if now - float(checked_at) < UPDATE_CHECK_TTL:
            return JSONResponse(cached)

    from ....shims.sources import check_remote_updates

    import asyncio
    import functools

    # The check rides the same transport as the fetch it describes: a
    # deployment that reaches GitHub only through server.proxy would otherwise
    # read "Check failed" on every badge while its shims load fine.
    config = getattr(request.app, "gateway_config", None)
    proxy = getattr(config, "proxy", None)

    loop = asyncio.get_running_loop()
    entries = await loop.run_in_executor(
        None, functools.partial(check_remote_updates, report, proxy=proxy)
    )
    payload = {"sources": entries, "checked_at": now}
    request.app.shim_update_check = payload
    return JSONResponse(payload)


async def post_shim_package_upgrade(request: Any, key: str) -> Response:
    """Move one catalogue package's source to its repository's current head.

    The guard is the server's, not the panel's: the verdict is recomputed here,
    so a request that arrives while the pin still equals the head — a stale
    page, or a hand-built call — changes nothing and returns ``changed:
    false``.  Only an answer that says the head moved re-fetches anything, and
    the refetch is the same forced one the Sources view runs.

    Args:
        request: The admin request.
        key: A ``SHIM_SOURCE_PACKAGES`` key.

    Returns:
        ``{"ok": True, "changed": False, "pin", "head"}`` when there is nothing
        to do; the reload result with ``changed: True`` after a refetch; a
        4xx naming the reason when the package is unknown, not loaded from a
        remote, or its head cannot be read.
    """
    import asyncio
    import functools

    from ....shims.catalog import get_source_package, package_repo_keys, repo_key
    from ....shims.sources import ShimLoadReport

    package = get_source_package(key)
    if package is None:
        return JSONResponse({"error": f"Unknown shim package {key!r}"}, status_code=404)

    report = getattr(request.app, "shim_source_report", None)
    if not isinstance(report, ShimLoadReport):
        return JSONResponse(
            {"error": "the shim load report is not ready", "key": key},
            status_code=409,
        )
    # A package runs either from its repository (a remote, possibly spelled
    # with a renamed repository's old URL) or from its manifest source, as
    # one vendor install per vendor (#1660).
    repos = package_repo_keys(package)
    source = next(
        (
            status
            for status in getattr(report, "sources", []) or []
            if getattr(status, "kind", "") == "remote"
            and (
                (
                    not getattr(status, "vendor", None)
                    and repo_key(getattr(status, "location", "")) in repos
                )
                or (
                    getattr(status, "vendor", None) in package.vendors
                    and getattr(status, "source", None) == package.source
                )
            )
        ),
        None,
    )
    if source is None:
        return JSONResponse(
            {
                "error": f"{package.label} is not loaded from a declared remote",
                "key": key,
            },
            status_code=409,
        )

    from ....shims.sources import check_remote_updates

    config = getattr(request.app, "gateway_config", None)
    proxy = getattr(config, "proxy", None)
    loop = asyncio.get_running_loop()
    entries = await loop.run_in_executor(
        None, functools.partial(check_remote_updates, report, proxy=proxy)
    )
    entry = next(
        (
            e
            for e in entries
            if repo_key(e.get("location")) == repo_key(source.location)
        ),
        None,
    )
    if entry is None:
        return JSONResponse(
            {"error": f"{package.label} has no remote to check", "key": key},
            status_code=409,
        )
    if entry.get("error"):
        return JSONResponse(
            {
                "error": entry["error"],
                "key": key,
                "pin": entry.get("current"),
                "head": entry.get("head"),
            },
            status_code=409,
        )
    if not entry.get("update"):
        # Nothing to do: the pin is the head.  No config write, no refetch.
        return JSONResponse(
            {
                "ok": True,
                "changed": False,
                "key": key,
                "pin": entry.get("current"),
                "head": entry.get("head"),
            }
        )

    failure, new_config = await _refetch_and_reload(request, _get_config_path(request))
    if failure is not None:
        return failure
    # The verdict the panel holds is about the previous pin; drop it so the
    # next read asks again instead of repeating "update available".
    request.app.shim_update_check = None
    return JSONResponse(
        {
            "ok": True,
            "changed": True,
            "key": key,
            "pin": entry.get("current"),
            "head": entry.get("head"),
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
