"""Installed extensions (epic #1703).

``GET /admin/api/extensions`` lists what core found and what is installed: the
state and reason of each, the shims and transports it provides, the external
programs it drives and the extensions it still needs.  The listing is read-only;
the one write is ``POST /admin/api/extensions/<id>/install-cli``, which installs
a program the extension declares at the version its manifest pins (#1820).

The extensions themselves are managed here too (#1829): the plan of a removal,
the removal (with the programs the extension installed), the update check
against the manifest sources and the update.  Each write is recorded in the ops
log, and each one that changes what is installed reloads the shims.
"""

from __future__ import annotations

import asyncio
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response
from llm_rosetta_gateway.admin_api import ExtensionInstallRefused, admin_context

from .._shared import _get_config_path, _reload_gateway_config, parse_json_body


async def get_extensions(request: Any) -> Response:
    """List every extension, loaded or not.

    Returns:
        ``{"extensions": [...]}``; each record is documented on
        :meth:`llm_rosetta_gateway.admin_api.AdminExtensions.extension_listing`.
    """
    loop = asyncio.get_running_loop()
    try:
        rows = await loop.run_in_executor(
            None, admin_context(request).extension_listing
        )
    except Exception as exc:  # noqa: BLE001 - the page must still render
        return JSONResponse(
            {"error": f"Cannot list the extensions: {exc}"}, status_code=500
        )
    return JSONResponse({"extensions": rows})


async def post_extension_install_cli(request: Any, ext_id: str) -> Response:
    """Install a program the extension declares, at the version it pins.

    The body may hold only ``program`` (a name; optional when the extension has
    one installable program).  Nothing else is read: no version, URL, checksum
    or command comes from the request.

    Returns:
        ``{"extension", "program", "state", "version", "found"}`` on success;
        ``{"error", "reason"}`` with 400, 404, 409 or 502 otherwise.
    """
    body, error = parse_json_body(request)
    if error is not None:
        return error
    if not isinstance(body, dict) or set(body) - {"program"}:
        return JSONResponse(
            {"error": "Only 'program' may be given", "reason": "bad_body"},
            status_code=400,
        )
    program = body.get("program")
    if program is not None and not isinstance(program, str):
        return JSONResponse(
            {"error": "'program' must be a string", "reason": "bad_body"},
            status_code=400,
        )
    try:
        result = await admin_context(request).install_extension_cli(ext_id, program)
    except ExtensionInstallRefused as exc:
        return JSONResponse(
            {"error": exc.message, "reason": exc.reason}, status_code=exc.status
        )
    return JSONResponse(result)


def _refusal(exc: ExtensionInstallRefused) -> Response:
    return JSONResponse(
        {"error": exc.message, "reason": exc.reason}, status_code=exc.status
    )


def _bad_body(message: str) -> Response:
    return JSONResponse({"error": message, "reason": "bad_body"}, status_code=400)


async def _reload_shims(request: Any) -> str | None:
    """Re-register the shims after the installed extensions changed.

    Returns:
        ``None`` on success, else one sentence: the files changed, so a failed
        reload is reported to the operator, never swallowed.
    """
    try:
        await _reload_gateway_config(
            request, _get_config_path(request), force_shims=True
        )
    except Exception as exc:  # noqa: BLE001 - reported in the answer
        return f"shim reload failed: {exc}"
    return None


async def get_extension_removal_plan(request: Any, ext_id: str) -> Response:
    """Say what removing the extension deletes and keeps, changing nothing.

    Returns:
        The plan (see
        :meth:`llm_rosetta_gateway.admin_api.AdminExtensions.extension_removal_plan`),
        or 404 for an extension that is not installed.
    """
    loop = asyncio.get_running_loop()
    plan = await loop.run_in_executor(
        None, admin_context(request).extension_removal_plan, ext_id
    )
    if plan is None:
        return JSONResponse(
            {"error": f"no extension {ext_id!r}", "reason": "unknown_extension"},
            status_code=404,
        )
    return JSONResponse(plan)


async def post_extension_remove(request: Any, ext_id: str) -> Response:
    """Remove the extension and the programs it installed, then reload the shims.

    The body may hold only ``replace`` (a boolean that marks the removal as the
    first half of a replacement, for the audit record).

    Returns:
        ``{"extension", "deleted", "failed", "reload_error"}`` on success;
        ``{"error", "reason"}`` with 404, 409 or 502 otherwise.
    """
    body, error = parse_json_body(request)
    if error is not None:
        return error
    if not isinstance(body, dict) or set(body) - {"replace"}:
        return _bad_body("Only 'replace' may be given")
    replace = body.get("replace", False)
    if not isinstance(replace, bool):
        return _bad_body("'replace' must be a boolean")
    try:
        result = await admin_context(request).remove_extension(ext_id, replace=replace)
    except ExtensionInstallRefused as exc:
        return _refusal(exc)
    result["reload_error"] = await _reload_shims(request)
    return JSONResponse(result)


async def post_extension_check_updates(request: Any) -> Response:
    """Compare the installed extensions with the manifest sources.

    The body may hold only ``ids`` (a list of extension ids; every extension by
    default).  The sources are read again, not from the cache.

    Returns:
        ``{"updates": [...]}``; see
        :meth:`llm_rosetta_gateway.admin_api.AdminExtensions.check_extension_updates`.
    """
    body, error = parse_json_body(request)
    if error is not None:
        return error
    if not isinstance(body, dict) or set(body) - {"ids"}:
        return _bad_body("Only 'ids' may be given")
    ids = body.get("ids")
    if ids is not None and not (
        isinstance(ids, list) and all(isinstance(i, str) for i in ids)
    ):
        return _bad_body("'ids' must be a list of strings")
    config = getattr(request.app, "gateway_config", None)
    report = getattr(request.app, "shim_source_report", None)
    try:
        updates = await admin_context(request).check_extension_updates(
            config, report, ids
        )
    except Exception as exc:  # noqa: BLE001 - the answer says why, never silent
        return JSONResponse(
            {"error": f"The update check failed: {exc}", "reason": "failed"},
            status_code=502,
        )
    return JSONResponse({"updates": updates})


async def post_extension_update(request: Any, ext_id: str) -> Response:
    """Install the newest release (or ``version``) of the extension, then reload.

    The body may hold only ``version`` (one the extension's source publishes).

    Returns:
        ``{"extension", "before", "after", "reload_error"}`` on success;
        ``{"error", "reason"}`` with 400, 404, 409 or 502 otherwise.
    """
    body, error = parse_json_body(request)
    if error is not None:
        return error
    if not isinstance(body, dict) or set(body) - {"version"}:
        return _bad_body("Only 'version' may be given")
    version = body.get("version")
    if version is not None and not isinstance(version, str):
        return _bad_body("'version' must be a string")
    config = getattr(request.app, "gateway_config", None)
    report = getattr(request.app, "shim_source_report", None)
    try:
        result = await admin_context(request).update_extension(
            ext_id, config, report, version
        )
    except ExtensionInstallRefused as exc:
        return _refusal(exc)
    result["reload_error"] = await _reload_shims(request)
    return JSONResponse(result)
