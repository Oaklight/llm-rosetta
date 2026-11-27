"""Installed extensions (epic #1703).

``GET /admin/api/extensions`` lists what core found and what is installed: the
state and reason of each, the shims and transports it provides, the external
programs it drives and the extensions it still needs.  The listing is read-only;
the one write is ``POST /admin/api/extensions/<id>/install-cli``, which installs
a program the extension declares at the version its manifest pins (#1820).
Installing and removing the extensions themselves stay on the manifest sources
and the command line.
"""

from __future__ import annotations

import asyncio
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response
from llm_rosetta_gateway.admin_api import ExtensionInstallRefused, admin_context

from .._shared import parse_json_body


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
