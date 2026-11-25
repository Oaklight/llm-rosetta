"""Installed extensions (epic #1703).

``GET /admin/api/extensions`` lists what core found and what is installed: the
state and reason of each, the shims and transports it provides, the external
programs it drives and the extensions it still needs.  Read-only: installing and
removing stay on the manifest sources and the command line.
"""

from __future__ import annotations

import asyncio
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response
from llm_rosetta_gateway.admin_api import admin_context


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
