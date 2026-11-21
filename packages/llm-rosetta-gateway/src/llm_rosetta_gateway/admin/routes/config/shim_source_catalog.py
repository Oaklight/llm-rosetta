"""Manifest sources and every vendor they publish (#1660 S9).

``GET /admin/api/config/shims/source-catalog`` feeds the Shims page's
per-source vendor cards and the ``shims.sources`` / ``shims.install`` editor.
The manifests are read through the installer's cache (reused for
``shims.remote_ttl``), so an open panel costs no network call while the cache
is fresh; ``?refresh=1`` forces a new read, as the editor's Re-read does.

When the config reads no source at all, the catalogue sources are listed as
implied (``configured: false``), so the panel can offer their vendors before
the operator selects one.
"""

from __future__ import annotations

import asyncio
import functools
from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response
from llm_rosetta_gateway.admin_api import admin_context

from .._shared import _qp


async def get_shim_source_catalog(request: Any) -> Response:
    """List every manifest source the gateway reads, with all of its vendors.

    Returns:
        ``{"core": {surface, ir, version}, "sources": [...], "install": [...]}``.
        A source whose manifest cannot be read carries ``ok: false`` and the
        error, and no vendors.
    """
    refresh = str(_qp(request, "refresh", "") or "").lower() in ("1", "true", "yes")
    config = getattr(request.app, "gateway_config", None)
    report = getattr(request.app, "shim_source_report", None)
    loop = asyncio.get_running_loop()
    try:
        payload = await loop.run_in_executor(
            None,
            functools.partial(
                admin_context(request).shim_source_catalogue,
                config,
                report,
                refresh=refresh,
            ),
        )
    except Exception as exc:  # noqa: BLE001 - the page must still render
        return JSONResponse(
            {"error": f"Cannot read the shim sources: {exc}"}, status_code=500
        )
    return JSONResponse(payload)
