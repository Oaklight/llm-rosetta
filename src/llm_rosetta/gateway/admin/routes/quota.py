"""Provider quota reading handler.

One endpoint, ``GET /admin/api/config/providers/<name>/quota``.  It answers
with the normalized reading of every credential the provider holds; the raw
keys never leave the server.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response
from llm_rosetta.observability.sanitize import scrub_credential_patterns

from ._shared import _qp
from .config.payload import _get_gateway_config


def _wants_refresh(request: Any) -> bool:
    """Whether the caller asked to bypass the reading cache."""
    value = _qp(request, "refresh", "0")
    return (value or "").lower() not in ("", "0", "false", "no")


async def get_provider_quota(request: Any, name: str) -> Response:
    """Return the quota of every credential of *name*.

    Query parameters:
        refresh: Any truthy value bypasses the cache and re-probes.
    """
    config = _get_gateway_config(request)
    if config is None:
        return JSONResponse({"error": "No config loaded"}, status_code=500)
    if name not in getattr(config, "providers", {}):
        return JSONResponse({"error": f"Provider '{name}' not found"}, status_code=404)

    from ...quota.service import read_provider_quota

    try:
        payload = await read_provider_quota(config, name, force=_wants_refresh(request))
    except Exception as exc:  # noqa: BLE001 - report, never 500 the panel
        return JSONResponse(
            {
                "provider": name,
                "supported": True,
                "status": "error",
                "error": scrub_credential_patterns(str(exc)),
            },
            status_code=200,
        )
    return JSONResponse(payload)
