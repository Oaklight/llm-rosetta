"""The console's update probe and its self-update action.

Three routes: what PyPI offers (``GET /admin/api/updates``), the changelog
bundled with this build (``GET /admin/api/changelog``), and the install that
follows a click (``POST /admin/api/updates/apply``).  The install is the only
route in the panel that changes the running code, so every refusal happens
before pip starts and a failed install leaves the old process serving.
"""

from __future__ import annotations

import asyncio
from typing import Any

from llm_rosetta_gateway import updates
from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response
from llm_rosetta_gateway.ops.security import OpsSelfUpdate

from ._shared import _qp, parse_json_body

#: One install at a time: two clicks must not race two pip runs into the same
#: site-packages, and the second click gets a 409 rather than a queue.
_install_lock = asyncio.Lock()


def _config(request: Any) -> Any:
    return getattr(request.app, "config", None)


async def get_updates(request: Any) -> Response:
    """Report the running versions and the newest release PyPI lists."""
    config = _config(request)
    if config is not None and not getattr(config, "update_check", True):
        return JSONResponse({"checked": False, "reason": "disabled"})
    force = (_qp(request, "force") or "") in ("1", "true")
    payload = await updates.check(getattr(config, "proxy", None), force=force)
    return JSONResponse(payload)


async def get_changelog(request: Any) -> Response:
    """Return the changelog bundled with this build, as markdown.

    A build without the file answers with an empty string; the panel then
    hides the "what changed" entry rather than showing an empty dialog.
    """
    language = _qp(request, "lang") or "en"
    return JSONResponse({"markdown": updates.read_changelog(language)})


async def apply_update(request: Any) -> Response:
    """Install the offered release and restart the process into it.

    The body names the version the panel was offered; a version that is not
    the one on offer is refused, so the endpoint cannot be used to install an
    arbitrary release.  On success the response is written first and the
    process image is replaced 0.8 s later; the panel then polls ``/health``
    until the new version answers.
    """
    config = _config(request)
    if config is not None and not getattr(config, "self_update", True):
        return JSONResponse(
            {"error": "Self-update is disabled", "reason": "disabled"},
            status_code=403,
        )
    body, error = parse_json_body(request)
    if error is not None:
        return error
    if _install_lock.locked():
        return JSONResponse(
            {"error": "An update is already running", "reason": "busy"},
            status_code=409,
        )
    proxy = getattr(config, "proxy", None)
    async with _install_lock:
        status = await updates.check(proxy)
        offered = status.get("update") or {}
        offered_version = offered.get("version")
        requested = body.get("version")
        if not isinstance(offered_version, str) or not offered_version:
            return JSONResponse(
                {"error": "No update is on offer", "reason": "no_update"},
                status_code=409,
            )
        if requested and requested != offered_version:
            # The body may only confirm what the probe offered; anything else
            # would let the endpoint install an arbitrary release.
            return JSONResponse(
                {"error": "No update is on offer", "reason": "no_update"},
                status_code=409,
            )
        target = offered_version
        gate = status.get("self_update") or {}
        if not gate.get("allowed"):
            return JSONResponse(
                {
                    "error": "This process cannot update itself",
                    "reason": gate.get("reason") or "refused",
                },
                status_code=409,
            )
        result = await updates.install(target, proxy)
        if not result["ok"]:
            return JSONResponse(
                {"status": "failed", "output": result["output"]},
                status_code=502,
            )
        ops_ctx = getattr(request.app, "ops_ctx", None)
        if ops_ctx is not None:
            await OpsSelfUpdate(ops_ctx, version=target).execute()
        # The browser needs the response before the interpreter it is talking
        # to is replaced; half a second is one poll interval, not a timeout.
        asyncio.get_running_loop().call_later(0.8, updates.restart_process)
        return JSONResponse(
            {
                "status": "restarting",
                "version": target,
                "dists": result["dists"],
            }
        )
