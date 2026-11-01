"""Health and readiness endpoints."""

from __future__ import annotations

import os
import time
from functools import lru_cache
from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    pass

from llm_rosetta_gateway._vendor.httpserver import (
    JSONResponse,
    Response,
)

from ..logging import get_logger


logger = get_logger()


def version_info() -> dict[str, str]:
    """The gateway and core versions this process runs.

    They are released together (epic #1127) but can drift by a patch when a
    gateway-only fix ships; exposing both lets an operator confirm the pair.
    """
    from llm_rosetta import __version__ as core_version

    from .. import __version__ as gateway_version

    return {"gateway": gateway_version, "core": core_version}


@lru_cache(maxsize=1)
def _known_zones() -> frozenset[str]:
    """The zone names this machine's zone database knows, computed once."""
    try:
        from zoneinfo import available_timezones
    except ImportError:  # pragma: no cover - zoneinfo is 3.9+
        return frozenset()
    try:
        return frozenset(available_timezones())
    except Exception:  # pragma: no cover - a broken tzpath is not fatal here
        return frozenset()


def _tz_name() -> str:
    """The gateway's zone as an IANA name, or its abbreviation.

    The panel shows the name instead of a bare GMT offset when asked to, so
    the operator can tell ``Asia/Shanghai`` from ``Asia/Tokyo``.  The name is
    read from ``TZ`` or the ``/etc/localtime`` link; a process without either
    reports the platform abbreviation (``CST``).

    ``TZ`` is free text, not a zone key: a POSIX rule (``CST-8``) or a path
    (``TZ=:/etc/localtime``) is not a name, and ``/health`` is public, so only
    a value the zone database knows — or the ``/etc/localtime`` target, which
    is one by construction — is reported as the name.
    """
    zones = _known_zones()
    tz = os.environ.get("TZ", "").strip().lstrip(":")
    if "zoneinfo/" in tz:
        tz = tz.rsplit("zoneinfo/", 1)[1]
    if tz and (tz in zones or (not zones and "/" in tz and not tz.startswith("/"))):
        return tz
    try:
        path = os.path.realpath("/etc/localtime")
    except OSError:
        path = ""
    marker = "/zoneinfo/"
    if marker in path:
        return path.split(marker, 1)[1]
    return time.tzname[0]


def _clock_info() -> dict[str, Any]:
    """The gateway's own clock, for the admin panel's footer.

    ``server_time`` is epoch milliseconds, ``server_tz_offset_minutes`` the
    process's UTC offset and ``server_tz_name`` its zone, so the panel can
    render the gateway's wall time without guessing the browser's zone.  All
    three are public facts about the process, not about any provider.
    """
    now = time.time()
    return {
        "server_time": int(now * 1000),
        "server_tz_offset_minutes": int(time.localtime(now).tm_gmtoff // 60),
        "server_tz_name": _tz_name(),
    }


async def handle_health(request: Any) -> Response:
    """Return aggregate health status without exposing provider details.

    Always returns HTTP 200. Use ``status: "degraded"`` in the payload
    to signal provider issues without breaking existing monitors.
    For a 503-on-unhealthy probe use ``/health/ready``.

    Per-provider health details are available via the authenticated
    admin metrics endpoint (``/admin/api/metrics``).
    """
    metrics = getattr(request.app, "metrics", None)
    if metrics is None:
        return JSONResponse(
            {"status": "ok", "versions": version_info(), **_clock_info()}
        )

    snap = metrics.snapshot(series_seconds=3600)
    errors_last_hour = sum(
        pt["errors"] for pt in snap.get("series", []) if pt.get("errors", 0)
    )

    critical = metrics.any_critical_provider()
    overall_status = "degraded" if critical else "ok"

    deferred = getattr(request.app, "deferred_startup", None)
    payload = {
        "status": overall_status,
        "ready": deferred.is_fully_ready() if deferred else True,
        "uptime_seconds": snap["uptime_seconds"],
        "requests_total": snap["total_requests"],
        "errors_last_hour": errors_last_hour,
        "detail_url": "/admin/api/metrics",
        "versions": version_info(),
        **_clock_info(),
    }
    return JSONResponse(payload, status_code=200)


async def handle_health_live(request: Any) -> Response:
    """Kubernetes liveness probe — always 200 while the process is up."""
    return JSONResponse({"status": "ok"})


async def handle_health_ready(request: Any) -> Response:
    """Kubernetes readiness probe — 200 if fully operational, 503 if not.

    Returns 503 when deferred startup tasks are still running (token
    seeding, counter rebuild, backfills) or when provider health has
    degraded critically.

    A deferred step that *failed* does not make this a 503: those steps are
    best-effort repairs, each one safe to retry, and refusing traffic over one
    is the outage the policy exists to avoid.  The ready answer therefore
    carries ``maintenance`` — ``done``, or ``partial`` with the steps that
    failed — so that a probe, or whoever reads the probe, can tell the two
    apart.
    """
    deferred = getattr(request.app, "deferred_startup", None)
    if deferred is not None and not deferred.is_fully_ready():
        return JSONResponse(
            {"status": "not_ready", **deferred.status()},
            status_code=503,
        )

    payload: dict[str, Any] = {}
    if deferred is not None:
        payload["maintenance"] = deferred.maintenance_summary()

    metrics = getattr(request.app, "metrics", None)
    if metrics is None:
        return JSONResponse({"status": "ok", **payload})

    health = metrics.provider_health_snapshot()
    critical_count = sum(1 for v in health.values() if v.get("status") == "critical")
    if critical_count:
        return JSONResponse(
            {"status": "not_ready", "critical_providers": critical_count},
            status_code=503,
        )
    return JSONResponse({"status": "ready", **payload})
