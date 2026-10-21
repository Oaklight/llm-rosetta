"""Health and readiness endpoints."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any


if TYPE_CHECKING:
    pass

from llm_rosetta._vendor.httpserver import (
    JSONResponse,
    Response,
)

from ..logging import get_logger


logger = get_logger()


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
        return JSONResponse({"status": "ok"})

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
