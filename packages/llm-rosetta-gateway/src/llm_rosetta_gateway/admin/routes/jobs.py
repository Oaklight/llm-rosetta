"""Background job routes: list, inspect, cancel, retry, resume, unhide."""

from __future__ import annotations

from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response

from ._shared import _qp


def _manager(request: Any) -> Any:
    """The job manager, or ``None`` when this app has no persistence."""
    return getattr(request.app, "jobs", None)


def _unavailable() -> Response:
    return JSONResponse(
        {"error": "Background jobs require persistence"}, status_code=400
    )


def _job_id(request: Any, kwargs: dict[str, Any]) -> str:
    return kwargs.get("job_id") or request.path_params["job_id"]


async def get_jobs(request: Any) -> Response:
    """List jobs newest first, with the active count for the tab badge."""
    manager = _manager(request)
    if manager is None:
        return _unavailable()
    try:
        limit = int(_qp(request, "limit", "50"))
        offset = int(_qp(request, "offset", "0"))
    except ValueError:
        return JSONResponse(
            {"error": "limit and offset must be integers"}, status_code=400
        )
    jobs, total = await manager.store.list_jobs(
        status=_qp(request, "status") or None,
        kind=_qp(request, "kind") or None,
        limit=max(1, min(limit, 200)),
        offset=max(0, offset),
    )
    return JSONResponse(
        {
            "jobs": [job.to_payload() for job in jobs],
            "total": total,
            "active": await manager.store.active_count(),
        }
    )


async def get_job(request: Any, **kwargs: Any) -> Response:
    """Return one job, including its result or error."""
    manager = _manager(request)
    if manager is None:
        return _unavailable()
    job = await manager.store.get(_job_id(request, kwargs))
    if job is None:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return JSONResponse(job.to_payload())


async def cancel_job(request: Any, **kwargs: Any) -> Response:
    """Cancel a queued job, or ask a running one to stop at its checkpoint."""
    manager = _manager(request)
    if manager is None:
        return _unavailable()
    found, status = await manager.cancel(_job_id(request, kwargs))
    if not found:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    return JSONResponse({"ok": True, "status": status})


async def retry_job(request: Any, **kwargs: Any) -> Response:
    """Queue a failed or cancelled job again from the start."""
    manager = _manager(request)
    if manager is None:
        return _unavailable()
    ok = await manager.store.retry(_job_id(request, kwargs))
    if not ok:
        return JSONResponse(
            {"error": "Only a failed or cancelled job can be retried"},
            status_code=409,
        )
    return JSONResponse({"ok": True, "status": "queued"})


async def resume_job(request: Any, **kwargs: Any) -> Response:
    """Continue a cancelled job, keeping the rows it has not deleted yet."""
    manager = _manager(request)
    if manager is None:
        return _unavailable()
    ok = await manager.store.resume(_job_id(request, kwargs))
    if not ok:
        return JSONResponse(
            {"error": "Only a cancelled job can be resumed"}, status_code=409
        )
    return JSONResponse({"ok": True, "status": "queued"})


async def unhide_job(request: Any, **kwargs: Any) -> Response:
    """Make a job's still-hidden rows visible again."""
    manager = _manager(request)
    if manager is None:
        return _unavailable()
    try:
        removed = await manager.store.unhide(_job_id(request, kwargs))
    except KeyError:
        return JSONResponse({"error": "Job not found"}, status_code=404)
    except ValueError as exc:
        return JSONResponse({"error": str(exc)}, status_code=409)
    return JSONResponse({"ok": True, "ranges_removed": removed})
