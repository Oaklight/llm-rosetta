"""In-memory content capture route handlers.

Provides admin API handlers for enabling/disabling content capture and
retrieving captured request/response flows.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response
from llm_rosetta.observability.capture import CaptureState  # noqa: F401


async def get_capture_status(request: Any) -> Response:
    """Return current capture status."""
    state: CaptureState = request.app.capture_state
    return JSONResponse(state.status())


async def enable_capture(request: Any) -> Response:
    """Enable content capture for the next N requests."""
    state: CaptureState = request.app.capture_state
    try:
        body = request.json()
    except Exception:
        body = {}
    requests = int(body.get("requests", 5))
    requests = max(1, min(requests, 100))  # clamp to [1, 100]
    return JSONResponse(state.enable(requests))


async def disable_capture(request: Any) -> Response:
    """Disable content capture."""
    state: CaptureState = request.app.capture_state
    return JSONResponse(state.disable())


async def get_capture_results(request: Any) -> Response:
    """Return capture result summaries (no bodies)."""
    state: CaptureState = request.app.capture_state
    summaries = [r.summary() for r in state.results]
    return JSONResponse({"results": summaries, "total": len(summaries)})


async def get_capture_result(request: Any, **kwargs: Any) -> Response:
    """Return a single capture result by index (full detail)."""
    state: CaptureState = request.app.capture_state
    try:
        index = int(request.path_params["index"])
    except (ValueError, TypeError, KeyError):
        return JSONResponse({"error": "Invalid index"}, status_code=400)

    if index < 0 or index >= len(state.results):
        return JSONResponse({"error": "Index out of range"}, status_code=404)

    return JSONResponse(state.results[index].to_dict())


async def clear_capture_results(request: Any) -> Response:
    """Clear all capture results."""
    state: CaptureState = request.app.capture_state
    state.clear_results()
    return JSONResponse({"ok": True})
