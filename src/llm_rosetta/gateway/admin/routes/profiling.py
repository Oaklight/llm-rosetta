"""On-demand deep profiling route handlers.

Provides admin API handlers for enabling/disabling profiling and
retrieving results.  The data-layer :class:`ProfilerState` lives in
:mod:`llm_rosetta.observability.profiling` and is re-exported here
for backward compatibility.

The :class:`ProfilerState` is instantiated in :func:`~admin.setup_admin`
and attached to ``app.profiler_state``.  Route handlers access it via
``request.app.profiler_state``.  ``app.py`` accesses it the same way
(never imports :class:`ProfilerState` directly).
"""

from __future__ import annotations

from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse, Response
from llm_rosetta.observability.profiling import ProfilerState  # noqa: F401 (re-exported)

from ._shared import _qp


# ---------------------------------------------------------------------------
# Route handlers
# ---------------------------------------------------------------------------


async def get_profiling_status(request: Any) -> Response:
    """Return current profiling status."""
    state: ProfilerState = request.app.profiler_state
    return JSONResponse(state.status())


async def enable_profiling(request: Any) -> Response:
    """Enable profiling for the next N requests."""
    state: ProfilerState = request.app.profiler_state

    # Pre-check: is pyinstrument available?
    try:
        import pyinstrument  # noqa: F401
    except ImportError:
        return JSONResponse(
            {
                "error": "pyinstrument is not installed. "
                "Install with: pip install llm-rosetta[profiling]"
            },
            status_code=400,
        )

    try:
        body = request.json()
    except Exception:
        body = {}
    requests = int(body.get("requests", 5))
    requests = max(1, min(requests, 100))  # clamp to [1, 100]
    return JSONResponse(state.enable(requests))


async def disable_profiling(request: Any) -> Response:
    """Disable profiling."""
    state: ProfilerState = request.app.profiler_state
    return JSONResponse(state.disable())


async def get_profiling_results(request: Any) -> Response:
    """Return profiling results (summaries without HTML/text bodies)."""
    state: ProfilerState = request.app.profiler_state
    summaries = [
        {k: v for k, v in r.items() if k not in ("html", "text")} for r in state.results
    ]
    return JSONResponse({"results": summaries, "total": len(summaries)})


async def get_profiling_result(request: Any, **kwargs: Any) -> Response:
    """Return a single profiling result by index.

    Query param ``format=html`` returns the flamegraph HTML directly
    (with ``text/html`` content type).
    """
    state: ProfilerState = request.app.profiler_state
    try:
        index = int(request.path_params["index"])
    except (ValueError, TypeError, KeyError):
        return JSONResponse({"error": "Invalid index"}, status_code=400)

    if index < 0 or index >= len(state.results):
        return JSONResponse({"error": "Index out of range"}, status_code=404)

    result = state.results[index]

    if _qp(request, "format") == "html":
        return Response(
            body=result["html"].encode("utf-8"),
            status_code=200,
            content_type="text/html; charset=utf-8",
        )

    return JSONResponse(result)


async def download_profiling_results(request: Any) -> Response:
    """Download all profiling results as a ZIP archive."""
    import io
    import zipfile

    state: ProfilerState = request.app.profiler_state
    if not state.results:
        return JSONResponse({"error": "No results to download"}, status_code=404)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for i, r in enumerate(state.results):
            model = r.get("model", "unknown").replace("/", "_").replace(":", "_")
            ts = r.get("timestamp", "")[:19].replace(":", "")
            filename = f"profile-{i}-{model}-{ts}.html"
            zf.writestr(filename, r.get("html", ""))
    buf.seek(0)

    return Response(
        body=buf.getvalue(),
        status_code=200,
        content_type="application/zip",
        headers={"Content-Disposition": "attachment; filename=profiling-results.zip"},
    )


async def clear_profiling_results(request: Any) -> Response:
    """Clear all profiling results."""
    state: ProfilerState = request.app.profiler_state
    state.clear_results()
    return JSONResponse({"ok": True})
