"""Async model test task route handlers."""

from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any

from llm_rosetta._vendor.httpclient import AsyncClient, Response as HttpResponse
from llm_rosetta._vendor.httpserver import JSONResponse, Response

from ...config import GatewayConfig

# In-memory store: task_id → {status, result, asyncio_task, started, ...}
_test_tasks: dict[str, dict[str, Any]] = {}
_MAX_TASK_AGE = 300  # seconds — auto-cleanup threshold
_TASK_TIMEOUT = 120  # seconds — per-task execution timeout


def _cleanup_stale_tasks() -> None:
    """Remove tasks older than _MAX_TASK_AGE."""
    now = time.monotonic()
    expired = [
        tid
        for tid, t in _test_tasks.items()
        if now - t.get("started", 0) > _MAX_TASK_AGE
    ]
    for tid in expired:
        task = _test_tasks.pop(tid, None)
        if task and not task.get("_task_obj", asyncio.Future()).done():
            task["_task_obj"].cancel()


async def _run_test_task(
    task_id: str,
    endpoint: str,
    payload: dict[str, Any],
    internal_token: str,
    proxy_url: str | None,  # noqa: ARG001 — reserved for future use
) -> None:
    """Execute an upstream test request via a self-call to the gateway.

    Results are stored in ``_test_tasks[task_id]``.  This runs as an
    ``asyncio.Task`` so the admin API handler can return immediately.

    Important: we create a **dedicated** ``AsyncClient`` per task instead
    of reusing the shared pool from ``proxy.get_client()``.  The shared
    client serialises non-streaming requests with an ``asyncio.Lock``,
    which would deadlock when the self-call triggers the proxy handler
    that itself needs the same lock to call the upstream provider.
    """
    try:
        # Use a per-task client to avoid lock contention / deadlock
        # with the shared proxy client.
        client = AsyncClient(timeout=float(_TASK_TIMEOUT))
        try:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {internal_token}",
            }

            base_url = _test_tasks[task_id].get("_base_url", "http://127.0.0.1:28765")
            url = f"{base_url}{endpoint}"

            # Enforce per-task timeout so hung upstream calls don't
            # linger for the full _MAX_TASK_AGE cleanup window.
            resp = await asyncio.wait_for(
                client.post(url, json=payload, headers=headers),
                timeout=_TASK_TIMEOUT,
            )
            assert isinstance(resp, HttpResponse)

            # Try to parse JSON body
            try:
                body = resp.json()
            except Exception:
                body = resp.text

            _test_tasks[task_id].update(
                {
                    "status": "done",
                    "status_code": resp.status_code,
                    "body": body,
                }
            )
        finally:
            await client.aclose()
    except asyncio.CancelledError:
        _test_tasks[task_id]["status"] = "cancelled"
    except asyncio.TimeoutError:
        _test_tasks[task_id].update(
            {
                "status": "error",
                "error": f"Test timed out after {_TASK_TIMEOUT}s",
            }
        )
    except Exception as exc:
        _test_tasks[task_id].update(
            {
                "status": "error",
                "error": str(exc),
            }
        )


def _get_gateway_config(request: Any) -> GatewayConfig | None:
    """Return the live GatewayConfig from the app module."""
    import llm_rosetta.gateway.app as _app_mod

    return _app_mod._config


async def start_test(request: Any) -> Response:
    """Start an async model test.  Returns a task_id immediately.

    POST /admin/api/test
    Body: {endpoint: "/v1/...", payload: {...}}
    """
    _cleanup_stale_tasks()

    try:
        body = request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON"}, status_code=400)

    endpoint = body.get("endpoint")
    payload = body.get("payload")
    if not endpoint or not isinstance(payload, dict):
        return JSONResponse({"error": "Missing endpoint or payload"}, status_code=400)

    internal_token = getattr(request.app, "internal_token", "")
    if not internal_token:
        return JSONResponse({"error": "No internal token available"}, status_code=500)

    # Determine the base URL the gateway is listening on so the test
    # task can POST back to ourselves.
    host = getattr(request.app, "_bind_host", "127.0.0.1")
    port = getattr(request.app, "_bind_port", 28765)
    # Always use 127.0.0.1 for self-calls — even if bound to 0.0.0.0
    if host in ("0.0.0.0", "::"):
        host = "127.0.0.1"
    base_url = f"http://{host}:{port}"

    task_id = uuid.uuid4().hex[:12]
    _test_tasks[task_id] = {
        "status": "pending",
        "started": time.monotonic(),
        "_base_url": base_url,
    }

    # Determine proxy from gateway config
    gw_config = _get_gateway_config(request)
    proxy_url = gw_config.proxy if gw_config else None

    asyncio_task = asyncio.create_task(
        _run_test_task(task_id, endpoint, payload, internal_token, proxy_url)
    )
    _test_tasks[task_id]["_task_obj"] = asyncio_task

    return JSONResponse({"task_id": task_id})


async def get_test_result(request: Any, task_id: str = "") -> Response:
    """Poll for a test task result.

    GET /admin/api/test/<task_id>
    """
    task = _test_tasks.get(task_id)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)

    # Return only serialisable fields (skip _task_obj, _base_url)
    result: dict[str, Any] = {k: v for k, v in task.items() if not k.startswith("_")}
    return JSONResponse(result)


async def cancel_test(request: Any, task_id: str = "") -> Response:
    """Cancel a running test task.

    DELETE /admin/api/test/<task_id>
    """
    task = _test_tasks.get(task_id)
    if not task:
        return JSONResponse({"error": "Task not found"}, status_code=404)

    task_obj = task.get("_task_obj")
    if task_obj and not task_obj.done():
        task_obj.cancel()
        task["status"] = "cancelled"

    return JSONResponse({"ok": True})
