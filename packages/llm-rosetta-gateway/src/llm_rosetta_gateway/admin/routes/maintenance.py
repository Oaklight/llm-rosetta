"""The maintenance surface: what the deferred steps did, and re-running one.

The five repairs that run after the socket is open each report a
:class:`~llm_rosetta_gateway.observability.maintenance.StepResult` — what changed, how
long it took, and why it failed if it did — and those results are what this
serves, so "did the backfills run" is a request rather than a log search.

``POST /admin/api/maintenance/<name>`` runs one step by name.  The two
single-purpose routes that predate this surface
(``/admin/api/keys/backfill-last-used`` and
``/admin/api/error-dumps/backfill-log-ids``) are handlers over the same steps
rather than a second implementation, and keep their own response shape because
the admin panel's buttons read ``updated`` from them.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response
from ...admin_api import MaintenanceLoadTimeStep, admin_context


def _result_payload(result: Any) -> dict[str, Any]:
    return {
        "name": result.name,
        "store": result.store,
        "ok": result.ok,
        "changed": result.changed,
        "seconds": round(result.seconds, 3),
        "error": result.error,
    }


def _step_payload(registry: Any, name: str) -> dict[str, Any]:
    """One step as the surface reports it, run or not."""
    step = next(step for step in registry.steps if step.name == name)
    payload: dict[str, Any] = {
        "name": step.name,
        "description": step.description,
        "store": step.store,
        "ran": False,
    }
    for result in registry.results:
        if result.name == name:
            payload.update(_result_payload(result))
            payload["ran"] = True
    return payload


async def get_maintenance(request: Any, **kwargs: Any) -> Response:
    """What every step of every registry did, with the ones that never ran marked.

    One surface for all of them — the schema of each store, the repairs that run
    after startup, and the steps that correct a config as it loads — which is
    what the epic meant by not having to know which mechanism lives where.
    """
    return JSONResponse(
        {
            "registries": [
                {
                    "name": registry.name,
                    "policy": registry.policy.value,
                    "steps": [
                        _step_payload(registry, step.name) for step in registry.steps
                    ],
                }
                for registry in admin_context(request).maintenance_registries()
            ],
        }
    )


async def post_maintenance_run(request: Any, **kwargs: Any) -> Response:
    """Re-run one deferred step by name.

    Deferred steps are the ones that are safe to repeat — each one's own
    ``WHERE`` clause stops it matching once it has run, which is what makes the
    idempotence property assertable — so re-running one is always allowed.
    """
    name = kwargs.get("name") or request.path_params["name"]
    try:
        result = await admin_context(request).run_maintenance_step(name)
    except MaintenanceLoadTimeStep:
        return JSONResponse(
            {
                "error": (
                    f"'{name}' runs when a config is loaded, not on demand: "
                    f"its result belongs to the document that was read"
                )
            },
            status_code=409,
        )
    if result is None:
        return JSONResponse(
            {"error": f"No maintenance step named '{name}'"}, status_code=404
        )
    return JSONResponse(_result_payload(result))


async def backfill_keys_last_used(request: Any, **kwargs: Any) -> Response:
    """The keys tab's button, over the maintenance step rather than beside it."""
    if admin_context(request).persistence() is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)
    result = await admin_context(request).run_maintenance_step(
        "api_keys.backfill_last_used"
    )
    if result is None:
        return JSONResponse({"error": "No key store configured"}, status_code=400)
    return JSONResponse({"updated": result.changed})


async def backfill_dump_log_ids(request: Any, **kwargs: Any) -> Response:
    """The logs tab's button, over the maintenance step rather than beside it."""
    if admin_context(request).persistence() is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)
    result = await admin_context(request).run_maintenance_step(
        "error_dumps.backfill_log_ids"
    )
    if result is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)
    return JSONResponse({"updated": result.changed})
