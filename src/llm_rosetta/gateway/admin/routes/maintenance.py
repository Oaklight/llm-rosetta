"""The maintenance surface: what the deferred steps did, and re-running one.

The five repairs that run after the socket is open each report a
:class:`~llm_rosetta.observability.maintenance.StepResult` — what changed, how
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

from llm_rosetta._vendor.httpserver import JSONResponse, Response
from llm_rosetta.observability.maintenance import Registry, StepContext, StepResult

from ...deferred_backfills import build_registry


def registry_for(app: Any) -> Registry:
    """The deferred registry this app runs, built once if nobody has yet.

    A process that reached deferred startup has one, holding the results of
    that run.  An app assembled without one — a test, or a gateway whose
    startup was cut short — gets a registry over the stores it does have, kept
    on the app so a re-run's result is there for the next reader rather than
    thrown away with the object that produced it.
    """
    deferred = getattr(app, "deferred_startup", None)
    registry = getattr(deferred, "maintenance", None)
    if registry is not None:
        return registry

    cached = getattr(app, "_maintenance_registry", None)
    if cached is None:
        cached = build_registry(app)
        try:
            setattr(app, "_maintenance_registry", cached)
        except AttributeError:  # pragma: no cover - a frozen app is still servable
            pass
    return cached


def context_for(app: Any) -> StepContext:
    """What a step reads: the stores and the config this app has."""
    return StepContext(
        config=getattr(app, "gateway_config", None),
        persistence=getattr(app, "persistence", None),
        keystore=getattr(app, "keystore", None),
        app=app,
    )


def _result_payload(result: StepResult) -> dict[str, Any]:
    return {
        "name": result.name,
        "store": result.store,
        "ok": result.ok,
        "changed": result.changed,
        "seconds": round(result.seconds, 3),
        "error": result.error,
    }


def _step_payload(registry: Registry, name: str) -> dict[str, Any]:
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


async def run_step(app: Any, name: str) -> StepResult | None:
    """Run one step, returning its result, or ``None`` when it is not here.

    A step that fails has still answered the request: the registry records why
    (and re-raises, for a policy that raises), so the caller gets the recorded
    result rather than a failed response.  The operator asked a question; "it
    failed, here is the error" is the answer to it.
    """
    registry = registry_for(app)
    if name not in {step.name for step in registry.steps}:
        return None
    try:
        return await registry.run_step(name, context_for(app))
    except Exception:  # noqa: BLE001 - the result is the answer, not the raise
        for result in registry.results:
            if result.name == name:
                return result
        raise  # pragma: no cover - run_step records before it re-raises


async def get_maintenance(request: Any, **kwargs: Any) -> Response:
    """What every deferred step did, with the ones that never ran marked so."""
    registry = registry_for(request.app)
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
    result = await run_step(request.app, name)
    if result is None:
        return JSONResponse(
            {"error": f"No maintenance step named '{name}'"}, status_code=404
        )
    return JSONResponse(_result_payload(result))


async def backfill_keys_last_used(request: Any, **kwargs: Any) -> Response:
    """The keys tab's button, over the maintenance step rather than beside it."""
    if getattr(request.app, "persistence", None) is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)
    result = await run_step(request.app, "api_keys.backfill_last_used")
    if result is None:
        return JSONResponse({"error": "No key store configured"}, status_code=400)
    return JSONResponse({"updated": result.changed})


async def backfill_dump_log_ids(request: Any, **kwargs: Any) -> Response:
    """The logs tab's button, over the maintenance step rather than beside it."""
    if getattr(request.app, "persistence", None) is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)
    result = await run_step(request.app, "error_dumps.backfill_log_ids")
    if result is None:
        return JSONResponse({"error": "No persistence configured"}, status_code=400)
    return JSONResponse({"updated": result.changed})
