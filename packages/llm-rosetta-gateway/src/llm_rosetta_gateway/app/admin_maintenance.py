"""Find and run the maintenance steps of an app for the admin panel.

Every store carries its own registry of steps, and the deferred set and the
load-time set live elsewhere; this module is the one place that knows where,
so the admin surface asks ``AdminContext.maintenance_registries`` and
``run_maintenance_step`` instead of reaching into the app.
"""

from __future__ import annotations

from typing import Any

from ..admin_api import MaintenanceLoadTimeStep
from ..keystore.maintenance import build_registry as build_keystore_registry
from ..maintenance.backfills import build_registry as build_deferred_registry
from ..observability.maintenance import Policy, Registry, StepContext, StepResult


def _as_registry(candidate: Any) -> Registry | None:
    """A registry, or nothing — never a stand-in that merely has the attribute.

    The stores on an app are not always the real objects (a test holds a mock, a
    deployment mid-startup holds none), and ``hasattr(store, "maintenance")``
    would happily return a mock's attribute and report a registry that does not
    exist.
    """
    return candidate if isinstance(candidate, Registry) else None


def registry_for(app: Any) -> Registry:
    """The deferred registry this app runs, built once if nobody has yet.

    A process that reached deferred startup has one, holding the results of
    that run.  An app assembled without one — a test, or a gateway whose
    startup was cut short — gets a registry over the stores it does have, kept
    on the app so a re-run's result is there for the next reader rather than
    thrown away with the object that produced it.
    """
    deferred = getattr(app, "deferred_startup", None)
    registry = _as_registry(getattr(deferred, "maintenance", None))
    if registry is not None:
        return registry

    cached = _as_registry(getattr(app, "_maintenance_registry", None))
    if cached is None:
        cached = build_deferred_registry(app)
        try:
            setattr(app, "_maintenance_registry", cached)
        except AttributeError:  # pragma: no cover - a frozen app is still servable
            pass
    return cached


def _keystore_registry(app: Any) -> Registry | None:
    """The key store's own registry, if this app has a key store.

    Built and kept here when the store did not open through it (a test, or a
    ``KeyStore`` handed to the app already open): the store is where its
    registry belongs, so a re-run's result is not thrown away with the object
    that produced it.
    """
    keystore = getattr(app, "keystore", None)
    if keystore is None:
        return None
    registry = _as_registry(getattr(keystore, "maintenance", None))
    if registry is None:
        registry = build_keystore_registry()
        try:
            keystore.maintenance = registry
        except AttributeError:  # pragma: no cover - an immutable store still serves
            pass
    return registry


def registries_for(app: Any) -> list[Registry]:
    """Every registry this app can report on, in the order an operator reads it.

    The deferred set first — what runs after startup, and the one whose failures
    an operator is looking for — then each store's own schema registry, then the
    load-time set the config carries.  A store whose registry never ran, and a
    config that was never loaded through one, contribute nothing: "no registry"
    and "a registry that did nothing" are different answers, and reporting the
    second for the first would be a lie at the exact moment it matters.
    """
    found: list[Registry] = []
    seen: set[str] = set()
    candidates = [
        registry_for(app),
        _as_registry(getattr(getattr(app, "persistence", None), "maintenance", None)),
        _keystore_registry(app),
        _as_registry(
            getattr(getattr(app, "gateway_config", None), "maintenance", None)
        ),
    ]
    for registry in candidates:
        if registry is not None and registry.name not in seen:
            seen.add(registry.name)
            found.append(registry)
    return found


def context_for(app: Any) -> StepContext:
    """What a step reads: the stores and the config this app has."""
    return StepContext(
        config=getattr(app, "gateway_config", None),
        persistence=getattr(app, "persistence", None),
        keystore=getattr(app, "keystore", None),
        app=app,
    )


def _find(app: Any, name: str) -> tuple[Registry, Any] | None:
    """The registry that owns a step name, and the step."""
    for registry in registries_for(app):
        for step in registry.steps:
            if step.name == name:
                return registry, step
    return None


async def run_step(app: Any, name: str) -> StepResult | None:
    """Run one step, returning its result, or ``None`` when it is not here.

    A step that fails has still answered the request: the registry records why
    (and re-raises, for a policy that raises), so the caller gets the recorded
    result rather than a failed response.  The operator asked a question; "it
    failed, here is the error" is the answer to it.

    Raises:
        MaintenanceLoadTimeStep: For a step of the load-time registry, which is not
            re-runnable from a request.
    """
    found = _find(app, name)
    if found is None:
        return None
    registry, _step = found
    if registry.policy is Policy.LOAD_TIME:
        raise MaintenanceLoadTimeStep(name)
    try:
        return await registry.run_step(name, context_for(app))
    except Exception:  # noqa: BLE001 - the result is the answer, not the raise
        for result in registry.results:
            if result.name == name:
                return result
        raise  # pragma: no cover - run_step records before it re-raises
