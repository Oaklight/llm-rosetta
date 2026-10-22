"""The config's load-time steps: the same vocabulary, a different store.

Two mechanisms bring a config document up to what the code expects, and they
differ from the database ones in every way that matters:

- they run while the document is *loaded*, on the calling thread, because a
  reader of the config cannot wait for a background task and the load is not
  asynchronous anywhere up the chain (:class:`Policy.LOAD_TIME <...Policy>`,
  and :meth:`~...Registry.run_sync` exists for exactly this);
- their ledger is ``config_version`` *in the document*, not a table beside it —
  a migration here is recorded by the file it corrected, so the file can be read
  by an older release without leaving a database behind;
- the name correction has no ledger at all and needs none: it re-derives every
  value and reports whether any differed, which is what makes it idempotent.

They are steps rather than two calls in a row so that the order is declared
(versioned structure first, names second), each one says whether it changed
anything, and one surface can report both alongside the repairs the databases
needed.  Their ``store`` is the document, not a file they own: a load-time step
changes the bytes that were read, and the caller decides whether to write them.

Nothing here re-runs a step against a document it did not read.  A load-time
step is run by a load — the admin surface lists these but refuses to re-run one,
because the document a previous load held is not the document on disk.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta_gateway.observability.maintenance import (
    Policy,
    Registry,
    Step,
    StepContext,
)

#: What the admin surface and the log lines call this registry.
NAME = "config"

#: A load-time step's store is the document itself.  There is no filename here
#: because there is no single one: the same steps run for whichever config the
#: caller is reading.
STORE = "config"

#: The registry's steps, in the order they must run.
STEPS = ["config.migrate_structure", "config.correct_legacy_names"]


def _migrate_structure(context: StepContext) -> int:
    """Apply the versioned config migrations, which own ``config_version``.

    The migrations themselves stay where they are (:mod:`~...gateway.maintenance.migrations`):
    the table is written once, with its own log line per migration, and this
    step is what puts it in the load's order and reports whether it moved.
    """
    from .migrations import migrate

    raw: dict[str, Any] = context.config
    _raw, changed = migrate(raw)
    return 1 if changed else 0


def _correct_legacy_names(context: StepContext) -> int:
    """Rewrite provider and shim names that were renamed, in place.

    Unversioned and idempotent by construction: it re-derives each value and
    reports whether any differed.  It is deliberately not part of
    ``config_version`` — it corrects *names*, not structure, and it must run for
    a config whose version is already current.
    """
    from .names import correct_legacy_names

    raw: dict[str, Any] = context.config
    return 1 if correct_legacy_names(raw) else 0


def build_registry() -> Registry:
    """The load-time steps, over the document the caller passes in the context."""
    return Registry(
        name=NAME,
        steps=[
            Step(
                name="config.migrate_structure",
                description=(
                    "bring an older config document up to the current structure"
                ),
                store=STORE,
                run=_migrate_structure,
            ),
            Step(
                name="config.correct_legacy_names",
                description=("rewrite provider and shim names that have been renamed"),
                store=STORE,
                run=_correct_legacy_names,
            ),
        ],
        policy=Policy.LOAD_TIME,
    )


class ConfigMigrationError(RuntimeError):
    """A load-time step could not bring this document forward.

    The step's own exception is the cause; this one says *which* step and *which
    document*, because the exception that comes out of a migration body is about
    the shape it found (``'str' object has no attribute 'items'``) and not about
    where it found it.  A config that cannot be brought forward has no correct
    answer to serve, so this is raised out of the load.
    """

    def __init__(self, path: str, step: str, cause: BaseException) -> None:
        super().__init__(
            f"config {path!r} could not be brought forward by step {step!r}: {cause}"
        )
        self.path = path
        self.step = step
        self.cause = cause


def apply(
    raw: dict[str, Any], *, path: str = "<document>"
) -> tuple[dict[str, Any], bool, Registry]:
    """Run the load-time steps over one raw config document.

    What the loader calls, in place of the two calls it used to make: the steps
    mutate ``raw`` and each says whether it moved, and the registry that ran is
    returned so the caller can keep it (the admin surface lists it; the config
    object holds it) instead of the load's work being visible only in the log.

    Args:
        raw: The document, corrected in place.
        path: What to name it in :class:`ConfigMigrationError`, when a step
            fails.  The document may not be a file at all.

    Returns:
        The document (the same object), whether anything changed, and the
        registry whose results say which step changed what.

    Raises:
        ConfigMigrationError: When a step fails.  ``Policy.LOAD_TIME`` means the
            load fails; this is what the caller fails *with*.
    """
    registry = build_registry()
    try:
        results = registry.run_sync(StepContext(config=raw))
    except Exception as exc:
        failed = next((r for r in registry.results if not r.ok), None)
        raise ConfigMigrationError(
            path, failed.name if failed else "unknown", exc
        ) from exc
    return raw, any(result.changed for result in results), registry
