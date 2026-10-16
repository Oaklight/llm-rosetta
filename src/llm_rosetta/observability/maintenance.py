"""One vocabulary for "bring existing state up to what the code expects".

This repository already has five mechanisms that do that job — config
migrations, the legacy name correction, the request-log DDL migrations, the
keystore's guarded DDL plus a backfill, and the five startup backfills — and
each grew its own way of saying the same three things: *what ran*, *did it
already run*, and *what does a failure mean*.  This module is the shared
vocabulary; the mechanisms keep their homes and their ledgers.

A :class:`Step` is one such correction: named, ordered, idempotent, and able to
say how much it changed.  A :class:`Registry` holds the steps of one store with
one :class:`Policy`, and :meth:`Registry.run` is the one runner:

- :attr:`Policy.MANDATORY` — synchronous, and a failure **aborts**.  The store
  is in an unknown state and serving from it is worse than not serving, so the
  step's own exception propagates unchanged.
- :attr:`Policy.LOAD_TIME` — runs while a config is loaded, corrects it, writes
  it back.  A failure fails that load.
- :attr:`Policy.DEFERRED` — runs after the socket is open; a failure is
  recorded and the next step still runs.  Failure isolation is the point: one
  bad backfill must not silently skip the repairs behind it.

The policies are not decoration.  A mandatory DDL step that continues after a
failure corrupts a store, and a best-effort backfill that aborts startup is a
self-inflicted outage, so the policy is a property of the registry rather than
a flag a caller remembers to pass.

**A step that succeeds is silent; a step that fails is not.**  A registry whose
steps already log what they did gets no per-step line — the request-log
migrations' two INFO lines per migration *are* the record, and a line per step
would only duplicate them — while the deferred set, whose steps have no voice of
their own, reports each one at INFO.  A failure is always a WARNING naming the
step, the registry and how long it ran, logged before the exception is re-raised
for a policy that raises: an abort whose only evidence is a traceback says what
went wrong but not which step did it.  That WARNING is the one log line this
module adds to a failing startup.

Registries are per store: ``request_log`` and ``api_keys`` are different SQLite
files with different ledgers, so each declares its own steps (a single global
list would be wrong), and the runner here is shared.  The module imports
neither the gateway nor a store — :class:`StepContext` carries whatever a step
may need — so both sides can use it, which is also what the package's
no-gateway-import guard requires.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "Policy",
    "Registry",
    "Step",
    "StepContext",
    "StepResult",
]

logger = logging.getLogger("llm-rosetta.observability.maintenance")


class Policy(Enum):
    """What a failure in a registry means, and when its steps run."""

    #: Synchronous, and a failure aborts the caller's startup.
    MANDATORY = "mandatory"
    #: Runs while a config is loaded; a failure fails that load.
    LOAD_TIME = "load_time"
    #: Runs in the background after serving starts; a failure is recorded and
    #: the remaining steps still run.
    DEFERRED = "deferred"


@dataclass(frozen=True)
class StepContext:
    """What a step may need, resolved by the caller that has it.

    The fields are deliberately untyped beyond "whatever it is": this module
    may not import the gateway, or a store, to name them, and a step knows what
    it asked for.  A step takes what it needs from here rather than the caller
    assembling arguments per step, so adding a step does not change the runner.
    """

    config: Any = None
    persistence: Any = None
    keystore: Any = None
    app: Any = None


@dataclass(frozen=True)
class StepResult:
    """What one step did, kept so that an operator can be told.

    ``changed`` is the step's own unit — rows, statements, whatever it counts —
    and 0 is the answer that matters most: it is what an idempotent step returns
    when there was nothing left to do.  A step that failed reports 0 there too,
    because what it half-did is not a count; the failure is the ``error``.
    """

    name: str
    changed: int
    seconds: float
    ok: bool
    error: str | None = None


#: A step's work: given the context, make its change and say how much it
#: changed.  Returning 0 is how a step says "already applied".
StepRun = Callable[[StepContext], Awaitable[int]]


@dataclass(frozen=True)
class Step:
    """One ordered, named, idempotent correction.

    ``description`` is why the step exists, in the terms a reader of the store
    needs — it ends up in a log line and in an admin surface, so it is written
    for that reader rather than for the author.
    """

    name: str
    description: str
    run: StepRun


@dataclass
class Registry:
    """The steps of one store, under one policy.

    The results of the last run are kept (:attr:`results`) so that "did the
    backfills run, and how much did they move" has an answer that is not a
    log search.
    """

    name: str
    steps: Sequence[Step]
    policy: Policy
    results: list[StepResult] = field(default_factory=list)

    async def run(self, context: StepContext) -> list[StepResult]:
        """Run every step in order, recording what each one did.

        Returns:
            One :class:`StepResult` per step, in order.

        Raises:
            Exception: For :attr:`Policy.MANDATORY` and
                :attr:`Policy.LOAD_TIME`, the failing step's own exception,
                re-raised unchanged and after its result was recorded: a caller
                that aborts startup today must keep aborting, with the error it
                aborted with.  A :attr:`Policy.DEFERRED` registry does not
                raise — a failed step is reported in its result and the next
                step runs.
        """
        results: list[StepResult] = []
        for step in self.steps:
            result, failure = await self._run_one(step, context)
            results.append(result)
            if failure is not None and self.policy is not Policy.DEFERRED:
                self.results = results
                raise failure
        self.results = results
        return results

    async def _run_one(
        self, step: Step, context: StepContext
    ) -> tuple[StepResult, Exception | None]:
        """Run one step, turning whatever it does into a result and a failure."""
        started = time.monotonic()
        try:
            changed = await step.run(context)
        except Exception as exc:
            seconds = time.monotonic() - started
            logger.warning(
                "maintenance step %s (%s) failed after %.3fs: %s",
                step.name,
                self.name,
                seconds,
                exc,
            )
            result = StepResult(
                name=step.name,
                changed=0,
                seconds=seconds,
                ok=False,
                error=f"{type(exc).__name__}: {exc}",
            )
            return result, exc
        seconds = time.monotonic() - started
        # A deferred registry is the one whose steps have no voice of their own,
        # so it reports each of them; a mandatory registry's steps already log
        # what they did, and a line per step here would change the output of a
        # path whose logs are pinned.
        level = logging.INFO if self.policy is Policy.DEFERRED else logging.DEBUG
        if changed:
            logger.log(
                level,
                "maintenance step %s (%s) changed %d in %.3fs",
                step.name,
                self.name,
                changed,
                seconds,
            )
        else:
            logger.debug(
                "maintenance step %s (%s) had nothing to do", step.name, self.name
            )
        return StepResult(
            name=step.name, changed=changed, seconds=seconds, ok=True
        ), None
