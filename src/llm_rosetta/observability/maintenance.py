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

import inspect
import logging
import time
from collections.abc import Callable, Sequence
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

    ``store`` is carried on the result rather than looked up from the step, so a
    registry whose steps span more than one store still answers the question
    that follows a ``partial`` run — *which* store is half-repaired — without
    the reader joining two objects or parsing it back out of a step name.
    """

    name: str
    store: str
    changed: int
    seconds: float
    ok: bool
    error: str | None = None


#: A step's work: given the context, make its change and say how much it
#: changed.  Returning 0 is how a step says "already applied".
#:
#: A step may be a plain function rather than a coroutine function: the
#: load-time set runs on a synchronous path (a config is read without a loop
#: running), and requiring it to be async would be ceremony around a pure
#: computation over a document.  A registry drives whichever it is —
#: :meth:`Registry.run` awaits an awaitable result and takes a plain one, and
#: :meth:`Registry.run_sync` refuses an asynchronous step rather than silently
#: skipping it.
StepRun = Callable[[StepContext], Any]


@dataclass(frozen=True)
class Step:
    """One ordered, named, idempotent correction.

    ``description`` is why the step exists, in the terms a reader of the store
    needs — it ends up in a log line and in an admin surface, so it is written
    for that reader rather than for the author.  ``store`` names the store the
    step repairs (a file, in this repository's stores), and travels onto every
    :class:`StepResult` the step produces.
    """

    name: str
    description: str
    store: str
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

    def run_sync(self, context: StepContext) -> list[StepResult]:
        """Run every step in order, for a registry whose steps are synchronous.

        A config is read on a synchronous path — the loader is called by readers
        that have no loop running — so the load-time registry cannot be driven
        by :meth:`run`.  Rather than make the config path asynchronous, or reach
        for a loop that may not exist, the runner takes the steps' own answer: a
        step that returns without awaiting runs here.

        What a failure means is the policy's, exactly as in :meth:`run`.

        Raises:
            TypeError: When a step's body is asynchronous.  A synchronous run
                that quietly dropped one would report the document as repaired
                by a step that never ran.
            Exception: The failing step's own exception, for
                :attr:`Policy.MANDATORY` and :attr:`Policy.LOAD_TIME`.
        """
        results: list[StepResult] = []
        for step in self.steps:
            result, failure = self._run_one_sync(step, context)
            results.append(result)
            if failure is not None and self.policy is not Policy.DEFERRED:
                self.results = results
                raise failure
        self.results = results
        return results

    async def run_step(self, name: str, context: StepContext) -> StepResult:
        """Run one step by name, for an operator who asked for it.

        The named step is the only one that runs, and its result replaces that
        step's in :attr:`results` so the surface that asked still reads one
        answer per step.  What a failure means is still the policy's — a
        raising registry raises, after recording, exactly as :meth:`run` does —
        which leaves the caller free to decide that an on-demand re-run reports
        the failure rather than failing itself.

        Raises:
            KeyError: When no registered step has that name.
        """
        order = {step.name: index for index, step in enumerate(self.steps)}
        for step in self.steps:
            if step.name == name:
                break
        else:
            raise KeyError(name)

        result, failure = await self._run_one(step, context)
        if any(existing.name == name for existing in self.results):
            self.results = [
                result if existing.name == name else existing
                for existing in self.results
            ]
        else:
            self.results = sorted([*self.results, result], key=lambda r: order[r.name])
        if failure is not None and self.policy is not Policy.DEFERRED:
            raise failure
        return result

    async def _run_one(
        self, step: Step, context: StepContext
    ) -> tuple[StepResult, Exception | None]:
        """Run one step, turning whatever it does into a result and a failure."""
        started = time.monotonic()
        try:
            changed = await _outcome(step, context)
        except Exception as exc:
            return self._failed(step, exc, time.monotonic() - started), exc
        return self._succeeded(step, changed, time.monotonic() - started), None

    def _run_one_sync(
        self, step: Step, context: StepContext
    ) -> tuple[StepResult, Exception | None]:
        """One step of a registry whose steps are synchronous."""
        started = time.monotonic()
        try:
            changed = _outcome_sync(step, context)
        except Exception as exc:
            return self._failed(step, exc, time.monotonic() - started), exc
        return self._succeeded(step, changed, time.monotonic() - started), None

    def _succeeded(self, step: Step, changed: int, seconds: float) -> StepResult:
        """Record a step that ran, logging it where the policy calls for it."""
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
            name=step.name,
            store=step.store,
            changed=changed,
            seconds=seconds,
            ok=True,
        )

    def _failed(self, step: Step, exc: Exception, seconds: float) -> StepResult:
        """Record a step that failed, naming it in the log before it travels."""
        logger.warning(
            "maintenance step %s (%s) failed after %.3fs: %s",
            step.name,
            self.name,
            seconds,
            exc,
        )
        return StepResult(
            name=step.name,
            store=step.store,
            changed=0,
            seconds=seconds,
            ok=False,
            error=f"{type(exc).__name__}: {exc}",
        )


async def _outcome(step: Step, context: StepContext) -> int:
    """What a step produced, awaited when it is awaitable and taken when not."""
    outcome = step.run(context)
    if inspect.isawaitable(outcome):
        return await outcome
    return outcome


def _outcome_sync(step: Step, context: StepContext) -> int:
    """The same, for a run that has no loop to offer a coroutine to.

    Raises:
        TypeError: When the step is asynchronous.  A synchronous run that
            dropped such a step would report a store as repaired by a step that
            never ran, which is worse than refusing to run at all.
    """
    outcome = step.run(context)
    if inspect.isawaitable(outcome):
        if inspect.iscoroutine(outcome):
            outcome.close()  # would otherwise warn that it was never awaited
        raise TypeError(
            f"maintenance step {step.name!r} is asynchronous and cannot run in "
            f"a synchronous registry ({context!r})"
        )
    return outcome
