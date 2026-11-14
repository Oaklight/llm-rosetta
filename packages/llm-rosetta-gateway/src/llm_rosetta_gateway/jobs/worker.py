"""The dispatcher: claim a job, run its handler, record how it ended.

One asyncio task per lane slot.  The ``sqlite`` lane runs one worker on purpose
— every statement shares one connection and one writer, so a second worker
would interleave statements without doing more work.  The lane is a name in the
model, not a constant in this loop, so a network-bound lane can run several
workers when the Batch API arrives (#1509).

Shutdown is cooperative.  ``stop()`` cancels the lane tasks; a task inside a
handler catches the cancellation, releases its job back to ``queued`` and
re-raises.  A delete already handed to SQLite is not recalled — cancellation is
batch-granular, which is why the handler calls ``checkpoint()`` between
batches.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from .model import (
    DEFAULT_LEASE_SECONDS,
    JOB_CANCELLED,
    JOB_FAILED,
    JOB_SUCCEEDED,
    LANE_SQLITE,
    Job,
    JobCancelled,
    LeaseLost,
)
from .registry import Handler, JobRegistry, UnknownJobKind, default_handlers
from .store import JobStore

logger = logging.getLogger(__name__)

#: A handler may report progress this often; the row shares the one connection
#: with request logging, so an unthrottled `progress()` per row would be the
#: bottleneck it is meant to observe.
_PROGRESS_INTERVAL = 1.0

#: A lease is extended at most this often.  Shorter than the lease itself, so a
#: slow batch cannot let the lease lapse mid-delete.
_HEARTBEAT_INTERVAL = 15.0

#: How long an idle worker sleeps before looking for work again.  New work also
#: wakes it immediately.
_WAIT_SECONDS = 30.0
#: The file whose exclusive lock says "this process owns the job queue here".
INSTANCE_LOCK_NAME = ".jobs.lock"
#: How often the lane re-checks for work that a statement blocked.
_CLAIM_RETRY_SECONDS = 1.0
#: How often an interrupted run is looked for while the process is up.
_RECOVERY_INTERVAL = 30.0


def _acquire_instance_lock(data_dir: Path) -> int | None:
    """Take the queue's instance lock, or report that someone else holds it.

    Startup recovery takes a lease away from its owner, which assumes this
    process is the only one running jobs in this database.  The lock turns
    that assumption into something the code checks: a second gateway on the
    same data directory cannot take a live job away from the first, and says
    so instead.  POSIX only; where ``fcntl`` is missing the assumption holds
    by convention and the caller logs it.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        return None
    path = data_dir / INSTANCE_LOCK_NAME
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        os.close(fd)
        return None
    return fd


def _release_instance_lock(fd: int | None) -> None:
    """Drop the lock; closing the descriptor is what actually frees it."""
    if fd is None:
        return
    try:
        import fcntl

        fcntl.flock(fd, fcntl.LOCK_UN)
    except (ImportError, OSError):  # pragma: no cover - best effort
        pass
    finally:
        try:
            os.close(fd)
        except OSError:  # pragma: no cover
            pass


class JobContext:
    """What a handler is given: the job, its progress, and a checkpoint."""

    __slots__ = (
        "job",
        "_store",
        "_owner",
        "_cancel",
        "_progress_interval",
        "_heartbeat_interval",
        "_last_progress",
        "_last_heartbeat",
        "_lease_lost",
        "_deleted",
    )

    def __init__(
        self,
        job: Job,
        store: JobStore,
        *,
        owner: str,
        cancel: asyncio.Event,
        progress_interval: float = _PROGRESS_INTERVAL,
        heartbeat_interval: float = _HEARTBEAT_INTERVAL,
    ) -> None:
        self.job = job
        self._store = store
        self._owner = owner
        self._cancel = cancel
        self._progress_interval = progress_interval
        self._heartbeat_interval = heartbeat_interval
        self._last_progress = 0.0
        self._last_heartbeat = time.monotonic()
        self._lease_lost = False
        self._deleted = 0

    @property
    def cancelled(self) -> bool:
        """True once the operator asked this job to stop."""
        return self._cancel.is_set()

    @property
    def deleted(self) -> int:
        """Rows the handler has deleted so far — decides what cancel means."""
        return self._deleted

    def note_deleted(self, count: int) -> None:
        """Add *count* to the running deletion total."""
        self._deleted += count

    async def progress(
        self,
        done: int,
        *,
        total: int | None = None,
        unit: str | None = None,
    ) -> None:
        """Record progress, at most once a second unless the job just finished."""
        now = time.monotonic()
        finished = total is not None and done >= total
        if not finished and now - self._last_progress < self._progress_interval:
            return
        self._last_progress = now
        await self._store.set_progress(self.job.id, done, total=total, unit=unit)

    async def heartbeat(self, *, force: bool = False) -> None:
        """Extend the lease, at most every ``_HEARTBEAT_INTERVAL`` seconds.

        A loss is remembered: the throttle must not turn the next checkpoint
        back into a silent one, because the checkpoint is the handler's own
        safety gate and a handler may catch the exception and carry on.
        """
        if self._lease_lost:
            raise LeaseLost()
        now = time.monotonic()
        if not force and now - self._last_heartbeat < self._heartbeat_interval:
            return
        self._last_heartbeat = now
        held = await self._store.heartbeat(
            self.job.id, self._owner, lease_seconds=DEFAULT_LEASE_SECONDS
        )
        if not held:
            # Someone else owns the row now (recovery after an expiry).  Stop
            # writing on a lease we no longer hold.
            self._lease_lost = True
            raise LeaseLost()

    async def checkpoint(self) -> None:
        """Stop if asked, and keep the lease alive.

        Call this between batches, never in the middle of one: the delete that
        is already running cannot be recalled.
        """
        if self._cancel.is_set():
            raise JobCancelled()
        await self.heartbeat()

    def partial_result(self) -> dict[str, Any]:
        """What a cancelled job reports about the work it did finish."""
        return {"deleted": self._deleted} if self._deleted else {}


class JobManager:
    """Owns the store, the registry and the lane tasks."""

    def __init__(
        self,
        persistence: Any,
        *,
        handlers: dict[str, Handler] | None = None,
        owner: str | None = None,
        lanes: dict[str, int] | None = None,
        lease_seconds: float = DEFAULT_LEASE_SECONDS,
        wait_seconds: float = _WAIT_SECONDS,
        recovery_interval: float = _RECOVERY_INTERVAL,
    ) -> None:
        self.store = JobStore(persistence)
        self.registry = JobRegistry(default_handlers())
        for kind, handler in (handlers or {}).items():
            self.registry.register(kind, handler)
        self.owner = owner or uuid.uuid4().hex
        self._lanes = dict(lanes or {LANE_SQLITE: 1})
        self._lease = lease_seconds
        self._wait = wait_seconds
        self._recovery_interval = recovery_interval
        self._tasks: list[asyncio.Task[None]] = []
        self._cancel: dict[str, asyncio.Event] = {}
        self._wake = asyncio.Event()
        self._stopping = False
        self.recovered = 0
        self.orphaned_ranges = 0
        self._last_prune = 0.0
        self._lock_fd: int | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _data_dir(self) -> Path | None:
        """Where this manager's database lives; the lock sits beside it.

        ``None`` when the store has no file: there is nothing for a second
        process to contend over, so there is nothing to lock, and falling
        back to the current directory would make two unrelated managers
        fight over one file.
        """
        db_path = self.store.db_path
        return Path(db_path).parent if db_path else None

    async def start(self) -> None:
        """Repair orphaned ranges, recover interrupted jobs, start the lanes."""
        self.orphaned_ranges = await self.store.sweep_orphan_ranges()
        # Taking a live lease away from its owner is only correct when this
        # process is the only one running jobs in this database, so the
        # assumption is checked rather than declared: whoever holds the
        # instance lock may sweep foreign leases, everyone else waits for the
        # lease to expire like a peer would.
        data_dir = self._data_dir()
        self._lock_fd = _acquire_instance_lock(data_dir) if data_dir else None
        if data_dir is not None and self._lock_fd is None:
            logger.warning(
                "Jobs: another process holds %s; startup recovery is limited "
                "to expired leases",
                INSTANCE_LOCK_NAME,
            )
        self.recovered = await self.store.recover(
            owner=self.owner if self._lock_fd is not None else None
        )
        if self.recovered:
            logger.info(
                "Jobs: %d interrupted job(s) returned to the queue", self.recovered
            )
        self._stopping = False
        self._tasks.append(
            asyncio.create_task(self._recovery_loop(), name="jobs-recovery")
        )
        for lane, concurrency in self._lanes.items():
            for index in range(max(1, concurrency)):
                self._tasks.append(
                    asyncio.create_task(
                        self._lane_loop(lane), name=f"jobs-{lane}-{index}"
                    )
                )

    async def stop(self) -> None:
        """Stop the lane tasks; running jobs return to the queue."""
        self._stopping = True
        self._wake.set()
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        _release_instance_lock(self._lock_fd)
        self._lock_fd = None

    # ------------------------------------------------------------------
    # Public operations
    # ------------------------------------------------------------------

    async def enqueue(self, *args: Any, **kwargs: Any) -> Job:
        """Record a job and wake a worker."""
        job = await self.store.enqueue(*args, **kwargs)
        self._wake.set()
        return job

    async def maybe_prune(self, interval: float = 3600.0, keep: int = 200) -> int:
        """Delete old terminal jobs, at most once per *interval*.

        The board is a recent-history view; an unbounded table of finished
        jobs is a slow leak.  A job that still owns a hidden range is never
        pruned, so a failed job stays retryable however old it is.
        """
        now = time.monotonic()
        if self._last_prune and now - self._last_prune < interval:
            return 0
        self._last_prune = now
        return await self.store.prune(keep)

    def wake(self) -> None:
        """Tell the lanes to look for work now, not at the next timeout.

        Retry and resume change a row's state from a request handler, which
        the lane has no way to observe; without this the job waits out the
        30-second poll first.
        """
        self._wake.set()

    async def cancel(self, job_id: str) -> tuple[bool, str | None]:
        """Ask a job to stop; a running one stops at its next checkpoint."""
        found, status = await self.store.request_cancel(job_id)
        event = self._cancel.get(job_id)
        if event is not None:
            event.set()
        return found, status

    async def run_once(self) -> Job | None:
        """Claim and run at most one job.  Deterministic for tests."""
        job = await self.store.claim(LANE_SQLITE, self.owner, lease_seconds=self._lease)
        if job is None:
            return None
        await self._run(job)
        return job

    async def drain(self) -> int:
        """Run until nothing is queued; returns how many jobs ran."""
        ran = 0
        while await self.run_once() is not None:
            ran += 1
        return ran

    # ------------------------------------------------------------------
    # The lane loop
    # ------------------------------------------------------------------

    async def _lane_loop(self, lane: str) -> None:
        """Claim and run, and survive anything that is not a shutdown.

        A lane that dies on a transient error — a locked database, a failed
        finish — stops the whole queue until the process restarts, which is
        invisible on the board: jobs sit `queued` with nothing to click.  So a
        failure is logged and retried, and only cancellation ends the loop.
        """
        while not self._stopping:
            # Clear before claiming, not after: a wake that arrives while the
            # claim is in flight (a retry landing on an idle lane) has to
            # survive into the wait, or the job sits out the whole poll.
            self._wake.clear()
            try:
                job = await self.store.claim(
                    lane, self.owner, lease_seconds=self._lease
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Jobs: %s lane could not claim", lane)
                await asyncio.sleep(_CLAIM_RETRY_SECONDS)
                continue
            if job is None:
                await self._wait_for_work()
                continue
            try:
                await self._run(job)
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Jobs: %s lane could not finish %s", lane, job.id)
                await asyncio.sleep(_CLAIM_RETRY_SECONDS)

    async def _recovery_loop(self) -> None:
        """Look for an interrupted run while the process is up.

        Startup is not the only moment a lease can expire: a worker can be
        killed by the OS, and a deployment that grew a second process would
        lose it the same way.  One UPDATE every half minute costs nothing.
        """
        while not self._stopping:
            await asyncio.sleep(self._recovery_interval)
            if self._stopping:
                return
            try:
                # A job this process is executing is not an interrupted run:
                # a heartbeat stall past the lease must not hand a live
                # destructive handler a second copy of itself.
                recovered = await self.store.recover(skip=tuple(self._cancel))
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("Jobs: recovery sweep failed")
                continue
            if recovered:
                logger.warning(
                    "Jobs: %d interrupted job(s) returned to the queue", recovered
                )
                self._wake.set()

    async def _wait_for_work(self) -> None:
        # The caller cleared the event before it claimed; clearing here would
        # drop a wake that arrived in between.
        try:
            await asyncio.wait_for(self._wake.wait(), timeout=self._wait)
        except TimeoutError:
            pass

    async def _run(self, job: Job) -> None:
        event = asyncio.Event()
        self._cancel[job.id] = event
        ctx = JobContext(job, self.store, owner=self.owner, cancel=event)
        try:
            try:
                handler = self.registry.get(job.kind)
            except UnknownJobKind as exc:
                await self.store.finish(job.id, self.owner, JOB_FAILED, error=str(exc))
                return
            try:
                result = await handler(ctx)
            except JobCancelled:
                await self.store.finish(
                    job.id,
                    self.owner,
                    JOB_CANCELLED,
                    result=ctx.partial_result(),
                    deleted=ctx.deleted,
                )
            except LeaseLost:
                # Not ours to finish: the row belongs to recovery now, and
                # writing a result would overwrite whoever reclaimed it.
                logger.warning(
                    "Jobs: %s lost its lease; leaving it to recovery", job.id
                )
            except asyncio.CancelledError:
                # Shutting down: hand the job back rather than fail it.
                try:
                    await self.store.release(job.id, self.owner)
                except Exception:  # pragma: no cover - shutdown best effort
                    logger.warning("Jobs: could not release %s", job.id, exc_info=True)
                raise
            except Exception as exc:
                logger.exception("Jobs: %s (%s) failed", job.id, job.kind)
                await self.store.finish(
                    job.id,
                    self.owner,
                    JOB_FAILED,
                    error=str(exc) or exc.__class__.__name__,
                    deleted=ctx.deleted,
                )
            else:
                wrote = await self.store.finish(
                    job.id,
                    self.owner,
                    JOB_SUCCEEDED,
                    result=result or ctx.partial_result() or None,
                    deleted=ctx.deleted,
                )
                if not wrote:
                    # The row belongs to recovery now: this run's result is
                    # discarded, and the board will show the other run's.
                    logger.warning(
                        "Jobs: %s finished after its lease moved; the result "
                        "was discarded",
                        job.id,
                    )
        finally:
            self._cancel.pop(job.id, None)
            self._wake.set()
