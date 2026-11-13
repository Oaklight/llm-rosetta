"""The worker: run a handler, record how it ended, and stop when asked."""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Any, cast

import pytest

from llm_rosetta_gateway.jobs import (
    JOB_CANCELLED,
    JOB_FAILED,
    JOB_QUEUED,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    JobCancelled,
    JobManager,
    LeaseLost,
)
from llm_rosetta_gateway.observability.persistence import PersistenceManager


@pytest.fixture
async def pm(tmp_path):
    manager = await PersistenceManager.create(str(tmp_path), success_max=1000)
    yield manager
    await manager.close()


async def _ok(ctx):
    await ctx.progress(1, total=1, unit="rows")
    return {"deleted": 1}


async def _boom(ctx):
    raise ValueError("upstream exploded")


async def _slow_until_cancelled(ctx):
    while True:
        await ctx.checkpoint()
        await asyncio.sleep(0.005)


async def _block_forever(ctx):
    await asyncio.Event().wait()


async def _wait_status(store, job_id, expected, *, timeout: float = 3.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = await store.get(job_id)
        if job is not None and job.status == expected:
            return
        await asyncio.sleep(0.01)
    raise AssertionError(f"job {job_id} never reached {expected}")


async def test_run_once_executes_the_handler(pm):
    manager = JobManager(pm, handlers={"demo.ok": _ok})
    job = await manager.enqueue("demo.ok")

    ran = await manager.run_once()

    assert ran is not None and ran.id == job.id
    finished = await manager.store.get(job.id)
    assert finished is not None
    assert finished.status == JOB_SUCCEEDED
    assert finished.result == {"deleted": 1}
    assert finished.progress_done == 1


async def test_an_unknown_kind_fails_cleanly(pm):
    manager = JobManager(pm)
    job = await manager.enqueue("demo.missing")

    await manager.run_once()

    finished = await manager.store.get(job.id)
    assert finished is not None
    assert finished.status == JOB_FAILED
    assert "demo.missing" in (finished.error or "")


async def test_a_raising_handler_fails_the_job(pm):
    manager = JobManager(pm, handlers={"demo.boom": _boom})
    job = await manager.enqueue("demo.boom")

    await manager.run_once()

    finished = await manager.store.get(job.id)
    assert finished is not None
    assert finished is not None
    assert finished.status == JOB_FAILED
    assert finished.error == "upstream exploded"


async def test_a_handler_stops_at_its_next_checkpoint(pm):
    manager = JobManager(pm, handlers={"demo.slow": _slow_until_cancelled})
    await manager.start()
    try:
        job = await manager.enqueue("demo.slow")
        await _wait_status(manager.store, job.id, JOB_RUNNING)

        found, status = await manager.cancel(job.id)
        assert (found, status) == (True, JOB_RUNNING)
        await _wait_status(manager.store, job.id, JOB_CANCELLED)
    finally:
        await manager.stop()


async def test_shutdown_puts_a_running_job_back_in_the_queue(pm):
    manager = JobManager(pm, handlers={"demo.block": _block_forever})
    await manager.start()
    job = await manager.enqueue("demo.block")
    await _wait_status(manager.store, job.id, JOB_RUNNING)

    await manager.stop()

    recovered = await manager.store.get(job.id)
    assert recovered is not None
    assert recovered.status == JOB_QUEUED
    assert recovered.attempts == 0


async def test_start_recovers_a_job_interrupted_by_the_last_process(pm):
    first = JobManager(pm)
    job = await first.enqueue("demo.ok")
    await first.store.claim("sqlite", "dead-process")
    # The process died holding the lease, so the claim never finished and the
    # lease eventually lapsed.
    past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    await pm.connection.execute(
        "UPDATE jobs SET lease_expires_at = ? WHERE id = ?", (past, job.id)
    )
    await pm.connection.commit()

    second = JobManager(pm, handlers={"demo.ok": _ok})
    await second.start()
    try:
        assert second.recovered == 1
        await _wait_status(second.store, job.id, JOB_SUCCEEDED)
    finally:
        await second.stop()


async def test_drain_runs_every_queued_job(pm):
    manager = JobManager(pm, handlers={"demo.ok": _ok})
    for _ in range(3):
        await manager.enqueue("demo.ok")

    assert await manager.drain() == 3
    assert await manager.store.active_count() == 0


async def test_checkpoint_raises_once_cancelled():
    manager = JobManager(object())
    # A context whose cancel event is already set: the first checkpoint stops.
    job = await _context_job(manager)

    await job.checkpoint()
    job._cancel.set()
    with pytest.raises(JobCancelled):
        await job.checkpoint()


async def test_start_repairs_an_orphaned_range(pm):
    manager = JobManager(pm)
    job = await manager.enqueue(
        "data.clear", hidden_ranges=[("request_log", "", "9999-01-01")]
    )
    await pm.connection.execute("DELETE FROM jobs WHERE id = ?", (job.id,))
    await pm.connection.commit()

    await manager.start()
    try:
        assert manager.orphaned_ranges == 1
        assert pm.visibility.ranges("request_log") == []
    finally:
        await manager.stop()


async def test_pruning_finished_jobs_happens_at_most_once_per_interval(pm):
    manager = JobManager(pm, handlers={"demo.ok": _ok})
    for _ in range(3):
        await manager.enqueue("demo.ok")
    await manager.drain()

    assert await manager.maybe_prune(interval=3600, keep=0) == 3
    assert await manager.maybe_prune(interval=3600, keep=0) == 0


async def _context_job(manager):
    """A JobContext over a harmless stand-in store."""
    from llm_rosetta_gateway.jobs.model import Job
    from llm_rosetta_gateway.jobs.worker import JobContext

    class _Store:
        async def set_progress(self, *args, **kwargs):
            return None

        async def heartbeat(self, *args, **kwargs):
            return True

    job = Job(
        id="j",
        kind="demo",
        payload={},
        lane="sqlite",
        status=JOB_RUNNING,
        priority=100,
        created_at="2026-01-01T00:00:00+00:00",
    )
    return JobContext(
        job,
        cast(Any, _Store()),
        owner=manager.owner,
        cancel=asyncio.Event(),
    )


async def test_a_lost_lease_is_remembered_by_every_later_checkpoint():
    """A loss must not be swallowed by the heartbeat throttle.

    The checkpoint is the handler's own safety gate, and a handler is free to
    catch the first exception and call again inside the 15 s window.
    """
    from llm_rosetta_gateway.jobs.model import Job
    from llm_rosetta_gateway.jobs.worker import JobContext

    class _Store:
        async def set_progress(self, *args, **kwargs):
            return None

        async def heartbeat(self, *args, **kwargs):
            return False

    job = Job(
        id="j",
        kind="demo",
        payload={},
        lane="sqlite",
        status=JOB_RUNNING,
        priority=100,
        created_at="2026-01-01T00:00:00+00:00",
    )
    ctx = JobContext(
        job,
        cast(Any, _Store()),
        owner="owner",
        cancel=asyncio.Event(),
    )

    # The forced call is the one that reaches the store and learns the row
    # moved; the checkpoint that follows is inside the throttle window.
    with pytest.raises(LeaseLost):
        await ctx.heartbeat(force=True)
    with pytest.raises(LeaseLost):
        await ctx.checkpoint()


async def test_a_lost_lease_stops_the_handler(manager=None):
    """A heartbeat that says the lease is gone is not a warning."""
    from llm_rosetta_gateway.jobs.model import Job
    from llm_rosetta_gateway.jobs.worker import JobContext

    class _Store:
        async def set_progress(self, *args, **kwargs):
            return None

        async def heartbeat(self, *args, **kwargs):
            return False

    job = Job(
        id="j",
        kind="demo",
        payload={},
        lane="sqlite",
        status=JOB_RUNNING,
        priority=100,
        created_at="2026-01-01T00:00:00+00:00",
    )
    ctx = JobContext(
        job,
        cast(Any, _Store()),
        owner="owner",
        cancel=asyncio.Event(),
        heartbeat_interval=0,
    )

    with pytest.raises(LeaseLost):
        await ctx.checkpoint()
