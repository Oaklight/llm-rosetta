"""The periodic cleanup's restart guard.

The task sleeps for a week, but the gateway may restart long before that.
The jobs table is the only schedule that survives a restart.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from llm_rosetta_gateway.app.tasks import (
    _CLEANUP_INITIAL_DELAY,
    _CLEANUP_INTERVAL,
    _periodic_cleanup,
)
from llm_rosetta_gateway.jobs import JobManager
from llm_rosetta_gateway.observability.persistence import PersistenceManager


@pytest.fixture
async def pm(tmp_path):
    manager = await PersistenceManager.create(str(tmp_path))
    yield manager
    await manager.close()


def _app(manager):
    config = SimpleNamespace(_raw={"server": {"request_log": {"max_age_days": 90}}})
    return SimpleNamespace(jobs=manager, gateway_config=config)


async def _stop_after_two_sleeps():
    calls = 0

    async def sleep(_seconds):
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise asyncio.CancelledError

    return sleep


async def _finish(manager, job, status):
    await manager.store.claim("sqlite", "owner")
    await manager.store.finish(job.id, "owner", status)


async def test_a_recent_successful_periodic_cleanup_suppresses_the_restart(pm):
    manager = JobManager(pm)
    job = await manager.enqueue("data.cleanup", {"source": "periodic"})
    await _finish(manager, job, "succeeded")
    app = _app(manager)
    sleep = await _stop_after_two_sleeps()

    with patch("llm_rosetta_gateway.app.tasks.asyncio.sleep", sleep):
        with pytest.raises(asyncio.CancelledError):
            await _periodic_cleanup(app)

    jobs, total = await manager.store.list_jobs()
    assert total == 1 and jobs[0].kind == "data.cleanup"


async def test_a_deferral_sleeps_only_the_remainder_of_the_interval(pm):
    """A wake a few seconds early must not add a whole interval to the gap."""
    manager = JobManager(pm)
    job = await manager.enqueue("data.cleanup", {"source": "periodic"})
    await _finish(manager, job, "succeeded")
    finished = datetime.now(timezone.utc) - timedelta(seconds=_CLEANUP_INTERVAL - 60)
    # Ten hours of runtime must not move the next due time earlier.
    created = finished - timedelta(hours=10)
    await pm.connection.execute(
        "UPDATE jobs SET created_at = ?, finished_at = ? WHERE id = ?",
        (created.isoformat(), finished.isoformat(), job.id),
    )
    await pm.connection.commit()

    sleeps: list[float] = []

    async def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 2:
            raise asyncio.CancelledError

    with patch("llm_rosetta_gateway.app.tasks.asyncio.sleep", sleep):
        with pytest.raises(asyncio.CancelledError):
            await _periodic_cleanup(_app(manager))

    assert sleeps[0] == _CLEANUP_INITIAL_DELAY
    assert 55 < sleeps[1] <= 61
    _, total = await manager.store.list_jobs()
    assert total == 1


@pytest.mark.parametrize("status", ["failed", "cancelled"])
async def test_a_cleanup_that_did_not_succeed_does_not_suppress_the_restart(pm, status):
    manager = JobManager(pm)
    job = await manager.enqueue("data.cleanup", {"source": "periodic"})
    await _finish(manager, job, status)
    app = _app(manager)
    sleep = await _stop_after_two_sleeps()

    with patch("llm_rosetta_gateway.app.tasks.asyncio.sleep", sleep):
        with pytest.raises(asyncio.CancelledError):
            await _periodic_cleanup(app)

    jobs, total = await manager.store.list_jobs()
    assert total == 2
    assert [job.payload.get("source") for job in jobs] == ["periodic", "periodic"]


async def test_a_successful_manual_cleanup_does_not_suppress_the_weekly_pass(pm):
    """A one-hour range delete is not the weekly age sweep."""
    manager = JobManager(pm)
    job = await manager.enqueue("data.cleanup", {"source": "admin"})
    await _finish(manager, job, "succeeded")
    app = _app(manager)
    sleep = await _stop_after_two_sleeps()

    with patch("llm_rosetta_gateway.app.tasks.asyncio.sleep", sleep):
        with pytest.raises(asyncio.CancelledError):
            await _periodic_cleanup(app)

    jobs, total = await manager.store.list_jobs()
    assert total == 2
    assert jobs[0].payload.get("source") == "periodic"


async def test_no_recent_cleanup_queues_one(pm):
    manager = JobManager(pm)
    app = _app(manager)
    sleep = await _stop_after_two_sleeps()

    with patch("llm_rosetta_gateway.app.tasks.asyncio.sleep", sleep):
        with pytest.raises(asyncio.CancelledError):
            await _periodic_cleanup(app)

    jobs, total = await manager.store.list_jobs()
    assert total == 1 and jobs[0].kind == "data.cleanup"
    assert jobs[0].payload["source"] == "periodic"
