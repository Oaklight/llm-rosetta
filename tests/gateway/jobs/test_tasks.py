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


def _app_two_ages(manager):
    config = SimpleNamespace(
        _raw={
            "server": {
                "request_log": {"max_age_days": 90},
                "ops_log": {"max_age_days": 30},
            }
        }
    )
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
    job = await manager.enqueue(
        "data.cleanup",
        {"source": "periodic", "tables": ["request_log", "error_dumps"]},
    )
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
    job = await manager.enqueue(
        "data.cleanup",
        {"source": "periodic", "tables": ["request_log", "error_dumps"]},
    )
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
    job = await manager.enqueue(
        "data.cleanup",
        {"source": "periodic", "tables": ["request_log", "error_dumps"]},
    )
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


async def test_one_successful_half_does_not_suppress_the_other(pm):
    """The two ages queue two passes; one success cannot answer for both."""
    manager = JobManager(pm)
    request = await manager.enqueue(
        "data.cleanup",
        {"source": "periodic", "tables": ["request_log", "error_dumps"]},
    )
    await _finish(manager, request, "succeeded")
    ops = await manager.enqueue(
        "data.cleanup",
        {"source": "periodic", "tables": ["ops_log"]},
    )
    await _finish(manager, ops, "failed")
    old = (datetime.now(timezone.utc) - timedelta(days=6)).isoformat()
    await pm.connection.execute(
        "UPDATE jobs SET created_at = ?, finished_at = ? "
        "WHERE status IN ('succeeded', 'failed')",
        (old, old),
    )
    await pm.connection.commit()

    sleeps: list[float] = []

    async def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 2:
            raise asyncio.CancelledError

    with patch("llm_rosetta_gateway.app.tasks.asyncio.sleep", sleep):
        with pytest.raises(asyncio.CancelledError):
            await _periodic_cleanup(_app_two_ages(manager))

    assert sleeps[0] == _CLEANUP_INITIAL_DELAY
    assert sleeps[1] == _CLEANUP_INTERVAL
    jobs, total = await manager.store.list_jobs()
    assert total == 4
    assert sum(1 for job in jobs if job.status == "queued") == 2


async def test_two_successful_passes_defer_to_the_earliest_due(pm):
    """With both halves done, wake at the first deadline, not the last."""
    manager = JobManager(pm)
    now = datetime.now(timezone.utc)
    request = await manager.enqueue(
        "data.cleanup",
        {"source": "periodic", "tables": ["request_log", "error_dumps"]},
    )
    await _finish(manager, request, "succeeded")
    await pm.connection.execute(
        "UPDATE jobs SET created_at = ?, finished_at = ? WHERE id = ?",
        (
            (now - timedelta(days=6)).isoformat(),
            (now - timedelta(days=6)).isoformat(),
            request.id,
        ),
    )
    await pm.connection.commit()
    ops = await manager.enqueue(
        "data.cleanup",
        {"source": "periodic", "tables": ["ops_log"]},
    )
    await _finish(manager, ops, "succeeded")
    await pm.connection.execute(
        "UPDATE jobs SET created_at = ?, finished_at = ? WHERE id = ?",
        (
            (now - timedelta(days=5)).isoformat(),
            (now - timedelta(days=5)).isoformat(),
            ops.id,
        ),
    )
    await pm.connection.commit()

    sleeps: list[float] = []

    async def sleep(seconds):
        sleeps.append(seconds)
        if len(sleeps) >= 2:
            raise asyncio.CancelledError

    with patch("llm_rosetta_gateway.app.tasks.asyncio.sleep", sleep):
        with pytest.raises(asyncio.CancelledError):
            await _periodic_cleanup(_app_two_ages(manager))

    assert sleeps[0] == _CLEANUP_INITIAL_DELAY
    assert 86300 < sleeps[1] <= 86460
    _, total = await manager.store.list_jobs()
    assert total == 2
