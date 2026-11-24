"""The rebuild flag's lifetime across the periodic flush and the job.

The flag is a request, not a latch: only a rebuild that scanned the log after
the request may clear it.  These tests pin both failure modes from #1698 —
an enqueue that raises and an enqueue deduplicated onto a running job — and
the token rule that keeps a request made mid-scan alive.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from llm_rosetta_gateway.app.tasks import _periodic_flush
from llm_rosetta_gateway.jobs import JobManager
from llm_rosetta_gateway.jobs.enqueue import enqueue_rebuild
from llm_rosetta_gateway.jobs.handlers.data import data_handlers
from llm_rosetta_gateway.observability.metrics import MetricsCollector
from llm_rosetta_gateway.observability.ops_log import OpsLog
from llm_rosetta_gateway.observability.persistence import PersistenceManager
from llm_rosetta_gateway.ops.base import OpsContext
from llm_rosetta_gateway.ops.data import OpsRebuildMetrics


@pytest.fixture
async def pm(tmp_path):
    manager = await PersistenceManager.create(str(tmp_path))
    yield manager
    await manager.close()


def _app(pm, jobs, metrics):
    return SimpleNamespace(persistence=pm, jobs=jobs, metrics=metrics)


def _ops(pm, metrics=None):
    return OpsContext(
        ops_log=OpsLog(persistence=pm),
        metrics=metrics or MetricsCollector(),
        persistence=pm,
    )


def _one_iteration():
    """A sleep that lets one loop body run, then stops the loop."""
    calls = 0

    async def sleep(_seconds):
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise asyncio.CancelledError

    return sleep


async def test_an_enqueue_failure_leaves_the_request_set(pm):
    jobs = JobManager(pm)
    metrics = MetricsCollector()
    await pm.set_rebuild_flag()

    with patch("llm_rosetta_gateway.app.tasks.asyncio.sleep", _one_iteration()):
        with patch(
            "llm_rosetta_gateway.jobs.enqueue.enqueue_rebuild",
            AsyncMock(side_effect=RuntimeError("store offline")),
        ):
            with pytest.raises(asyncio.CancelledError):
                await _periodic_flush(_app(pm, jobs, metrics))

    assert await pm.rebuild_flag_token() is not None
    _, total = await jobs.store.list_jobs()
    assert total == 0


async def test_a_running_rebuild_does_not_consume_a_newer_request(pm):
    """Dedupe returns the running job; the flag it cannot answer stays."""
    jobs = JobManager(pm)
    job = await enqueue_rebuild(jobs, source="persistence")
    await jobs.store.claim("sqlite", jobs.owner)
    await pm.set_rebuild_flag()

    with patch("llm_rosetta_gateway.app.tasks.asyncio.sleep", _one_iteration()):
        with pytest.raises(asyncio.CancelledError):
            await _periodic_flush(_app(pm, jobs, MetricsCollector()))

    _, total = await jobs.store.list_jobs()
    assert total == 1
    still = await jobs.store.get(job.id)
    assert still is not None and still.status == "running"
    assert await pm.rebuild_flag_token() is not None


async def test_the_rebuild_job_clears_the_request_it_runs(pm):
    jobs = JobManager(pm, handlers=data_handlers(_ops(pm)))
    await pm.set_rebuild_flag()

    with patch("llm_rosetta_gateway.app.tasks.asyncio.sleep", _one_iteration()):
        with pytest.raises(asyncio.CancelledError):
            await _periodic_flush(_app(pm, jobs, MetricsCollector()))

    # Queued, not answered yet: the request survives until the scan runs.
    assert await pm.rebuild_flag_token() is not None
    await jobs.drain()

    assert await pm.rebuild_flag_token() is None
    job = (await jobs.store.list_jobs())[0][0]
    assert job.status == "succeeded"


async def test_a_request_made_mid_scan_survives_the_pass(pm):
    """The running scan answers its own token, not the one that replaced it."""
    await pm.set_rebuild_flag()
    first = await pm.rebuild_flag_token()
    assert isinstance(first, str)

    class _MidScanFlag(MetricsCollector):
        async def rebuild_from_stream(self, rows):
            await pm.set_rebuild_flag()
            return await super().rebuild_from_stream(rows)

    await OpsRebuildMetrics(_ops(pm, _MidScanFlag())).execute()

    second = await pm.rebuild_flag_token()
    assert isinstance(second, str) and second != first
