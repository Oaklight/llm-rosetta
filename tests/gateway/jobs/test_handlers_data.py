"""The data jobs: hide at enqueue, delete in the worker, audit once.

These are the end-to-end tests for slice 3 — the behaviour the operator sees.
They run a real store, a real ops context and a real worker, so a change that
breaks "the rows disappear now and are deleted later" fails here rather than in
production.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from llm_rosetta_gateway.jobs import JobManager
from llm_rosetta_gateway.jobs.enqueue import (
    enqueue_cleanup,
    enqueue_clear,
    enqueue_rebuild,
    enqueue_trim,
    enqueue_vacuum,
)
from llm_rosetta_gateway.jobs.handlers.data import data_handlers
from llm_rosetta_gateway.observability.metrics import MetricsCollector
from llm_rosetta_gateway.observability.ops_log import OpsLog
from llm_rosetta_gateway.observability.persistence import PersistenceManager
from llm_rosetta_gateway.observability.request_log import RequestLog
from llm_rosetta_gateway.ops.base import OpsContext


def _ago(**kwargs: Any) -> str:
    return (datetime.now(timezone.utc) - timedelta(**kwargs)).isoformat()


@pytest.fixture
async def pm(tmp_path):
    manager = await PersistenceManager.create(str(tmp_path), success_max=1000)
    yield manager
    await manager.close()


@pytest.fixture
def ops(pm):
    return OpsContext(
        ops_log=OpsLog(persistence=pm),
        request_log=RequestLog(persistence=pm),
        metrics=MetricsCollector(),
        persistence=pm,
    )


@pytest.fixture
def manager(pm, ops):
    return JobManager(pm, handlers=data_handlers(ops))


async def _seed(pm: PersistenceManager, *, days: tuple[float, ...] = (0.1, 2, 40)):
    await pm.insert_log_entries(
        [
            {
                "id": f"r{index}",
                "timestamp": _ago(days=age),
                "model": "m",
                "source_provider": "openai",
                "target_provider": "messages",
                "is_stream": False,
                "status_code": 200,
                "duration_ms": 1.0,
                "api_key_label": "k",
                "total_tokens": 10,
                "model_type": "llm",
            }
            for index, age in enumerate(days)
        ]
    )


async def test_clear_hides_at_once_and_deletes_in_the_worker(pm, ops, manager):
    await _seed(pm)
    job = await enqueue_clear(manager, "request_log")

    # The operator's view changed before any deletion happened.
    assert (await pm.query_log_entries(limit=10))[1] == 0
    assert await pm.count_log_entries(include_hidden=True) == 3
    assert (await manager.store.get(job.id)).status == "queued"

    await manager.drain()

    assert await pm.count_log_entries(include_hidden=True) == 0
    assert pm.visibility.ranges("request_log") == []
    entries, _ = await pm.query_ops_log_entries(limit=10)
    cleared = [e for e in entries if e["event_type"] == "data_cleared"]
    assert cleared and cleared[0]["details"]["deleted"] == 3
    assert await pm.check_and_clear_rebuild_flag() is True


async def test_cleanup_age_deletes_only_the_hidden_window(pm, ops, manager):
    await _seed(pm, days=(0.1, 2, 40))
    await enqueue_cleanup(manager, mode="age", tables=("request_log",), max_age_days=30)

    # The forty-day row is older than the cutoff; the other two are not.
    visible, total = await pm.query_log_entries(limit=10)
    assert total == 2
    assert {entry["id"] for entry in visible} == {"r0", "r1"}

    await manager.drain()

    remaining, _ = await pm.query_log_entries(limit=10)
    assert {entry["id"] for entry in remaining} == {"r0", "r1"}
    assert await pm.count_log_entries(include_hidden=True) == 2


async def test_cleanup_range_deletes_only_that_window(pm, ops, manager):
    await _seed(pm, days=(0.1, 2, 40))
    await enqueue_cleanup(
        manager,
        mode="range",
        tables=("request_log",),
        start=_ago(days=3),
        end=_ago(days=1),
    )

    assert (await pm.query_log_entries(limit=10))[1] == 2
    await manager.drain()
    remaining, _ = await pm.query_log_entries(limit=10)
    assert {entry["id"] for entry in remaining} == {"r0", "r2"}


async def test_cancelling_before_the_worker_runs_unhides_everything(pm, ops, manager):
    await _seed(pm)
    job = await enqueue_clear(manager, "request_log")

    await manager.cancel(job.id)

    assert (await pm.query_log_entries(limit=10))[1] == 3
    assert pm.visibility.ranges("request_log") == []
    assert (await manager.store.get(job.id)).status == "cancelled"


async def test_trim_runs_as_a_job(pm, ops, manager):
    await _seed(pm, days=(0.1, 1, 2, 3))
    pm.success_max = 2

    job = await enqueue_trim(manager)
    assert job.kind == "data.trim"
    await manager.drain()

    assert await pm.count_success_entries(include_hidden=True) == 2
    assert (await manager.store.get(job.id)).status == "succeeded"


async def test_rebuild_streams_the_visible_rows(pm, ops, manager):
    await _seed(pm, days=(0.1, 2, 40))
    await pm.add_hidden_range("request_log", _ago(days=1), "", job_id="earlier")

    job = await enqueue_rebuild(manager)
    await manager.drain()

    finished = await manager.store.get(job.id)
    assert finished is not None and finished.status == "succeeded"
    assert finished.result is not None and finished.result["rebuilt_from"] == 2
    assert ops.metrics.total_requests == 2


async def test_vacuum_runs_as_a_job(pm, ops, manager):
    job = await enqueue_vacuum(manager)
    await manager.drain()

    finished = await manager.store.get(job.id)
    assert finished is not None and finished.status == "succeeded"


async def test_clearing_dumps_sweeps_their_bodies(pm, ops, manager):
    await pm.insert_dump_body("hash-1", b"payload", len(b"payload"))
    await pm.insert_error_dump(
        dump_id="d0",
        request_log_id=None,
        timestamp=_ago(hours=1),
        model="m",
        source_provider="openai",
        target_provider="messages",
        provider_name="up",
        status_code=502,
        error_phase="upstream",
        body_hash="hash-1",
        response_text="boom",
        upstream_url="http://example.test",
    )

    await enqueue_clear(manager, "error_dumps")
    assert await pm.count_error_dumps() == 0
    await manager.drain()

    assert await pm.count_error_dumps(include_hidden=True) == 0
    assert await pm.get_dump_body("hash-1") is None


async def test_a_failed_job_keeps_its_range_and_offers_a_retry(pm, ops):
    await _seed(pm)

    async def boom(ctx):
        raise RuntimeError("disk full")

    manager = JobManager(pm, handlers={"data.clear": boom})
    job = await enqueue_clear(manager, "request_log")
    await manager.drain()

    finished = await manager.store.get(job.id)
    assert finished is not None and finished.status == "failed"
    assert "disk full" in (finished.error or "")
    ranges = pm.visibility.ranges("request_log")
    assert len(ranges) == 1 and ranges[0][0] == ""

    # Retrying with a working handler finishes what the first attempt hid.
    healthy = JobManager(pm, handlers=data_handlers(ops))
    assert await healthy.store.retry(job.id) is True
    await healthy.drain()
    assert await pm.count_log_entries(include_hidden=True) == 0
    assert pm.visibility.ranges("request_log") == []
