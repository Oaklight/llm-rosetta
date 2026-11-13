"""The job store: enqueue, claim, lease, finish, and what a terminal state means."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pytest

from llm_rosetta_gateway.jobs import (
    JOB_CANCELLED,
    JOB_FAILED,
    JOB_QUEUED,
    JOB_RUNNING,
    JOB_SUCCEEDED,
    JobStore,
)
from llm_rosetta_gateway.observability.persistence import PersistenceManager
from llm_rosetta_gateway.observability.visibility import MAX_TS, MIN_TS


@pytest.fixture
async def pm(tmp_path):
    manager = await PersistenceManager.create(str(tmp_path), success_max=1000)
    yield manager
    await manager.close()


@pytest.fixture
def store(pm):
    return JobStore(pm)


async def _expire_lease(pm: PersistenceManager, job_id: str) -> None:
    past = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    await pm.connection.execute(
        "UPDATE jobs SET lease_expires_at = ? WHERE id = ?", (past, job_id)
    )
    await pm.connection.commit()


async def test_enqueue_hides_its_ranges_immediately(store, pm):
    job = await store.enqueue(
        "data.clear",
        {"table": "request_log"},
        hidden_ranges=[("request_log", MIN_TS, MAX_TS)],
    )

    assert job.status == JOB_QUEUED
    assert job.payload == {"table": "request_log"}
    assert pm.visibility.ranges("request_log") == [(MIN_TS, MAX_TS)]
    row = await pm.connection.execute_fetchone(
        "SELECT job_id FROM hidden_ranges WHERE id = ?", (job.hidden_range_ids[0],)
    )
    assert row[0] == job.id


async def test_a_dedupe_key_returns_the_active_job(store):
    first = await store.enqueue("data.trim", dedupe_key="trim")
    second = await store.enqueue("data.trim", dedupe_key="trim")
    assert second.id == first.id

    await store.claim("sqlite", "owner")
    await store.finish(first.id, "owner", JOB_SUCCEEDED)
    third = await store.enqueue("data.trim", dedupe_key="trim")
    assert third.id != first.id


async def test_claim_takes_one_job_at_a_time(store):
    await store.enqueue("data.clear", priority=100)
    low = await store.enqueue("data.clear", priority=10)

    claimed = await store.claim("sqlite", "owner")
    assert claimed is not None and claimed.id == low.id
    assert claimed.status == JOB_RUNNING and claimed.attempts == 1
    assert await store.claim("sqlite", "owner") is not None
    assert await store.claim("sqlite", "owner") is None


async def test_heartbeat_renews_only_the_owners_lease(store):
    job = await store.enqueue("data.clear")
    await store.claim("sqlite", "owner")

    assert await store.heartbeat(job.id, "owner") is True
    assert await store.heartbeat(job.id, "someone-else") is False


async def test_progress_is_recorded(store):
    job = await store.enqueue("data.clear")
    await store.claim("sqlite", "owner")

    await store.set_progress(job.id, 5000, total=20000, unit="rows")
    job = await store.get(job.id)
    assert (job.progress_done, job.progress_total, job.progress_unit) == (
        5000,
        20000,
        "rows",
    )


async def test_success_drops_the_ranges(store, pm):
    job = await store.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )
    await store.claim("sqlite", "owner")

    assert await store.finish(job.id, "owner", JOB_SUCCEEDED, result={"deleted": 3})
    finished = await store.get(job.id)
    assert finished.status == JOB_SUCCEEDED
    assert finished.result == {"deleted": 3}
    assert finished.hidden_range_ids == []
    assert pm.visibility.ranges("request_log") == []


async def test_failure_keeps_the_ranges_so_a_retry_finishes_the_job(store, pm):
    job = await store.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )
    await store.claim("sqlite", "owner")

    assert await store.finish(job.id, "owner", JOB_FAILED, error="disk full")
    finished = await store.get(job.id)
    assert finished.error == "disk full"
    assert len(finished.hidden_range_ids) == 1
    assert pm.visibility.ranges("request_log") == [(MIN_TS, MAX_TS)]


async def test_cancelling_a_queued_job_is_a_clean_undo(store, pm):
    job = await store.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )

    found, status = await store.request_cancel(job.id)
    assert (found, status) == (True, JOB_CANCELLED)
    assert pm.visibility.ranges("request_log") == []
    assert (await store.get(job.id)).hidden_range_ids == []


async def test_cancelling_a_running_job_only_flags_it(store, pm):
    job = await store.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )
    await store.claim("sqlite", "owner")

    found, status = await store.request_cancel(job.id)
    assert (found, status) == (True, JOB_RUNNING)
    assert (await store.get(job.id)).cancel_requested is True
    assert pm.visibility.ranges("request_log") == [(MIN_TS, MAX_TS)]


async def test_a_cancel_after_a_partial_delete_keeps_the_ranges(store, pm):
    job = await store.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )
    await store.claim("sqlite", "owner")

    await store.finish(job.id, "owner", JOB_CANCELLED, deleted=2)
    finished = await store.get(job.id)
    assert len(finished.hidden_range_ids) == 1
    assert pm.visibility.ranges("request_log") == [(MIN_TS, MAX_TS)]


async def test_release_gives_the_attempt_back(store):
    job = await store.enqueue("data.clear")
    claimed = await store.claim("sqlite", "owner")
    assert claimed.attempts == 1

    assert await store.release(job.id, "owner") is True
    job = await store.get(job.id)
    assert job.status == JOB_QUEUED and job.attempts == 0


async def test_recovery_requeues_an_expired_lease(store, pm):
    job = await store.enqueue("data.clear")
    await store.claim("sqlite", "owner")
    await _expire_lease(pm, job.id)

    assert await store.recover() == 1
    recovered = await store.get(job.id)
    assert recovered.status == JOB_QUEUED
    row = await pm.connection.execute_fetchone(
        "SELECT lease_owner FROM jobs WHERE id = ?", (job.id,)
    )
    assert row[0] is None


async def test_recovery_fails_a_job_out_of_attempts(store, pm):
    job = await store.enqueue("data.clear", max_attempts=1)
    await store.claim("sqlite", "owner")
    await _expire_lease(pm, job.id)

    assert await store.recover() == 1
    recovered = await store.get(job.id)
    assert recovered.status == JOB_FAILED
    assert "interrupted" in (recovered.error or "")


async def test_retry_resets_and_resume_continues(store, pm):
    job = await store.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )
    claimed = await store.claim("sqlite", "owner")
    assert claimed.attempts == 1
    await store.finish(job.id, "owner", JOB_CANCELLED, deleted=1)

    assert await store.resume(job.id) is True
    resumed = await store.get(job.id)
    assert resumed.status == JOB_QUEUED and resumed.attempts == 1
    assert pm.visibility.ranges("request_log") == [(MIN_TS, MAX_TS)]

    await store.claim("sqlite", "owner")
    await store.finish(job.id, "owner", JOB_FAILED, error="boom")
    assert await store.retry(job.id) is True
    retried = await store.get(job.id)
    assert retried.status == JOB_QUEUED and retried.attempts == 0
    assert retried.error is None


async def test_resume_after_a_clean_undo_hides_the_rows_again(store, pm):
    """A clean-undo cancel dropped the range; resuming must put it back.

    Otherwise the resumed job deletes rows that no read is hiding: they
    disappear on the first batch instead of leaving the listings when the
    operator resumes, which is the whole promise of the hide-at-enqueue design.
    """
    job = await store.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )
    await store.request_cancel(job.id)
    assert pm.visibility.ranges("request_log") == []

    assert await store.resume(job.id) is True
    assert pm.visibility.ranges("request_log") == [(MIN_TS, MAX_TS)]
    resumed = await store.get(job.id)
    assert len(resumed.hidden_range_ids) == 1
    assert "_hide" not in resumed.to_payload()["payload"]


async def test_a_requeue_that_loses_the_race_restores_no_range(store, pm):
    """The status fence comes before the range comes back.

    A job that left the terminal states between the operator's click and the
    UPDATE must not have its window re-hidden on the way out: the rows would go
    invisible with no worker coming to delete them.
    """
    job = await store.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )
    await store.request_cancel(job.id)
    assert pm.visibility.ranges("request_log") == []
    await pm.connection.execute(
        "UPDATE jobs SET status = 'running', lease_owner = 'someone' WHERE id = ?",
        (job.id,),
    )
    await pm.connection.commit()

    assert await store.resume(job.id) is False
    assert pm.visibility.ranges("request_log") == []


async def test_resume_of_a_job_with_no_window_adds_no_range(store, pm):
    """Trim and VACUUM never hide anything, so there is nothing to put back."""
    job = await store.enqueue("data.trim")
    await store.claim("sqlite", "owner")
    await store.finish(job.id, "owner", JOB_CANCELLED, deleted=0)

    assert await store.resume(job.id) is True
    assert (await store.get(job.id)).hidden_range_ids == []
    assert pm.visibility.is_empty()


async def test_unhide_makes_the_rows_visible_again(store, pm):
    job = await store.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )
    await store.claim("sqlite", "owner")
    await store.finish(job.id, "owner", JOB_FAILED, error="boom")

    assert await store.unhide(job.id) == 1
    assert pm.visibility.ranges("request_log") == []
    assert (await store.get(job.id)).hidden_range_ids == []


async def test_prune_keeps_a_job_that_owns_a_range(store, pm):
    owner = await store.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )
    await store.claim("sqlite", "owner")
    await store.finish(owner.id, "owner", JOB_FAILED, error="boom")
    for _ in range(3):
        job = await store.enqueue("data.trim")
        await store.claim("sqlite", "owner")
        await store.finish(job.id, "owner", JOB_SUCCEEDED)

    removed = await store.prune(keep=0)

    assert removed == 3
    assert await store.get(owner.id) is not None
    assert await store.get(job.id) is None


async def test_list_filters_and_counts(store):
    first = await store.enqueue("data.clear")
    await store.enqueue("data.trim")
    await store.claim("sqlite", "owner")
    await store.finish(first.id, "owner", JOB_SUCCEEDED)

    jobs, total = await store.list_jobs()
    assert total == 2 and len(jobs) == 2
    only_clear, count = await store.list_jobs(kind="data.clear")
    assert count == 1 and only_clear[0].id == first.id
    succeeded, count = await store.list_jobs(status=JOB_SUCCEEDED)
    assert count == 1 and succeeded[0].id == first.id
    assert await store.active_count() == 1


async def test_claim_is_atomic_under_two_claimers(store):
    await store.enqueue("data.clear")

    first, second = await asyncio.gather(
        store.claim("sqlite", "a"), store.claim("sqlite", "b")
    )
    claimed = [job for job in (first, second) if job is not None]
    assert len(claimed) == 1
    assert await store.active_count() == 1


async def test_success_while_the_range_still_covers_rows_fails_the_job(store, pm):
    """A handler that returns without deleting must not resurrect its rows."""
    await pm.insert_log_entries(
        [
            {
                "id": "r1",
                "timestamp": "2026-01-01T00:00:00+00:00",
                "model": "m",
                "source_provider": "s",
                "target_provider": "t",
                "is_stream": False,
                "status_code": 200,
                "duration_ms": 1.0,
            }
        ]
    )
    job = await store.enqueue(
        "data.clear",
        hidden_ranges=[("request_log", "", "2027-01-01T00:00:00+00:00")],
    )
    await store.claim("sqlite", "owner")

    # The handler did nothing; `deleted` may even look fine.
    assert await store.finish(job.id, "owner", JOB_SUCCEEDED, deleted=1) is True

    finished = await store.get(job.id)
    assert finished.status == JOB_FAILED
    assert "hidden rows remain" in (finished.error or "")
    assert len(finished.hidden_range_ids) == 1
    assert pm.visibility.ranges("request_log") != []
    assert await pm.count_log_entries() == 0


async def test_success_after_the_rows_are_gone_is_accepted(store, pm):
    """The count is the evidence, not the handler's own counter."""
    job = await store.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )
    await store.claim("sqlite", "owner")

    # A handler that removed the rows but never called note_deleted.
    assert await store.finish(job.id, "owner", JOB_SUCCEEDED, deleted=0) is True

    finished = await store.get(job.id)
    assert finished.status == JOB_SUCCEEDED
    assert finished.hidden_range_ids == []
    assert pm.visibility.ranges("request_log") == []


async def test_recover_settles_a_job_the_operator_cancelled(store, pm):
    """Cancel-while-running followed by a restart must not wedge the job."""
    job = await store.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )
    await store.claim("sqlite", "owner")
    await store.request_cancel(job.id)
    await _expire_lease(pm, job.id)

    assert await store.recover() == 1

    settled = await store.get(job.id)
    assert settled.status == JOB_CANCELLED
    assert settled.cancel_requested is True
    # The ranges stay: deletion may have run partially before the restart.
    assert len(settled.hidden_range_ids) == 1
    assert await store.claim("sqlite", "owner") is None


async def test_unhiding_a_queued_job_cancels_it(store, pm):
    """The only honest unhide for a job that has not run is to cancel it.

    Leaving it queued would bring the rows back now and delete them again on
    the job's first batch, without anything in the listing having changed.
    """
    job = await store.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )

    assert await store.unhide(job.id) == 1
    assert pm.visibility.ranges("request_log") == []
    cancelled = await store.get(job.id)
    assert cancelled.status == JOB_CANCELLED
    assert cancelled.cancel_requested is True
    assert cancelled.hidden_range_ids == []
    assert await store.claim("sqlite", "owner") is None


async def test_unhiding_a_running_job_is_refused(store, pm):
    job = await store.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )
    await store.claim("sqlite", "owner")

    with pytest.raises(ValueError, match="running"):
        await store.unhide(job.id)

    assert pm.visibility.ranges("request_log") == [(MIN_TS, MAX_TS)]
