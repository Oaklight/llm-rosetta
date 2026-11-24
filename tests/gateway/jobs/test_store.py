"""The job store: enqueue, claim, lease, finish, and what a terminal state means."""

from __future__ import annotations

import asyncio
import json
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
from llm_rosetta_gateway.jobs.enqueue import CLEANUP_KIND, CLEAR_KIND
from llm_rosetta_gateway.jobs.model import HIDING_KINDS
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


def test_the_store_names_the_database_it_writes(store, pm):
    """The worker places its instance lock beside this path."""
    assert store.db_path == pm.db_path


def test_a_store_without_a_file_names_none():
    from llm_rosetta_gateway.jobs.store import JobStore

    assert JobStore(object()).db_path is None


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


async def test_failure_releases_the_ranges_and_a_retry_puts_them_back(store, pm):
    job = await store.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )
    await store.claim("sqlite", "owner")

    assert await store.finish(job.id, "owner", JOB_FAILED, error="disk full")
    finished = await store.get(job.id)
    assert finished.error == "disk full"
    assert finished.hidden_range_ids == []
    assert pm.visibility.ranges("request_log") == []

    assert await store.retry(job.id) is True
    assert pm.visibility.ranges("request_log") == [(MIN_TS, MAX_TS)]
    assert len((await store.get(job.id)).hidden_range_ids) == 1


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


async def test_a_cancel_after_a_partial_delete_releases_the_ranges(store, pm):
    job = await store.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )
    await store.claim("sqlite", "owner")

    await store.finish(job.id, "owner", JOB_CANCELLED)
    finished = await store.get(job.id)
    assert finished.hidden_range_ids == []
    assert pm.visibility.ranges("request_log") == []


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


async def test_recovery_at_startup_takes_a_live_foreign_lease(store, pm):
    """A crash-restart inside the lease window must not strand the job.

    The lease is still valid for a minute, so an expiry-only sweep leaves the
    row `running` with its rows hidden and no worker deleting them.
    """
    job = await store.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )
    await pm.connection.execute(
        "UPDATE jobs SET status = 'running', lease_owner = 'a-dead-process', "
        "lease_expires_at = '9999-01-01T00:00:00+00:00' WHERE id = ?",
        (job.id,),
    )
    await pm.connection.commit()

    assert await store.recover() == 0
    assert (await store.get(job.id)).status == JOB_RUNNING

    assert await store.recover(owner="this-process") == 1
    recovered = await store.get(job.id)
    assert recovered.status == JOB_QUEUED
    assert pm.visibility.ranges("request_log") == [(MIN_TS, MAX_TS)]


async def test_recovery_fails_a_job_out_of_attempts(store, pm):
    job = await store.enqueue("data.clear", max_attempts=1)
    await store.claim("sqlite", "owner")
    await _expire_lease(pm, job.id)

    assert await store.recover() == 1
    recovered = await store.get(job.id)
    assert recovered.status == JOB_FAILED
    assert "interrupted" in (recovered.error or "")


async def test_retry_resets_attempts_and_restores_the_window(store, pm):
    job = await store.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )
    claimed = await store.claim("sqlite", "owner")
    assert claimed.attempts == 1
    await store.finish(job.id, "owner", JOB_FAILED, error="boom")
    assert pm.visibility.ranges("request_log") == []

    assert await store.retry(job.id) is True
    retried = await store.get(job.id)
    assert retried.status == JOB_QUEUED and retried.attempts == 0
    assert retried.error is None
    assert pm.visibility.ranges("request_log") == [(MIN_TS, MAX_TS)]


async def test_a_cancelled_job_is_terminal_and_cannot_be_retried(store, pm):
    """Cancelling is final: the operation has to be submitted again."""
    job = await store.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )
    await store.request_cancel(job.id)
    assert pm.visibility.ranges("request_log") == []

    assert await store.retry(job.id) is False
    assert (await store.get(job.id)).status == JOB_CANCELLED
    assert pm.visibility.is_empty()


async def test_a_retry_that_loses_the_race_restores_no_range(store, pm):
    """The status fence comes before the range comes back.

    A job that left the terminal states between the operator's click and the
    UPDATE must not have its window re-hidden on the way out: the rows would go
    invisible with no worker coming to delete them.
    """
    job = await store.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )
    await store.claim("sqlite", "owner")
    await store.finish(job.id, "owner", JOB_FAILED, error="boom")
    await pm.connection.execute(
        "UPDATE jobs SET status = 'running', lease_owner = 'someone' WHERE id = ?",
        (job.id,),
    )
    await pm.connection.commit()

    assert await store.retry(job.id) is False
    assert pm.visibility.ranges("request_log") == []


async def test_retry_of_a_job_with_no_window_adds_no_range(store, pm):
    """Trim and VACUUM never hide anything, so there is nothing to put back."""
    job = await store.enqueue("data.trim")
    await store.claim("sqlite", "owner")
    await store.finish(job.id, "owner", JOB_FAILED, error="boom")

    assert await store.retry(job.id) is True
    assert (await store.get(job.id)).hidden_range_ids == []
    assert pm.visibility.is_empty()


async def test_a_lost_window_refuses_a_retry_instead_of_deleting_visible_rows(
    store, pm
):
    """A pre-`_hide` job whose range is gone must not be retried blindly."""
    job = await store.enqueue(
        "data.clear",
        {"table": "request_log"},
        hidden_ranges=[("request_log", MIN_TS, MAX_TS)],
    )
    await store.claim("sqlite", "owner")
    await store.finish(job.id, "owner", JOB_FAILED, error="boom")
    raw = await pm.connection.execute_fetchone(
        "SELECT payload FROM jobs WHERE id = ?", (job.id,)
    )
    payload = json.loads(raw[0])
    payload.pop("_hide", None)
    await pm.connection.execute(
        "UPDATE jobs SET payload = ? WHERE id = ?", (json.dumps(payload), job.id)
    )
    await pm.connection.commit()

    with pytest.raises(ValueError, match="no recorded hide window"):
        await store.retry(job.id)

    assert (await store.get(job.id)).status == JOB_FAILED
    assert pm.visibility.is_empty()


async def test_startup_repair_releases_a_pre_upgrade_terminal_range(store, pm):
    """A failed job written by the old code kept its range with no route back."""
    job = await store.enqueue(
        "data.clear",
        {"table": "request_log"},
        hidden_ranges=[("request_log", MIN_TS, MAX_TS)],
    )
    await store.claim("sqlite", "owner")
    raw = await pm.connection.execute_fetchone(
        "SELECT payload FROM jobs WHERE id = ?", (job.id,)
    )
    payload = json.loads(raw[0])
    payload.pop("_hide", None)
    await pm.connection.execute(
        "UPDATE jobs SET status = ?, error = 'disk full', payload = ?, "
        "lease_owner = NULL, lease_expires_at = NULL WHERE id = ?",
        (JOB_FAILED, json.dumps(payload), job.id),
    )
    await pm.connection.commit()
    assert pm.visibility.ranges("request_log") == [(MIN_TS, MAX_TS)]

    assert await store.release_terminal_ranges() == 1

    assert pm.visibility.ranges("request_log") == []
    row = await pm.connection.execute_fetchone(
        "SELECT payload, hidden_range_ids FROM jobs WHERE id = ?", (job.id,)
    )
    assert json.loads(row[0])["_hide"] == [["request_log", MIN_TS, MAX_TS]]
    assert json.loads(row[1]) == []
    assert await store.release_terminal_ranges() == 0
    assert await store.retry(job.id) is True
    assert pm.visibility.ranges("request_log") == [(MIN_TS, MAX_TS)]


async def test_startup_repair_keeps_a_maintenance_job_requeueable(store, pm):
    job = await store.enqueue("data.trim")
    await store.claim("sqlite", "owner")
    raw = await pm.connection.execute_fetchone(
        "SELECT payload FROM jobs WHERE id = ?", (job.id,)
    )
    payload = json.loads(raw[0])
    payload.pop("_hide", None)
    await pm.connection.execute(
        "UPDATE jobs SET status = ?, error = 'boom', payload = ?, "
        "lease_owner = NULL, lease_expires_at = NULL WHERE id = ?",
        (JOB_FAILED, json.dumps(payload), job.id),
    )
    await pm.connection.commit()

    assert await store.release_terminal_ranges() == 1

    row = await pm.connection.execute_fetchone(
        "SELECT payload FROM jobs WHERE id = ?", (job.id,)
    )
    assert json.loads(row[0])["_hide"] == []
    assert await store.retry(job.id) is True
    assert pm.visibility.is_empty()


async def test_prune_removes_a_terminal_job_after_it_releases_its_range(store, pm):
    job = await store.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )
    await store.claim("sqlite", "owner")
    await store.finish(job.id, "owner", JOB_FAILED, error="boom")
    assert pm.visibility.ranges("request_log") == []

    removed = await store.prune(keep=0)

    assert removed == 1
    assert await store.get(job.id) is None


async def test_last_job_at_is_kind_status_and_source_scoped(store):
    job = await store.enqueue("data.cleanup", {"source": "periodic"})

    assert await store.last_job_at("data.cleanup") == job.created_at
    assert await store.last_job_at("data.trim") is None
    assert await store.last_job_at("data.cleanup", source="periodic") == (
        job.created_at
    )

    # A queued pass has not done the work yet, and a failed one did not.
    assert (
        await store.last_job_at("data.cleanup", status=JOB_SUCCEEDED, source="periodic")
        is None
    )
    await store.claim("sqlite", "owner")
    await store.finish(job.id, "owner", JOB_FAILED, error="database is locked")
    assert (
        await store.last_job_at("data.cleanup", status=JOB_SUCCEEDED, source="periodic")
        is None
    )

    manual = await store.enqueue("data.cleanup", {"source": "admin"})
    await store.claim("sqlite", "owner")
    await store.finish(manual.id, "owner", JOB_SUCCEEDED)
    assert (
        await store.last_job_at("data.cleanup", status=JOB_SUCCEEDED, source="periodic")
        is None
    )

    periodic = await store.enqueue("data.cleanup", {"source": "periodic"})
    await store.claim("sqlite", "owner")
    await store.finish(periodic.id, "owner", JOB_SUCCEEDED)
    done = await store.get(periodic.id)
    assert done is not None and done.finished_at is not None
    assert (
        await store.last_job_at("data.cleanup", status=JOB_SUCCEEDED, source="periodic")
        == done.finished_at
    )


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


async def test_category_is_filtered_in_sql_so_one_half_cannot_evict_the_other(
    store,
):
    request = await store.enqueue("batch.submit", lane="network")
    network_ops = await store.enqueue("data.clear", lane="network")
    ops = await store.enqueue("data.trim")

    requests, total = await store.list_jobs(category="requests")
    assert total == 1 and [job.id for job in requests] == [request.id]
    assert await store.active_count(category="requests") == 1

    everything_else, total = await store.list_jobs(category="ops")
    assert total == 2
    assert {job.id for job in everything_else} == {network_ops.id, ops.id}
    assert await store.active_count(category="ops") == 2

    with pytest.raises(ValueError, match="unknown job category"):
        await store.list_jobs(category="everything")
    with pytest.raises(ValueError, match="unknown job category"):
        await store.active_count(category="everything")


def test_hiding_kinds_match_the_enqueue_producers():
    """Startup repair must keep naming the kinds that actually hide rows."""
    assert HIDING_KINDS == {CLEAR_KIND, CLEANUP_KIND}


async def test_claim_is_atomic_under_two_claimers(store):
    await store.enqueue("data.clear")

    first, second = await asyncio.gather(
        store.claim("sqlite", "a"), store.claim("sqlite", "b")
    )
    claimed = [job for job in (first, second) if job is not None]
    assert len(claimed) == 1
    assert await store.active_count() == 1


async def test_sweeping_drops_a_range_whose_job_is_gone(store, pm):
    job = await store.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )
    # A job deleted by something other than its own terminal transition.
    await pm.connection.execute("DELETE FROM jobs WHERE id = ?", (job.id,))
    await pm.connection.commit()
    assert pm.visibility.ranges("request_log") == [(MIN_TS, MAX_TS)]

    assert await store.sweep_orphan_ranges() == 1

    assert pm.visibility.ranges("request_log") == []
    assert await pm.count_hidden_entries("request_log") == 0


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
    assert await store.finish(job.id, "owner", JOB_SUCCEEDED) is True

    finished = await store.get(job.id)
    assert finished.status == JOB_FAILED
    assert "hidden rows remain" in (finished.error or "")
    assert finished.hidden_range_ids == []
    assert pm.visibility.ranges("request_log") == []
    assert await pm.count_log_entries() == 1


async def test_success_after_the_rows_are_gone_is_accepted(store, pm):
    """The count is the evidence, not the handler's own counter."""
    job = await store.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )
    await store.claim("sqlite", "owner")

    # A handler that removed the rows but never called note_deleted.
    assert await store.finish(job.id, "owner", JOB_SUCCEEDED) is True

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
    # A settled job releases its reservation; a cancelled one is not requeued.
    assert settled.hidden_range_ids == []
    assert pm.visibility.ranges("request_log") == []
    assert await store.claim("sqlite", "owner") is None
