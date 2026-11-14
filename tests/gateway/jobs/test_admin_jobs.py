"""The job routes: the payloads the admin panel reads and the actions it offers."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import pytest

from llm_rosetta_gateway.admin.routes import jobs as routes
from llm_rosetta_gateway.jobs import JOB_CANCELLED, JobManager
from llm_rosetta_gateway.observability.persistence import PersistenceManager
from llm_rosetta_gateway.observability.visibility import MAX_TS, MIN_TS


class _App:
    def __init__(self, jobs: Any = None) -> None:
        self.jobs = jobs


class _Request:
    def __init__(self, app: _App, **path_params: str) -> None:
        self.app = app
        self.path_params = path_params
        self.query_params: dict[str, str] = {}


def _body(response) -> dict[str, Any]:
    return json.loads(response.body)


@pytest.fixture
async def pm(tmp_path):
    manager = await PersistenceManager.create(str(tmp_path), success_max=1000)
    yield manager
    await manager.close()


@pytest.fixture
def manager(pm):
    return JobManager(pm)


async def test_listing_reports_the_active_count(manager):
    await manager.enqueue("data.clear")
    await manager.enqueue("data.trim")
    claimed = await manager.store.claim("sqlite", "owner")
    await manager.store.finish(claimed.id, "owner", "succeeded")

    body = _body(await routes.get_jobs(_Request(_App(manager))))

    assert body["total"] == 2
    assert body["active"] == 1
    assert {job["kind"] for job in body["jobs"]} == {"data.clear", "data.trim"}
    assert all("payload" in job for job in body["jobs"])


async def test_retry_wakes_the_lane(manager, monkeypatch):
    """Without the wake the job waits out the lane's 30-second poll."""
    woken: list[bool] = []
    monkeypatch.setattr(manager, "wake", lambda: woken.append(True))
    await manager.enqueue("data.clear")
    claimed = await manager.store.claim("sqlite", "owner")
    await manager.store.finish(claimed.id, "owner", "failed", error="boom")

    response = await routes.retry_job(_Request(_App(manager), job_id=claimed.id))

    assert _body(response)["status"] == "queued"
    assert woken == [True]


async def test_retry_is_picked_up_without_waiting_out_the_poll(pm):
    """The latency the wake exists for, measured through the route.

    A spy proves the call, not the effect; this drives a real lane with a
    2-second poll and requires the retried job to finish well inside it.
    """
    calls = {"n": 0}

    async def flaky(ctx):
        calls["n"] += 1
        if calls["n"] == 1:
            raise ValueError("first run fails")
        return {"deleted": 1}

    lane_manager = JobManager(pm, handlers={"demo.flaky": flaky}, wait_seconds=2.0)
    await lane_manager.start()
    try:
        job = await lane_manager.enqueue("demo.flaky")
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            current = await lane_manager.store.get(job.id)
            assert current is not None
            if current.status == "failed":
                break
            await asyncio.sleep(0.01)

        started = time.monotonic()
        await routes.retry_job(_Request(_App(lane_manager), job_id=job.id))
        deadline = time.monotonic() + 1.5
        status = "queued"
        while time.monotonic() < deadline:
            current = await lane_manager.store.get(job.id)
            assert current is not None
            status = current.status
            if status == "succeeded":
                break
            await asyncio.sleep(0.01)
        elapsed = time.monotonic() - started
        assert status == "succeeded", f"retry left the job {status}"
        assert elapsed < 1.0, f"retry waited {elapsed:.2f}s with a 2s poll"
    finally:
        await lane_manager.stop()


async def test_resume_wakes_the_lane(manager, monkeypatch):
    woken: list[bool] = []
    monkeypatch.setattr(manager, "wake", lambda: woken.append(True))
    await manager.enqueue("data.clear")
    claimed = await manager.store.claim("sqlite", "owner")
    await manager.store.finish(claimed.id, "owner", "cancelled", deleted=1)

    response = await routes.resume_job(_Request(_App(manager), job_id=claimed.id))

    assert _body(response)["status"] == "queued"
    assert woken == [True]


async def test_an_unknown_job_is_a_404(manager):
    response = await routes.get_job(_Request(_App(manager), job_id="nope"))

    assert response.status_code == 404
    assert _body(response)["error"] == "Job not found"


async def test_cancelling_a_queued_job_unhides_its_rows(manager, pm):
    job = await manager.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )

    response = await routes.cancel_job(_Request(_App(manager), job_id=job.id))

    assert _body(response) == {"ok": True, "status": JOB_CANCELLED}
    assert pm.visibility.ranges("request_log") == []


async def test_retry_rejects_a_job_that_has_not_failed(manager):
    job = await manager.enqueue("data.clear")

    response = await routes.retry_job(_Request(_App(manager), job_id=job.id))

    assert response.status_code == 409
    assert "failed or cancelled" in _body(response)["error"]


async def test_unhide_reports_the_ranges_it_removed(manager, pm):
    job = await manager.enqueue(
        "data.clear", hidden_ranges=[("request_log", MIN_TS, MAX_TS)]
    )
    await manager.store.claim("sqlite", "owner")
    await manager.store.finish(job.id, "owner", "failed", error="boom")

    response = await routes.unhide_job(_Request(_App(manager), job_id=job.id))

    assert _body(response) == {"ok": True, "ranges_removed": 1}
    assert pm.visibility.ranges("request_log") == []
    assert (await manager.store.get(job.id)).status == "failed"


async def test_without_persistence_the_routes_say_so():
    response = await routes.get_jobs(_Request(_App(None)))

    assert response.status_code == 400
    assert "persistence" in _body(response)["error"]


async def test_a_bad_limit_is_a_400(manager):
    request = _Request(_App(manager))
    request.query_params = {"limit": "many"}

    response = await routes.get_jobs(request)

    assert response.status_code == 400
    assert "integers" in _body(response)["error"]


async def test_cancel_after_a_job_was_cleared_reports_its_state(manager):
    job = await manager.enqueue("data.clear")
    await manager.store.request_cancel(job.id)

    response = await routes.cancel_job(_Request(_App(manager), job_id=job.id))

    assert _body(response) == {"ok": True, "status": JOB_CANCELLED}
    assert (await manager.store.get(job.id)).status == JOB_CANCELLED
