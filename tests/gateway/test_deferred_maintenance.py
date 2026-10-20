"""The deferred steps: failure isolation, per-step results, and the summary.

Slice 2 of #1027.  The five repairs that used to run under a single ``try`` now
run as a registry, so what is pinned here is the behaviour the epic exists for —
one failing step no longer skips the ones behind it — plus the two things an
operator reads afterwards: the per-step results and the readiness summary.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest
from unittest.mock import AsyncMock, MagicMock

from llm_rosetta.gateway.maintenance.backfills import build_registry
from llm_rosetta.gateway.maintenance.startup import DeferredStartup
from llm_rosetta.gateway.routing import ModelRoute, ProviderEntry
from llm_rosetta.observability.maintenance import Policy, StepContext

_ALL_STEPS = [
    "request_log.backfill_provider_names",
    "api_keys.backfill_last_used",
    "request_log.backfill_total_tokens",
    "request_log.backfill_usage_buckets",
    "error_dumps.backfill_log_ids",
]


class _App:
    def __init__(self, *, persistence: Any = None, keystore: Any = None) -> None:
        self.persistence = persistence
        self.keystore = keystore
        self.gateway_config = None
        self.metrics = None
        self._counter_rebuild_needed = False
        self._token_refresh_tasks: list = []
        # Attached by the readiness-probe tests, as run_gateway attaches it.
        self.deferred_startup: Any = None


class _Config:
    def __init__(self, *, models: dict | None = None, aliases: dict | None = None):
        self.providers: dict = {}
        self.models = models if models is not None else {}
        self.model_upstream_names = aliases if aliases is not None else {}


def _config_with_model() -> _Config:
    return _Config(
        models={
            "gpt-4o": ModelRoute(
                providers=[ProviderEntry(name="openai")], route_id="gpt-4o"
            )
        }
    )


def _stores(*, keystore: bool = True) -> tuple[Any, Any]:
    persistence = MagicMock()
    persistence.db_path = "/tmp/gateway.db"
    persistence.backfill_provider_names = AsyncMock(return_value=3)
    persistence.backfill_total_tokens = AsyncMock(return_value=7)
    persistence.backfill_usage_buckets = AsyncMock(return_value=4)
    persistence.backfill_error_dump_log_ids = AsyncMock(return_value=1)
    key_store = None
    if keystore:
        key_store = MagicMock()
        key_store.backfill_last_used = AsyncMock(return_value=2)
        key_store.label_bucket_map = AsyncMock(return_value={"wired": "bucket-1"})
    return persistence, key_store


class TestWhichStepsExist:
    """The registry is the steps this deployment can actually run."""

    def test_every_step_is_registered_when_both_stores_are_there(self):
        persistence, key_store = _stores()
        registry = build_registry(_App(persistence=persistence, keystore=key_store))
        assert [step.name for step in registry.steps] == _ALL_STEPS
        assert registry.policy is Policy.DEFERRED, (
            "the whole point: a failure here is recorded, never fatal"
        )

    def test_every_step_says_which_store_it_repairs(self):
        """This registry spans two stores, so each result has to name its own.

        "Which store is half-repaired" is the first question after a partial
        run, and the answer must not require parsing it back out of a step name.
        """
        persistence, key_store = _stores()
        registry = build_registry(_App(persistence=persistence, keystore=key_store))
        stores = {step.name: step.store for step in registry.steps}

        assert stores["api_keys.backfill_last_used"] == "keys.db"
        assert stores["request_log.backfill_provider_names"] == "gateway.db"
        assert stores["error_dumps.backfill_log_ids"] == "gateway.db"
        assert set(stores.values()) == {"gateway.db", "keys.db"}

    def test_the_key_store_steps_are_absent_without_a_key_store(self):
        persistence, _ = _stores(keystore=False)
        registry = build_registry(_App(persistence=persistence))
        assert [step.name for step in registry.steps] == [
            "request_log.backfill_provider_names",
            "request_log.backfill_total_tokens",
            "error_dumps.backfill_log_ids",
        ]

    def test_a_store_with_nothing_to_repair_has_no_steps(self):
        assert build_registry(_App()).steps == []


class TestWhatTheStepsRead:
    """A step resolves its own inputs, so re-running one needs only the context."""

    async def test_the_provider_map_comes_from_the_config(self):
        persistence, key_store = _stores()
        registry = build_registry(_App(persistence=persistence, keystore=key_store))
        await registry.run_step(
            "request_log.backfill_provider_names",
            StepContext(config=_config_with_model(), persistence=persistence),
        )
        persistence.backfill_provider_names.assert_awaited_once_with(
            {"gpt-4o": "openai"}
        )

    async def test_the_bucket_map_comes_from_the_key_store(self):
        persistence, key_store = _stores()
        registry = build_registry(_App(persistence=persistence, keystore=key_store))
        await registry.run_step(
            "request_log.backfill_usage_buckets",
            StepContext(persistence=persistence, keystore=key_store),
        )
        persistence.backfill_usage_buckets.assert_awaited_once_with(
            {"wired": "bucket-1"}
        )

    async def test_the_aliases_come_from_the_config(self):
        persistence, key_store = _stores()
        registry = build_registry(_App(persistence=persistence, keystore=key_store))
        await registry.run_step(
            "error_dumps.backfill_log_ids",
            StepContext(
                config=_Config(aliases={"gpt-4o": "openai/gpt-4o"}),
                persistence=persistence,
            ),
        )
        persistence.backfill_error_dump_log_ids.assert_awaited_once_with(
            model_aliases={"gpt-4o": "openai/gpt-4o"}
        )

    async def test_a_step_with_no_config_repairs_nothing(self):
        """Which is the same answer as "already repaired", and not an error."""
        persistence, key_store = _stores()
        registry = build_registry(_App(persistence=persistence, keystore=key_store))
        result = await registry.run_step(
            "request_log.backfill_provider_names",
            StepContext(persistence=persistence),
        )
        assert result.changed == 0
        persistence.backfill_provider_names.assert_not_awaited()


class TestTheFailureIsolation:
    """The defect the epic names: one failure used to skip the other four."""

    async def test_a_failing_step_does_not_skip_the_ones_behind_it(self):
        persistence, key_store = _stores()
        persistence.backfill_provider_names = AsyncMock(
            side_effect=RuntimeError("database is locked")
        )
        registry = build_registry(_App(persistence=persistence, keystore=key_store))

        results = await registry.run(
            StepContext(
                config=_config_with_model(), persistence=persistence, keystore=key_store
            )
        )

        assert [result.ok for result in results] == [False, True, True, True, True]
        assert results[0].error == "RuntimeError: database is locked"
        assert results[0].changed == 0, "what a failed step half-did is not a count"
        persistence.backfill_total_tokens.assert_awaited_once()
        persistence.backfill_usage_buckets.assert_awaited_once()
        persistence.backfill_error_dump_log_ids.assert_awaited_once()
        key_store.backfill_last_used.assert_awaited_once()

    async def test_the_steps_that_did_run_still_report_what_they_changed(self):
        persistence, key_store = _stores()
        registry = build_registry(_App(persistence=persistence, keystore=key_store))
        await registry.run(
            StepContext(
                config=_config_with_model(), persistence=persistence, keystore=key_store
            )
        )
        assert {result.name: result.changed for result in registry.results} == {
            "request_log.backfill_provider_names": 3,
            "api_keys.backfill_last_used": 2,
            "request_log.backfill_total_tokens": 7,
            "request_log.backfill_usage_buckets": 4,
            "error_dumps.backfill_log_ids": 1,
        }


class TestRunningThroughTheStartupTask:
    """``_run_backfills`` is the runner, and readiness still clears."""

    async def test_the_registry_runs_and_stays_on_the_manager(self):
        persistence, key_store = _stores()
        app = _App(persistence=persistence, keystore=key_store)
        startup = DeferredStartup(_config_with_model(), app)
        startup._pending_tasks.add("backfills")

        await startup._run_backfills()

        assert startup.maintenance is not None
        assert len(startup.maintenance.results) == len(_ALL_STEPS)
        assert "backfills" not in startup._pending_tasks

    async def test_a_failure_still_clears_the_pending_flag(self):
        """Readiness is about the task finishing, not about the repairs working."""
        persistence, _ = _stores(keystore=False)
        persistence.backfill_total_tokens = AsyncMock(
            side_effect=RuntimeError("db error")
        )
        startup = DeferredStartup(_config_with_model(), _App(persistence=persistence))
        startup._pending_tasks.add("backfills")

        await startup._run_backfills()

        assert "backfills" not in startup._pending_tasks
        assert startup.maintenance_summary()["state"] == "partial"

    async def test_without_persistence_there_is_nothing_to_run(self):
        startup = DeferredStartup(_config_with_model(), _App())
        startup._pending_tasks.add("backfills")
        await startup._run_backfills()
        assert startup.maintenance is None
        assert "backfills" not in startup._pending_tasks


class TestTheReadinessSummary:
    """How a probe tells "done" from "done, with three repairs failed"."""

    def test_before_the_steps_run_it_says_so(self):
        startup = DeferredStartup(_config_with_model(), _App())
        assert startup.maintenance_summary() == {"state": "not_started", "steps": []}

    async def test_a_clean_run_is_done(self):
        persistence, key_store = _stores()
        startup = DeferredStartup(
            _config_with_model(), _App(persistence=persistence, keystore=key_store)
        )
        await startup._run_backfills()
        summary = startup.maintenance_summary()
        assert summary["state"] == "done"
        assert summary["failed"] == []
        assert [step["name"] for step in summary["steps"]] == _ALL_STEPS

    async def test_a_failed_step_makes_it_partial_and_is_named(self):
        persistence, key_store = _stores()
        persistence.backfill_usage_buckets = AsyncMock(
            side_effect=RuntimeError("no such table")
        )
        startup = DeferredStartup(
            _config_with_model(), _App(persistence=persistence, keystore=key_store)
        )
        await startup._run_backfills()
        summary = startup.maintenance_summary()
        assert summary["state"] == "partial"
        assert summary["failed"] == ["request_log.backfill_usage_buckets"]

    async def test_the_summary_says_which_store_each_step_touched(self):
        """The reader of a partial run is asking which database to look at."""
        persistence, key_store = _stores()
        key_store.backfill_last_used = AsyncMock(
            side_effect=RuntimeError("database is locked")
        )
        startup = DeferredStartup(
            _config_with_model(), _App(persistence=persistence, keystore=key_store)
        )
        await startup._run_backfills()

        summary = startup.maintenance_summary()
        stores = {step["name"]: step["store"] for step in summary["steps"]}
        assert summary["state"] == "partial"
        assert stores["api_keys.backfill_last_used"] == "keys.db", (
            "the failed step's store is the one an operator has to open"
        )
        assert stores["request_log.backfill_total_tokens"] == "gateway.db"

    async def test_nothing_ran_is_not_success(self):
        """A registry that recorded no result is not a finished registry.

        Reachable in production by a cancellation before the first step: the
        runner re-raises ``CancelledError`` rather than recording it, so the
        summary is asked about a registry holding steps and no results.  A probe
        branching on ``done`` must not read that as repaired.
        """
        persistence, key_store = _stores()
        startup = DeferredStartup(
            _config_with_model(), _App(persistence=persistence, keystore=key_store)
        )
        startup.maintenance = build_registry(_App(persistence=persistence))

        summary = startup.maintenance_summary()

        assert summary["state"] == "not_started"
        assert summary["steps"] == []

    async def test_a_partly_recorded_run_is_partial_not_done(self):
        """Every registered step has to have a result for ``done``."""
        persistence, key_store = _stores()
        startup = DeferredStartup(
            _config_with_model(), _App(persistence=persistence, keystore=key_store)
        )
        registry = build_registry(_App(persistence=persistence, keystore=key_store))
        await registry.run_step("request_log.backfill_total_tokens", _context(startup))
        startup.maintenance = registry

        assert startup.maintenance_summary()["state"] == "partial"

    async def test_the_summary_carries_no_failure_text(self):
        """``/health/ready`` is unguarded; the text belongs on the admin surface."""
        persistence, key_store = _stores()
        persistence.backfill_total_tokens = AsyncMock(
            side_effect=RuntimeError("no such table: /var/data/secret/gateway.db")
        )
        startup = DeferredStartup(
            _config_with_model(), _App(persistence=persistence, keystore=key_store)
        )
        await startup._run_backfills()

        summary = startup.maintenance_summary()
        assert summary["failed"] == ["request_log.backfill_total_tokens"]
        assert "secret" not in json.dumps(summary), (
            "an exception string can carry a path; the probe must not"
        )
        assert all("error" not in step for step in summary["steps"])

    def test_while_the_task_is_still_running_it_says_running(self):
        persistence, key_store = _stores()
        startup = DeferredStartup(
            _config_with_model(), _App(persistence=persistence, keystore=key_store)
        )
        asyncio.run(startup._run_backfills())
        startup._pending_tasks.add("backfills")
        assert startup.maintenance_summary()["state"] == "running"

    async def test_the_status_payload_carries_the_summary(self):
        persistence, key_store = _stores()
        startup = DeferredStartup(
            _config_with_model(), _App(persistence=persistence, keystore=key_store)
        )
        await startup._run_backfills()
        assert startup.status()["maintenance"]["state"] == "done"


@pytest.mark.parametrize("step", _ALL_STEPS)
def test_every_step_says_what_it_is_for(step):
    """The admin surface and the log line both show the description."""
    persistence, key_store = _stores()
    registry = build_registry(_App(persistence=persistence, keystore=key_store))
    described = {s.name: s.description for s in registry.steps}
    assert described[step], f"{step} has no description for an operator to read"


class _Request:
    def __init__(self, app: Any) -> None:
        self.app = app


class TestTheReadinessProbe:
    """The probe answers "is traffic safe", not "did every repair work"."""

    async def test_a_failed_repair_is_reported_without_withholding_traffic(self):
        import json

        from llm_rosetta.gateway.app import handle_health_ready
        from llm_rosetta.observability import MetricsCollector

        persistence, key_store = _stores()
        persistence.backfill_total_tokens = AsyncMock(
            side_effect=RuntimeError("database is locked")
        )
        app = _App(persistence=persistence, keystore=key_store)
        app.metrics = MetricsCollector()
        startup = DeferredStartup(_config_with_model(), app)
        app.deferred_startup = startup
        await startup._run_backfills()

        response = await handle_health_ready(_Request(app))

        assert response.status_code == 200, (
            "a deferred repair can be retried; an outage cannot be undone"
        )
        body = json.loads(response.body)
        assert body["status"] == "ready"
        assert body["maintenance"]["state"] == "partial"
        assert body["maintenance"]["failed"] == ["request_log.backfill_total_tokens"]

    async def test_a_clean_run_says_done(self):
        import json

        from llm_rosetta.gateway.app import handle_health_ready
        from llm_rosetta.observability import MetricsCollector

        persistence, key_store = _stores()
        app = _App(persistence=persistence, keystore=key_store)
        app.metrics = MetricsCollector()
        startup = DeferredStartup(_config_with_model(), app)
        app.deferred_startup = startup
        await startup._run_backfills()

        body = json.loads((await handle_health_ready(_Request(app))).body)

        assert body["maintenance"]["state"] == "done"
        assert body["maintenance"]["failed"] == []

    async def test_the_repairs_still_running_are_a_503(self):
        """Which is the pre-existing behaviour: the *task* is not finished."""
        import json

        from llm_rosetta.gateway.app import handle_health_ready

        persistence, key_store = _stores()
        app = _App(persistence=persistence, keystore=key_store)
        startup = DeferredStartup(_config_with_model(), app)
        app.deferred_startup = startup
        startup._pending_tasks.add("backfills")

        response = await handle_health_ready(_Request(app))

        assert response.status_code == 503
        body = json.loads(response.body)
        assert body["status"] == "not_ready"
        assert body["maintenance"]["state"] == "not_started"


class TestTheAlertableLine:
    """``partial`` in a probe body is invisible to a monitor; a log line is not."""

    async def test_a_failed_run_is_logged_at_error_naming_the_failed_steps(
        self, caplog
    ):
        import logging

        persistence, key_store = _stores()
        persistence.backfill_total_tokens = AsyncMock(
            side_effect=RuntimeError("database is locked")
        )
        startup = DeferredStartup(
            _config_with_model(), _App(persistence=persistence, keystore=key_store)
        )
        with caplog.at_level(logging.INFO):
            await startup._run_backfills()

        errors = [
            record for record in caplog.records if record.levelno >= logging.ERROR
        ]
        assert errors, "a run that did not repair everything has to be alertable"
        assert "request_log.backfill_total_tokens" in caplog.text

    async def test_a_clean_run_does_not_log_an_error(self, caplog):
        import logging

        persistence, key_store = _stores()
        startup = DeferredStartup(
            _config_with_model(), _App(persistence=persistence, keystore=key_store)
        )
        with caplog.at_level(logging.INFO):
            await startup._run_backfills()

        assert not [
            record for record in caplog.records if record.levelno >= logging.ERROR
        ], "an all-green run must not page anyone"
        assert "Deferred maintenance complete" in caplog.text


def _context(startup: DeferredStartup) -> StepContext:
    """The context the startup task hands its steps."""
    return StepContext(
        config=startup._config,
        persistence=startup._app.persistence,
        keystore=startup._app.keystore,
        app=startup._app,
    )
