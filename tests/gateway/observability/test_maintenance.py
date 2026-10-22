"""Tests for the maintenance vocabulary, and for the migrations in it.

Two things are asserted here.  The framework's own contract — what a policy
means for a failure, and that a step's result is kept — and the property every
registered step must have: **run the registry twice and the second pass changes
nothing**.  The property is asserted by iterating the registry rather than by
naming its steps, so a step added later is covered by being registered instead
of by someone remembering to test it.
"""

from __future__ import annotations

import asyncio
import logging
import sqlite3
from pathlib import Path

import pytest

from llm_rosetta_gateway.observability import PersistenceManager
from llm_rosetta_gateway.observability.maintenance import (
    Policy,
    Registry,
    Step,
    StepContext,
)

_MIGRATIONS = PersistenceManager._MIGRATIONS
_CURRENT = max(version for version, _description, _sql in _MIGRATIONS)


def _step(name: str, run, store: str = "test.db") -> Step:
    return Step(name=name, description=f"a step called {name}", store=store, run=run)


async def _connection(store: Path):
    """A store opened the way the app opens one, and its connection."""
    manager = await PersistenceManager.create(str(store))
    return manager


class TestThePolicies:
    """What a failure means is the policy, not a flag on the caller."""

    async def test_a_mandatory_step_that_fails_raises_its_own_error(self, tmp_path):
        """A caller that aborts startup keeps aborting, with its own error."""
        boom = RuntimeError("the store is not what this code expects")

        async def _fail(context):
            raise boom

        registry = Registry(
            name="test", steps=[_step("fails", _fail)], policy=Policy.MANDATORY
        )
        with pytest.raises(RuntimeError) as caught:
            await registry.run(StepContext())
        assert caught.value is boom, "the caller must see the error it saw before"

    async def test_a_deferred_step_that_fails_does_not_stop_the_next(self, tmp_path):
        """Which is the whole point of the deferred set (epic #1027)."""
        ran: list[str] = []

        async def _fail(context):
            raise RuntimeError("backfill one is unhappy")

        async def _after(context):
            ran.append("after")
            return 3

        registry = Registry(
            name="test",
            steps=[_step("first", _fail), _step("second", _after)],
            policy=Policy.DEFERRED,
        )
        results = await registry.run(StepContext())

        assert ran == ["after"], "a failed step must not skip the ones behind it"
        assert [result.ok for result in results] == [False, True]
        assert results[0].changed == 0
        assert results[1].changed == 3

    async def test_a_failure_is_reported_in_the_result_and_not_only_logged(
        self, caplog
    ):
        async def _fail(context):
            raise ValueError("no such column")

        registry = Registry(
            name="test", steps=[_step("fails", _fail)], policy=Policy.DEFERRED
        )
        with caplog.at_level(logging.WARNING):
            results = await registry.run(StepContext())

        assert results[0].ok is False
        assert results[0].error == "ValueError: no such column"
        assert "maintenance step fails (test) failed" in caplog.text

    async def test_a_mandatory_failure_names_the_step_before_it_aborts(self, caplog):
        """Which step killed the boot is what an operator needs first.

        The step's own exception is re-raised unchanged, so on its own it says
        what went wrong but not where — the log line has to carry the name.
        """

        async def _fail(context):
            raise RuntimeError("the store is not what this code expects")

        registry = Registry(
            name="request_log",
            steps=[_step("request_log.v7", _fail)],
            policy=Policy.MANDATORY,
        )
        with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError):
            await registry.run(StepContext())

        assert "maintenance step request_log.v7 (request_log) failed" in caplog.text

    async def test_the_results_of_the_last_run_are_kept(self):
        """So "did the backfills run" is not a log search."""

        async def _nothing(context):
            return 0

        registry = Registry(
            name="test", steps=[_step("quiet", _nothing)], policy=Policy.DEFERRED
        )
        assert registry.results == []
        await registry.run(StepContext())
        assert [result.name for result in registry.results] == ["quiet"]

    async def test_a_step_is_told_how_long_it_took(self):
        async def _quick(context):
            return 1

        registry = Registry(
            name="test", steps=[_step("quick", _quick)], policy=Policy.MANDATORY
        )
        results = await registry.run(StepContext())
        assert results[0].seconds >= 0.0
        assert results[0].ok is True

    async def test_a_result_says_which_store_the_step_touched(self):
        """A partial run of a registry over two stores has to name the store.

        The name carries it too, but a reader under pressure should not have to
        parse it back out — and a step's name is allowed to be anything.
        """

        async def _nothing(context):
            return 0

        async def _fail(context):
            raise RuntimeError("database is locked")

        registry = Registry(
            name="test",
            steps=[
                _step("one", _nothing, store="first.db"),
                _step("two", _fail, store="second.db"),
            ],
            policy=Policy.DEFERRED,
        )
        results = await registry.run(StepContext())

        assert [result.store for result in results] == ["first.db", "second.db"]
        assert results[1].store == "second.db", (
            "a failed result names the store it failed on"
        )

    async def test_a_re_run_keeps_the_store_on_the_result(self):
        async def _nothing(context):
            return 0

        registry = Registry(
            name="test",
            steps=[_step("one", _nothing, store="only.db")],
            policy=Policy.DEFERRED,
        )
        result = await registry.run_step("one", StepContext())
        assert result.store == "only.db"


class TestTheSynchronousRun:
    """A load-time registry runs without a loop, and refuses what it cannot run."""

    def test_a_plain_function_step_runs(self):
        ran: list[str] = []

        def _plain(context):
            ran.append("plain")
            return 2

        registry = Registry(
            name="config", steps=[_step("plain", _plain)], policy=Policy.LOAD_TIME
        )
        results = registry.run_sync(StepContext(config={"a": 1}))

        assert ran == ["plain"]
        assert results[0].changed == 2
        assert results[0].ok is True
        assert registry.results == results, "the run is kept, as in the async one"

    def test_an_asynchronous_step_is_refused_rather_than_skipped(self):
        """Silently dropping it would report a document as repaired by nothing."""

        async def _async(context):
            return 1  # pragma: no cover - never awaited

        registry = Registry(
            name="config", steps=[_step("async", _async)], policy=Policy.LOAD_TIME
        )
        with pytest.raises(TypeError) as caught:
            registry.run_sync(StepContext())
        assert "asynchronous" in str(caught.value)
        assert "async" in str(caught.value), "the refusal names the step"

    def test_a_load_time_failure_fails_the_load(self):
        """LOAD_TIME means the load fails — it is not a 'record and continue'."""

        def _boom(context):
            raise ValueError("the config is not what this code expects")

        registry = Registry(
            name="config", steps=[_step("boom", _boom)], policy=Policy.LOAD_TIME
        )
        with pytest.raises(ValueError):
            registry.run_sync(StepContext())
        assert registry.results[0].ok is False

    def test_the_async_runner_takes_a_plain_step_too(self):
        """So a registry can hold both, and the runner is the one that decides."""

        def _plain(context):
            return 3

        async def _async(context):
            return 4

        registry = Registry(
            name="mixed",
            steps=[_step("plain", _plain), _step("async", _async)],
            policy=Policy.DEFERRED,
        )
        results = asyncio.run(registry.run(StepContext()))

        assert [result.changed for result in results] == [3, 4]

    def test_a_sync_run_still_obeys_the_deferred_policy(self):
        def _boom(context):
            raise RuntimeError("database is locked")

        def _after(context):
            return 5

        registry = Registry(
            name="mixed",
            steps=[_step("boom", _boom), _step("after", _after)],
            policy=Policy.DEFERRED,
        )
        results = registry.run_sync(StepContext())

        assert [result.ok for result in results] == [False, True]
        assert results[1].changed == 5

    def test_a_sync_run_still_obeys_the_mandatory_policy(self):
        """The store's own error travels, and the step after it never runs.

        The keystore is the synchronous caller: one of its migrations fails, the
        store is not opened, and the failure is on the registry for the surface
        that reports it.
        """
        ran: list[str] = []

        def _boom(context):
            raise sqlite3.OperationalError("no such column: usage_bucket")

        def _after(context):  # pragma: no cover - the abort above is the point
            ran.append("after")
            return 1

        registry = Registry(
            name="api_keys",
            steps=[_step("boom", _boom), _step("after", _after)],
            policy=Policy.MANDATORY,
        )
        with pytest.raises(sqlite3.OperationalError) as caught:
            registry.run_sync(StepContext())

        assert "usage_bucket" in str(caught.value), (
            "the caller that aborts gets the store's error, not a wrapped one"
        )
        assert ran == []
        assert [result.ok for result in registry.results] == [False], (
            "the failure is recorded before it is raised"
        )


class TestRunningOneStep:
    """An operator re-runs a step by name; the vocabulary has to allow it."""

    async def test_only_the_named_step_runs(self):
        ran: list[str] = []

        async def _one(context):
            ran.append("one")
            return 1

        async def _two(context):
            ran.append("two")
            return 2

        registry = Registry(
            name="test",
            steps=[_step("one", _one), _step("two", _two)],
            policy=Policy.DEFERRED,
        )
        result = await registry.run_step("two", StepContext())

        assert ran == ["two"], "a re-run is one step, not a registry run"
        assert result.name == "two"
        assert result.changed == 2

    async def test_an_unknown_name_is_a_key_error(self):
        async def _nothing(context):
            return 0

        registry = Registry(
            name="test", steps=[_step("known", _nothing)], policy=Policy.DEFERRED
        )
        with pytest.raises(KeyError):
            await registry.run_step("unknown", StepContext())

    async def test_the_result_replaces_that_step_and_leaves_the_others(self):
        """The surface that asked still reads one answer per step, in order."""

        async def _nothing(context):
            return 0

        async def _later(context):
            return 5

        registry = Registry(
            name="test",
            steps=[_step("first", _nothing), _step("second", _nothing)],
            policy=Policy.DEFERRED,
        )
        await registry.run(StepContext())
        re_run = _step("second", _later)
        rerun_registry = Registry(
            name="test",
            steps=[registry.steps[0], re_run],
            policy=Policy.DEFERRED,
            results=list(registry.results),
        )
        await rerun_registry.run_step("second", StepContext())

        assert [result.name for result in rerun_registry.results] == [
            "first",
            "second",
        ]
        assert [result.changed for result in rerun_registry.results] == [0, 5]

    async def test_a_step_that_was_never_run_is_added_in_step_order(self):
        async def _nothing(context):
            return 0

        registry = Registry(
            name="test",
            steps=[_step("first", _nothing), _step("second", _nothing)],
            policy=Policy.DEFERRED,
        )
        await registry.run_step("second", StepContext())

        assert [result.name for result in registry.results] == ["second"]

    async def test_a_raising_policy_still_raises_after_recording(self):
        """A re-run is not a way to make a mandatory step quiet."""

        async def _fail(context):
            raise RuntimeError("the store is not what this code expects")

        registry = Registry(
            name="test", steps=[_step("fails", _fail)], policy=Policy.MANDATORY
        )
        with pytest.raises(RuntimeError):
            await registry.run_step("fails", StepContext())
        assert registry.results[0].ok is False
        assert registry.results[0].error == (
            "RuntimeError: the store is not what this code expects"
        )


class TestTheIdempotenceProperty:
    """One property, asserted for every step by iterating the registry."""

    async def test_a_fresh_store_lands_on_the_current_schema(self, tmp_path):
        store = tmp_path / "fresh"
        manager = await _connection(store)
        try:
            assert await manager.schema_version() == _CURRENT
            assert manager.maintenance is not None, (
                "create() must run the registry and keep it"
            )
            assert [result.name for result in manager.maintenance.results], (
                "the steps that ran must be readable afterwards"
            )
        finally:
            await manager.close()

    @pytest.mark.parametrize("runs", [2, 3])
    async def test_the_second_pass_of_every_registered_step_changes_nothing(
        self, tmp_path, runs
    ):
        """The property the epic asks for, over the registry rather than a list.

        Every step is asserted by being registered: the test runs the store's
        registry *again* against the schema the first run produced, and a step
        that is not idempotent shows up as a non-zero ``changed``.
        """
        manager = await _connection(tmp_path / "twice")
        try:
            registry = manager.migration_registry()
            assert registry.steps, "the registry must have steps to test"

            for attempt in range(runs):
                results = await registry.run(StepContext(persistence=manager))
                assert [result.name for result in results] == [
                    step.name for step in registry.steps
                ], "every registered step has to report, or the property skips it"
                if attempt:
                    assert [result.changed for result in results] == [0] * len(
                        results
                    ), "a repeated run must change nothing"
            assert await manager.schema_version() == _CURRENT
        finally:
            await manager.close()

    async def test_the_property_has_teeth(self, tmp_path):
        """The assertion above fails for a step that is not idempotent."""
        inserts = 0

        async def _not_idempotent(context):
            nonlocal inserts
            inserts += 1
            return 1

        registry = Registry(
            name="test",
            steps=[_step("appends-every-time", _not_idempotent)],
            policy=Policy.MANDATORY,
        )
        await registry.run(StepContext())
        second = await registry.run(StepContext())
        assert second[0].changed == 1, (
            "a step that always changes something must not pass the property"
        )


class TestTheMigrationsAreUnchanged:
    """Slice 1 is behaviour-preserving, so the behaviour is pinned."""

    async def test_a_fresh_store_records_every_version(self, tmp_path):
        manager = await _connection(tmp_path / "fresh")
        try:
            rows = await manager._conn.execute_fetchall(
                "SELECT version FROM schema_version ORDER BY version"
            )
            assert [row[0] for row in rows] == sorted(
                version for version, _d, _s in _MIGRATIONS
            )
        finally:
            await manager.close()

    async def test_the_log_lines_are_the_ones_the_migrations_always_wrote(
        self, tmp_path, caplog
    ):
        """``create()`` on a fresh store says the same two lines per migration."""
        with caplog.at_level(logging.INFO):
            manager = await _connection(tmp_path / "logged")
        try:
            for version, description, _statements in _MIGRATIONS:
                assert f"Applying migration v{version}: {description}" in caplog.text
                assert f"Migration v{version} applied" in caplog.text
        finally:
            await manager.close()

    @pytest.mark.parametrize("stopped_at", [0, 1, 2, 3, 4, 5])
    async def test_a_store_from_any_earlier_version_is_brought_forward(
        self, tmp_path, stopped_at
    ):
        """Every version a store can be at, not just the one before this one.

        The columns a later migration adds are present already — that is what a
        store left by an older release looks like when the columns were added
        without the version being recorded — so this also exercises the
        duplicate-column tolerance for each of them.
        """
        store = tmp_path / f"at-{stopped_at}"
        manager = await _connection(store)
        await manager.close()

        conn = sqlite3.connect(store / "gateway.db")
        try:
            conn.execute("DELETE FROM schema_version WHERE version > ?", (stopped_at,))
            conn.commit()
        finally:
            conn.close()

        again = await _connection(store)
        try:
            assert await again.schema_version() == _CURRENT
            versions = {
                row[0]
                for row in await again._conn.execute_fetchall(
                    "SELECT version FROM schema_version"
                )
            }
            assert versions == {version for version, _d, _s in _MIGRATIONS}
        finally:
            await again.close()

    async def test_the_forward_migration_leaves_the_store_shaped_right(self, tmp_path):
        """The shape, not only the ledger: the column and its default."""
        store = tmp_path / "shape"
        manager = await _connection(store)
        await manager.close()

        conn = sqlite3.connect(store / "gateway.db")
        try:
            conn.execute("DELETE FROM schema_version WHERE version = ?", (6,))
            conn.commit()
        finally:
            conn.close()

        again = await _connection(store)
        try:
            described = {
                row[1]: row[4]
                for row in await (
                    await again._conn.execute("PRAGMA table_info(request_log)")
                ).fetchall()
            }
            assert described["model_type"] == "'llm'", (
                "the default is what keeps rows that predate the column meaning chat"
            )
        finally:
            await again.close()

    async def test_a_pre_registry_store_is_adopted_before_migrating(self, tmp_path):
        """Columns without a version row: v1 is inferred, the rest is applied."""
        store = tmp_path / "legacy"
        manager = await _connection(store)
        await manager.close()

        conn = sqlite3.connect(store / "gateway.db")
        try:
            # What the old unversioned _migrate_add_columns left behind: v1's
            # columns, and no record of them.
            conn.execute("DELETE FROM schema_version")
            conn.commit()
        finally:
            conn.close()

        adopted = await _connection(store)
        try:
            assert await adopted.schema_version() == _CURRENT
            runs = {
                result.name: result.changed for result in adopted.maintenance.results
            }
            assert runs["request_log.adopt_pre_registry_schema"] == 1, (
                "the open has to infer the columns it found"
            )
            # And the migrations that followed were applied, not skipped: the
            # adoption records v1, so v2..v6 still have to run.
            assert runs[f"request_log.v{_CURRENT}"] > 0
        finally:
            await adopted.close()

    async def test_an_unexpected_failure_still_aborts(self, tmp_path):
        """Policy.MANDATORY, on the real store: the error is not swallowed."""
        manager = await _connection(tmp_path / "aborts")
        try:
            registry = manager.migration_registry()

            async def _explode(context):
                raise RuntimeError("a step that cannot run")

            broken = Registry(
                name=registry.name,
                steps=[*registry.steps, _step("explodes", _explode)],
                policy=Policy.MANDATORY,
            )
            with pytest.raises(RuntimeError):
                await broken.run(StepContext(persistence=manager))
            assert broken.results[-1].ok is False
        finally:
            await manager.close()

    async def test_the_store_whose_migration_fails_does_not_open(self, tmp_path):
        """The wiring, not a runner the test built: this is the abort contract.

        ``test_an_unexpected_failure_still_aborts`` constructs its own registry
        with ``policy=Policy.MANDATORY``, so it pins the runner and not what
        :meth:`PersistenceManager.create` is wired to — a change that made the
        store's registry deferred would leave that test green.  This opens a
        store with a migration that cannot run and asserts both halves of the
        contract: the open raises the store's own error, and the store keeps the
        version it had rather than claiming one it never applied.
        """
        store = tmp_path / "half-migrated"
        manager = await _connection(store)
        await manager.close()
        assert PersistenceManager.migration_registry().policy is Policy.MANDATORY

        saved = list(PersistenceManager._MIGRATIONS)
        PersistenceManager._MIGRATIONS = [
            *saved,
            (
                _CURRENT + 1,
                "a migration that cannot run",
                ["INSERT INTO no_such_table (x) VALUES (1)"],
            ),
        ]
        try:
            with pytest.raises(sqlite3.OperationalError) as caught:
                await PersistenceManager.create(str(store))
            assert "no_such_table" in str(caught.value), (
                "the caller must see the store's error, not a wrapper"
            )
        finally:
            PersistenceManager._MIGRATIONS = saved

        reopened = await _connection(store)
        try:
            versions = {
                row[0]
                for row in await reopened._conn.execute_fetchall(
                    "SELECT version FROM schema_version"
                )
            }
            assert versions == {version for version, _d, _s in saved}, (
                "the version that failed must not be recorded"
            )
            assert _CURRENT + 1 not in versions
        finally:
            await reopened.close()
