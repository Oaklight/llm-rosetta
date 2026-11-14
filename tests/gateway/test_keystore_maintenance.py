"""The keystore's own registry: the ledger, the adoption, and the backfill.

Slice 3 of #1027.  Until now ``keys.db`` was upgraded by two ``PRAGMA``-guarded
helpers that recorded nothing and logged nothing — the epic's claim that no test
pinned their idempotence was true.  What is asserted here is what replaced them:
a version ledger in the store, an adoption step for a store written before the
ledger existed, and a backfill that reports what it repaired.

The property every registry has to satisfy is asserted for this store together
with the others in ``tests/gateway/test_maintenance_registries.py``.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any

import pytest

from llm_rosetta_gateway.keystore import VERSIONS, KeyStore, key_bucket
from llm_rosetta_gateway.maintenance.keystore import COLUMN_OF, build_registry
from llm_rosetta_gateway.observability.maintenance import Policy

_STEPS = [
    "api_keys.adopt_pre_registry_schema",
    "api_keys.v1",
    "api_keys.v2",
    "api_keys.v3",
    "api_keys.v4",
    "api_keys.v5",
    "api_keys.backfill_usage_bucket",
]

#: The table as it was before either migration: what an old ``keys.db`` has.
_BASE_COLUMNS = (
    "id TEXT PRIMARY KEY, key_hash TEXT NOT NULL UNIQUE, "
    "label TEXT NOT NULL DEFAULT '', allowed_shims TEXT NOT NULL DEFAULT '[\"*\"]', "
    "created TEXT NOT NULL, rotated TEXT"
)


def _legacy_db(path: Path, *, extra_columns: str = "", rows: int = 0) -> None:
    """A keys.db as an older release left it: no ledger, and its own columns."""
    columns = _BASE_COLUMNS + (f", {extra_columns}" if extra_columns else "")
    conn = sqlite3.connect(path)
    try:
        conn.execute(f"CREATE TABLE api_keys ({columns})")
        for index in range(rows):
            conn.execute(
                "INSERT INTO api_keys (id, key_hash, label, created) "
                "VALUES (?, ?, ?, ?)",
                (f"key{index}", f"hash{index}", f"label{index}", "2026-01-01"),
            )
        conn.commit()
    finally:
        conn.close()


async def _columns(store: KeyStore) -> set[str]:
    cursor = await store._conn.execute("PRAGMA table_info(api_keys)")
    return {row[1] for row in await cursor.fetchall()}


async def _tables(store: KeyStore) -> set[str]:
    rows = await store._conn.execute_fetchall(
        "SELECT name FROM sqlite_master WHERE type = 'table'"
    )
    return {row[0] for row in rows}


async def _versions(store: KeyStore) -> set[int]:
    rows = await store._conn.execute_fetchall("SELECT version FROM schema_version")
    return {row[0] for row in rows}


class TestTheRegistry:
    def test_it_is_mandatory_and_names_this_store(self):
        registry = build_registry()
        assert registry.policy is Policy.MANDATORY, (
            "a store whose schema is not what this code expects must not be served from"
        )
        assert [step.name for step in registry.steps] == _STEPS
        assert {step.store for step in registry.steps} == {"keys.db"}

    def test_every_version_has_adoption_evidence(self):
        """The adoption step infers from columns; the two lists cannot drift."""
        assert sorted(version for version, _d, _s in VERSIONS) == sorted(COLUMN_OF)


class TestAFreshStore:
    async def test_it_lands_on_the_current_schema_and_records_it(self, tmp_path):
        store = await KeyStore.create(tmp_path / "keys.db")
        try:
            assert await _columns(store) >= {"last_used", "usage_bucket"}
            assert await _versions(store) == {1, 2, 3, 4, 5}
            assert store.maintenance is not None, (
                "create() must keep what it ran, so an operator can read it"
            )
            assert [result.name for result in store.maintenance.results] == _STEPS
        finally:
            await store.close()

    async def test_the_dropped_column_is_not_in_a_fresh_store(self, tmp_path):
        store = await KeyStore.create(tmp_path / "keys.db")
        try:
            assert "allowed_shims" not in await _columns(store)
        finally:
            await store.close()

    async def test_a_store_with_an_index_on_the_column_is_brought_forward(
        self, tmp_path
    ):
        """The rebuild does not care that the column is indexed; the ALTER
        would refuse, and this step is mandatory."""
        db = tmp_path / "keys.db"
        _legacy_db(db, rows=1)
        conn = sqlite3.connect(db)
        try:
            conn.execute("CREATE INDEX idx_shims ON api_keys(allowed_shims)")
            conn.commit()
        finally:
            conn.close()

        store = await KeyStore.create(db)
        try:
            assert "allowed_shims" not in await _columns(store)
            assert await store._conn.execute_fetchall("SELECT id FROM api_keys") == [
                ("key0",)
            ]
            assert await _versions(store) == {1, 2, 3, 4, 5}
        finally:
            await store.close()

    async def test_the_migrations_are_what_added_the_columns(self, tmp_path):
        """Not the CREATE TABLE: the same route serves a fresh store and an old."""
        store = await KeyStore.create(tmp_path / "keys.db")
        try:
            assert store.maintenance is not None
            changed = {r.name: r.changed for r in store.maintenance.results}
            assert changed["api_keys.v1"] > 0, "v1 adds last_used"
            assert changed["api_keys.v2"] > 0, "v2 adds usage_bucket"
            assert changed["api_keys.adopt_pre_registry_schema"] == 1, (
                "a fresh store has no dropped column, which is v5's evidence"
            )
            assert changed["api_keys.v5"] == 0, (
                "so the adoption already recorded it and v5 does not run"
            )
        finally:
            await store.close()


class TestARebuildThatHasNothingToDo:
    async def test_an_absent_dropped_column_only_records_the_version(self, tmp_path):
        """Defence in depth: the adoption step normally records v5 first, so a
        store this code created never runs the rebuild.  If the ledger row is
        missing anyway, the version must record itself without touching rows.
        """
        from llm_rosetta_gateway.maintenance.keystore import _apply_migration
        from llm_rosetta_gateway.observability.maintenance import StepContext

        store = await KeyStore.create(tmp_path / "keys.db")
        try:
            await store._conn.execute("DELETE FROM schema_version WHERE version = 5")
            await store._conn.commit()
            changed = await _apply_migration(
                StepContext(keystore=store),
                version=5,
                description="drop api_keys.allowed_shims",
                statements=["INSERT INTO no_such_table (x) VALUES (1)"],
                drops="allowed_shims",
            )
            assert changed == 1, "one ledger row and no statement"
            assert 5 in await _versions(store)
        finally:
            await store.close()


class TestAnInterruptedRebuild:
    #: What the rebuild copies: the v4 columns, without the one being dropped.
    _COLS = (
        "id, key_hash, label, created, rotated, last_used, usage_bucket, "
        "allowed_models, rate_limit"
    )

    async def test_a_rebuild_that_dies_midway_leaves_the_file_as_it_was(self, tmp_path):
        """The v5 statements are one transaction: a process death between the
        drop and the rename comes back as the v4 table, never as an empty one
        with the keys stranded in a table nothing reads."""
        db = tmp_path / "keys.db"
        store = await KeyStore.create(db)
        try:
            await store.create_key(label="one", manual_key="key-one")
            await store.create_key(label="two", manual_key="key-two")
            # Put the file back to a v4 shape: the column, no v5 in the ledger.
            await store._conn.execute(
                "ALTER TABLE api_keys ADD COLUMN allowed_shims TEXT NOT NULL "
                "DEFAULT '[\"*\"]'"
            )
            await store._conn.execute("DELETE FROM schema_version WHERE version = 5")
            await store._conn.commit()
        finally:
            await store.close()

        # The rebuild's statements, up to the drop, then a close without a
        # commit — what a death inside the transaction leaves behind.
        conn = sqlite3.connect(db)
        try:
            conn.execute("BEGIN IMMEDIATE")
            conn.execute(
                "CREATE TABLE api_keys_v5 (id TEXT PRIMARY KEY, "
                "key_hash TEXT NOT NULL UNIQUE, label TEXT NOT NULL DEFAULT '', "
                "created TEXT NOT NULL, rotated TEXT, last_used TEXT, "
                "usage_bucket TEXT, allowed_models TEXT NOT NULL DEFAULT '[\"*\"]', "
                "rate_limit TEXT NOT NULL DEFAULT '')"
            )
            conn.execute(
                f"INSERT INTO api_keys_v5 ({self._COLS}) SELECT {self._COLS} FROM api_keys"
            )
            conn.execute("DROP TABLE api_keys")
        finally:
            conn.close()

        reopened = await KeyStore.create(db)
        try:
            assert "api_keys_v5" not in await _tables(reopened)
            assert "allowed_shims" not in await _columns(reopened)
            rows = await reopened._conn.execute_fetchall(
                "SELECT label FROM api_keys ORDER BY label"
            )
            assert [row[0] for row in rows] == ["one", "two"], (
                "the keys are still there"
            )
            assert reopened.validate("key-one") is not None
        finally:
            await reopened.close()

    async def test_a_rebuild_that_fails_rolls_back_whole(self, tmp_path):
        """The transaction is the guard: without it, a statement failing after
        the drop leaves a live table with no rows."""
        from llm_rosetta_gateway.keystore import VERSIONS
        from llm_rosetta_gateway.maintenance.keystore import _apply_migration
        from llm_rosetta_gateway.observability.maintenance import StepContext

        db = tmp_path / "keys.db"
        store = await KeyStore.create(db)
        try:
            await store.create_key(label="one", manual_key="key-one")
            await store._conn.execute(
                "ALTER TABLE api_keys ADD COLUMN allowed_shims TEXT NOT NULL "
                "DEFAULT '[\"*\"]'"
            )
            await store._conn.execute("DELETE FROM schema_version WHERE version = 5")
            await store._conn.commit()

            rebuild = next(s for v, _d, s in VERSIONS if v == 5)
            with pytest.raises(sqlite3.OperationalError):
                await _apply_migration(
                    StepContext(keystore=store),
                    version=5,
                    description="drop api_keys.allowed_shims",
                    statements=[*rebuild, "INSERT INTO no_such_table (x) VALUES (1)"],
                    drops="allowed_shims",
                )

            assert "api_keys_v5" not in await _tables(store), "nothing half-built"
            assert "allowed_shims" in await _columns(store), "the v4 table is back"
            rows = await store._conn.execute_fetchall("SELECT label FROM api_keys")
            assert [row[0] for row in rows] == ["one"]
            assert 5 not in await _versions(store), "and the step will run again"
        finally:
            await store.close()


class TestAStoreFromAnOlderRelease:
    async def test_a_store_missing_both_columns_is_brought_forward(self, tmp_path):
        db = tmp_path / "keys.db"
        _legacy_db(db, rows=2)

        store = await KeyStore.create(db)
        try:
            assert await _columns(store) >= {"last_used", "usage_bucket"}
            assert await _versions(store) == {1, 2, 3, 4, 5}
            rows = await store._conn.execute_fetchall(
                "SELECT key_hash, usage_bucket FROM api_keys ORDER BY id"
            )
            assert [bucket for _hash, bucket in rows] == [
                key_bucket("hash0"),
                key_bucket("hash1"),
            ], "the buckets are derived from the hashes, not left NULL"
        finally:
            await store.close()

    async def test_a_store_whose_columns_predate_the_ledger_is_adopted(self, tmp_path):
        """``usage_bucket`` present, no ledger: v2 is recorded, not re-applied."""
        db = tmp_path / "keys.db"
        _legacy_db(db, extra_columns="usage_bucket TEXT", rows=1)

        store = await KeyStore.create(db)
        try:
            assert store.maintenance is not None
            changed = {r.name: r.changed for r in store.maintenance.results}
            assert changed["api_keys.adopt_pre_registry_schema"] == 1, (
                "the presence of the column is the evidence for v2"
            )
            assert changed["api_keys.v2"] == 0, "and v2 does not run again"
            assert changed["api_keys.v1"] > 0, "while v1 still has work to do"
            assert await _versions(store) == {1, 2, 3, 4, 5}
        finally:
            await store.close()

    async def test_a_store_that_already_has_both_columns_is_adopted_wholesale(
        self, tmp_path
    ):
        """No ledger, and both versions' columns: both are recorded in one step.

        A ``keys.db`` the older release had already brought forward.  It is the
        other half of why the ledger is read per version and not as a ladder:
        here the adoption records 1 *and* 2, where a store carrying only
        ``usage_bucket`` records 2 while v1 still has work to do.

        ``allowed_models`` and ``rate_limit`` are not among the columns this
        release brought forward, so v3 and v4 each still alter the table and
        record themselves: 2 apiece, one statement plus the ledger row.
        """
        db = tmp_path / "keys.db"
        _legacy_db(db, extra_columns="last_used TEXT, usage_bucket TEXT", rows=1)
        conn = sqlite3.connect(db)
        try:
            conn.execute("UPDATE api_keys SET usage_bucket = ?", (key_bucket("hash0"),))
            conn.commit()
        finally:
            conn.close()

        store = await KeyStore.create(db)
        try:
            assert store.maintenance is not None
            # v5 drops the column the server never read; the row survives it.
            assert "allowed_shims" not in await _columns(store)
            assert await store._conn.execute_fetchall("SELECT id FROM api_keys") == [
                ("key0",)
            ]
            changed = {r.name: r.changed for r in store.maintenance.results}
            assert changed == {
                "api_keys.adopt_pre_registry_schema": 2,
                "api_keys.v1": 0,
                "api_keys.v2": 0,
                "api_keys.v3": 2,
                "api_keys.v4": 2,
                "api_keys.v5": 6,
                "api_keys.backfill_usage_bucket": 0,
            }
            assert await _versions(store) == {1, 2, 3, 4, 5}
        finally:
            await store.close()

    async def test_the_backfill_repairs_rows_written_before_the_column(self, tmp_path):
        db = tmp_path / "keys.db"
        _legacy_db(db, extra_columns="usage_bucket TEXT", rows=1)

        store = await KeyStore.create(db)
        try:
            assert store.maintenance is not None
            changed = {r.name: r.changed for r in store.maintenance.results}
            assert changed["api_keys.backfill_usage_bucket"] == 1
            row = await store._conn.execute_fetchone(
                "SELECT usage_bucket FROM api_keys"
            )
            assert row is not None
            assert row[0] == key_bucket("hash0")
        finally:
            await store.close()

    async def test_a_repaired_store_reports_nothing_on_the_next_open(self, tmp_path):
        """The backfill's own ``WHERE`` clause is the whole of its ledger."""
        db = tmp_path / "keys.db"
        _legacy_db(db, rows=1)
        first = await KeyStore.create(db)
        await first.close()

        again = await KeyStore.create(db)
        try:
            assert again.maintenance is not None
            assert [r.changed for r in again.maintenance.results] == [0] * len(
                _STEPS
            ), "every step of a store that is already current has nothing to do"
            assert [r.ok for r in again.maintenance.results] == [True] * len(_STEPS)
        finally:
            await again.close()


class TestAFailingMigration:
    async def test_it_aborts_the_open_and_leaves_the_ledger_alone(
        self, tmp_path, monkeypatch
    ):
        """MANDATORY here, as the whole open always was: it raises out of create()."""
        db = tmp_path / "keys.db"
        store = await KeyStore.create(db)
        await store.close()

        from llm_rosetta_gateway import keystore as keystore_module

        broken: list[Any] = [
            *keystore_module.VERSIONS,
            (
                6,
                "a migration that cannot run",
                ["INSERT INTO no_such_table (x) VALUES (1)"],
            ),
        ]
        with monkeypatch.context() as patch:
            patch.setattr(keystore_module, "VERSIONS", broken)
            with pytest.raises(sqlite3.OperationalError) as caught:
                await KeyStore.create(db)
            assert "no_such_table" in str(caught.value)

        reopened = await KeyStore.create(db)
        try:
            assert await _versions(reopened) == {1, 2, 3, 4, 5}, (
                "the version that failed is not recorded"
            )
        finally:
            await reopened.close()
