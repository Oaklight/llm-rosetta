"""Tests for the observability PersistenceManager (standalone, no gateway)."""

import pytest

from llm_rosetta_gateway.observability import PersistenceManager, RequestLogEntry


@pytest.fixture
async def pm(tmp_path):
    """Create a PersistenceManager using a temp directory."""
    return await PersistenceManager.create(str(tmp_path), success_max=100)


class TestPersistenceManager:
    @pytest.mark.asyncio
    async def test_insert_and_query(self, pm):
        entry = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=False,
            status_code=200,
            duration_ms=50.0,
        )
        await pm.insert_log_entries([entry.to_dict()])
        entries, total = await pm.query_log_entries(limit=10)
        assert total == 1
        assert entries[0]["model"] == "gpt-4o"

    @pytest.mark.asyncio
    async def test_metrics_save_load(self, pm):
        data = {"total_requests": 42, "total_errors": 3}
        await pm.save_metrics(data)
        loaded = await pm.load_metrics()
        assert loaded == data

    @pytest.mark.asyncio
    async def test_count_methods(self, pm):
        for sc in [200, 200, 500]:
            entry = RequestLogEntry.create(
                model="gpt-4o",
                source_provider="openai_chat",
                target_provider="anthropic",
                is_stream=False,
                status_code=sc,
                duration_ms=10.0,
            )
            await pm.insert_log_entries([entry.to_dict()])
        assert await pm.count_log_entries() == 3
        assert await pm.count_success_entries() == 2
        assert await pm.count_error_entries() == 1

    @pytest.mark.asyncio
    async def test_clear_log(self, pm):
        entry = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=False,
            status_code=200,
            duration_ms=10.0,
        )
        await pm.insert_log_entries([entry.to_dict()])
        assert await pm.count_log_entries() == 1
        await pm.clear_log()
        assert await pm.count_log_entries() == 0

    @pytest.mark.asyncio
    async def test_db_file_sizes(self, pm):
        sizes = pm.db_file_sizes()
        assert "db_bytes" in sizes
        assert sizes["db_bytes"] >= 0

    @pytest.mark.asyncio
    async def test_close(self, pm):
        await pm.close()
        # Should not raise on double close
        await pm.close()


class TestPersistenceRetention:
    @pytest.mark.asyncio
    async def test_prune_success(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path), success_max=5)
        for i in range(10):
            entry = RequestLogEntry.create(
                model=f"model-{i}",
                source_provider="openai_chat",
                target_provider="anthropic",
                is_stream=False,
                status_code=200,
                duration_ms=10.0,
            )
            await pm.insert_log_entries([entry.to_dict()])
        # After pruning, should have at most success_max
        assert await pm.count_success_entries() <= 5

    @pytest.mark.asyncio
    async def test_errors_not_pruned_by_count(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path), success_max=5)
        # Add errors
        for i in range(10):
            entry = RequestLogEntry.create(
                model=f"err-{i}",
                source_provider="openai_chat",
                target_provider="anthropic",
                is_stream=False,
                status_code=500,
                duration_ms=10.0,
            )
            await pm.insert_log_entries([entry.to_dict()])
        assert await pm.count_error_entries() == 10


class TestSchemaVersioning:
    async def test_new_db_has_latest_version(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        version = await pm.schema_version()
        # Read off the table rather than restating the number, so adding a
        # migration does not need this test edited.
        assert version == PersistenceManager._MIGRATIONS[-1][0]

    async def test_schema_version_table_exists(self, pm):
        row = await pm._conn.execute_fetchone(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='schema_version'"
        )
        assert row is not None

    async def test_migration_history_recorded(self, pm):
        cursor = await pm._conn.execute(
            "SELECT version, applied_at FROM schema_version ORDER BY version"
        )
        rows = await cursor.fetchall()
        assert [r[0] for r in rows] == [
            v for v, _d, _s in PersistenceManager._MIGRATIONS
        ]
        # applied_at should be ISO timestamps
        assert "T" in rows[0][1]

    async def test_error_dumps_indexes_exist(self, pm):
        cursor = await pm._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' "
            "AND tbl_name='error_dumps'"
        )
        indexes = {r[0] for r in await cursor.fetchall()}
        assert "idx_ed_model" in indexes
        assert "idx_ed_error_phase" in indexes

    async def test_error_dumps_has_tool_name_collisions_column(self, pm):
        """Migration v3 adds the tool-name collision column."""
        cursor = await pm._conn.execute("PRAGMA table_info(error_dumps)")
        columns = {r[1] for r in await cursor.fetchall()}
        assert "tool_name_collisions" in columns

    async def test_request_log_has_api_key_bucket_column(self, pm):
        """Migration v4 adds the per-key usage attribution column."""
        cursor = await pm._conn.execute("PRAGMA table_info(request_log)")
        columns = {r[1] for r in await cursor.fetchall()}
        assert "api_key_bucket" in columns

    async def test_foreign_keys_enabled(self, pm):
        row = await pm._conn.execute_fetchone("PRAGMA foreign_keys")
        assert row[0] == 1

    async def test_error_dumps_fk_on_new_db(self, tmp_path):
        """New databases should have FK constraint on error_dumps.request_log_id."""
        pm = await PersistenceManager.create(str(tmp_path))
        cursor = await pm._conn.execute("PRAGMA foreign_key_list(error_dumps)")
        fks = await cursor.fetchall()
        assert len(fks) >= 1
        # FK should reference request_log(id)
        fk = fks[0]
        assert fk[2] == "request_log"  # table
        assert fk[3] == "request_log_id"  # from
        assert fk[4] == "id"  # to

    async def test_idempotent_migrations(self, tmp_path):
        """Running migrations twice should not fail."""
        pm = await PersistenceManager.create(str(tmp_path))
        v1 = await pm.schema_version()
        await pm.close()
        # Re-open — migrations should detect already-applied and skip
        pm2 = await PersistenceManager.create(str(tmp_path))
        v2 = await pm2.schema_version()
        assert v1 == v2

    async def test_bootstrap_detects_old_style_migration(self, tmp_path):
        """Databases that had _migrate_add_columns should get v1 auto-detected."""
        from llm_rosetta_gateway._vendor import aiosqlite

        db_path = tmp_path / "gateway.db"
        # Simulate old database: create tables manually without schema_version data
        conn = await aiosqlite.connect(str(db_path))
        await conn.executescript("""
            CREATE TABLE request_log (
                id TEXT PRIMARY KEY, timestamp TEXT NOT NULL,
                model TEXT NOT NULL, source_provider TEXT NOT NULL,
                target_provider TEXT NOT NULL, is_stream INTEGER NOT NULL,
                status_code INTEGER NOT NULL, duration_ms REAL NOT NULL,
                error_detail TEXT, api_key_label TEXT,
                target_provider_name TEXT, client_ip TEXT,
                profile TEXT, input_tokens INTEGER,
                output_tokens INTEGER, total_tokens INTEGER,
                cache_read_tokens INTEGER, cache_creation_tokens INTEGER,
                reasoning_tokens INTEGER
            );
            CREATE TABLE metrics (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE dump_bodies (
                hash TEXT PRIMARY KEY, data BLOB NOT NULL,
                orig_bytes INTEGER NOT NULL, created TEXT NOT NULL
            );
            CREATE TABLE error_dumps (
                id TEXT PRIMARY KEY, request_log_id TEXT,
                timestamp TEXT NOT NULL, model TEXT,
                source_provider TEXT, target_provider TEXT,
                provider_name TEXT, status_code INTEGER,
                error_phase TEXT, body_hash TEXT,
                response_text TEXT, upstream_url TEXT,
                converted_body_hash TEXT
            );
            CREATE TABLE ops_log (
                id TEXT PRIMARY KEY, timestamp TEXT NOT NULL,
                event_type TEXT NOT NULL, severity TEXT NOT NULL,
                message TEXT NOT NULL, details TEXT, source TEXT
            );
        """)
        await conn.close()

        # Now open with PersistenceManager — should bootstrap v1 and apply
        # every migration after it.
        pm = await PersistenceManager.create(str(tmp_path))
        assert await pm.schema_version() == PersistenceManager._MIGRATIONS[-1][0]


class TestWALCheckpoint:
    async def test_wal_size_initially_zero_or_small(self, pm):
        assert pm.wal_size() >= 0

    async def test_wal_checkpoint_returns_status(self, pm):
        result = await pm.wal_checkpoint()
        assert result["ok"] is True
        assert "wal_before" in result
        assert "wal_after" in result

    async def test_wal_checkpoint_after_inserts(self, pm):
        for i in range(10):
            entry = RequestLogEntry.create(
                model=f"m-{i}",
                source_provider="openai_chat",
                target_provider="anthropic",
                is_stream=False,
                status_code=200,
                duration_ms=10.0,
            )
            await pm.insert_log_entries([entry.to_dict()])
        result = await pm.wal_checkpoint()
        assert result["ok"] is True

    async def test_start_stop_wal_task(self, pm):
        pm.start_wal_task()
        assert pm._wal_task is not None
        assert not pm._wal_task.done()
        await pm.stop_wal_task()
        assert pm._wal_task is None

    async def test_close_stops_wal_task(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        pm.start_wal_task()
        assert pm._wal_task is not None
        await pm.close()
        assert pm._wal_task is None


class TestBackupAndIntegrity:
    async def test_backup_created_on_open(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        await pm.close()
        # Re-open: should create a backup of existing DB
        pm2 = await PersistenceManager.create(str(tmp_path))
        backup_dir = tmp_path / "backups"
        assert backup_dir.exists()
        backups = list(backup_dir.glob("gateway-*.db"))
        assert len(backups) == 1
        assert backups[0].stat().st_size > 0
        await pm2.close()

    async def test_backup_keeps_max_copies(self, tmp_path):
        # Create initial DB
        pm = await PersistenceManager.create(str(tmp_path))
        await pm.close()
        # Re-open 5 times to create 5 backups
        import asyncio

        for _ in range(5):
            await asyncio.sleep(0.01)  # ensure unique timestamps
            pm = await PersistenceManager.create(str(tmp_path))
            await pm.close()
        backup_dir = tmp_path / "backups"
        backups = list(backup_dir.glob("gateway-*.db"))
        assert len(backups) <= 3  # _BACKUP_MAX_KEEP

    async def test_integrity_check_passes_on_new_db(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        await pm.close()
        # Re-open triggers integrity check
        pm2 = await PersistenceManager.create(str(tmp_path))
        await pm2.close()  # no error = integrity ok

    async def test_no_backup_on_fresh_db(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        backup_dir = tmp_path / "backups"
        assert not backup_dir.exists()
        await pm.close()


class TestBackfillTotalTokens:
    """The backfill and the writer have to apply one rule (review on #1014).

    The old backfill rewrote *any* stored total below input + output + cache,
    which for the OpenAI-shaped standards is the provider's own answer — their
    cached counters sit inside the prompt count.  Every restart then moved
    those rows, and a row's meaning depended on whether a restart had touched
    it.
    """

    @staticmethod
    def _row(row_id, *, target="chat_completions", total=None, inp=40, out=5, **extra):
        entry = RequestLogEntry.create(
            model="m",
            source_provider="chat_completions",
            target_provider=target,
            is_stream=False,
            status_code=200,
            duration_ms=1.0,
            input_tokens=inp,
            output_tokens=out,
            total_tokens=total,
        )
        row = entry.to_dict() | {"id": row_id}
        row.update(extra)
        return row

    @pytest.mark.asyncio
    async def test_a_provider_total_below_the_additive_sum_survives(self, pm):
        """OpenAI reports cached tokens inside the prompt: 50 is the answer."""
        await pm.insert_log_entries(
            [self._row("openai", total=50, inp=40, out=5, cache_read_tokens=20)]
        )

        assert await pm.backfill_total_tokens() == 0

        entries, _ = await pm.query_log_entries(limit=10)
        assert entries[0]["total_tokens"] == 50

    @pytest.mark.asyncio
    async def test_a_derived_anthropic_total_is_corrected(self, pm):
        """Messages reports no total, and its cache tokens are additive."""
        await pm.insert_log_entries(
            [
                self._row(
                    "anthropic", target="messages", total=45, cache_read_tokens=20
                ),
            ]
        )

        assert await pm.backfill_total_tokens() == 1

        entries, _ = await pm.query_log_entries(limit=10)
        assert entries[0]["total_tokens"] == 65

    @pytest.mark.asyncio
    async def test_the_legacy_anthropic_spelling_is_corrected_too(self, pm):
        await pm.insert_log_entries(
            [
                self._row("legacy", target="anthropic", total=45, cache_read_tokens=20),
            ]
        )
        assert await pm.backfill_total_tokens() == 1

    @pytest.mark.asyncio
    async def test_a_missing_total_is_filled(self, pm):
        """A NULL counts as nothing in every view that sums the column."""
        await pm.insert_log_entries([self._row("gap", total=None)])

        assert await pm.backfill_total_tokens() == 1

        entries, _ = await pm.query_log_entries(limit=10)
        assert entries[0]["total_tokens"] == 45

    @pytest.mark.asyncio
    async def test_a_row_with_no_usage_at_all_is_left_alone(self, pm):
        """An error entry has nothing to total; it must not become 0."""
        entry = RequestLogEntry.create(
            model="m",
            source_provider="chat_completions",
            target_provider="chat_completions",
            is_stream=False,
            status_code=502,
            duration_ms=1.0,
        )
        await pm.insert_log_entries([entry.to_dict()])

        assert await pm.backfill_total_tokens() == 0

    @pytest.mark.asyncio
    async def test_second_run_is_a_noop(self, pm):
        await pm.insert_log_entries(
            [
                self._row(
                    "anthropic", target="messages", total=45, cache_read_tokens=20
                ),
                self._row("openai", total=50, cache_read_tokens=20),
                self._row("gap", total=None),
            ]
        )

        first = await pm.backfill_total_tokens()
        after_first = {
            e["id"]: e["total_tokens"]
            for e in (await pm.query_log_entries(limit=10))[0]
        }
        assert await pm.backfill_total_tokens() == 0

        after_second = {
            e["id"]: e["total_tokens"]
            for e in (await pm.query_log_entries(limit=10))[0]
        }
        assert after_first == after_second
        # 1 derived Messages row + 1 blank row; the OpenAI row keeps its own.
        assert first == 2


class TestUsageByModel:
    """The reading that answers "which models moved the tokens".

    The subject cut answers who spent them; the model name is a column of the
    row rather than an attribution, so it is a second read.  It feeds the usage
    CLI's model table, which is why the two must agree about a row's model
    type and about what a window means.
    """

    @staticmethod
    def _row(
        row_id: str,
        model: str,
        total: int,
        *,
        model_type: str | None = None,
        minutes: int = 5,
    ) -> dict:
        from datetime import datetime, timedelta, timezone

        entry = RequestLogEntry.create(
            model=model,
            source_provider="chat_completions",
            target_provider="chat_completions",
            is_stream=False,
            status_code=200,
            duration_ms=1.0,
            input_tokens=total,
            output_tokens=0,
            total_tokens=total,
        )
        row = entry.to_dict() | {"id": row_id}
        row["timestamp"] = (
            datetime.now(timezone.utc) - timedelta(minutes=minutes)
        ).isoformat()
        if model_type is not None:
            row["model_type"] = model_type
        return row

    @pytest.mark.asyncio
    async def test_one_model_keeps_its_types_apart(self, pm):
        """The same name can serve two model types, and each is its own row."""
        await pm.insert_log_entries(
            [
                self._row("chat", "shared-name", 100, model_type="llm"),
                self._row("embed", "shared-name", 300, model_type="embeddings"),
            ]
        )

        groups = (await pm.query_usage_by_model())["groups"]
        by_type = {group["model_type"]: group["windows"]["24h"] for group in groups}
        assert by_type["llm"]["total_tokens"] == 100
        assert by_type["llm"]["request_count"] == 1
        assert by_type["embeddings"]["total_tokens"] == 300

    @pytest.mark.asyncio
    async def test_the_biggest_model_comes_first(self, pm):
        """Ordered by the widest window, so the leader is the month's leader."""
        await pm.insert_log_entries(
            [
                self._row("small", "gpt-5-nano", 10),
                self._row("big", "gpt-5", 900),
            ]
        )

        groups = (await pm.query_usage_by_model())["groups"]
        assert [group["model"] for group in groups] == ["gpt-5", "gpt-5-nano"]

    @pytest.mark.asyncio
    async def test_a_row_predating_the_column_counts_as_chat(self, pm):
        """A row written before ``model_type`` existed has no type to report."""
        await pm.insert_log_entries([self._row("old", "gpt-5", 40)])

        groups = (await pm.query_usage_by_model())["groups"]
        assert groups[0]["model_type"] == "llm"

    @pytest.mark.asyncio
    async def test_the_windows_narrow_with_age(self, pm):
        """A row older than a day is in the week and out of the day."""
        await pm.insert_log_entries(
            [
                self._row("today", "gpt-5", 100, minutes=10),
                self._row("yesterday", "gpt-5", 7, minutes=60 * 30),
            ]
        )

        windows = (await pm.query_usage_by_model())["groups"][0]["windows"]
        assert windows["24h"]["total_tokens"] == 100
        assert windows["7d"]["total_tokens"] == 107
