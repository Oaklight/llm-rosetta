"""Tests for the observability PersistenceManager (standalone, no gateway)."""

import pytest

from llm_rosetta.observability import PersistenceManager, RequestLogEntry


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
        assert version == 2  # latest migration version

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
        assert len(rows) == 2
        assert rows[0][0] == 1
        assert rows[1][0] == 2
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
        from llm_rosetta._vendor import aiosqlite

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

        # Now open with PersistenceManager — should bootstrap v1 and apply v2
        pm = await PersistenceManager.create(str(tmp_path))
        assert await pm.schema_version() == 2


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
