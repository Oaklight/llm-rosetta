"""Tests for SQLite-based persistence and request log integration."""

import gzip
import json
import time

import pytest

from llm_rosetta.gateway.admin.persistence import (
    DEFAULT_SUCCESS_MAX,
    PersistenceManager,
)
from llm_rosetta.gateway.admin.request_log import RequestLog, RequestLogEntry


# -- Helpers --


def _make_entry_dict(
    model: str = "gpt-4o",
    status: int = 200,
    provider: str = "openai_chat",
    error_detail: str | None = None,
    api_key_label: str | None = None,
) -> dict:
    e = RequestLogEntry.create(
        model=model,
        source_provider="openai_chat",
        target_provider=provider,
        is_stream=False,
        status_code=status,
        duration_ms=10.0,
        error_detail=error_detail,
        api_key_label=api_key_label,
    )
    return e.to_dict()


def _make_entry(
    model: str = "gpt-4o",
    status: int = 200,
    provider: str = "openai_chat",
) -> RequestLogEntry:
    return RequestLogEntry.create(
        model=model,
        source_provider="openai_chat",
        target_provider=provider,
        is_stream=False,
        status_code=status,
        duration_ms=10.0,
    )


# -- PersistenceManager tests --


class TestPersistenceManagerSchema:
    @pytest.mark.asyncio
    async def test_creates_db_file(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        assert pm.db_path.exists()
        await pm.close()

    @pytest.mark.asyncio
    async def test_wal_mode(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        row = await pm._conn.execute_fetchone("PRAGMA journal_mode")
        assert row[0] == "wal"
        await pm.close()


class TestPersistenceManagerRequestLog:
    @pytest.mark.asyncio
    async def test_insert_and_query(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        entries = [_make_entry_dict(model=f"m-{i}") for i in range(5)]
        await pm.insert_log_entries(entries)

        results, total = await pm.query_log_entries(limit=10)
        assert total == 5
        assert len(results) == 5
        await pm.close()

    @pytest.mark.asyncio
    async def test_newest_first(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        e1 = _make_entry_dict(model="first")
        time.sleep(0.01)  # ensure distinct timestamps
        e2 = _make_entry_dict(model="second")
        await pm.insert_log_entries([e1, e2])

        results, _ = await pm.query_log_entries()
        assert results[0]["model"] == "second"
        assert results[1]["model"] == "first"
        await pm.close()

    @pytest.mark.asyncio
    async def test_filter_by_model(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        await pm.insert_log_entries(
            [
                _make_entry_dict(model="gpt-4o"),
                _make_entry_dict(model="claude"),
                _make_entry_dict(model="gpt-4o"),
            ]
        )

        results, total = await pm.query_log_entries(model="gpt-4o")
        assert total == 2
        assert all(r["model"] == "gpt-4o" for r in results)
        await pm.close()

    @pytest.mark.asyncio
    async def test_filter_by_provider(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        await pm.insert_log_entries(
            [
                _make_entry_dict(provider="openai_chat"),
                _make_entry_dict(provider="anthropic"),
            ]
        )

        results, total = await pm.query_log_entries(provider="anthropic")
        assert total == 1
        assert results[0]["target_provider"] == "anthropic"
        await pm.close()

    @pytest.mark.asyncio
    async def test_filter_by_status(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        await pm.insert_log_entries(
            [
                _make_entry_dict(status=200),
                _make_entry_dict(status=500),
                _make_entry_dict(status=404),
            ]
        )

        ok_results, ok_total = await pm.query_log_entries(status="ok")
        assert ok_total == 1

        err_results, err_total = await pm.query_log_entries(status="error")
        assert err_total == 2
        await pm.close()

    @pytest.mark.asyncio
    async def test_filter_by_api_key_label(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        await pm.insert_log_entries(
            [
                _make_entry_dict(api_key_label="alice"),
                _make_entry_dict(api_key_label="bob"),
                _make_entry_dict(api_key_label="alice"),
                _make_entry_dict(),  # no label
            ]
        )

        results, total = await pm.query_log_entries(api_key_label="alice")
        assert total == 2
        assert all(r["api_key_label"] == "alice" for r in results)

        results, total = await pm.query_log_entries(api_key_label="bob")
        assert total == 1
        await pm.close()

    @pytest.mark.asyncio
    async def test_get_api_key_labels(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        await pm.insert_log_entries(
            [
                _make_entry_dict(api_key_label="bob"),
                _make_entry_dict(api_key_label="alice"),
                _make_entry_dict(api_key_label="bob"),
                _make_entry_dict(),
            ]
        )

        assert await pm.get_api_key_labels() == ["alice", "bob"]
        await pm.close()

    @pytest.mark.asyncio
    async def test_pagination(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        entries = [_make_entry_dict(model=f"m-{i}") for i in range(20)]
        await pm.insert_log_entries(entries)

        page1, total = await pm.query_log_entries(limit=5, offset=0)
        assert total == 20
        assert len(page1) == 5

        page2, _ = await pm.query_log_entries(limit=5, offset=5)
        assert len(page2) == 5
        assert page1[0]["id"] != page2[0]["id"]
        await pm.close()

    @pytest.mark.asyncio
    async def test_get_log_entry(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        entry = _make_entry_dict()
        await pm.insert_log_entries([entry])

        found = await pm.get_log_entry(entry["id"])
        assert found is not None
        assert found["id"] == entry["id"]
        assert found["model"] == entry["model"]
        await pm.close()

    @pytest.mark.asyncio
    async def test_get_log_entry_not_found(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        assert await pm.get_log_entry("nonexistent") is None
        await pm.close()

    @pytest.mark.asyncio
    async def test_clear_log(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        await pm.insert_log_entries([_make_entry_dict() for _ in range(5)])
        assert await pm.count_log_entries() == 5

        await pm.clear_log()
        assert await pm.count_log_entries() == 0
        await pm.close()

    @pytest.mark.asyncio
    async def test_prune(self, tmp_path):
        # Legacy max_entries=N caps successes only; emits DeprecationWarning.
        with pytest.warns(DeprecationWarning):
            pm = await PersistenceManager.create(str(tmp_path), max_entries=10)
        # Insert 150 successful entries in batches to trigger prune.
        for batch in range(3):
            entries = [_make_entry_dict(model=f"m-{batch}-{i}") for i in range(50)]
            await pm.insert_log_entries(entries)

        assert await pm.count_success_entries() <= 10
        await pm.close()


class TestPersistenceManagerRetention:
    """Dual-threshold prune: success and error caps are independent."""

    @pytest.mark.asyncio
    async def test_defaults(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        assert pm.success_max == DEFAULT_SUCCESS_MAX
        assert pm.dump_max == PersistenceManager.DEFAULT_DUMP_MAX
        await pm.close()

    @pytest.mark.asyncio
    async def test_explicit_caps(self, tmp_path):
        pm = await PersistenceManager.create(
            str(tmp_path), success_max=123, dump_max=45
        )
        assert pm.success_max == 123
        assert pm.dump_max == 45
        await pm.close()

    @pytest.mark.asyncio
    async def test_legacy_max_entries_maps_to_success(self, tmp_path):
        with pytest.warns(DeprecationWarning, match="success_max"):
            pm = await PersistenceManager.create(str(tmp_path), max_entries=77)
        assert pm.success_max == 77
        assert pm.dump_max == PersistenceManager.DEFAULT_DUMP_MAX
        await pm.close()

    @pytest.mark.asyncio
    async def test_legacy_does_not_override_explicit_success_max(self, tmp_path):
        with pytest.warns(DeprecationWarning):
            pm = await PersistenceManager.create(
                str(tmp_path), success_max=200, max_entries=77
            )
        # Explicit success_max wins over legacy alias.
        assert pm.success_max == 200
        await pm.close()

    @pytest.mark.asyncio
    async def test_errors_not_evicted_by_success_flood(self, tmp_path):
        # Tiny success cap, generous error cap: a flood of successes must
        # not evict the rare error rows.
        pm = await PersistenceManager.create(str(tmp_path), success_max=20)

        err_entries = [_make_entry_dict(status=500, model=f"e-{i}") for i in range(5)]
        await pm.insert_log_entries(err_entries)

        for batch in range(2):
            ok_entries = [_make_entry_dict(model=f"ok-{batch}-{i}") for i in range(100)]
            await pm.insert_log_entries(ok_entries)

        assert await pm.count_success_entries() <= 20
        assert await pm.count_error_entries() == 5
        await pm.close()

    @pytest.mark.asyncio
    async def test_errors_not_pruned_by_count(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path), success_max=1000)
        # 150 errors, batched to trigger periodic prune at 100.
        for batch in range(3):
            entries = [
                _make_entry_dict(status=500, model=f"e-{batch}-{i}") for i in range(50)
            ]
            await pm.insert_log_entries(entries)

        assert await pm.count_error_entries() == 150
        assert await pm.count_success_entries() == 0
        await pm.close()

    @pytest.mark.asyncio
    async def test_count_success_and_error_separately(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        await pm.insert_log_entries(
            [
                _make_entry_dict(status=200),
                _make_entry_dict(status=201),
                _make_entry_dict(status=404),
                _make_entry_dict(status=500),
                _make_entry_dict(status=502),
            ]
        )
        assert await pm.count_log_entries() == 5
        assert await pm.count_success_entries() == 2
        assert await pm.count_error_entries() == 3
        await pm.close()

    @pytest.mark.asyncio
    async def test_prune_batching_large_excess(self, tmp_path):
        """Large excess is pruned via batched deletion."""
        pm = await PersistenceManager.create(str(tmp_path), success_max=100)
        for batch in range(12):
            entries = [_make_entry_dict(model=f"m-{batch}-{i}") for i in range(1000)]
            await pm.insert_log_entries(entries)

        assert await pm.count_success_entries() == 100
        await pm.close()

    @pytest.mark.asyncio
    async def test_prune_idempotent(self, tmp_path):
        """Calling _prune() when already under cap is a no-op."""
        pm = await PersistenceManager.create(str(tmp_path), success_max=50)
        entries = [_make_entry_dict(model=f"m-{i}") for i in range(30)]
        await pm.insert_log_entries(entries)

        assert await pm.count_success_entries() == 30
        await pm._prune()
        assert await pm.count_success_entries() == 30
        await pm.close()


class TestPersistenceManagerSizes:
    @pytest.mark.asyncio
    async def test_db_file_sizes_keys(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        sizes = pm.db_file_sizes()
        assert set(sizes.keys()) == {"db_bytes", "wal_bytes", "shm_bytes"}
        assert all(isinstance(v, int) for v in sizes.values())
        await pm.close()

    @pytest.mark.asyncio
    async def test_db_file_sizes_nonzero_after_insert(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        await pm.insert_log_entries(
            [_make_entry_dict(model=f"m-{i}") for i in range(50)]
        )
        sizes = pm.db_file_sizes()
        # Main db file always exists after init; WAL is created on first write.
        assert sizes["db_bytes"] > 0
        assert sizes["wal_bytes"] >= 0
        await pm.close()

    @pytest.mark.asyncio
    async def test_bool_roundtrip(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        e = RequestLogEntry.create(
            model="test",
            source_provider="a",
            target_provider="b",
            is_stream=True,
            status_code=200,
            duration_ms=1.0,
        )
        await pm.insert_log_entries([e.to_dict()])

        results, _ = await pm.query_log_entries()
        assert results[0]["is_stream"] is True
        await pm.close()

    @pytest.mark.asyncio
    async def test_error_detail_stored(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        await pm.insert_log_entries(
            [
                _make_entry_dict(error_detail="upstream 500: internal error"),
            ]
        )

        results, _ = await pm.query_log_entries()
        assert results[0]["error_detail"] == "upstream 500: internal error"
        await pm.close()

    @pytest.mark.asyncio
    async def test_none_fields_omitted(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        await pm.insert_log_entries([_make_entry_dict()])

        results, _ = await pm.query_log_entries()
        assert "error_detail" not in results[0]
        assert "api_key_label" not in results[0]
        assert "client_ip" not in results[0]
        await pm.close()


class TestPersistenceManagerMetrics:
    @pytest.mark.asyncio
    async def test_save_and_load(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        data = {"total_requests": 42, "total_errors": 3}
        await pm.save_metrics(data)

        loaded = await pm.load_metrics()
        assert loaded == data
        await pm.close()

    @pytest.mark.asyncio
    async def test_load_empty(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        assert await pm.load_metrics() is None
        await pm.close()

    @pytest.mark.asyncio
    async def test_overwrite(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        await pm.save_metrics({"total_requests": 10})
        await pm.save_metrics({"total_requests": 20})

        loaded = await pm.load_metrics()
        assert loaded is not None
        assert loaded["total_requests"] == 20
        await pm.close()


class TestRebuildFlag:
    @pytest.mark.asyncio
    async def test_flag_not_set_by_default(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        assert await pm.check_and_clear_rebuild_flag() is False
        await pm.close()

    @pytest.mark.asyncio
    async def test_set_then_check_clears(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        await pm.set_rebuild_flag()
        assert await pm.check_and_clear_rebuild_flag() is True
        assert await pm.check_and_clear_rebuild_flag() is False
        await pm.close()

    @pytest.mark.asyncio
    async def test_flag_persists_across_connections(self, tmp_path):
        pm1 = await PersistenceManager.create(str(tmp_path))
        await pm1.set_rebuild_flag()
        await pm1.close()

        pm2 = await PersistenceManager.create(str(tmp_path))
        assert await pm2.check_and_clear_rebuild_flag() is True
        await pm2.close()

    @pytest.mark.asyncio
    async def test_flag_does_not_interfere_with_metrics(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        await pm.save_metrics({"total_requests": 42})
        await pm.set_rebuild_flag()

        assert await pm.load_metrics() == {"total_requests": 42}
        assert await pm.check_and_clear_rebuild_flag() is True
        assert await pm.load_metrics() == {"total_requests": 42}
        await pm.close()

    @pytest.mark.asyncio
    async def test_cleanup_sets_flag_and_rebuild_fixes_counters(self, tmp_path):
        from llm_rosetta.gateway.admin.metrics import MetricsCollector

        pm = await PersistenceManager.create(str(tmp_path))
        rl = RequestLog(persistence=pm)

        for i in range(5):
            await rl.add(
                RequestLogEntry.create(
                    model="m",
                    source_provider="openai_chat",
                    target_provider="openai_chat",
                    is_stream=False,
                    status_code=200,
                    duration_ms=10.0,
                )
            )
        for i in range(3):
            await rl.add(
                RequestLogEntry.create(
                    model="m",
                    source_provider="openai_chat",
                    target_provider="openai_chat",
                    is_stream=False,
                    status_code=503,
                    duration_ms=10.0,
                )
            )

        mc = MetricsCollector()
        rows = [r async for r in pm.iter_log_rows_for_rebuild()]
        mc.rebuild_counters(iter(rows))
        assert mc.total_requests == 8
        assert mc.total_errors == 3

        await pm.cleanup_logs_by_age(0)
        await pm.set_rebuild_flag()

        assert await pm.check_and_clear_rebuild_flag() is True
        rows = [r async for r in pm.iter_log_rows_for_rebuild()]
        mc.rebuild_counters(iter(rows))
        assert mc.total_requests == 0
        assert mc.total_errors == 0
        await pm.close()


# -- Legacy migration tests --


class TestLegacyMigration:
    @pytest.mark.asyncio
    async def test_migrate_jsonl(self, tmp_path):
        # Write legacy JSONL
        entries = [_make_entry_dict(model=f"legacy-{i}") for i in range(3)]
        jsonl_path = tmp_path / "request_log.jsonl"
        with open(jsonl_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        pm = await PersistenceManager.create(str(tmp_path))
        assert await pm.count_log_entries() == 3

        # Legacy file renamed
        assert not jsonl_path.exists()
        assert (tmp_path / "request_log.migrated").exists()
        await pm.close()

    @pytest.mark.asyncio
    async def test_migrate_metrics_json(self, tmp_path):
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps({"total_requests": 99}))

        pm = await PersistenceManager.create(str(tmp_path))
        loaded = await pm.load_metrics()
        assert loaded is not None
        assert loaded["total_requests"] == 99

        assert not metrics_path.exists()
        assert (tmp_path / "metrics.migrated").exists()
        await pm.close()

    @pytest.mark.asyncio
    async def test_migrate_gzip_backups(self, tmp_path):
        # Write gzipped backup
        entries = [_make_entry_dict(model=f"gz-{i}") for i in range(5)]
        gz_path = tmp_path / "request_log.1.jsonl.gz"
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        # Also need the main file to trigger migration
        (tmp_path / "request_log.jsonl").write_text("")

        pm = await PersistenceManager.create(str(tmp_path))
        assert await pm.count_log_entries() == 5
        assert not gz_path.exists()
        await pm.close()

    @pytest.mark.asyncio
    async def test_no_migration_when_clean(self, tmp_path):
        # No legacy files — should just start clean
        pm = await PersistenceManager.create(str(tmp_path))
        assert await pm.count_log_entries() == 0
        assert await pm.load_metrics() is None
        await pm.close()


# -- RequestLog with persistence integration --


class TestRequestLogWithPersistence:
    @pytest.mark.asyncio
    async def test_add_and_get(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        log = RequestLog(persistence=pm)
        await log.add(_make_entry())

        entries, total = await log.get_entries()
        assert total == 1
        assert len(entries) == 1
        await pm.close()

    @pytest.mark.asyncio
    async def test_filter_by_model(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        log = RequestLog(persistence=pm)
        await log.add(_make_entry(model="gpt-4o"))
        await log.add(_make_entry(model="claude"))
        await log.add(_make_entry(model="gpt-4o"))

        entries, total = await log.get_entries(model="gpt-4o")
        assert total == 2
        assert all(e["model"] == "gpt-4o" for e in entries)
        await pm.close()

    @pytest.mark.asyncio
    async def test_filter_by_status(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        log = RequestLog(persistence=pm)
        await log.add(_make_entry(status=200))
        await log.add(_make_entry(status=500))
        await log.add(_make_entry(status=404))

        _, ok_total = await log.get_entries(status="ok")
        assert ok_total == 1
        _, err_total = await log.get_entries(status="error")
        assert err_total == 2
        await pm.close()

    @pytest.mark.asyncio
    async def test_clear(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        log = RequestLog(persistence=pm)
        await log.add(_make_entry())
        await log.add(_make_entry())
        assert await pm.count_log_entries() == 2
        await log.clear()
        assert await pm.count_log_entries() == 0
        await pm.close()

    @pytest.mark.asyncio
    async def test_get_entry_by_id(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        log = RequestLog(persistence=pm)
        e = _make_entry()
        await log.add(e)

        found = await log.get_entry(e.id)
        assert found is not None
        assert found["id"] == e.id
        await pm.close()

    @pytest.mark.asyncio
    async def test_pending_returns_empty(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        log = RequestLog(persistence=pm)
        await log.add(_make_entry())
        assert log.pending_entries() == []
        await pm.close()

    @pytest.mark.asyncio
    async def test_newest_first(self, tmp_path):
        pm = await PersistenceManager.create(str(tmp_path))
        log = RequestLog(persistence=pm)
        await log.add(_make_entry(model="first"))
        time.sleep(0.01)
        await log.add(_make_entry(model="second"))

        entries, _ = await log.get_entries()
        assert entries[0]["model"] == "second"
        assert entries[1]["model"] == "first"
        await pm.close()


class TestCleanupByAge:
    @pytest.mark.asyncio
    async def test_deletes_old_records(self, tmp_path):
        """Records older than max_age_days are deleted."""
        from datetime import datetime, timedelta, timezone

        pm = await PersistenceManager.create(str(tmp_path))

        # Insert entries with explicit old timestamps
        old_ts = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()
        new_ts = datetime.now(timezone.utc).isoformat()

        await pm._conn.execute(
            "INSERT INTO request_log (id, timestamp, model, source_provider, "
            "target_provider, is_stream, status_code, duration_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("old-1", old_ts, "gpt-4o", "openai_chat", "openai_chat", 0, 200, 100),
        )
        await pm._conn.execute(
            "INSERT INTO request_log (id, timestamp, model, source_provider, "
            "target_provider, is_stream, status_code, duration_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("new-1", new_ts, "gpt-4o", "openai_chat", "openai_chat", 0, 200, 50),
        )
        await pm._conn.commit()

        result = await pm.cleanup_by_age(90)
        assert result["request_log_deleted"] == 1
        assert result["max_age_days"] == 90

        # Only the new entry remains
        rows = await pm._conn.execute_fetchall("SELECT id FROM request_log")
        assert len(rows) == 1
        assert rows[0][0] == "new-1"
        await pm.close()

    @pytest.mark.asyncio
    async def test_cleans_orphaned_dump_bodies(self, tmp_path):
        """Orphaned dump_bodies are removed after error_dumps are deleted."""
        from datetime import datetime, timedelta, timezone

        pm = await PersistenceManager.create(str(tmp_path))

        old_ts = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()

        await pm._conn.execute(
            "INSERT INTO dump_bodies (hash, data, orig_bytes, created) "
            "VALUES (?, ?, ?, ?)",
            ("hash-old", b"old body data", 13, old_ts),
        )
        await pm._conn.execute(
            "INSERT INTO error_dumps (id, timestamp, model, source_provider, "
            "target_provider, provider_name, status_code, body_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "ed-1",
                old_ts,
                "gpt-4o",
                "openai_chat",
                "openai_chat",
                "openai",
                500,
                "hash-old",
            ),
        )
        await pm._conn.commit()

        result = await pm.cleanup_by_age(90)
        assert result["error_dumps_deleted"] == 1
        assert result["dump_bodies_deleted"] == 1

        bodies = await pm._conn.execute_fetchall("SELECT hash FROM dump_bodies")
        assert len(bodies) == 0
        await pm.close()

    @pytest.mark.asyncio
    async def test_nothing_deleted_when_all_recent(self, tmp_path):
        """No records are deleted when everything is within the age window."""
        from datetime import datetime, timezone

        pm = await PersistenceManager.create(str(tmp_path))

        new_ts = datetime.now(timezone.utc).isoformat()
        await pm._conn.execute(
            "INSERT INTO request_log (id, timestamp, model, source_provider, "
            "target_provider, is_stream, status_code, duration_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("new-1", new_ts, "gpt-4o", "openai_chat", "openai_chat", 0, 200, 50),
        )
        await pm._conn.commit()

        result = await pm.cleanup_by_age(90)
        assert result["request_log_deleted"] == 0
        assert result["error_dumps_deleted"] == 0
        assert result["dump_bodies_deleted"] == 0

        rows = await pm._conn.execute_fetchall("SELECT id FROM request_log")
        assert len(rows) == 1
        await pm.close()


class TestCleanupLogsIndependent:
    @pytest.mark.asyncio
    async def test_only_logs_deleted(self, tmp_path):
        """cleanup_logs_by_age deletes logs but leaves error dumps untouched."""
        from datetime import datetime, timedelta, timezone

        pm = await PersistenceManager.create(str(tmp_path))
        old_ts = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()

        await pm._conn.execute(
            "INSERT INTO request_log (id, timestamp, model, source_provider, "
            "target_provider, is_stream, status_code, duration_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("old-log", old_ts, "gpt-4o", "openai_chat", "openai_chat", 0, 200, 100),
        )
        await pm._conn.execute(
            "INSERT INTO error_dumps (id, timestamp, model, source_provider, "
            "target_provider, provider_name, status_code) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("old-dump", old_ts, "gpt-4o", "openai_chat", "openai_chat", "openai", 500),
        )
        await pm._conn.commit()

        result = await pm.cleanup_logs_by_age(90)
        assert result["deleted"] == 1

        # Error dump should still be there
        dumps = await pm._conn.execute_fetchall("SELECT id FROM error_dumps")
        assert len(dumps) == 1
        assert dumps[0][0] == "old-dump"
        await pm.close()


class TestCleanupErrorsIndependent:
    @pytest.mark.asyncio
    async def test_only_errors_deleted(self, tmp_path):
        """cleanup_error_dumps_by_age deletes dumps but leaves logs untouched."""
        from datetime import datetime, timedelta, timezone

        pm = await PersistenceManager.create(str(tmp_path))
        old_ts = (datetime.now(timezone.utc) - timedelta(days=100)).isoformat()

        await pm._conn.execute(
            "INSERT INTO request_log (id, timestamp, model, source_provider, "
            "target_provider, is_stream, status_code, duration_ms) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            ("old-log", old_ts, "gpt-4o", "openai_chat", "openai_chat", 0, 200, 100),
        )
        await pm._conn.execute(
            "INSERT INTO dump_bodies (hash, data, orig_bytes, created) "
            "VALUES (?, ?, ?, ?)",
            ("hash-old", b"body data", 9, old_ts),
        )
        await pm._conn.execute(
            "INSERT INTO error_dumps (id, timestamp, model, source_provider, "
            "target_provider, provider_name, status_code, body_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "old-dump",
                old_ts,
                "gpt-4o",
                "openai_chat",
                "openai_chat",
                "openai",
                500,
                "hash-old",
            ),
        )
        await pm._conn.commit()

        result = await pm.cleanup_error_dumps_by_age(90)
        assert result["error_dumps_deleted"] == 1
        assert result["dump_bodies_deleted"] == 1

        # Request log should still be there
        logs = await pm._conn.execute_fetchall("SELECT id FROM request_log")
        assert len(logs) == 1
        assert logs[0][0] == "old-log"
        await pm.close()


class TestExportErrorDumps:
    @pytest.mark.asyncio
    async def test_export_creates_valid_archive(self, tmp_path):
        """export_error_dumps returns a valid tar.gz with metadata and bodies."""
        import io
        import tarfile
        from datetime import datetime, timezone

        pm = await PersistenceManager.create(str(tmp_path))
        ts = datetime.now(timezone.utc).isoformat()

        await pm._conn.execute(
            "INSERT INTO dump_bodies (hash, data, orig_bytes, created) "
            "VALUES (?, ?, ?, ?)",
            ("abc123", b"test body content", 17, ts),
        )
        await pm._conn.execute(
            "INSERT INTO error_dumps (id, timestamp, model, source_provider, "
            "target_provider, provider_name, status_code, body_hash) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "dump-1",
                ts,
                "gpt-4o",
                "openai_chat",
                "openai_chat",
                "openai",
                500,
                "abc123",
            ),
        )
        await pm._conn.commit()

        data = await pm.export_error_dumps()
        assert len(data) > 0

        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            names = tar.getnames()
            assert "metadata.json" in names
            assert "bodies/abc123.bin" in names

            f = tar.extractfile("metadata.json")
            assert f is not None
            meta = json.loads(f.read())
            assert len(meta) == 1
            assert meta[0]["id"] == "dump-1"
            assert meta[0]["body_hash"] == "abc123"

            f = tar.extractfile("bodies/abc123.bin")
            assert f is not None
            body = f.read()
            assert body == b"test body content"
        await pm.close()

    @pytest.mark.asyncio
    async def test_export_with_date_range(self, tmp_path):
        """export_error_dumps respects start/end filters."""

        pm = await PersistenceManager.create(str(tmp_path))

        await pm._conn.execute(
            "INSERT INTO error_dumps (id, timestamp, model, source_provider, "
            "target_provider, provider_name, status_code) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("old", "2026-01-01T00:00:00Z", "gpt-4o", "oc", "oc", "openai", 500),
        )
        await pm._conn.execute(
            "INSERT INTO error_dumps (id, timestamp, model, source_provider, "
            "target_provider, provider_name, status_code) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("new", "2026-08-01T00:00:00Z", "gpt-4o", "oc", "oc", "openai", 500),
        )
        await pm._conn.commit()

        data = await pm.export_error_dumps(start="2026-07-01T00:00:00Z")
        import io
        import tarfile

        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            f = tar.extractfile("metadata.json")
            assert f is not None
            meta = json.loads(f.read())
            assert len(meta) == 1
            assert meta[0]["id"] == "new"
        await pm.close()

    @pytest.mark.asyncio
    async def test_export_empty(self, tmp_path):
        """export_error_dumps returns valid empty archive when no matches."""
        import io
        import tarfile

        pm = await PersistenceManager.create(str(tmp_path))
        data = await pm.export_error_dumps()
        assert len(data) > 0

        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
            f = tar.extractfile("metadata.json")
            assert f is not None
            meta = json.loads(f.read())
            assert meta == []
        await pm.close()
