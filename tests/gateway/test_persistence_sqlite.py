"""Tests for SQLite-based persistence and request log integration."""

import gzip
import json
import time

from llm_rosetta.gateway.admin.persistence import PersistenceManager
from llm_rosetta.gateway.admin.request_log import RequestLog, RequestLogEntry


# -- Helpers --


def _make_entry_dict(
    model: str = "gpt-4o",
    status: int = 200,
    provider: str = "openai_chat",
    error_detail: str | None = None,
) -> dict:
    e = RequestLogEntry.create(
        model=model,
        source_provider="openai_chat",
        target_provider=provider,
        is_stream=False,
        status_code=status,
        duration_ms=10.0,
        error_detail=error_detail,
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
    def test_creates_db_file(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        assert pm.db_path.exists()
        pm.close()

    def test_wal_mode(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        row = pm._conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal"
        pm.close()


class TestPersistenceManagerRequestLog:
    def test_insert_and_query(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        entries = [_make_entry_dict(model=f"m-{i}") for i in range(5)]
        pm.insert_log_entries(entries)

        results, total = pm.query_log_entries(limit=10)
        assert total == 5
        assert len(results) == 5
        pm.close()

    def test_newest_first(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        e1 = _make_entry_dict(model="first")
        time.sleep(0.01)  # ensure distinct timestamps
        e2 = _make_entry_dict(model="second")
        pm.insert_log_entries([e1, e2])

        results, _ = pm.query_log_entries()
        assert results[0]["model"] == "second"
        assert results[1]["model"] == "first"
        pm.close()

    def test_filter_by_model(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        pm.insert_log_entries(
            [
                _make_entry_dict(model="gpt-4o"),
                _make_entry_dict(model="claude"),
                _make_entry_dict(model="gpt-4o"),
            ]
        )

        results, total = pm.query_log_entries(model="gpt-4o")
        assert total == 2
        assert all(r["model"] == "gpt-4o" for r in results)
        pm.close()

    def test_filter_by_provider(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        pm.insert_log_entries(
            [
                _make_entry_dict(provider="openai_chat"),
                _make_entry_dict(provider="anthropic"),
            ]
        )

        results, total = pm.query_log_entries(provider="anthropic")
        assert total == 1
        assert results[0]["target_provider"] == "anthropic"
        pm.close()

    def test_filter_by_status(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        pm.insert_log_entries(
            [
                _make_entry_dict(status=200),
                _make_entry_dict(status=500),
                _make_entry_dict(status=404),
            ]
        )

        ok_results, ok_total = pm.query_log_entries(status="ok")
        assert ok_total == 1

        err_results, err_total = pm.query_log_entries(status="error")
        assert err_total == 2
        pm.close()

    def test_pagination(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        entries = [_make_entry_dict(model=f"m-{i}") for i in range(20)]
        pm.insert_log_entries(entries)

        page1, total = pm.query_log_entries(limit=5, offset=0)
        assert total == 20
        assert len(page1) == 5

        page2, _ = pm.query_log_entries(limit=5, offset=5)
        assert len(page2) == 5
        assert page1[0]["id"] != page2[0]["id"]
        pm.close()

    def test_get_log_entry(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        entry = _make_entry_dict()
        pm.insert_log_entries([entry])

        found = pm.get_log_entry(entry["id"])
        assert found is not None
        assert found["id"] == entry["id"]
        assert found["model"] == entry["model"]
        pm.close()

    def test_get_log_entry_not_found(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        assert pm.get_log_entry("nonexistent") is None
        pm.close()

    def test_clear_log(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        pm.insert_log_entries([_make_entry_dict() for _ in range(5)])
        assert pm.count_log_entries() == 5

        pm.clear_log()
        assert pm.count_log_entries() == 0
        pm.close()

    def test_prune(self, tmp_path):
        pm = PersistenceManager(str(tmp_path), max_entries=10)
        # Insert 15 entries in batches to trigger prune
        for batch in range(3):
            entries = [_make_entry_dict(model=f"m-{batch}-{i}") for i in range(50)]
            pm.insert_log_entries(entries)

        assert pm.count_log_entries() <= 10
        pm.close()

    def test_bool_roundtrip(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        e = RequestLogEntry.create(
            model="test",
            source_provider="a",
            target_provider="b",
            is_stream=True,
            status_code=200,
            duration_ms=1.0,
        )
        pm.insert_log_entries([e.to_dict()])

        results, _ = pm.query_log_entries()
        assert results[0]["is_stream"] is True
        pm.close()

    def test_error_detail_stored(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        pm.insert_log_entries(
            [
                _make_entry_dict(error_detail="upstream 500: internal error"),
            ]
        )

        results, _ = pm.query_log_entries()
        assert results[0]["error_detail"] == "upstream 500: internal error"
        pm.close()

    def test_none_fields_omitted(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        pm.insert_log_entries([_make_entry_dict()])

        results, _ = pm.query_log_entries()
        assert "error_detail" not in results[0]
        assert "api_key_label" not in results[0]
        pm.close()


class TestPersistenceManagerMetrics:
    def test_save_and_load(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        data = {"total_requests": 42, "total_errors": 3}
        pm.save_metrics(data)

        loaded = pm.load_metrics()
        assert loaded == data
        pm.close()

    def test_load_empty(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        assert pm.load_metrics() is None
        pm.close()

    def test_overwrite(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        pm.save_metrics({"total_requests": 10})
        pm.save_metrics({"total_requests": 20})

        loaded = pm.load_metrics()
        assert loaded["total_requests"] == 20
        pm.close()


# -- Legacy migration tests --


class TestLegacyMigration:
    def test_migrate_jsonl(self, tmp_path):
        # Write legacy JSONL
        entries = [_make_entry_dict(model=f"legacy-{i}") for i in range(3)]
        jsonl_path = tmp_path / "request_log.jsonl"
        with open(jsonl_path, "w") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")

        pm = PersistenceManager(str(tmp_path))
        assert pm.count_log_entries() == 3

        # Legacy file renamed
        assert not jsonl_path.exists()
        assert (tmp_path / "request_log.migrated").exists()
        pm.close()

    def test_migrate_metrics_json(self, tmp_path):
        metrics_path = tmp_path / "metrics.json"
        metrics_path.write_text(json.dumps({"total_requests": 99}))

        pm = PersistenceManager(str(tmp_path))
        loaded = pm.load_metrics()
        assert loaded["total_requests"] == 99

        assert not metrics_path.exists()
        assert (tmp_path / "metrics.migrated").exists()
        pm.close()

    def test_migrate_gzip_backups(self, tmp_path):
        # Write gzipped backup
        entries = [_make_entry_dict(model=f"gz-{i}") for i in range(5)]
        gz_path = tmp_path / "request_log.1.jsonl.gz"
        with gzip.open(gz_path, "wt", encoding="utf-8") as f:
            for e in entries:
                f.write(json.dumps(e) + "\n")
        # Also need the main file to trigger migration
        (tmp_path / "request_log.jsonl").write_text("")

        pm = PersistenceManager(str(tmp_path))
        assert pm.count_log_entries() == 5
        assert not gz_path.exists()
        pm.close()

    def test_no_migration_when_clean(self, tmp_path):
        # No legacy files — should just start clean
        pm = PersistenceManager(str(tmp_path))
        assert pm.count_log_entries() == 0
        assert pm.load_metrics() is None
        pm.close()


# -- RequestLog with persistence integration --


class TestRequestLogWithPersistence:
    def test_add_and_get(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        log = RequestLog(persistence=pm)
        log.add(_make_entry())

        entries, total = log.get_entries()
        assert total == 1
        assert len(entries) == 1
        pm.close()

    def test_filter_by_model(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        log = RequestLog(persistence=pm)
        log.add(_make_entry(model="gpt-4o"))
        log.add(_make_entry(model="claude"))
        log.add(_make_entry(model="gpt-4o"))

        entries, total = log.get_entries(model="gpt-4o")
        assert total == 2
        assert all(e["model"] == "gpt-4o" for e in entries)
        pm.close()

    def test_filter_by_status(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        log = RequestLog(persistence=pm)
        log.add(_make_entry(status=200))
        log.add(_make_entry(status=500))
        log.add(_make_entry(status=404))

        _, ok_total = log.get_entries(status="ok")
        assert ok_total == 1
        _, err_total = log.get_entries(status="error")
        assert err_total == 2
        pm.close()

    def test_clear(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        log = RequestLog(persistence=pm)
        log.add(_make_entry())
        log.add(_make_entry())
        assert len(log) == 2
        log.clear()
        assert len(log) == 0
        pm.close()

    def test_get_entry_by_id(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        log = RequestLog(persistence=pm)
        e = _make_entry()
        log.add(e)

        found = log.get_entry(e.id)
        assert found is not None
        assert found["id"] == e.id
        pm.close()

    def test_pending_returns_empty(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        log = RequestLog(persistence=pm)
        log.add(_make_entry())
        assert log.pending_entries() == []
        pm.close()

    def test_newest_first(self, tmp_path):
        pm = PersistenceManager(str(tmp_path))
        log = RequestLog(persistence=pm)
        log.add(_make_entry(model="first"))
        time.sleep(0.01)
        log.add(_make_entry(model="second"))

        entries, _ = log.get_entries()
        assert entries[0]["model"] == "second"
        assert entries[1]["model"] == "first"
        pm.close()
