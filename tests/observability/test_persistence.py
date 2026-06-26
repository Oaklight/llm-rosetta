"""Tests for the observability PersistenceManager (standalone, no gateway)."""

import pytest

from llm_rosetta.observability import PersistenceManager, RequestLogEntry


@pytest.fixture
def pm(tmp_path):
    """Create a PersistenceManager using a temp directory."""
    return PersistenceManager(str(tmp_path), success_max=100, error_max=50)


class TestPersistenceManager:
    def test_insert_and_query(self, pm):
        entry = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=False,
            status_code=200,
            duration_ms=50.0,
        )
        pm.insert_log_entries([entry.to_dict()])
        entries, total = pm.query_log_entries(limit=10)
        assert total == 1
        assert entries[0]["model"] == "gpt-4o"

    def test_metrics_save_load(self, pm):
        data = {"total_requests": 42, "total_errors": 3}
        pm.save_metrics(data)
        loaded = pm.load_metrics()
        assert loaded == data

    def test_count_methods(self, pm):
        for sc in [200, 200, 500]:
            entry = RequestLogEntry.create(
                model="gpt-4o",
                source_provider="openai_chat",
                target_provider="anthropic",
                is_stream=False,
                status_code=sc,
                duration_ms=10.0,
            )
            pm.insert_log_entries([entry.to_dict()])
        assert pm.count_log_entries() == 3
        assert pm.count_success_entries() == 2
        assert pm.count_error_entries() == 1

    def test_clear_log(self, pm):
        entry = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=False,
            status_code=200,
            duration_ms=10.0,
        )
        pm.insert_log_entries([entry.to_dict()])
        assert pm.count_log_entries() == 1
        pm.clear_log()
        assert pm.count_log_entries() == 0

    def test_db_file_sizes(self, pm):
        sizes = pm.db_file_sizes()
        assert "db_bytes" in sizes
        assert sizes["db_bytes"] >= 0

    def test_close(self, pm):
        pm.close()
        # Should not raise on double close
        pm.close()


class TestPersistenceRetention:
    def test_prune_success(self, tmp_path):
        pm = PersistenceManager(str(tmp_path), success_max=5, error_max=5)
        for i in range(10):
            entry = RequestLogEntry.create(
                model=f"model-{i}",
                source_provider="openai_chat",
                target_provider="anthropic",
                is_stream=False,
                status_code=200,
                duration_ms=10.0,
            )
            pm.insert_log_entries([entry.to_dict()])
        # After pruning, should have at most success_max
        assert pm.count_success_entries() <= 5

    def test_prune_errors_independently(self, tmp_path):
        pm = PersistenceManager(str(tmp_path), success_max=5, error_max=3)
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
            pm.insert_log_entries([entry.to_dict()])
        assert pm.count_error_entries() <= 3
