"""Tests for token usage tracking across the observability stack."""

from __future__ import annotations

import pytest

from llm_rosetta.observability.metrics import MetricsCollector
from llm_rosetta.observability.request_log import RequestLog, RequestLogEntry


class TestRequestLogEntryTokenFields:
    def test_create_with_tokens(self):
        entry = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=False,
            status_code=200,
            duration_ms=150.0,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
        assert entry.input_tokens == 100
        assert entry.output_tokens == 50
        assert entry.total_tokens == 150

    def test_create_without_tokens(self):
        entry = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=False,
            status_code=200,
            duration_ms=150.0,
        )
        assert entry.input_tokens is None
        assert entry.output_tokens is None
        assert entry.total_tokens is None

    def test_to_dict_includes_tokens(self):
        entry = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=False,
            status_code=200,
            duration_ms=150.0,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
        d = entry.to_dict()
        assert d["input_tokens"] == 100
        assert d["output_tokens"] == 50
        assert d["total_tokens"] == 150

    def test_to_dict_omits_none_tokens(self):
        entry = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=False,
            status_code=200,
            duration_ms=150.0,
        )
        d = entry.to_dict()
        assert "input_tokens" not in d
        assert "output_tokens" not in d
        assert "total_tokens" not in d


class TestRequestLogUpdateUsage:
    def test_update_usage_in_memory(self):
        log = RequestLog(persistence=None)
        entry = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=True,
            status_code=200,
            duration_ms=500.0,
        )
        log.add(entry)

        log.update_usage(entry.id, 200, 100, 300)

        d = log.get_entry(entry.id)
        assert d is not None
        assert d["input_tokens"] == 200
        assert d["output_tokens"] == 100
        assert d["total_tokens"] == 300

    def test_update_usage_nonexistent_id(self):
        log = RequestLog(persistence=None)
        log.update_usage("nonexistent", 100, 50, 150)


class TestMetricsCollectorTokenTracking:
    def test_record_request_with_tokens(self):
        m = MetricsCollector()
        m.record_request(
            model="gpt-4o",
            source="openai_chat",
            target="anthropic",
            status_code=200,
            duration_ms=150.0,
            is_stream=False,
            input_tokens=100,
            output_tokens=50,
        )
        assert m.total_input_tokens == 100
        assert m.total_output_tokens == 50
        assert m.by_model_tokens["gpt-4o"]["input_tokens"] == 100
        assert m.by_model_tokens["gpt-4o"]["output_tokens"] == 50

    def test_record_request_without_tokens(self):
        m = MetricsCollector()
        m.record_request(
            model="gpt-4o",
            source="openai_chat",
            target="anthropic",
            status_code=200,
            duration_ms=150.0,
            is_stream=False,
        )
        assert m.total_input_tokens == 0
        assert m.total_output_tokens == 0
        assert m.by_model_tokens == {}

    def test_token_accumulation(self):
        m = MetricsCollector()
        m.record_request(
            model="gpt-4o",
            source="openai_chat",
            target="anthropic",
            status_code=200,
            duration_ms=100.0,
            is_stream=False,
            input_tokens=100,
            output_tokens=50,
        )
        m.record_request(
            model="gpt-4o",
            source="openai_chat",
            target="anthropic",
            status_code=200,
            duration_ms=200.0,
            is_stream=True,
            input_tokens=200,
            output_tokens=100,
        )
        assert m.total_input_tokens == 300
        assert m.total_output_tokens == 150
        assert m.by_model_tokens["gpt-4o"]["input_tokens"] == 300
        assert m.by_model_tokens["gpt-4o"]["output_tokens"] == 150

    def test_record_usage_separate(self):
        m = MetricsCollector()
        m.record_usage(model="gpt-4o", input_tokens=500, output_tokens=200)
        assert m.total_input_tokens == 500
        assert m.total_output_tokens == 200
        assert m.by_model_tokens["gpt-4o"]["input_tokens"] == 500
        assert m.by_model_tokens["gpt-4o"]["output_tokens"] == 200

    def test_snapshot_includes_tokens(self):
        m = MetricsCollector()
        m.record_request(
            model="gpt-4o",
            source="openai_chat",
            target="anthropic",
            status_code=200,
            duration_ms=100.0,
            is_stream=False,
            input_tokens=1000,
            output_tokens=500,
        )
        snap = m.snapshot(series_seconds=5)
        assert snap["total_input_tokens"] == 1000
        assert snap["total_output_tokens"] == 500
        assert snap["by_model_tokens"]["gpt-4o"]["input_tokens"] == 1000
        assert snap["by_model_tokens"]["gpt-4o"]["output_tokens"] == 500

    def test_export_load_preserves_tokens(self):
        m = MetricsCollector()
        m.record_request(
            model="gpt-4o",
            source="openai_chat",
            target="anthropic",
            status_code=200,
            duration_ms=100.0,
            is_stream=False,
            input_tokens=1000,
            output_tokens=500,
        )
        exported = m.export_counters()

        m2 = MetricsCollector()
        m2.load_counters(exported)
        assert m2.total_input_tokens == 1000
        assert m2.total_output_tokens == 500
        assert m2.by_model_tokens["gpt-4o"]["input_tokens"] == 1000

    def test_rebuild_counters_with_tokens(self):
        m = MetricsCollector()
        rows = [
            {
                "model": "gpt-4o",
                "source_provider": "openai_chat",
                "target_provider": "anthropic",
                "target_provider_name": None,
                "is_stream": False,
                "status_code": 200,
                "input_tokens": 100,
                "output_tokens": 50,
            },
            {
                "model": "gpt-4o",
                "source_provider": "openai_chat",
                "target_provider": "anthropic",
                "target_provider_name": None,
                "is_stream": True,
                "status_code": 200,
                "input_tokens": 200,
                "output_tokens": 100,
            },
        ]
        count = m.rebuild_counters(rows)
        assert count == 2
        assert m.total_input_tokens == 300
        assert m.total_output_tokens == 150
        assert m.by_model_tokens["gpt-4o"]["input_tokens"] == 300


class TestPersistenceTokenColumns:
    @pytest.fixture()
    def pm(self, tmp_path):
        from llm_rosetta.observability.persistence import PersistenceManager

        return PersistenceManager(data_dir=tmp_path)

    def test_insert_and_query_with_tokens(self, pm):
        entry = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=False,
            status_code=200,
            duration_ms=150.0,
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
        )
        pm.insert_log_entries([entry.to_dict()])

        entries, total = pm.query_log_entries(limit=10)
        assert total == 1
        e = entries[0]
        assert e["input_tokens"] == 100
        assert e["output_tokens"] == 50
        assert e["total_tokens"] == 150

    def test_insert_without_tokens(self, pm):
        entry = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=False,
            status_code=200,
            duration_ms=150.0,
        )
        pm.insert_log_entries([entry.to_dict()])

        entries, total = pm.query_log_entries(limit=10)
        assert total == 1
        e = entries[0]
        assert "input_tokens" not in e
        assert "output_tokens" not in e
        assert "total_tokens" not in e

    def test_update_entry_usage(self, pm):
        entry = RequestLogEntry.create(
            model="gpt-4o",
            source_provider="openai_chat",
            target_provider="anthropic",
            is_stream=True,
            status_code=200,
            duration_ms=500.0,
        )
        pm.insert_log_entries([entry.to_dict()])

        pm.update_entry_usage(entry.id, 200, 100, 300)

        e = pm.get_log_entry(entry.id)
        assert e is not None
        assert e["input_tokens"] == 200
        assert e["output_tokens"] == 100
        assert e["total_tokens"] == 300

    def test_migration_adds_token_columns(self, tmp_path):
        """Simulate an old DB without token columns and verify migration."""
        import sqlite3

        db_path = tmp_path / "gateway.db"
        conn = sqlite3.connect(str(db_path))
        conn.executescript("""
            CREATE TABLE request_log (
                id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                model TEXT NOT NULL,
                source_provider TEXT NOT NULL,
                target_provider TEXT NOT NULL,
                is_stream INTEGER NOT NULL,
                status_code INTEGER NOT NULL,
                duration_ms REAL NOT NULL,
                error_detail TEXT,
                api_key_label TEXT,
                target_provider_name TEXT,
                client_ip TEXT,
                profile TEXT
            );
            CREATE TABLE metrics (key TEXT PRIMARY KEY, value TEXT NOT NULL);
        """)
        conn.close()

        from llm_rosetta.observability.persistence import PersistenceManager

        pm = PersistenceManager(data_dir=tmp_path)

        cursor = pm._conn.execute("PRAGMA table_info(request_log)")
        columns = {row[1] for row in cursor.fetchall()}
        assert "input_tokens" in columns
        assert "output_tokens" in columns
        assert "total_tokens" in columns


class TestStreamProcessorUsageTracking:
    def test_get_accumulated_usage_from_ir_event(self):
        from unittest.mock import MagicMock

        from llm_rosetta.pipeline import StreamProcessor

        target_conv = MagicMock()
        source_conv = MagicMock()
        from_ctx = MagicMock()
        to_ctx = MagicMock()

        processor = StreamProcessor(
            target_converter=target_conv,
            source_converter=source_conv,
            from_ctx=from_ctx,
            to_ctx=to_ctx,
        )

        assert processor.get_accumulated_usage() is None

        target_conv.stream_response_from_provider.return_value = [
            {
                "type": "usage",
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            }
        ]
        source_conv.stream_response_to_provider.return_value = []
        from_ctx.metadata = {}
        to_ctx.metadata = {}

        processor.process_chunk({"type": "test"})

        usage = processor.get_accumulated_usage()
        assert usage is not None
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150


class TestPassthroughStreamProcessorUsage:
    def test_extract_openai_usage(self):
        from llm_rosetta.pipeline import PassthroughStreamProcessor

        p = PassthroughStreamProcessor()

        p.process_chunk(
            {
                "choices": [{"delta": {}, "finish_reason": "stop"}],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                },
            }
        )

        usage = p.get_accumulated_usage()
        assert usage is not None
        assert usage["prompt_tokens"] == 100
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 150

    def test_extract_anthropic_usage(self):
        from llm_rosetta.pipeline import PassthroughStreamProcessor

        p = PassthroughStreamProcessor()
        p.process_chunk(
            {
                "type": "message_delta",
                "usage": {"input_tokens": 200, "output_tokens": 80},
            }
        )

        usage = p.get_accumulated_usage()
        assert usage is not None
        assert usage["prompt_tokens"] == 200
        assert usage["completion_tokens"] == 80

    def test_extract_google_usage(self):
        from llm_rosetta.pipeline import PassthroughStreamProcessor

        p = PassthroughStreamProcessor()
        p.process_chunk(
            {
                "usageMetadata": {
                    "promptTokenCount": 300,
                    "candidatesTokenCount": 120,
                    "totalTokenCount": 420,
                }
            }
        )

        usage = p.get_accumulated_usage()
        assert usage is not None
        assert usage["prompt_tokens"] == 300
        assert usage["completion_tokens"] == 120
        assert usage["total_tokens"] == 420

    def test_no_usage_returns_none(self):
        from llm_rosetta.pipeline import PassthroughStreamProcessor

        p = PassthroughStreamProcessor()
        p.process_chunk({"choices": [{"delta": {"content": "hi"}}]})

        assert p.get_accumulated_usage() is None


class TestPassthroughZeroTokens:
    def test_zero_prompt_tokens_preserved(self):
        from llm_rosetta.pipeline import PassthroughStreamProcessor

        p = PassthroughStreamProcessor()
        p.process_chunk(
            {
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 50,
                    "total_tokens": 50,
                },
            }
        )
        usage = p.get_accumulated_usage()
        assert usage is not None
        assert usage["prompt_tokens"] == 0
        assert usage["completion_tokens"] == 50
        assert usage["total_tokens"] == 50


# ---- Provider-level token tracking (multi-provider support) ----


class TestProviderTokenTracking:
    """by_provider_tokens aggregation in MetricsCollector."""

    def test_record_request_tracks_provider_tokens(self):
        m = MetricsCollector()
        m.record_request(
            model="gpt-4o",
            source="openai_chat",
            target="openai_chat",
            status_code=200,
            duration_ms=100,
            is_stream=False,
            provider_name="openai_a",
            input_tokens=100,
            output_tokens=50,
        )
        assert m.by_provider_tokens["openai_a"]["input_tokens"] == 100
        assert m.by_provider_tokens["openai_a"]["output_tokens"] == 50

    def test_record_request_without_provider_name(self):
        m = MetricsCollector()
        m.record_request(
            model="gpt-4o",
            source="openai_chat",
            target="openai_chat",
            status_code=200,
            duration_ms=100,
            is_stream=False,
            input_tokens=100,
            output_tokens=50,
        )
        assert m.by_provider_tokens == {}

    def test_provider_token_accumulation(self):
        m = MetricsCollector()
        for _ in range(3):
            m.record_request(
                model="gpt-4o",
                source="openai_chat",
                target="openai_chat",
                status_code=200,
                duration_ms=100,
                is_stream=False,
                provider_name="openai_a",
                input_tokens=100,
                output_tokens=50,
            )
        assert m.by_provider_tokens["openai_a"]["input_tokens"] == 300
        assert m.by_provider_tokens["openai_a"]["output_tokens"] == 150

    def test_multi_provider_separation(self):
        m = MetricsCollector()
        m.record_request(
            model="gpt-4o",
            source="openai_chat",
            target="openai_chat",
            status_code=200,
            duration_ms=100,
            is_stream=False,
            provider_name="openai_a",
            input_tokens=100,
            output_tokens=50,
        )
        m.record_request(
            model="gpt-4o",
            source="openai_chat",
            target="openai_chat",
            status_code=200,
            duration_ms=100,
            is_stream=False,
            provider_name="openai_b",
            input_tokens=200,
            output_tokens=100,
        )
        assert m.by_provider_tokens["openai_a"]["input_tokens"] == 100
        assert m.by_provider_tokens["openai_b"]["input_tokens"] == 200

    def test_record_usage_with_provider_name(self):
        m = MetricsCollector()
        m.record_usage(
            model="gpt-4o",
            input_tokens=500,
            output_tokens=200,
            provider_name="openai_a",
        )
        assert m.by_provider_tokens["openai_a"]["input_tokens"] == 500
        assert m.by_provider_tokens["openai_a"]["output_tokens"] == 200

    def test_record_usage_without_provider_name(self):
        m = MetricsCollector()
        m.record_usage(model="gpt-4o", input_tokens=500, output_tokens=200)
        assert m.by_provider_tokens == {}

    def test_snapshot_includes_provider_tokens(self):
        m = MetricsCollector()
        m.record_request(
            model="gpt-4o",
            source="openai_chat",
            target="openai_chat",
            status_code=200,
            duration_ms=100,
            is_stream=False,
            provider_name="openai_a",
            input_tokens=1000,
            output_tokens=500,
        )
        snap = m.snapshot()
        assert snap["by_provider_tokens"]["openai_a"]["input_tokens"] == 1000

    def test_export_load_preserves_provider_tokens(self):
        m = MetricsCollector()
        m.record_request(
            model="gpt-4o",
            source="openai_chat",
            target="openai_chat",
            status_code=200,
            duration_ms=100,
            is_stream=False,
            provider_name="openai_a",
            input_tokens=1000,
            output_tokens=500,
        )
        exported = m.export_counters()
        m2 = MetricsCollector()
        m2.load_counters(exported)
        assert m2.by_provider_tokens["openai_a"]["input_tokens"] == 1000

    def test_rebuild_counters_with_provider_tokens(self):
        m = MetricsCollector()
        rows = [
            {
                "model": "gpt-4o",
                "source_provider": "openai_chat",
                "target_provider": "openai_chat",
                "target_provider_name": "openai_a",
                "is_stream": False,
                "status_code": 200,
                "duration_ms": 100,
                "input_tokens": 100,
                "output_tokens": 50,
            },
            {
                "model": "gpt-4o",
                "source_provider": "openai_chat",
                "target_provider": "openai_chat",
                "target_provider_name": "openai_b",
                "is_stream": False,
                "status_code": 200,
                "duration_ms": 100,
                "input_tokens": 200,
                "output_tokens": 100,
            },
        ]
        m.rebuild_counters(rows)
        assert m.by_provider_tokens["openai_a"]["input_tokens"] == 100
        assert m.by_provider_tokens["openai_b"]["input_tokens"] == 200
