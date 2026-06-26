"""Tests for the observability MetricsCollector (standalone, no gateway)."""

from llm_rosetta.observability import MetricsCollector
from llm_rosetta.observability.metrics import _RollingWindow


class TestRollingWindow:
    def test_record_and_get_series(self):
        w = _RollingWindow(window_seconds=300)
        w.record(100.0, is_error=False)
        w.record(200.0, is_error=True)
        series = w.get_series(seconds=5)
        assert len(series) == 5
        last = series[-1]
        assert last["count"] == 2
        assert last["errors"] == 1
        assert last["avg_ms"] == 150.0

    def test_empty_series(self):
        w = _RollingWindow()
        series = w.get_series(seconds=10)
        assert len(series) == 10
        assert all(s["count"] == 0 for s in series)


class TestMetricsCollector:
    def test_record_and_snapshot(self):
        m = MetricsCollector()
        m.record_request(
            model="gpt-4o",
            source="openai_chat",
            target="anthropic",
            status_code=200,
            duration_ms=150.0,
            is_stream=False,
        )
        snap = m.snapshot(series_seconds=5)
        assert snap["total_requests"] == 1
        assert snap["total_errors"] == 0

    def test_export_load_roundtrip(self):
        m = MetricsCollector()
        m.record_request(
            model="gpt-4o",
            source="openai_chat",
            target="anthropic",
            status_code=200,
            duration_ms=50.0,
            is_stream=True,
        )
        exported = m.export_counters()

        m2 = MetricsCollector()
        m2.load_counters(exported)
        assert m2.total_requests == 1
        assert m2.total_streams == 1

    def test_provider_health(self):
        m = MetricsCollector()
        for _ in range(15):
            m.record_request(
                model="gpt-4o",
                source="openai_chat",
                target="anthropic",
                status_code=500,
                duration_ms=100.0,
                is_stream=False,
                provider_name="test-provider",
                error_detail="fail",
            )
        assert m.any_critical_provider()
        health = m.provider_health_snapshot()
        assert health["test-provider"]["status"] == "critical"

    def test_rebuild_counters(self):
        m = MetricsCollector()
        rows = [
            {
                "model": "gpt-4o",
                "source_provider": "openai_chat",
                "target_provider": "anthropic",
                "target_provider_name": "My Anthropic",
                "is_stream": False,
                "status_code": 200,
            },
            {
                "model": "gpt-4o",
                "source_provider": "openai_chat",
                "target_provider": "anthropic",
                "target_provider_name": "My Anthropic",
                "is_stream": True,
                "status_code": 500,
            },
        ]
        count = m.rebuild_counters(rows)
        assert count == 2
        assert m.total_requests == 2
        assert m.total_errors == 1
        assert m.total_streams == 1
