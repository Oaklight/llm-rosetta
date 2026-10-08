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

    def test_record_disconnect(self):
        m = MetricsCollector()
        assert m.total_client_disconnects == 0
        m.record_disconnect()
        m.record_disconnect()
        assert m.total_client_disconnects == 2

    def test_disconnect_in_snapshot(self):
        m = MetricsCollector()
        m.record_disconnect()
        snap = m.snapshot(series_seconds=1)
        assert snap["total_client_disconnects"] == 1

    def test_disconnect_export_load_roundtrip(self):
        m = MetricsCollector()
        m.record_disconnect()
        m.record_disconnect()
        m.record_disconnect()
        exported = m.export_counters()
        assert exported["total_client_disconnects"] == 3

        m2 = MetricsCollector()
        m2.load_counters(exported)
        assert m2.total_client_disconnects == 3

    def test_disconnect_not_rebuilt_from_rows(self):
        m = MetricsCollector()
        m.record_disconnect()
        m.rebuild_counters([])
        assert m.total_client_disconnects == 0


class TestLifetimeCounters:
    def _record(self, m, status=200):
        m.record_request(
            model="m",
            source="s",
            target="t",
            status_code=status,
            duration_ms=1.0,
            is_stream=False,
            provider_name="p",
        )

    def test_lifetime_increments_with_total(self):
        m = MetricsCollector()
        self._record(m)
        self._record(m, status=500)
        assert m.total_requests == 2
        assert m.lifetime_total_requests == 2
        assert m.total_errors == 1
        assert m.lifetime_total_errors == 1

    def test_rebuild_does_not_reset_lifetime(self):
        m = MetricsCollector()
        for _ in range(10):
            self._record(m)
        self._record(m, status=500)
        assert m.lifetime_total_requests == 11
        assert m.lifetime_total_errors == 1

        # Rebuild with only 5 rows (simulating prune)
        rows = [
            {
                "model": "m",
                "source_provider": "s",
                "target_provider": "t",
                "is_stream": False,
                "status_code": 200,
            }
            for _ in range(5)
        ]
        m.rebuild_counters(iter(rows))
        assert m.total_requests == 5  # reset to log count
        assert m.lifetime_total_requests == 11  # preserved

    def test_export_load_roundtrip(self):
        m = MetricsCollector()
        for _ in range(5):
            self._record(m)
        self._record(m, status=500)
        exported = m.export_counters()
        assert exported["lifetime_total_requests"] == 6
        assert exported["lifetime_total_errors"] == 1

        m2 = MetricsCollector()
        m2.load_counters(exported)
        assert m2.lifetime_total_requests == 6
        assert m2.lifetime_total_errors == 1

    def test_load_old_data_without_lifetime(self):
        """Loading data from before lifetime counters bootstraps from totals."""
        m = MetricsCollector()
        m.load_counters({"total_requests": 100, "total_errors": 10})
        assert m.lifetime_total_requests == 100
        assert m.lifetime_total_errors == 10

    def test_merge_rebuild_preserves_lifetime(self):
        m = MetricsCollector()
        for _ in range(20):
            self._record(m)
        pre = m.export_counters()
        # Simulate background rebuild with fewer rows
        baseline = {"total_requests": 10, "lifetime_total_requests": 20}
        m.merge_rebuild(baseline, pre)
        assert m.lifetime_total_requests == 20

    def test_snapshot_includes_lifetime(self):
        m = MetricsCollector()
        self._record(m)
        snap = m.snapshot()
        assert "lifetime_total_requests" in snap
        assert "lifetime_total_errors" in snap
        assert snap["lifetime_total_requests"] == 1
