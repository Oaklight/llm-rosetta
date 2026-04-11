"""Tests for the admin panel MetricsCollector."""

from llm_rosetta.gateway.admin.metrics import MetricsCollector, _RollingWindow


class TestRollingWindow:
    def test_record_and_get_series(self):
        w = _RollingWindow(window_seconds=300)
        w.record(100.0, is_error=False)
        w.record(200.0, is_error=True)
        series = w.get_series(seconds=5)
        assert len(series) == 5
        # The last bucket should have our data
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
    def test_record_request(self):
        m = MetricsCollector()
        m.record_request(
            model="gpt-4o",
            source="openai_chat",
            target="anthropic",
            status_code=200,
            duration_ms=150.0,
            is_stream=False,
        )
        assert m.total_requests == 1
        assert m.total_errors == 0
        assert m.total_streams == 0
        assert m.by_model["gpt-4o"] == 1
        assert m.by_source_provider["openai_chat"] == 1
        assert m.by_target_provider["anthropic"] == 1
        assert m.by_status_code[200] == 1

    def test_record_error(self):
        m = MetricsCollector()
        m.record_request(
            model="gpt-4o",
            source="openai_chat",
            target="anthropic",
            status_code=500,
            duration_ms=50.0,
            is_stream=True,
        )
        assert m.total_requests == 1
        assert m.total_errors == 1
        assert m.total_streams == 1

    def test_snapshot(self):
        m = MetricsCollector()
        m.record_request(
            model="gpt-4o",
            source="openai_chat",
            target="anthropic",
            status_code=200,
            duration_ms=100.0,
            is_stream=False,
        )
        snap = m.snapshot(series_seconds=5)
        assert snap["total_requests"] == 1
        assert snap["error_rate"] == 0
        assert "series" in snap
        assert len(snap["series"]) == 5
        assert snap["uptime_seconds"] >= 0
        assert snap["by_model"] == {"gpt-4o": 1}

    def test_active_streams_gauge(self):
        m = MetricsCollector()
        m.active_streams += 1
        assert m.active_streams == 1
        m.active_streams -= 1
        assert m.active_streams == 0

    def test_multiple_models(self):
        m = MetricsCollector()
        for _ in range(3):
            m.record_request(
                model="gpt-4o",
                source="openai_chat",
                target="openai_chat",
                status_code=200,
                duration_ms=10.0,
                is_stream=False,
            )
        for _ in range(2):
            m.record_request(
                model="claude",
                source="anthropic",
                target="anthropic",
                status_code=200,
                duration_ms=20.0,
                is_stream=True,
            )
        assert m.total_requests == 5
        assert m.by_model["gpt-4o"] == 3
        assert m.by_model["claude"] == 2
        assert m.total_streams == 2
