"""Tests for the enhanced /health, /health/live, and /health/ready endpoints."""

from __future__ import annotations

import asyncio

from llm_rosetta.gateway.admin.metrics import MetricsCollector, _ProviderStats


# ---------------------------------------------------------------------------
# Unit tests for _ProviderStats
# ---------------------------------------------------------------------------


class TestProviderStats:
    def test_empty_success_rate(self):
        s = _ProviderStats()
        assert s.success_rate == 1.0
        assert s.avg_latency_ms == 0.0
        assert s.sample_size == 0
        assert not s.is_critical()

    def test_all_success(self):
        s = _ProviderStats()
        for _ in range(20):
            s.record(100.0, is_error=False)
        assert s.success_rate == 1.0
        assert s.avg_latency_ms == 100.0
        assert not s.is_critical()

    def test_critical_threshold(self):
        s = _ProviderStats()
        # 10 samples, 6 errors → success_rate = 0.4 < 0.5 threshold
        for _ in range(10):
            s.record(50.0, is_error=True)
        assert s.is_critical()

    def test_not_critical_below_min_sample(self):
        s = _ProviderStats()
        # Only 5 samples (< 10 minimum), even all errors should not be critical
        for _ in range(5):
            s.record(50.0, is_error=True)
        assert not s.is_critical()

    def test_last_error_captured(self):
        s = _ProviderStats()
        s.record(100.0, is_error=True, error_detail="timeout")
        s.record(100.0, is_error=True, error_detail="connection refused")
        assert s.last_error == "connection refused"

    def test_last_error_not_overwritten_by_success(self):
        s = _ProviderStats()
        s.record(100.0, is_error=True, error_detail="timeout")
        s.record(100.0, is_error=False)  # success should not clear last_error
        assert s.last_error == "timeout"

    def test_window_circular_buffer(self):
        """Buffer should cap at window_size and drop oldest entries."""
        s = _ProviderStats(window_size=5)
        # Fill with successes
        for _ in range(5):
            s.record(10.0, is_error=False)
        assert s.sample_size == 5
        assert s.success_rate == 1.0

        # Now add errors; oldest successes get evicted
        for _ in range(5):
            s.record(10.0, is_error=True)
        # All 5 slots are now errors
        assert s.sample_size == 5
        assert s.success_rate == 0.0


# ---------------------------------------------------------------------------
# Unit tests for MetricsCollector per-provider tracking
# ---------------------------------------------------------------------------


class TestMetricsCollectorPerProvider:
    def test_provider_stats_populated(self):
        m = MetricsCollector()
        m.record_request(
            model="gpt-4o",
            source="openai_chat",
            target="anthropic",
            status_code=200,
            duration_ms=123.0,
            is_stream=False,
            provider_name="myargo",
        )
        health = m.provider_health_snapshot()
        assert "myargo" in health
        assert health["myargo"]["status"] == "ok"
        assert health["myargo"]["success_rate"] == 1.0
        assert health["myargo"]["avg_latency_ms"] == 123.0
        assert health["myargo"]["sample_size"] == 1
        assert health["myargo"]["last_error"] is None

    def test_provider_stats_falls_back_to_target(self):
        """When provider_name is omitted, 'target' is used as key."""
        m = MetricsCollector()
        m.record_request(
            model="m",
            source="openai_chat",
            target="anthropic",
            status_code=200,
            duration_ms=50.0,
            is_stream=False,
        )
        health = m.provider_health_snapshot()
        assert "anthropic" in health

    def test_any_critical_provider_false_when_healthy(self):
        m = MetricsCollector()
        for _ in range(20):
            m.record_request(
                model="m",
                source="openai_chat",
                target="openai_chat",
                status_code=200,
                duration_ms=50.0,
                is_stream=False,
                provider_name="good_provider",
            )
        assert not m.any_critical_provider()

    def test_any_critical_provider_true_when_unhealthy(self):
        m = MetricsCollector()
        for _ in range(10):
            m.record_request(
                model="m",
                source="openai_chat",
                target="openai_chat",
                status_code=500,
                duration_ms=50.0,
                is_stream=False,
                provider_name="bad_provider",
                error_detail="server error",
            )
        assert m.any_critical_provider()

    def test_snapshot_includes_providers(self):
        m = MetricsCollector()
        m.record_request(
            model="gpt-4o",
            source="openai_chat",
            target="anthropic",
            status_code=200,
            duration_ms=200.0,
            is_stream=False,
            provider_name="myargo",
        )
        snap = m.snapshot()
        assert "providers" in snap
        assert "myargo" in snap["providers"]

    def test_error_detail_recorded_in_provider_stats(self):
        m = MetricsCollector()
        m.record_request(
            model="m",
            source="openai_chat",
            target="openai_chat",
            status_code=503,
            duration_ms=10.0,
            is_stream=False,
            provider_name="flaky",
            error_detail="upstream timeout",
        )
        health = m.provider_health_snapshot()
        assert health["flaky"]["last_error"] == "upstream timeout"


# ---------------------------------------------------------------------------
# Functional tests for health handler (unit-level, no HTTP server)
# ---------------------------------------------------------------------------


class _FakeApp:
    """Minimal stand-in for the App object."""

    def __init__(self, metrics: MetricsCollector | None = None):
        self.metrics = metrics


class _FakeRequest:
    def __init__(self, metrics: MetricsCollector | None = None):
        self.app = _FakeApp(metrics=metrics)


def _run(coro):
    """Run an async coroutine synchronously (no pytest-asyncio needed)."""
    return asyncio.run(coro)


class TestHandleHealthFunction:
    """Test handle_health / handle_health_live / handle_health_ready directly."""

    def setup_method(self):
        from llm_rosetta.gateway.app import (
            handle_health,
            handle_health_live,
            handle_health_ready,
        )

        self.handle_health = handle_health
        self.handle_health_live = handle_health_live
        self.handle_health_ready = handle_health_ready

    def test_health_no_metrics(self):
        import json

        req = _FakeRequest(metrics=None)
        resp = _run(self.handle_health(req))
        body = json.loads(resp.body)
        assert body["status"] == "ok"
        assert resp.status_code == 200

    def test_health_with_healthy_metrics(self):
        import json

        m = MetricsCollector()
        for _ in range(5):
            m.record_request(
                model="gpt-4o",
                source="openai_chat",
                target="anthropic",
                status_code=200,
                duration_ms=150.0,
                is_stream=False,
                provider_name="myargo",
            )
        req = _FakeRequest(metrics=m)
        resp = _run(self.handle_health(req))
        body = json.loads(resp.body)
        assert body["status"] == "ok"
        assert resp.status_code == 200
        assert body["requests_total"] == 5
        assert "uptime_seconds" in body
        assert "errors_last_hour" in body
        assert "providers" in body
        assert "myargo" in body["providers"]

    def test_health_returns_200_degraded_for_critical_provider(self):
        import json

        m = MetricsCollector()
        # Push 10 errors to trigger critical threshold
        for _ in range(10):
            m.record_request(
                model="m",
                source="openai_chat",
                target="openai_chat",
                status_code=500,
                duration_ms=50.0,
                is_stream=False,
                provider_name="bad_provider",
                error_detail="error",
            )
        req = _FakeRequest(metrics=m)
        resp = _run(self.handle_health(req))
        body = json.loads(resp.body)
        assert resp.status_code == 200  # /health always 200; use /health/ready for 503
        assert body["status"] == "degraded"

    def test_health_live_always_200(self):
        import json

        req = _FakeRequest(metrics=None)
        resp = _run(self.handle_health_live(req))
        body = json.loads(resp.body)
        assert resp.status_code == 200
        assert body["status"] == "ok"

    def test_health_ready_200_when_healthy(self):
        import json

        m = MetricsCollector()
        req = _FakeRequest(metrics=m)
        resp = _run(self.handle_health_ready(req))
        body = json.loads(resp.body)
        assert resp.status_code == 200
        assert body["status"] == "ready"

    def test_health_ready_503_when_critical(self):
        import json

        m = MetricsCollector()
        for _ in range(10):
            m.record_request(
                model="m",
                source="openai_chat",
                target="openai_chat",
                status_code=500,
                duration_ms=50.0,
                is_stream=False,
                provider_name="bad_provider",
                error_detail="error",
            )
        req = _FakeRequest(metrics=m)
        resp = _run(self.handle_health_ready(req))
        body = json.loads(resp.body)
        assert resp.status_code == 503
        assert body["status"] == "not_ready"
        assert "providers" in body
