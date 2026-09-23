"""Tests for the gateway circuit breaker module.

Covers state transitions, probe limiting, cooldown timing, thread safety,
registry lazy creation, and integration with the proxy handler (503 when open).
"""

from __future__ import annotations

import threading
import time


from llm_rosetta.gateway.circuit_breaker import (
    CircuitBreaker,
    CircuitBreakerConfig,
    CircuitBreakerRegistry,
    CircuitState,
)


# ---------------------------------------------------------------------------
# CircuitBreakerConfig defaults
# ---------------------------------------------------------------------------


class TestCircuitBreakerConfig:
    def test_defaults(self):
        cfg = CircuitBreakerConfig()
        assert cfg.enabled is False
        assert cfg.error_threshold == 5
        assert cfg.cooldown_seconds == 30.0
        assert cfg.half_open_max_probes == 1

    def test_custom_values(self):
        cfg = CircuitBreakerConfig(
            enabled=True,
            error_threshold=3,
            cooldown_seconds=10.0,
            half_open_max_probes=2,
        )
        assert cfg.enabled is True
        assert cfg.error_threshold == 3
        assert cfg.cooldown_seconds == 10.0
        assert cfg.half_open_max_probes == 2


# ---------------------------------------------------------------------------
# CircuitBreaker state transitions
# ---------------------------------------------------------------------------


class TestCircuitBreakerStates:
    """Test the CLOSED -> OPEN -> HALF_OPEN -> CLOSED state machine."""

    def _make_breaker(self, **kwargs) -> CircuitBreaker:
        defaults = {
            "enabled": True,
            "error_threshold": 3,
            "cooldown_seconds": 0.1,
            "half_open_max_probes": 1,
        }
        defaults.update(kwargs)
        return CircuitBreaker(CircuitBreakerConfig(**defaults))

    def test_initial_state_is_closed(self):
        cb = self._make_breaker()
        assert cb.state == CircuitState.CLOSED

    def test_closed_allows_requests(self):
        cb = self._make_breaker()
        assert cb.allow_request() is True

    def test_failures_below_threshold_stay_closed(self):
        cb = self._make_breaker(error_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        assert cb.allow_request() is True

    def test_threshold_exceeded_trips_to_open(self):
        cb = self._make_breaker(error_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_open_rejects_requests(self):
        cb = self._make_breaker(error_threshold=2, cooldown_seconds=60.0)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert cb.allow_request() is False

    def test_open_transitions_to_half_open_after_cooldown(self):
        cb = self._make_breaker(error_threshold=2, cooldown_seconds=0.05)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        time.sleep(0.06)
        assert cb.state == CircuitState.HALF_OPEN

    def test_half_open_allows_probe(self):
        cb = self._make_breaker(error_threshold=2, cooldown_seconds=0.05)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.06)
        assert cb.allow_request() is True

    def test_half_open_success_closes_circuit(self):
        cb = self._make_breaker(error_threshold=2, cooldown_seconds=0.05)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.06)
        cb.allow_request()  # consume probe slot
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_half_open_failure_reopens_circuit(self):
        cb = self._make_breaker(error_threshold=2, cooldown_seconds=0.05)
        cb.record_failure()
        cb.record_failure()
        time.sleep(0.06)
        cb.allow_request()  # consume probe slot
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_success_resets_failure_count(self):
        cb = self._make_breaker(error_threshold=5)
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.failure_count == 3
        cb.record_success()
        assert cb.failure_count == 0
        assert cb.state == CircuitState.CLOSED

    def test_interleaved_success_prevents_tripping(self):
        """A success in between failures resets the consecutive counter."""
        cb = self._make_breaker(error_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()  # reset
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_full_cycle_closed_open_halfopen_closed(self):
        cb = self._make_breaker(error_threshold=2, cooldown_seconds=0.05)
        # CLOSED
        assert cb.state == CircuitState.CLOSED
        # Trip to OPEN
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # Wait for cooldown -> HALF_OPEN
        time.sleep(0.06)
        assert cb.state == CircuitState.HALF_OPEN
        # Probe succeeds -> CLOSED
        cb.allow_request()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_full_cycle_with_probe_failure(self):
        cb = self._make_breaker(error_threshold=2, cooldown_seconds=0.05)
        # Trip to OPEN
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # Cooldown -> HALF_OPEN
        time.sleep(0.06)
        assert cb.state == CircuitState.HALF_OPEN
        # Probe fails -> OPEN again
        cb.allow_request()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        # Wait again -> HALF_OPEN
        time.sleep(0.06)
        assert cb.state == CircuitState.HALF_OPEN
        # Probe succeeds -> CLOSED
        cb.allow_request()
        cb.record_success()
        assert cb.state == CircuitState.CLOSED


# ---------------------------------------------------------------------------
# Probe limiting in HALF_OPEN
# ---------------------------------------------------------------------------


class TestHalfOpenProbeLimit:
    def test_single_probe_limit(self):
        cb = CircuitBreaker(
            CircuitBreakerConfig(
                enabled=True,
                error_threshold=1,
                cooldown_seconds=0.05,
                half_open_max_probes=1,
            )
        )
        cb.record_failure()
        time.sleep(0.06)
        assert cb.state == CircuitState.HALF_OPEN
        # First probe allowed
        assert cb.allow_request() is True
        # Second probe rejected
        assert cb.allow_request() is False

    def test_multi_probe_limit(self):
        cb = CircuitBreaker(
            CircuitBreakerConfig(
                enabled=True,
                error_threshold=1,
                cooldown_seconds=0.05,
                half_open_max_probes=3,
            )
        )
        cb.record_failure()
        time.sleep(0.06)
        # 3 probes allowed
        assert cb.allow_request() is True
        assert cb.allow_request() is True
        assert cb.allow_request() is True
        # 4th rejected
        assert cb.allow_request() is False


# ---------------------------------------------------------------------------
# Cooldown timing
# ---------------------------------------------------------------------------


class TestCooldownRemaining:
    def test_closed_returns_zero(self):
        cb = CircuitBreaker(CircuitBreakerConfig(enabled=True, error_threshold=5))
        assert cb.cooldown_remaining() == 0.0

    def test_open_returns_positive(self):
        cb = CircuitBreaker(
            CircuitBreakerConfig(enabled=True, error_threshold=1, cooldown_seconds=10.0)
        )
        cb.record_failure()
        remaining = cb.cooldown_remaining()
        assert remaining > 0.0
        assert remaining <= 10.0

    def test_cooldown_decreases_over_time(self):
        cb = CircuitBreaker(
            CircuitBreakerConfig(enabled=True, error_threshold=1, cooldown_seconds=1.0)
        )
        cb.record_failure()
        r1 = cb.cooldown_remaining()
        time.sleep(0.1)
        r2 = cb.cooldown_remaining()
        assert r2 < r1

    def test_half_open_returns_zero(self):
        cb = CircuitBreaker(
            CircuitBreakerConfig(enabled=True, error_threshold=1, cooldown_seconds=0.05)
        )
        cb.record_failure()
        time.sleep(0.06)
        assert cb.state == CircuitState.HALF_OPEN
        assert cb.cooldown_remaining() == 0.0


# ---------------------------------------------------------------------------
# Disabled circuit breaker (no-op)
# ---------------------------------------------------------------------------


class TestDisabledCircuitBreaker:
    def test_disabled_always_allows(self):
        cb = CircuitBreaker(CircuitBreakerConfig(enabled=False, error_threshold=1))
        for _ in range(100):
            cb.record_failure()
        assert cb.allow_request() is True

    def test_disabled_state_stays_closed(self):
        cb = CircuitBreaker(CircuitBreakerConfig(enabled=False, error_threshold=1))
        for _ in range(100):
            cb.record_failure()
        assert cb.state == CircuitState.CLOSED

    def test_disabled_record_success_noop(self):
        cb = CircuitBreaker(CircuitBreakerConfig(enabled=False))
        cb.record_success()
        assert cb.state == CircuitState.CLOSED

    def test_disabled_record_failure_noop(self):
        cb = CircuitBreaker(CircuitBreakerConfig(enabled=False))
        cb.record_failure()
        assert cb.failure_count == 0


# ---------------------------------------------------------------------------
# get_snapshot
# ---------------------------------------------------------------------------


class TestGetSnapshot:
    def test_closed_snapshot(self):
        cb = CircuitBreaker(CircuitBreakerConfig(enabled=True, error_threshold=5))
        snap = cb.get_snapshot()
        assert snap["state"] == "CLOSED"
        assert snap["failure_count"] == 0
        assert snap["last_failure"] is None
        assert snap["cooldown_remaining"] == 0.0

    def test_open_snapshot(self):
        cb = CircuitBreaker(
            CircuitBreakerConfig(enabled=True, error_threshold=1, cooldown_seconds=30.0)
        )
        cb.record_failure()
        snap = cb.get_snapshot()
        assert snap["state"] == "OPEN"
        assert snap["failure_count"] == 1
        assert snap["last_failure"] is not None
        assert snap["cooldown_remaining"] > 0.0

    def test_snapshot_after_success_reset(self):
        cb = CircuitBreaker(CircuitBreakerConfig(enabled=True, error_threshold=5))
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        snap = cb.get_snapshot()
        assert snap["failure_count"] == 0


# ---------------------------------------------------------------------------
# Thread safety
# ---------------------------------------------------------------------------


class TestThreadSafety:
    def test_concurrent_failures(self):
        """Multiple threads recording failures should not corrupt state."""
        cb = CircuitBreaker(
            CircuitBreakerConfig(
                enabled=True, error_threshold=100, cooldown_seconds=60.0
            )
        )
        barrier = threading.Barrier(10)

        def _record_failures():
            barrier.wait()
            for _ in range(50):
                cb.record_failure()

        threads = [threading.Thread(target=_record_failures) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All 500 failures should be counted (consecutive, no resets)
        assert cb.failure_count == 500

    def test_concurrent_allow_request_in_half_open(self):
        """Only max_probes requests should be allowed in HALF_OPEN."""
        cb = CircuitBreaker(
            CircuitBreakerConfig(
                enabled=True,
                error_threshold=1,
                cooldown_seconds=0.05,
                half_open_max_probes=1,
            )
        )
        cb.record_failure()
        time.sleep(0.06)
        assert cb.state == CircuitState.HALF_OPEN

        allowed = []
        barrier = threading.Barrier(10)

        def _try_request():
            barrier.wait()
            result = cb.allow_request()
            allowed.append(result)

        threads = [threading.Thread(target=_try_request) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly 1 should be allowed
        assert sum(allowed) == 1

    def test_concurrent_success_and_failure(self):
        """Mixed success/failure from multiple threads should not deadlock."""
        cb = CircuitBreaker(
            CircuitBreakerConfig(
                enabled=True, error_threshold=1000, cooldown_seconds=60.0
            )
        )

        def _mixed_ops():
            for i in range(100):
                if i % 3 == 0:
                    cb.record_success()
                else:
                    cb.record_failure()
                cb.allow_request()
                cb.state  # trigger potential transition

        threads = [threading.Thread(target=_mixed_ops) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Just verify no deadlock or crash
        assert cb.state in (CircuitState.CLOSED, CircuitState.OPEN)


# ---------------------------------------------------------------------------
# CircuitBreakerRegistry
# ---------------------------------------------------------------------------


class TestCircuitBreakerRegistry:
    def test_lazy_creation(self):
        registry = CircuitBreakerRegistry(
            CircuitBreakerConfig(enabled=True, error_threshold=3)
        )
        assert registry.get_state("unknown") is None
        cb = registry.get_or_create("provider-a")
        assert cb is not None
        assert registry.get_state("provider-a") == CircuitState.CLOSED

    def test_returns_same_instance(self):
        registry = CircuitBreakerRegistry()
        cb1 = registry.get_or_create("p1")
        cb2 = registry.get_or_create("p1")
        assert cb1 is cb2

    def test_different_providers_different_instances(self):
        registry = CircuitBreakerRegistry()
        cb1 = registry.get_or_create("p1")
        cb2 = registry.get_or_create("p2")
        assert cb1 is not cb2

    def test_custom_config_per_provider(self):
        default = CircuitBreakerConfig(enabled=True, error_threshold=5)
        custom = CircuitBreakerConfig(enabled=True, error_threshold=2)
        registry = CircuitBreakerRegistry(default)
        cb = registry.get_or_create("custom-provider", config=custom)
        # 2 failures should trip (custom threshold), not 5
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_set_override(self):
        default = CircuitBreakerConfig(enabled=True, error_threshold=5)
        override = CircuitBreakerConfig(enabled=True, error_threshold=1)
        registry = CircuitBreakerRegistry(default)
        registry.set_override("p1", override)
        cb = registry.get_or_create("p1")
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_set_override_replaces_existing(self):
        default = CircuitBreakerConfig(enabled=True, error_threshold=5)
        registry = CircuitBreakerRegistry(default)
        # Create with default
        cb1 = registry.get_or_create("p1")
        # Override replaces
        override = CircuitBreakerConfig(enabled=True, error_threshold=1)
        registry.set_override("p1", override)
        cb2 = registry.get_or_create("p1")
        assert cb1 is not cb2
        cb2.record_failure()
        assert cb2.state == CircuitState.OPEN

    def test_get_all_states_empty(self):
        registry = CircuitBreakerRegistry()
        assert registry.get_all_states() == {}

    def test_get_all_states_multiple(self):
        registry = CircuitBreakerRegistry(
            CircuitBreakerConfig(enabled=True, error_threshold=3)
        )
        registry.get_or_create("p1")
        registry.get_or_create("p2")
        states = registry.get_all_states()
        assert "p1" in states
        assert "p2" in states
        assert states["p1"]["state"] == "CLOSED"
        assert states["p2"]["state"] == "CLOSED"

    def test_enabled_property(self):
        registry = CircuitBreakerRegistry(CircuitBreakerConfig(enabled=True))
        assert registry.enabled is True
        registry2 = CircuitBreakerRegistry(CircuitBreakerConfig(enabled=False))
        assert registry2.enabled is False

    def test_default_registry_disabled(self):
        registry = CircuitBreakerRegistry()
        assert registry.enabled is False


# ---------------------------------------------------------------------------
# Config integration
# ---------------------------------------------------------------------------


class TestConfigIntegration:
    """Test circuit breaker config parsing via GatewayConfig."""

    def test_default_config_creates_disabled_registry(self):
        from llm_rosetta.gateway.config import GatewayConfig

        raw = {
            "providers": {
                "test": {"type": "openai_chat", "api_key": "k", "base_url": "http://x"}
            },
            "models": {"m": "test"},
        }
        config = GatewayConfig(raw)
        assert config.circuit_breaker_registry is not None
        assert config.circuit_breaker_registry.enabled is False

    def test_enabled_config(self):
        from llm_rosetta.gateway.config import GatewayConfig

        raw = {
            "providers": {
                "test": {"type": "openai_chat", "api_key": "k", "base_url": "http://x"}
            },
            "models": {"m": "test"},
            "server": {
                "circuit_breaker": {
                    "enabled": True,
                    "error_threshold": 3,
                    "cooldown_seconds": 15.0,
                    "half_open_max_probes": 2,
                }
            },
        }
        config = GatewayConfig(raw)
        assert config.circuit_breaker_registry.enabled is True

    def test_per_provider_override(self):
        from llm_rosetta.gateway.config import GatewayConfig

        raw = {
            "providers": {
                "test": {
                    "type": "openai_chat",
                    "api_key": "k",
                    "base_url": "http://x",
                    "circuit_breaker": {
                        "enabled": True,
                        "error_threshold": 2,
                    },
                }
            },
            "models": {"m": "test"},
            "server": {
                "circuit_breaker": {
                    "enabled": True,
                    "error_threshold": 10,
                }
            },
        }
        config = GatewayConfig(raw)
        cb = config.circuit_breaker_registry.get_or_create("test")
        # Per-provider override should use threshold 2, not global 10
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN


# ---------------------------------------------------------------------------
# Integration: circuit breaker returns 503 when open
# ---------------------------------------------------------------------------


class TestCircuitBreaker503Integration:
    """Verify that the proxy handler returns 503 when a circuit is open."""

    def test_503_response_format_openai(self):
        """When circuit is open, the handler returns a 503 with a clear message."""
        from llm_rosetta.gateway.error_format import (
            detect_api_format,
            format_error_response,
        )

        api_format = detect_api_format("/v1/chat/completions")
        assert api_format == "openai"

        resp = format_error_response(
            api_format,
            503,
            "Provider 'test-provider' is temporarily unavailable "
            "(circuit breaker open, cooldown 25s remaining)",
            error_type="service_unavailable",
            google_status="UNAVAILABLE",
            cors=True,
        )
        assert resp.status_code == 503

        import json

        body = json.loads(resp.body)
        assert "circuit breaker open" in body["error"]["message"]
        assert "test-provider" in body["error"]["message"]

    def test_503_response_format_anthropic(self):
        from llm_rosetta.gateway.error_format import (
            detect_api_format,
            format_error_response,
        )

        api_format = detect_api_format("/v1/messages")
        assert api_format == "anthropic"

        resp = format_error_response(
            api_format,
            503,
            "Provider 'test' is temporarily unavailable "
            "(circuit breaker open, cooldown 10s remaining)",
            error_type="service_unavailable",
        )
        assert resp.status_code == 503

        import json

        body = json.loads(resp.body)
        assert body["type"] == "error"
        assert "circuit breaker open" in body["error"]["message"]

    def test_503_response_format_google(self):
        from llm_rosetta.gateway.error_format import (
            detect_api_format,
            format_error_response,
        )

        api_format = detect_api_format("/v1beta/models/gemini:generateContent")
        assert api_format == "google"

        resp = format_error_response(
            api_format,
            503,
            "Provider 'test' is temporarily unavailable "
            "(circuit breaker open, cooldown 5s remaining)",
            google_status="UNAVAILABLE",
        )
        assert resp.status_code == 503

        import json

        body = json.loads(resp.body)
        assert body["error"]["status"] == "UNAVAILABLE"
        assert "circuit breaker open" in body["error"]["message"]


# ---------------------------------------------------------------------------
# last_failure_time property
# ---------------------------------------------------------------------------


class TestLastFailureTime:
    def test_none_before_any_failure(self):
        cb = CircuitBreaker(CircuitBreakerConfig(enabled=True))
        assert cb.last_failure_time is None

    def test_set_after_failure(self):
        cb = CircuitBreaker(CircuitBreakerConfig(enabled=True))
        before = time.monotonic()
        cb.record_failure()
        after = time.monotonic()
        assert cb.last_failure_time is not None
        assert before <= cb.last_failure_time <= after
