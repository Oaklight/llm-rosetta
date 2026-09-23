"""Per-provider circuit breaker for the gateway proxy layer.

Implements a three-state circuit breaker (CLOSED / OPEN / HALF_OPEN) that
short-circuits requests to providers experiencing sustained failures.
This is complementary to the observational health tracking in
:class:`~llm_rosetta.observability.metrics.MetricsCollector` — the circuit
breaker *acts* on failure patterns rather than merely reporting them.

Thread-safety: each :class:`CircuitBreaker` instance uses a
:class:`threading.Lock` to serialise state transitions and counter updates.

The feature is **disabled by default** and must be opted-in via gateway config.
"""

from __future__ import annotations

import enum
import threading
import time
from dataclasses import dataclass
from typing import Any


class CircuitState(enum.Enum):
    """Three-state circuit breaker states."""

    CLOSED = "CLOSED"
    OPEN = "OPEN"
    HALF_OPEN = "HALF_OPEN"


@dataclass
class CircuitBreakerConfig:
    """Configuration for a single circuit breaker instance.

    Args:
        enabled: Whether the circuit breaker is active.
        error_threshold: Consecutive failures required to trip CLOSED -> OPEN.
        cooldown_seconds: Seconds in OPEN before transitioning to HALF_OPEN.
        half_open_max_probes: Max concurrent probe requests allowed in
            HALF_OPEN state.
    """

    enabled: bool = False
    error_threshold: int = 5
    cooldown_seconds: float = 30.0
    half_open_max_probes: int = 1


class CircuitBreaker:
    """Per-provider circuit breaker with thread-safe state transitions.

    State machine::

        CLOSED ──[threshold exceeded]──> OPEN
        OPEN   ──[cooldown elapsed]────> HALF_OPEN
        HALF_OPEN ──[probe success]────> CLOSED
        HALF_OPEN ──[probe failure]────> OPEN (reset cooldown)

    Args:
        config: Circuit breaker configuration parameters.
    """

    def __init__(self, config: CircuitBreakerConfig) -> None:
        self._config = config
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._consecutive_failures: int = 0
        self._last_failure_time: float | None = None
        self._opened_at: float | None = None
        self._half_open_probes: int = 0

    @property
    def state(self) -> CircuitState:
        """Current circuit state (may transition OPEN -> HALF_OPEN on read)."""
        with self._lock:
            self._maybe_transition_to_half_open()
            return self._state

    @property
    def failure_count(self) -> int:
        """Current consecutive failure count."""
        with self._lock:
            return self._consecutive_failures

    @property
    def last_failure_time(self) -> float | None:
        """Monotonic timestamp of the last recorded failure, or None."""
        with self._lock:
            return self._last_failure_time

    def allow_request(self) -> bool:
        """Check whether a request should be allowed through.

        Returns:
            True if the circuit is CLOSED, or HALF_OPEN with available
            probe slots.  False if the circuit is OPEN (before cooldown).
        """
        if not self._config.enabled:
            return True

        with self._lock:
            self._maybe_transition_to_half_open()

            if self._state == CircuitState.CLOSED:
                return True

            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_probes < self._config.half_open_max_probes:
                    self._half_open_probes += 1
                    return True
                return False

            # OPEN
            return False

    def record_success(self) -> None:
        """Record a successful upstream response.

        Resets the consecutive failure counter.  Transitions
        HALF_OPEN -> CLOSED on a successful probe.
        """
        if not self._config.enabled:
            return

        with self._lock:
            self._consecutive_failures = 0
            if self._state == CircuitState.HALF_OPEN:
                self._state = CircuitState.CLOSED
                self._half_open_probes = 0
                self._opened_at = None

    def record_failure(self) -> None:
        """Record a failed upstream response (5xx, timeout, connection error).

        Increments the consecutive failure counter.  Trips CLOSED -> OPEN
        when the threshold is reached.  Transitions HALF_OPEN -> OPEN on
        a probe failure (resets the cooldown timer).
        """
        if not self._config.enabled:
            return

        with self._lock:
            self._consecutive_failures += 1
            self._last_failure_time = time.monotonic()

            if self._state == CircuitState.HALF_OPEN:
                # Probe failed — re-open the circuit
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._half_open_probes = 0
                return

            if self._state == CircuitState.CLOSED:
                if self._consecutive_failures >= self._config.error_threshold:
                    self._state = CircuitState.OPEN
                    self._opened_at = time.monotonic()

    def cooldown_remaining(self) -> float:
        """Seconds remaining before the circuit transitions to HALF_OPEN.

        Returns 0.0 when CLOSED, HALF_OPEN, or the cooldown has elapsed.
        """
        with self._lock:
            if self._state != CircuitState.OPEN or self._opened_at is None:
                return 0.0
            elapsed = time.monotonic() - self._opened_at
            remaining = self._config.cooldown_seconds - elapsed
            return max(0.0, remaining)

    def get_snapshot(self) -> dict[str, Any]:
        """Return a JSON-serialisable snapshot of the circuit state."""
        with self._lock:
            self._maybe_transition_to_half_open()
            remaining = 0.0
            if self._state == CircuitState.OPEN and self._opened_at is not None:
                elapsed = time.monotonic() - self._opened_at
                remaining = max(0.0, self._config.cooldown_seconds - elapsed)
            return {
                "state": self._state.value,
                "failure_count": self._consecutive_failures,
                "last_failure": self._last_failure_time,
                "cooldown_remaining": round(remaining, 1),
            }

    # -- internal helpers --

    def _maybe_transition_to_half_open(self) -> None:
        """Transition OPEN -> HALF_OPEN if the cooldown period has elapsed.

        Must be called while holding ``self._lock``.
        """
        if self._state != CircuitState.OPEN or self._opened_at is None:
            return
        if time.monotonic() - self._opened_at >= self._config.cooldown_seconds:
            self._state = CircuitState.HALF_OPEN
            self._half_open_probes = 0


class CircuitBreakerRegistry:
    """Manages per-provider :class:`CircuitBreaker` instances.

    Lazily creates a breaker on first access for each provider.

    Args:
        default_config: Fallback config used when no per-provider override
            exists.
    """

    def __init__(self, default_config: CircuitBreakerConfig | None = None) -> None:
        self._default_config = default_config or CircuitBreakerConfig()
        self._breakers: dict[str, CircuitBreaker] = {}
        self._overrides: dict[str, CircuitBreakerConfig] = {}
        self._lock = threading.Lock()

    def set_override(self, provider_name: str, config: CircuitBreakerConfig) -> None:
        """Set a per-provider config override (replaces any existing one)."""
        with self._lock:
            self._overrides[provider_name] = config
            # If a breaker already exists for this provider, replace it
            if provider_name in self._breakers:
                self._breakers[provider_name] = CircuitBreaker(config)

    def get_or_create(
        self,
        provider_name: str,
        config: CircuitBreakerConfig | None = None,
    ) -> CircuitBreaker:
        """Return the breaker for *provider_name*, creating one if needed.

        Args:
            provider_name: Provider identifier (matches config key).
            config: Optional config override for this provider.  If not
                given, uses any previously set per-provider override or
                falls back to the registry default.

        Returns:
            The :class:`CircuitBreaker` for the provider.
        """
        with self._lock:
            if provider_name not in self._breakers:
                effective = (
                    config or self._overrides.get(provider_name) or self._default_config
                )
                self._breakers[provider_name] = CircuitBreaker(effective)
            return self._breakers[provider_name]

    def get_state(self, provider_name: str) -> CircuitState | None:
        """Return the current state for *provider_name*, or None if unknown."""
        with self._lock:
            breaker = self._breakers.get(provider_name)
        if breaker is None:
            return None
        return breaker.state

    def get_all_states(self) -> dict[str, dict[str, Any]]:
        """Return a snapshot of all tracked providers for dashboard display."""
        with self._lock:
            names = list(self._breakers.keys())
        return {name: self._breakers[name].get_snapshot() for name in names}

    @property
    def enabled(self) -> bool:
        """Whether the global default config has circuit breaking enabled."""
        return self._default_config.enabled
