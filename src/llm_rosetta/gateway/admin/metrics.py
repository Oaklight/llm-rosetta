"""In-process metrics collector for the gateway admin panel.

All data structures are plain Python objects.  Since the gateway runs
on a single asyncio event loop thread, no locks are required.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Rolling time-series window
# ---------------------------------------------------------------------------


@dataclass
class _Bucket:
    """Aggregation for a single second."""

    count: int = 0
    total_duration_ms: float = 0.0
    error_count: int = 0


class _RollingWindow:
    """Per-second resolution time-series with auto-expiring buckets.

    Buckets are keyed by ``int(time.monotonic())``.  Expired buckets are
    cleaned up lazily during read operations — never on the hot write
    path.
    """

    def __init__(self, window_seconds: int = 300) -> None:
        self._buckets: dict[int, _Bucket] = {}
        self._window = window_seconds

    def record(self, duration_ms: float, *, is_error: bool = False) -> None:
        """Record a single request in the current second's bucket."""
        key = int(time.monotonic())
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket()
            self._buckets[key] = bucket
        bucket.count += 1
        bucket.total_duration_ms += duration_ms
        if is_error:
            bucket.error_count += 1

    def get_series(self, seconds: int = 60) -> list[dict]:
        """Return per-second datapoints for the last *seconds*.

        Each element is ``{"t": <offset_seconds_ago>, "count": …,
        "avg_ms": …, "errors": …}``.

        Also performs lazy cleanup of expired buckets.
        """
        now = int(time.monotonic())
        cutoff = now - self._window

        # Lazy cleanup
        expired = [k for k in self._buckets if k < cutoff]
        for k in expired:
            del self._buckets[k]

        start = now - seconds
        series: list[dict] = []
        for offset in range(seconds):
            key = start + offset + 1
            bucket = self._buckets.get(key)
            if bucket is not None:
                avg_ms = bucket.total_duration_ms / bucket.count if bucket.count else 0
                series.append(
                    {
                        "t": key - now,  # negative offset (seconds ago)
                        "count": bucket.count,
                        "avg_ms": round(avg_ms, 2),
                        "errors": bucket.error_count,
                    }
                )
            else:
                series.append({"t": key - now, "count": 0, "avg_ms": 0, "errors": 0})
        return series


# ---------------------------------------------------------------------------
# Main collector
# ---------------------------------------------------------------------------


@dataclass
class MetricsCollector:
    """Lightweight in-process metrics for the gateway."""

    # Counters
    total_requests: int = 0
    total_errors: int = 0
    total_streams: int = 0

    # Breakdowns
    by_model: dict[str, int] = field(default_factory=dict)
    by_source_provider: dict[str, int] = field(default_factory=dict)
    by_target_provider: dict[str, int] = field(default_factory=dict)
    by_status_code: dict[int, int] = field(default_factory=dict)

    # Gauge
    active_streams: int = 0

    # Time-series
    _window: _RollingWindow = field(default_factory=_RollingWindow)

    # Timing
    _start_time: float = field(default_factory=time.monotonic)

    def record_request(
        self,
        *,
        model: str,
        source: str,
        target: str,
        status_code: int,
        duration_ms: float,
        is_stream: bool,
    ) -> None:
        """Record a completed proxy request."""
        self.total_requests += 1
        is_error = status_code >= 400
        if is_error:
            self.total_errors += 1
        if is_stream:
            self.total_streams += 1

        self.by_model[model] = self.by_model.get(model, 0) + 1
        self.by_source_provider[source] = self.by_source_provider.get(source, 0) + 1
        self.by_target_provider[target] = self.by_target_provider.get(target, 0) + 1
        self.by_status_code[status_code] = self.by_status_code.get(status_code, 0) + 1

        self._window.record(duration_ms, is_error=is_error)

    def snapshot(self, series_seconds: int = 60) -> dict:
        """Return a JSON-serializable metrics snapshot."""
        uptime = time.monotonic() - self._start_time
        error_rate = (
            round(self.total_errors / self.total_requests, 4)
            if self.total_requests
            else 0
        )

        return {
            "uptime_seconds": round(uptime, 1),
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "total_streams": self.total_streams,
            "error_rate": error_rate,
            "active_streams": self.active_streams,
            "by_model": dict(self.by_model),
            "by_source_provider": dict(self.by_source_provider),
            "by_target_provider": dict(self.by_target_provider),
            "by_status_code": {str(k): v for k, v in self.by_status_code.items()},
            "series": self._window.get_series(series_seconds),
        }
