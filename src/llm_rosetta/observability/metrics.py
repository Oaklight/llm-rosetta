"""In-process metrics collector.

All data structures are plain Python objects — no framework
dependencies.  Since typical consumers run on a single asyncio event
loop thread, no locks are required.

This module is framework-agnostic and can be used by any consumer
(the llm-rosetta gateway, argo-proxy, or standalone scripts).
"""

from __future__ import annotations

import time
from collections.abc import Iterable
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
    latencies: list[float] = field(default_factory=list)


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
        bucket.latencies.append(round(duration_ms, 2))
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
                        "t": key - now,
                        "count": bucket.count,
                        "avg_ms": round(avg_ms, 2),
                        "errors": bucket.error_count,
                        "latencies": bucket.latencies[:],
                    }
                )
            else:
                series.append(
                    {
                        "t": key - now,
                        "count": 0,
                        "avg_ms": 0,
                        "errors": 0,
                        "latencies": [],
                    }
                )
        return series


# ---------------------------------------------------------------------------
# Per-provider stats
# ---------------------------------------------------------------------------


@dataclass
class _ProviderStats:
    """Recent rolling stats for a single target provider.

    Tracks the last *window_size* requests so that ``success_rate`` and
    ``avg_latency_ms`` are computed over a bounded, recent sample rather
    than the entire uptime of the process.
    """

    window_size: int = 100

    # Circular buffer tracking (duration_ms, is_error) for recent requests.
    _durations: list[float] = field(default_factory=list)
    _errors: list[bool] = field(default_factory=list)
    _pos: int = 0  # next write position (only relevant when buffer is full)
    _full: bool = False  # True once the buffer has been filled once

    # Most recent error message (if any)
    last_error: str | None = None

    def record(
        self, duration_ms: float, *, is_error: bool, error_detail: str | None = None
    ) -> None:
        if len(self._durations) < self.window_size:
            self._durations.append(duration_ms)
            self._errors.append(is_error)
            if len(self._durations) == self.window_size:
                self._full = True
                self._pos = 0
        else:
            self._durations[self._pos] = duration_ms
            self._errors[self._pos] = is_error
            self._pos = (self._pos + 1) % self.window_size
        if is_error and error_detail:
            self.last_error = error_detail

    @property
    def success_rate(self) -> float:
        if not self._errors:
            return 1.0
        errors = sum(1 for e in self._errors if e)
        return round(1.0 - errors / len(self._errors), 4)

    @property
    def avg_latency_ms(self) -> float:
        if not self._durations:
            return 0.0
        return round(sum(self._durations) / len(self._durations), 1)

    @property
    def sample_size(self) -> int:
        return len(self._durations)

    def is_critical(self, threshold: float = 0.5) -> bool:
        """True when success_rate has dropped below *threshold* on a meaningful sample."""
        return self.sample_size >= 10 and self.success_rate < threshold


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

    # Token usage
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    total_cache_read_tokens: int = 0
    total_cache_creation_tokens: int = 0
    total_reasoning_tokens: int = 0
    by_model_tokens: dict[str, dict[str, int]] = field(default_factory=dict)
    by_provider_tokens: dict[str, dict[str, int]] = field(default_factory=dict)

    # Gauge
    active_streams: int = 0

    # Time-series
    _window: _RollingWindow = field(default_factory=_RollingWindow)

    # Per-provider rolling stats (keyed by provider_name)
    _provider_stats: dict[str, _ProviderStats] = field(default_factory=dict)

    # Timing
    _start_time: float = field(default_factory=time.monotonic)

    def _get_provider_stats(self, provider_name: str) -> _ProviderStats:
        stats = self._provider_stats.get(provider_name)
        if stats is None:
            stats = _ProviderStats()
            self._provider_stats[provider_name] = stats
        return stats

    # Token field keys tracked in by_model_tokens / by_provider_tokens dicts
    _TOKEN_KEYS: tuple[str, ...] = (
        "input_tokens",
        "output_tokens",
        "cache_read_tokens",
        "cache_creation_tokens",
        "reasoning_tokens",
    )

    def _accumulate_tokens(
        self,
        model: str,
        provider_name: str | None,
        *,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_read_tokens: int | None = None,
        cache_creation_tokens: int | None = None,
        reasoning_tokens: int | None = None,
    ) -> None:
        """Add token counts to totals and per-model/per-provider breakdowns."""
        vals = {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_read_tokens": cache_read_tokens,
            "cache_creation_tokens": cache_creation_tokens,
            "reasoning_tokens": reasoning_tokens,
        }
        any_set = False
        for key, val in vals.items():
            if val is not None:
                attr = f"total_{key}"
                setattr(self, attr, getattr(self, attr) + val)
                any_set = True

        if not any_set:
            return

        mt = self.by_model_tokens.get(model)
        if mt is None:
            mt = {k: 0 for k in self._TOKEN_KEYS}
            self.by_model_tokens[model] = mt
        for key, val in vals.items():
            mt[key] = mt.get(key, 0) + (val or 0)

        if provider_name:
            pt = self.by_provider_tokens.get(provider_name)
            if pt is None:
                pt = {k: 0 for k in self._TOKEN_KEYS}
                self.by_provider_tokens[provider_name] = pt
            for key, val in vals.items():
                pt[key] = pt.get(key, 0) + (val or 0)

    def record_request(
        self,
        *,
        model: str,
        source: str,
        target: str,
        status_code: int,
        duration_ms: float,
        is_stream: bool,
        provider_name: str | None = None,
        error_detail: str | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_read_tokens: int | None = None,
        cache_creation_tokens: int | None = None,
        reasoning_tokens: int | None = None,
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
        # Use provider_name (e.g. "Argo Claude") when available,
        # fall back to API type (e.g. "anthropic") for backward compat.
        target_key = provider_name or target
        self.by_target_provider[target_key] = (
            self.by_target_provider.get(target_key, 0) + 1
        )
        self.by_status_code[status_code] = self.by_status_code.get(status_code, 0) + 1

        self._accumulate_tokens(
            model,
            provider_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            reasoning_tokens=reasoning_tokens,
        )

        self._window.record(duration_ms, is_error=is_error)

        # Per-provider stats (use provider_name if available, fall back to target)
        pname = provider_name or target
        self._get_provider_stats(pname).record(
            duration_ms, is_error=is_error, error_detail=error_detail
        )

    def record_usage(
        self,
        *,
        model: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        cache_read_tokens: int | None = None,
        cache_creation_tokens: int | None = None,
        reasoning_tokens: int | None = None,
        provider_name: str | None = None,
    ) -> None:
        """Record token usage separately (for streaming write-back)."""
        self._accumulate_tokens(
            model,
            provider_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_creation_tokens=cache_creation_tokens,
            reasoning_tokens=reasoning_tokens,
        )

    def provider_health_snapshot(self) -> dict[str, dict]:
        """Return a JSON-serializable per-provider health snapshot."""
        out: dict[str, dict] = {}
        for name, stats in self._provider_stats.items():
            out[name] = {
                "status": "critical" if stats.is_critical() else "ok",
                "success_rate": stats.success_rate,
                "avg_latency_ms": stats.avg_latency_ms,
                "sample_size": stats.sample_size,
                "last_error": stats.last_error,
            }
        return out

    def any_critical_provider(self) -> bool:
        """Return True if any tracked provider is critically unhealthy."""
        return any(s.is_critical() for s in self._provider_stats.values())

    def export_counters(self) -> dict:
        """Return counters suitable for persistence (no time-series)."""
        return {
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "total_streams": self.total_streams,
            "by_model": dict(self.by_model),
            "by_source_provider": dict(self.by_source_provider),
            "by_target_provider": dict(self.by_target_provider),
            "by_status_code": {str(k): v for k, v in self.by_status_code.items()},
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cache_read_tokens": self.total_cache_read_tokens,
            "total_cache_creation_tokens": self.total_cache_creation_tokens,
            "total_reasoning_tokens": self.total_reasoning_tokens,
            "by_model_tokens": {k: dict(v) for k, v in self.by_model_tokens.items()},
            "by_provider_tokens": {
                k: dict(v) for k, v in self.by_provider_tokens.items()
            },
        }

    def load_counters(self, data: dict) -> None:
        """Restore counters from a previously exported dict."""
        self.total_requests = data.get("total_requests", 0)
        self.total_errors = data.get("total_errors", 0)
        self.total_streams = data.get("total_streams", 0)
        self.by_model = dict(data.get("by_model", {}))
        self.by_source_provider = dict(data.get("by_source_provider", {}))
        self.by_target_provider = dict(data.get("by_target_provider", {}))
        self.by_status_code = {
            int(k): v for k, v in data.get("by_status_code", {}).items()
        }
        self.total_input_tokens = data.get("total_input_tokens", 0)
        self.total_output_tokens = data.get("total_output_tokens", 0)
        self.total_cache_read_tokens = data.get("total_cache_read_tokens", 0)
        self.total_cache_creation_tokens = data.get("total_cache_creation_tokens", 0)
        self.total_reasoning_tokens = data.get("total_reasoning_tokens", 0)
        self.by_model_tokens = {
            k: dict(v) for k, v in data.get("by_model_tokens", {}).items()
        }
        self.by_provider_tokens = {
            k: dict(v) for k, v in data.get("by_provider_tokens", {}).items()
        }

    def rebuild_counters(self, rows: Iterable[dict]) -> int:
        """Rebuild all counters from request log rows.

        Replaces current counter values with aggregates computed from
        *rows*.  Each row must have ``model``, ``source_provider``,
        ``target_provider``, ``target_provider_name``, ``is_stream``,
        and ``status_code`` keys.

        Accepts any iterable (including generators) so the caller can
        stream rows in batches without loading the entire table into
        memory at once.

        Time-series and per-provider rolling stats are NOT rebuilt
        (they only make sense for recent data).

        Args:
            rows: Iterable of request log entry dicts.

        Returns:
            Number of rows processed.
        """
        # Build in temporaries, then swap atomically.  The gateway runs
        # on a single asyncio thread so there are no true data races,
        # but building first avoids exposing half-rebuilt counters to
        # concurrent ``snapshot()`` calls between await points.
        total_requests = 0
        total_errors = 0
        total_streams = 0
        by_model: dict[str, int] = {}
        by_source: dict[str, int] = {}
        by_target: dict[str, int] = {}
        by_status: dict[int, int] = {}
        tmp = MetricsCollector()

        for r in rows:
            total_requests += 1
            sc = r.get("status_code", 200)
            if sc >= 400:
                total_errors += 1
            if r.get("is_stream"):
                total_streams += 1

            model = r.get("model", "unknown")
            by_model[model] = by_model.get(model, 0) + 1

            source = r.get("source_provider", "unknown")
            by_source[source] = by_source.get(source, 0) + 1

            target = r.get("target_provider_name") or r.get(
                "target_provider", "unknown"
            )
            by_target[target] = by_target.get(target, 0) + 1

            by_status[sc] = by_status.get(sc, 0) + 1

            pn = r.get("target_provider_name") or r.get("target_provider", "unknown")
            tmp._accumulate_tokens(
                model,
                pn,
                **{tk: r.get(tk) for tk in self._TOKEN_KEYS},
            )

        # Atomic swap — active_streams is live state, not rebuilt.
        self.total_requests = total_requests
        self.total_errors = total_errors
        self.total_streams = total_streams
        self.by_model = by_model
        self.by_source_provider = by_source
        self.by_target_provider = by_target
        self.by_status_code = by_status
        self.total_input_tokens = tmp.total_input_tokens
        self.total_output_tokens = tmp.total_output_tokens
        self.total_cache_read_tokens = tmp.total_cache_read_tokens
        self.total_cache_creation_tokens = tmp.total_cache_creation_tokens
        self.total_reasoning_tokens = tmp.total_reasoning_tokens
        self.by_model_tokens = tmp.by_model_tokens
        self.by_provider_tokens = tmp.by_provider_tokens

        return total_requests

    def merge_rebuild(self, baseline: dict, pre_snapshot: dict) -> None:
        """Atomically merge a background-rebuilt baseline with live traffic.

        *baseline* is the result of :meth:`export_counters` from a
        temporary :class:`MetricsCollector` that processed the full
        request log in a background thread.

        *pre_snapshot* is the result of :meth:`export_counters` captured
        on the event-loop thread **before** the background rebuild
        started.  The delta ``current - pre_snapshot`` represents
        requests that arrived during the rebuild window.

        The final value for each counter is ``baseline + delta``, applied
        via a single-assignment swap so concurrent readers never see
        partial state.

        Must be called from the event-loop thread (not from an executor).
        """

        def _merge_int(base_key: str) -> int:
            delta = getattr(self, base_key) - pre_snapshot.get(base_key, 0)
            return baseline.get(base_key, 0) + max(0, delta)

        def _merge_dict(base_key: str) -> dict:
            b = dict(baseline.get(base_key, {}))
            pre = pre_snapshot.get(base_key, {})
            cur = getattr(self, base_key)
            for k, v in cur.items():
                delta = v - pre.get(k, 0)
                if delta > 0:
                    b[k] = b.get(k, 0) + delta
            return b

        def _merge_token_dict(base_key: str) -> dict[str, dict[str, int]]:
            b: dict[str, dict[str, int]] = {
                k: dict(v) for k, v in baseline.get(base_key, {}).items()
            }
            pre = pre_snapshot.get(base_key, {})
            cur = getattr(self, base_key)
            for k, cur_toks in cur.items():
                pre_toks = pre.get(k, {})
                if k not in b:
                    b[k] = {tk: 0 for tk in self._TOKEN_KEYS}
                for tok_key in self._TOKEN_KEYS:
                    delta = cur_toks.get(tok_key, 0) - pre_toks.get(tok_key, 0)
                    if delta > 0:
                        b[k][tok_key] = b[k].get(tok_key, 0) + delta
            return b

        # Build all values before swapping
        total_requests = _merge_int("total_requests")
        total_errors = _merge_int("total_errors")
        total_streams = _merge_int("total_streams")
        by_model = _merge_dict("by_model")
        by_source = _merge_dict("by_source_provider")
        by_target = _merge_dict("by_target_provider")
        total_input = _merge_int("total_input_tokens")
        total_output = _merge_int("total_output_tokens")
        total_cache_read = _merge_int("total_cache_read_tokens")
        total_cache_creation = _merge_int("total_cache_creation_tokens")
        total_reasoning = _merge_int("total_reasoning_tokens")
        by_model_tokens = _merge_token_dict("by_model_tokens")
        by_provider_tokens = _merge_token_dict("by_provider_tokens")

        # by_status_code uses int keys, same merge pattern
        b_status: dict[int, int] = {
            int(k): v for k, v in baseline.get("by_status_code", {}).items()
        }
        pre_status = {
            int(k): v for k, v in pre_snapshot.get("by_status_code", {}).items()
        }
        for k, v in self.by_status_code.items():
            delta = v - pre_status.get(k, 0)
            if delta > 0:
                b_status[k] = b_status.get(k, 0) + delta

        # Atomic swap
        self.total_requests = total_requests
        self.total_errors = total_errors
        self.total_streams = total_streams
        self.by_model = by_model
        self.by_source_provider = by_source
        self.by_target_provider = by_target
        self.by_status_code = b_status
        self.total_input_tokens = total_input
        self.total_output_tokens = total_output
        self.total_cache_read_tokens = total_cache_read
        self.total_cache_creation_tokens = total_cache_creation
        self.total_reasoning_tokens = total_reasoning
        self.by_model_tokens = by_model_tokens
        self.by_provider_tokens = by_provider_tokens

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
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_cache_read_tokens": self.total_cache_read_tokens,
            "total_cache_creation_tokens": self.total_cache_creation_tokens,
            "total_reasoning_tokens": self.total_reasoning_tokens,
            "by_model_tokens": {k: dict(v) for k, v in self.by_model_tokens.items()},
            "by_provider_tokens": {
                k: dict(v) for k, v in self.by_provider_tokens.items()
            },
            "series": self._window.get_series(series_seconds),
            "providers": self.provider_health_snapshot(),
        }
