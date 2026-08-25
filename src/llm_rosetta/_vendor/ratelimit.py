# /// zerodep
# version = "0.1.0"
# deps = []
# tier = "subsystem"
# category = "network"
# note = "Install/update via `zerodep add ratelimit`"
# ///
"""Multi-algorithm rate limiter with zero dependencies.

Provides four in-memory rate-limiting algorithms behind a common
``RateLimiter`` protocol, plus convenience wrappers (decorator, context
manager, async support, string quota notation).

Algorithms:

- **Token bucket** — smooth rate with configurable burst tolerance.
- **Fixed window** — simplest counter-per-window, lowest overhead.
- **Sliding window counter** — accurate without boundary-burst artifacts.
- **GCRA** (Generic Cell Rate Algorithm) — constant-rate shaping via a
  single TAT (Theoretical Arrival Time) timestamp.

Typical usage::

    limiter = TokenBucketLimiter(rate=10.0, capacity=20)
    result = limiter.acquire("client-ip")
    if not result.allowed:
        return 429, {"Retry-After": str(result.retry_after)}

    # Or with convenience API
    @ratelimit("10/s", algorithm="sliding_window")
    def handle_request(): ...

Thread safety: individual limiter classes are **not** thread-safe.
For concurrent access from multiple threads, wrap with
:class:`ThreadSafeLimiter` or pass a pre-wrapped instance via the
``limiter`` parameter of :class:`ratelimit`.

Part of zerodep: https://github.com/Oaklight/zerodep
Copyright (c) 2026 Peng Ding. MIT License.
"""

from __future__ import annotations

import asyncio
import math
import re
import threading
import time
from abc import abstractmethod
from collections.abc import Callable
from functools import wraps
from typing import Any, Protocol, runtime_checkable

__all__: list[str] = [
    "RateLimitResult",
    "RateLimiter",
    "TokenBucketLimiter",
    "FixedWindowLimiter",
    "SlidingWindowLimiter",
    "GCRALimiter",
    "ThreadSafeLimiter",
    "RateLimitExceeded",
    "ratelimit",
    "create_limiter",
    "parse_quota",
]

# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------


class RateLimitResult:
    """Outcome of a rate-limit check.

    Instances are **not hashable** (mutable ``__slots__`` class).

    Attributes:
        allowed: Whether the request is permitted.
        limit: Total quota (bucket capacity or window limit).
        remaining: Remaining quota (>= 0).  May be ``float`` for
            algorithms with fractional token counts (token bucket,
            sliding window, GCRA).
        reset_at: Monotonic timestamp when quota fully replenishes or
            the current window ends.
        retry_after: Seconds to wait before retrying (raw ``float``,
            not rounded).  ``None`` when ``allowed`` is ``True``.
    """

    __slots__ = ("allowed", "limit", "remaining", "reset_at", "retry_after")

    def __init__(
        self,
        allowed: bool,
        limit: int | float,
        remaining: int | float,
        reset_at: float,
        retry_after: float | None,
    ) -> None:
        self.allowed = allowed
        self.limit = limit
        self.remaining = remaining
        self.reset_at = reset_at
        self.retry_after = retry_after

    def __repr__(self) -> str:
        return (
            f"RateLimitResult(allowed={self.allowed!r}, limit={self.limit!r}, "
            f"remaining={self.remaining!r}, reset_at={self.reset_at!r}, "
            f"retry_after={self.retry_after!r})"
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, RateLimitResult):
            return NotImplemented
        return (
            self.allowed == other.allowed
            and self.limit == other.limit
            and self.remaining == other.remaining
            and self.reset_at == other.reset_at
            and self.retry_after == other.retry_after
        )


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class RateLimiter(Protocol):
    """Common interface for all rate-limiter implementations."""

    def acquire(self, key: str, tokens: int = 1) -> RateLimitResult:
        """Check and consume *tokens* units of quota for *key*."""
        ...

    def peek(self, key: str) -> RateLimitResult:
        """Return current quota state for *key* without consuming."""
        ...

    async def aacquire(self, key: str, tokens: int = 1) -> RateLimitResult:
        """Async version of :meth:`acquire`."""
        ...

    async def apeek(self, key: str) -> RateLimitResult:
        """Async version of :meth:`peek`."""
        ...


# ---------------------------------------------------------------------------
# Eviction mixin
# ---------------------------------------------------------------------------

_EVICT_INTERVAL = 128
_DEFAULT_RETRY_WAIT = 0.1  # fallback wait (seconds) when retry_after is None


class _AsyncMixin:
    """Adds ``aacquire`` and ``apeek`` coroutines that delegate to sync methods.

    The sync methods are pure in-memory computation (no I/O), so the
    async wrappers yield to the event loop once before calling them.
    This ensures the coroutine is a proper awaitable with a real
    suspension point, avoiding starvation in tight loops.
    """

    @abstractmethod
    def acquire(self, key: str, tokens: int = 1) -> RateLimitResult: ...
    @abstractmethod
    def peek(self, key: str) -> RateLimitResult: ...

    async def aacquire(self, key: str, tokens: int = 1) -> RateLimitResult:
        await asyncio.sleep(0)
        return self.acquire(key, tokens)

    async def apeek(self, key: str) -> RateLimitResult:
        await asyncio.sleep(0)
        return self.peek(key)


class _EvictionMixin:
    """Amortised stale-key eviction for in-memory stores.

    Subclasses must set ``_call_count: int`` and implement
    ``_evict_stale(now)`` to remove expired entries.
    """

    _call_count: int

    def _maybe_evict(self, now: float) -> None:
        self._call_count += 1
        if self._call_count >= _EVICT_INTERVAL:
            self._call_count = 0
            self._evict_stale(now)

    def _evict_stale(self, now: float) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Token bucket
# ---------------------------------------------------------------------------


class _Bucket:
    __slots__ = ("tokens", "last_refill")

    def __init__(self, tokens: float, last_refill: float) -> None:
        self.tokens = tokens
        self.last_refill = last_refill


class TokenBucketLimiter(_AsyncMixin, _EvictionMixin):
    """Token-bucket rate limiter.

    Tokens are replenished lazily on each ``acquire`` call.

    Args:
        rate: Tokens added per second.
        capacity: Maximum tokens the bucket can hold (burst size).
        clock: Callable returning current monotonic time.
    """

    def __init__(
        self,
        rate: float,
        capacity: int,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.rate = rate
        self.capacity = capacity
        self._clock = clock or time.monotonic
        self._buckets: dict[str, _Bucket] = {}
        self._call_count = 0

    def acquire(self, key: str, tokens: int = 1) -> RateLimitResult:
        now = self._clock()
        self._maybe_evict(now)

        # Inlined _get_or_create + _refill for hot-path performance
        buckets = self._buckets
        bucket = buckets.get(key)
        if bucket is None:
            bucket = _Bucket(float(self.capacity), now)
            buckets[key] = bucket
        else:
            elapsed = now - bucket.last_refill
            if elapsed > 0:
                bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.rate)
                bucket.last_refill = now

        cur = bucket.tokens
        capacity = self.capacity
        rate = self.rate
        if cur >= tokens:
            cur -= tokens
            bucket.tokens = cur
            return RateLimitResult(
                True,
                capacity,
                cur,
                now + (capacity - cur) / rate,
                None,
            )

        return RateLimitResult(
            False,
            capacity,
            cur,
            now + capacity / rate,
            (tokens - cur) / rate,
        )

    def peek(self, key: str) -> RateLimitResult:
        """Return current quota state without consuming.

        Note: initializes internal state for previously unseen keys
        (the bucket starts full, so the result is correct).
        """
        now = self._clock()
        bucket = self._get_or_create(key, now)
        self._refill(bucket, now)
        return RateLimitResult(
            allowed=bucket.tokens >= 1,
            limit=self.capacity,
            remaining=int(bucket.tokens),
            reset_at=now + (self.capacity - bucket.tokens) / self.rate,
            retry_after=None,
        )

    def _get_or_create(self, key: str, now: float) -> _Bucket:
        bucket = self._buckets.get(key)
        if bucket is None:
            bucket = _Bucket(tokens=float(self.capacity), last_refill=now)
            self._buckets[key] = bucket
        return bucket

    def _refill(self, bucket: _Bucket, now: float) -> None:
        elapsed = now - bucket.last_refill
        if elapsed > 0:
            bucket.tokens = min(self.capacity, bucket.tokens + elapsed * self.rate)
            bucket.last_refill = now

    def _evict_stale(self, now: float) -> None:
        stale = [
            k
            for k, b in self._buckets.items()
            if b.tokens + (now - b.last_refill) * self.rate >= self.capacity
        ]
        for k in stale:
            del self._buckets[k]


# ---------------------------------------------------------------------------
# Fixed window
# ---------------------------------------------------------------------------


class _FixedWindow:
    __slots__ = ("count", "window_start")

    def __init__(self, count: int, window_start: float) -> None:
        self.count = count
        self.window_start = window_start


class FixedWindowLimiter(_AsyncMixin, _EvictionMixin):
    """Fixed-window rate limiter.

    Divides time into non-overlapping windows.  Each key gets at most
    ``limit`` requests per window.

    Args:
        limit: Maximum requests per window.
        window_seconds: Window duration in seconds.
        clock: Callable returning current monotonic time.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")
        self.limit = limit
        self.window_seconds = window_seconds
        self._clock = clock or time.monotonic
        self._windows: dict[str, _FixedWindow] = {}
        self._call_count = 0

    def acquire(self, key: str, tokens: int = 1) -> RateLimitResult:
        now = self._clock()
        self._maybe_evict(now)

        # Inlined _get_or_reset for hot-path performance
        windows = self._windows
        window = windows.get(key)
        ws = self.window_seconds
        if window is None or now - window.window_start >= ws:
            window = _FixedWindow(0, now)
            windows[key] = window

        limit = self.limit
        if window.count + tokens <= limit:
            window.count += tokens
            return RateLimitResult(
                True,
                limit,
                limit - window.count,
                window.window_start + ws,
                None,
            )

        reset_at = window.window_start + ws
        return RateLimitResult(
            False,
            limit,
            max(0, limit - window.count),
            reset_at,
            max(0.0, reset_at - now),
        )

    def peek(self, key: str) -> RateLimitResult:
        now = self._clock()
        window = self._get_or_reset(key, now)
        remaining = max(0, self.limit - window.count)
        reset_at = window.window_start + self.window_seconds
        return RateLimitResult(
            allowed=remaining >= 1,
            limit=self.limit,
            remaining=remaining,
            reset_at=reset_at,
            retry_after=None,
        )

    def _get_or_reset(self, key: str, now: float) -> _FixedWindow:
        window = self._windows.get(key)
        if window is None or now - window.window_start >= self.window_seconds:
            window = _FixedWindow(count=0, window_start=now)
            self._windows[key] = window
        return window

    def _evict_stale(self, now: float) -> None:
        stale = [
            k
            for k, w in self._windows.items()
            if now - w.window_start >= self.window_seconds * 2
        ]
        for k in stale:
            del self._windows[k]


# ---------------------------------------------------------------------------
# Sliding window counter
# ---------------------------------------------------------------------------


class _SlidingState:
    __slots__ = ("prev_count", "prev_start", "curr_count", "curr_start")

    def __init__(
        self,
        prev_count: int,
        prev_start: float,
        curr_count: int,
        curr_start: float,
    ) -> None:
        self.prev_count = prev_count
        self.prev_start = prev_start
        self.curr_count = curr_count
        self.curr_start = curr_start


class SlidingWindowLimiter(_AsyncMixin, _EvictionMixin):
    """Sliding-window counter rate limiter.

    Blends the previous window's count with the current window's count
    weighted by overlap fraction, eliminating boundary-burst artifacts.

    Args:
        limit: Maximum requests per window.
        window_seconds: Window duration in seconds.
        clock: Callable returning current monotonic time.
    """

    def __init__(
        self,
        limit: int,
        window_seconds: float,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if limit <= 0:
            raise ValueError(f"limit must be positive, got {limit}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be positive, got {window_seconds}")
        self.limit = limit
        self.window_seconds = window_seconds
        self._clock = clock or time.monotonic
        self._states: dict[str, _SlidingState] = {}
        self._call_count = 0

    def acquire(self, key: str, tokens: int = 1) -> RateLimitResult:
        now = self._clock()
        self._maybe_evict(now)
        ws = self.window_seconds
        limit = self.limit

        # Inlined _get_or_create + _advance + _effective_count
        states = self._states
        state = states.get(key)
        if state is None:
            state = _SlidingState(0, now - ws, 0, now)
            states[key] = state
        elif now >= state.curr_start + ws:
            ew = int((now - state.curr_start) / ws)
            if ew == 1:
                state.prev_count = state.curr_count
                state.prev_start = state.curr_start
                state.curr_count = 0
                state.curr_start = state.prev_start + ws
            else:
                state.prev_count = 0
                state.prev_start = state.curr_start + (ew - 1) * ws
                state.curr_count = 0
                state.curr_start = state.prev_start + ws

        elapsed_in_curr = now - state.curr_start
        overlap = max(0.0, 1.0 - elapsed_in_curr / ws)
        effective = state.curr_count + state.prev_count * overlap

        if effective + tokens <= limit:
            state.curr_count += tokens
            new_eff = state.curr_count + state.prev_count * overlap
            return RateLimitResult(
                True,
                limit,
                max(0.0, limit - new_eff),
                state.curr_start + ws,
                None,
            )

        reset_at = state.curr_start + ws
        return RateLimitResult(
            False,
            limit,
            max(0.0, limit - effective),
            reset_at,
            max(0.0, reset_at - now),
        )

    def peek(self, key: str) -> RateLimitResult:
        now = self._clock()
        state = self._get_or_create(key, now)
        self._advance(state, now)
        effective = self._effective_count(state, now)
        remaining = max(0, self.limit - math.ceil(effective))
        reset_at = state.curr_start + self.window_seconds
        return RateLimitResult(
            allowed=remaining >= 1,
            limit=self.limit,
            remaining=remaining,
            reset_at=reset_at,
            retry_after=None,
        )

    def _get_or_create(self, key: str, now: float) -> _SlidingState:
        state = self._states.get(key)
        if state is None:
            state = _SlidingState(
                prev_count=0,
                prev_start=now - self.window_seconds,
                curr_count=0,
                curr_start=now,
            )
            self._states[key] = state
        return state

    def _advance(self, state: _SlidingState, now: float) -> None:
        """Roll forward, jumping directly to the correct window."""
        if now < state.curr_start + self.window_seconds:
            return
        elapsed_windows = int((now - state.curr_start) / self.window_seconds)
        if elapsed_windows == 1:
            state.prev_count = state.curr_count
            state.prev_start = state.curr_start
            state.curr_count = 0
            state.curr_start = state.prev_start + self.window_seconds
        else:
            state.prev_count = 0
            state.prev_start = (
                state.curr_start + (elapsed_windows - 1) * self.window_seconds
            )
            state.curr_count = 0
            state.curr_start = state.prev_start + self.window_seconds

    def _effective_count(self, state: _SlidingState, now: float) -> float:
        elapsed_in_curr = now - state.curr_start
        overlap = max(0.0, 1.0 - elapsed_in_curr / self.window_seconds)
        return state.curr_count + state.prev_count * overlap

    def _evict_stale(self, now: float) -> None:
        cutoff = now - self.window_seconds * 3
        stale = [k for k, s in self._states.items() if s.curr_start < cutoff]
        for k in stale:
            del self._states[k]


# ---------------------------------------------------------------------------
# GCRA (Generic Cell Rate Algorithm)
# ---------------------------------------------------------------------------


class GCRALimiter(_AsyncMixin, _EvictionMixin):
    """GCRA (Generic Cell Rate Algorithm) rate limiter.

    Maintains a single TAT (Theoretical Arrival Time) per key.
    Mathematically equivalent to a leaky bucket but with simpler state.

    Args:
        rate: Requests allowed per second.
        burst: Maximum burst size above the steady rate.
        clock: Callable returning current monotonic time.
    """

    def __init__(
        self,
        rate: float,
        burst: int = 0,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if rate <= 0:
            raise ValueError(f"rate must be positive, got {rate}")
        if burst < 0:
            raise ValueError(f"burst must be non-negative, got {burst}")
        self.rate = rate
        self.burst = burst
        self._emission_interval = 1.0 / rate
        self._delay_tolerance = burst * self._emission_interval
        self._limit = burst + 1
        self._clock = clock or time.monotonic
        self._tats: dict[str, float] = {}
        self._call_count = 0

    def acquire(self, key: str, tokens: int = 1) -> RateLimitResult:
        now = self._clock()
        self._maybe_evict(now)
        ei = self._emission_interval
        dt = self._delay_tolerance
        limit = self._limit
        tat = self._tats.get(key, now)

        new_tat = max(tat, now) + ei * tokens
        allow_at = new_tat - dt - ei

        if now < allow_at:
            return RateLimitResult(
                False,
                limit,
                max(0.0, (dt - (tat - now)) / ei),
                tat,
                allow_at - now,
            )

        self._tats[key] = new_tat
        return RateLimitResult(
            True,
            limit,
            max(0.0, (dt - (new_tat - now)) / ei),
            new_tat,
            None,
        )

    def peek(self, key: str) -> RateLimitResult:
        now = self._clock()
        tat = self._tats.get(key, now)
        remaining = max(
            0,
            int(
                (self._delay_tolerance - (max(tat, now) - now))
                / self._emission_interval
            ),
        )
        return RateLimitResult(
            allowed=remaining >= 1,
            limit=self._limit,
            remaining=remaining,
            reset_at=max(tat, now),
            retry_after=None,
        )

    def _evict_stale(self, now: float) -> None:
        stale = [k for k, tat in self._tats.items() if tat <= now]
        for k in stale:
            del self._tats[k]


# ---------------------------------------------------------------------------
# Quota string parsing
# ---------------------------------------------------------------------------

_UNIT_MAP: dict[str, float] = {
    "s": 1.0,
    "sec": 1.0,
    "second": 1.0,
    "m": 60.0,
    "min": 60.0,
    "minute": 60.0,
    "h": 3600.0,
    "hr": 3600.0,
    "hour": 3600.0,
    "d": 86400.0,
    "day": 86400.0,
}

_QUOTA_RE = re.compile(
    r"^\s*(\d+)\s*(?:/|per)\s*([a-z]+)"
    r"(?:\s+burst\s+(\d+))?\s*$",
    re.IGNORECASE,
)


def parse_quota(quota: str) -> dict[str, Any]:
    """Parse a quota string like ``"100/s"`` or ``"10 per minute burst 20"``.

    Returns:
        Dict with keys ``limit`` (int), ``period`` (float in seconds),
        and ``burst`` (int or None).
    """
    m = _QUOTA_RE.match(quota)
    if not m:
        raise ValueError(
            f"invalid quota string: {quota!r}  "
            f"(expected: '<N>/<unit>' or '<N> per <unit> [burst <B>]')"
        )
    limit = int(m.group(1))
    unit = m.group(2).lower()
    # strip trailing "s" for plurals (e.g. "seconds" -> "second") but
    # not for single-char units like "s" itself
    if len(unit) > 1 and unit.endswith("s"):
        unit = unit[:-1]
    if unit not in _UNIT_MAP:
        raise ValueError(f"unknown time unit: {m.group(2)!r}")
    period = _UNIT_MAP[unit]
    burst = int(m.group(3)) if m.group(3) else None
    return {"limit": limit, "period": period, "burst": burst}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

_ALGORITHMS = {"token_bucket", "fixed_window", "sliding_window", "gcra"}


def create_limiter(
    algorithm: str,
    quota: str,
    *,
    clock: Callable[[], float] | None = None,
) -> RateLimiter:
    """Create a rate limiter from an algorithm name and quota string.

    Args:
        algorithm: One of ``"token_bucket"``, ``"fixed_window"``,
            ``"sliding_window"``, or ``"gcra"``.
        quota: Quota string like ``"100/s"`` or ``"10/m burst 20"``.
        clock: Optional clock override for testing.

    Returns:
        A :class:`RateLimiter` instance.

    How ``burst`` maps per algorithm:

    - **token_bucket**: ``burst`` sets bucket capacity directly
      (default: ``limit``).
    - **fixed_window** / **sliding_window**: ``burst`` is ignored.
    - **gcra**: ``burst`` is total capacity, so internal burst
      parameter = ``burst - 1`` (extra slots above steady rate).

    Note:
        The ``timeout`` parameter for wait-and-retry always uses
        wall-clock time (``time.monotonic``), even when the limiter
        has an injected clock.
    """
    algo = algorithm.lower()
    if algo not in _ALGORITHMS:
        raise ValueError(
            f"unknown algorithm: {algorithm!r}  "
            f"(choices: {', '.join(sorted(_ALGORITHMS))})"
        )
    q = parse_quota(quota)
    limit, period, burst = q["limit"], q["period"], q["burst"]

    if algo == "token_bucket":
        rate = limit / period
        cap = burst if burst is not None else limit
        return TokenBucketLimiter(rate=rate, capacity=cap, clock=clock)  # type: ignore[return-value]
    elif algo == "fixed_window":
        return FixedWindowLimiter(limit=limit, window_seconds=period, clock=clock)  # type: ignore[return-value]
    elif algo == "sliding_window":
        return SlidingWindowLimiter(limit=limit, window_seconds=period, clock=clock)  # type: ignore[return-value]
    else:  # gcra
        rate = limit / period
        # GCRA burst = "extra slots above steady rate", so user's
        # "burst N" (total capacity) maps to burst=N-1 internally.
        b = (burst - 1) if burst is not None and burst > 0 else 0
        return GCRALimiter(rate=rate, burst=b, clock=clock)  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Decorator / context manager
# ---------------------------------------------------------------------------


class RateLimitExceeded(Exception):
    """Raised when a rate limit is exceeded in decorator/context-manager mode."""

    def __init__(self, result: RateLimitResult) -> None:
        self.result = result
        super().__init__(
            f"rate limit exceeded: remaining={result.remaining}, "
            f"retry_after={result.retry_after}"
        )


class ratelimit:
    """Rate-limit decorator and context manager.

    Can be used as a decorator::

        @ratelimit("10/s")
        def handle(): ...

    Or as a sync/async context manager::

        with ratelimit("5/m", key="user-1"):
            do_work()

        async with ratelimit("5/m", key="user-1"):
            await do_work()

    Args:
        quota: Quota string (e.g. ``"10/s"``, ``"100/m burst 200"``).
        algorithm: Algorithm name (default: ``"token_bucket"``).
        key: Rate-limit key (default: ``"__default__"``).
        timeout: If set, wait up to *timeout* seconds for quota
            instead of raising immediately.
        limiter: Pre-built limiter instance (overrides *quota*
            and *algorithm*).
    """

    def __init__(
        self,
        quota: str | None = None,
        *,
        algorithm: str = "token_bucket",
        key: str = "__default__",
        timeout: float | None = None,
        limiter: RateLimiter | None = None,
    ) -> None:
        if limiter is not None:
            self._limiter = limiter
        elif quota is not None:
            self._limiter = create_limiter(algorithm, quota)
        else:
            raise ValueError("either 'quota' or 'limiter' must be provided")
        self._key = key
        self._timeout = timeout

    def _acquire_or_wait(self, tokens: int = 1) -> RateLimitResult:
        result = self._limiter.acquire(self._key, tokens)
        if result.allowed:
            return result
        if self._timeout is None:
            raise RateLimitExceeded(result)

        deadline = time.monotonic() + self._timeout
        while not result.allowed:
            wait = result.retry_after or _DEFAULT_RETRY_WAIT
            remaining_budget = deadline - time.monotonic()
            if remaining_budget <= 0:
                raise RateLimitExceeded(result)
            time.sleep(min(wait, remaining_budget))
            result = self._limiter.acquire(self._key, tokens)

        return result

    async def _acquire_or_wait_async(self, tokens: int = 1) -> RateLimitResult:
        result = await self._limiter.aacquire(self._key, tokens)
        if result.allowed:
            return result
        if self._timeout is None:
            raise RateLimitExceeded(result)

        deadline = time.monotonic() + self._timeout
        while not result.allowed:
            wait = result.retry_after or _DEFAULT_RETRY_WAIT
            remaining_budget = deadline - time.monotonic()
            if remaining_budget <= 0:
                raise RateLimitExceeded(result)
            await asyncio.sleep(min(wait, remaining_budget))
            result = await self._limiter.aacquire(self._key, tokens)

        return result

    def __call__(self, func: Any) -> Any:
        if asyncio.iscoroutinefunction(func):

            @wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                await self._acquire_or_wait_async()
                return await func(*args, **kwargs)

            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            self._acquire_or_wait()
            return func(*args, **kwargs)

        return sync_wrapper

    def __enter__(self) -> RateLimitResult:
        return self._acquire_or_wait()

    def __exit__(self, *exc: Any) -> None:
        pass

    async def __aenter__(self) -> RateLimitResult:
        return await self._acquire_or_wait_async()

    async def __aexit__(self, *exc: Any) -> None:
        pass


# ---------------------------------------------------------------------------
# Thread-safe wrapper
# ---------------------------------------------------------------------------


class ThreadSafeLimiter:
    """Wraps any :class:`RateLimiter` with per-key ``threading.Lock`` instances.

    Different keys are fully concurrent; only requests to the same key
    serialize.  Uses ``threading.Lock`` (not ``asyncio.Lock``) so the
    wrapper is safe for mixed sync+async access.  For pure-async
    high-contention scenarios on a single key, the lock may briefly
    block the event loop; in practice the hold time is negligible
    (microseconds of in-memory computation).

    Args:
        limiter: The underlying rate limiter.
    """

    def __init__(self, limiter: RateLimiter) -> None:
        self._limiter = limiter
        self._locks: dict[str, threading.Lock] = {}
        self._meta_lock = threading.Lock()
        # Wrap _evict_stale so eviction always runs under _meta_lock
        # and stale per-key locks are cleaned up alongside data.
        if hasattr(limiter, "_evict_stale"):
            original_evict = limiter._evict_stale  # type: ignore[union-attr]
            locks = self._locks
            meta = self._meta_lock

            def _safe_evict(
                now: float,
                _orig: Any = original_evict,
                _lock: threading.Lock = meta,
                _locks: dict[str, threading.Lock] = locks,
            ) -> None:
                with _lock:
                    _orig(now)
                    # Purge locks for keys that were just evicted.
                    # After _orig runs, any key still in the limiter's
                    # state dict is alive; the rest can be dropped.
                    alive = set(_get_state_keys(limiter))
                    stale = [k for k in _locks if k not in alive]
                    for k in stale:
                        del _locks[k]

            object.__setattr__(limiter, "_evict_stale", _safe_evict)

    def _get_lock(self, key: str) -> threading.Lock:
        lock = self._locks.get(key)
        if lock is not None:
            return lock
        with self._meta_lock:
            lock = self._locks.get(key)
            if lock is None:
                lock = threading.Lock()
                self._locks[key] = lock
            return lock

    def acquire(self, key: str, tokens: int = 1) -> RateLimitResult:
        with self._get_lock(key):
            return self._limiter.acquire(key, tokens)

    def peek(self, key: str) -> RateLimitResult:
        with self._get_lock(key):
            return self._limiter.peek(key)

    async def aacquire(self, key: str, tokens: int = 1) -> RateLimitResult:
        await asyncio.sleep(0)
        with self._get_lock(key):
            return self._limiter.acquire(key, tokens)

    async def apeek(self, key: str) -> RateLimitResult:
        await asyncio.sleep(0)
        with self._get_lock(key):
            return self._limiter.peek(key)


def _get_state_keys(limiter: Any) -> set[str]:
    """Return the set of active keys in a limiter's internal state."""
    for attr in ("_buckets", "_windows", "_states", "_tats"):
        d = getattr(limiter, attr, None)
        if d is not None:
            return set(d.keys())
    return set()
