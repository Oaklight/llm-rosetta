"""Tests for the gateway rate-limiter integration (zerodep-backed)."""

from __future__ import annotations

import pytest

from llm_rosetta.gateway.ratelimit import (
    FixedWindowLimiter,
    GCRALimiter,
    RateLimitResult,
    RateLimiter,
    SlidingWindowLimiter,
    TokenBucketLimiter,
    create_limiter,
    parse_quota,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeClock:
    """Deterministic clock for testing."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# ---------------------------------------------------------------------------
# Protocol conformance
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    """All four implementations must satisfy the RateLimiter protocol."""

    def test_token_bucket_is_rate_limiter(self):
        assert isinstance(TokenBucketLimiter(rate=1.0, capacity=1), RateLimiter)

    def test_fixed_window_is_rate_limiter(self):
        assert isinstance(FixedWindowLimiter(limit=1, window_seconds=1.0), RateLimiter)

    def test_sliding_window_is_rate_limiter(self):
        assert isinstance(
            SlidingWindowLimiter(limit=1, window_seconds=1.0), RateLimiter
        )

    def test_gcra_is_rate_limiter(self):
        assert isinstance(GCRALimiter(rate=1.0, burst=1), RateLimiter)


# ---------------------------------------------------------------------------
# RateLimitResult
# ---------------------------------------------------------------------------


class TestRateLimitResult:
    def test_allowed_result(self):
        r = RateLimitResult(
            allowed=True, limit=10, remaining=9, reset_at=100.0, retry_after=None
        )
        assert r.allowed is True
        assert r.remaining == 9
        assert r.retry_after is None

    def test_denied_result(self):
        r = RateLimitResult(
            allowed=False, limit=10, remaining=0, reset_at=100.0, retry_after=5.0
        )
        assert r.allowed is False
        assert r.retry_after == 5.0


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


class TestValidation:
    def test_token_bucket_rejects_zero_rate(self):
        with pytest.raises(ValueError, match="rate must be positive"):
            TokenBucketLimiter(rate=0, capacity=10)

    def test_token_bucket_rejects_negative_capacity(self):
        with pytest.raises(ValueError, match="capacity must be positive"):
            TokenBucketLimiter(rate=1.0, capacity=-1)

    def test_fixed_window_rejects_zero_limit(self):
        with pytest.raises(ValueError, match="limit must be positive"):
            FixedWindowLimiter(limit=0, window_seconds=60.0)

    def test_fixed_window_rejects_negative_window(self):
        with pytest.raises(ValueError, match="window_seconds must be positive"):
            FixedWindowLimiter(limit=10, window_seconds=-1.0)

    def test_sliding_window_rejects_zero_limit(self):
        with pytest.raises(ValueError, match="limit must be positive"):
            SlidingWindowLimiter(limit=0, window_seconds=60.0)


# ---------------------------------------------------------------------------
# Token Bucket
# ---------------------------------------------------------------------------


class TestTokenBucket:
    def test_initial_burst(self):
        clock = FakeClock()
        limiter = TokenBucketLimiter(rate=1.0, capacity=5, clock=clock)
        for i in range(5):
            r = limiter.acquire("k")
            assert r.allowed, f"request {i} should be allowed"
            assert r.remaining == 5 - i - 1

    def test_denied_after_capacity(self):
        clock = FakeClock()
        limiter = TokenBucketLimiter(rate=1.0, capacity=3, clock=clock)
        for _ in range(3):
            limiter.acquire("k")
        r = limiter.acquire("k")
        assert r.allowed is False
        assert r.remaining == 0
        assert r.retry_after is not None
        assert r.retry_after > 0

    def test_refill_over_time(self):
        clock = FakeClock()
        limiter = TokenBucketLimiter(rate=2.0, capacity=4, clock=clock)
        for _ in range(4):
            limiter.acquire("k")
        assert limiter.acquire("k").allowed is False

        clock.advance(1.0)
        r = limiter.acquire("k")
        assert r.allowed is True
        assert r.remaining == 1

    def test_refill_caps_at_capacity(self):
        clock = FakeClock()
        limiter = TokenBucketLimiter(rate=10.0, capacity=5, clock=clock)
        limiter.acquire("k")
        clock.advance(100.0)
        r = limiter.peek("k")
        assert r.remaining == 5

    def test_multi_token_acquire(self):
        clock = FakeClock()
        limiter = TokenBucketLimiter(rate=1.0, capacity=10, clock=clock)
        r = limiter.acquire("k", tokens=7)
        assert r.allowed is True
        assert r.remaining == 3

        r = limiter.acquire("k", tokens=5)
        assert r.allowed is False
        assert r.retry_after is not None

    def test_key_isolation(self):
        clock = FakeClock()
        limiter = TokenBucketLimiter(rate=1.0, capacity=2, clock=clock)
        limiter.acquire("a")
        limiter.acquire("a")
        assert limiter.acquire("a").allowed is False
        assert limiter.acquire("b").allowed is True

    def test_peek_does_not_consume(self):
        clock = FakeClock()
        limiter = TokenBucketLimiter(rate=1.0, capacity=3, clock=clock)
        r1 = limiter.peek("k")
        r2 = limiter.peek("k")
        assert r1.remaining == r2.remaining == 3

    def test_retry_after_accuracy(self):
        clock = FakeClock()
        limiter = TokenBucketLimiter(rate=2.0, capacity=2, clock=clock)
        limiter.acquire("k")
        limiter.acquire("k")
        r = limiter.acquire("k")
        assert r.allowed is False
        assert r.retry_after == 0.5


# ---------------------------------------------------------------------------
# Fixed Window
# ---------------------------------------------------------------------------


class TestFixedWindow:
    def test_allows_within_limit(self):
        clock = FakeClock()
        limiter = FixedWindowLimiter(limit=5, window_seconds=60.0, clock=clock)
        for i in range(5):
            r = limiter.acquire("k")
            assert r.allowed, f"request {i} should be allowed"

    def test_denies_over_limit(self):
        clock = FakeClock()
        limiter = FixedWindowLimiter(limit=3, window_seconds=60.0, clock=clock)
        for _ in range(3):
            limiter.acquire("k")
        r = limiter.acquire("k")
        assert r.allowed is False
        assert r.retry_after is not None
        assert r.retry_after > 0

    def test_window_reset(self):
        clock = FakeClock()
        limiter = FixedWindowLimiter(limit=2, window_seconds=10.0, clock=clock)
        limiter.acquire("k")
        limiter.acquire("k")
        assert limiter.acquire("k").allowed is False

        clock.advance(10.0)
        r = limiter.acquire("k")
        assert r.allowed is True
        assert r.remaining == 1

    def test_multi_token_acquire(self):
        clock = FakeClock()
        limiter = FixedWindowLimiter(limit=10, window_seconds=60.0, clock=clock)
        r = limiter.acquire("k", tokens=4)
        assert r.allowed is True
        assert r.remaining == 6
        r = limiter.acquire("k", tokens=7)
        assert r.allowed is False

    def test_key_isolation(self):
        clock = FakeClock()
        limiter = FixedWindowLimiter(limit=1, window_seconds=60.0, clock=clock)
        limiter.acquire("a")
        assert limiter.acquire("a").allowed is False
        assert limiter.acquire("b").allowed is True

    def test_peek_does_not_consume(self):
        clock = FakeClock()
        limiter = FixedWindowLimiter(limit=3, window_seconds=60.0, clock=clock)
        limiter.acquire("k")
        r1 = limiter.peek("k")
        r2 = limiter.peek("k")
        assert r1.remaining == r2.remaining == 2

    def test_retry_after_points_to_window_end(self):
        clock = FakeClock(start=100.0)
        limiter = FixedWindowLimiter(limit=1, window_seconds=30.0, clock=clock)
        limiter.acquire("k")
        clock.advance(5.0)
        r = limiter.acquire("k")
        assert r.allowed is False
        assert r.retry_after == pytest.approx(25.0, abs=0.1)


# ---------------------------------------------------------------------------
# Sliding Window
# ---------------------------------------------------------------------------


class TestSlidingWindow:
    def test_allows_within_limit(self):
        clock = FakeClock()
        limiter = SlidingWindowLimiter(limit=5, window_seconds=60.0, clock=clock)
        for i in range(5):
            r = limiter.acquire("k")
            assert r.allowed, f"request {i} should be allowed"

    def test_denies_over_limit(self):
        clock = FakeClock()
        limiter = SlidingWindowLimiter(limit=3, window_seconds=60.0, clock=clock)
        for _ in range(3):
            limiter.acquire("k")
        r = limiter.acquire("k")
        assert r.allowed is False

    def test_window_roll(self):
        clock = FakeClock()
        limiter = SlidingWindowLimiter(limit=4, window_seconds=10.0, clock=clock)
        for _ in range(4):
            limiter.acquire("k")
        assert limiter.acquire("k").allowed is False

        clock.advance(10.0)
        r = limiter.acquire("k")
        assert r.allowed is False

        clock.advance(5.0)
        r = limiter.acquire("k")
        assert r.allowed is True

    def test_boundary_smoothing(self):
        """Sliding window should not allow 2x burst at boundaries."""
        clock = FakeClock()
        limiter = SlidingWindowLimiter(limit=10, window_seconds=10.0, clock=clock)

        clock.advance(9.0)
        for _ in range(8):
            limiter.acquire("k")

        clock.advance(1.5)
        r = limiter.acquire("k")
        assert r.allowed is True
        r = limiter.acquire("k")
        assert r.allowed is True
        r = limiter.acquire("k")
        assert r.allowed is False

    def test_multi_token_acquire(self):
        clock = FakeClock()
        limiter = SlidingWindowLimiter(limit=10, window_seconds=60.0, clock=clock)
        r = limiter.acquire("k", tokens=6)
        assert r.allowed is True
        r = limiter.acquire("k", tokens=5)
        assert r.allowed is False

    def test_key_isolation(self):
        clock = FakeClock()
        limiter = SlidingWindowLimiter(limit=1, window_seconds=60.0, clock=clock)
        limiter.acquire("a")
        assert limiter.acquire("a").allowed is False
        assert limiter.acquire("b").allowed is True

    def test_peek_does_not_consume(self):
        clock = FakeClock()
        limiter = SlidingWindowLimiter(limit=5, window_seconds=60.0, clock=clock)
        limiter.acquire("k")
        limiter.acquire("k")
        r1 = limiter.peek("k")
        r2 = limiter.peek("k")
        assert r1.remaining == r2.remaining


# ---------------------------------------------------------------------------
# GCRA
# ---------------------------------------------------------------------------


class TestGCRA:
    def test_allows_within_rate(self):
        clock = FakeClock()
        limiter = GCRALimiter(rate=10.0, burst=0, clock=clock)
        r = limiter.acquire("k")
        assert r.allowed is True

    def test_denies_over_rate(self):
        clock = FakeClock()
        limiter = GCRALimiter(rate=1.0, burst=0, clock=clock)
        limiter.acquire("k")
        r = limiter.acquire("k")
        assert r.allowed is False
        assert r.retry_after is not None
        assert r.retry_after > 0

    def test_burst_tolerance(self):
        clock = FakeClock()
        limiter = GCRALimiter(rate=1.0, burst=3, clock=clock)
        for i in range(4):
            r = limiter.acquire("k")
            assert r.allowed, f"request {i} should be allowed"
        r = limiter.acquire("k")
        assert r.allowed is False

    def test_recovers_after_wait(self):
        clock = FakeClock()
        limiter = GCRALimiter(rate=2.0, burst=0, clock=clock)
        limiter.acquire("k")
        assert limiter.acquire("k").allowed is False
        clock.advance(0.5)
        assert limiter.acquire("k").allowed is True

    def test_key_isolation(self):
        clock = FakeClock()
        limiter = GCRALimiter(rate=1.0, burst=0, clock=clock)
        limiter.acquire("a")
        assert limiter.acquire("a").allowed is False
        assert limiter.acquire("b").allowed is True

    def test_peek_does_not_consume(self):
        clock = FakeClock()
        limiter = GCRALimiter(rate=1.0, burst=2, clock=clock)
        r1 = limiter.peek("k")
        r2 = limiter.peek("k")
        assert r1.remaining == r2.remaining


# ---------------------------------------------------------------------------
# Convenience API
# ---------------------------------------------------------------------------


class TestConvenienceAPI:
    def test_parse_quota(self):
        q = parse_quota("10/s")
        assert q["limit"] == 10
        assert q["period"] == 1.0

    def test_parse_quota_with_burst(self):
        q = parse_quota("100/m burst 20")
        assert q["limit"] == 100
        assert q["period"] == 60.0
        assert q["burst"] == 20

    def test_create_limiter_token_bucket(self):
        limiter = create_limiter("token_bucket", "10/s")
        assert isinstance(limiter, TokenBucketLimiter)

    def test_create_limiter_fixed_window(self):
        limiter = create_limiter("fixed_window", "10/m")
        assert isinstance(limiter, FixedWindowLimiter)

    def test_create_limiter_sliding_window(self):
        limiter = create_limiter("sliding_window", "10/m")
        assert isinstance(limiter, SlidingWindowLimiter)

    def test_create_limiter_gcra(self):
        limiter = create_limiter("gcra", "5/s burst 3")
        assert isinstance(limiter, GCRALimiter)

    def test_create_limiter_unknown_raises(self):
        with pytest.raises(ValueError):
            create_limiter("unknown", "1/s")


# ---------------------------------------------------------------------------
# Eviction
# ---------------------------------------------------------------------------


class TestEviction:
    def test_token_bucket_evicts_stale_keys(self):
        clock = FakeClock()
        limiter = TokenBucketLimiter(rate=1.0, capacity=2, clock=clock)
        for i in range(200):
            limiter.acquire(f"k{i}")
        initial_count = len(limiter._buckets)
        assert initial_count == 200

        clock.advance(100.0)
        for _ in range(128):
            limiter.acquire("trigger")
        assert len(limiter._buckets) < initial_count

    def test_fixed_window_evicts_stale_keys(self):
        clock = FakeClock()
        limiter = FixedWindowLimiter(limit=10, window_seconds=5.0, clock=clock)
        for i in range(200):
            limiter.acquire(f"k{i}")
        assert len(limiter._windows) == 200

        clock.advance(15.0)
        for _ in range(128):
            limiter.acquire("trigger")
        assert len(limiter._windows) < 200

    def test_sliding_window_evicts_stale_keys(self):
        clock = FakeClock()
        limiter = SlidingWindowLimiter(limit=10, window_seconds=5.0, clock=clock)
        for i in range(200):
            limiter.acquire(f"k{i}")
        assert len(limiter._states) == 200

        clock.advance(20.0)
        for _ in range(128):
            limiter.acquire("trigger")
        assert len(limiter._states) < 200
