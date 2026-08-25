"""Rate-limiting support for the gateway.

Re-exports the core rate-limiter classes from the vendored ``zerodep``
ratelimit module.  Gateway middleware will use these to enforce
per-IP, per-model, or per-key request quotas.

Usage::

    from llm_rosetta.gateway.ratelimit import TokenBucketLimiter

    limiter = TokenBucketLimiter(rate=10.0, capacity=20)
    result = limiter.acquire("client-ip-1.2.3.4")
    if not result.allowed:
        return 429, {"Retry-After": str(result.retry_after)}
"""

from __future__ import annotations

from llm_rosetta._vendor.ratelimit import (
    FixedWindowLimiter,
    GCRALimiter,
    RateLimitExceeded,
    RateLimiter,
    RateLimitResult,
    SlidingWindowLimiter,
    ThreadSafeLimiter,
    TokenBucketLimiter,
    create_limiter,
    parse_quota,
    ratelimit,
)

__all__ = [
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
