"""Rate-limiting middleware for the gateway.

Re-exports the core rate-limiter classes from the vendored ``zerodep``
ratelimit module, and provides the gateway-specific middleware hooks
that enforce per-IP, per-model, or per-key request quotas.

Usage (standalone)::

    from llm_rosetta.gateway.middleware.ratelimit import TokenBucketLimiter

    limiter = TokenBucketLimiter(rate=10.0, capacity=20)
    result = limiter.acquire("client-ip-1.2.3.4")
    if not result.allowed:
        return 429, {"Retry-After": str(result.retry_after)}

Usage (gateway integration)::

    state = RateLimitState()
    state.rebuild(config)
    app.before_request(create_rate_limit_hook(state))
    app.after_request(create_rate_limit_after_hook())
"""

from __future__ import annotations

import contextvars
import logging
import math
import re
from typing import TYPE_CHECKING, Any

from llm_rosetta._vendor.httpserver import Response

from .error_format import detect_api_format, format_error_response, is_admin_path
from .request_context import request_context_var

from llm_rosetta._vendor.ratelimit import (
    CompositeLimiter,
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

if TYPE_CHECKING:
    from ..config import GatewayConfig

logger = logging.getLogger("llm-rosetta-gateway")

__all__ = [
    "RateLimitResult",
    "RateLimiter",
    "TokenBucketLimiter",
    "FixedWindowLimiter",
    "SlidingWindowLimiter",
    "GCRALimiter",
    "CompositeLimiter",
    "ThreadSafeLimiter",
    "RateLimitExceeded",
    "ratelimit",
    "create_limiter",
    "parse_quota",
    "RateLimitState",
    "create_rate_limit_hook",
    "create_rate_limit_after_hook",
]

# Per-request rate limit result for the after-hook to attach headers.
_rate_limit_result_var: contextvars.ContextVar[RateLimitResult | None] = (
    contextvars.ContextVar("_rate_limit_result", default=None)
)

_GOOGLE_MODEL_RE = re.compile(r"/v1beta/models/([^/:]+)")


# ---------------------------------------------------------------------------
# RateLimitState — mutable container for hot-reloadable limiters
# ---------------------------------------------------------------------------


def _build_limiter(algorithm: str, quota: str | list[str] | None) -> RateLimiter | None:
    if not quota:
        return None
    if isinstance(quota, list):
        limiters = [create_limiter(algorithm, q) for q in quota]
        return ThreadSafeLimiter(CompositeLimiter(limiters))
    return ThreadSafeLimiter(create_limiter(algorithm, quota))


class _LimiterSnapshot:
    """Immutable snapshot of limiter state — assigned atomically."""

    __slots__ = ("gl", "ip", "key", "model")

    def __init__(
        self,
        gl: RateLimiter | None,
        ip: RateLimiter | None,
        key: RateLimiter | None,
        model: RateLimiter | None,
    ) -> None:
        self.gl = gl
        self.ip = ip
        self.key = key
        self.model = model


class RateLimitState:
    """Holds the active rate limiters, swappable on config reload.

    The hook reads ``_snap`` — a single reference that is replaced
    atomically by ``rebuild()``, so concurrent requests never see a
    mix of old and new limiters.
    """

    __slots__ = ("enabled", "exclude_prefixes", "trust_proxy", "_snap")

    def __init__(self) -> None:
        self.enabled: bool = False
        self.exclude_prefixes: list[str] = ["/health", "/admin"]
        self.trust_proxy: bool = False
        self._snap: _LimiterSnapshot = _LimiterSnapshot(None, None, None, None)

    def rebuild(self, config: GatewayConfig) -> None:
        """Recreate limiters from the current config (counters reset)."""
        algo = config.rate_limit_algorithm
        snap = _LimiterSnapshot(
            gl=_build_limiter(algo, config.rate_limit_global),
            ip=_build_limiter(algo, config.rate_limit_per_ip),
            key=_build_limiter(algo, config.rate_limit_per_key),
            model=_build_limiter(algo, config.rate_limit_per_model),
        )
        # Single reference assignment — atomic under the GIL.
        self._snap = snap
        self.exclude_prefixes = list(config.rate_limit_exclude)
        self.trust_proxy = config.rate_limit_trust_proxy
        self.enabled = config.rate_limit_enabled
        if self.enabled:
            dims = []
            if snap.gl:
                dims.append("global")
            if snap.ip:
                dims.append("per-ip")
            if snap.key:
                dims.append("per-key")
            if snap.model:
                dims.append("per-model")
            logger.info(
                "Rate limiting enabled (%s, dimensions: %s)",
                algo,
                ", ".join(dims) or "none",
            )


# ---------------------------------------------------------------------------
# Format-aware 429 response
# ---------------------------------------------------------------------------


def _rate_limit_response(
    path: str, result: RateLimitResult, dimension: str
) -> Response:
    """Build a format-aware 429 response with standard rate-limit headers."""
    retry_secs = math.ceil(max(result.retry_after or 0, 1))
    message = f"Rate limit exceeded ({dimension}). Please retry after {retry_secs}s."

    return format_error_response(
        detect_api_format(path),
        429,
        message,
        error_type="rate_limit_error",
        error_code="rate_limit_exceeded",
        google_status="RESOURCE_EXHAUSTED",
        extra_headers={
            "Retry-After": str(retry_secs),
            "X-RateLimit-Limit": str(int(result.limit)),
            "X-RateLimit-Remaining": str(max(0, int(result.remaining))),
            "X-RateLimit-Reset": str(int(math.ceil(result.reset_at))),
        },
        cors=not is_admin_path(path),
    )


# ---------------------------------------------------------------------------
# Model extraction (for per-model limiting)
# ---------------------------------------------------------------------------


def _extract_model(request: Any) -> str | None:
    """Extract model name from the request path or body."""
    m = _GOOGLE_MODEL_RE.search(request.path)
    if m:
        return m.group(1)
    try:
        return request.json().get("model")
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Before-request hook
# ---------------------------------------------------------------------------


def _check_limiter(
    limiter: RateLimiter | None,
    key: str,
    dimension: str,
    path: str,
    tightest: RateLimitResult | None,
) -> tuple[Response | None, RateLimitResult | None]:
    """Acquire from a single limiter; return (429 response, updated tightest).

    Each acquire() consumes quota even if a later dimension denies the
    request.  This is intentional — global quota reflects total server
    load including rejected requests from individual dimensions.
    """
    if limiter is None:
        return None, tightest
    result = limiter.acquire(key)
    if not result.allowed:
        return _rate_limit_response(path, result, dimension), tightest
    if tightest is None or result.remaining < tightest.remaining:
        tightest = result
    return None, tightest


def create_rate_limit_hook(
    state: RateLimitState,
) -> Any:
    """Return a before-request handler that enforces rate limits.

    Must be registered *after* the auth hook so that
    ``api_key_context_var`` is already populated.
    """
    from .auth import api_key_context_var

    async def rate_limit_hook(request: Any) -> Response | None:
        if not state.enabled:
            return None

        if request.method == "OPTIONS":
            return None

        path = request.path
        for prefix in state.exclude_prefixes:
            if path.startswith(prefix):
                return None

        # Read snapshot once — atomic under the GIL.
        snap = state._snap
        tightest: RateLimitResult | None = None

        denied, tightest = _check_limiter(
            snap.gl, "__global__", "global", path, tightest
        )
        if denied:
            return denied

        rctx = request_context_var.get()
        client_ip = rctx.client_ip if rctx else "unknown"
        denied, tightest = _check_limiter(
            snap.ip,
            client_ip,
            "per_ip",
            path,
            tightest,
        )
        if denied:
            return denied

        ctx = api_key_context_var.get()
        if ctx is not None:
            denied, tightest = _check_limiter(
                snap.key, ctx.label, "per_key", path, tightest
            )
            if denied:
                return denied

        model = _extract_model(request)
        if model:
            denied, tightest = _check_limiter(
                snap.model, model, "per_model", path, tightest
            )
            if denied:
                return denied

        _rate_limit_result_var.set(tightest)
        return None

    return rate_limit_hook


# ---------------------------------------------------------------------------
# After-request hook — attach rate-limit headers to successful responses
# ---------------------------------------------------------------------------


def create_rate_limit_after_hook() -> Any:
    """Return an after-request handler that adds X-RateLimit-* headers."""

    async def rate_limit_after_hook(request: Any, response: Any) -> Any:
        result = _rate_limit_result_var.get()
        if result is None:
            return response
        response.headers.setdefault("X-RateLimit-Limit", str(int(result.limit)))
        response.headers.setdefault(
            "X-RateLimit-Remaining", str(max(0, int(result.remaining)))
        )
        response.headers.setdefault(
            "X-RateLimit-Reset", str(int(math.ceil(result.reset_at)))
        )
        return response

    return rate_limit_after_hook
