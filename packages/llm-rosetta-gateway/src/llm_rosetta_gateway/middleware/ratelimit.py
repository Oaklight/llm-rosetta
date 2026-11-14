"""Rate-limiting middleware for the gateway.

Re-exports the core rate-limiter classes from the vendored ``zerodep``
ratelimit module, and provides the gateway-specific middleware hooks
that enforce per-IP, per-model, or per-key request quotas.

Usage (standalone)::

    from llm_rosetta_gateway.middleware.ratelimit import TokenBucketLimiter

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
import time
from typing import TYPE_CHECKING, Any

from llm_rosetta_gateway._vendor.httpserver import Response

from .error_format import detect_api_format, format_error_response, is_admin_path
from .request_context import request_context_var

from llm_rosetta_gateway._vendor.ratelimit import (
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

    __slots__ = ("enabled", "exclude_prefixes", "trust_proxy", "_snap", "_key_limits")

    def __init__(self) -> None:
        self.enabled: bool = False
        self.exclude_prefixes: list[str] = ["/health", "/admin"]
        self.trust_proxy: bool = False
        self._snap: _LimiterSnapshot = _LimiterSnapshot(None, None, None, None)
        #: (key identity, quota spec) → its windows, when they were last used
        #: and how long the longest one runs.  A key whose limit changes gets
        #: fresh windows rather than inheriting the old counts; a key that
        #: stops calling is pruned once its own window has closed.
        self._key_limits: dict[tuple[str, str], tuple[RateLimiter, float, float]] = {}

    #: How long an idle key's windows are kept.  Pruning is what keeps a
    #: deleted key from holding a limiter for the process lifetime.
    KEY_LIMIT_IDLE_S = 3600.0

    def key_limiter(self, ctx: Any) -> RateLimiter:
        """The windows for one key's own limit, built on first use.

        The spec is the settings fields' ``N/unit`` form and may name several
        windows (``"30/m, 500/h"``); each is a fixed window, so a key's limit
        reads the same whatever algorithm the deployment-level limiting runs.
        A spec with ``burst`` keeps only its limit: a fixed window has no
        burst to spend.
        """
        now = time.monotonic()
        cache_key = (ctx.key_hash or ctx.label, ctx.rate_limit)
        entry = self._key_limits.get(cache_key)
        if entry is None:
            limiters = []
            periods: list[float] = []
            for part in str(ctx.rate_limit).split(","):
                part = part.strip()
                if not part:
                    continue
                parsed = parse_quota(part)  # once per part
                periods.append(float(parsed["period"]))
                limiters.append(
                    FixedWindowLimiter(
                        limit=int(parsed["limit"]),
                        window_seconds=float(parsed["period"]),
                    )
                )
            limiter: RateLimiter = ThreadSafeLimiter(
                limiters[0] if len(limiters) == 1 else CompositeLimiter(limiters)
            )
            entry = (limiter, now, max(periods))
            self._key_limits[cache_key] = entry
            self._prune_key_limits(now)
        else:
            self._key_limits[cache_key] = (entry[0], now, entry[2])
        return entry[0]

    def _prune_key_limits(self, now: float) -> None:
        """Drop the windows of keys whose own window closed while they idled.

        ``KEY_LIMIT_IDLE_S`` is a floor, not the rule: a ``2/d`` key is idle
        for most of its window, and dropping those counters would hand the
        cap back.  An entry goes only once its longest window has rolled over.
        """
        stale = [
            key
            for key, (_, seen, window) in self._key_limits.items()
            if now - seen > max(self.KEY_LIMIT_IDLE_S, window)
        ]
        for key in stale:
            del self._key_limits[key]

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
        self._prune_key_limits(time.monotonic())
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

    def snapshot(self, key: str = "__global__") -> dict[str, Any]:
        """Return introspection data for all rate limit dimensions.

        Args:
            key: The limiter key to peek.  Defaults to ``"__global__"``.
        """
        snap = self._snap

        def _limiter_info(lim: RateLimiter | None, key: str) -> dict[str, Any] | None:
            if lim is None:
                return None
            inner = lim.limiter if isinstance(lim, ThreadSafeLimiter) else lim
            if isinstance(inner, CompositeLimiter):
                details = inner.detail(key)
                return {
                    "windows": [
                        {
                            "limit": int(r.limit),
                            "remaining": max(0, int(r.remaining)),
                            "reset_at": round(r.reset_at, 2),
                            "allowed": r.allowed,
                        }
                        for r in details
                    ],
                }
            result = lim.peek(key)
            return {
                "limit": int(result.limit),
                "remaining": max(0, int(result.remaining)),
                "reset_at": round(result.reset_at, 2),
                "allowed": result.allowed,
            }

        return {
            "enabled": self.enabled,
            "dimensions": {
                "global": _limiter_info(snap.gl, "__global__"),
                "per_ip": _limiter_info(snap.ip, key),
                "per_key": _limiter_info(snap.key, key),
                "per_model": _limiter_info(snap.model, key),
            },
        }


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


def _check_key_quota(state: RateLimitState, path: str) -> Response | None:
    """The 429 a key's own quota earns, or ``None``.

    A key's limit is a property of the key, not of the deployment: it applies
    whether or not global limiting is on, which is why it is checked before
    the switch.
    """
    from .auth import api_key_context_var

    ctx = api_key_context_var.get()
    if ctx is None or not ctx.rate_limit:
        return None
    return _check_limiter(
        state.key_limiter(ctx),
        ctx.key_hash or ctx.label,
        "per_key_record",
        path,
        None,
    )[0]


def create_rate_limit_hook(
    state: RateLimitState,
) -> Any:
    """Return a before-request handler that enforces rate limits.

    Must be registered *after* the auth hook so that
    ``api_key_context_var`` is already populated.
    """
    from .auth import api_key_context_var

    async def rate_limit_hook(request: Any) -> Response | None:
        if request.method == "OPTIONS":
            return None

        path = request.path
        for prefix in state.exclude_prefixes:
            if path.startswith(prefix):
                return None

        denied = _check_key_quota(state, path)
        if denied:
            return denied

        if not state.enabled:
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
