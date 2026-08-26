"""Rate-limiting middleware for the gateway.

Re-exports the core rate-limiter classes from the vendored ``zerodep``
ratelimit module, and provides the gateway-specific middleware hooks
that enforce per-IP, per-model, or per-key request quotas.

Usage (standalone)::

    from llm_rosetta.gateway.ratelimit import TokenBucketLimiter

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

from llm_rosetta._vendor.httpserver import JSONResponse, Response
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

if TYPE_CHECKING:
    from .config import GatewayConfig

logger = logging.getLogger("llm-rosetta-gateway")

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
    "RateLimitState",
    "create_rate_limit_hook",
    "create_rate_limit_after_hook",
]

# Per-request rate limit result for the after-hook to attach headers.
_rate_limit_result_var: contextvars.ContextVar[RateLimitResult | None] = (
    contextvars.ContextVar("_rate_limit_result", default=None)
)

# Route prefix → API format (mirrors auth.py's _ROUTE_EXTRACTORS)
_ROUTE_FORMATS: list[tuple[str, str]] = [
    ("/v1beta/models", "google"),
    ("/v1/messages", "anthropic"),
    ("/v1/", "openai"),
]

_GOOGLE_MODEL_RE = re.compile(r"/v1beta/models/([^/:]+)")


# ---------------------------------------------------------------------------
# RateLimitState — mutable container for hot-reloadable limiters
# ---------------------------------------------------------------------------


def _build_limiter(algorithm: str, quota: str | None) -> RateLimiter | None:
    if not quota:
        return None
    return ThreadSafeLimiter(create_limiter(algorithm, quota))


class RateLimitState:
    """Holds the active rate limiters, swappable on config reload."""

    __slots__ = (
        "enabled",
        "exclude_prefixes",
        "_global",
        "_per_ip",
        "_per_key",
        "_per_model",
    )

    def __init__(self) -> None:
        self.enabled: bool = False
        self.exclude_prefixes: list[str] = ["/health", "/admin"]
        self._global: RateLimiter | None = None
        self._per_ip: RateLimiter | None = None
        self._per_key: RateLimiter | None = None
        self._per_model: RateLimiter | None = None

    def rebuild(self, config: GatewayConfig) -> None:
        """Recreate limiters from the current config (counters reset)."""
        self.enabled = config.rate_limit_enabled
        self.exclude_prefixes = list(config.rate_limit_exclude)
        algo = config.rate_limit_algorithm
        self._global = _build_limiter(algo, config.rate_limit_global)
        self._per_ip = _build_limiter(algo, config.rate_limit_per_ip)
        self._per_key = _build_limiter(algo, config.rate_limit_per_key)
        self._per_model = _build_limiter(algo, config.rate_limit_per_model)
        if self.enabled:
            dims = []
            if self._global:
                dims.append("global")
            if self._per_ip:
                dims.append("per-ip")
            if self._per_key:
                dims.append("per-key")
            if self._per_model:
                dims.append("per-model")
            logger.info(
                "Rate limiting enabled (%s, dimensions: %s)",
                algo,
                ", ".join(dims) or "none",
            )


# ---------------------------------------------------------------------------
# Format-aware 429 response
# ---------------------------------------------------------------------------


def _detect_format(path: str) -> str:
    for prefix, fmt in _ROUTE_FORMATS:
        if path.startswith(prefix):
            return fmt
    return "openai"


def _rate_limit_response(
    path: str, result: RateLimitResult, dimension: str
) -> Response:
    """Build a format-aware 429 response with standard rate-limit headers."""
    retry_secs = math.ceil(result.retry_after or 1)
    message = f"Rate limit exceeded ({dimension}). Please retry after {retry_secs}s."
    fmt = _detect_format(path)

    if fmt == "anthropic":
        body = {
            "type": "error",
            "error": {"type": "rate_limit_error", "message": message},
        }
    elif fmt == "google":
        body = {
            "error": {
                "code": 429,
                "message": message,
                "status": "RESOURCE_EXHAUSTED",
            }
        }
    else:
        body = {
            "error": {
                "message": message,
                "type": "rate_limit_error",
                "code": "rate_limit_exceeded",
            }
        }

    resp = JSONResponse(body, status_code=429)
    resp.headers["Retry-After"] = str(retry_secs)
    resp.headers["X-RateLimit-Limit"] = str(int(result.limit))
    resp.headers["X-RateLimit-Remaining"] = str(max(0, int(result.remaining)))
    resp.headers["X-RateLimit-Reset"] = str(int(math.ceil(result.reset_at)))

    if not (path.startswith("/admin/") or path == "/admin"):
        resp.headers["Access-Control-Allow-Origin"] = "*"
        resp.headers["Access-Control-Allow-Methods"] = "*"
        resp.headers["Access-Control-Allow-Headers"] = "*"

    return resp


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
# Client IP extraction
# ---------------------------------------------------------------------------


def _extract_client_ip(request: Any) -> str:
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    real_ip = request.headers.get("x-real-ip")
    if real_ip:
        return real_ip.strip()
    return request.client_addr[0] if request.client_addr else "unknown"


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
    """Acquire from a single limiter; return (429 response, updated tightest)."""
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

        tightest: RateLimitResult | None = None

        denied, tightest = _check_limiter(
            state._global, "__global__", "global", path, tightest
        )
        if denied:
            return denied

        denied, tightest = _check_limiter(
            state._per_ip, _extract_client_ip(request), "per_ip", path, tightest
        )
        if denied:
            return denied

        ctx = api_key_context_var.get()
        if ctx is not None:
            denied, tightest = _check_limiter(
                state._per_key, ctx.label, "per_key", path, tightest
            )
            if denied:
                return denied

        model = _extract_model(request)
        if model:
            denied, tightest = _check_limiter(
                state._per_model, model, "per_model", path, tightest
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
