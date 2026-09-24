"""Per-request context extraction middleware for the gateway.

Consolidates duplicated context extraction (client IP, API format,
admin path detection, request ID) into a single ``before_request`` hook
that populates a :data:`request_context_var` contextvar.  Downstream
hooks and handlers read from the contextvar instead of re-extracting.

This module depends on :mod:`error_format` for ``detect_api_format``
and ``is_admin_path``, and on :mod:`headers` for ``get_request_id``.
"""

from __future__ import annotations

import contextvars
import time
from dataclasses import dataclass
from typing import Any

from .error_format import detect_api_format, is_admin_path
from .headers import get_request_id


# ---------------------------------------------------------------------------
# RequestContext dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RequestContext:
    """Immutable per-request context populated by the early middleware hook.

    Attributes:
        request_id: Unique identifier for the request, either extracted
            from the ``X-Request-Id`` header or auto-generated.
        client_ip: Best-effort client IP address.  When *trust_proxy*
            is enabled, proxy headers (``X-Forwarded-For``,
            ``X-Real-IP``) take precedence over the TCP peer address.
        api_format: Detected API format based on the request path
            (``"openai"``, ``"anthropic"``, ``"google"``), or ``None``
            for paths where format detection is not meaningful (e.g.
            admin, health).
        is_admin: Whether the request targets an admin panel path.
        request_start: Monotonic timestamp captured when the request
            context is created, used to compute gateway-level TTFB.
    """

    request_id: str
    client_ip: str
    api_format: str | None
    is_admin: bool
    request_start: float = 0.0


# ---------------------------------------------------------------------------
# Contextvar
# ---------------------------------------------------------------------------

request_context_var: contextvars.ContextVar[RequestContext | None] = (
    contextvars.ContextVar("request_context", default=None)
)


# ---------------------------------------------------------------------------
# Client IP extraction (canonical implementation)
# ---------------------------------------------------------------------------


def extract_client_ip(request: Any, *, trust_proxy: bool = True) -> str:
    """Extract the client IP from *request*.

    When *trust_proxy* is ``True`` (the default), ``X-Forwarded-For``
    and ``X-Real-IP`` headers are checked first.  When ``False``, only
    the TCP peer address is used — appropriate for deployments where
    the gateway faces the internet directly and proxy headers cannot
    be trusted.

    Args:
        request: The incoming HTTP request object.
        trust_proxy: Whether to trust reverse-proxy headers for client
            IP determination.

    Returns:
        The client IP as a string.  Falls back to ``"unknown"`` when no
        address can be determined.
    """
    if trust_proxy:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            # X-Forwarded-For may contain a chain: "client, proxy1, proxy2"
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
    addr = getattr(request, "client_addr", None)
    if addr and isinstance(addr, (tuple, list)) and addr[0]:
        return str(addr[0])
    return "unknown"


# ---------------------------------------------------------------------------
# Context setup hook
# ---------------------------------------------------------------------------


def setup_request_context(*, trust_proxy: bool = True) -> Any:
    """Return a ``before_request`` hook that populates :data:`request_context_var`.

    The returned hook should be registered as the **earliest**
    ``before_request`` handler so that auth, rate-limiting, and all
    downstream code can read the context.

    Args:
        trust_proxy: Forwarded to :func:`extract_client_ip`.

    Returns:
        An async callable suitable for ``app.before_request()``.
    """

    async def _setup_request_context_hook(request: Any) -> None:
        path: str = request.path
        admin = is_admin_path(path)

        # For admin and public/health paths the api_format is not
        # meaningful — set to None so consumers can distinguish.
        api_format: str | None = None if admin else detect_api_format(path)

        ctx = RequestContext(
            request_id=get_request_id(request),
            client_ip=extract_client_ip(request, trust_proxy=trust_proxy),
            api_format=api_format,
            is_admin=admin,
            request_start=time.monotonic(),
        )
        request_context_var.set(ctx)

    return _setup_request_context_hook
