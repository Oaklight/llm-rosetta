"""Hop-count middleware to prevent routing loops.

Checks the ``X-Rosetta-Hop-Count`` header on every inbound request.
If the count meets or exceeds ``MAX_HOPS``, the request is rejected
with 508 Loop Detected before it reaches any handler.

The outbound side (incrementing the header on upstream forwarding) lives
in :func:`~headers.build_upstream_extra_headers`.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta._vendor.httpserver import JSONResponse

from .headers import MAX_HOPS, get_hop_count
from .request_context import request_context_var
from ..logging import get_logger

logger = get_logger()

_ADMIN_PREFIX = "/admin/"
_HEALTH_PATHS = frozenset({"/health", "/ready", "/healthz"})


def create_hop_limit_hook() -> Any:
    """Return a ``before_request`` hook that enforces the hop-count limit.

    Admin and health-check paths are excluded — they are never
    forwarded upstream and cannot participate in loops.
    """

    async def _hop_limit_hook(request: Any) -> JSONResponse | None:
        path: str = request.path
        if path.startswith(_ADMIN_PREFIX) or path in _HEALTH_PATHS:
            return None

        hop_count = get_hop_count(request)
        if hop_count < MAX_HOPS:
            return None

        rctx = request_context_var.get()
        request_id = rctx.request_id if rctx else "?"
        logger.warning(
            "[%s] Loop detected: hop count %d >= %d, returning 508",
            request_id,
            hop_count,
            MAX_HOPS,
        )
        return JSONResponse(
            {
                "error": {
                    "message": (
                        f"Loop detected: request has been forwarded"
                        f" {hop_count} times (max {MAX_HOPS})"
                    ),
                    "type": "loop_detected",
                }
            },
            status_code=508,
        )

    return _hop_limit_hook
