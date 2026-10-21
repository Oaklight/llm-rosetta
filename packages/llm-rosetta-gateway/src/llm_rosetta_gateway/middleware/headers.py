"""Helpers for gateway request header forwarding."""

from __future__ import annotations

from typing import Any
import uuid

HOP_COUNT_HEADER = "x-rosetta-hop-count"
# Generous for legitimate chained gateways, catches infinite loops within a few round-trips.
MAX_HOPS = 4


def get_request_id(request: Any) -> str:
    """Return the client request ID or generate one."""
    return request.headers.get("x-request-id") or str(uuid.uuid4())


def get_hop_count(request: Any) -> int:
    """Return the current hop count from the inbound request."""
    raw = request.headers.get(HOP_COUNT_HEADER)
    if raw is None:
        return 0
    try:
        return max(0, int(raw))
    except (ValueError, TypeError):
        return 0


def build_upstream_extra_headers(
    request: Any,
    request_id: str,
    session_header: str | None = None,
) -> dict[str, str]:
    """Build the explicit request headers that may be forwarded upstream.

    When *session_header* is set (a shim-declared per-conversation header, e.g.
    ``x-opencode-session``) the client's value for it is forwarded verbatim,
    overriding the gateway-derived one in the transport.
    """
    extra_headers: dict[str, str] = {}

    if request_id:
        extra_headers["x-request-id"] = request_id

    # Propagate hop count, incremented by one for this hop.
    extra_headers[HOP_COUNT_HEADER] = str(get_hop_count(request) + 1)

    user_agent = request.headers.get("user-agent")
    if user_agent:
        extra_headers["User-Agent"] = user_agent

    or_version = request.headers.get("openresponses-version")
    if or_version:
        extra_headers["OpenResponses-Version"] = or_version

    if session_header:
        client_session = request.headers.get(session_header)
        if client_session:
            extra_headers[session_header] = client_session

    return extra_headers


def get_preflight_tokens_override(request: Any) -> bool | None:
    """Return the per-request preflight token count override, or None."""
    value = request.headers.get("x-rosetta-preflight-tokens")
    if value is None:
        return None
    return value.strip().lower() in ("true", "1", "yes")
