"""Helpers for gateway request header forwarding."""

from __future__ import annotations

from typing import Any


def build_upstream_extra_headers(request: Any, request_id: str) -> dict[str, str]:
    """Build the explicit request headers that may be forwarded upstream."""
    extra_headers: dict[str, str] = {}

    if request_id:
        extra_headers["x-request-id"] = request_id

    user_agent = request.headers.get("user-agent")
    if user_agent:
        extra_headers["User-Agent"] = user_agent

    or_version = request.headers.get("openresponses-version")
    if or_version:
        extra_headers["OpenResponses-Version"] = or_version

    return extra_headers
