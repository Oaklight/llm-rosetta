"""HTTP client pool and upstream request construction.

Manages a shared pool of :class:`AsyncClient` instances (keyed by proxy
URL) and provides :func:`prepare_upstream` to assemble the URL, headers,
and body for an upstream provider call.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta._vendor.httpclient import AsyncClient
from llm_rosetta.auto_detect import ProviderType

from .provider_info import ProviderInfo

# Shared HTTP clients keyed by proxy URL (None = direct connection)
_http_clients: dict[str | None, AsyncClient] = {}


def get_client(proxy_url: str | None = None) -> AsyncClient:
    """Get or create an ``AsyncClient`` for the given proxy URL."""
    if proxy_url not in _http_clients:
        _http_clients[proxy_url] = AsyncClient(
            timeout=300.0,
            proxy=proxy_url,
        )
    return _http_clients[proxy_url]


async def close_clients() -> None:
    """Close all pooled HTTP clients."""
    for client in _http_clients.values():
        await client.aclose()
    _http_clients.clear()


def prepare_upstream(
    target_provider: ProviderType,
    provider_info: ProviderInfo,
    provider_request: dict[str, Any],
    model: str,
    *,
    stream: bool,
    extra_headers: dict[str, str] | None = None,
) -> tuple[str, dict[str, str], dict[str, Any]]:
    """Return ``(url, headers, body)`` ready for the upstream HTTP call."""
    url = provider_info.upstream_url(model, stream=stream)
    headers = {
        "Content-Type": "application/json",
        **provider_info.auth_headers(),
    }
    if extra_headers:
        headers.update(extra_headers)

    body = dict(provider_request)

    # Inject stream flag into the body for providers that use it
    if stream:
        if target_provider in ("openai_chat",):
            body["stream"] = True
            body["stream_options"] = {"include_usage": True}
        elif target_provider in ("openai_responses", "open_responses", "anthropic"):
            body["stream"] = True
        # Google streaming is signaled via URL, not body

    return url, headers, body
