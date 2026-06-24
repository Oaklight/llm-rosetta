"""HTTP client pool — manages :class:`AsyncClient` instances.

Thin wrapper over :mod:`llm_rosetta._vendor.httpclient` that pools
``AsyncClient`` instances by proxy URL so multiple requests to the
same upstream reuse the same connection pool.
"""

from __future__ import annotations

from llm_rosetta._vendor.httpclient import AsyncClient


class HttpClientPool:
    """Manages :class:`AsyncClient` instances keyed by proxy URL.

    Each unique ``proxy_url`` (including ``None`` for direct connections)
    gets its own ``AsyncClient`` with connection pooling.
    """

    def __init__(self, *, timeout: float = 300.0) -> None:
        self._clients: dict[str | None, AsyncClient] = {}
        self._timeout = timeout

    def get(self, proxy_url: str | None = None) -> AsyncClient:
        """Get or create an ``AsyncClient`` for the given proxy URL."""
        if proxy_url not in self._clients:
            self._clients[proxy_url] = AsyncClient(
                timeout=self._timeout,
                proxy=proxy_url,
            )
        return self._clients[proxy_url]

    async def close_all(self) -> None:
        """Close all pooled HTTP clients."""
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()
