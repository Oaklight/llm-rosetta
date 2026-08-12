"""HTTP/SSE transport implementation.

Implements the :class:`~transport._base.UpstreamTransport` protocol for
HTTP REST + SSE streaming, backed by the vendored ``httpclient`` and ``sse``
modules.

The transport layer is URL-agnostic — callers construct the upstream URL
and inject any provider-specific body modifications (e.g. stream flags)
before calling :meth:`send` or :meth:`send_streaming`.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from llm_rosetta._vendor.httpclient import (
    HttpClientError,
    HttpTimeoutError,
    Response as HttpResponse,
    StreamingResponse as HttpStreamingResponse,
)
from llm_rosetta._vendor.sse import AsyncEventSource

from .._base import (
    UpstreamConnectionError,
    UpstreamResponse,
    UpstreamStream,
    UpstreamTimeoutError,
)
from ..provider_info import ProviderInfo
from .client_pool import HttpClientPool

logger = logging.getLogger("llm-rosetta-gateway")


# ---------------------------------------------------------------------------
# Streaming response wrapper
# ---------------------------------------------------------------------------


class HttpUpstreamStream(UpstreamStream):
    """Streaming response backed by HTTP/SSE.

    Wraps a :class:`~httpclient.StreamingResponse` and uses the vendored
    :class:`~sse.AsyncEventSource` to parse SSE events into JSON chunks.
    """

    def __init__(self, resp: HttpStreamingResponse) -> None:
        self.status_code = resp.status_code
        self._resp = resp

    async def read_error(self) -> str:
        """Read the error body as a string."""
        raw = await self._resp.aread()
        return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else raw

    async def __aiter__(self) -> AsyncIterator[dict[str, Any]]:  # type: ignore[override]
        """Yield parsed JSON chunks from the upstream SSE stream.

        Uses the vendored W3C-compliant SSE parser.  Detects OpenAI's
        ``[DONE]`` marker and stops iteration.
        """
        async for event in AsyncEventSource(self._resp.aiter_lines()):
            if event.data == "[DONE]":
                break
            try:
                yield json.loads(event.data)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed SSE data: %s", event.data[:200])

    async def close(self) -> None:
        """Close the underlying HTTP streaming response."""
        await self._resp.aclose()


# ---------------------------------------------------------------------------
# HttpTransport
# ---------------------------------------------------------------------------


class HttpTransport:
    """Upstream transport implementation for HTTP REST + SSE streaming.

    Implements the :class:`~transport._base.UpstreamTransport` protocol.
    """

    def __init__(self, *, timeout: float = 300.0) -> None:
        self._pool = HttpClientPool(timeout=timeout)

    def _build_headers(
        self,
        provider_info: ProviderInfo,
        extra_headers: dict[str, str] | None = None,
    ) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            **provider_info.auth_headers(),
        }
        if extra_headers:
            headers.update(extra_headers)
        return headers

    async def send(
        self,
        provider_info: ProviderInfo,
        url: str,
        body: dict[str, Any],
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> UpstreamResponse:
        """Send a non-streaming request and return the full response."""
        headers = self._build_headers(provider_info, extra_headers)
        client = self._pool.get(provider_info.proxy_url)
        try:
            kwargs: dict[str, Any] = {}
            if provider_info.timeout is not None:
                kwargs["timeout"] = provider_info.timeout
            resp = await client.post(url, json=body, headers=headers, **kwargs)
        except HttpTimeoutError as exc:
            raise UpstreamTimeoutError(str(exc)) from exc
        except HttpClientError as exc:
            raise UpstreamConnectionError(str(exc)) from exc

        assert isinstance(resp, HttpResponse)
        return UpstreamResponse(
            status_code=resp.status_code,
            body=resp.json() if resp.status_code < 400 else None,
            raw_content=resp.content,
        )

    async def send_streaming(
        self,
        provider_info: ProviderInfo,
        url: str,
        body: dict[str, Any],
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> HttpUpstreamStream:
        """Send a streaming request and return an async chunk iterator."""
        headers = self._build_headers(provider_info, extra_headers)
        client = self._pool.get(provider_info.proxy_url)
        try:
            kwargs: dict[str, Any] = {}
            if provider_info.timeout is not None:
                kwargs["timeout"] = provider_info.timeout
            resp = await client.post(
                url, json=body, headers=headers, stream=True, **kwargs
            )
        except HttpTimeoutError as exc:
            raise UpstreamTimeoutError(str(exc)) from exc
        except HttpClientError as exc:
            raise UpstreamConnectionError(str(exc)) from exc

        assert isinstance(resp, HttpStreamingResponse)
        return HttpUpstreamStream(resp)

    async def close(self) -> None:
        """Close all pooled HTTP clients."""
        await self._pool.close_all()
