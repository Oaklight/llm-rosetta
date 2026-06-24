"""HTTP/SSE transport implementation.

Implements the :class:`~transport._base.UpstreamTransport` protocol for
HTTP REST + SSE streaming, backed by the vendored ``httpclient`` and ``sse``
modules.
"""

from __future__ import annotations

import json
import logging
from collections.abc import AsyncIterator
from typing import Any

from llm_rosetta._vendor.httpclient import (
    HttpClientError,
    Response as HttpResponse,
    StreamingResponse as HttpStreamingResponse,
)
from llm_rosetta._vendor.sse import AsyncEventSource
from llm_rosetta.auto_detect import ProviderType

from .._base import UpstreamConnectionError, UpstreamResponse, UpstreamStream
from ..provider_info import ProviderInfo
from .client_pool import HttpClientPool

logger = logging.getLogger("llm-rosetta-gateway")


# ---------------------------------------------------------------------------
# Upstream request assembly
# ---------------------------------------------------------------------------


def _prepare_upstream(
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

    async def send_request(
        self,
        provider_info: ProviderInfo,
        target_provider: ProviderType,
        body: dict[str, Any],
        model: str,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> UpstreamResponse:
        """Send a non-streaming request and return the full response."""
        url, headers, req_body = _prepare_upstream(
            target_provider,
            provider_info,
            body,
            model,
            stream=False,
            extra_headers=extra_headers,
        )
        client = self._pool.get(provider_info.proxy_url)
        try:
            resp = await client.post(url, json=req_body, headers=headers)
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
        target_provider: ProviderType,
        body: dict[str, Any],
        model: str,
        *,
        extra_headers: dict[str, str] | None = None,
    ) -> HttpUpstreamStream:
        """Send a streaming request and return an async chunk iterator."""
        url, headers, req_body = _prepare_upstream(
            target_provider,
            provider_info,
            body,
            model,
            stream=True,
            extra_headers=extra_headers,
        )
        client = self._pool.get(provider_info.proxy_url)
        try:
            resp = await client.post(url, json=req_body, headers=headers, stream=True)
        except HttpClientError as exc:
            raise UpstreamConnectionError(str(exc)) from exc

        assert isinstance(resp, HttpStreamingResponse)
        return HttpUpstreamStream(resp)

    async def close(self) -> None:
        """Close all pooled HTTP clients."""
        await self._pool.close_all()
