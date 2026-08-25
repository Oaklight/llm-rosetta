"""Tests for redirect security: upstream redirects are blocked to prevent credential leakage."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest

from llm_rosetta._vendor.httpclient import TooManyRedirects
from llm_rosetta.gateway.transport.http.client_pool import HttpClientPool
from llm_rosetta.gateway.transport.http.transport import HttpTransport
from llm_rosetta.gateway.transport._base import UpstreamConnectionError
from llm_rosetta.gateway.transport.provider_info import ProviderInfo, openai_auth


def _test_provider() -> ProviderInfo:
    return ProviderInfo(
        name="test",
        api_key="sk-test-key-12345",
        base_url="https://api.example.com",
        auth_header_fn=openai_auth,
        url_template="{base_url}/v1/chat/completions",
    )


class TestClientPoolRedirects:
    # Intentionally access private attr — security-critical setting with no public accessor
    def test_pool_creates_client_with_zero_redirects(self):
        pool = HttpClientPool()
        client = pool.get()
        assert client._max_redirects == 0

    def test_pool_creates_proxy_client_with_zero_redirects(self):
        pool = HttpClientPool()
        client = pool.get("socks5://proxy:1080")
        assert client._max_redirects == 0


class TestTransportRedirectBlocking:
    def test_send_raises_on_redirect(self):
        transport = HttpTransport(timeout=10.0)
        provider = _test_provider()
        url = provider.upstream_url("gpt-4")

        async def run():
            with patch.object(
                transport._pool.get(provider.proxy_url),
                "post",
                side_effect=TooManyRedirects(url, 0),
            ):
                with pytest.raises(UpstreamConnectionError, match="redirect"):
                    await transport.send(provider, url, {"model": "gpt-4"})

        asyncio.run(run())

    def test_send_streaming_raises_on_redirect(self):
        transport = HttpTransport(timeout=10.0)
        provider = _test_provider()
        url = provider.upstream_url("gpt-4")

        async def run():
            with patch.object(
                transport._pool.get(provider.proxy_url),
                "post",
                side_effect=TooManyRedirects(url, 0),
            ):
                with pytest.raises(UpstreamConnectionError, match="redirect"):
                    await transport.send_streaming(provider, url, {"model": "gpt-4"})

        asyncio.run(run())

    def test_redirect_error_includes_security_message(self):
        transport = HttpTransport(timeout=10.0)
        provider = _test_provider()
        url = provider.upstream_url("gpt-4")

        async def run():
            with patch.object(
                transport._pool.get(provider.proxy_url),
                "post",
                side_effect=TooManyRedirects(url, 0),
            ):
                with pytest.raises(
                    UpstreamConnectionError, match="blocked for security"
                ):
                    await transport.send(provider, url, {"model": "gpt-4"})

        asyncio.run(run())

    def test_redirect_error_does_not_leak_api_key(self):
        transport = HttpTransport(timeout=10.0)
        provider = _test_provider()
        url = provider.upstream_url("gpt-4")

        async def run():
            with patch.object(
                transport._pool.get(provider.proxy_url),
                "post",
                side_effect=TooManyRedirects(url, 0),
            ):
                with pytest.raises(UpstreamConnectionError) as exc_info:
                    await transport.send(provider, url, {"model": "gpt-4"})
                assert "sk-test-key-12345" not in str(exc_info.value)

        asyncio.run(run())
