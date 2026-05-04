"""SOCKS5 proxy integration tests for the vendored httpclient.

Verifies that the gateway's proxy code path (AsyncClient with proxy=)
works correctly with SOCKS5 proxies, mirroring how proxy.py:get_client()
creates clients for upstream requests.
"""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

from llm_rosetta._vendor.httpclient import (
    AsyncClient,
    HttpConnectionError,
    Response as HttpResponse,
    Socks5Error,
    StreamingResponse as HttpStreamingResponse,
)


def _run(coro: Any) -> Any:
    return asyncio.get_event_loop().run_until_complete(coro)


class TestAsyncSocks5:
    """Async SOCKS5 proxy tests — mirrors gateway's AsyncClient usage."""

    def test_get_through_socks5(self, httpbin_url, socks5_url):
        async def _t():
            async with AsyncClient(proxy=socks5_url) as client:
                r = await client.get(f"{httpbin_url}/get")
                assert isinstance(r, HttpResponse)
                assert r.status_code == 200
                assert "url" in r.json()

        _run(_t())

    def test_post_json_through_socks5(self, httpbin_url, socks5_url):
        async def _t():
            async with AsyncClient(proxy=socks5_url) as client:
                r = await client.post(f"{httpbin_url}/post", json={"key": "value"})
                assert isinstance(r, HttpResponse)
                assert r.status_code == 200
                assert r.json()["json"] == {"key": "value"}

        _run(_t())

    def test_streaming_through_socks5(self, httpbin_url, socks5_url):
        async def _t():
            async with AsyncClient(proxy=socks5_url) as client:
                r = await client.get(f"{httpbin_url}/stream-bytes/1024", stream=True)
                assert isinstance(r, HttpStreamingResponse)
                assert r.status_code == 200
                data = b""
                async for chunk in r.aiter_bytes():
                    data += chunk
                assert len(data) == 1024

        _run(_t())

    def test_socks5_with_auth(self, httpbin_url, socks5_auth_url):
        async def _t():
            async with AsyncClient(proxy=socks5_auth_url) as client:
                r = await client.get(f"{httpbin_url}/get")
                assert r.status_code == 200

        _run(_t())

    def test_socks5_auth_missing(self, httpbin_url, socks5_auth_url):
        """SOCKS5 server requires auth but client provides none."""
        no_auth_url = socks5_auth_url.split("@")[-1]
        no_auth_url = f"socks5://{no_auth_url}"

        async def _t():
            async with AsyncClient(proxy=no_auth_url) as client:
                await client.get(f"{httpbin_url}/get")

        with pytest.raises((Socks5Error, HttpConnectionError)):
            _run(_t())

    def test_socks5_wrong_credentials(self, httpbin_url, socks5_auth_url):
        """SOCKS5 server requires auth but client provides wrong credentials."""
        parts = socks5_auth_url.split("@")
        wrong_url = f"socks5://wrong:creds@{parts[-1]}"

        async def _t():
            async with AsyncClient(proxy=wrong_url) as client:
                await client.get(f"{httpbin_url}/get")

        with pytest.raises((Socks5Error, HttpConnectionError)):
            _run(_t())
