"""What the panel's own files answer, and where its logos come from (#1522).

The static routes used to re-send every byte on every reload, and the panel
loaded provider logos straight from jsDelivr — so a reload paid for 967 KB and
the page told a third party which providers the gateway holds.  These cases pin
the two halves of the fix: a conditional request is a 304 with no body, text is
compressed when the client asks, and the logo route only fetches from hosts
this gateway can name.
"""

from __future__ import annotations

import gzip
from typing import Any

import pytest

from llm_rosetta_gateway.admin.routes import logos
from llm_rosetta_gateway.admin.routes.auth import serve_admin_html, serve_admin_static


class _Request:
    """What the handlers read: headers, query params, path params, the app."""

    def __init__(
        self,
        *,
        headers: dict[str, str] | None = None,
        query: dict[str, list[str]] | None = None,
        path_params: dict[str, str] | None = None,
        app: Any = None,
    ) -> None:
        self.headers = headers or {}
        self.query_params = query or {}
        self.path_params = path_params or {}
        self.app = app


@pytest.mark.asyncio
async def test_a_static_file_carries_a_validator_and_revalidates() -> None:
    request = _Request(path_params={"path": "js/core/init.js"})
    first = await serve_admin_static(request)
    assert first.status_code == 200
    assert first.headers["Cache-Control"] == "no-cache"
    etag = first.headers["ETag"]
    assert etag.startswith('"')

    again = await serve_admin_static(
        _Request(
            path_params={"path": "js/core/init.js"}, headers={"if-none-match": etag}
        )
    )
    assert again.status_code == 304
    assert again.body == b""
    assert again.headers["ETag"] == etag


@pytest.mark.asyncio
async def test_text_is_compressed_when_the_client_asks() -> None:
    plain = await serve_admin_static(
        _Request(path_params={"path": "css/components.css"})
    )
    zipped = await serve_admin_static(
        _Request(
            path_params={"path": "css/components.css"},
            headers={"accept-encoding": "gzip"},
        )
    )
    assert zipped.headers["Content-Encoding"] == "gzip"
    assert zipped.headers["Vary"] == "Accept-Encoding"
    assert len(zipped.body) < len(plain.body)
    assert gzip.decompress(zipped.body) == plain.body


@pytest.mark.asyncio
async def test_the_document_revalidates_too() -> None:
    first = await serve_admin_html(_Request())
    assert first.status_code == 200
    second = await serve_admin_html(
        _Request(headers={"if-none-match": first.headers["ETag"]})
    )
    assert second.status_code == 304


@pytest.mark.asyncio
async def test_a_logo_needs_a_source_and_an_allowed_host() -> None:
    assert (await logos.get_logo(_Request(query={}))).status_code == 400
    refused = await logos.get_logo(
        _Request(query={"src": ["http://127.0.0.1:1/secret.svg"]})
    )
    assert refused.status_code == 403


def _fake_answer(url: str):
    """A real client ``Response``, so the handler's own type check passes."""
    from llm_rosetta_gateway._vendor.httpclient import CaseInsensitiveDict, Response

    return Response(
        status_code=200,
        headers=CaseInsensitiveDict({"content-type": "image/svg+xml"}),
        content=b"<svg/>",
        url=url,
    )


@pytest.mark.asyncio
async def test_an_allowed_logo_is_fetched_once(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def get(self, url: str):
            calls.append(url)
            return _fake_answer(url)

    import llm_rosetta_gateway._vendor.httpclient as httpclient

    monkeypatch.setattr(httpclient, "AsyncClient", _FakeClient)
    logos._cache.clear()
    src = "https://cdn.jsdelivr.net/npm/pkg/icons/x.svg"
    first = await logos.get_logo(_Request(query={"src": [src]}))
    second = await logos.get_logo(_Request(query={"src": [src]}))
    assert first.status_code == 200
    assert first.body == b"<svg/>"
    assert second.status_code == 200
    assert calls == [src], "the second request must come from the cache"


@pytest.mark.asyncio
async def test_a_redirect_out_of_the_allow_list_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An allowed host is a door if it can answer with someone else's bytes.

    The check used to run on the URL the caller asked for only; the client
    follows redirects, so `allowed.example` could serve `http://127.0.0.1/…`
    and the gateway would hand the panel bytes from a host it cannot name
    (#1522 review).
    """

    class _RedirectingClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _RedirectingClient:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def get(self, url: str):
            return _fake_answer("http://127.0.0.1:8102/logo.svg")

    import llm_rosetta_gateway._vendor.httpclient as httpclient

    monkeypatch.setattr(httpclient, "AsyncClient", _RedirectingClient)
    logos._cache.clear()
    answer = await logos.get_logo(
        _Request(query={"src": ["https://cdn.jsdelivr.net/npm/pkg/icons/x.svg"]})
    )
    assert answer.status_code == 403


@pytest.mark.asyncio
async def test_a_proxied_logo_cannot_run_anything(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The bytes are served from the admin origin, so they may not execute."""

    class _FakeClient:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        async def __aenter__(self) -> _FakeClient:
            return self

        async def __aexit__(self, *exc: Any) -> None:
            return None

        async def get(self, url: str):
            from llm_rosetta_gateway._vendor.httpclient import (
                CaseInsensitiveDict,
                Response,
            )

            return Response(
                status_code=200,
                headers=CaseInsensitiveDict({"content-type": "image/svg+xml"}),
                content=b"<svg><script>alert(1)</script></svg>",
                url=url,
            )

    import llm_rosetta_gateway._vendor.httpclient as httpclient

    monkeypatch.setattr(httpclient, "AsyncClient", _FakeClient)
    logos._cache.clear()
    answer = await logos.get_logo(
        _Request(query={"src": ["https://cdn.jsdelivr.net/npm/pkg/icons/x.svg"]})
    )
    assert answer.status_code == 200
    assert "script-src 'none'" in answer.headers["Content-Security-Policy"] or (
        "default-src 'none'" in answer.headers["Content-Security-Policy"]
    )
    assert answer.headers["X-Content-Type-Options"] == "nosniff"
