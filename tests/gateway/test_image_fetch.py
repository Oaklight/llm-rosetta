"""The gateway supplies conversion's image fetcher (#1155).

Conversion must not perform network I/O of its own on the gateway path; the
gateway hands the pipeline a fetcher built on the vendored client instead.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from llm_rosetta_gateway.proxy.image_fetch import fetch_image


def test_fetch_image_returns_the_body_and_content_type(monkeypatch):
    seen: dict[str, Any] = {}

    class _Client:
        def __init__(self, **kwargs):
            seen["client"] = kwargs

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url, **kwargs):
            seen["url"] = url
            seen["headers"] = kwargs.get("headers")
            return SimpleNamespace(
                status_code=200,
                headers={"Content-Type": "image/png; charset=binary"},
                content=b"\x89PNG",
            )

    monkeypatch.setattr("llm_rosetta._vendor.httpclient.Client", _Client, raising=True)

    assert fetch_image("https://example.com/a.png") == (
        b"\x89PNG",
        "image/png; charset=binary",
    )
    assert seen["url"] == "https://example.com/a.png"
    assert "llm-rosetta" in seen["headers"]["User-Agent"]


def test_fetch_image_raises_on_an_error_status(monkeypatch):
    class _Client:
        def __init__(self, **kwargs):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def get(self, url, **kwargs):
            return SimpleNamespace(status_code=404, headers={}, content=b"")

    monkeypatch.setattr("llm_rosetta._vendor.httpclient.Client", _Client, raising=True)

    with pytest.raises(RuntimeError, match="404"):
        fetch_image("https://example.com/missing.png")
