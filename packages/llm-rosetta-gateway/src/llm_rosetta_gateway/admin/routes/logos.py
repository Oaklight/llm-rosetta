"""Provider logos, fetched by the gateway instead of by the browser.

The panel used to render every provider's logo straight from the host the
config named — in practice ``cdn.jsdelivr.net`` for the shims, plus whatever
host a deployment's own entries point at.  That made the page contact third
parties on load, told them which providers this gateway holds, and left an
air-gapped deployment with broken images (#1522).

This route is the one door: the browser asks the gateway, the gateway asks the
host, and the answer is cached.  Only hosts this gateway can name are fetched —
jsDelivr for the built-in icon set, and the hosts the configured entries
declare — and the check is repeated on the URL the bytes were read from, so a
redirect out of the allow-list is refused rather than followed.
"""

from __future__ import annotations

import time
from typing import Any
from urllib.parse import urlparse

from llm_rosetta_gateway._vendor.httpserver import JSONResponse, Response

from ._shared import _qp

#: Hosts the built-in shims and the icon picker use.
_BUILTIN_HOSTS = {"cdn.jsdelivr.net", "data.jsdelivr.com"}

#: A fetched logo, kept for a day: a logo is not worth a request per paint.
_TTL_SECONDS = 86_400
#: Served from the admin origin, so the bytes may not run anything, and the
#: browser may not re-sniff them into a type that would (#1522 review).
_RESPONSE_HEADERS = {
    "Cache-Control": "public, max-age=86400",
    "Content-Security-Policy": "default-src 'none'; style-src 'unsafe-inline'",
    "X-Content-Type-Options": "nosniff",
}
_MAX_BYTES = 512 * 1024
_cache: dict[str, tuple[bytes, str, float]] = {}


def _configured_logo_hosts(app: Any) -> set[str]:
    """Every host the configured provider entries name in their ``logo``."""
    hosts: set[str] = set()
    config = getattr(app, "config", None)
    raw = getattr(config, "_raw_providers", None) if config is not None else None
    for entry in (raw or {}).values():
        logo = (entry or {}).get("logo") if isinstance(entry, dict) else None
        if isinstance(logo, str) and "://" in logo:
            host = urlparse(logo).hostname
            if host:
                hosts.add(host)
    return hosts


def _allowed(src: str, app: Any) -> bool:
    parsed = urlparse(src)
    if parsed.scheme not in ("http", "https") or not parsed.hostname:
        return False
    return parsed.hostname in _BUILTIN_HOSTS | _configured_logo_hosts(app)


async def get_logo(request: Any) -> Response:
    """Return one image from an allowed host, cached.

    Query: ``src`` — the absolute URL to fetch.
    """
    src = (_qp(request, "src", "") or "").strip()
    if not src:
        return JSONResponse({"error": "src is required"}, status_code=400)
    if not _allowed(src, request.app):
        return JSONResponse(
            {"error": "That host is not a logo source"}, status_code=403
        )

    now = time.monotonic()
    cached = _cache.get(src)
    if cached and now - cached[2] < _TTL_SECONDS:
        data, content_type, _ = cached
        return Response(
            body=data,
            status_code=200,
            content_type=content_type,
            headers=dict(_RESPONSE_HEADERS),
        )

    from llm_rosetta_gateway._vendor.httpclient import (
        AsyncClient,
        Response as HttpAnswer,
    )

    try:
        async with AsyncClient(timeout=5) as client:
            answer = await client.get(src)
    except Exception as exc:  # pragma: no cover - network failure path
        return JSONResponse({"error": f"Logo fetch failed: {exc}"}, status_code=502)
    if not isinstance(answer, HttpAnswer):
        return JSONResponse({"error": "Logo fetch failed"}, status_code=502)
    # The allow-list has to hold for the URL the bytes came from, not the one
    # the caller asked for: an allowed host that redirects is still a door to
    # every host this gateway cannot name (#1522 review).
    if not _allowed(answer.url, request.app):
        return JSONResponse(
            {"error": "That host is not a logo source"}, status_code=403
        )
    if answer.status_code != 200 or not answer.content:
        return JSONResponse({"error": "Logo fetch failed"}, status_code=502)
    content_type = (answer.headers.get("content-type") or "image/svg+xml").split(";")[0]
    if not content_type.startswith(("image/", "application/octet-stream")):
        return JSONResponse({"error": "That URL is not an image"}, status_code=415)
    if len(answer.content) > _MAX_BYTES:
        return JSONResponse({"error": "That image is too large"}, status_code=413)
    _cache[src] = (answer.content, content_type, now)
    return Response(
        body=answer.content,
        status_code=200,
        content_type=content_type,
        headers=dict(_RESPONSE_HEADERS),
    )
