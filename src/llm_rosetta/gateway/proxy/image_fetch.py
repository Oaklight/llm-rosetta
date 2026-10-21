"""Image downloads for conversions that need inline bytes.

``ConversionPipeline`` runs request conversion synchronously, but the gateway
owns an HTTP stack (the vendored client, with the deployment's proxy).  Passing
this function as the pipeline's ``image_fetcher`` keeps conversion free of
network I/O: the Google Generate converter asks for the bytes instead of
reaching for ``urllib`` itself.
"""

from __future__ import annotations

import os
from typing import Any


def fetch_image(url: str) -> tuple[bytes, str] | None:
    """Download *url* through the vendored HTTP client.

    Args:
        url: Absolute image URL (``data:`` URIs never reach this function).

    Returns:
        ``(body, content-type)``; the content type may be empty when the
        upstream does not send one.

    Raises:
        RuntimeError: If the upstream answers with a non-success status.  The
            converter logs it and drops the part, the same result as a failed
            ``urllib`` fetch.
    """
    from llm_rosetta._vendor import httpclient

    proxy = os.environ.get("HTTPS_PROXY") or os.environ.get("HTTP_PROXY")
    with httpclient.Client(timeout=30.0, proxy=proxy) as client:
        response: Any = client.get(
            url, headers={"User-Agent": "llm-rosetta/1.0 (image fetch)"}
        )
        if response.status_code >= 400:
            raise RuntimeError(f"HTTP {response.status_code} for {url}")
        content_type = response.headers.get("Content-Type", "")
    return response.content, content_type
