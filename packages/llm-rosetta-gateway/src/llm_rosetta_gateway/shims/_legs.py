"""Transport-failure classification, origin comparison and auth headers."""

from __future__ import annotations

import os
from urllib.parse import urlsplit


from ._types import (
    ShimSourceError,
)


def _is_unreachable(exc: BaseException) -> bool:
    """Whether *exc* is a transport failure — no response from the server.

    Only this class of failure falls back to the mirror.  An HTTP status is a
    response from the source itself, and a bad archive is the source's
    verdict; neither is a network symptom to route around.

    The transport subclasses are named rather than taking ``OSError`` whole:
    what is left of ``OSError`` here is local (a permission error, a path of
    the wrong type), which says nothing about whether the host answers.
    """
    from llm_rosetta_gateway._vendor.httpclient import (
        HTTPError,
        HttpConnectionError,
        HttpTimeoutError,
    )

    if isinstance(exc, HTTPError):
        return False
    return isinstance(
        exc, (HttpConnectionError, HttpTimeoutError, ConnectionError, TimeoutError)
    )


def _same_origin(first: str, second: str) -> bool:
    """Whether two URLs share a scheme, host and port."""
    left, right = _url_origin(first), _url_origin(second)
    return left is not None and left == right


#: The port a URL scheme implies when it names none.
_DEFAULT_PORTS = {"http": 80, "https": 443}


def _url_origin(url: str) -> tuple[str, str, int | None] | None:
    """The ``(scheme, host, port)`` an Authorization header would reach.

    A scheme's default port is filled in, so ``https://h`` and ``https://h:443``
    are one origin: writing the default explicitly must not cost the credential.
    """
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return None
    if not parts.hostname:
        return None
    scheme = parts.scheme.lower()
    if port is None:
        port = _DEFAULT_PORTS.get(scheme)
    return scheme, parts.hostname.lower(), port


def _auth_headers(token_env: str | None) -> dict[str, str]:
    """Build request headers, adding a bearer token when configured."""
    headers = {"User-Agent": "llm-rosetta-shim-loader", "Accept": "*/*"}
    if token_env:
        headers.update(_bearer_headers(token_env))
    return headers


def _bearer_headers(token_env: str) -> dict[str, str]:
    """The ``Authorization`` header for *token_env*, or an error when it is unset."""
    token = os.environ.get(token_env)
    if not token:
        raise ShimSourceError(f"token_env {token_env!r} is not set in the environment")
    return {"Authorization": f"Bearer {token}"}
