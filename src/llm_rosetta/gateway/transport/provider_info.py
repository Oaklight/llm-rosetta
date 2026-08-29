"""Provider runtime configuration — connection info, auth, and key rotation.

This module contains the data classes that describe *how* to talk to an
upstream provider at the transport level:

* :class:`KeyRing` — round-robin API key selector.
* :class:`ProviderInfo` — base URL, auth headers, URL templates.
* Auth header builder functions (``openai_auth``, ``anthropic_auth``,
  ``google_auth``).

Higher-level factory logic (shim resolution, config parsing) stays in
``gateway.providers``.
"""

from __future__ import annotations

import re
from collections.abc import Callable

# Type alias for auth-header builder callables
AuthHeaderFn = Callable[[str], dict[str, str]]


# ---------------------------------------------------------------------------
# API key rotation (round-robin)
# ---------------------------------------------------------------------------


class KeyRing:
    """Round-robin API key selector.

    Accepts a single key string **or** a comma-separated list of keys.
    Each call to :meth:`next` returns the next key in rotation.
    """

    def __init__(self, keys_csv: str) -> None:
        self._keys = [k.strip() for k in keys_csv.split(",") if k.strip()]
        self._idx = 0

    def next(self) -> str:
        """Return the next API key."""
        if not self._keys:
            raise ValueError("No API keys configured")
        key = self._keys[self._idx]
        self._idx = (self._idx + 1) % len(self._keys)
        return key

    def __len__(self) -> int:
        return len(self._keys)


# ---------------------------------------------------------------------------
# Provider descriptor
# ---------------------------------------------------------------------------


# Version-prefix segments that may appear at the end of base_url AND
# at the start of a url_template path, causing duplication.
_VERSION_SUFFIXES = re.compile(r"/v\d+(?:beta\d?)?$", re.IGNORECASE)


def normalize_base_url(base_url: str, url_template: str) -> str:
    """Strip trailing version prefix from *base_url* when it would
    duplicate the start of *url_template*.

    Example: ``base_url="https://api.openai.com/v1"`` with
    ``url_template="{base_url}/v1/embeddings"`` → strips ``/v1`` from
    base_url to avoid ``/v1/v1/embeddings``.

    Templates like ``"{base_url}/chat/completions"`` (no version prefix
    after ``{base_url}``) are left alone — the ``/v1`` in base_url is
    needed.
    """
    m = _VERSION_SUFFIXES.search(base_url)
    if not m:
        return base_url
    suffix = m.group()  # e.g. "/v1", "/v1beta", "/v3"
    # Check if the template path (after {base_url}) starts with the same segment
    tpl_path = (
        url_template.split("{base_url}", 1)[-1] if "{base_url}" in url_template else ""
    )
    if tpl_path.startswith(suffix + "/") or tpl_path == suffix:
        return base_url[: m.start()]
    return base_url


class ProviderInfo:
    """Runtime representation of a single configured provider.

    Encapsulates base_url, key rotation, auth-header construction,
    and upstream URL building.
    """

    def __init__(
        self,
        name: str,
        *,
        api_key: str,
        base_url: str,
        auth_header_fn: AuthHeaderFn,
        url_template: str,
        stream_url_template: str | None = None,
        proxy_url: str | None = None,
        timeout: float | None = None,
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(
                f"Provider '{name}': base_url must start with http:// or https://, "
                f"got '{base_url}'"
            )
        self.name = name
        self.base_url = normalize_base_url(base_url.rstrip("/"), url_template)
        self.key_ring = KeyRing(api_key)
        self._auth_header_fn = auth_header_fn
        self._url_template = url_template
        self._stream_url_template = stream_url_template
        self.proxy_url = proxy_url
        self.timeout = timeout

    # -- public helpers used by the proxy -----------------------------------

    def auth_headers(self) -> dict[str, str]:
        """Return auth headers using the next rotated key."""
        return self._auth_header_fn(self.key_ring.next())

    def upstream_url(self, model: str, *, stream: bool = False) -> str:
        """Build the upstream URL for the given model."""
        tpl = (
            self._stream_url_template
            if (stream and self._stream_url_template)
            else self._url_template
        )
        return tpl.format(base_url=self.base_url, model=model)

    def with_url_templates(
        self,
        url_template: str | None = None,
        stream_url_template: str | None = None,
    ) -> ProviderInfo:
        """Return a shallow copy with overridden URL template(s).

        The new instance shares the same :class:`KeyRing` (round-robin
        state is preserved).  Fields not overridden keep their current
        values.
        """
        import copy

        clone = copy.copy(self)
        if url_template is not None:
            clone._url_template = url_template
        if stream_url_template is not None:
            clone._stream_url_template = stream_url_template
        return clone

    def with_timeout(self, timeout: float | None) -> ProviderInfo:
        """Return a shallow copy with an overridden timeout.

        The new instance shares the same :class:`KeyRing`.
        Returns ``self`` unchanged if *timeout* is ``None``.
        """
        if timeout is None:
            return self
        import copy

        clone = copy.copy(self)
        clone.timeout = timeout
        return clone


# ---------------------------------------------------------------------------
# Per-provider auth header builders
# ---------------------------------------------------------------------------


def openai_auth(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def anthropic_auth(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }


def google_auth(api_key: str) -> dict[str, str]:
    return {"x-goog-api-key": api_key}
