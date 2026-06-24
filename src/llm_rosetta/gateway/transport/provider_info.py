"""Provider runtime configuration — connection info, auth, and key rotation.

This module contains the data classes that describe *how* to talk to an
upstream provider at the transport level:

* :class:`KeyRing` — round-robin API key selector.
* :class:`ProviderInfo` — base URL, auth headers, URL templates.
* Auth header builder functions (``_openai_auth``, ``_anthropic_auth``,
  ``_google_auth``).

Higher-level factory logic (shim resolution, config parsing) stays in
``gateway.providers``.
"""

from __future__ import annotations

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
    ) -> None:
        if not base_url.startswith(("http://", "https://")):
            raise ValueError(
                f"Provider '{name}': base_url must start with http:// or https://, "
                f"got '{base_url}'"
            )
        self.name = name
        self.base_url = base_url.rstrip("/")
        self.key_ring = KeyRing(api_key)
        self._auth_header_fn = auth_header_fn
        self._url_template = url_template
        self._stream_url_template = stream_url_template
        self.proxy_url = proxy_url

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


# ---------------------------------------------------------------------------
# Per-provider auth header builders
# ---------------------------------------------------------------------------


def _openai_auth(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}


def _anthropic_auth(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }


def _google_auth(api_key: str) -> dict[str, str]:
    return {"x-goog-api-key": api_key}
