"""Gateway provider definitions — auth, URLs, defaults, and key rotation."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("llm-rosetta-gateway")

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


# ---------------------------------------------------------------------------
# Provider registry — known provider types and their characteristics
# ---------------------------------------------------------------------------

_PROVIDER_REGISTRY: dict[str, dict[str, Any]] = {
    "openai_chat": {
        "default_base_url": "https://api.openai.com/v1",
        "default_api_key_env": "OPENAI_API_KEY",
        "auth_header_fn": _openai_auth,
        "url_template": "{base_url}/chat/completions",
    },
    "openai_responses": {
        "default_base_url": "https://api.openai.com/v1",
        "default_api_key_env": "OPENAI_API_KEY",
        "auth_header_fn": _openai_auth,
        "url_template": "{base_url}/responses",
    },
    "open_responses": {
        "default_base_url": "https://api.openai.com/v1",
        "default_api_key_env": "OPENAI_API_KEY",
        "auth_header_fn": _openai_auth,
        "url_template": "{base_url}/responses",
    },
    "anthropic": {
        "default_base_url": "https://api.anthropic.com",
        "default_api_key_env": "ANTHROPIC_API_KEY",
        "auth_header_fn": _anthropic_auth,
        "url_template": "{base_url}/v1/messages",
    },
    "google": {
        "default_base_url": "https://generativelanguage.googleapis.com",
        "default_api_key_env": "GOOGLE_API_KEY",
        "auth_header_fn": _google_auth,
        "url_template": "{base_url}/v1beta/models/{model}:generateContent",
        "stream_url_template": "{base_url}/v1beta/models/{model}:streamGenerateContent?alt=sse",
    },
}


def get_default_base_url(provider_type: str) -> str:
    """Return the default base URL for a known provider type, or ``""``."""
    entry = _PROVIDER_REGISTRY.get(provider_type)
    return entry["default_base_url"] if entry else ""


def get_default_api_key_env(provider_type: str) -> str:
    """Return the default env-var name for a provider's API key."""
    entry = _PROVIDER_REGISTRY.get(provider_type)
    return entry["default_api_key_env"] if entry else f"{provider_type.upper()}_API_KEY"


def known_provider_types() -> list[str]:
    """Return the list of built-in provider type names."""
    return list(_PROVIDER_REGISTRY)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_provider_info(
    provider_type: str,
    cfg: dict[str, str],
    *,
    global_proxy: str | None = None,
) -> ProviderInfo:
    """Create a :class:`ProviderInfo` from a provider config dict.

    *provider_type* may be a base converter type (e.g. ``"openai_chat"``)
    or a registered shim name (e.g. ``"deepseek"``).  When a shim is
    found, its ``default_base_url`` and ``default_api_key_env`` are used
    as fallbacks when the config does not specify them.

    *cfg* is the dict from the JSONC config, e.g.
    ``{"api_key": "sk-...", "base_url": "https://..."}``

    *global_proxy* is the server-level proxy URL (from ``server.proxy``).
    A per-provider ``"proxy"`` key in *cfg* takes precedence.

    For known provider types the auth and URL logic is looked up from the
    registry.  Unknown types fall back to Bearer-token auth and a simple
    ``{base_url}/`` URL template.
    """
    import os

    from llm_rosetta.shims import get_shim

    # Resolve through shim registry for defaults
    shim = get_shim(provider_type)
    if shim is not None:
        base_type = shim.base
        # Apply shim defaults where config is missing
        if "base_url" not in cfg and shim.default_base_url:
            cfg = {**cfg, "base_url": shim.default_base_url}
        if "api_key" not in cfg and shim.default_api_key_env:
            env_val = os.environ.get(shim.default_api_key_env, "")
            if env_val:
                cfg = {**cfg, "api_key": env_val}
    else:
        base_type = provider_type

    reg = _PROVIDER_REGISTRY.get(base_type)

    if reg:
        auth_fn = reg["auth_header_fn"]
        url_tpl = reg["url_template"]
        stream_tpl = reg.get("stream_url_template")
    else:
        # Unknown / custom provider — best-effort defaults
        auth_fn = _openai_auth
        url_tpl = "{base_url}/"
        stream_tpl = None
        logger.warning(
            "Unknown provider type '%s'; using Bearer auth and generic URL template",
            base_type,
        )

    # Fall back to base-type defaults if still missing
    if "base_url" not in cfg:
        default_url = get_default_base_url(base_type)
        if default_url:
            cfg = {**cfg, "base_url": default_url}
    if "api_key" not in cfg:
        default_env = get_default_api_key_env(base_type)
        env_val = os.environ.get(default_env, "")
        if env_val:
            cfg = {**cfg, "api_key": env_val}

    # Per-provider proxy overrides global proxy
    proxy_url = cfg.get("proxy") or global_proxy or None

    return ProviderInfo(
        name=provider_type,
        api_key=cfg["api_key"],
        base_url=cfg["base_url"],
        auth_header_fn=auth_fn,
        url_template=url_tpl,
        stream_url_template=stream_tpl,
        proxy_url=proxy_url,
    )
