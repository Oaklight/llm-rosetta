"""Gateway provider definitions — registry, factory, and defaults.

Transport-level classes (:class:`ProviderInfo`, :class:`KeyRing`) and auth
header builders live in :mod:`gateway.transport.provider_info`.  This module
keeps the provider *registry* and *factory* that resolve shim config into
runtime :class:`ProviderInfo` instances.
"""

from __future__ import annotations

import logging
from typing import Any

from .transport.provider_info import (
    ProviderInfo,
    anthropic_auth,
    google_auth,
    no_auth,
    openai_auth,
)

from llm_rosetta.provider_names import normalize_provider_name

# Re-export ProviderInfo so existing ``from .providers import ProviderInfo``
# continues to work without changes across the codebase.
__all__ = ["ProviderInfo", "build_provider_info"]

logger = logging.getLogger("llm-rosetta-gateway")


# ---------------------------------------------------------------------------
# Provider registry — known provider types and their characteristics
# ---------------------------------------------------------------------------

_PROVIDER_REGISTRY: dict[str, dict[str, Any]] = {
    "chat_completions": {
        "default_base_url": "https://api.openai.com/v1",
        "default_api_key_env": "OPENAI_API_KEY",
        "auth_header_fn": openai_auth,
        "url_template": "{base_url}/chat/completions",
    },
    "openai_responses": {
        "default_base_url": "https://api.openai.com/v1",
        "default_api_key_env": "OPENAI_API_KEY",
        "auth_header_fn": openai_auth,
        "url_template": "{base_url}/responses",
    },
    "open_responses": {
        # The spec is vendor-neutral and has no canonical host of its own, so
        # the gateway defaults to OpenRouter (a multi-vendor router that speaks
        # the spec).  A deployment targeting another host must set ``base_url``
        # in its provider config or a shim ``connection.base_url``.
        "default_base_url": "https://openrouter.ai/api/v1",
        "default_api_key_env": "OPENROUTER_API_KEY",
        "auth_header_fn": openai_auth,
        "url_template": "{base_url}/responses",
    },
    "anthropic": {
        "default_base_url": "https://api.anthropic.com",
        "default_api_key_env": "ANTHROPIC_API_KEY",
        "auth_header_fn": anthropic_auth,
        "url_template": "{base_url}/v1/messages",
    },
    "google": {
        "default_base_url": "https://generativelanguage.googleapis.com",
        "default_api_key_env": "GOOGLE_API_KEY",
        "auth_header_fn": google_auth,
        "url_template": "{base_url}/v1beta/models/{model}:generateContent",
        "stream_url_template": "{base_url}/v1beta/models/{model}:streamGenerateContent?alt=sse",
    },
    "google_generate": {
        "default_base_url": "https://generativelanguage.googleapis.com",
        "default_api_key_env": "GOOGLE_API_KEY",
        "auth_header_fn": google_auth,
        "url_template": "{base_url}/v1beta/models/{model}:generateContent",
        "stream_url_template": "{base_url}/v1beta/models/{model}:streamGenerateContent?alt=sse",
    },
    "google_interactions": {
        "default_base_url": "https://generativelanguage.googleapis.com",
        "default_api_key_env": "GOOGLE_API_KEY",
        "auth_header_fn": google_auth,
        "url_template": "{base_url}/v1beta/interactions",
        "stream_url_template": "{base_url}/v1beta/interactions?alt=sse",
    },
}


def get_default_base_url(provider_type: str) -> str:
    """Return the default base URL for a known provider type, or ``""``."""
    entry = _PROVIDER_REGISTRY.get(normalize_provider_name(provider_type))
    return entry.get("default_base_url", "") if entry else ""


def get_default_api_key_env(provider_type: str) -> str:
    """Return the default env-var name for a provider's API key."""
    provider_type = normalize_provider_name(provider_type)
    entry = _PROVIDER_REGISTRY.get(provider_type)
    return (
        entry.get("default_api_key_env", f"{provider_type.upper()}_API_KEY")
        if entry
        else f"{provider_type.upper()}_API_KEY"
    )


def known_provider_types() -> list[str]:
    """Return the list of built-in provider type names."""
    return list(_PROVIDER_REGISTRY)


# Whether each base format's standard natively accepts ``{type: "custom"}``
# tool definitions.  A bare format uses this as its default when no vendor shim
# supplies one — the OpenAI-shaped standards do, Anthropic/Google do not.
# In the picker this only decides formats without an eponymous shim, since a
# shim entry wins in the merged lookup; the table is the format's own default
# regardless of whether a shim shadows it.
_CUSTOM_TOOLS_BY_TYPE: dict[str, bool] = {
    "chat_completions": True,
    "openai_responses": True,
    "open_responses": True,
    "anthropic": False,
    "google": False,
    "google_generate": False,
    "google_interactions": False,
}


def get_default_supports_custom_tools(provider_type: str) -> bool:
    """Default custom-tools support for a base format (``False`` if unknown)."""
    return _CUSTOM_TOOLS_BY_TYPE.get(normalize_provider_name(provider_type), False)


# Base formats offered as bare formats in the admin provider picker, in display
# order — the chat API standards.  A vendor-named shim may also exist for the
# same standard (see ``registered_shims``).  ``google`` is a plain alias of
# ``google_generate`` and stays reachable through the ``google`` shim, so it is
# not listed separately here.
BASE_FORMATS: tuple[str, ...] = (
    "chat_completions",
    "openai_responses",
    "open_responses",
    "anthropic",
    "google_generate",
    "google_interactions",
)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def _resolve_token_command(
    provider_type: str, cfg: dict[str, Any]
) -> tuple[str, list[str] | None, int]:
    """Resolve token_command config into (api_key, token_command, interval).

    Returns the initial API key (either from static config or by running the
    command), the command list (or None), and the refresh interval.

    When ``token_command`` is set, the actual subprocess call is **not**
    performed here — a sentinel placeholder is returned instead, and the
    real token is fetched asynchronously after the event loop starts (via
    :class:`~gateway.deferred_startup.DeferredStartup`).

    Raises ValueError on invalid config (bad types, missing binary, etc.).
    """
    import shutil

    from .transport.provider_info import TOKEN_PENDING_SENTINEL

    token_command = cfg.get("token_command")
    token_refresh_interval = int(cfg.get("token_refresh_interval", 3600))

    if token_command is None:
        # ``api_key`` may legitimately be absent; return an empty string so the
        # caller reaches its own clear error rather than a bare ``KeyError``.
        return cfg.get("api_key", ""), None, token_refresh_interval

    if isinstance(token_command, str):
        raise ValueError(
            f"Provider '{provider_type}': token_command must be an array, "
            f'not a string (e.g., ["python3", "scripts/alcf-token.py"])'
        )
    if not isinstance(token_command, list) or not token_command:
        raise ValueError(
            f"Provider '{provider_type}': token_command must be a non-empty array"
        )
    if token_refresh_interval < 60:
        raise ValueError(
            f"Provider '{provider_type}': token_refresh_interval must be "
            f">= 60, got {token_refresh_interval}"
        )
    if "api_key" in cfg:
        raise ValueError(
            f"Provider '{provider_type}': token_command and api_key are "
            f"mutually exclusive"
        )
    if shutil.which(token_command[0]) is None:
        logger.warning(
            "Provider '%s': token_command executable '%s' not found on "
            "PATH — the command may still work if the binary is available "
            "at runtime (e.g. via volume mount)",
            provider_type,
            token_command[0],
        )
    logger.info(
        "Provider '%s' uses token_command — token will be fetched "
        "asynchronously after startup",
        provider_type,
    )
    return TOKEN_PENDING_SENTINEL, token_command, token_refresh_interval


def _apply_shim_defaults(cfg: dict[str, Any], shim: Any) -> dict[str, Any]:
    """Fill base_url / api_key from the shim when the config omits them.

    An empty string counts as omitted: the admin writes ``""`` keys, and an
    empty base_url must fall back to the shim rather than reach ProviderInfo,
    which would raise on a URL that doesn't start with http.
    """
    import os

    if not cfg.get("base_url") and shim.connection.base_url:
        cfg = {**cfg, "base_url": shim.connection.base_url}
    if cfg.get("api_key") is None and shim.connection.api_key_env:
        env_val = os.environ.get(shim.connection.api_key_env, "")
        if env_val:
            cfg = {**cfg, "api_key": env_val}
    return cfg


def _apply_base_defaults(
    cfg: dict[str, Any], base_type: str, keyless_shim: bool
) -> dict[str, Any]:
    """Fill base_url / api_key from the base type's registry defaults.

    The base-type api_key env var is a generic convenience, not this provider's
    own credential, so it is skipped for a keyless shim — an unrelated
    ``OPENAI_API_KEY`` in the environment must not silently enable auth against
    a free pool.  ``is None`` (not falsy) leaves a key the operator deliberately
    cleared (``"api_key": ""``) cleared rather than refilling it.
    """
    import os

    if not cfg.get("base_url"):
        default_url = get_default_base_url(base_type)
        if default_url:
            cfg = {**cfg, "base_url": default_url}
    if cfg.get("api_key") is None and not keyless_shim:
        env_val = os.environ.get(get_default_api_key_env(base_type), "")
        if env_val:
            cfg = {**cfg, "api_key": env_val}
    return cfg


def build_provider_info(
    provider_type: str,
    cfg: dict[str, Any],
    *,
    global_proxy: str | None = None,
    shim_name: str | None = None,
) -> ProviderInfo:
    """Create a :class:`ProviderInfo` from a provider config dict.

    *provider_type* is the base converter type (e.g. ``"chat_completions"``).
    When a shim is found, its ``default_base_url`` and ``default_api_key_env``
    are used as fallbacks when the config does not specify them.

    *shim_name* is the registered shim name (e.g. ``"deepseek"``) when the
    provider was configured via a shim.  Shim defaults are looked up from it;
    when omitted, *provider_type* is tried as a shim name too (callers that
    pass a shim name as *provider_type* keep working).  Gateway callers should
    pass *shim_name* explicitly: they resolve the shim to its base type first,
    so otherwise the shim — and its connection defaults — would be lost.

    *cfg* is the dict from the JSONC config, e.g.
    ``{"api_key": "sk-...", "base_url": "https://..."}``

    *global_proxy* is the server-level proxy URL (from ``server.proxy``).
    A per-provider ``"proxy"`` key in *cfg* takes precedence.

    For known provider types the auth and URL logic is looked up from the
    registry.  Unknown types fall back to Bearer-token auth and a simple
    ``{base_url}/`` URL template.
    """
    from llm_rosetta.shims import get_shim

    # Legacy spellings (e.g. ``openai_chat``) migrate to canonical names so the
    # registry lookup below hits.
    provider_type = normalize_provider_name(provider_type)

    # Resolve through shim registry for defaults
    shim = get_shim(shim_name or provider_type)
    if shim is not None:
        base_type = normalize_provider_name(shim.base)
        # Apply shim defaults where the config value is missing *or empty*.
        cfg = _apply_shim_defaults(cfg, shim)
    else:
        base_type = provider_type

    keyless_shim = bool(shim and shim.connection.keyless)

    reg = _PROVIDER_REGISTRY.get(base_type)

    if reg:
        auth_fn = reg["auth_header_fn"]
        url_tpl = reg["url_template"]
        stream_tpl = reg.get("stream_url_template")
    else:
        auth_fn = openai_auth
        url_tpl = "{base_url}/"
        stream_tpl = None
        logger.warning(
            "Unknown provider type '%s'; using Bearer auth and generic URL template",
            base_type,
        )

    # Per-shim auth_header override: when the shim declares a custom
    # auth header name, build a simple key→value header function that
    # replaces the base-type default.
    if shim is not None and shim.connection.auth_header:
        header_name = shim.connection.auth_header
        auth_fn = lambda key, _h=header_name: {_h: key}  # noqa: E731

    # Per-provider url_template / stream_url_template override from config
    if "url_template" in cfg:
        url_tpl = cfg["url_template"]
    if "stream_url_template" in cfg:
        stream_tpl = cfg["stream_url_template"]

    # Fall back to base-type defaults if still missing or empty
    cfg = _apply_base_defaults(cfg, base_type, keyless_shim)

    # Keyless upstream (a free public gateway): send no credential while no
    # api_key and no token_command is configured.  Computed only after every
    # api_key source (config, shim env, base-type env) so a key resolved from
    # the environment still disables keyless.  Supplying a key or a
    # token_command switches back to the base type's auth, so a paid account
    # can use the same shim.
    keyless = bool(
        keyless_shim and not cfg.get("api_key") and not cfg.get("token_command")
    )
    if keyless:
        auth_fn = no_auth

    # Per-provider proxy overrides global proxy
    proxy_url = cfg.get("proxy") or global_proxy or None

    # -- token_command: dynamic key refresh ------------------------------------
    if keyless:
        # No token_command, so no refresher is ever scheduled and the interval
        # is inert — 0 rather than the 3600 default, so it cannot read as
        # "refreshes hourly".  (It skips _resolve_token_command's >= 60 check,
        # which only guards the command path.)
        api_key, token_command, token_refresh_interval = "", None, 0
    else:
        api_key, token_command, token_refresh_interval = _resolve_token_command(
            provider_type, cfg
        )

    if not cfg.get("base_url"):
        raise ValueError(
            f"Provider {provider_type!r} (base type {base_type!r}) has no "
            "base_url configured; set it in the provider config or via a shim "
            "connection.base_url."
        )

    return ProviderInfo(
        name=provider_type,
        api_key=api_key,
        base_url=cfg["base_url"],
        auth_header_fn=auth_fn,
        url_template=url_tpl,
        stream_url_template=stream_tpl,
        proxy_url=proxy_url,
        timeout=float(cfg["timeout"]) if "timeout" in cfg else None,
        token_command=token_command,
        token_refresh_interval=token_refresh_interval,
        keyless=keyless,
    )
