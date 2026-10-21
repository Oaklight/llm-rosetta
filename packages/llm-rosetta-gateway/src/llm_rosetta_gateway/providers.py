"""Gateway provider definitions — registry, factory, and defaults.

Transport-level classes (:class:`ProviderInfo`, :class:`KeyRing`) and auth
header builders live in :mod:`gateway.transport.provider_info`.  This module
keeps the provider *registry* and *factory* that resolve shim config into
runtime :class:`ProviderInfo` instances.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from .transport.provider_info import (
    ProviderInfo,
    anthropic_auth,
    google_auth,
    no_auth,
    openai_auth,
)

from llm_rosetta.formats import FORMAT_SPECS
from llm_rosetta.provider_names import canonical_provider_name

# Re-export ProviderInfo so existing ``from .providers import ProviderInfo``
# continues to work without changes across the codebase.
__all__ = ["ProviderInfo", "build_provider_info"]

logger = logging.getLogger("llm-rosetta-gateway")


# ---------------------------------------------------------------------------
# Provider registry — known provider types and their characteristics
# ---------------------------------------------------------------------------

_AUTH_BUILDERS = {
    "openai": openai_auth,
    "anthropic": anthropic_auth,
    "google": google_auth,
}

# Derived from the format catalog (``llm_rosetta.formats``).  The registry key
# is the format; the legacy ``google`` spelling stays here because provider
# validation historically accepts it as a registry key as well as through
# ``canonical_provider_name``.
_PROVIDER_REGISTRY: dict[str, dict[str, Any]] = {}
for _spec in FORMAT_SPECS:
    if _spec.model_type != "llm" or _spec.url_template is None:
        continue
    _entry: dict[str, Any] = {
        "auth_header_fn": _AUTH_BUILDERS[_spec.auth or "openai"],
        "url_template": _spec.url_template,
    }
    if _spec.stream_url_template:
        _entry["stream_url_template"] = _spec.stream_url_template
    _PROVIDER_REGISTRY[_spec.format] = _entry

_PROVIDER_REGISTRY["google"] = _PROVIDER_REGISTRY["google_generate"]

# Vendor hints for bare base formats, surfaced by the admin provider picker as
# input *placeholders* only.  They are never runtime fallbacks: a bare base
# type with no ``base_url`` fails to load rather than silently inheriting a
# vendor host, and its credential env var is not consulted either.
_BASE_TYPE_HINTS: dict[str, dict[str, str]] = {
    spec.format: {
        "base_url": spec.base_url_hint,
        "api_key_env": spec.api_key_env_hint,
    }
    for spec in FORMAT_SPECS
    if spec.model_type == "llm" and spec.base_url_hint and spec.api_key_env_hint
}
_BASE_TYPE_HINTS["google"] = _BASE_TYPE_HINTS["google_generate"]


def get_base_type_hints(provider_type: str) -> dict[str, str]:
    """Return UI placeholder hints (``base_url`` / ``api_key_env``) for a base.

    These suggest a common vendor for a bare base format but are **not**
    runtime defaults — see :data:`_BASE_TYPE_HINTS`.  Empty for unknown types.
    """
    return dict(_BASE_TYPE_HINTS.get(canonical_provider_name(provider_type), {}))


def get_default_base_url(provider_type: str) -> str:
    """Return ``\"\"`` — a base format has no implied host.

    Base formats are vendor-neutral; the vendor URL shown in the admin picker
    is only a placeholder hint (:func:`get_base_type_hints`).  A provider must
    set ``base_url`` explicitly, or use a vendor ``<vendor>--<base>`` shim
    whose ``connection.base_url`` supplies it.
    """
    return ""


def get_default_api_key_env(provider_type: str) -> str:
    """Return ``\"\"`` — a base format has no implied credential env var.

    See :func:`get_default_base_url`.
    """
    return ""


def default_models_path(provider_type: str) -> str:
    """Default model-listing path appended to ``base_url`` for *provider_type*.

    Derived from the format catalog's ``models_path``; the generic
    ``/models`` is the fallback for a provider type the catalog does not know.
    """
    from llm_rosetta.formats import BY_PAIR

    fmt = canonical_provider_name(provider_type)
    for model_type in ("llm", "embedding"):
        spec = BY_PAIR.get((model_type, fmt))
        if spec is not None and spec.models_path:
            return spec.models_path
    return "/models"


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
    spec.format: spec.supports_custom_tools
    for spec in FORMAT_SPECS
    if spec.model_type == "llm"
}
_CUSTOM_TOOLS_BY_TYPE["google"] = _CUSTOM_TOOLS_BY_TYPE["google_generate"]


def get_default_supports_custom_tools(provider_type: str) -> bool:
    """Default custom-tools support for a base format (``False`` if unknown)."""
    return _CUSTOM_TOOLS_BY_TYPE.get(canonical_provider_name(provider_type), False)


# Base formats offered as bare formats in the admin provider picker, in display
# order — the chat API standards.  A vendor-named shim may also exist for the
# same standard (see ``registered_shims``).  ``google`` is a plain alias of
# ``google_generate`` and stays reachable through the ``google`` shim, so it is
# not listed separately here.
BASE_FORMATS: tuple[str, ...] = tuple(
    spec.format
    for spec in sorted(
        (
            spec
            for spec in FORMAT_SPECS
            if spec.model_type == "llm" and spec.url_template is not None
        ),
        key=lambda spec: spec.picker_order,
    )
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
    :class:`~gateway.maintenance.startup.DeferredStartup`).

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


def _credential_token(
    *,
    token_command: list[str] | None,
    api_key: str,
    keyless: bool,
) -> str | None:
    """Derive an opaque, non-secret account identity for a provider.

    Two providers with the same token share one upstream account; see
    :attr:`GatewayConfig.provider_account_tokens`.  ``None`` means "no stable
    credential identity" — a keyless upstream, or an empty api_key — so such
    providers are never grouped.

    Note an unresolved ``${VAR}`` is not empty: config loading leaves the
    placeholder text in place when the variable is unset, so two providers
    sharing one do group.  That is harmless — neither can authenticate.

    The token comes from the credential *source*, not its resolved value where
    the two differ: a ``token_command`` is hashed as-is so token rotation does
    not move the identity, while a static key is hashed by value so an
    env-resolved key and a literal key with the same value group together.
    Only the digest leaves this function; the raw key is never stored or logged.

    The endpoint is deliberately not part of the identity: the same credential
    pointed at two different ``base_url`` values is one account, so tier-2
    routing may realign between those endpoints.  Tier 1 is where the endpoint
    matters — its key pairs the shim name with ``base_url``.
    """
    if keyless:
        return None
    if token_command is not None:
        return "cmd:" + hashlib.sha256("\x00".join(token_command).encode()).hexdigest()
    keys = sorted(k.strip() for k in api_key.split(",") if k.strip())
    if not keys:
        return None
    return "key:" + hashlib.sha256("\x00".join(keys).encode()).hexdigest()


def build_provider_info(
    provider_type: str,
    cfg: dict[str, Any],
    *,
    global_proxy: str | None = None,
    shim_name: str | None = None,
) -> ProviderInfo:
    """Create a :class:`ProviderInfo` from a provider config dict.

    *provider_type* is the base converter type (e.g. ``"chat_completions"``).
    When a shim is found, its ``connection.base_url`` and
    ``connection.api_key_env`` are used as fallbacks when the config does not
    specify them.  A bare base type has no such fallback (it is vendor-neutral),
    so a missing ``base_url`` raises.

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

    from .shims import (
        find_source_package,
        legacy_shim_migration_advice,
        source_package_advice,
    )

    # Legacy spellings (e.g. ``openai_chat``) migrate to canonical names so the
    # registry lookup below hits.
    provider_type = canonical_provider_name(provider_type)

    # Resolve through shim registry for defaults
    shim = get_shim(shim_name or provider_type)
    if shim is not None:
        base_type = shim.format
        # Apply shim defaults where the config value is missing *or empty*.
        cfg = _apply_shim_defaults(cfg, shim)
    else:
        base_type = provider_type

    # A name that a standalone package provides but the registry does not hold
    # means the source was not declared (or failed to load).  The missing
    # base_url error below is the first thing such a config hits, so it is the
    # right place to name the package and the entry to declare.
    missing_package = (
        None if shim is not None else find_source_package(shim_name or provider_type)
    )

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
        # A shim that is simply not registered — usually one that moved to a
        # standalone package, or an older spelling of one — lands here and runs
        # on generic defaults, so this is where the operator has to be told
        # what to declare.  Without it the warning names the type and nothing
        # else, and a legacy spelling looks like a typo.
        logger.warning(
            "Unknown provider type '%s'; using Bearer auth and generic URL template%s",
            base_type,
            legacy_shim_migration_advice(shim_name)
            or source_package_advice(shim_name or base_type),
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

    # Keyless upstream (a free public gateway): send no credential while no
    # api_key and no token_command is configured.  Computed only after every
    # api_key source (config, shim env) so a key resolved from the environment
    # still disables keyless.  Supplying a key or a token_command switches back
    # to the base type's auth, so a paid account can use the same shim.
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
        message = (
            f"Provider {provider_type!r} (base type {base_type!r}) has no "
            "base_url configured; set it in the provider config or via a shim "
            "connection.base_url."
        )
        if missing_package is not None:
            name = shim_name or provider_type
            message += (
                f" Shim {name!r} is not built in: it ships in the standalone "
                f"package {missing_package.label!r} ({missing_package.repo}). "
                f"Declare `shims.presets: [{missing_package.key!r}]` (or the "
                f"equivalent `shims.remotes` entry: {missing_package.remote!r}); "
                "Settings -> Shim Sources offers both. If the source is already "
                "declared, check the shim-source status for a load failure."
            )
        raise ValueError(message)

    quota_raw = cfg.get("quota")
    quota_cfg: dict[str, Any] = quota_raw if isinstance(quota_raw, dict) else {}
    shim_quota = getattr(shim, "quota", None) if shim is not None else None
    quota_probe = quota_cfg.get("probe") or (
        shim_quota.probe if shim_quota is not None else None
    )
    quota_url = quota_cfg.get("url") or (
        shim_quota.url if shim_quota is not None else None
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
        credential_token=_credential_token(
            token_command=token_command, api_key=api_key, keyless=keyless
        ),
        session_header=cfg.get("session_header")
        or (shim.connection.session_header if shim is not None else None),
        extra_headers=(shim.connection.headers if shim is not None else None),
        request_header=(shim.connection.request_header if shim is not None else None),
        session_id_format=(
            shim.connection.session_id_format if shim is not None else None
        ),
        request_id_format=(
            shim.connection.request_id_format if shim is not None else None
        ),
        stream_required=bool(shim.connection.stream_required if shim else False),
        quota_probe_id=quota_probe,
        quota_url=quota_url,
        quota_probes=(shim.quota_probes if shim is not None else ()),
    )
