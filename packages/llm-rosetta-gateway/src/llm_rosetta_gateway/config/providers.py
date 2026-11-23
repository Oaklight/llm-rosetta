"""Gateway provider definitions — registry, factory, and defaults.

Transport-level classes (:class:`ProviderInfo`, :class:`KeyRing`) and auth
header builders live in :mod:`gateway.transport.provider_info`.  This module
keeps the provider *registry* and *factory* that resolve shim config into
runtime :class:`ProviderInfo` instances.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
from typing import Any

from ..transport.provider_info import (
    ProviderInfo,
    anthropic_auth,
    google_auth,
    no_auth,
    openai_auth,
)

from llm_rosetta.api import FORMAT_SPECS, spec_for

# Re-export ProviderInfo so existing ``from .providers import ProviderInfo``
# continues to work without changes across the codebase.
__all__ = [
    "ProviderInfo",
    "build_provider_info",
    "duplicate_profile_errors",
    "lane_profile_is_managed",
    "managed_acp_state_dir",
    "managed_auth_state_dir",
    "validate_provider_profiles",
]

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
# is the format.
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


def get_base_type_hints(provider_type: str) -> dict[str, str]:
    """Return UI placeholder hints (``base_url`` / ``api_key_env``) for a base.

    These suggest a common vendor for a bare base format but are **not**
    runtime defaults — see :data:`_BASE_TYPE_HINTS`.  Empty for unknown types.
    """
    return dict(_BASE_TYPE_HINTS.get(provider_type, {}))


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
    for model_type in ("llm", "embeddings"):
        spec = spec_for(model_type, provider_type)
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


def get_default_supports_custom_tools(provider_type: str) -> bool:
    """Default custom-tools support for a base format (``False`` if unknown)."""
    return _CUSTOM_TOOLS_BY_TYPE.get(provider_type, False)


# Base formats offered as bare formats in the admin provider picker, in display
# order — the chat API standards.  A vendor-named shim may also exist for the
# same standard (see ``registered_shims``).
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
    name: str, cfg: dict[str, Any], *, quiet: bool = False
) -> tuple[str, list[str] | None, int]:
    """Resolve token_command config into (api_key, token_command, interval).

    *name* is what the operator wrote — the shim name when there is one — so
    the messages below name the provider they can actually find in the config
    rather than its base format.

    Returns the initial API key (either from static config or by running the
    command), the command list (or None), and the refresh interval.

    When ``token_command`` is set, the actual subprocess call is **not**
    performed here — a sentinel placeholder is returned instead, and the
    real token is fetched asynchronously after the event loop starts (via
    :class:`~gateway.maintenance.startup.DeferredStartup`).

    Raises ValueError on invalid config (bad types, missing binary, etc.).
    """
    import shutil

    from ..transport.provider_info import TOKEN_PENDING_SENTINEL

    token_command = cfg.get("token_command")
    token_refresh_interval = int(cfg.get("token_refresh_interval", 3600))

    if token_command is None:
        # ``api_key`` may legitimately be absent; return an empty string so the
        # caller reaches its own clear error rather than a bare ``KeyError``.
        return cfg.get("api_key", ""), None, token_refresh_interval

    if isinstance(token_command, str):
        raise ValueError(
            f"Provider '{name}': token_command must be an array, "
            f'not a string (e.g., ["python3", "token.py"])'
        )
    if not isinstance(token_command, list) or not token_command:
        raise ValueError(f"Provider '{name}': token_command must be a non-empty array")
    if token_refresh_interval < 60:
        raise ValueError(
            f"Provider '{name}': token_refresh_interval must be "
            f">= 60, got {token_refresh_interval}"
        )
    # Only a real key conflicts with the command, and only when the command
    # does not consume it: ``token_command_env`` marks an exchange that takes
    # the key as input (AskSage mints a token from it), where the key is not a
    # second credential.  An entry that carries ``"api_key": ""`` says "no
    # static key" — the admin writes that key for a provider the operator never
    # gave a credential — and treating it as a second source made every saved
    # command provider unusable on reload.
    if cfg.get("api_key") and cfg.get("token_command_env") is None:
        raise ValueError(
            f"Provider '{name}': token_command and api_key are mutually exclusive"
        )
    # The admin's pre-write validation rebuilds every enabled provider on each
    # save.  These two lines describe the *provider*, not that save, so they
    # would repeat once per save; the validator asks for them at DEBUG.
    if shutil.which(token_command[0]) is None:
        logger.log(
            logging.DEBUG if quiet else logging.WARNING,
            "Provider '%s': token_command executable '%s' not found on "
            "PATH — the command may still work if the binary is available "
            "at runtime (e.g. via volume mount)",
            name,
            token_command[0],
        )
    logger.log(
        logging.DEBUG if quiet else logging.INFO,
        "Provider '%s' uses token_command — token will be fetched "
        "asynchronously after startup",
        name,
    )
    return TOKEN_PENDING_SENTINEL, token_command, token_refresh_interval


def _exchange_consumes_key(exchange: Any) -> bool:
    """Whether a shim exchange takes the operator's key as its *input*."""
    return bool(getattr(exchange, "consumes_key", False))


def _apply_shim_defaults(
    cfg: dict[str, Any], shim: Any, exchange: Any | None = None
) -> dict[str, Any]:
    """Fill base_url / credential defaults from the shim when absent.

    An empty string counts as omitted: the admin writes ``""`` keys, and an
    empty base_url must fall back to the shim rather than reach ProviderInfo,
    which would raise on a URL that doesn't start with http.

    A shim has two ways to declare a credential exchange — the argv
    ``connection.token_command`` and the module form ``auth.py`` — and the
    provider's own ``token_command`` overrides either.  Both own the same slot,
    so only one default is ever installed; the loader already rejects a shim
    that ships both.  *exchange* is the shim's ``auth.py`` object already
    resolved by the caller, or ``None``.
    """
    if not cfg.get("base_url") and shim.connection.base_url:
        cfg = {**cfg, "base_url": shim.connection.base_url}

    # Does the provider name its own command?  Asked before the shim default
    # is installed, so a shim's declaration cannot read as the operator's.
    provider_owns_command = cfg.get("token_command") is not None

    # A shim that declares the exchange ships its default credential source.
    # Installing it before the static-key default below keeps the two from
    # ever being set together — the loader rejects that pair.
    if exchange is None and not provider_owns_command and shim.connection.token_command:
        cfg = {**cfg, "token_command": list(shim.connection.token_command)}
    command_slot_taken = cfg.get("token_command") is not None or exchange is not None
    # A command that *takes* the operator's key is not a second credential:
    # the key is its input, and its output is what goes upstream.
    if cfg.get("token_command_env") is None and shim.connection.token_command_env:
        cfg = {**cfg, "token_command_env": shim.connection.token_command_env}
    # The interval is the vendor's credential lifetime, so it stays a default
    # even when the operator names their own command.
    if (
        (cfg.get("token_command") is not None or exchange is not None)
        and cfg.get("token_refresh_interval") is None
        and shim.connection.token_refresh_interval is not None
    ):
        cfg = {
            **cfg,
            "token_refresh_interval": shim.connection.token_refresh_interval,
        }
    # A shim's ``api_key_env`` is a *default* credential, and the command slot
    # above is a different source for the same slot: injecting here would hand
    # ``_resolve_token_command`` an ``api_key`` it has to reject on sight, so a
    # config that names a command (e.g. the documented ALCF setup) would fail
    # merely because the shim's variable happened to be exported.  The one
    # exception is an exchange whose key is an *input* — AskSage mints a token
    # from the account key — where the key is not a second credential.
    # ``is None`` rather than a falsy test: an explicit ``""``/``null`` is the
    # operator saying "no static key", and ``"not in cfg"`` would turn an
    # explicit ``null`` into an empty key ring.
    key_is_input = cfg.get("token_command_env") is not None or (
        exchange is not None and _exchange_consumes_key(exchange)
    )
    if (
        cfg.get("api_key") is None
        and (not command_slot_taken or key_is_input)
        and shim.connection.api_key_env
    ):
        env_val = os.environ.get(shim.connection.api_key_env, "")
        if env_val:
            cfg = {**cfg, "api_key": env_val}
    return cfg


def _auth_state_dir(data_dir: str | None, provider: str) -> str | None:
    """Return the directory a shim exchange may persist its record in.

    One directory per configured provider, under the gateway's data dir.
    *data_dir* must already be resolved: a relative value would be read against
    the process's working directory, which is not where the gateway keeps its
    state.  ``None`` leaves the location to the exchange's own default
    (typically under ``~/.config/llm-rosetta``).
    """
    if not data_dir:
        return None
    # Readable rather than opaque: an operator looking for a credential
    # record should find it under the provider name they configured.  Only
    # characters a path cannot carry are replaced, and a leading dot is
    # dropped so the directory is never hidden.
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", provider).lstrip(".") or "provider"
    return os.path.join(data_dir, "shim-auth", slug)


def _acp_state_dir(
    data_dir: str | None, provider: str, cfg: dict[str, Any] | None = None
) -> str:
    """Return the persistent home for this provider's ACP lane.

    The default is one directory per configured provider,
    ``<data_dir>/acp/<slug>``, so two entries are two profiles, two vendor
    login stores and two processes.  Because it lives under the data
    directory, a deployment that mounts that directory keeps the vendor's
    login across restarts and across image upgrades.

    The ``acp.home`` provider setting names another directory.  An absolute
    path is used as written; a relative one is resolved under the data
    directory (or the per-user fallback when the gateway has no data
    directory).  A tilde is expanded.  A relative path may not contain ``..``:
    a profile that escapes the data directory silently stops surviving the
    image upgrade this home exists for, and an absolute path is the explicit
    way to point elsewhere.

    Raises:
        ValueError: ``acp`` is present but is not a mapping, ``acp.home`` is
            present but is not a non-empty path, a relative home climbs out
            of the data directory, or the named path is not a directory.
    """
    from ..transport.acp_profile import fallback_root, state_dir_for

    raw = (cfg or {}).get("acp")
    if raw is None:
        return state_dir_for(data_dir, provider)
    if not isinstance(raw, dict):
        raise ValueError(
            f"Provider '{provider}': acp must be a mapping of settings, got {raw!r}"
        )
    home = raw.get("home")
    if home is None:
        return state_dir_for(data_dir, provider)
    if not isinstance(home, str) or not home.strip():
        raise ValueError(
            f"Provider '{provider}': acp.home must be a non-empty path, got {home!r}"
        )
    path = os.path.expanduser(home.strip())
    if not os.path.isabs(path):
        if ".." in re.split(r"[\\/]", path):
            raise ValueError(
                f"Provider '{provider}': acp.home must stay under the data "
                f"directory; use an absolute path to point elsewhere, got {home!r}"
            )
        root = os.path.abspath(data_dir) if data_dir else str(fallback_root().parent)
        path = os.path.join(root, path)
    path = os.path.abspath(path)
    if os.path.exists(path) and not os.path.isdir(path):
        raise ValueError(f"Provider '{provider}': acp.home is not a directory: {path}")
    return path


def lane_profile_is_managed(provider: str, cfg: dict[str, Any]) -> bool:
    """Whether this entry's ACP home is derived by the gateway.

    A managed home (``<data>/acp/<slug>``) is the entry's own profile: the
    gateway may move it when the entry is renamed and remove it when the entry
    is deleted.  An ``acp.home`` or a ``--gemini-dir`` in the agent command is
    the operator's path; lifecycle actions never touch it.
    """
    acp = cfg.get("acp")
    if isinstance(acp, dict) and acp.get("home") is not None:
        return False
    argv = cfg.get("acp_argv")
    if argv is None and isinstance(acp, dict):
        argv = acp.get("argv")
    if isinstance(argv, (list, tuple)) and argv:
        from ..transport.acp_profile import gemini_dir_values

        try:
            if gemini_dir_values([str(item) for item in argv]):
                return False
        except (TypeError, ValueError):
            # A malformed command already fails the build.  Treat it as
            # operator-owned here so a cleanup never guesses a path.
            return False
    return True


def managed_acp_state_dir(
    provider: str, cfg: dict[str, Any], data_dir: str | None
) -> str | None:
    """The ACP profile the gateway owns for *provider*, or ``None``.

    ``None`` covers both "there is no managed profile" (an explicit
    ``acp.home`` or ``--gemini-dir``) and "there is no data directory to put
    one under".
    """
    if not data_dir or not lane_profile_is_managed(provider, cfg):
        return None
    from ..transport.acp_profile import state_dir_for

    return state_dir_for(data_dir, provider)


def managed_auth_state_dir(provider: str, data_dir: str | None) -> str | None:
    """The shim-exchange record directory the gateway owns for *provider*."""
    return _auth_state_dir(data_dir, provider)


def _managed_profile_dirs(
    provider_key: str,
    cfg: dict[str, Any],
    data_dir: str | None,
    *,
    lane_transport: str,
    exchange: Any | None,
) -> tuple[str | None, str | None]:
    """The two profile directories one entry owns, if any.

    The ACP home is derived only for an agent lane; the shim-exchange record
    directory exists only when the shim ships an exchange.  Either is ``None``
    when the operator named a path of their own or there is no data directory.
    """
    acp_home = (
        managed_acp_state_dir(provider_key, cfg, data_dir)
        if lane_transport == "acp"
        else None
    )
    auth_home = (
        _auth_state_dir(data_dir, provider_key) if exchange is not None else None
    )
    return acp_home, auth_home


def _profile_key(path: str) -> str:
    """A comparison key that treats one directory written two ways as one."""
    return os.path.normcase(os.path.abspath(os.path.expanduser(path)))


def duplicate_profile_errors(providers: Any) -> dict[str, str]:
    """Name each entry that repeats an earlier entry's login profile.

    The first entry in *providers* keeps the profile; every later one that
    resolves to the same ACP home or the same shim-exchange record directory
    is reported.  ``validate_provider_profiles`` raises the first message for a
    save, and the load-time recovery disables the reported entries instead of
    refusing to start.

    Args:
        providers: ``{config_name: ProviderInfo}`` for the enabled entries.

    Returns:
        ``{config_name: message}`` for the entries that must not serve.
    """
    errors: dict[str, str] = {}
    acp_homes: dict[str, str] = {}
    auth_dirs: dict[str, str] = {}
    for name, pinfo in providers.items():
        home = getattr(pinfo, "acp_state_dir", None)
        if home and getattr(pinfo, "transport", "http") != "http":
            key = _profile_key(str(home))
            other = acp_homes.get(key)
            if other is not None and other != name:
                if getattr(pinfo, "managed_acp_state_dir", None) and getattr(
                    providers[other], "managed_acp_state_dir", None
                ):
                    errors[name] = (
                        f"Providers '{other}' and '{name}' resolve to the same "
                        f"ACP profile {home!r} because their names collide "
                        "after slugging. Rename one entry or give one an "
                        "explicit acp.home."
                    )
                else:
                    errors[name] = (
                        f"Providers '{other}' and '{name}' share the ACP profile "
                        f"{home!r}. Each entry must have its own login profile; "
                        "remove one acp.home or give each entry a different path."
                    )
            else:
                acp_homes[key] = name

        context = getattr(pinfo, "auth_context", None)
        state_dir = getattr(context, "state_dir", None) if context is not None else None
        if state_dir and getattr(pinfo, "auth_exchange", None) is not None:
            key = _profile_key(str(state_dir))
            other = auth_dirs.get(key)
            if other is not None and other != name:
                errors.setdefault(
                    name,
                    f"Providers '{other}' and '{name}' share the credential "
                    f"record directory {state_dir!r}. Each entry must have its "
                    "own sign-in; rename one of the entries.",
                )
            else:
                auth_dirs[key] = name
    return errors


def validate_provider_profiles(providers: Any) -> None:
    """Refuse two entries that resolve to one login profile.

    Every entry has its own profile by default.  An explicit ``acp.home`` (or
    ``--gemini-dir``) can point two entries at one directory, and two names
    can slugify to one shim-auth directory.  Either case makes a sign-in for
    one entry silently serve the other, so it is a configuration error.

    Args:
        providers: ``{config_name: ProviderInfo}`` for the enabled entries.

    Raises:
        ValueError: Two entries resolve to the same ACP home or the same
            shim-exchange record directory.
    """
    errors = duplicate_profile_errors(providers)
    if errors:
        raise ValueError(next(iter(errors.values())))


def _auth_context(
    provider: str,
    cfg: dict[str, Any],
    state_dir: str | None,
    acp_env: dict[str, str] | None = None,
) -> Any:
    """Build the context an exchange is handed.

    The operator's key travels in ``input_key`` rather than in ``config``, so
    the secret lives in exactly one place even though the exchange is trusted.

    *acp_env* is the environment the lane's agent will actually be started
    with, after shim defaults, provider overrides and ``{acp_home}``
    expansion.  A vendor-managed exchange drives the vendor CLI itself, so it
    must see that same map: otherwise its sign-in would write to one profile
    while the agent read another.
    """
    from llm_rosetta.contracts.auth import AuthContext

    config = {k: v for k, v in cfg.items() if k != "api_key"}
    if acp_env is not None:
        config["acp_env"] = dict(acp_env)
    return AuthContext(
        provider=provider,
        config=config,
        state_dir=state_dir,
        input_key=cfg.get("api_key") or "",
    )


def account_token_for(account_id: str | None) -> str | None:
    """Return the opaque form of a shim exchange's account identity.

    The subject an exchange reports is an identity a panel may want to show;
    it is not something to publish in a log line or a metrics label.  Every
    account identity the gateway keeps is this digest, so two surfaces naming
    the same account name it the same way and no surface carries the raw
    subject.  ``None`` means the exchange names no account.
    """
    if not account_id:
        return None
    return "acct:" + hashlib.sha256(str(account_id).encode()).hexdigest()


def _credential_token(
    *,
    token_command: list[str] | None,
    api_key: str,
    keyless: bool,
    exchange: bool = False,
    account_id: str | None = None,
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
    if account_id:
        # A shim exchange names its own account identity (the signed-in
        # subject, not the bearer that subject currently holds — the access
        # token rotates, the account does not).
        return account_token_for(account_id)
    if exchange:
        # An exchange with no account identity is one provider per identity,
        # exactly like a provider with no credential.
        return None
    if token_command is not None:
        return "cmd:" + hashlib.sha256("\x00".join(token_command).encode()).hexdigest()
    keys = sorted(k.strip() for k in api_key.split(",") if k.strip())
    if not keys:
        return None
    return "key:" + hashlib.sha256("\x00".join(keys).encode()).hexdigest()


def _is_keyless(keyless_shim: bool, exchange: Any | None, cfg: dict[str, Any]) -> bool:
    """Whether the provider sends no credential at all.

    A free public gateway: keyless while no ``api_key`` and no credential
    source is configured.  Asked only after every ``api_key`` source (config,
    shim env) has been applied, so a key resolved from the environment still
    disables keyless — and an exchange that mints a credential disables it
    too.  A ``vendor_managed`` exchange is the exception: the vendor CLI owns
    the credential, so the lane stays keyless and the gateway never seeds it.
    """
    return bool(
        keyless_shim
        and (exchange is None or getattr(exchange, "vendor_managed", False))
        and not cfg.get("api_key")
        and not cfg.get("token_command")
    )


def _credential_source_name(
    *,
    keyless: bool,
    provider_owns_command: bool,
    exchange: Any | None,
    token_command: list[str] | None,
    vendor_managed: bool = False,
) -> str:
    """Name the effective credential source for the admin editor.

    The builder, not the panel, knows the precedence: a provider-level command
    outranks the shim's declaration, and a shim ships either ``auth.py`` or a
    command, never both.  A vendor-managed exchange is named for itself even
    though the lane is keyless: it is still the only surface the editor has.
    """
    if vendor_managed:
        return "auth_exchange"
    if keyless:
        return "none"
    if provider_owns_command:
        return "config_command"
    if exchange is not None:
        return "auth_exchange"
    if token_command is not None:
        return "shim_command"
    return "static_key"


def _resolve_credential(
    name: str,
    cfg: dict[str, Any],
    exchange: Any | None,
    *,
    keyless: bool,
    quiet: bool = False,
) -> tuple[str, list[str] | None, int, bool]:
    """Decide where the provider's credential comes from.

    Returns ``(api_key, token_command, refresh_interval, exchange_takes_key)``:
    a static key, an argv command to run, or a key-ring placeholder the shim's
    exchange fills in after the event loop starts.

    Raises:
        ValueError: An ``api_key`` sits beside an exchange that does not take
            it as input, or an exchange's interval is below the minimum.
    """
    exchange_takes_key = bool(exchange is not None and _exchange_consumes_key(exchange))
    if exchange is not None and cfg.get("api_key") and not exchange_takes_key:
        raise ValueError(
            f"Provider '{name}': a shim-declared credential exchange and "
            "api_key are mutually exclusive; remove api_key, or have the shim "
            "declare consumes_key if the key is the exchange's input"
        )

    if keyless:
        # No credential source, so no refresher is ever scheduled and the
        # interval is inert — 0 rather than the 3600 default, so it cannot read
        # as "refreshes hourly".  (It skips _resolve_token_command's >= 60
        # check, which only guards the command path.)
        return "", None, 0, False
    if exchange is not None:
        from ..transport.provider_info import TOKEN_PENDING_SENTINEL

        interval = int(cfg.get("token_refresh_interval") or 3600)
        if interval < 60:
            raise ValueError(
                f"Provider '{name}': token_refresh_interval must be >= 60, "
                f"got {interval}"
            )
        logger.log(
            logging.DEBUG if quiet else logging.INFO,
            "Provider '%s' mints its credential in the shim's auth.py — "
            "credential will be fetched asynchronously after startup",
            name,
        )
        # Until the deferred seed runs, the ring holds the pending sentinel and
        # the provider is not ready.
        return TOKEN_PENDING_SENTINEL, None, interval, exchange_takes_key

    api_key, token_command, interval = _resolve_token_command(name, cfg, quiet=quiet)
    return api_key, token_command, interval, exchange_takes_key


def _resolve_lane_transport(
    cfg: dict[str, Any], shim: Any | None, provider_key: str
) -> str:
    """Return the transport that carries one provider's lane.

    A config key wins over the shim's declaration; a name the gateway has no
    transport for is refused here, naming the provider it belongs to, rather
    than on that provider's first request.

    Args:
        cfg: The provider's own config block.
        shim: The provider's shim, whose ``connection.transport`` supplies the
            default when the config names none.
        provider_key: The config key, used in the error message.

    Returns:
        The transport name.

    Raises:
        ValueError: The name is not one the gateway ships.
    """
    from ..transport import TRANSPORT_NAMES

    lane_transport = str(
        cfg.get("transport")
        or (shim.connection.transport if shim is not None else "http")
    )
    if lane_transport not in TRANSPORT_NAMES:
        raise ValueError(
            f"Provider '{provider_key}': unknown transport {lane_transport!r}; "
            f"available: {', '.join(TRANSPORT_NAMES)}"
        )
    return lane_transport


def _argv_value(value: Any, provider_key: str, label: str) -> tuple[str, ...] | None:
    """Check a declared agent command and normalise it to a tuple.

    A command is a non-empty list of non-empty strings.  Refusing a typo here
    keeps it beside the provider that declared it, rather than surfacing as a
    start failure on the lane's first request.

    Raises:
        ValueError: The value is not a list/tuple of non-empty strings.
    """
    if value is None:
        return None
    if (
        not isinstance(value, (list, tuple))
        or not value
        or not all(isinstance(part, str) and part for part in value)
    ):
        raise ValueError(
            f"Provider '{provider_key}': {label} must be a non-empty list of "
            f"strings, got {value!r}"
        )
    return tuple(value)


def _resolve_acp_argv(
    cfg: dict[str, Any], shim: Any | None, provider_key: str
) -> tuple[str, ...] | None:
    """The command that starts this lane's agent, and the version it selects.

    Three sources, in order: ``acp.argv`` in the provider entry, the older
    top-level ``acp_argv``, then the shim's ``connection.acp_argv``.  The
    command names an *install*, not a profile: pointing it at another version
    of the vendor CLI leaves ``acp.home`` and the vendor's login store where
    they are, so an image upgrade or a version switch does not cost a sign-in.
    """
    block = cfg.get("acp")
    if isinstance(block, dict):
        declared = _argv_value(block.get("argv"), provider_key, "acp.argv")
        if declared is not None:
            return declared
    declared = _argv_value(cfg.get("acp_argv"), provider_key, "acp_argv")
    if declared is not None:
        return declared
    if shim is not None:
        return _argv_value(
            shim.connection.acp_argv, provider_key, "connection.acp_argv"
        )
    return None


def _resolve_acp_harden(cfg: dict[str, Any], provider_key: str) -> bool | None:
    """The entry's explicit hardening choice, or ``None`` for "by name".

    The three states are deliberate.  Absent (``None``) hardens a command
    whose executable is named ``agy`` — the ordinary install.  ``True``
    forces the gateway-owned profile on whatever command the entry names, for
    a renamed binary (``agy-1.7``) or a wrapper that forwards agy's flags.
    ``False`` runs the command as written.

    The effective answer is :func:`_effective_harden`; this function only
    reads and validates the operator's declaration.

    Raises:
        ValueError: ``acp.harden`` is present but is not a boolean.
    """
    block = cfg.get("acp")
    if not isinstance(block, dict):
        return None
    value = block.get("harden")
    if value is None:
        return None
    if not isinstance(value, bool):
        raise ValueError(
            f"Provider '{provider_key}': acp.harden must be true or false, "
            f"got {value!r}"
        )
    return value


def _effective_harden(declared: bool | None, argv: tuple[str, ...] | None) -> bool:
    """Whether the gateway gives this lane's agent its own profile.

    An absent ``acp.harden`` follows the command's name, so the shim's default
    ``agy`` command is hardened without the operator repeating it.  An
    explicit ``True`` forces the profile on a renamed or wrapped command; an
    explicit ``False`` runs it as written.
    """
    if declared is not None:
        return declared
    if argv is None:
        # An ACP lane with no command of its own runs the transport's default
        # agent, which is ``agy`` and therefore hardened.
        return True
    from ..transport.acp_profile import is_agy

    return is_agy(list(argv))


#: Provider/argv pairs already warned about.  ``build_provider_info`` runs on
#: the request path, so an unhardened wrapper must not log on every turn.
_NOT_HARDENED_WARNED: set[tuple[str, tuple[str, ...]]] = set()


def _warn_not_hardened(provider_key: str, argv: tuple[str, ...]) -> None:
    """Say once that a replaced agent command does not get the profile.

    A shim that starts ``agy`` declares the profile the lane expects.  When
    the entry replaces that command with something not named ``agy`` — a
    wrapper, a renamed download — the name-based default cannot tell it is
    the same agent, so the tool gate is silently absent.  The operator either
    forces the profile with ``acp.harden: true`` or accepts the command as
    written with ``false``.
    """
    key = (provider_key, tuple(argv))
    if key in _NOT_HARDENED_WARNED:
        return
    _NOT_HARDENED_WARNED.add(key)
    logger.warning(
        "Provider '%s': the agent command %r is not named 'agy', so the "
        "gateway-owned profile is not applied; set acp.harden: true to force "
        "it on this command, or false to accept it.",
        provider_key,
        list(argv),
    )


def _resolve_acp_install(
    cfg: dict[str, Any],
    shim: Any | None,
    provider_key: str,
    lane_transport: str,
) -> tuple[tuple[str, ...] | None, bool]:
    """Return the lane's install command and whether its profile is ours.

    Only an ACP lane has an install to resolve.  A shim that starts ``agy``
    but is overridden by a command of another name gets one warning: the
    name-based default cannot see that the unknown command is the same agent.
    """
    if lane_transport != "acp":
        return None, True
    argv = _resolve_acp_argv(cfg, shim, provider_key)
    declared = _resolve_acp_harden(cfg, provider_key)
    if (
        declared is None
        and shim is not None
        and shim.connection.acp_argv
        and argv is not None
    ):
        from ..transport.acp_profile import is_agy

        if is_agy(list(shim.connection.acp_argv)) and not is_agy(list(argv)):
            _warn_not_hardened(provider_key, argv)
    return argv, _effective_harden(declared, argv)


def _acp_env(
    cfg: dict[str, Any],
    shim: Any | None,
    provider_key: str,
    home: str | None = None,
    *,
    inject_home: bool = False,
) -> dict[str, str] | None:
    """Environment overrides for this provider's ACP agent process.

    The shim declares a vendor switch (Claude Code needs
    ``ENABLE_TOOL_SEARCH=false`` and its config directory); the provider entry
    may add deployment facts without repeating the vendor's.  The operator's
    values win.

    *home* is the lane's persistent directory.  A shim may name it as
    ``{acp_home}`` (``{acp_state_dir}`` is kept as an alias), which keeps a
    vendor's login in the gateway's data directory instead of the process's
    ``HOME``.  With *inject_home*, the gateway publishes it as
    ``LLM_ROSETTA_ACP_HOME`` for a shim that drives the vendor CLI itself.
    That variable is gateway-owned: an ``acp_env`` value is overwritten, so
    the agent and the sign-in driver cannot be split across two profiles.
    ``acp.home`` is the supported way to move the profile.
    """
    env = dict(shim.connection.acp_env or {}) if shim is not None else {}
    override = cfg.get("acp_env")
    if override is not None:
        if not isinstance(override, dict):
            raise ValueError(
                f"Provider '{provider_key}': acp_env must be a mapping of "
                f"environment names to strings, got {override!r}"
            )
        for name, value in override.items():
            if not isinstance(name, str) or not isinstance(value, str):
                raise ValueError(
                    f"Provider '{provider_key}': acp_env must map strings to "
                    f"strings; {name!r} has value {value!r}"
                )
        env.update(override)
    if home:
        env = {
            name: value.replace("{acp_home}", home).replace("{acp_state_dir}", home)
            for name, value in env.items()
        }
        if inject_home:
            env["LLM_ROSETTA_ACP_HOME"] = home
    return env or None


def _lane_environment(
    cfg: dict[str, Any],
    shim: Any | None,
    provider_key: str,
    data_dir: str | None,
    lane_transport: str,
    argv: tuple[str, ...] | None = None,
    harden: bool = False,
) -> tuple[str, dict[str, str] | None]:
    """Return the persistent home and the environment one lane starts with.

    An ACP lane gets a home under the data directory, its ``acp.home``
    override and the ``{acp_home}`` expansion.  A lane that speaks HTTP starts
    no agent, so it ignores those settings and keeps its ``acp_env`` map as
    written.

    An ``agy`` command may already name ``--gemini-dir``.  That directory is
    the profile the transport writes into, so it is also the home the sign-in
    and the environment must name.  The flag is validated here — one
    occurrence, a non-empty absolute path — because a malformed one would
    otherwise let hardening consume the next injected flag as its value.  When
    ``acp.home`` names a different directory, the two are refused rather than
    split silently across the agent and its sign-in.

    An unhardened ``agy`` command with no ``--gemini-dir`` is refused: the
    agent would read the CLI's own profile while the gateway publishes a home
    nobody signs in to.  Naming the directory, or leaving hardening on, keeps
    the contract true on every branch.
    """
    if lane_transport != "acp":
        return _acp_state_dir(data_dir, provider_key), _acp_env(cfg, shim, provider_key)
    home = _acp_state_dir(data_dir, provider_key, cfg)
    if argv:
        from ..transport.acp_profile import gemini_dir_values, is_agy

        agy = is_agy(list(argv))
        named: str | None = None
        # Any command may already name the profile it reads, a wrapper
        # included.  The home the sign-in and the panel use must be that
        # directory whether or not the gateway injects the flag itself.
        values = gemini_dir_values(list(argv))
        if len(values) > 1:
            raise ValueError(
                f"Provider '{provider_key}': the agent command names "
                f"--gemini-dir more than once ({values!r}); keep one "
                f"profile"
            )
        if values:
            named = values[0]
            if not named or named.startswith("-"):
                raise ValueError(
                    f"Provider '{provider_key}': --gemini-dir in the agent "
                    f"command must be followed by a directory, not a flag "
                    f"or nothing"
                )
        if agy and not harden and named is None:
            raise ValueError(
                f"Provider '{provider_key}': acp.harden: false runs agy as "
                f"written, so the profile it reads is not knowable here. Name "
                f"it with --gemini-dir <dir>, or leave hardening on."
            )
        if named is not None:
            expanded = os.path.expanduser(named)
            if not os.path.isabs(expanded):
                raise ValueError(
                    f"Provider '{provider_key}': the --gemini-dir in the agent "
                    f"command must be an absolute path (or move the profile "
                    f"with acp.home), got {named!r}"
                )
            resolved = os.path.abspath(expanded)
            block = cfg.get("acp")
            explicit = isinstance(block, dict) and block.get("home") is not None
            if explicit and home != resolved:
                raise ValueError(
                    f"Provider '{provider_key}': acp.home and the "
                    f"--gemini-dir in the agent command name different "
                    f"directories ({home!r} and {resolved!r}); keep one, or "
                    f"make them equal"
                )
            home = resolved
    return home, _acp_env(cfg, shim, provider_key, home, inject_home=True)


def _resolve_acp_limits(cfg: dict[str, Any], provider_key: str) -> dict[str, Any]:
    """Read one provider's agent-process and session limits.

    The ``acp`` block is the operator's, not the shim's: how many processes to
    run is a fact about this deployment's memory, and how long to keep a
    session is a fact about how its clients work.  Every key is optional, and
    an absent one leaves the transport's default in place.

    Args:
        cfg: The provider's own config block.
        provider_key: The config key, used in error messages.

    Returns:
        The keyword arguments :class:`ProviderInfo` takes for these limits.

    Raises:
        ValueError: A value is present but is not a positive number.
    """
    raw = cfg.get("acp")
    block: dict[str, Any] = raw if isinstance(raw, dict) else {}
    limits: dict[str, Any] = {}
    processes = block.get("processes")
    if processes is not None:
        if not isinstance(processes, int) or isinstance(processes, bool):
            raise ValueError(
                f"Provider '{provider_key}': acp.processes must be an integer, "
                f"got {processes!r}"
            )
        if processes < 1:
            raise ValueError(
                f"Provider '{provider_key}': acp.processes must be >= 1, "
                f"got {processes}"
            )
        limits["acp_processes"] = processes
    max_sessions = block.get("max_sessions")
    if max_sessions is not None:
        if not isinstance(max_sessions, int) or isinstance(max_sessions, bool):
            raise ValueError(
                f"Provider '{provider_key}': acp.max_sessions must be an "
                f"integer, got {max_sessions!r}"
            )
        limits["acp_max_sessions"] = max_sessions
    ttl = block.get("session_ttl")
    if ttl is not None:
        if not isinstance(ttl, (int, float)) or isinstance(ttl, bool):
            raise ValueError(
                f"Provider '{provider_key}': acp.session_ttl must be a number "
                f"of seconds, got {ttl!r}"
            )
        limits["acp_session_ttl"] = float(ttl)
    return limits


def _bind_auth_exchange(
    provider_key: str,
    cfg: dict[str, Any],
    exchange: Any | None,
    data_dir: str | None,
    acp_env: dict[str, str] | None = None,
) -> tuple[Any | None, str | None]:
    """Build an exchange's context and read the account identity it implies.

    Returns ``(context, account_id)``; both ``None`` without an exchange.  A
    record the exchange cannot read costs the provider its account identity,
    not its configuration — the credential may yet be minted.

    *acp_env* is forwarded to the context, so a vendor-managed exchange drives
    the vendor CLI in the same profile the ACP transport will start it with.
    """
    if exchange is None:
        return None, None
    context = _auth_context(
        provider_key, cfg, _auth_state_dir(data_dir, provider_key), acp_env
    )
    try:
        return context, exchange.account_id(context)
    except Exception as exc:  # noqa: BLE001 - a broken record must not abort config
        logger.warning(
            "Provider '%s': credential exchange could not read its account "
            "identity: %s",
            provider_key,
            exc,
        )
        return context, None


def build_provider_info(
    provider_type: str,
    cfg: dict[str, Any],
    *,
    global_proxy: str | None = None,
    shim_name: str | None = None,
    data_dir: str | None = None,
    config_name: str | None = None,
    quiet: bool = False,
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

    *data_dir* is the gateway's data directory, already resolved to an
    absolute path.  A shim-declared credential exchange persists its record
    under it, one directory per provider; ``None`` (the default) leaves the
    location to the exchange's own default.

    *config_name* is the provider's key in the config (``"chatgpt-personal"``),
    as opposed to *provider_type* (the format, ``"open_responses"``).  A shim
    that mints its own credential needs it to tell two configured providers of
    the same shim apart — they are two accounts, not one.

    *quiet* asks the builder not to repeat its per-provider notices (the
    ``token_command`` executable check, the exchange notice) at their normal
    level.  The admin's pre-write validation rebuilds every enabled provider
    on each save, and those lines describe the provider rather than that save.

    For known provider types the auth and URL logic is looked up from the
    registry.  Unknown types fall back to Bearer-token auth and a simple
    ``{base_url}/`` URL template.
    """
    from llm_rosetta.api import get_shim

    from ..shims import legacy_shim_migration_advice, source_package_advice

    # A retired spelling (e.g. ``openai_chat``) is an error that names its
    # replacement, not a name the registry lookup below could resolve.
    from llm_rosetta.api import retired_names

    retired_names.check_not_retired(provider_type, retired_names.PROVIDER_TYPE)

    # Resolve through shim registry for defaults
    shim = get_shim(shim_name or provider_type)
    # A provider-level ``token_command`` is the operator's own exchange and
    # outranks the shim's, whichever form the shim declared.
    provider_owns_command = cfg.get("token_command") is not None
    shim_exchange = getattr(shim, "auth", None) if shim is not None else None
    # Read this before a provider command hides the shim's exchange: the
    # vendor-managed rule is a property of the shim, not of the resolved one.
    vendor_managed = bool(getattr(shim_exchange, "vendor_managed", False))
    exchange = shim_exchange if shim is not None and not provider_owns_command else None
    if shim is not None:
        base_type = shim.format
        # Apply shim defaults where the config value is missing *or empty*.
        cfg = _apply_shim_defaults(cfg, shim, exchange)
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

    # Keyless upstream (a free public gateway).  Supplying a key or a
    # credential source switches back to the base type's auth, so a paid
    # account can use the same shim.
    keyless = _is_keyless(keyless_shim, exchange, cfg)
    if keyless:
        auth_fn = no_auth
    if vendor_managed:
        if not keyless_shim:
            raise ValueError(
                f"Provider '{config_name or shim_name or provider_type}': its "
                "auth.py is vendor-managed, so the vendor CLI owns the credential "
                "and the shim must declare connection.keyless: true"
            )
        if provider_owns_command:
            # A command would mint a gateway token again, which is the one
            # thing a vendor-managed lane says it never does.  The exchange
            # was dropped above, so report the shim's rule here.
            raise ValueError(
                f"Provider '{config_name or shim_name or provider_type}': its "
                "auth.py is vendor-managed, so the vendor CLI owns the "
                "credential; remove token_command, or use a shim without that "
                "exchange"
            )

    # Per-provider proxy overrides global proxy
    proxy_url = cfg.get("proxy") or global_proxy or None

    # -- the credential: static key, command, or shim exchange -----------------
    api_key, token_command, token_refresh_interval, exchange_takes_key = (
        _resolve_credential(
            shim_name or provider_type, cfg, exchange, keyless=keyless, quiet=quiet
        )
    )

    if not cfg.get("base_url"):
        raise ValueError(
            _missing_base_url_message(provider_type, base_type, shim_name, shim)
        )

    # The config key, when the caller has one: two providers of the same shim
    # are two accounts, and the exchange must be able to tell them apart.
    provider_key = config_name or shim_name or provider_type

    # Which transport carries this lane.  A config key wins over the shim's
    # declaration, and a name with no transport behind it is refused here —
    # beside the provider it belongs to — rather than on its first request.
    lane_transport = _resolve_lane_transport(cfg, shim, provider_key)
    acp_limits = _resolve_acp_limits(cfg, provider_key)

    quota_raw = cfg.get("quota")
    quota_cfg: dict[str, Any] = quota_raw if isinstance(quota_raw, dict) else {}
    shim_quota = getattr(shim, "quota", None) if shim is not None else None
    quota_probe = quota_cfg.get("probe") or (
        shim_quota.probe if shim_quota is not None else None
    )
    quota_url = quota_cfg.get("url") or (
        shim_quota.url if shim_quota is not None else None
    )
    # A lane with no usage endpoint: the probe is kept for its in-band
    # classifier, and nothing polls it.
    quota_in_band = bool(
        quota_cfg.get("in_band", False)
        or (shim_quota.in_band if shim_quota is not None else False)
    )

    command_takes_key = bool(token_command and cfg.get("token_command_env"))

    # The command names an install; the home names the profile.  They are read
    # separately so that moving to another vendor CLI version — another argv —
    # keeps the login store, and so the transport knows which command it may
    # harden.  The home lives under the data directory by default, so a mounted
    # volume carries the login across restarts and image upgrades.  The
    # effective environment is built once: the ACP spawn, the vendor sign-in
    # driver and the admin's status probe must all name the same profile.
    acp_argv, acp_harden = _resolve_acp_install(cfg, shim, provider_key, lane_transport)
    acp_home, effective_acp_env = _lane_environment(
        cfg, shim, provider_key, data_dir, lane_transport, acp_argv, acp_harden
    )

    # Name the effective source for the admin editor.
    credential_source = _credential_source_name(
        keyless=keyless,
        provider_owns_command=provider_owns_command,
        exchange=exchange,
        token_command=token_command,
        vendor_managed=vendor_managed,
    )

    # The config key, when the caller has one: two providers of the same shim
    # are two accounts, and the exchange must be able to tell them apart.
    auth_context, auth_account_id = _bind_auth_exchange(
        provider_key, cfg, exchange, data_dir, effective_acp_env
    )

    # Which profile directories this entry owns, for the admin lifecycle.
    # Computed here, where the builder already knows whether the path was
    # chosen by the gateway or written by the operator, so the admin layer
    # needs no config import to answer the same question.
    managed_acp_home, managed_auth_home = _managed_profile_dirs(
        provider_key,
        cfg,
        data_dir,
        lane_transport=lane_transport,
        exchange=exchange,
    )

    return ProviderInfo(
        name=provider_type,
        config_name=provider_key,
        api_key=api_key,
        base_url=cfg["base_url"],
        auth_header_fn=auth_fn,
        url_template=url_tpl,
        stream_url_template=stream_tpl,
        proxy_url=proxy_url,
        timeout=float(cfg["timeout"]) if "timeout" in cfg else None,
        token_command=token_command,
        token_refresh_interval=token_refresh_interval,
        token_command_env=(cfg.get("token_command_env") if token_command else None),
        # The operator's key, when a credential source consumes it as input
        # (the command's environment channel, or an exchange that declares
        # ``consumes_key``).  It is never sent upstream itself.
        token_input=(
            (cfg.get("api_key") or "")
            if (token_command and cfg.get("token_command_env")) or exchange_takes_key
            else ""
        ),
        auth_exchange=exchange,
        auth_context=auth_context,
        keyless=keyless,
        # When the exchange consumes the operator's key, the account is that
        # key's account — hashing the command alone would merge every account
        # of the vendor into one identity.
        credential_token=_credential_token(
            token_command=(None if command_takes_key else token_command),
            api_key=(cfg.get("api_key") or "") if command_takes_key else api_key,
            keyless=keyless,
            exchange=exchange is not None,
            account_id=auth_account_id,
        ),
        credential_source=credential_source,
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
        transport=lane_transport,
        acp_argv=acp_argv,
        acp_argv_custom=(
            cfg.get("acp_argv") is not None
            or (isinstance(cfg.get("acp"), dict) and cfg["acp"].get("argv") is not None)
        ),
        acp_harden=acp_harden,
        acp_env=effective_acp_env,
        acp_state_dir=acp_home,
        managed_acp_state_dir=managed_acp_home,
        managed_auth_state_dir=managed_auth_home,
        **acp_limits,
        quota_probe_id=quota_probe,
        quota_url=quota_url,
        quota_in_band=quota_in_band,
        quota_probes=_quota_probes(shim, quota_cfg.get("probe")),
    )


def _missing_base_url_message(
    provider_type: str,
    base_type: str,
    shim_name: str | None,
    shim: Any,
) -> str:
    """The error for a provider with no ``base_url`` from its config or its shim.

    A shim that a standalone package provides, or whose vendor moved out of
    core (#1660), is named with where it comes from: that is the usual reason
    no shim supplied the URL.
    """
    from llm_rosetta.api import retired_names

    from ..shims.catalog import find_standalone

    missing_package = (
        None if shim is not None else find_standalone(shim_name or provider_type)
    )

    message = (
        f"Provider {provider_type!r} (base type {base_type!r}) has no "
        "base_url configured; set it in the provider config or via a shim "
        "connection.base_url."
    )
    if missing_package is not None:
        name = shim_name or provider_type
        message += (
            f" Shim {name!r} is not built in: it ships in the standalone "
            f"package {missing_package.label!r}. Declare "
            f"`shims.install: [{missing_package.install!r}]`; "
            "Settings -> Shims offers it. If the source is already "
            "declared, check the shim-source status for a load failure."
        )
    moved = (
        retired_names.moved_shim_message(shim_name or provider_type)
        if shim is None
        else None
    )
    if moved is not None:
        message += (
            f" The {moved}. If it is still missing, check the shim-source "
            "status for the install failure."
        )
    return message


def _quota_probes(shim: Any, probe_id: Any) -> tuple[Any, ...]:
    """The probes a provider can resolve ``quota.probe`` against.

    A shim's own ``quota.py`` comes first.  A probe the config names
    explicitly that the shim does not ship is looked up among the other
    registered shims: a probe that moved out of core with its vendor
    (``openai_codex`` now ships with ``openai_chatgpt`` in contrib, #1660)
    still serves a provider on another shim that names it, once that vendor
    is installed.  Only an explicit config value reaches this far, so no
    provider picks up a foreign probe by itself.
    """
    own = tuple(shim.quota_probes) if shim is not None else ()
    if not isinstance(probe_id, str) or any(p.id == probe_id for p in own):
        return own
    from llm_rosetta.api import list_shims, retired_names

    for other in list_shims():
        for probe in other.quota_probes:
            if probe.id == probe_id:
                return (*own, probe)
    moved = retired_names.moved_probe_vendor(probe_id)
    if moved is not None:
        logger.warning(
            "Quota probe %r moved from core to %s with vendor %r and is not "
            "installed; the gateway installs that vendor automatically, so check "
            "the shim-source status",
            probe_id,
            moved.package,
            moved.vendor,
        )
    return own


#: The credential state a second endpoint of one provider reads from the lane
#: that owns it.  Kept as a name list rather than a copy loop over ``__dict__``
#: so a field added to :class:`ProviderInfo` later is shared only by an
#: explicit decision.
_CREDENTIAL_FIELDS: tuple[str, ...] = (
    "config_name",
    "token_command",
    "token_refresh_interval",
    "token_command_env",
    "token_input",
    "auth_exchange",
    "auth_context",
    "keyless",
    "credential_token",
    "credential_source",
    "quota_identity",
    "quota_provider_blocked",
)


def share_credential(target: ProviderInfo, owner: ProviderInfo) -> None:
    """Make *target* read *owner*'s credential instead of holding its own.

    A typed endpoint (embeddings, rerank, decisions) is a second endpoint of a
    provider entry, not a second provider: it must forward whatever the entry
    minted.  Sharing the **key ring object** — not a copy of its keys — is what
    gives that for free, because every path that renews a credential
    (``DeferredStartup`` seeding, the scheduled refresher, the reactive 401
    retry, the panel's "refresh now") writes through ``owner.key_ring``.  No
    path is added for typed endpoints, so one credential source keeps exactly
    one mint and one refresh loop.

    The routing state that belongs to the account rather than the endpoint is
    carried over too: a provider-wide quota block or a rate-limit disable is a
    fact about the credential, so it must apply to every endpoint of it.

    **The quota boundary.**  What is shared is the credential's *eligibility*:
    the key ring's ``usable`` predicate (a demoted or 429-disabled key is
    skipped), the provider-wide block, and the credential identity.  What is
    not shared is the allowance *measurement*: ``quota_probe`` and
    ``quota_interval`` stay on the owner, so ``provider_headroom`` is ``None``
    for a typed endpoint and ``provider_spent`` never excludes one.  Typed
    routing therefore spends no part of the headroom-weighting budget — it is
    excluded only by what the credential itself reports.  Copying the probe
    here would leave a declaration nothing reads: the poller walks
    ``config.providers`` only.

    Args:
        target: The typed endpoint's own :class:`ProviderInfo`.
        owner: The provider entry in the LLM pool that mints the credential.
    """
    target.key_ring = owner.key_ring
    for field in _CREDENTIAL_FIELDS:
        setattr(target, field, getattr(owner, field))
    # ``token_status`` reads through this link rather than copying the dict:
    # a copy would stop describing the credential the moment it refreshed.
    target.credential_owner = owner
