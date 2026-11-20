"""The typed (non-LLM) endpoint pools: parsing, and the shim resolver.

Driven by the model-type descriptor, so embedding, rerank and decision share
one implementation; :func:`parse_typed_config` documents the precedence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..transport import ProviderInfo
from .errors import RetiredConfigName
from .parse import first_pool_provider

if TYPE_CHECKING:
    from ..model_types import ModelTypeDescriptor

logger = logging.getLogger("llm-rosetta-gateway")


def retired_spellings(type_name: str) -> tuple[str, ...]:
    """The retired spellings of the model type *type_name*.

    ``embeddings`` was ``embedding``.  Empty for a type that was never renamed.
    """
    from llm_rosetta.api import retired_names

    return tuple(
        row.name
        for row in retired_names.RETIRED_NAMES
        if row.kind == retired_names.MODEL_TYPE and row.replacement == type_name
    )


def reject_retired_keys(
    cfg: dict[str, Any], type_name: str, where: str | None = None
) -> None:
    """Raise if *cfg* carries a key spelled with a retired model type.

    ``embedding_format``, ``embedding_path``, ``embedding_shim`` and the nested
    ``embedding:`` block are no longer read: the config loader answers with the
    key that replaces each, and the ``llm-rosetta-migrate`` command.

    Args:
        cfg: A provider entry.
        type_name: The current model type.
        where: The entry's JSON path (``providers.<name>``), named in the error.

    Raises:
        RetiredConfigName: Naming the first such key and its replacement.
    """
    from llm_rosetta.api import retired_names

    for old in retired_spellings(type_name):
        prefix = f"{old}_"
        for key in cfg:
            if key == old or (isinstance(key, str) and key.startswith(prefix)):
                new = type_name if key == old else f"{type_name}_" + key[len(prefix) :]
                raise RetiredConfigName(
                    retired_names.retired_key_message(str(key), new),
                    where=f"{where}.{key}" if where else None,
                )


def reject_retired_sections(raw: dict[str, Any], type_name: str) -> None:
    """Raise if the config carries a key spelled with a retired model type.

    The top-level ``embedding_providers``, ``embedding_models`` and
    ``default_embedding_format``, and the ``embedding_*`` keys and ``embedding:``
    block of any provider entry, are no longer read as the pools and endpoint
    settings of ``embeddings``.  The check runs at load: an endpoint that fails
    to parse is only disabled, and a retired spelling must not be quietly
    dropped that way.

    Raises:
        RetiredConfigName: Naming the first such key and the key that replaces it.
    """
    from llm_rosetta.api import retired_names

    for name, entry in (raw.get("providers") or {}).items():
        if isinstance(entry, dict):
            reject_retired_keys(entry, type_name, f"providers.{name}")
    for section in (
        f"{type_name}_providers",
        *(f"{old}_providers" for old in retired_spellings(type_name)),
    ):
        for name, entry in (raw.get(section) or {}).items():
            if isinstance(entry, dict):
                reject_retired_keys(entry, type_name, f"{section}.{name}")

    for old in retired_spellings(type_name):
        for key in raw:
            if not isinstance(key, str):
                continue
            if key.startswith(f"{old}_"):
                new = f"{type_name}_" + key[len(old) + 1 :]
            elif key.startswith(f"default_{old}_"):
                new = f"default_{type_name}_" + key[len(old) + 9 :]
            else:
                continue
            raise RetiredConfigName(
                retired_names.retired_key_message(key, new), where=key
            )


def _declared(cfg: dict[str, Any], key: str, type_name: str) -> Any:
    """Read *key*: present and non-empty, or ``None``.

    Presence decides, so ``False`` stays a value for an ``*_enabled`` key.
    """
    if key in cfg:
        return cfg[key]
    return None


def face_section(cfg: dict[str, Any], type_name: str) -> dict[str, Any]:
    """The ``<type>:`` block one provider entry declares for an endpoint.

    A provider is an *account*; each model type is a *face* of it, and the two
    are not the same fact.  An aggregator key signs one vendor's chat lane and
    another vendor's embeddings lane on the same account, so the endpoint's
    host is a property of the face:

    .. code-block:: jsonc

        "frankenservice": {
          "type": "google_generate",
          "base_url": "https://generativelanguage.googleapis.com",
          "api_key": "...",
          "embeddings": {"shim": "cohere--embedding",
                         "base_url": "https://api.cohere.com/v2"}
        }

    The flat ``embeddings_shim`` / ``embeddings_format`` / ``embeddings_path``
    / ``embeddings_enabled`` keys are the same block spelled without the
    nesting, kept readable (and written into the block by the migration) for
    configs written before it existed — and are what the legacy
    ``<type>_providers`` sections use, since each of those is one face already.

    Args:
        cfg: A provider entry.
        type_name: The model type, e.g. ``"embeddings"``.

    Returns:
        The block as a mapping (possibly empty).  A block that is not a
        mapping is reported and treated as absent rather than crashing the
        parse of every other endpoint on the entry.
    """
    reject_retired_keys(cfg, type_name)
    block = cfg.get(type_name)
    if block is None:
        return {}
    if not isinstance(block, dict):
        logger.warning(
            "config: %r must be a mapping of endpoint settings, got %s; "
            "the key is ignored",
            type_name,
            type(block).__name__,
        )
        return {}
    return block


def face_value(cfg: dict[str, Any], key: str, type_name: str) -> Any:
    """Read one endpoint setting: the face block first, then the flat key.

    The block's field name is the flat key without its ``<type>_`` prefix —
    ``embeddings_shim`` → ``embeddings: {shim: …}``, ``embeddings_path`` →
    ``path``, ``embeddings_enabled`` → ``enabled``.

    An explicit value in the block wins over the flat key, and an empty one
    falls through — the same "an empty value means the operator said nothing"
    rule :func:`_declared` applies, so a form that writes ``""`` into the
    block cannot hide a value the flat key still holds.  ``enabled`` decides
    by presence, so ``false`` is a value there like it is in the flat key.
    """
    field = key[len(type_name) + 1 :] if key.startswith(f"{type_name}_") else key
    block = face_section(cfg, type_name)
    if field in block:
        value = block[field]
        if field == "enabled" or isinstance(value, bool) or value:
            return value
    return _declared(cfg, key, type_name)


@dataclass(frozen=True)
class _TypedKeys:
    """The config keys and defaults one typed endpoint is parsed with."""

    type_name: str
    format_key: str
    path_key: str
    shim_key: str
    enabled_key: str
    default_path: str
    default_format: str

    @classmethod
    def for_descriptor(
        cls, descriptor: ModelTypeDescriptor, format_key: str | None = None
    ) -> _TypedKeys:
        """The keys *descriptor* declares, with *format_key* overriding one."""
        name = descriptor.name
        return cls(
            type_name=name,
            format_key=format_key or descriptor.config_format_key or f"{name}_format",
            path_key=descriptor.config_path_key or f"{name}_path",
            shim_key=descriptor.config_shim_key or f"{name}_shim",
            enabled_key=f"{name}_enabled",
            default_path=descriptor.default_path or "",
            default_format=descriptor.default_format or "",
        )


def _typed_descriptors() -> list[ModelTypeDescriptor]:
    """The registered model types that carry a typed (non-LLM) endpoint.

    A descriptor joins the loop by declaring ``config_format_key``; ``llm``
    routes through the chat pipeline instead and has none.
    """
    from ..model_types import all_model_types

    return [
        desc
        for desc in all_model_types()
        if not desc.is_llm and desc.config_format_key is not None
    ]


# ---- Typed (non-LLM) endpoint parsing --------------------------------
def parse_typed_config(
    raw_providers: dict[str, Any],
    raw_models: dict[str, Any],
    *,
    descriptor: ModelTypeDescriptor,
    format_key: str | None = None,
    require_declared_format: bool = False,
    map_models: bool = True,
    global_proxy: str | None = None,
    credential_owner: dict[str, ProviderInfo] | None = None,
    refuse_unowned_minted: bool = True,
) -> tuple[dict[str, dict[str, Any]], dict[str, ProviderInfo], dict[str, str]]:
    """Parse one typed endpoint out of a provider section.

    The unified ``providers:`` entries and the legacy
    ``<type>_providers`` section both come through here, so embedding,
    rerank and decision cannot drift apart.  :meth:`_parse_typed_provider`
    documents the shim / format / path / connection precedence.

    Args:
        raw_providers: Provider entries to read.
        raw_models: Model entries to map onto the surviving providers.
        descriptor: The model type's registry descriptor; it supplies the
            category, the config keys, the default format and the default
            and per-format paths.
        format_key: Config key naming the endpoint format, when it is not
            the descriptor's (the legacy sections spell it ``format``).
        require_declared_format: Skip an entry that declares neither a
            format nor a shim — the unified pool's participation rule.
        map_models: Return the model→provider mapping; ``False`` when the
            caller distributes typed models separately.
        global_proxy: Server-level proxy URL.
        credential_owner: The LLM pool the entries draw their credential
            from, keyed by config name.  An entry that mints its credential
            (a ``token_command`` or the shim's ``auth.py``) shares its owner's
            key ring, so one mint serves both lanes.  ``None`` for a caller
            parsing a document without a live pool — the migration's
            "which providers serve this type" pass.
        refuse_unowned_minted: Disable an entry whose credential would be
            minted but has no owner to share it from.  The legacy
            ``<type>_providers`` section has no ``providers:`` entry, so its
            endpoints keep today's refusal with a reason that names the
            section.  A caller that only needs membership (the migration)
            passes ``False``.

    Returns:
        ``(providers, provider_infos, models)``.
    """
    keys = _TypedKeys.for_descriptor(descriptor, format_key)
    providers: dict[str, dict[str, Any]] = {}
    provider_infos: dict[str, ProviderInfo] = {}
    for name, cfg in raw_providers.items():
        try:
            parsed = parse_typed_provider(
                name,
                cfg,
                keys,
                descriptor=descriptor,
                require_declared_format=require_declared_format,
                global_proxy=global_proxy,
                credential_owner=(
                    credential_owner.get(name) if credential_owner is not None else None
                ),
                refuse_unowned_minted=refuse_unowned_minted,
            )
        except (ValueError, KeyError, TypeError) as exc:
            # One broken endpoint must not take the provider's other
            # endpoints — or every other provider — down with it.  The
            # provider stays in the pools it does parse; the failure is
            # recorded on its config entry for the admin payload.
            reason = str(exc)
            _record_endpoint_failure(cfg, keys.type_name, reason)
            logger.warning(
                "Provider %r: %s endpoint disabled: %s",
                name,
                keys.type_name,
                reason,
            )
            continue
        if parsed is not None:
            providers[name] = parsed[0]
            provider_infos[name] = parsed[1]

    models: dict[str, str] = {}
    if map_models:
        for model_name, value in raw_models.items():
            provider_name = first_pool_provider(value, providers)
            if provider_name:
                models[model_name] = provider_name
    return providers, provider_infos, models


_ENDPOINT_CATEGORIES = (
    ("names unknown shim", "unknown_shim"),
    ("serves ", "wrong_model_type"),
    ("no base_url", "missing_base_url"),
)


def _endpoint_failure_category(reason: str) -> str:
    """Classify an endpoint failure for the admin payload."""
    for needle, category in _ENDPOINT_CATEGORIES:
        if needle in reason:
            return category
    return "invalid_endpoint_config"


def _record_endpoint_failure(cfg: Any, type_name: str, reason: str) -> None:
    """Attach a per-endpoint diagnostic to the provider's config entry."""
    if not isinstance(cfg, dict):
        return
    endpoints = cfg.setdefault("_auto_disabled_endpoints", {})
    endpoints[type_name] = {
        "reason": reason,
        "category": _endpoint_failure_category(reason),
    }


def parse_typed_provider(
    name: str,
    cfg: Any,
    keys: _TypedKeys,
    *,
    descriptor: ModelTypeDescriptor,
    require_declared_format: bool,
    global_proxy: str | None,
    credential_owner: ProviderInfo | None = None,
    refuse_unowned_minted: bool = True,
) -> tuple[dict[str, Any], ProviderInfo] | None:
    """Parse one provider entry, or ``None`` when it does not participate.

    Precedence:

    1. shim: *shim_key*, else the provider-wide ``shim`` / ``type`` when
       its category matches;
    2. format: an explicit *format_key* > the shim's format > the
       provider-wide ``type`` > *default_format*, canonicalised;
    3. path: an explicit *path_key* > the shim's ``default_path`` when the
       effective format is the shim's > the descriptor's per-format path >
       *default_path*;
    4. connection: a non-empty config value > the shim's default, and a
       per-provider ``proxy`` > the server proxy;
    5. auth: Google's header for a ``google_generate`` upstream, OpenAI
       Bearer otherwise.

    An entry declaring ``<type>_enabled: false`` does not participate, even
    when it declares a format or shim: the operator keeps the endpoint's
    configuration and stops routing to it, and the provider's other endpoints
    are untouched.
    """
    if not isinstance(cfg, dict) or cfg.get("enabled") is False:
        return None
    endpoint_flag = face_value(cfg, keys.enabled_key, keys.type_name)
    if endpoint_flag is False:
        return None
    if endpoint_flag not in (None, "") and not isinstance(endpoint_flag, bool):
        # A quoted "false" is a string, not a switch: the endpoint stays on and
        # the operator is told, rather than silently serving what they believe
        # is off.
        logger.warning(
            "config: %r for the %s endpoint must be true or false, got %r; "
            "the endpoint stays enabled",
            keys.enabled_key,
            keys.type_name,
            endpoint_flag,
        )
    shim = resolve_endpoint_shim(cfg, type_name=keys.type_name, shim_key=keys.shim_key)
    explicit_format = face_value(cfg, keys.format_key, keys.type_name) or ""
    if require_declared_format and not explicit_format and shim is None:
        return None

    wire_format = typed_wire_format(cfg, keys, explicit_format, shim)
    path = typed_path(cfg, keys, descriptor, shim, wire_format)
    info = typed_provider_info(
        name,
        cfg,
        keys,
        wire_format,
        path,
        shim,
        global_proxy,
        credential_owner=credential_owner,
        refuse_unowned_minted=refuse_unowned_minted,
    )
    # The shim's defaults describe its own dialect; an explicit format that
    # overrides the shim's makes them a guess, so they apply only when the
    # effective format is the shim's (the same rule ``typed_path`` follows).
    parsed: dict[str, Any] = {"format": wire_format, keys.path_key: path}
    if shim is not None and wire_format == shim.format and shim.params:
        parsed["params"] = dict(shim.params)
    return parsed, info


def typed_wire_format(
    cfg: dict[str, Any], keys: _TypedKeys, explicit_format: str, shim: Any
) -> str:
    """The wire format an entry resolves to."""
    if explicit_format:
        return explicit_format
    if shim is not None:
        return shim.format
    return cfg.get("type") or keys.default_format


def typed_path(
    cfg: dict[str, Any],
    keys: _TypedKeys,
    descriptor: ModelTypeDescriptor,
    shim: Any,
    wire_format: str,
) -> str:
    """The upstream endpoint path an entry resolves to."""
    path = face_value(cfg, keys.path_key, keys.type_name) or ""
    if not path and shim is not None and wire_format == shim.format:
        path = shim.default_path or ""
    if not path:
        path = descriptor.path_for(wire_format) or ""
    return path or keys.default_path


def _refuse_unowned_minted_credential(
    name: str,
    effective: dict[str, Any],
    shim: Any,
    type_name: str,
) -> None:
    """Refuse a minted credential this parse has no lane to share it from.

    A shim can mint its credential from a command (``connection.token_command``)
    or from an exchange (``auth.py``), and a provider entry can name its own
    command.  A ``providers:`` entry shares the LLM lane's key ring, so its
    typed endpoints forward the minted credential
    (:func:`~llm_rosetta_gateway.providers.share_credential`).

    The legacy ``<type>_providers`` section has no counterpart in
    ``providers:`` — nothing there mints, seeds or refreshes — so an entry
    there that mints would forward an empty ``Authorization`` header and fail
    as a 401 far from the cause.  ``parse_typed_config`` catches the
    ``ValueError`` and auto-disables the endpoint with this reason, which names
    the section and the fix.

    Args:
        name: The configured provider, named in the message.
        effective: The entry after shim defaults are applied.
        shim: The resolved endpoint shim, or ``None``.
        type_name: The model type whose section is being parsed.

    Raises:
        ValueError: The effective credential is produced by a command or an
            exchange, and this parse has no owner to share it from.
    """
    exchange = getattr(shim, "auth", None) if shim is not None else None
    command = effective.get("token_command")
    if exchange is None and command is None:
        return
    kind = getattr(exchange, "kind", None) or "auth.py"
    source = "a token_command" if command is not None else f"an {kind} exchange"
    raise ValueError(
        f"provider {name!r} mints its credential with {source}, but the legacy "
        f"{type_name}_providers section has no provider entry in `providers:` "
        f"to share the minted credential from; move this provider to "
        f"`providers:` so its {type_name} endpoint can use it"
    )


def typed_provider_info(
    name: str,
    cfg: dict[str, Any],
    keys: _TypedKeys,
    wire_format: str,
    path: str,
    shim: Any,
    global_proxy: str | None,
    *,
    credential_owner: ProviderInfo | None = None,
    refuse_unowned_minted: bool = True,
) -> ProviderInfo:
    """The connection a parsed entry forwards through.

    An endpoint of a ``providers:`` entry forwards with that entry's
    credential: *credential_owner* is the entry in the LLM pool, and this
    object reads its key ring, minting declaration and credential status.  One
    credential source therefore mints once, seeds once and refreshes once, and
    every endpoint of it sees the same value.

    Args:
        name: The provider's config key.
        cfg: The raw provider entry.
        keys: The config keys this type is parsed with.
        wire_format: The effective wire format.
        path: The upstream endpoint path.
        shim: The resolved endpoint shim, or ``None``.
        global_proxy: Server-level proxy URL.
        credential_owner: The provider entry that mints this endpoint's
            credential, or ``None`` when the parse has no such lane.
        refuse_unowned_minted: Disable the endpoint when it mints but has no
            owner — see :func:`_refuse_unowned_minted_credential`.

    Raises:
        ValueError: The credential would be minted and there is no owner to
            share it from.
    """
    from ..providers import _apply_shim_defaults, share_credential
    from ..transport.provider_info import google_auth, openai_auth

    effective = _apply_shim_defaults(cfg, shim) if shim is not None else cfg
    if credential_owner is None and refuse_unowned_minted:
        _refuse_unowned_minted_credential(name, effective, shim, keys.type_name)
    base_url = _face_base_url(name, cfg, keys.type_name, effective, shim, wire_format)
    info = ProviderInfo(
        name=f"{keys.type_name}:{name}",
        api_key=effective.get("api_key", ""),
        base_url=base_url,
        auth_header_fn=(
            google_auth if wire_format == "google_generate" else openai_auth
        ),
        url_template="{base_url}" + path,
        proxy_url=cfg.get("proxy") or global_proxy,
    )
    if credential_owner is not None:
        share_credential(info, credential_owner)
    return info


def _face_base_url(
    name: str,
    cfg: dict[str, Any],
    type_name: str,
    effective: dict[str, Any],
    shim: Any,
    wire_format: str,
) -> str:
    """The host one typed endpoint forwards to.

    The face's own ``base_url`` wins; without one the endpoint inherits the
    entry's, which is what it always did — a shim's default host never
    substitutes for the operator's URL.  Inheritance is checked rather than
    assumed: an endpoint whose own declaration belongs to another place is
    almost always an entry written before the face existed, and sending it the
    chat lane's host is a silent misroute rather than a configuration error —
    so the mismatch is logged here and shown on the endpoint's row in the panel.

    Args:
        name: The provider's config key, for the warning.
        cfg: The raw provider entry.
        type_name: The model type whose face this is.
        effective: The entry after shim defaults, i.e. what the connection is
            built from.
        shim: The endpoint's shim, or ``None``.
        wire_format: The format the endpoint resolves to; used to find the
            vendor shim when the face names a format and no shim.

    Returns:
        The base URL with any trailing slash removed.
    """
    declared = face_section(cfg, type_name).get("base_url") or ""
    base_url = str(declared or effective.get("base_url", ""))
    if declared:
        return base_url.rstrip("/")
    candidates = _endpoint_hosts(shim, type_name, wire_format)
    if candidates and _origin(base_url) not in {_origin(host) for host in candidates}:
        logger.warning(
            "Provider %r: the %s endpoint inherits base_url %r, but %s; "
            "give the endpoint its own `%s: {base_url: ...}` if the two are "
            "different vendors",
            name,
            type_name,
            base_url,
            _endpoint_subject(shim, wire_format, candidates),
            type_name,
        )
    return base_url.rstrip("/")


def _endpoint_subject(shim: Any, wire_format: str, candidates: list[str]) -> str:
    """How the warning names what the endpoint forwards with."""
    if shim is not None:
        return f"its shim {getattr(shim, 'name', '?')!r} forwards to {candidates[0]!r}"
    listed = ", ".join(f"{host!r}" for host in sorted(candidates))
    return (
        f"no shim serving format {wire_format!r} forwards there (one of {listed} would)"
    )


def _endpoint_hosts(shim: Any, type_name: str, wire_format: str) -> list[str]:
    """The hosts this endpoint's declaration would forward to.

    A face that names a shim takes that shim's host.  A face that names only a
    *format* has no shim of its own to ask, and several vendors ship a shim in
    their dialect — ``embeddings: {format: "openai"}`` is served by
    ``openai--embedding``, by ``siliconflow--embedding`` and by three more — so
    *every* registered shim of the ``(model_type, format)`` pair answers.  The
    inherited host is accepted when it is any of them and reported only when it
    is none: answering with the first registered shim made SiliconFlow, Zhipu,
    Z.ai and Volcengine warn while routing correctly (#1595 review).
    """
    connection = getattr(shim, "connection", None) if shim is not None else None
    host = getattr(connection, "base_url", None) or ""
    if host:
        return [str(host)]
    from llm_rosetta.api import list_shims

    hosts: list[str] = []
    for candidate in list_shims():
        if (
            getattr(candidate, "model_type", None) == type_name
            and getattr(candidate, "format", None) == wire_format
        ):
            candidate_host = getattr(candidate.connection, "base_url", None) or ""
            if candidate_host and str(candidate_host) not in hosts:
                hosts.append(str(candidate_host))
    return hosts


def _origin(url: str) -> str:
    """The origin of *url*: scheme, host and port.

    Hostname alone called ``http://api.cohere.com`` and
    ``https://api.cohere.com`` the same place, and two local services on
    different ports the same place too — each a silent misroute, which is the
    failure this check exists to name (#1595 review).
    """
    from urllib.parse import urlsplit

    try:
        parts = urlsplit(str(url))
    except ValueError:
        return str(url)
    if not parts.scheme or not parts.hostname:
        return str(url)
    port = parts.port or {"http": 80, "https": 443}.get(parts.scheme)
    return f"{parts.scheme}://{parts.hostname.lower()}:{port}"


def resolve_endpoint_shim(cfg: dict[str, Any], *, type_name: str, shim_key: str) -> Any:
    """Return the shim a non-LLM endpoint of *type_name* is configured through.

    An explicit ``cfg[shim_key]`` must name a registered shim serving
    *type_name*; a mismatch is a configuration error.  Otherwise the
    provider-wide ``shim`` / ``type`` is used only when its category
    matches, so a chat provider named ``openai`` keeps its chat identity
    while still serving embeddings through ``embeddings_format``.

    Args:
        cfg: Raw provider config entry.
        type_name: Registry category the endpoint belongs to.
        shim_key: Provider config key naming the endpoint shim
            (e.g. ``"embeddings_shim"``).

    Returns:
        The effective :class:`~llm_rosetta.shims.provider_shim.ProviderShim`,
        or ``None``.

    Raises:
        ValueError: If ``cfg[shim_key]`` is unknown or serves another type.
    """
    from llm_rosetta.api import get_shim

    declared = face_value(cfg, shim_key, type_name)
    if declared:
        shim = get_shim(declared)
        if shim is None:
            raise ValueError(f"config: {shim_key} names unknown shim '{declared}'")
        if shim.model_type != type_name:
            raise ValueError(
                f"config: {shim_key} shim '{declared}' serves "
                f"{shim.model_type}, not {type_name}"
            )
        return shim
    for key in ("shim", "type"):
        value = cfg.get(key)
        if not value:
            continue
        shim = get_shim(value)
        if shim is not None and shim.model_type == type_name:
            return shim
    return None
