"""The typed (non-LLM) endpoint pools: parsing, and the shim resolver.

Driven by the model-type descriptor, so embedding, rerank and decision share
one implementation; :func:`parse_typed_config` documents the precedence.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..transport import ProviderInfo
from .parse import first_pool_provider

if TYPE_CHECKING:
    from ..model_types import ModelTypeDescriptor

logger = logging.getLogger("llm-rosetta-gateway")


def legacy_config_key(key: str, type_name: str) -> str | None:
    """The pre-rename spelling of *key*, when *type_name* was renamed.

    ``embeddings_format`` → ``embedding_format``.  A config written before
    the model type was renamed is rewritten by the load-time migration; this
    is the read-side fallback for a programmatic config, or a file that could
    not be written back (a read-only mount).

    Args:
        key: A config key derived from *type_name* (``embeddings_path``).
        type_name: The canonical model type.

    Returns:
        The legacy key, or ``None`` when the type was never renamed or the
        key does not carry the type's prefix.
    """
    from llm_rosetta.formats import legacy_model_type

    old = legacy_model_type(type_name)
    prefix = f"{type_name}_"
    if old is None or not key.startswith(prefix):
        return None
    return f"{old}_" + key[len(prefix) :]


def _declared(cfg: dict[str, Any], key: str, type_name: str) -> Any:
    """Read *key*, falling back to its pre-rename spelling.

    The fallback logs once per read so an operator sees which key to change;
    the migration normally rewrites the file before this runs.
    """
    value = cfg.get(key)
    if value:
        return value
    legacy = legacy_config_key(key, type_name)
    if legacy is None:
        return value
    value = cfg.get(legacy)
    if value:
        logger.warning(
            "config: %r is deprecated for the %s endpoint; use %r",
            legacy,
            type_name,
            key,
        )
    return value


@dataclass(frozen=True)
class _TypedKeys:
    """The config keys and defaults one typed endpoint is parsed with."""

    type_name: str
    format_key: str
    path_key: str
    shim_key: str
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
    """
    if not isinstance(cfg, dict) or cfg.get("enabled") is False:
        return None
    shim = resolve_endpoint_shim(cfg, type_name=keys.type_name, shim_key=keys.shim_key)
    explicit_format = _declared(cfg, keys.format_key, keys.type_name) or ""
    if require_declared_format and not explicit_format and shim is None:
        return None

    wire_format = typed_wire_format(cfg, keys, explicit_format, shim)
    path = typed_path(cfg, keys, descriptor, shim, wire_format)
    info = typed_provider_info(name, cfg, keys, wire_format, path, shim, global_proxy)
    return {"format": wire_format, keys.path_key: path}, info


def typed_wire_format(
    cfg: dict[str, Any], keys: _TypedKeys, explicit_format: str, shim: Any
) -> str:
    """The wire format an entry resolves to, canonicalised."""
    from llm_rosetta.provider_names import canonical_provider_name

    if explicit_format:
        return canonical_provider_name(explicit_format)
    if shim is not None:
        return shim.format
    return canonical_provider_name(cfg.get("type") or keys.default_format)


def typed_path(
    cfg: dict[str, Any],
    keys: _TypedKeys,
    descriptor: ModelTypeDescriptor,
    shim: Any,
    wire_format: str,
) -> str:
    """The upstream endpoint path an entry resolves to."""
    path = _declared(cfg, keys.path_key, keys.type_name) or ""
    if not path and shim is not None and wire_format == shim.format:
        path = shim.default_path or ""
    if not path:
        path = descriptor.path_for(wire_format) or ""
    return path or keys.default_path


def typed_provider_info(
    name: str,
    cfg: dict[str, Any],
    keys: _TypedKeys,
    wire_format: str,
    path: str,
    shim: Any,
    global_proxy: str | None,
) -> ProviderInfo:
    """The connection a parsed entry forwards through."""
    from ..providers import _apply_shim_defaults
    from ..transport.provider_info import google_auth, openai_auth

    effective = _apply_shim_defaults(cfg, shim) if shim is not None else cfg
    return ProviderInfo(
        name=f"{keys.type_name}:{name}",
        api_key=effective.get("api_key", ""),
        base_url=effective.get("base_url", "").rstrip("/"),
        auth_header_fn=(
            google_auth if wire_format == "google_generate" else openai_auth
        ),
        url_template="{base_url}" + path,
        proxy_url=cfg.get("proxy") or global_proxy,
    )


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
    from llm_rosetta.shims import get_shim

    declared = _declared(cfg, shim_key, type_name)
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
