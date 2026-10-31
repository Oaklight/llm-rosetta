"""Runtime resolution: selection context, provider choice, loop detection.

What turns a request's model and source format into the provider that serves
it, plus the identity map the admin panel and the ops layer read.
"""

from __future__ import annotations

from datetime import datetime

import logging
from typing import TYPE_CHECKING, Any

from ..routing import (
    ModelRoute,
    SelectionContext,
)
from llm_rosetta.auto_detect import ProviderType
from llm_rosetta.routing import ResolvedRoute

from ..providers import BASE_FORMATS
from ..transport import ProviderInfo
from .schema import ProviderIdentity
from .typed import resolve_endpoint_shim

if TYPE_CHECKING:
    from .core import GatewayConfig


logger = logging.getLogger("llm-rosetta-gateway")


def identity_category(wire_format: str, cfg: dict[str, Any]) -> str | None:
    """Return the category a bare *wire_format* names, or ``None`` if unknown.

    Chat formats resolve to ``llm``.  A non-chat format name is known but
    may span categories (``cohere``, ``jina`` and ``voyage`` are both
    embedding and rerank formats), so prefer a category the provider
    declares an endpoint for and otherwise the registry order.  A name that
    appears in no category stays unknown, which lets the caller keep the
    legacy ``llm`` default for a provider with no declared type.
    """
    from llm_rosetta.model_vocabulary import FORMATS_BY_TYPE

    from ..model_types import all_model_types
    from .typed import _declared

    if wire_format in FORMATS_BY_TYPE["llm"]:
        return "llm"
    candidates = [
        model_type
        for model_type, formats in FORMATS_BY_TYPE.items()
        if wire_format in formats
    ]
    if not candidates:
        return None
    declared = {
        desc.name
        for desc in all_model_types()
        if not desc.is_llm
        and (
            (
                desc.config_format_key
                and _declared(cfg, desc.config_format_key, desc.name)
            )
            or (
                desc.config_shim_key and _declared(cfg, desc.config_shim_key, desc.name)
            )
        )
    }
    for candidate in candidates:
        if candidate in declared:
            return candidate
    return candidates[0]


def endpoint_identity(
    cfg: dict[str, Any],
    endpoint_types: list[tuple[str, str, str | None]],
    provider_shim_names: dict[str, str | None],
    name: str,
) -> tuple[str, str] | None:
    """Return the ``(model_type, format)`` a non-LLM endpoint declares.

    A populated ``<type>_format`` key declares the endpoint just as a
    ``<type>_shim`` does.  Reading only the shim keys left such a provider
    with the provider *name* as its format and an ``llm`` identity, which
    hid it from the admin picker and from the LLM-model validator.
    """
    from llm_rosetta.provider_names import canonical_provider_name

    from .typed import _declared

    for type_name, shim_key, format_key in endpoint_types:
        declared_format = (
            _declared(cfg, format_key, type_name) if format_key else ""
        ) or ""
        if not declared_format and not _declared(cfg, shim_key, type_name):
            continue
        shim = resolve_endpoint_shim(cfg, type_name=type_name, shim_key=shim_key)
        if shim is None and not declared_format:
            continue
        if declared_format:
            wire_format = canonical_provider_name(declared_format)
        else:
            assert shim is not None
            wire_format = shim.format
        if shim is not None:
            # The LLM-side ProviderInfo still needs the shim's connection
            # defaults (base_url, key env) so an endpoint-only provider
            # loads without duplicating them in the config.
            provider_shim_names[name] = shim.name
        return type_name, wire_format
    return None


def resolve_provider_identities(
    raw_providers: dict[str, dict[str, str]],
) -> tuple[dict[str, ProviderIdentity], dict[str, str | None]]:
    """Resolve each provider's ``(model_type, format)`` identity.

    ``GatewayConfig.provider_types`` is derived from these values as the
    format, so the historical field keeps its meaning for existing
    consumers; read this mapping when the capability category matters.
    For the decision shims the format is the dialect (``systemone`` /
    ``openai_decisions``) rather than the category the old ``base`` field
    happened to carry.

    Resolution order per provider:
      1. ``shim`` field → the shim's pair
      2. ``type`` field → the shim's pair, or the format's category
      3. provider name itself (backward-compatible fallback)
      4. a non-LLM ``*_shim`` when the provider has no chat identity

    Step 4 matters for a provider configured only as an endpoint, e.g.
    ``{"embeddings_shim": "jina--embedding"}``.  It never overrides a chat
    identity: a provider named ``openai`` that also declares an embedding
    shim stays an LLM provider and serves embeddings through the endpoint
    config.

    Legacy provider/format names are resolved to canonical spellings
    (see :func:`llm_rosetta.provider_names.canonical_provider_name`), and
    legacy shim names to their canonical ``<provider>--<base>`` spelling.
    A config declaring e.g. ``type: google`` is migrated to
    ``google_generate``; a ``shim:`` / ``type:`` naming a legacy shim (e.g.
    ``volcengine--openai_chat``) is migrated to ``volcengine--chat_completions``.
    Each emits one ``DeprecationWarning`` per distinct legacy name so
    operators migrate before 0.16.0 (#919).

    Returns:
        Tuple of (provider_identities, provider_shim_names).
    """
    from llm_rosetta.provider_names import (
        canonical_provider_name,
        warn_legacy_provider_names,
        warn_legacy_shim_names,
    )
    from llm_rosetta.shims import resolve_format, resolve_model_type

    from ..model_types import all_model_types
    from ..maintenance.names import correct_legacy_names, legacy_name_error

    # Migrate legacy names in place.  File-backed configs are also
    # corrected and written back by ``JsoncConfigIO``; this covers
    # programmatic configs.  A name that survives was one we could not map,
    # so reject it rather than resolving a legacy alias at runtime.
    correct_legacy_names({"providers": raw_providers})
    error = legacy_name_error(raw_providers)
    if error is not None:
        raise ValueError(error)

    endpoint_types = [
        (desc.name, desc.config_shim_key, desc.config_format_key)
        for desc in all_model_types()
        if not desc.is_llm and desc.config_shim_key
    ]

    identities: dict[str, ProviderIdentity] = {}
    provider_shim_names: dict[str, str | None] = {}
    for name, cfg in raw_providers.items():
        # ``shim:`` / ``type:`` values were already canonicalised above, so
        # no deprecation warning fires for them — the gateway *corrects*
        # the config (and a file-backed one is rewritten on disk), logging
        # each rewrite at INFO.  Only a legacy provider *key* still warns:
        # keys are user identifiers and are not rewritten.
        if "shim" in cfg:
            source = cfg["shim"]
            provider_shim_names[name] = source
        elif "type" in cfg:
            source = cfg["type"]
            provider_shim_names[name] = source
        else:
            warn_legacy_provider_names(name)
            if name not in BASE_FORMATS:
                warn_legacy_shim_names(name)
            # Resolve through the shim registry too: a provider named after
            # a shim (e.g. ``openai--open_responses``, or the legacy
            # ``openai_responses`` alias of it) uses that shim's base and
            # keeps its identity, rather than being read as a bare format
            # name.
            source = name
            provider_shim_names[name] = name

        wire_format = canonical_provider_name(resolve_format(source))
        model_type = resolve_model_type(source)

        # A provider with no chat identity may be defined by a non-LLM
        # endpoint alone.  The provider-wide shim/type wins when it names
        # a category, so this only fills in the unknown case.
        if model_type is None:
            endpoint = endpoint_identity(cfg, endpoint_types, provider_shim_names, name)
            if endpoint is not None:
                model_type, wire_format = endpoint

        if model_type is None:
            # A known non-chat format still names a category even when the
            # bare name is ambiguous.  Never turn that "I do not know the
            # chat format" into a confident ``llm`` identity.
            model_type = identity_category(wire_format, cfg)

        identities[name] = ProviderIdentity(
            model_type=model_type or "llm", format=wire_format
        )
    return identities, provider_shim_names


def selection_context(
    config: GatewayConfig,
    model: str,
    source_format: str,
    pinned_provider: str | None = None,
) -> SelectionContext:
    """Build one request's selection state, carrying this config's tables.

    This is the one place the two halves of selection are assembled: the
    context is created here, used to choose a provider before any pipeline
    exists, and carried on to the proxy, which fills in what it learns from
    the converted request.

    Args:
        model: Model name as specified by the client.
        source_format: API standard the request arrived in.  Legacy
            spellings normalise here.
        pinned_provider: Serve the request from this provider only.  The
            admin test path sets it to reach one provider of a
            multi-provider model; ordinary requests leave it unset.

    Returns:
        A context holding the model's route and this config's provider
        tables.

    Raises:
        KeyError: If the model is not in the routing table.
    """
    ctx = SelectionContext(model, source_format)
    ctx.install_route(
        model_route=config.models[model],
        providers=config.providers,
        provider_types=config.provider_types,
        provider_shim_names=config.provider_shim_names,
        provider_account_tokens=config.provider_account_tokens,
    )
    ctx.with_pinned_provider(pinned_provider)
    return ctx


def _quota_blocked_until(unready: list[Any]) -> datetime | None:
    """The end of a quota block covering every unready provider, if any.

    Returns the **latest** end, because the route is unavailable until the last
    of its providers is back.
    """
    from ..quota.state import QUOTA_STATE

    ends: list[datetime] = []
    for pinfo in unready:
        blocked = getattr(pinfo, "quota_provider_blocked", None)
        if blocked is None or not blocked():
            return None
        identity = getattr(pinfo, "quota_identity", None) or f"@{pinfo.name}"
        block = QUOTA_STATE.provider_block(identity)
        if block is None:
            return None
        ends.append(block.until)
    return max(ends) if ends else None


def _quota_exhausted_until(
    config: GatewayConfig, ctx: SelectionContext
) -> datetime | None:
    """The end of a spent allowance covering every provider, if one is known.

    The route is unavailable until the **last** provider comes back, so this is
    the latest of their earliest serve times; any provider whose vendor named
    no reset makes the end unknown, and the gateway will not invent one.
    """
    from ..quota.service import provider_exhausted_until

    ends: list[datetime] = []
    for entry in ctx.candidates:
        pinfo = config.providers.get(entry.name)
        if pinfo is None:
            return None
        moment = provider_exhausted_until(pinfo)
        if moment is None:
            return None
        ends.append(moment)
    return max(ends) if ends else None


def resolve(
    config: GatewayConfig, ctx: SelectionContext
) -> tuple[ResolvedRoute, ProviderInfo]:
    """Resolve *ctx* to a :class:`ResolvedRoute` and :class:`ProviderInfo`.

    Consolidates model lookup, provider type resolution, shim binding,
    capability detection, and reasoning overrides into a single typed
    result.

    Args:
        ctx: The request's selection state, as built by
            :meth:`selection_context`.  Its ``chosen`` provider is set —
            or left unset, which is reported as not-ready.

    Returns:
        ``(route, provider_info)`` — the route contains all
        pipeline-relevant fields; ``provider_info`` is the
        transport-level connection config.

    Raises:
        ProviderNotReady: If no provider for the model is ready.
    """
    from typing import cast

    from ..maintenance.startup import ProviderInitState, ProviderNotReady

    model = ctx.model
    source_provider = cast(ProviderType, ctx.source_format)

    # The anchor is what routing would pick with the preference off, and the
    # full route's counter advances exactly as it would there.  Tier
    # selection can still move the served request onto another account
    # (tier 3), so what is preserved is the anchor selection, not the
    # per-account served share.
    config._selection.run(ctx)
    model_route = ctx.model_route
    assert model_route is not None  # install_route() put it there
    if ctx.chosen is None:
        unready = [
            config.providers[p.name]
            for p in model_route.providers
            if not config.providers[p.name].ready
        ]
        # Every route is out because of a quota block: that has a known end,
        # and telling the client "initializing" would be a different story.
        blocked_until = _quota_blocked_until(unready)
        if blocked_until is not None and len(unready) == len(model_route.providers):
            raise ProviderNotReady(model, blocked_until=blocked_until)
        # Every candidate removed by a reading rather than by readiness.  A
        # reading knows when its window resets, which is exactly what
        # ``Retry-After`` is for; "initializing" would send the client looking
        # for a startup problem that does not exist.
        route_names = {p.name for p in model_route.providers}
        if route_names and route_names <= set(ctx.quota_excluded):
            until = _quota_exhausted_until(config, ctx)
            raise ProviderNotReady(
                model, exhausted_until=until, exhausted=until is None
            )
        deferred = getattr(config, "_deferred_startup", None)
        all_failed = deferred is not None and all(
            deferred.provider_state(p.name) == ProviderInitState.FAILED for p in unready
        )
        raise ProviderNotReady(model, failed=all_failed)

    entry = ctx.chosen
    provider_name = entry.name
    provider_type = config.provider_types[provider_name]
    shim_name = config.provider_shim_names.get(provider_name)
    upstream_model = entry.upstream_model or config.model_upstream_names.get(model)
    caps = config.model_capabilities.get(model, list(config.DEFAULT_CAPABILITIES))
    reasoning = config.model_reasoning_overrides.get(model)
    flatten_system = config.model_flatten_system.get(model, False)
    from llm_rosetta.shims import resolve_shim

    _shim = resolve_shim(shim_name) if shim_name else None
    custom_tools = config.provider_supports_custom_tools.get(
        provider_name, _shim.tools.custom_tools if _shim else False
    )
    hoist_system = config.provider_hoist_system_messages.get(
        provider_name, _shim.hoist_system_messages if _shim else True
    )
    preflight = config.provider_preflight_token_count.get(provider_name, False)

    _mtdl_model = config.model_max_tool_description_length.get(model)
    _mtdl_prov = config.provider_max_tool_description_length.get(provider_name)
    _mtdl_shim = _shim.tools.max_description_length if _shim else None
    max_tool_desc = (
        _mtdl_model
        if _mtdl_model is not None
        else (_mtdl_prov if _mtdl_prov is not None else _mtdl_shim)
    )

    route = ResolvedRoute(
        source_provider=source_provider,
        target_provider=cast(ProviderType, provider_type),
        provider_name=provider_name,
        shim_name=shim_name,
        upstream_model=upstream_model,
        model_capabilities=caps,
        reasoning_override=reasoning,
        flatten_system=flatten_system,
        supports_custom_tools=custom_tools,
        hoist_system_messages=hoist_system,
        preflight_token_count=preflight,
        max_tool_description_length=max_tool_desc,
        soft_error_patterns=_shim.soft_error_patterns if _shim else (),
    )

    pinfo = config.providers[provider_name]

    # Per-model URL template override: create a shallow copy of the
    # provider info with modified template(s) so the transport layer
    # hits the right endpoint.
    model_url_tpl = config.model_url_templates.get(model)
    model_stream_tpl = config.model_stream_url_templates.get(model)
    if model_url_tpl or model_stream_tpl:
        pinfo = pinfo.with_url_templates(model_url_tpl, model_stream_tpl)

    model_timeout = config.model_timeouts.get(model)
    if model_timeout is not None:
        pinfo = pinfo.with_timeout(model_timeout)

    return route, pinfo


def detect_routing_loops(
    models: dict[str, ModelRoute],
    upstream_names: dict[str, str],
) -> list[str]:
    """Detect potential routing loops in the model configuration.

    A loop risk exists when model A's upstream_model is also a registered
    model name — if the upstream provider's base_url points back at this
    gateway, the request would be re-routed internally.  Also detects
    transitive chains (A → B → C → A).

    Returns a list of human-readable warning strings (empty if clean).
    """
    warnings: list[str] = []

    # Collect all edges: model_name → upstream_model (where upstream is
    # also a model key).  Include both model-level and per-entry overrides.
    edges: dict[str, str] = {}
    for model_name, route in models.items():
        up = upstream_names.get(model_name)
        if up and up in models:
            edges[model_name] = up
            continue
        for entry in route.providers:
            if entry.upstream_model and entry.upstream_model in models:
                edges[model_name] = entry.upstream_model
                break

    # Report every edge as a potential loop — even non-cyclic chains like
    # A→B are dangerous when the upstream provider points back at this
    # gateway.  True cycles (A→B→A) get a stronger message.
    reported: set[str] = set()
    for source, target in edges.items():
        if source in reported:
            continue
        # Walk forward to detect cycles.
        chain = [source]
        node: str | None = target
        chain_set = {source}
        while node and node not in chain_set:
            chain.append(node)
            chain_set.add(node)
            node = edges.get(node)
        if node and node in chain_set:
            cycle_start = chain.index(node)
            cycle = chain[cycle_start:] + [node]
            warnings.append(f"routing loop: {' → '.join(cycle)}")
            reported.update(cycle)
        else:
            warnings.append(
                f"upstream_model collision: '{source}' maps to"
                f" '{target}' which is also a routable model"
            )
            reported.add(source)

    return warnings
