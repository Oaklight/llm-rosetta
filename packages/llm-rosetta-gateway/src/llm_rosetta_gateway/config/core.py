"""Gateway configuration: the parsed and validated runtime config.

:class:`GatewayConfig` itself.  Loading lives in :mod:`config.loader`, the
value types in :mod:`config.schema`, and the parsing / resolution it performs
in :mod:`config.parse`, :mod:`config.typed`, :mod:`config.settings` and
:mod:`config.resolve`.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from llm_rosetta_gateway.observability.maintenance import Registry
from llm_rosetta.routing import ResolvedRoute

from ..providers import build_provider_info
from ..routing import (
    AnchorStage,
    EligibilityStage,
    RoutingChain,
    RoutingStage,
    SameFormatStage,
    SelectionContext,
    RulesStage,
)
from ..transport import ProviderInfo
from .parse import (
    DEFAULT_CAPABILITIES as _DEFAULT_CAPABILITIES,
)
from .parse import (
    declared_model_lanes,
    distribute_typed_models,
    parse_model_failover,
    parse_model_overrides,
    parse_models,
)
from .resolve import (
    resolve,
    resolve_provider_identities,
    selection_context,
)
from .schema import TypedRoute
from .settings import (
    apply_auth_settings,
    apply_circuit_breaker_settings,
    apply_debug_settings,
    apply_rate_limit_settings,
    apply_routing_settings,
    apply_server_settings,
    parse_custom_tools_overrides,
    parse_hoist_system_overrides,
    parse_key_affinity_overrides,
    parse_max_tool_description_length_overrides,
    parse_preflight_token_count_overrides,
    validate,
)
from .typed import _typed_descriptors, parse_typed_config

logger = logging.getLogger("llm-rosetta-gateway")

if TYPE_CHECKING:
    from ..quota.rate_limit import RateLimitPolicy
    from ..shim_sources import ShimSources


class GatewayConfig:
    """Parsed and validated gateway configuration."""

    # Default capabilities when not specified in config.
    DEFAULT_CAPABILITIES: list[str] = _DEFAULT_CAPABILITIES

    # Typed (non-LLM) endpoint pools.  The registry-driven loop in __init__
    # fills one set per registered type; the built-in three are declared here
    # so their attributes are visible to a type checker.  A further type reads
    # its pools through :meth:`typed_pool`.
    #: Per type, then per model: the providers that serve that type, with the
    #: weights the model declared.  The typed pools keep only the first one as
    #: ``<type>_models[model]``; this is the pool the selector scores.
    typed_model_candidates: dict[str, dict[str, list[Any]]]
    embeddings_providers: dict[str, dict[str, Any]]
    embeddings_provider_infos: dict[str, ProviderInfo]
    embeddings_models: dict[str, str]
    rerank_providers: dict[str, dict[str, Any]]
    rerank_provider_infos: dict[str, ProviderInfo]
    rerank_models: dict[str, str]
    decisions_providers: dict[str, dict[str, Any]]
    decisions_provider_infos: dict[str, ProviderInfo]
    decisions_models: dict[str, str]
    default_embeddings_format: str
    default_rerank_format: str
    default_decisions_format: str

    # Runtime state, filled by the parsing and resolution modules.
    _raw_providers: dict[str, dict[str, str]]
    # Every lane each model names, disabled providers included: the Routing
    # board reads this so a lane the gateway took out does not disappear.
    declared_model_lanes: dict[str, list[dict[str, Any]]]
    # What the load-time provider validation did.  Written by the CLI startup
    # path before the ops log exists; run_gateway records it (issue #1386).
    provider_validation_result: dict[str, Any] | None
    _same_format: SameFormatStage
    _selection: RoutingChain
    admin_cors_origins: list[str]
    admin_custom_css: str
    admin_password: str | None
    admin_session_idle_minutes: int
    admin_session_remember_days: int
    api_keys_db: str | None
    circuit_breaker_registry: Any
    credential_visible: bool
    data_dir: str | None
    error_dumps_enabled: bool
    host: str
    log_bodies: bool
    log_format: str
    maintenance: Registry | None
    model_capabilities: Any
    model_failover: Any
    model_flatten_system: Any
    model_max_tool_description_length: Any
    model_reasoning_overrides: Any
    model_stream_url_templates: Any
    model_timeouts: Any
    model_upstream_names: Any
    model_url_templates: Any
    open_on_no_keys: bool
    ops_log: dict[str, Any]
    port: int
    prefer_same_format: bool
    provider_account_tokens: dict[str, str]
    provider_hoist_system_messages: Any
    provider_identities: Any
    provider_key_affinity: Any
    provider_max_tool_description_length: Any
    provider_preflight_token_count: Any
    provider_shim_names: Any
    provider_supports_custom_tools: Any
    provider_types: dict[str, str]
    proxy: str | None
    rate_limit_algorithm: str
    rate_limit_enabled: bool
    rate_limit_exclude: list[str]
    rate_limit_global: str | list[str] | None
    rate_limit_per_ip: str | list[str] | None
    rate_limit_per_key: str | list[str] | None
    rate_limit_per_model: str | list[str] | None
    rate_limit_trust_proxy: bool
    raw_providers: dict[str, dict[str, str]]
    read_timeout: float
    request_log: dict[str, Any]
    root_redirect: str | None
    routing_rate_limit: RateLimitPolicy
    self_update: bool
    shims: ShimSources
    socket: str | None
    tls_cert: str | None
    tls_key: str | None
    update_check: bool
    upstream_timeout: float
    verbose: bool

    @staticmethod
    def _formats_for_type(type_name: str) -> list[str]:
        """Return supported formats for a model type from the registry.

        Falls back to an empty list if the type is not registered.
        """
        from ..model_types import get_model_type

        desc = get_model_type(type_name)
        return list(desc.formats) if desc else []

    @staticmethod
    def embeddings_formats() -> list[str]:
        """Return supported embeddings formats from the model type registry."""
        return GatewayConfig._formats_for_type("embeddings")

    @staticmethod
    def embedding_formats() -> list[str]:
        """Deprecated alias for :meth:`embeddings_formats`."""
        return GatewayConfig.embeddings_formats()

    @staticmethod
    def rerank_formats() -> list[str]:
        """Return supported rerank formats from the model type registry."""
        return GatewayConfig._formats_for_type("rerank")

    def selection_context(
        self,
        model: str,
        source_provider: str,
        pinned_provider: str | None = None,
    ) -> SelectionContext:
        """Build the selection context for one request."""
        return selection_context(self, model, source_provider, pinned_provider)

    def resolve(self, ctx: SelectionContext) -> tuple[ResolvedRoute, ProviderInfo]:
        """Pick the provider that serves *ctx*."""
        return resolve(self, ctx)

    def __init__(
        self,
        raw: dict[str, Any],
        *,
        maintenance: Registry | None = None,
        config_path: str | None = None,
    ) -> None:
        # What the load of *this* document ran, when a reader ran it: the
        # load-time steps' results, for the surface that reports them.  Passed
        # in by the caller that read the document rather than claimed from a
        # slot, so a config built from a literal reports nothing.
        self.maintenance: Registry | None = maintenance
        # Where this document lives, when the caller read it from a file: the
        # anchor a relative ``server.data_dir`` resolves against, and with it
        # every provider's credential-record directory.
        self.config_path: str | None = config_path

        # Declarative external shim sources, parsed and validated here so the
        # document the panel reads and the one the apply path loads cannot
        # disagree about its shape (#1330).  The signature is computed from
        # the raw section, because that is what the reload path compares.
        from ..shim_sources import parse_shims_section, shims_signature

        raw_shims = raw.get("shims")
        self.shims = parse_shims_section(raw_shims, config_path)
        self.shims_signature = shims_signature({"shims": raw_shims})

        all_providers: dict[str, dict[str, str]] = raw.get("providers", {})

        # The unfiltered view, for consumers that have to see a *disabled*
        # provider too: the shim-removal advisory warns about a config that would
        # break the moment it is switched back on.  It is the caller's mapping,
        # so `_resolve_provider_types` correcting legacy names in place is
        # visible here as well.
        self.raw_providers: dict[str, dict[str, str]] = all_providers

        # Filter out disabled providers (enabled defaults to True)
        self._raw_providers: dict[str, dict[str, str]] = {
            name: cfg
            for name, cfg in all_providers.items()
            if cfg.get("enabled", True) is not False
        }

        self.provider_identities, self.provider_shim_names = (
            resolve_provider_identities(self._raw_providers)
        )
        # ``provider_types`` keeps meaning the wire format for existing
        # consumers (routing, proxy, observability, admin).  Read the pair from
        # ``provider_identities`` when the category matters.
        self.provider_types: dict[str, str] = {
            name: ident.format for name, ident in self.provider_identities.items()
        }

        self.provider_supports_custom_tools = parse_custom_tools_overrides(
            self._raw_providers
        )

        self.provider_hoist_system_messages = parse_hoist_system_overrides(
            self._raw_providers
        )

        self.provider_preflight_token_count = parse_preflight_token_count_overrides(
            self._raw_providers
        )

        self.provider_max_tool_description_length = (
            parse_max_tool_description_length_overrides(self._raw_providers)
        )

        self.provider_key_affinity = parse_key_affinity_overrides(self._raw_providers)

        self.models, self.model_capabilities, self.model_upstream_names = parse_models(
            raw.get("models", {}), self._raw_providers
        )

        raw_models = raw.get("models", {})
        self.declared_model_lanes = declared_model_lanes(raw_models, all_providers)
        # Filled by the CLI's load-time validation run, read once by the
        # factory when it opens the ops log (issue #1386).
        self.provider_validation_result: dict[str, Any] | None = None
        (
            self.model_url_templates,
            self.model_stream_url_templates,
            self.model_reasoning_overrides,
            self.model_flatten_system,
            self.model_timeouts,
            self.model_max_tool_description_length,
        ) = parse_model_overrides(raw_models)
        self.model_failover = parse_model_failover(raw_models)

        _server = raw.get("server", {})
        apply_server_settings(self, _server)
        apply_auth_settings(self, _server)
        apply_rate_limit_settings(self, _server)
        apply_circuit_breaker_settings(self, _server)

        apply_debug_settings(self, raw.get("debug", {}))

        # The top-level routing defaults.  Parsed here so a malformed section
        # fails the load, not the first request, and so the panel can show the
        # same values the router will apply.
        apply_routing_settings(self, raw.get("routing"))

        validate(self, raw.get("models", {}))

        # Build ProviderInfo objects (with key rotation support)
        self.providers: dict[str, ProviderInfo] = {
            name: build_provider_info(
                self.provider_types[name],
                cfg,
                global_proxy=self.proxy,
                shim_name=self.provider_shim_names.get(name),
                config_name=name,
                data_dir=self.data_dir,
            )
            for name, cfg in self._raw_providers.items()
        }

        # Attach the process-wide quota state: the key ring learns which
        # credentials a probe or an in-band 429 has demoted.  State outlives a
        # config reload, so it is bound here rather than owned by the config.
        from ..quota.service import bind_provider_quota

        from ..quota.service import quota_intervals

        intervals = quota_intervals(self)
        for provider_name, provider_info in self.providers.items():
            bind_provider_quota(provider_name, provider_info)
            provider_info.quota_interval = intervals.get(provider_name, 0)

        # Upstream-account identity per provider, derived from the credential
        # source at build time.  Two providers sharing a token front one
        # account — the key tier-2 same-format routing compares on.  Computed
        # here because the env var *name* is already expanded away by then.
        self.provider_account_tokens: dict[str, str] = {
            name: pinfo.account_token(name) for name, pinfo in self.providers.items()
        }

        # Provider selection, as stages over one request's selection context.
        # The order is checked when the chain is built, so a stage added later
        # fails at construction rather than on the first request.
        self._same_format = SameFormatStage()
        stages: list[RoutingStage] = [EligibilityStage(), AnchorStage()]
        if self.prefer_same_format:
            stages.append(self._same_format)
        # Rules run last: they start from the provider the anchor and the
        # tiers chose, so a rule composes with them instead of replacing them.
        stages.append(RulesStage())
        self._selection = RoutingChain(stages)

        # The typed pools' candidates, filled while the pools are built.
        self.typed_model_candidates = {}
        # Typed models are scored by the same two stages the chat route starts
        # with: eligibility, then the anchor.  No same-format tier and no rules
        # apply to them.
        self._typed_selection = RoutingChain([EligibilityStage(), AnchorStage()])
        self._typed_routes: dict[tuple[str, str], Any] = {}

        # --- Typed (non-LLM) endpoint pools ---
        # One loop over the model-type registry: every non-LLM type gets its
        # providers / provider_infos / models pools from the unified provider
        # entries plus its legacy ``<type>_providers`` section, and its
        # default format from the registry.  A newly registered type needs no
        # edit here.
        #
        # Order dependency: the typed endpoints of a ``providers:`` entry draw
        # their credential from that entry's ProviderInfo, so ``self.providers``
        # must already be built above.  The pool is read straight off the
        # attribute — no ``getattr`` fallback — so moving this loop earlier
        # fails the load with an AttributeError instead of quietly leaving every
        # typed endpoint without an owner.  A test pins the shared key ring.
        from llm_rosetta.formats import legacy_model_type

        credential_owners = self.providers
        for descriptor in _typed_descriptors():
            name = descriptor.name
            # A legacy spelling of the type names the same pools (the v1
            # config's ``embedding_providers`` / ``embedding_models`` /
            # ``default_embedding_format``).  Normalise it here so the loop
            # stays one shape and a programmatic config that skipped the
            # file migration still loads.
            old_name = legacy_model_type(name)
            old_providers = raw.get(f"{old_name}_providers", {}) if old_name else {}
            old_models = raw.get(f"{old_name}_models", {}) if old_name else {}
            pools = parse_typed_config(
                self._raw_providers,
                {},
                descriptor=descriptor,
                require_declared_format=True,
                map_models=False,
                global_proxy=self.proxy,
                credential_owner=credential_owners,
            )
            legacy = parse_typed_config(
                raw.get(f"{name}_providers") or old_providers,
                raw.get(f"{name}_models") or old_models,
                descriptor=descriptor,
                format_key="format",
                global_proxy=self.proxy,
            )
            # Unified entries win; the legacy section fills the gaps.
            providers, provider_infos, models = pools
            for key, value in legacy[0].items():
                providers.setdefault(key, value)
            for key, value in legacy[1].items():
                provider_infos.setdefault(key, value)
            for key, value in legacy[2].items():
                models.setdefault(key, value)
            setattr(self, f"{name}_providers", providers)
            setattr(self, f"{name}_provider_infos", provider_infos)
            setattr(self, f"{name}_models", models)
            default_format = raw.get(f"default_{name}_format")
            if default_format is None and old_name is not None:
                default_format = raw.get(f"default_{old_name}_format")
            setattr(
                self,
                f"default_{name}_format",
                default_format
                if default_format is not None
                else (descriptor.default_format or ""),
            )

        # --- Extract type-tagged models from unified models pool ---
        distribute_typed_models(self, raw.get("models", {}))

    _VALID_RL_ALGORITHMS = frozenset(
        {"token_bucket", "fixed_window", "sliding_window", "gcra"}
    )

    # ---- Unified endpoint extraction ------------------------------------

    def configured_secrets(self) -> frozenset[str]:
        """Every credential the gateway holds for its upstream providers.

        Read at call time, so keys refreshed or minted since startup are
        included.  For scrubbing text that may echo one, never for display.
        """
        pools = [self.providers]
        pools += [
            getattr(self, f"{d.name}_provider_infos") for d in _typed_descriptors()
        ]
        found: set[str] = set()
        for pool in pools:
            for info in pool.values():
                found.update(info.key_ring.keys())
                if info.token_input:
                    found.add(info.token_input)
        return frozenset(found)

    def typed_pool(
        self, type_name: str
    ) -> tuple[dict[str, dict[str, Any]], dict[str, ProviderInfo], dict[str, str]]:
        """The ``(providers, provider_infos, models)`` pools of *type_name*.

        The registry-driven counterpart of the per-type attributes, for a
        caller that only has the type's name.  A legacy spelling names the
        same pools.
        """
        from llm_rosetta.formats import canonical_model_type

        type_name = canonical_model_type(type_name)
        return (
            getattr(self, f"{type_name}_providers"),
            getattr(self, f"{type_name}_provider_infos"),
            getattr(self, f"{type_name}_models"),
        )

    def _typed_model_route(self, type_name: str, model: str) -> Any:
        """The selection route for one typed model, built once per config.

        Raises:
            KeyError: If the model is not configured for that type.
        """
        from ..routing import ModelRoute, ProviderEntry

        key = (type_name, model)
        cached = self._typed_routes.get(key)
        if cached is not None:
            return cached
        models: dict[str, str] = getattr(self, f"{type_name}_models")
        infos: dict[str, ProviderInfo] = getattr(self, f"{type_name}_provider_infos")
        if model not in models:
            raise KeyError(model)
        entries = list(self.typed_model_candidates.get(type_name, {}).get(model) or [])
        if not entries:
            # A model from a legacy ``<type>_models`` section names one
            # provider and carries no weight.
            entries = [ProviderEntry(name=models[model])]
        entries = [entry for entry in entries if entry.name in infos]
        if not entries:
            raise KeyError(model)
        route = ModelRoute(providers=entries, route_id=f"{type_name}:{model}")
        self._typed_routes[key] = route
        return route

    def resolve_typed(
        self,
        type_name: str,
        model: str,
        *,
        excluded: frozenset[str] = frozenset(),
    ) -> TypedRoute:
        """Resolve *model* to its provider config for *type_name*.

        The one implementation behind ``resolve_embedding``, ``resolve_rerank``
        and ``resolve_decision``; a newly registered type resolves through it
        without a new method.  A typed model with several providers goes
        through the same selector the chat route uses: eligibility first, then
        the anchor over the configured weights, so a spent or blocked first
        provider is passed over instead of always being chosen.

        Args:
            type_name: The model type (``embeddings``, ``rerank``, …).
            model: The model name the client asked for.
            excluded: Providers this request already failed over from.

        Raises:
            KeyError: If the model is not configured for that type.
            ProviderNotReady: If no candidate can serve it right now.
        """
        from llm_rosetta.formats import canonical_model_type

        from ..model_types import get_model_type

        type_name = canonical_model_type(type_name)
        descriptor = get_model_type(type_name)
        path_key = (
            descriptor.config_path_key if descriptor is not None else None
        ) or f"{type_name}_path"
        route = self._typed_model_route(type_name, model)
        providers: dict[str, dict[str, Any]] = getattr(self, f"{type_name}_providers")
        infos: dict[str, ProviderInfo] = getattr(self, f"{type_name}_provider_infos")

        ctx = SelectionContext(model, "")
        ctx.install_route(
            model_route=route,
            providers=infos,
            provider_types={},
            provider_shim_names={},
            provider_account_tokens={},
        )
        if excluded:
            ctx.with_failover_excluded(excluded)
        self._typed_selection.run(ctx)
        if ctx.chosen is None:
            from ..maintenance.startup import ProviderNotReady
            from ..routing.unavailable import classify

            raise ProviderNotReady(
                model, **classify(self, ctx, providers=infos).as_kwargs()
            )

        provider_name = ctx.chosen.name
        pcfg = providers[provider_name]
        return TypedRoute(
            provider_name=provider_name,
            format=pcfg["format"],
            path=pcfg[path_key],
            provider_info=infos[provider_name],
            params=pcfg.get("params") or None,
            upstream_model=ctx.chosen.upstream_model
            or self.model_upstream_names.get(model),
        )

    def resolve_embedding(
        self, model: str, *, excluded: frozenset[str] = frozenset()
    ) -> TypedRoute:
        """Resolve an embedding model to its provider config.

        Args:
            model: The model name the client asked for.
            excluded: Providers this request already failed over from.

        Raises:
            KeyError: If the model is not in embeddings_models.
            ProviderNotReady: If no candidate can serve it right now.
        """
        return self.resolve_typed("embeddings", model, excluded=excluded)

    def resolve_rerank(
        self, model: str, *, excluded: frozenset[str] = frozenset()
    ) -> TypedRoute:
        """Resolve a rerank model to its provider config.

        Args:
            model: The model name the client asked for.
            excluded: Providers this request already failed over from.

        Raises:
            KeyError: If the model is not in rerank_models.
            ProviderNotReady: If no candidate can serve it right now.
        """
        return self.resolve_typed("rerank", model, excluded=excluded)

    def resolve_decision(
        self, model: str, *, excluded: frozenset[str] = frozenset()
    ) -> TypedRoute:
        """Resolve a decision model to its provider config.

        Args:
            model: The model name the client asked for.
            excluded: Providers this request already failed over from.

        Raises:
            KeyError: If the model is not in decisions_models.
            ProviderNotReady: If no candidate can serve it right now.
        """
        return self.resolve_typed("decisions", model, excluded=excluded)
