"""Same-format preference: serve a request in the format its client speaks.

The tiers are relative to the **anchor** — the provider routing would have
picked with the preference off — so turning the preference on does not change
which provider the full route's counter advances to.  What it changes is
whether the request is then realigned onto a provider that speaks the client's
format, and how far it is willing to move to do it:

1. same shim and endpoint as the anchor,
2. the same upstream account, possibly another format,
3. any provider in the client's format.

Tier 1 cannot change the format — a shim fixes its base — so it only orders
failover between identical endpoints.  Tier 2 is the tier that realigns the
format without moving to another account.  When no tier can serve the request,
the anchor stands and the request is converted through the IR.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

from .select import select_ready_entry
from .stage import RoutingStage
from .strategy import ModelRoute

if TYPE_CHECKING:
    from .context import SelectionContext
    from .strategy import ProviderEntry


class FormatGroups(NamedTuple):
    """Static partitions of a model's providers in one source format.

    Built once per ``(model, source_format)`` and reused across requests; the
    per-request part of tier selection (which provider is the anchor) is not
    part of this.
    """

    covers_all: bool
    """Every provider of the model already speaks the source format."""

    all_group: ModelRoute
    """Every provider in the source format (tier 3)."""

    by_shim: dict[tuple[str | None, str], ModelRoute]
    """Keyed by ``(shim name, base_url)`` — same endpoint (tier 1)."""

    by_account: dict[str, ModelRoute]
    """Keyed by the account token — same upstream account (tier 2)."""


def shim_group_key(ctx: SelectionContext, provider_name: str) -> tuple[str | None, str]:
    """Return the tier-1 key for *provider_name*: same shim, same endpoint.

    ``base_url`` is part of the key so two unrelated bare-format providers
    (both with no shim) are not mistaken for one endpoint.
    """
    return (
        ctx.provider_shim_names.get(provider_name),
        ctx.providers[provider_name].base_url,
    )


def _subset_route_id(model: str, kind: str, key: object) -> str:
    """Name one tier subset's route, so it keeps counters of its own.

    The model is part of the id because two models may list the same providers
    and must still take turns independently of each other.
    """
    return f"{model}\x1f{kind}\x1f{key}"


class SameFormatStage(RoutingStage):
    """Realign the anchor onto a provider that speaks the client's format.

    Only the *static* partitions are cached — they are a function of the config
    — while which provider is the anchor is per request.  The stage is built
    per config, so a reload starts with an empty cache, exactly as the former
    config attribute did.
    """

    __slots__ = ("_groups",)

    name = "same_format"
    requires = frozenset(
        {
            "chosen",
            "model_route",
            "source_format",
            "identity",
            "providers",
            "provider_types",
            "provider_shim_names",
            "provider_account_tokens",
        }
    )
    provides = frozenset({"chosen"})

    def __init__(self) -> None:
        self._groups: dict[tuple[str, str], FormatGroups | None] = {}

    @property
    def cache(self) -> dict[tuple[str, str], FormatGroups | None]:
        """The partitions built so far, keyed by ``(model, source_format)``."""
        return self._groups

    def groups_for(self, ctx: SelectionContext) -> FormatGroups | None:
        """Return the static tier partitions for the context's model and format.

        ``None`` when the model has no provider in that format.  The result is
        a function of the config alone, so it is cached.

        Provider types are compared as normalised base types: since #909
        ``open_responses`` is the only Responses base and
        ``openai--open_responses`` is a shim over it, so an OpenAI-profile
        provider and a neutral-spec provider count as the same format.
        """
        key = (ctx.model, ctx.source_format)
        # ``None`` is a meaningful cached value, so test membership.
        if key in self._groups:
            return self._groups[key]

        model_route = ctx.model_route
        assert model_route is not None  # seeded by install_route()
        matching = [
            p
            for p in model_route.providers
            if ctx.provider_types.get(p.name) == ctx.source_format
        ]
        if not matching:
            groups = None
        else:
            # Every subset shares the model's strategy instance and names its
            # own route, so each keeps a counter of its own while the strategy
            # itself stays one object per model.
            def _sub(
                members: list[ProviderEntry], kind: str, key: object
            ) -> ModelRoute:
                return ModelRoute(
                    members,
                    _subset_route_id(ctx.model, kind, key),
                    model_route.strategy,
                )

            by_shim: dict[tuple[str | None, str], list[ProviderEntry]] = {}
            by_account: dict[str, list[ProviderEntry]] = {}
            for provider in matching:
                by_shim.setdefault(shim_group_key(ctx, provider.name), []).append(
                    provider
                )
                by_account.setdefault(
                    ctx.provider_account_tokens[provider.name], []
                ).append(provider)

            groups = FormatGroups(
                covers_all=len(matching) == len(model_route.providers),
                all_group=_sub(matching, "format", ctx.source_format),
                by_shim={k: _sub(v, "shim", k) for k, v in by_shim.items()},
                by_account={k: _sub(v, "account", k) for k, v in by_account.items()},
            )
        self._groups[key] = groups
        return groups

    def select_tiered(
        self,
        groups: FormatGroups,
        anchor: ProviderEntry,
        ctx: SelectionContext,
    ) -> ProviderEntry | None:
        """Pick the ready provider closest to *anchor* among same-format ones.

        Tries tier 1 (same shim and endpoint as *anchor*), then tier 2 (same
        account, possibly another format), then tier 3 (any same-format
        provider).  ``None`` when no same-format provider is ready, leaving the
        caller to fall back to *anchor* and convert through IR.
        """
        for route in (
            groups.by_shim.get(shim_group_key(ctx, anchor.name)),
            groups.by_account.get(ctx.provider_account_tokens[anchor.name]),
            groups.all_group,
        ):
            if route is None:
                continue
            entry = select_ready_entry(
                route, is_ready=ctx.is_ready, identity=ctx.identity.client
            )
            if entry is not None:
                return entry
        return None

    def run(self, ctx: SelectionContext) -> None:
        """Realign the anchor, or leave it as the previous stage left it."""
        if ctx.chosen is None:
            # Nothing was ready; the caller reports that, and there is no
            # anchor to realign.
            return
        groups = self.groups_for(ctx)
        # A homogeneous route has nothing to realign — the anchor already
        # speaks the client's format — so it keeps the plain-route behavior,
        # counters included.
        if groups is None or groups.covers_all:
            return
        realigned = self.select_tiered(groups, ctx.chosen, ctx)
        if realigned is not None:
            ctx.choose(realigned)
