"""Multi-provider routing strategies.

Provides the :class:`RoutingStrategy` protocol, concrete strategy
implementations, and the :class:`ModelRoute` container that pairs a
list of provider entries with a strategy for selection.

A strategy keeps its counters **per route**, keyed by :attr:`ModelRoute.route_id`
rather than by the candidate list's identity.  One strategy instance can then
serve a model and every subset of that model's providers that tier selection
builds, each of them advancing a counter of its own — which is what the
candidates' *names* cannot express, and what object identity used to stand in
for.  The id belongs to the caller: it must be stable for exactly as long as
the counter should survive.  A config reload builds new route objects and a new
strategy with them, so counters start over either way.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from typing import Any, Protocol
from collections.abc import Mapping

from .model_routing import ModelRouting

_U64 = float(1 << 64)


@dataclass(slots=True)
class ProviderEntry:
    """A provider candidate with an associated weight.

    ``weight`` is a ratio, not a count: a request-time factor (an allowance
    headroom, say) scales it, and rounding a scaled weight back to an integer
    would erase the difference between "untouched" and "nearly spent" for the
    default of ``1``.
    """

    name: str
    weight: float = 1.0
    upstream_model: str | None = None


class RoutingStrategy(Protocol):
    """Interface for selecting a provider from a weighted list."""

    def select(
        self,
        providers: list[ProviderEntry],
        *,
        route_id: str,
        identity: str | None = None,
    ) -> str:
        """Return the name of the chosen provider.

        Args:
            providers: Candidate providers with weights.
            route_id: Names the route these candidates belong to.  Strategies
                that hold state key it on this, so two routes sharing a
                strategy instance keep separate state.
            identity: Optional client identity string for affinity-based
                routing.  Strategies that don't use affinity ignore it.
        """
        ...


class WeightedRoundRobinStrategy:
    """Smooth weighted round-robin (nginx-style), counted per route.

    Each route tracks a ``current_weight`` per candidate.  On every call the
    effective weight is added, the highest-weight candidate is chosen, and the
    total weight is subtracted from the winner.  This produces an even
    interleaving — e.g. weights [5, 1] yield the sequence A A A A A B rather
    than bursting all A's then all B's.

    A route whose candidate count changes starts its counters over: the
    positions the counters are indexed by no longer mean the same candidates.
    """

    def __init__(self) -> None:
        self._current_weights: dict[str, list[float]] = {}

    def select(
        self,
        providers: list[ProviderEntry],
        *,
        route_id: str,
        identity: str | None = None,
    ) -> str:
        n = len(providers)
        if n == 0:
            raise ValueError("No providers configured")
        if n == 1:
            return providers[0].name

        weights = self._current_weights.get(route_id)
        if weights is None or len(weights) != n:
            weights = [0.0] * n
            self._current_weights[route_id] = weights

        total = sum(p.weight for p in providers)
        best_idx = 0
        best_weight = float("-inf")
        for i, p in enumerate(providers):
            weights[i] += p.weight
            if weights[i] > best_weight:
                best_weight = weights[i]
                best_idx = i

        weights[best_idx] -= total
        return providers[best_idx].name


class AffinityRoundRobinStrategy:
    """Keep one client on one provider, in proportion to the weights.

    When an ``identity`` string is provided, the choice is weighted rendezvous
    hashing: each candidate draws ``-ln(u) / weight`` for a hash ``u`` of
    ``(identity, provider)``, and the **smallest** draw wins — the weight is
    the rate of an exponential race, so a heavier provider wins more often.
    That keeps the mapping stable for one identity — the same client keeps the
    same provider, and its prompt cache — while a heavier provider wins in
    proportion to its weight.  Hashing the identity modulo the candidate count,
    as this did before, ignored the weight entirely, so a nearly spent provider
    kept its full share of the affinity traffic.

    Falls back to weighted round-robin when no identity is provided
    or the provider list is trivial.
    """

    def __init__(self) -> None:
        self._fallback = WeightedRoundRobinStrategy()

    def select(
        self,
        providers: list[ProviderEntry],
        *,
        route_id: str,
        identity: str | None = None,
    ) -> str:
        if not providers:
            raise ValueError("No providers configured")
        if len(providers) == 1:
            return providers[0].name
        if identity is None:
            return self._fallback.select(providers, route_id=route_id)

        best = providers[0].name
        best_score = float("inf")
        for provider in providers:
            digest = hashlib.sha256(f"{identity}\0{provider.name}".encode()).digest()
            # A uniform draw from (0, 1]; the +1 keeps it away from zero, where
            # the logarithm would be infinite.
            draw = (int.from_bytes(digest[:8], "big") + 1) / _U64
            score = -math.log(draw) / max(provider.weight, 1e-9)
            if score < best_score:
                best_score = score
                best = provider.name
        return best


class OrderedStrategy:
    """Always the first candidate — the operator's order is the policy.

    The cheapest thing that can be called a strategy: the list is sorted by
    preference, and the first eligible provider serves every request.  Weights
    are ignored, because the point of naming this strategy is to say the
    order *is* the policy.
    """

    def select(
        self,
        providers: list[ProviderEntry],
        *,
        route_id: str,
        identity: str | None = None,
    ) -> str:
        if not providers:
            raise ValueError("No providers configured")
        return providers[0].name


class RotateStrategy:
    """Plain round-robin: every candidate takes its turn.

    Unlike weighted round-robin, a provider configured at ``weight: 5`` gets
    no larger a share than its peer at ``weight: 1``.  Counted per route, so a
    tier subset takes turns of its own.
    """

    def __init__(self) -> None:
        self._next: dict[str, int] = {}

    def select(
        self,
        providers: list[ProviderEntry],
        *,
        route_id: str,
        identity: str | None = None,
    ) -> str:
        n = len(providers)
        if n == 0:
            raise ValueError("No providers configured")
        index = self._next.get(route_id, 0) % n
        self._next[route_id] = (index + 1) % n
        return providers[index].name


class LeastUsedStrategy:
    """The candidate that has served the fewest requests.

    Counted per route from the moment the route was built — this is a
    process-local tie-breaker, not a load monitor: it knows what *this* gateway
    has sent, not what the upstream is doing.  That is enough to keep a fresh
    provider catch up with a long-serving one.
    """

    def __init__(self) -> None:
        self._counts: dict[str, list[int]] = {}

    def select(
        self,
        providers: list[ProviderEntry],
        *,
        route_id: str,
        identity: str | None = None,
    ) -> str:
        n = len(providers)
        if n == 0:
            raise ValueError("No providers configured")
        counts = self._counts.get(route_id)
        if counts is None or len(counts) != n:
            counts = [0] * n
            self._counts[route_id] = counts
        best = min(range(n), key=lambda i: counts[i])
        counts[best] += 1
        return providers[best].name


class PaceStrategy:
    """Weighted round-robin over the allowance headroom alone.

    The configured weight is *not* applied: this is the strategy that says
    "spend the provider with the most room left, whatever the operator
    configured".  A provider nothing has measured counts as having full room,
    so it is tried before one that is nearly spent.
    """

    #: Read by :class:`ModelRoute`: scale by headroom only, not by weight.
    uses_headroom_only = True

    def __init__(self) -> None:
        self._fallback = WeightedRoundRobinStrategy()

    def select(
        self,
        providers: list[ProviderEntry],
        *,
        route_id: str,
        identity: str | None = None,
    ) -> str:
        return self._fallback.select(providers, route_id=route_id)


# Strategy registry.  The old names stay: a config written before the model
# routing block still names them, and the block's own vocabulary is a superset.
_STRATEGIES: dict[str, type[RoutingStrategy]] = {
    "weighted": WeightedRoundRobinStrategy,
    "ordered": OrderedStrategy,
    "rotate": RotateStrategy,
    "least_used": LeastUsedStrategy,
    "pace": PaceStrategy,
    "weighted_round_robin": WeightedRoundRobinStrategy,
    "affinity_round_robin": AffinityRoundRobinStrategy,
}

DEFAULT_STRATEGY = "weighted"


def create_strategy(name: str) -> RoutingStrategy:
    """Instantiate a routing strategy by name.

    Raises:
        ValueError: If the strategy name is not recognized.
    """
    cls = _STRATEGIES.get(name)
    if cls is None:
        valid = ", ".join(sorted(_STRATEGIES))
        raise ValueError(
            f"Unknown routing strategy '{name}'. Valid strategies: {valid}"
        )
    return cls()


def strategy_for(routing: Any) -> RoutingStrategy:
    """Build the strategy a model's routing block names.

    ``affinity: session`` replaces the per-request choice with one client on
    one provider, which is why the parser only allows it with ``weighted``.
    """
    if getattr(routing, "affinity", "off") == "session":
        return AffinityRoundRobinStrategy()
    return create_strategy(routing.strategy)


@dataclass
class ModelRoute:
    """A model's provider list and routing strategy.

    Wraps the provider selection so that callers simply call
    :meth:`select` without knowing the strategy details.

    Attributes:
        providers: The candidates, in configuration order.
        route_id: Names this route to its strategy, which keys its per-route
            state on it.  A model and a subset of that model's providers are
            different routes and need different ids, or they would take turns
            out of one counter.
        strategy: The selection strategy.
    """

    providers: list[ProviderEntry]
    route_id: str
    strategy: RoutingStrategy = field(default_factory=WeightedRoundRobinStrategy)
    routing: ModelRouting = field(default_factory=ModelRouting)

    def __post_init__(self) -> None:
        self._entry_by_name = {p.name: p for p in self.providers}

    def _candidates(
        self,
        weights: Mapping[str, float] | None,
        excluded: frozenset[str] = frozenset(),
        sunk: frozenset[str] = frozenset(),
    ) -> list[ProviderEntry]:
        """The entries the strategy sees, with this request's weights applied.

        ``excluded`` removes candidates *before* the strategy counts them.
        Letting a strategy pick one and discarding the pick afterwards would
        leave the excluded provider consuming turns, so the survivors would
        not get the share the weights promise.

        ``sunk`` is an ordering input, not a weight: a provider that just
        answered 429 keeps its configured weight but is only offered to the
        strategy when every provider above it is gone.  Dropping it from the
        list is what "last" means to every strategy at once — a round-robin
        counter and a rendezvous hash both only ever see what they are given.
        A sunk provider still serves a route with nothing else, which is why
        the survivors are checked before the drop.

        The configured weight is the operator's statement of preference; a
        request-time factor only scales it, and never to zero — a live provider
        with a nearly spent window still serves its share.
        """
        entries = self.providers
        if excluded:
            entries = [p for p in entries if p.name not in excluded]
        if sunk and self.routing.sink:
            standing = [p for p in entries if p.name not in sunk]
            if standing:
                entries = standing
        headroom_only = getattr(self.strategy, "uses_headroom_only", False)
        if not weights and not headroom_only:
            return entries
        factors = weights or {}
        if headroom_only:
            # "Pace" spends the provider with the most room left; the
            # configured weight is exactly what it is told to ignore.
            return [
                ProviderEntry(
                    name=p.name,
                    weight=max(0.05, factors.get(p.name, 1.0)),
                    upstream_model=p.upstream_model,
                )
                for p in entries
            ]
        return [
            ProviderEntry(
                name=p.name,
                # Never zero: a provider with a spent window is still a
                # candidate — the quota state, not this factor, removes one.
                weight=max(0.05, p.weight * factors.get(p.name, 1.0)),
                upstream_model=p.upstream_model,
            )
            for p in entries
        ]

    def select(
        self,
        *,
        identity: str | None = None,
        weights: Mapping[str, float] | None = None,
        excluded: frozenset[str] = frozenset(),
        sunk: frozenset[str] = frozenset(),
    ) -> str:
        """Pick the next provider according to the strategy.

        Args:
            identity: Optional client identity for affinity routing.
            weights: Optional per-provider multiplier for this selection, e.g.
                an allowance headroom.  Absent means the configured weights.
            excluded: Providers that are not candidates at all this time.
            sunk: Providers that are candidates but are tried last.
        """
        return self.strategy.select(
            self._candidates(weights, excluded, sunk),
            route_id=self.route_id,
            identity=identity,
        )

    def select_entry(
        self,
        *,
        identity: str | None = None,
        weights: Mapping[str, float] | None = None,
        excluded: frozenset[str] = frozenset(),
        sunk: frozenset[str] = frozenset(),
    ) -> ProviderEntry:
        """Pick the next provider and return the full entry.

        Args:
            identity: Optional client identity for affinity routing.
            weights: Optional per-provider multiplier, as in :meth:`select`.
            excluded: Providers that are not candidates at all this time.
            sunk: Providers that are candidates but are tried last.
        """
        return self._entry_by_name[
            self.strategy.select(
                self._candidates(weights, excluded, sunk),
                route_id=self.route_id,
                identity=identity,
            )
        ]

    @property
    def provider_names(self) -> list[str]:
        """All provider names in this route."""
        return [p.name for p in self.providers]

    @property
    def is_multi(self) -> bool:
        """Whether this route has more than one provider."""
        return len(self.providers) > 1
