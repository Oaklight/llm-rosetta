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
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(slots=True)
class ProviderEntry:
    """A provider candidate with an associated weight."""

    name: str
    weight: int = 1
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
        self._current_weights: dict[str, list[int]] = {}

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
            weights = [0] * n
            self._current_weights[route_id] = weights

        total = sum(p.weight for p in providers)
        best_idx = 0
        best_weight = -1
        for i, p in enumerate(providers):
            weights[i] += p.weight
            if weights[i] > best_weight:
                best_weight = weights[i]
                best_idx = i

        weights[best_idx] -= total
        return providers[best_idx].name


class AffinityRoundRobinStrategy:
    """Hash client identity to a preferred provider for cache locality.

    When an ``identity`` string is provided, uses a deterministic hash
    to pick the same provider every time — preserving provider-side
    prompt cache across requests from the same client.

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
        idx = int.from_bytes(
            hashlib.sha256(identity.encode()).digest()[:8], "big"
        ) % len(providers)
        return providers[idx].name


# Strategy registry
_STRATEGIES: dict[str, type[RoutingStrategy]] = {
    "weighted_round_robin": WeightedRoundRobinStrategy,
    "affinity_round_robin": AffinityRoundRobinStrategy,
}

DEFAULT_STRATEGY = "weighted_round_robin"


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

    def __post_init__(self) -> None:
        self._entry_by_name = {p.name: p for p in self.providers}

    def select(self, *, identity: str | None = None) -> str:
        """Pick the next provider according to the strategy.

        Args:
            identity: Optional client identity for affinity routing.
        """
        return self.strategy.select(
            self.providers, route_id=self.route_id, identity=identity
        )

    def select_entry(self, *, identity: str | None = None) -> ProviderEntry:
        """Pick the next provider and return the full entry.

        Args:
            identity: Optional client identity for affinity routing.
        """
        return self._entry_by_name[
            self.strategy.select(
                self.providers, route_id=self.route_id, identity=identity
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
