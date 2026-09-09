"""Multi-provider routing strategies.

Provides the :class:`RoutingStrategy` protocol, concrete strategy
implementations, and the :class:`ModelRoute` container that pairs a
list of provider entries with a strategy for selection.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass(slots=True)
class ProviderEntry:
    """A provider candidate with an associated weight."""

    name: str
    weight: int = 1


class RoutingStrategy(Protocol):
    """Interface for selecting a provider from a weighted list."""

    def select(self, providers: list[ProviderEntry]) -> str:
        """Return the name of the chosen provider."""
        ...


class WeightedRoundRobinStrategy:
    """Smooth weighted round-robin (nginx-style).

    Each provider tracks a ``current_weight``.  On every call the
    effective weight is added, the highest-weight provider is chosen,
    and the total weight is subtracted from the winner.  This produces
    an even interleaving — e.g. weights [5, 1] yield the sequence
    A A A A A B rather than bursting all A's then all B's.
    """

    def __init__(self) -> None:
        self._current_weights: list[int] = []
        self._initialized_for: list[ProviderEntry] | None = None

    def select(self, providers: list[ProviderEntry]) -> str:
        n = len(providers)
        if n == 0:
            raise ValueError("No providers configured")
        if n == 1:
            return providers[0].name

        if self._initialized_for is not providers:
            self._current_weights = [0] * n
            self._initialized_for = providers

        total = sum(p.weight for p in providers)
        best_idx = 0
        best_weight = -1
        for i, p in enumerate(providers):
            self._current_weights[i] += p.weight
            if self._current_weights[i] > best_weight:
                best_weight = self._current_weights[i]
                best_idx = i

        self._current_weights[best_idx] -= total
        return providers[best_idx].name


# Strategy registry
_STRATEGIES: dict[str, type[RoutingStrategy]] = {
    "weighted_round_robin": WeightedRoundRobinStrategy,
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
    """

    providers: list[ProviderEntry]
    strategy: RoutingStrategy = field(default_factory=WeightedRoundRobinStrategy)

    def select(self) -> str:
        """Pick the next provider according to the strategy."""
        return self.strategy.select(self.providers)

    @property
    def provider_names(self) -> list[str]:
        """All provider names in this route."""
        return [p.name for p in self.providers]

    @property
    def is_multi(self) -> bool:
        """Whether this route has more than one provider."""
        return len(self.providers) > 1
