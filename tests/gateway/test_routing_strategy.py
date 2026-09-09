"""Tests for multi-provider routing strategies."""

from __future__ import annotations

import pytest

from llm_rosetta.gateway.routing_strategy import (
    DEFAULT_STRATEGY,
    ModelRoute,
    ProviderEntry,
    WeightedRoundRobinStrategy,
    create_strategy,
)


class TestProviderEntry:
    def test_defaults(self):
        e = ProviderEntry("openai")
        assert e.name == "openai"
        assert e.weight == 1

    def test_custom_weight(self):
        e = ProviderEntry("openai", weight=5)
        assert e.weight == 5


class TestWeightedRoundRobinStrategy:
    def test_single_provider(self):
        s = WeightedRoundRobinStrategy()
        providers = [ProviderEntry("a")]
        assert s.select(providers) == "a"
        assert s.select(providers) == "a"

    def test_equal_weight_alternates(self):
        s = WeightedRoundRobinStrategy()
        providers = [ProviderEntry("a"), ProviderEntry("b")]
        results = [s.select(providers) for _ in range(6)]
        assert results.count("a") == 3
        assert results.count("b") == 3

    def test_weighted_distribution(self):
        s = WeightedRoundRobinStrategy()
        providers = [ProviderEntry("a", weight=3), ProviderEntry("b", weight=1)]
        results = [s.select(providers) for _ in range(4)]
        assert results.count("a") == 3
        assert results.count("b") == 1

    def test_weighted_interleaving(self):
        """Smooth WRR should interleave, not burst."""
        s = WeightedRoundRobinStrategy()
        providers = [ProviderEntry("a", weight=2), ProviderEntry("b", weight=1)]
        results = [s.select(providers) for _ in range(6)]
        # Should produce a-a-b-a-a-b or a-b-a-a-b-a pattern, not a-a-a-a-b-b
        # Key invariant: no more than 2 consecutive "a"s
        for i in range(len(results) - 2):
            if results[i] == results[i + 1] == results[i + 2] == "a":
                pytest.fail(f"Three consecutive 'a' at index {i}: {results}")

    def test_three_providers(self):
        s = WeightedRoundRobinStrategy()
        providers = [
            ProviderEntry("a", weight=3),
            ProviderEntry("b", weight=2),
            ProviderEntry("c", weight=1),
        ]
        results = [s.select(providers) for _ in range(6)]
        assert results.count("a") == 3
        assert results.count("b") == 2
        assert results.count("c") == 1

    def test_empty_raises(self):
        s = WeightedRoundRobinStrategy()
        with pytest.raises(ValueError, match="No providers"):
            s.select([])

    def test_deterministic(self):
        """Same sequence of selections every time."""
        s1 = WeightedRoundRobinStrategy()
        s2 = WeightedRoundRobinStrategy()
        providers = [ProviderEntry("a", weight=3), ProviderEntry("b", weight=1)]
        r1 = [s1.select(providers) for _ in range(8)]
        r2 = [s2.select(providers) for _ in range(8)]
        assert r1 == r2

    def test_large_weight_ratio(self):
        s = WeightedRoundRobinStrategy()
        providers = [ProviderEntry("a", weight=99), ProviderEntry("b", weight=1)]
        results = [s.select(providers) for _ in range(100)]
        assert results.count("a") == 99
        assert results.count("b") == 1


class TestModelRoute:
    def test_single_provider(self):
        route = ModelRoute([ProviderEntry("openai")])
        assert route.select() == "openai"
        assert not route.is_multi
        assert route.provider_names == ["openai"]

    def test_multi_provider(self):
        route = ModelRoute([ProviderEntry("a"), ProviderEntry("b")])
        assert route.is_multi
        assert set(route.provider_names) == {"a", "b"}
        # Should return both over multiple calls
        results = {route.select() for _ in range(10)}
        assert results == {"a", "b"}

    def test_custom_strategy(self):
        strategy = WeightedRoundRobinStrategy()
        route = ModelRoute(
            [ProviderEntry("a", weight=1), ProviderEntry("b", weight=1)],
            strategy=strategy,
        )
        results = [route.select() for _ in range(4)]
        assert results.count("a") == 2
        assert results.count("b") == 2


class TestCreateStrategy:
    def test_default_strategy(self):
        s = create_strategy(DEFAULT_STRATEGY)
        assert isinstance(s, WeightedRoundRobinStrategy)

    def test_weighted_round_robin(self):
        s = create_strategy("weighted_round_robin")
        assert isinstance(s, WeightedRoundRobinStrategy)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown routing strategy"):
            create_strategy("does_not_exist")
