"""Tests for multi-provider routing strategies."""

from __future__ import annotations

import pytest

from llm_rosetta.gateway.routing_strategy import (
    DEFAULT_STRATEGY,
    AffinityRoundRobinStrategy,
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

    def test_identity_param_ignored(self):
        """WeightedRoundRobinStrategy accepts but ignores identity."""
        s = WeightedRoundRobinStrategy()
        providers = [ProviderEntry("a"), ProviderEntry("b")]
        results = [s.select(providers, identity="client-1") for _ in range(6)]
        assert results.count("a") == 3
        assert results.count("b") == 3


class TestAffinityRoundRobinStrategy:
    def test_deterministic(self):
        """Same identity always picks the same provider."""
        s = AffinityRoundRobinStrategy()
        providers = [ProviderEntry("a"), ProviderEntry("b"), ProviderEntry("c")]
        first = s.select(providers, identity="user-42")
        for _ in range(20):
            assert s.select(providers, identity="user-42") == first

    def test_no_identity_fallback(self):
        """Without identity, falls back to weighted round-robin."""
        s = AffinityRoundRobinStrategy()
        providers = [ProviderEntry("a"), ProviderEntry("b")]
        # No identity — should alternate like WRR
        results = [s.select(providers) for _ in range(6)]
        assert results.count("a") == 3
        assert results.count("b") == 3

    def test_none_identity_fallback(self):
        """Explicit identity=None falls back to weighted round-robin."""
        s = AffinityRoundRobinStrategy()
        providers = [ProviderEntry("a"), ProviderEntry("b")]
        results = [s.select(providers, identity=None) for _ in range(6)]
        assert results.count("a") == 3
        assert results.count("b") == 3

    def test_single_provider(self):
        """Single provider always returns it regardless of identity."""
        s = AffinityRoundRobinStrategy()
        providers = [ProviderEntry("only")]
        assert s.select(providers, identity="x") == "only"
        assert s.select(providers) == "only"

    def test_empty_raises(self):
        s = AffinityRoundRobinStrategy()
        with pytest.raises(ValueError, match="No providers"):
            s.select([], identity="x")

    def test_distribution(self):
        """Different identities distribute across providers."""
        s = AffinityRoundRobinStrategy()
        providers = [ProviderEntry("a"), ProviderEntry("b"), ProviderEntry("c")]
        chosen = {s.select(providers, identity=f"id-{i}") for i in range(100)}
        # With 100 distinct identities across 3 providers, all should appear
        assert chosen == {"a", "b", "c"}

    def test_stable_across_instances(self):
        """Two separate strategy instances produce the same mapping."""
        s1 = AffinityRoundRobinStrategy()
        s2 = AffinityRoundRobinStrategy()
        providers = [ProviderEntry("a"), ProviderEntry("b")]
        for identity in ("alice", "bob", "charlie", "key-hash-abc123"):
            assert s1.select(providers, identity=identity) == s2.select(
                providers, identity=identity
            )


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

    def test_select_with_identity(self):
        """ModelRoute.select passes identity through to strategy."""
        route = ModelRoute(
            [ProviderEntry("a"), ProviderEntry("b")],
            strategy=AffinityRoundRobinStrategy(),
        )
        first = route.select(identity="client-x")
        for _ in range(10):
            assert route.select(identity="client-x") == first

    def test_select_entry_with_identity(self):
        """ModelRoute.select_entry passes identity through to strategy."""
        route = ModelRoute(
            [ProviderEntry("a"), ProviderEntry("b")],
            strategy=AffinityRoundRobinStrategy(),
        )
        entry = route.select_entry(identity="client-y")
        assert isinstance(entry, ProviderEntry)
        for _ in range(10):
            assert route.select_entry(identity="client-y").name == entry.name


class TestCreateStrategy:
    def test_default_strategy(self):
        s = create_strategy(DEFAULT_STRATEGY)
        assert isinstance(s, WeightedRoundRobinStrategy)

    def test_weighted_round_robin(self):
        s = create_strategy("weighted_round_robin")
        assert isinstance(s, WeightedRoundRobinStrategy)

    def test_affinity_round_robin(self):
        s = create_strategy("affinity_round_robin")
        assert isinstance(s, AffinityRoundRobinStrategy)

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Unknown routing strategy"):
            create_strategy("does_not_exist")
