"""Tests for multi-provider routing strategies."""

from __future__ import annotations

import pytest

from llm_rosetta.gateway.routing import (
    DEFAULT_STRATEGY,
    AffinityRoundRobinStrategy,
    ModelRoute,
    ProviderEntry,
    WeightedRoundRobinStrategy,
    create_strategy,
)

ROUTE = "test-route"


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
        assert s.select(providers, route_id=ROUTE) == "a"
        assert s.select(providers, route_id=ROUTE) == "a"

    def test_equal_weight_alternates(self):
        s = WeightedRoundRobinStrategy()
        providers = [ProviderEntry("a"), ProviderEntry("b")]
        results = [s.select(providers, route_id=ROUTE) for _ in range(6)]
        assert results.count("a") == 3
        assert results.count("b") == 3

    def test_equal_weight_sequence(self):
        """The exact order, not just the counts."""
        s = WeightedRoundRobinStrategy()
        providers = [ProviderEntry("a"), ProviderEntry("b")]
        results = [s.select(providers, route_id=ROUTE) for _ in range(6)]
        assert results == ["a", "b", "a", "b", "a", "b"]

    def test_weighted_distribution(self):
        s = WeightedRoundRobinStrategy()
        providers = [ProviderEntry("a", weight=3), ProviderEntry("b", weight=1)]
        results = [s.select(providers, route_id=ROUTE) for _ in range(4)]
        assert results.count("a") == 3
        assert results.count("b") == 1

    def test_weighted_interleaving(self):
        """Smooth WRR should interleave, not burst."""
        s = WeightedRoundRobinStrategy()
        providers = [ProviderEntry("a", weight=2), ProviderEntry("b", weight=1)]
        results = [s.select(providers, route_id=ROUTE) for _ in range(6)]
        # Should produce a-a-b-a-a-b or a-b-a-a-b-a pattern, not a-a-a-a-b-b
        # Key invariant: no more than 2 consecutive "a"s
        for i in range(len(results) - 2):
            if results[i] == results[i + 1] == results[i + 2] == "a":
                msg = f"Three consecutive 'a' at index {i}: {results}"
                pytest.fail(msg)  # ty: ignore[invalid-argument-type]

    def test_weighted_sequence(self):
        """The exact interleaving, not just the counts."""
        s = WeightedRoundRobinStrategy()
        providers = [ProviderEntry("a", weight=2), ProviderEntry("b", weight=1)]
        results = [s.select(providers, route_id=ROUTE) for _ in range(6)]
        assert results == ["a", "b", "a", "a", "b", "a"]

    def test_three_providers(self):
        s = WeightedRoundRobinStrategy()
        providers = [
            ProviderEntry("a", weight=3),
            ProviderEntry("b", weight=2),
            ProviderEntry("c", weight=1),
        ]
        results = [s.select(providers, route_id=ROUTE) for _ in range(6)]
        assert results.count("a") == 3
        assert results.count("b") == 2
        assert results.count("c") == 1

    def test_empty_raises(self):
        s = WeightedRoundRobinStrategy()
        with pytest.raises(ValueError, match="No providers"):
            s.select([], route_id=ROUTE)

    def test_deterministic(self):
        """Same sequence of selections every time."""
        s1 = WeightedRoundRobinStrategy()
        s2 = WeightedRoundRobinStrategy()
        providers = [ProviderEntry("a", weight=3), ProviderEntry("b", weight=1)]
        r1 = [s1.select(providers, route_id=ROUTE) for _ in range(8)]
        r2 = [s2.select(providers, route_id=ROUTE) for _ in range(8)]
        assert r1 == r2

    def test_large_weight_ratio(self):
        s = WeightedRoundRobinStrategy()
        providers = [ProviderEntry("a", weight=99), ProviderEntry("b", weight=1)]
        results = [s.select(providers, route_id=ROUTE) for _ in range(100)]
        assert results.count("a") == 99
        assert results.count("b") == 1

    def test_identity_param_ignored(self):
        """WeightedRoundRobinStrategy accepts but ignores identity."""
        s = WeightedRoundRobinStrategy()
        providers = [ProviderEntry("a"), ProviderEntry("b")]
        results = [
            s.select(providers, route_id=ROUTE, identity="client-1") for _ in range(6)
        ]
        assert results.count("a") == 3
        assert results.count("b") == 3


class TestPerRouteCounters:
    """One strategy instance serves several routes, each counting on its own."""

    def test_routes_do_not_share_a_counter(self):
        """Another route advancing must not move this route's sequence."""
        providers = [ProviderEntry("a"), ProviderEntry("b")]
        solo = WeightedRoundRobinStrategy()
        expected = [solo.select(providers, route_id="alone") for _ in range(6)]

        shared = WeightedRoundRobinStrategy()
        got = []
        for _ in range(6):
            shared.select(providers, route_id="other")
            got.append(shared.select(providers, route_id="one"))

        assert got == expected

    def test_counter_starts_over_when_the_candidates_change(self):
        """Positions no longer mean the same providers, so the route resets."""
        s = WeightedRoundRobinStrategy()
        two = [ProviderEntry("a"), ProviderEntry("b")]
        three = [ProviderEntry("a"), ProviderEntry("b"), ProviderEntry("c")]
        s.select(two, route_id=ROUTE)
        results = [s.select(three, route_id=ROUTE) for _ in range(3)]
        assert results == ["a", "b", "c"]


class TestAffinityRoundRobinStrategy:
    def test_deterministic(self):
        """Same identity always picks the same provider."""
        s = AffinityRoundRobinStrategy()
        providers = [ProviderEntry("a"), ProviderEntry("b"), ProviderEntry("c")]
        first = s.select(providers, route_id=ROUTE, identity="user-42")
        for _ in range(20):
            assert s.select(providers, route_id=ROUTE, identity="user-42") == first

    def test_no_identity_fallback(self):
        """Without identity, falls back to weighted round-robin."""
        s = AffinityRoundRobinStrategy()
        providers = [ProviderEntry("a"), ProviderEntry("b")]
        # No identity — should alternate like WRR
        results = [s.select(providers, route_id=ROUTE) for _ in range(6)]
        assert results.count("a") == 3
        assert results.count("b") == 3

    def test_none_identity_fallback(self):
        """Explicit identity=None falls back to weighted round-robin."""
        s = AffinityRoundRobinStrategy()
        providers = [ProviderEntry("a"), ProviderEntry("b")]
        results = [s.select(providers, route_id=ROUTE, identity=None) for _ in range(6)]
        assert results.count("a") == 3
        assert results.count("b") == 3

    def test_fallback_counts_per_route(self):
        """The fallback is stateful too, and keyed the same way."""
        s = AffinityRoundRobinStrategy()
        providers = [ProviderEntry("a"), ProviderEntry("b")]
        solo = WeightedRoundRobinStrategy()
        expected = [solo.select(providers, route_id="alone") for _ in range(6)]

        got = []
        for _ in range(6):
            s.select(providers, route_id="other")
            got.append(s.select(providers, route_id="one"))

        assert got == expected

    def test_single_provider(self):
        """Single provider always returns it regardless of identity."""
        s = AffinityRoundRobinStrategy()
        providers = [ProviderEntry("only")]
        assert s.select(providers, route_id=ROUTE, identity="x") == "only"
        assert s.select(providers, route_id=ROUTE) == "only"

    def test_empty_raises(self):
        s = AffinityRoundRobinStrategy()
        with pytest.raises(ValueError, match="No providers"):
            s.select([], route_id=ROUTE, identity="x")

    def test_distribution(self):
        """Different identities distribute across providers."""
        s = AffinityRoundRobinStrategy()
        providers = [ProviderEntry("a"), ProviderEntry("b"), ProviderEntry("c")]
        chosen = {
            s.select(providers, route_id=ROUTE, identity=f"id-{i}") for i in range(100)
        }
        # With 100 distinct identities across 3 providers, all should appear
        assert chosen == {"a", "b", "c"}

    def test_stable_across_instances(self):
        """Two separate strategy instances produce the same mapping."""
        s1 = AffinityRoundRobinStrategy()
        s2 = AffinityRoundRobinStrategy()
        providers = [ProviderEntry("a"), ProviderEntry("b")]
        for identity in ("alice", "bob", "charlie", "key-hash-abc123"):
            assert s1.select(providers, route_id=ROUTE, identity=identity) == s2.select(
                providers, route_id=ROUTE, identity=identity
            )


class TestModelRoute:
    def test_single_provider(self):
        route = ModelRoute([ProviderEntry("openai")], "m")
        assert route.select() == "openai"
        assert not route.is_multi
        assert route.provider_names == ["openai"]

    def test_multi_provider(self):
        route = ModelRoute([ProviderEntry("a"), ProviderEntry("b")], "m")
        assert route.is_multi
        assert set(route.provider_names) == {"a", "b"}
        # Should return both over multiple calls
        results = {route.select() for _ in range(10)}
        assert results == {"a", "b"}

    def test_custom_strategy(self):
        strategy = WeightedRoundRobinStrategy()
        route = ModelRoute(
            [ProviderEntry("a", weight=1), ProviderEntry("b", weight=1)],
            "m",
            strategy=strategy,
        )
        results = [route.select() for _ in range(4)]
        assert results.count("a") == 2
        assert results.count("b") == 2

    def test_two_routes_share_a_strategy_without_sharing_counters(self):
        """The route id, not the instance, decides where the counter lives."""
        providers = [ProviderEntry("a"), ProviderEntry("b")]
        solo = ModelRoute(providers, "alone")
        expected = [solo.select() for _ in range(6)]

        strategy = WeightedRoundRobinStrategy()
        one = ModelRoute(providers, "one", strategy=strategy)
        other = ModelRoute(providers, "other", strategy=strategy)
        got = []
        for _ in range(6):
            other.select()
            got.append(one.select())
        assert got == expected

    def test_select_with_identity(self):
        """ModelRoute.select passes identity through to strategy."""
        route = ModelRoute(
            [ProviderEntry("a"), ProviderEntry("b")],
            "m",
            strategy=AffinityRoundRobinStrategy(),
        )
        first = route.select(identity="client-x")
        for _ in range(10):
            assert route.select(identity="client-x") == first

    def test_select_entry_with_identity(self):
        """ModelRoute.select_entry passes identity through to strategy."""
        route = ModelRoute(
            [ProviderEntry("a"), ProviderEntry("b")],
            "m",
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
