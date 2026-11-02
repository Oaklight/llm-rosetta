"""A model's ``routing`` block: how its providers are chosen between.

The block is the one place routing is configured.  These tests pin three
things: the parse (defaults, unknown keys, the legacy ``strategy`` key it
absorbs), the strategies the block can name, and the two switches the block
carries beyond the strategy — ``affinity`` and ``sink``.
"""

from __future__ import annotations

import pytest

from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.routing import (
    AffinityRoundRobinStrategy,
    Eligibility,
    ModelRoute,
    ModelRouting,
    OrderedStrategy,
    PaceStrategy,
    ProviderEntry,
    RotateStrategy,
    create_strategy,
    parse_model_routing,
    routing_from_legacy,
    strategy_for,
)
from llm_rosetta_gateway.routing.select import select_ready_entry


class TestSingleProviderEntry:
    """A model declared with one provider keeps its routing block.

    The panel writes the block through ``PUT /admin/api/routing/<model>``;
    before this was pinned, a single-provider entry parsed to a route with
    the default block, so the panel saved a policy the runtime ignored.
    """

    PROVS = {"a": {}, "b": {}}

    def test_a_dict_entry_reads_its_block(self):
        from llm_rosetta_gateway.config.parse import parse_model_route

        route = parse_model_route(
            "m",
            {"provider": "a", "routing": {"strategy": "ordered", "affinity": "off"}},
            self.PROVS,
        )

        assert route is not None
        assert [p.name for p in route.providers] == ["a"]
        assert route.routing.strategy == "ordered"
        assert isinstance(route.strategy, OrderedStrategy)

    def test_a_string_entry_keeps_the_default_block(self):
        from llm_rosetta_gateway.config.parse import parse_model_route

        route = parse_model_route("m", "a", self.PROVS)

        assert route is not None
        assert route.routing.strategy == "weighted"

    def test_a_bad_block_is_refused_by_name(self):
        from llm_rosetta_gateway.config.parse import parse_model_route

        with pytest.raises(ValueError, match="models.m.routing"):
            parse_model_route(
                "m", {"provider": "a", "routing": {"nope": 1}}, self.PROVS
            )


def _route(strategy: str, *, weights=(1, 1), routing: ModelRouting | None = None):
    entries = [ProviderEntry(f"p{i}", weight=float(w)) for i, w in enumerate(weights)]
    return ModelRoute(
        entries,
        f"r-{strategy}",
        create_strategy(strategy),
        routing=routing or ModelRouting(strategy=strategy),
    )


def _config(model_entry):
    return GatewayConfig(
        {
            "providers": {
                "a": {
                    "type": "chat_completions",
                    "api_key": "sk-a",
                    "base_url": "https://a.test",
                },
                "b": {
                    "type": "chat_completions",
                    "api_key": "sk-b",
                    "base_url": "https://b.test",
                },
            },
            "models": {"m": model_entry},
        }
    )


class TestTheParse:
    def test_no_block_is_todays_behaviour(self):
        routing = parse_model_routing(None, label="models.m.routing")

        assert routing.strategy == "weighted"
        assert routing.affinity == "off"
        assert routing.sink is True
        assert routing.first_token_ms is None
        assert routing.rules == ()

    def test_an_empty_block_is_the_same(self):
        assert parse_model_routing({}, label="x") == ModelRouting()

    def test_an_unknown_key_is_refused(self):
        with pytest.raises(ValueError, match="unknown key"):
            parse_model_routing({"stratgey": "weighted"}, label="models.m.routing")

    def test_the_section_must_be_an_object(self):
        with pytest.raises(ValueError, match="must be an object"):
            parse_model_routing([], label="models.m.routing")

    @pytest.mark.parametrize(
        "strategy", ["weighted", "ordered", "rotate", "least_used", "pace"]
    )
    def test_every_documented_strategy_parses(self, strategy):
        assert (
            parse_model_routing({"strategy": strategy}, label="x").strategy == strategy
        )

    def test_an_unknown_strategy_is_refused(self):
        with pytest.raises(ValueError, match="strategy must be one of"):
            parse_model_routing({"strategy": "fastest"}, label="models.m.routing")

    @pytest.mark.parametrize("affinity", ["turn", "strict"])
    def test_an_unimplemented_affinity_is_refused_by_name(self, affinity):
        """Accepting it and ignoring it would silently change nothing."""
        with pytest.raises(
            ValueError, match=f"affinity '{affinity}' is not implemented yet"
        ):
            parse_model_routing({"affinity": affinity}, label="models.m.routing")

    def test_session_affinity_requires_the_weighted_strategy(self):
        with pytest.raises(ValueError, match="cannot be combined"):
            parse_model_routing(
                {"strategy": "rotate", "affinity": "session"}, label="models.m.routing"
            )

    def test_rules_parse_into_the_block(self):
        routing = parse_model_routing(
            {"rules": [{"when": {"images": True}, "use": "b"}]},
            label="models.m.routing",
        )

        assert len(routing.rules) == 1
        assert routing.rules[0].use == "b"
        assert routing.rules[0].when == {"images": True}

    def test_sink_must_be_a_boolean(self):
        with pytest.raises(ValueError, match="sink must be true or false"):
            parse_model_routing({"sink": "yes"}, label="models.m.routing")

    def test_first_token_ms_is_kept_for_later(self):
        """Parsed now, acted on when the timeout work lands."""
        assert (
            parse_model_routing({"first_token_ms": 20000}, label="x").first_token_ms
            == 20000
        )

    def test_first_token_ms_must_be_positive(self):
        with pytest.raises(ValueError, match="greater than 0"):
            parse_model_routing({"first_token_ms": 0}, label="models.m.routing")


class TestTheLegacyKey:
    def test_weighted_round_robin_maps_to_weighted(self):
        assert routing_from_legacy("weighted_round_robin", label="x") == ModelRouting()

    def test_affinity_round_robin_maps_to_session_affinity(self):
        routing = routing_from_legacy("affinity_round_robin", label="x")

        assert routing.strategy == "weighted"
        assert routing.affinity == "session"
        assert isinstance(strategy_for(routing), AffinityRoundRobinStrategy)

    def test_an_unknown_legacy_name_is_refused(self):
        with pytest.raises(ValueError, match="must be one of"):
            routing_from_legacy("magic", label="models.m.strategy")

    def test_the_block_and_the_legacy_key_cannot_be_combined(self):
        with pytest.raises(ValueError, match="sets both 'strategy' and 'routing'"):
            _config({"providers": ["a"], "strategy": "ordered", "routing": {}})

    def test_a_legacy_model_keeps_its_strategy(self):
        config = _config({"providers": ["a", "b"], "strategy": "affinity_round_robin"})

        assert isinstance(config.models["m"].strategy, AffinityRoundRobinStrategy)
        assert config.models["m"].routing.affinity == "session"


class TestTheStrategies:
    def test_ordered_is_the_first_candidate_every_time(self):
        route = _route("ordered", weights=(5, 1))

        assert [route.select() for _ in range(5)] == ["p0"] * 5

    def test_rotate_ignores_the_weights(self):
        route = _route("rotate", weights=(5, 1))

        picks = [route.select() for _ in range(6)]

        assert picks.count("p0") == 3
        assert picks.count("p1") == 3

    def test_least_used_lets_the_fresh_provider_catch_up(self):
        route = _route("least_used", weights=(5, 1))

        picks = [route.select() for _ in range(6)]

        assert picks.count("p0") == 3
        assert picks.count("p1") == 3

    def test_pace_spends_the_provider_with_room(self):
        """Configured weight is exactly what ``pace`` ignores."""
        route = _route("pace", weights=(5, 1))

        picks = {"p0": 0, "p1": 0}
        headroom = {"p0": 0.1, "p1": 1.0}
        for _ in range(40):
            entry = select_ready_entry(
                route,
                eligibility={
                    name: Eligibility(name=name, eligible=True, headroom=share)
                    for name, share in headroom.items()
                },
            )
            assert entry is not None
            picks[entry.name] += 1

        assert picks["p1"] > picks["p0"] * 2

    def test_weighted_would_have_preferred_the_heavy_one(self):
        """The control for the pace test: weight still wins under ``weighted``."""
        route = _route("weighted", weights=(5, 1))

        picks = {"p0": 0, "p1": 0}
        for _ in range(40):
            entry = select_ready_entry(
                route,
                eligibility={
                    "p0": Eligibility(name="p0", eligible=True),
                    "p1": Eligibility(name="p1", eligible=True),
                },
            )
            assert entry is not None
            picks[entry.name] += 1

        assert picks["p0"] > picks["p1"] * 2


class TestTheSwitches:
    def test_sink_true_orders_a_sunk_provider_last(self):
        route = _route("weighted", routing=ModelRouting(sink=True))
        eligibility = {
            "p0": Eligibility(name="p0", eligible=True, sunk=True),
            "p1": Eligibility(name="p1", eligible=True),
        }

        picks = set()
        for _ in range(5):
            entry = select_ready_entry(route, eligibility=eligibility)
            assert entry is not None
            picks.add(entry.name)

        assert picks == {"p1"}

    def test_sink_false_keeps_the_strategy_in_charge(self):
        """A model that opted out of sinking still splits the traffic."""
        route = _route("weighted", routing=ModelRouting(sink=False))
        eligibility = {
            "p0": Eligibility(name="p0", eligible=True, sunk=True),
            "p1": Eligibility(name="p1", eligible=True),
        }

        picks = []
        for _ in range(4):
            entry = select_ready_entry(route, eligibility=eligibility)
            assert entry is not None
            picks.append(entry.name)

        assert picks.count("p0") == 2
        assert picks.count("p1") == 2

    def test_affinity_session_builds_the_affinity_strategy(self):
        assert isinstance(
            strategy_for(ModelRouting(affinity="session")), AffinityRoundRobinStrategy
        )

    def test_the_strategy_registry_keeps_the_old_names(self):
        assert type(create_strategy("weighted_round_robin")).__name__ == (
            "WeightedRoundRobinStrategy"
        )
        assert isinstance(
            create_strategy("affinity_round_robin"), AffinityRoundRobinStrategy
        )

    def test_each_new_strategy_has_a_class(self):
        assert isinstance(create_strategy("ordered"), OrderedStrategy)
        assert isinstance(create_strategy("rotate"), RotateStrategy)
        assert isinstance(create_strategy("pace"), PaceStrategy)
