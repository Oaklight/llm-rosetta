"""Two lanes of one provider are two candidates (#1430).

A model may route one provider to several upstream models — a cheap lane and a
fast one, say.  Selection used to key every intermediate map by the provider
name, so the second lane overwrote the first: one lane won every time and the
other could not be told apart in eligibility, exclusion or logging.  Candidate
identity is now ``(name, upstream_model)``.
"""

from __future__ import annotations

import pytest

from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.config.parse import parse_multi_provider
from llm_rosetta_gateway.routing import (
    AffinityRoundRobinStrategy,
    Eligibility,
    ModelRoute,
    OrderedStrategy,
    ProviderEntry,
    required_capabilities,
    select_ready_entry,
)
from llm_rosetta_gateway.routing.context import SelectionContext
from llm_rosetta_gateway.routing.eligibility import CAPABILITY, EligibilityStage
from llm_rosetta_gateway.routing.rules import RuleFacts
from llm_rosetta_gateway.transport.provider_info import ProviderInfo

ROUTE = "lanes"


def _lanes(*upstreams: str | None) -> list[ProviderEntry]:
    return [ProviderEntry("p", upstream_model=up) for up in upstreams]


def _names(entries) -> list[str]:
    return [entry.upstream_model or "" for entry in entries]


def _pinfo(name: str) -> ProviderInfo:
    return ProviderInfo(
        name,
        api_key="k1",
        base_url="https://example.test",
        auth_header_fn=lambda key: {"Authorization": f"Bearer {key}"},
        url_template="{base_url}/x",
    )


def _selection_context(
    entries: list[ProviderEntry], required: frozenset[str] = frozenset()
) -> SelectionContext:
    ctx = SelectionContext("m", "chat_completions")
    ctx.install_route(
        model_route=ModelRoute(providers=entries, route_id="m"),
        providers={entry.name: _pinfo(entry.name) for entry in entries},
        provider_types={},
        provider_shim_names={},
        provider_account_tokens={},
    )
    ctx.with_required_capabilities(required)
    return ctx


class TestTheKey:
    def test_two_lanes_of_one_provider_have_different_keys(self):
        fast, eco = _lanes("fast", "eco")

        assert fast.key == ("p", "fast")
        assert eco.key == ("p", "eco")
        assert fast.key != eco.key

    def test_an_absent_upstream_and_an_empty_one_are_one_key(self):
        assert ProviderEntry("p").key == ProviderEntry("p", upstream_model="").key

    def test_an_empty_upstream_is_stored_as_absent(self):
        assert ProviderEntry("p", upstream_model="").upstream_model is None


class TestEachStrategyCanPickBoth:
    """Every strategy must be able to reach either lane, not just the first."""

    def test_weighted_alternates(self):
        route = ModelRoute(_lanes("fast", "eco"), ROUTE)
        chosen = [route.select() for _ in range(6)]

        assert _names(chosen).count("fast") == 3
        assert _names(chosen).count("eco") == 3

    def test_rotate_takes_turns(self):
        route = ModelRoute(_lanes("fast", "eco"), ROUTE)
        route.strategy = __import__(
            "llm_rosetta_gateway.routing", fromlist=["RotateStrategy"]
        ).RotateStrategy()

        chosen = [route.select() for _ in range(4)]

        assert _names(chosen) == ["fast", "eco", "fast", "eco"]

    def test_least_used_serves_the_other_lane_next(self):
        from llm_rosetta_gateway.routing import LeastUsedStrategy

        route = ModelRoute(_lanes("fast", "eco"), ROUTE, strategy=LeastUsedStrategy())
        chosen = [route.select() for _ in range(4)]

        assert _names(chosen).count("fast") == 2
        assert _names(chosen).count("eco") == 2

    def test_ordered_keeps_the_operators_order(self):
        route = ModelRoute(_lanes("fast", "eco"), ROUTE, strategy=OrderedStrategy())
        chosen = [route.select() for _ in range(3)]

        assert _names(chosen) == ["fast", "fast", "fast"]

    def test_affinity_reaches_both_lanes(self):
        """The hash covers the lane, so identities spread over both."""
        strategy = AffinityRoundRobinStrategy()
        providers = _lanes("fast", "eco")

        picked = {
            strategy.select(providers, route_id=ROUTE, identity=f"client-{i}").key
            for i in range(64)
        }

        assert picked == {("p", "fast"), ("p", "eco")}

    def test_affinity_keeps_one_identity_on_one_lane(self):
        strategy = AffinityRoundRobinStrategy()
        providers = _lanes("fast", "eco")

        first = strategy.select(providers, route_id=ROUTE, identity="user-42")
        again = [
            strategy.select(providers, route_id=ROUTE, identity="user-42")
            for _ in range(5)
        ]

        assert all(entry.key == first.key for entry in again)


class TestEligibilityIsPerLane:
    def test_a_capability_filter_names_the_lane_it_drops(self):
        ctx = _selection_context(
            [
                ProviderEntry("p", upstream_model="fast", capabilities=("text",)),
                ProviderEntry(
                    "p", upstream_model="eco", capabilities=("text", "vision")
                ),
            ],
            required=frozenset({"vision"}),
        )

        EligibilityStage().run(ctx)

        assert set(ctx.eligibility) == {("p", "fast"), ("p", "eco")}
        assert ctx.eligibility[("p", "fast")].eligible is False
        assert ctx.eligibility[("p", "fast")].reason == CAPABILITY
        assert ctx.eligibility[("p", "eco")].eligible is True

    def test_selection_skips_the_ineligible_lane_only(self):
        route = ModelRoute(_lanes("fast", "eco"), ROUTE)
        eligibility: dict[tuple[str, str | None], Eligibility] = {
            ("p", "fast"): Eligibility(name="p", upstream="fast", eligible=False),
            ("p", "eco"): Eligibility(name="p", upstream="eco", eligible=True),
        }

        picked = {
            entry.key
            for entry in (
                select_ready_entry(route, eligibility=eligibility) for _ in range(6)
            )
            if entry is not None
        }

        assert picked == {("p", "eco")}


class TestCredentialLevelExclusionsCoverEveryLane:
    """Spent allowance and failover refusal are per credential, not per lane."""

    @staticmethod
    def _ctx():
        entries = [
            ProviderEntry("a", upstream_model="fast"),
            ProviderEntry("a", upstream_model="eco"),
            ProviderEntry("b"),
        ]
        ctx = _selection_context(entries)
        EligibilityStage().run(ctx)
        return ctx, entries

    def test_a_spent_provider_takes_every_lane_out(self):
        ctx, entries = self._ctx()
        ctx.with_quota_headroom({e.key: 1.0 for e in entries}, frozenset({"a"}))

        assert ctx.excluded_providers() == {("a", "fast"), ("a", "eco")}
        picked = {
            entry.key
            for entry in (
                select_ready_entry(
                    ctx.model_route,
                    eligibility=ctx.eligibility,
                    excluded=ctx.excluded_providers(),
                )
                for _ in range(6)
            )
            if entry is not None
        }
        assert picked == {("b", None)}

    def test_a_failover_exclusion_covers_every_lane_too(self):
        ctx, _entries = self._ctx()
        ctx.with_failover_excluded({"a"})

        assert ctx.excluded_providers() == {("a", "fast"), ("a", "eco")}


class TestRulesSpeakProviderWide:
    @staticmethod
    def _config(rules, lanes):
        return GatewayConfig(
            {
                "providers": {
                    "p": {
                        "type": "chat_completions",
                        "api_key": "sk-p",
                        "base_url": "https://p.test",
                    },
                    "q": {
                        "type": "chat_completions",
                        "api_key": "sk-q",
                        "base_url": "https://q.test",
                    },
                },
                "models": {"m": {"providers": lanes, "routing": {"rules": rules}}},
            }
        )

    @staticmethod
    def _resolve(config, facts: RuleFacts):
        selection = config.selection_context("m", "chat_completions")
        selection.with_rule_facts(facts)
        # The edge derives the request's needs from the same facts; this is
        # what routes_llm does before resolve().
        selection.with_required_capabilities(required_capabilities(facts))
        config.resolve(selection)
        assert selection.chosen is not None
        return selection.chosen

    def test_a_use_rule_promotes_the_first_eligible_lane(self):
        config = self._config(
            [{"when": {}, "use": "p"}],
            [
                {"name": "p", "upstream_model": "fast"},
                {"name": "p", "upstream_model": "eco"},
                {"name": "q"},
            ],
        )

        assert self._resolve(config, RuleFacts()).upstream_model == "fast"

    def test_a_use_rule_falls_to_the_second_lane_when_the_first_cannot_serve(self):
        config = self._config(
            [{"when": {"images": True}, "use": "p"}],
            [
                {"name": "p", "upstream_model": "fast", "capabilities": ["text"]},
                {
                    "name": "p",
                    "upstream_model": "eco",
                    "capabilities": ["text", "vision"],
                },
                {"name": "q"},
            ],
        )

        chosen = self._resolve(config, RuleFacts(images=True))

        assert (chosen.name, chosen.upstream_model) == ("p", "eco")

    def test_a_pause_rule_cannot_empty_the_route(self):
        """Pausing the only candidate is a no-op: the anchor stands.

        A rule is a preference applied on top of the anchor, not a veto over
        the route, so there is always a provider to serve the request.
        """
        config = self._config([{"when": {}, "pause": "p"}], [{"name": "p"}])

        assert self._resolve(config, RuleFacts()).name == "p"

    def test_a_pause_rule_drops_every_lane_of_the_named_provider(self):
        config = self._config(
            [{"when": {}, "pause": "p"}],
            [
                {"name": "p", "upstream_model": "fast"},
                {"name": "p", "upstream_model": "eco"},
                {"name": "q"},
            ],
        )

        assert self._resolve(config, RuleFacts()).name == "q"


class TestTheLoaderRefusesADuplicateLane:
    def _parse(self, providers):
        return parse_multi_provider("m", {"providers": providers}, {"p": {}})

    def test_the_same_lane_twice_is_a_config_error(self):
        with pytest.raises(ValueError, match="more than once"):
            self._parse(
                [
                    {"name": "p", "upstream_model": "fast"},
                    {"name": "p", "upstream_model": "fast"},
                ]
            )

    def test_the_same_provider_with_different_upstreams_loads(self):
        route = self._parse(
            [
                {"name": "p", "upstream_model": "fast"},
                {"name": "p", "upstream_model": "eco"},
            ]
        )

        assert route is not None
        assert [entry.key for entry in route.providers] == [
            ("p", "fast"),
            ("p", "eco"),
        ]

    def test_a_bare_duplicate_name_is_a_config_error(self):
        with pytest.raises(ValueError, match="more than once"):
            self._parse(["p", "p"])

    def test_a_typed_model_refuses_a_duplicate_lane_too(self):
        """Typed pools parse their entries with the same rule and label."""
        from llm_rosetta_gateway.config.parse import declared_provider_entries

        with pytest.raises(ValueError, match="model 'embeds' lists provider 'p'"):
            declared_provider_entries(
                {"providers": [{"name": "p"}, {"name": "p"}]},
                label="model 'embeds'",
            )
