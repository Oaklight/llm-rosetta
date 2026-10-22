"""Quota-informed weighting: a nearly spent provider serves less, not nothing.

The quota layer knows how much allowance each credential has left.  Routing
used to ignore it until a provider was entirely blocked, so two "ready"
providers split the traffic evenly even when one was at 96%.
"""

from __future__ import annotations

from typing import Any

from llm_rosetta_gateway.quota.models import QuotaReading, QuotaWindow
from llm_rosetta_gateway.quota.state import QUOTA_STATE
from llm_rosetta_gateway.routing import (
    AnchorStage,
    QuotaStage,
    SelectionContext,
    create_strategy,
)
from llm_rosetta_gateway.routing.select import quota_weight, select_ready_entry
from llm_rosetta_gateway.routing.strategy import ModelRoute, ProviderEntry
from llm_rosetta_gateway.transport.provider_info import ProviderInfo


def _pinfo(name: str, *, keys: str = "k1", probe: str | None = None) -> Any:
    pinfo = ProviderInfo(
        name,
        api_key=keys,
        base_url="https://example.test",
        auth_header_fn=lambda key: {"Authorization": f"Bearer {key}"},
        url_template="{base_url}/x",
    )
    pinfo.quota_probe = probe
    return pinfo


def _reading(probe: str, key: str, used: float) -> QuotaReading:
    return QuotaReading(
        probe=probe,
        credential="fingerprint",
        windows=[QuotaWindow(name="5h", used_percent=used, span_seconds=18000)],
    )


class TestTheWeight:
    def test_an_unmeasured_provider_keeps_its_weight(self):
        assert quota_weight(None) == 1.0

    def test_an_empty_window_keeps_it_too(self):
        assert quota_weight(1.0) == 1.0

    def test_a_spent_window_still_serves_some(self):
        """Zero would starve a live provider; the state removes it, not this."""
        assert quota_weight(0.0) == 0.25

    def test_it_is_monotonic_in_between(self):
        assert quota_weight(0.75) > quota_weight(0.5) > quota_weight(0.25)


class TestTheStage:
    def _ctx(self, providers: dict[str, Any], names: list[str]) -> SelectionContext:
        ctx = SelectionContext("m", "chat_completions")
        ctx.providers = providers
        ctx.model_route = ModelRoute(
            providers=[ProviderEntry(name=n) for n in names],
            route_id="r",
            strategy=create_strategy("weighted_round_robin"),
        )
        return ctx

    def test_a_measured_provider_reports_its_smallest_share(self):
        probe = "p-test-stage"
        pinfo = _pinfo("a", keys="k1,k2", probe=probe)
        QUOTA_STATE.put_reading(_reading(probe, "k1", 10))
        # The reading is keyed by fingerprint, so put it where the lookup looks.
        QUOTA_STATE.put_reading(
            QuotaReading(
                probe=probe,
                credential=__import__(
                    "llm_rosetta_gateway.quota.models", fromlist=["x"]
                ).credential_fingerprint("k2"),
                windows=[QuotaWindow(name="5h", used_percent=90, span_seconds=18000)],
            )
        )
        ctx = self._ctx({"a": pinfo}, ["a"])

        QuotaStage().run(ctx)

        assert ctx.quota_headroom["a"] == 0.1

    def test_an_unmeasured_provider_is_one_not_zero(self):
        ctx = self._ctx({"a": _pinfo("a"), "b": _pinfo("b")}, ["a", "b"])

        QuotaStage().run(ctx)

        assert ctx.quota_headroom == {"a": 1.0, "b": 1.0}


class TestTheAnchor:
    def test_the_nearly_spent_provider_gets_materially_less(self):
        route = ModelRoute(
            providers=[ProviderEntry(name="fresh"), ProviderEntry(name="spent")],
            route_id="r-quota-split",
            strategy=create_strategy("weighted_round_robin"),
        )
        picks = {"fresh": 0, "spent": 0}
        for _ in range(40):
            entry = select_ready_entry(
                route,
                is_ready=lambda name: True,
                headroom={"fresh": 1.0, "spent": 0.0},
            )
            assert entry is not None
            picks[entry.name] += 1

        assert picks["fresh"] > picks["spent"] * 2
        assert picks["spent"] > 0  # it is still a candidate, not excluded

    def test_without_headroom_the_split_is_even(self):
        route = ModelRoute(
            providers=[ProviderEntry(name="a"), ProviderEntry(name="b")],
            route_id="r-quota-even",
            strategy=create_strategy("weighted_round_robin"),
        )
        picks = {"a": 0, "b": 0}
        for _ in range(20):
            entry = select_ready_entry(route, is_ready=lambda name: True)
            assert entry is not None
            picks[entry.name] += 1

        assert picks == {"a": 10, "b": 10}


class TestTheChainOrder:
    def test_the_anchor_refuses_to_run_before_the_quota_stage(self):
        """A reversed chain would weight nothing and say nothing."""
        from llm_rosetta_gateway.routing import RoutingChain
        from llm_rosetta_gateway.routing.stage import StageOrderError

        try:
            RoutingChain([AnchorStage(), QuotaStage()])
        except StageOrderError as exc:
            assert "quota_headroom" in str(exc)
        else:  # pragma: no cover - the failure this test exists for
            raise AssertionError("the reversed chain was accepted")

    def test_the_real_order_builds(self):
        from llm_rosetta_gateway.routing import RoutingChain

        RoutingChain([QuotaStage(), AnchorStage()])
