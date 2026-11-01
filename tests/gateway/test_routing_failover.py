"""Re-selection after a refusal: exclude the refuser, ask the chain again.

The proxy decides *whether* a model may fail over; this decides *where* it
goes.  The two answers the caller depends on are pinned here: the refused
provider is out of the running for the rest of the request, and a route with
nothing left reports "no target" rather than raising out of the failure path.
"""

from __future__ import annotations

from llm_rosetta_gateway.maintenance.startup import ProviderNotReady
from llm_rosetta_gateway.routing import SelectionContext
from llm_rosetta_gateway.routing.failover import reselect_after_refusal
from llm_rosetta_gateway.routing.strategy import ModelRoute, ProviderEntry


def _ctx(names: list[str]) -> SelectionContext:
    ctx = SelectionContext("m", "chat_completions")
    ctx.model_route = ModelRoute(
        providers=[ProviderEntry(name=n) for n in names], route_id="r-failover"
    )
    ctx.chosen = ctx.model_route.providers[0]
    return ctx


class TestItExcludesAndAsksAgain:
    def test_the_refused_provider_does_not_come_back(self):
        ctx = _ctx(["a", "b"])
        seen: list[frozenset[str]] = []

        def resolve(selection: SelectionContext):
            seen.append(selection.failover_excluded)
            return "route", "pinfo"

        assert reselect_after_refusal(ctx, exclude="a", resolve=resolve) == (
            "route",
            "pinfo",
        )
        assert seen == [frozenset({"a"})]

    def test_the_previous_choice_is_voided_first(self):
        ctx = _ctx(["a", "b"])
        chosen_on_entry: list[object] = []

        def resolve(selection: SelectionContext):
            chosen_on_entry.append(selection.chosen)
            return "route", "pinfo"

        reselect_after_refusal(ctx, exclude="a", resolve=resolve)

        # A resolver that finds nothing must not leave the refused provider
        # standing in the context the caller reads afterwards.
        assert chosen_on_entry == [None]

    def test_an_earlier_exclusion_is_kept(self):
        ctx = _ctx(["a", "b", "c"])
        ctx.failover_excluded = frozenset({"c"})
        seen: list[frozenset[str]] = []

        def resolve(selection: SelectionContext):
            seen.append(selection.failover_excluded)
            return "route", "pinfo"

        reselect_after_refusal(ctx, exclude="a", resolve=resolve)

        assert seen == [frozenset({"a", "c"})]


class TestWhenNothingIsLeft:
    def test_a_route_with_nothing_ready_is_not_an_error(self):
        ctx = _ctx(["a"])

        def resolve(selection: SelectionContext):
            raise ProviderNotReady("m")

        assert reselect_after_refusal(ctx, exclude="a", resolve=resolve) is None

    def test_an_unknown_model_is_not_an_error_either(self):
        ctx = _ctx(["a"])

        def resolve(selection: SelectionContext):
            raise KeyError("m")

        assert reselect_after_refusal(ctx, exclude="a", resolve=resolve) is None

    def test_another_failure_still_reaches_the_caller(self):
        """Swallowing more than the two route-level refusals would hide a bug."""
        ctx = _ctx(["a"])

        def resolve(selection: SelectionContext):
            raise RuntimeError("boom")

        try:
            reselect_after_refusal(ctx, exclude="a", resolve=resolve)
        except RuntimeError:
            pass
        else:  # pragma: no cover - the failure is the point
            raise AssertionError("a non-route failure must not be swallowed")
