"""Re-selection after a refusal: exclude the refuser, ask the chain again.

The proxy decides *whether* a model may fail over; this decides *where* it
goes.  The two answers the caller depends on are pinned here: the refused
provider is out of the running for the rest of the request, and a route with
nothing left reports "no target" rather than raising out of the failure path.
"""

from __future__ import annotations

from llm_rosetta_gateway.maintenance.startup import ProviderNotReady
from llm_rosetta_gateway.routing import SelectionContext
from llm_rosetta_gateway.routing.failover import (
    reselect_after_credential_refusal,
    reselect_after_lane_refusal,
)
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

        assert reselect_after_credential_refusal(ctx, exclude="a", resolve=resolve) == (
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

        reselect_after_credential_refusal(ctx, exclude="a", resolve=resolve)

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

        reselect_after_credential_refusal(ctx, exclude="a", resolve=resolve)

        assert seen == [frozenset({"a", "c"})]


class TestWhenNothingIsLeft:
    def test_a_route_with_nothing_ready_is_not_an_error(self):
        ctx = _ctx(["a"])

        def resolve(selection: SelectionContext):
            raise ProviderNotReady("m")

        assert (
            reselect_after_credential_refusal(ctx, exclude="a", resolve=resolve) is None
        )

    def test_an_unknown_model_is_not_an_error_either(self):
        ctx = _ctx(["a"])

        def resolve(selection: SelectionContext):
            raise KeyError("m")

        assert (
            reselect_after_credential_refusal(ctx, exclude="a", resolve=resolve) is None
        )

    def test_another_failure_still_reaches_the_caller(self):
        """Swallowing more than the two route-level refusals would hide a bug."""
        ctx = _ctx(["a"])

        def resolve(selection: SelectionContext):
            raise RuntimeError("boom")

        try:
            reselect_after_credential_refusal(ctx, exclude="a", resolve=resolve)
        except RuntimeError:
            pass
        else:  # pragma: no cover - the failure is the point
            raise AssertionError("a non-route failure must not be swallowed")


def _lane_ctx() -> SelectionContext:
    """One provider with two lanes, and a second provider."""
    ctx = SelectionContext("m", "chat_completions")
    ctx.model_route = ModelRoute(
        providers=[
            ProviderEntry("a", upstream_model="fast"),
            ProviderEntry("a", upstream_model="eco"),
            ProviderEntry("b"),
        ],
        route_id="r-lane",
    )
    ctx.chosen = ctx.model_route.providers[0]
    return ctx


class TestTheLaneRefusal:
    """A content-level refusal leaves the provider standing (#1430).

    Only the lane that answered is out; the provider's siblings are asked for
    first, and another provider is reached only when none of them is left.
    """

    def test_the_sibling_is_looked_for_before_anyone_else(self):
        ctx = _lane_ctx()
        seen: list[tuple[str | None, frozenset[tuple[str, str | None]]]] = []

        def resolve(selection: SelectionContext):
            seen.append((selection.pinned, selection.lane_excluded))
            if selection.pinned != "a":
                raise ProviderNotReady("m")
            return "sibling", "pinfo"

        assert reselect_after_lane_refusal(
            ctx, lane=("a", "fast"), resolve=resolve
        ) == ("sibling", "pinfo")
        # The sibling search runs with the provider pinned, after the lane is
        # already out — so the search cannot land on the lane that refused.
        assert seen == [("a", frozenset({("a", "fast")}))]
        # The provider keeps its credential: nothing was excluded by name.
        assert ctx.failover_excluded == frozenset()
        assert ctx.lane_excluded == frozenset({("a", "fast")})
        # The pin is the caller's, not ours: it is put back.
        assert ctx.pinned is None

    def test_no_sibling_left_widens_the_search(self):
        ctx = _lane_ctx()
        seen: list[str | None] = []

        def resolve(selection: SelectionContext):
            seen.append(selection.pinned)
            if selection.pinned == "a":
                raise ProviderNotReady("m")
            return "elsewhere", "pinfo"

        assert reselect_after_lane_refusal(
            ctx, lane=("a", "fast"), resolve=resolve
        ) == ("elsewhere", "pinfo")
        assert seen == ["a", None]

    def test_the_sibling_step_can_be_skipped_once_it_has_refused(self):
        ctx = _lane_ctx()
        seen: list[str | None] = []

        def resolve(selection: SelectionContext):
            seen.append(selection.pinned)
            return "elsewhere", "pinfo"

        reselect_after_lane_refusal(
            ctx, lane=("a", "eco"), resolve=resolve, try_siblings=False
        )

        assert seen == [None]

    def test_a_pinned_request_never_widens(self):
        """A pin is the operator's instruction; a refusal does not lift it."""
        ctx = _lane_ctx()
        ctx.with_pinned_provider("a")
        seen: list[str | None] = []

        def resolve(selection: SelectionContext):
            seen.append(selection.pinned)
            raise ProviderNotReady("m")

        assert (
            reselect_after_lane_refusal(ctx, lane=("a", "fast"), resolve=resolve)
            is None
        )
        # Only the in-pin search ran: no unpinned resolve, which would have
        # been free to answer from ``b``.
        assert seen == ["a"]
        assert ctx.pinned == "a"
