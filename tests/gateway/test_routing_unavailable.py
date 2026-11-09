"""Why a route is down: four states, one answer, one place.

``resolve`` used to work this out inline: a quota block, a spent allowance with
or without a reset, a failed startup and a plain "still initializing".  The
order is the rule — a client that just received a plan refusal must not be told
the gateway is booting — so it is pinned here rather than through the exception
it ends up in.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from llm_rosetta_gateway.maintenance.startup import ProviderInitState
from llm_rosetta_gateway.quota.state import QUOTA_STATE
from llm_rosetta_gateway.routing import SelectionContext
from llm_rosetta_gateway.routing.strategy import ModelRoute, ProviderEntry
from llm_rosetta_gateway.routing.unavailable import (
    Unavailable,
    blocked_until,
    classify,
)

NOW = datetime(2026, 11, 1, 12, 0, tzinfo=timezone.utc)


def _ctx(names: list[str], excluded: frozenset[str] = frozenset()) -> SelectionContext:
    ctx = SelectionContext("m", "chat_completions")
    ctx.model_route = ModelRoute(
        providers=[ProviderEntry(name=n) for n in names],
        route_id="r-unavailable",
    )
    ctx.quota_excluded = excluded
    return ctx


def _config(providers: dict[str, object], deferred: object | None = None):
    return SimpleNamespace(providers=providers, _deferred_startup=deferred)


def _unready(name: str) -> SimpleNamespace:
    return SimpleNamespace(name=name, ready=False)


def _blocked(name: str, identity: str, seconds: int) -> SimpleNamespace:
    QUOTA_STATE.block_provider(identity, seconds, "usage limit")
    pinfo = SimpleNamespace(
        name=name,
        ready=False,
        quota_identity=identity,
        quota_provider_blocked=lambda: QUOTA_STATE.provider_block(identity) is not None,
    )
    return pinfo


class TestTheLaneExclusion:
    """A lane this request was refused by is out of the story, as a provider is.

    ``classify`` answers about the route the request can still use.  After a
    lane-level refusal that route is every candidate except that lane, so a
    full-route answer taken over the unfiltered list would name a candidate
    the retry may not touch.
    """

    def test_a_refused_lane_is_not_counted_as_available(self):
        refuser = SimpleNamespace(name="a", ready=True)
        blocked = _blocked("b", "@unavail-lane-block", 30)
        ctx = SelectionContext("m", "chat_completions")
        ctx.model_route = ModelRoute(
            providers=[ProviderEntry("a", upstream_model="fast"), ProviderEntry("b")],
            route_id="r-unavailable-lane",
        )
        ctx.lane_excluded = frozenset({("a", "fast")})

        state = classify(_config({"a": refuser, "b": blocked}), ctx)

        # Only ``b`` is left in the retry's route, and it is blocked: that is
        # the client's story, not "initializing" over a route that still
        # counted the lane that just refused.
        assert state.reason == "blocked"
        assert state.as_kwargs()["blocked_until"] is not None

    def test_a_sibling_of_the_refused_lane_keeps_the_story_partial(self):
        """The exclusion is one lane's, not the provider's.

        ``a/eco`` is still usable, so a route whose only *unready* provider is
        ``b`` is not a whole-route block.  Filtering by provider name instead
        of by lane would drop ``a`` from the route, leaving ``b`` alone and
        turning the same state into "blocked".
        """
        refuser = SimpleNamespace(name="a", ready=True)
        blocked = _blocked("b", "@unavail-lane-sibling", 30)
        ctx = SelectionContext("m", "chat_completions")
        ctx.model_route = ModelRoute(
            providers=[
                ProviderEntry("a", upstream_model="fast"),
                ProviderEntry("a", upstream_model="eco"),
                ProviderEntry("b"),
            ],
            route_id="r-unavailable-lane-2",
        )
        ctx.lane_excluded = frozenset({("a", "fast")})

        state = classify(_config({"a": refuser, "b": blocked}), ctx)

        assert state.reason == "initializing"


class TestTheBlock:
    def test_every_provider_blocked_is_a_block_with_the_latest_end(self):
        a = _blocked("a", "@unavail-block-a", 30)
        b = _blocked("b", "@unavail-block-b", 90)
        ctx = _ctx(["a", "b"])

        state = classify(_config({"a": a, "b": b}), ctx)

        assert state.reason == "blocked"
        assert state.until is not None
        first = QUOTA_STATE.provider_block("@unavail-block-a")
        second = QUOTA_STATE.provider_block("@unavail-block-b")
        assert first is not None and second is not None
        assert state.until == max(first.until, second.until)

    def test_one_provider_unblocked_is_not_a_block(self):
        """The route's story is not (only) a quota story, so it falls through."""
        blocked = _blocked("a", "@unavail-block-mixed", 30)
        other = SimpleNamespace(
            name="b", ready=False, quota_provider_blocked=lambda: False
        )

        assert blocked_until([blocked, other]) is None

    def test_a_block_that_cannot_be_dated_is_unknown(self):
        pinfo = SimpleNamespace(
            name="a", ready=False, quota_provider_blocked=lambda: True
        )
        assert blocked_until([pinfo]) is None

    def test_a_blocked_provider_that_is_also_spent_reports_the_block(self):
        """Both states can hold at once, and the order decides the answer.

        A blocked provider whose readings are also spent satisfies the
        ``exhausted`` branch as well; the client is told the block, because it
        has a known end.  Swapping the two checks in ``classify`` leaves the
        rest of the suite green, so this is what makes the order load-bearing.
        """
        pinfo = _blocked("a", "@unavail-block-both", 30)
        ctx = _ctx(["a"], excluded=frozenset({"a"}))

        state = classify(_config({"a": pinfo}), ctx)

        assert state.reason == "blocked"
        assert state.as_kwargs()["blocked_until"] is not None

    def test_a_retry_reports_the_block_it_lands_on(self):
        """A failed-over request must not be told the gateway is booting.

        The provider that refused is in ``failover_excluded``, so the route
        the retry can still use is the rest of it.  If the only one left is
        quota-blocked, that block is the client's story -- asking the question
        over the whole route would count the refuser as an available provider
        and answer "initializing".
        """
        refuser = SimpleNamespace(name="a", ready=True)
        blocked = _blocked("b", "@unavail-block-after-failover", 30)
        ctx = _ctx(["a", "b"])
        ctx.failover_excluded = frozenset({"a"})

        state = classify(_config({"a": refuser, "b": blocked}), ctx)

        assert state.reason == "blocked"
        assert state.as_kwargs()["blocked_until"] is not None

    def test_a_retry_with_another_candidate_left_is_not_a_full_block(self):
        refuser = SimpleNamespace(name="a", ready=True)
        blocked = _blocked("b", "@unavail-block-partial-retry", 30)
        spare = SimpleNamespace(name="c", ready=False)
        ctx = _ctx(["a", "b", "c"])
        ctx.failover_excluded = frozenset({"a"})

        state = classify(_config({"a": refuser, "b": blocked, "c": spare}), ctx)

        assert state.reason == "initializing"


class TestTheSpentAllowance:
    def test_a_known_reset_is_reported(self, monkeypatch):
        from llm_rosetta_gateway.quota import service

        monkeypatch.setattr(
            service,
            "provider_exhausted_until",
            lambda pinfo: NOW + timedelta(minutes=5),
        )
        pinfo = SimpleNamespace(name="a", ready=True)
        ctx = _ctx(["a"], excluded=frozenset({"a"}))

        state = classify(_config({"a": pinfo}), ctx)

        assert state.reason == "exhausted"
        assert state.until == NOW + timedelta(minutes=5)
        assert state.as_kwargs() == {"exhausted_until": state.until, "exhausted": False}

    def test_no_named_reset_is_still_exhausted(self, monkeypatch):
        """The gateway refuses to invent a reset it was not told."""
        from llm_rosetta_gateway.quota import service

        monkeypatch.setattr(service, "provider_exhausted_until", lambda pinfo: None)
        pinfo = SimpleNamespace(name="a", ready=True)
        ctx = _ctx(["a"], excluded=frozenset({"a"}))

        state = classify(_config({"a": pinfo}), ctx)

        assert state.reason == "exhausted"
        assert state.until is None
        assert state.as_kwargs() == {"exhausted_until": None, "exhausted": True}

    def test_one_candidate_left_out_is_not_exhausted(self, monkeypatch):
        from llm_rosetta_gateway.quota import service

        monkeypatch.setattr(service, "provider_exhausted_until", lambda pinfo: NOW)
        pinfo = SimpleNamespace(name="a", ready=True)
        ctx = _ctx(["a", "b"], excluded=frozenset({"a"}))

        state = classify(_config({"a": pinfo, "b": pinfo}), ctx)

        assert state.reason == "initializing"

    def test_the_reset_is_read_from_the_route_that_is_left(self, monkeypatch):
        """An excluded provider's unknown reset must not make the answer unknown.

        Both providers are spent; the one the request failed over from never
        named a reset, the one it can still use did.  The reset lookup follows
        the same reduced route as the two "every provider" questions, so the
        client gets the date instead of "no reset time is known".
        """
        from llm_rosetta_gateway.quota import service

        resets = {"a": None, "b": NOW + timedelta(minutes=7)}
        monkeypatch.setattr(
            service, "provider_exhausted_until", lambda pinfo: resets[pinfo.name]
        )
        spent = SimpleNamespace(name="a", ready=True)
        spent_b = SimpleNamespace(name="b", ready=True)
        ctx = _ctx(["a", "b"], excluded=frozenset({"a", "b"}))
        ctx.failover_excluded = frozenset({"a"})

        state = classify(_config({"a": spent, "b": spent_b}), ctx)

        assert state.reason == "exhausted"
        assert state.until == NOW + timedelta(minutes=7)

    def test_a_route_with_no_candidates_is_not_exhausted(self):
        ctx = _ctx([])

        state = classify(_config({}), ctx)

        assert state.reason == "initializing"


class TestTheStartupStates:
    def test_every_unready_provider_failed(self):
        deferred = SimpleNamespace(provider_state=lambda name: ProviderInitState.FAILED)
        ctx = _ctx(["a", "b"])

        state = classify(
            _config({"a": _unready("a"), "b": _unready("b")}, deferred), ctx
        )

        assert state.reason == "failed"
        assert state.as_kwargs() == {"failed": True}

    def test_one_provider_still_initializing(self):
        states = {"a": ProviderInitState.FAILED, "b": ProviderInitState.READY}
        deferred = SimpleNamespace(provider_state=lambda name: states[name])
        ctx = _ctx(["a", "b"])

        state = classify(
            _config({"a": _unready("a"), "b": _unready("b")}, deferred), ctx
        )

        assert state.reason == "initializing"
        assert state.as_kwargs() == {"failed": False}

    def test_no_deferred_startup_is_initializing(self):
        ctx = _ctx(["a"])

        assert classify(_config({"a": _unready("a")}), ctx).reason == "initializing"


class TestTheRateLimitLadder:
    """A ladder that took every candidate out is not "still starting".

    ``ready`` folds a 429 disable and a capacity disable into "not ready", so
    the block check used to answer "quota block" (a plan refusal the client
    never received) or, when the block could not be dated, "initializing".
    The ladder's own answers are what distinguish them.
    """

    @staticmethod
    def _limited(name, until):
        from llm_rosetta_gateway.routing.eligibility import Eligibility, RATE_LIMITED

        return Eligibility(name=name, eligible=False, reason=RATE_LIMITED, until=until)

    @staticmethod
    def _ready(name):
        return SimpleNamespace(name=name, ready=True)

    def test_every_candidate_limited_is_rate_limited_with_the_latest_end(self):
        ctx = _ctx(["a", "b"])
        later = NOW + timedelta(seconds=90)
        ctx.with_eligibility(
            {
                ("a", None): self._limited("a", NOW + timedelta(seconds=30)),
                ("b", None): self._limited("b", later),
            }
        )

        state = classify(_config({"a": self._ready("a"), "b": self._ready("b")}), ctx)

        assert state.reason == "rate_limited"
        assert state.until == later
        assert state.as_kwargs() == {"rate_limited_until": later, "rate_limited": False}

    def test_one_candidate_still_initializing_keeps_initializing(self):
        from llm_rosetta_gateway.routing.eligibility import Eligibility, INITIALIZING

        ctx = _ctx(["a", "b"])
        ctx.with_eligibility(
            {
                ("a", None): self._limited("a", NOW + timedelta(seconds=30)),
                ("b", None): Eligibility(name="b", eligible=False, reason=INITIALIZING),
            }
        )

        state = classify(_config({"a": self._ready("a"), "b": _unready("b")}), ctx)

        assert state.reason == "initializing"
        assert state.as_kwargs() == {"failed": False}

    def test_an_undated_disable_is_reported_without_a_deadline(self):
        ctx = _ctx(["a"])
        ctx.with_eligibility({("a", None): self._limited("a", None)})

        state = classify(_config({"a": self._ready("a")}), ctx)

        assert state.reason == "rate_limited"
        assert state.until is None
        assert state.as_kwargs() == {"rate_limited_until": None, "rate_limited": True}

    def test_a_quota_block_beside_a_disable_stays_a_block(self):
        from llm_rosetta_gateway.routing.eligibility import Eligibility, BLOCKED

        a = _blocked("a", "@unavail-mixed-block", 30)
        b = _blocked("b", "@unavail-mixed-block-b", 60)
        ctx = _ctx(["a", "b"])
        # ``b`` is blocked at the provider level, so its eligibility is not the
        # ladder's: the route is a block, not a rate limit.
        ctx.with_eligibility(
            {
                ("a", None): self._limited("a", NOW + timedelta(seconds=30)),
                ("b", None): Eligibility(name="b", eligible=False, reason=BLOCKED),
            }
        )

        state = classify(_config({"a": a, "b": b}), ctx)

        assert state.reason == "blocked"


class TestTheAnswerShape:
    @pytest.mark.parametrize(
        "state, expected",
        [
            (Unavailable("blocked", NOW), {"blocked_until": NOW}),
            (
                Unavailable("rate_limited", NOW),
                {"rate_limited_until": NOW, "rate_limited": False},
            ),
            (
                Unavailable("rate_limited"),
                {"rate_limited_until": None, "rate_limited": True},
            ),
            (
                Unavailable("exhausted", NOW),
                {"exhausted_until": NOW, "exhausted": False},
            ),
            (Unavailable("exhausted"), {"exhausted_until": None, "exhausted": True}),
            (Unavailable("failed"), {"failed": True}),
            (Unavailable("initializing"), {"failed": False}),
        ],
    )
    def test_as_kwargs_speaks_provider_not_ready(self, state, expected):
        assert state.as_kwargs() == expected
