"""Continuous 429s: the ladder, and the one predicate that reads it.

A single refusal is evidence about one request; a run of them is evidence about
the lane.  The rules under test:

- the first refusal **sinks** the provider — still a candidate, tried last;
- ``after`` refusals within the window **disable** it for ``cooldown_s``;
- each repeated disable doubles the cooldown, up to the cap;
- a success clears the count, the ladder and the sink.

The state is keyed per provider *and* credential, so the same rule covers a
probed lane, a keyless one, and one no probe can read — which is the point:
before this, only two of those three were classified at all.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.quota.models import (
    QuotaReading,
    QuotaWindow,
    credential_fingerprint,
)
from llm_rosetta_gateway.quota.rate_limit import RateLimitPolicy, RateLimitTracker
from llm_rosetta_gateway.quota.state import QUOTA_STATE
from llm_rosetta_gateway.routing import (
    BLOCKED,
    INITIALIZING,
    RATE_LIMITED,
    SPENT,
    EligibilityStage,
    provider_eligibility,
)
from llm_rosetta_gateway.routing.strategy import ModelRoute, ProviderEntry
from llm_rosetta_gateway.transport.provider_info import ProviderInfo

# The tracker judges a disable against the wall clock, so the timeline this
# file builds starts when the file is imported: a fixed date would age out
# of its own cooldown between two runs.
NOW = datetime.now(timezone.utc)
POLICY = RateLimitPolicy()


@pytest.fixture(autouse=True)
def _clean_quota_state():
    """The state is process-wide; no test may inherit another's refusals."""
    QUOTA_STATE.clear()
    yield
    QUOTA_STATE.clear()


@pytest.fixture(autouse=True)
def _pinned_clock(monkeypatch):
    """Every read judges against NOW, not the wall clock.

    ``record``, ``disabled``, ``latest_disable`` and ``sunk`` take a moment
    for callers that pass one; the eligibility stage does not pass one, so a
    fixed timeline would otherwise be true only while the file runs within the
    policy's own window of its import.
    """
    from llm_rosetta_gateway.quota import rate_limit as rate_limit_module

    monkeypatch.setattr(rate_limit_module, "_now", lambda: NOW)


def _pinfo(name: str = "a", *, keys: str = "k1", keyless: bool = False) -> ProviderInfo:
    return ProviderInfo(
        name,
        config_name=name,
        api_key=keys,
        base_url="https://example.test",
        auth_header_fn=lambda key: {"Authorization": f"Bearer {key}"},
        url_template="{base_url}/x",
        keyless=keyless,
    )


def _config(*names: str) -> GatewayConfig:
    return GatewayConfig(
        {
            "providers": {
                name: {
                    "type": "chat_completions",
                    "api_key": f"sk-{name}",
                    "base_url": f"https://{name}.test",
                }
                for name in names
            },
            "models": {"m": {"providers": list(names)}},
        }
    )


class TestTheLadder:
    def test_the_first_refusal_sinks(self):
        tracker = RateLimitTracker()

        outcome = tracker.note("a", None, POLICY, now=NOW)

        assert outcome == "sink"
        assert tracker.sunk("a") is True
        assert tracker.disabled("a") is None

    def test_after_refusals_disable_for_the_cooldown(self):
        tracker = RateLimitTracker()
        for _ in range(2):
            assert tracker.note("a", None, POLICY, now=NOW) == "sink"

        assert tracker.note("a", None, POLICY, now=NOW) == "disable"
        until = tracker.disabled("a", now=NOW)
        assert until == NOW + timedelta(seconds=60)

    def test_a_repeat_disable_doubles(self):
        tracker = RateLimitTracker()
        for _ in range(3):
            tracker.note("a", None, POLICY, now=NOW)
        assert tracker.disabled("a", now=NOW) == NOW + timedelta(seconds=60)

        later = NOW + timedelta(seconds=61)
        for _ in range(3):
            tracker.note("a", None, POLICY, now=later)

        assert tracker.disabled("a", now=later) == later + timedelta(seconds=120)

    def test_the_doubling_stops_at_the_cap(self):
        tracker = RateLimitTracker()
        policy = RateLimitPolicy(cooldown_s=60, cooldown_max_s=180)
        moment = NOW
        lengths: list[float] = []
        for _ in range(4):
            for _ in range(3):
                tracker.note("a", None, policy, now=moment)
            until = tracker.disabled("a", now=moment)
            assert until is not None
            lengths.append((until - moment).total_seconds())
            moment = until + timedelta(seconds=1)

        assert lengths == [60, 120, 180, 180]

    def test_a_window_that_ends_restarts_the_count(self):
        tracker = RateLimitTracker()
        tracker.note("a", None, POLICY, now=NOW)
        tracker.note("a", None, POLICY, now=NOW)

        after_window = NOW + timedelta(seconds=POLICY.window_s + 1)
        assert tracker.note("a", None, POLICY, now=after_window) == "sink"
        assert tracker.disabled("a", now=after_window) is None

    def test_a_success_clears_the_count_and_the_ladder(self):
        tracker = RateLimitTracker()
        for _ in range(2):
            tracker.note("a", None, POLICY, now=NOW)
        tracker.clear_lane("a", None)

        assert tracker.sunk("a") is False
        assert tracker.note("a", None, POLICY, now=NOW) == "sink"
        # The ladder restarted too: the next disable is the first one again.
        for _ in range(2):
            tracker.note("a", None, POLICY, now=NOW)
        assert tracker.disabled("a", now=NOW) == NOW + timedelta(seconds=60)

    def test_sink_can_be_off_without_turning_the_rule_off(self):
        tracker = RateLimitTracker()
        policy = RateLimitPolicy(sink=False)

        assert tracker.note("a", None, policy, now=NOW) == "off"
        assert tracker.sunk("a") is False

        tracker.note("a", None, policy, now=NOW)
        assert tracker.note("a", None, policy, now=NOW) == "disable"


class TestEligibility:
    def test_a_rate_limited_provider_is_not_a_candidate(self):
        pinfo = _pinfo()
        for _ in range(3):
            QUOTA_STATE.rate_limits.note("a", None, POLICY, now=NOW)

        answer = provider_eligibility(pinfo)

        assert answer.eligible is False
        assert answer.reason == RATE_LIMITED
        assert answer.until is not None

    def test_a_sunk_provider_is_still_a_candidate(self):
        pinfo = _pinfo()
        QUOTA_STATE.rate_limits.note("a", None, POLICY, now=NOW)

        answer = provider_eligibility(pinfo)

        assert answer.eligible is True
        assert answer.sunk is True

    def test_a_spent_allowance_is_reported_with_its_reset(self):
        pinfo = _pinfo()
        pinfo.quota_probe = "p-eligibility-spent"
        QUOTA_STATE.put_reading(
            QuotaReading(
                probe="p-eligibility-spent",
                credential=credential_fingerprint("k1"),
                windows=[
                    QuotaWindow(
                        name="5h",
                        used_percent=100,
                        resets_at=NOW + timedelta(minutes=5),
                    )
                ],
            )
        )

        answer = provider_eligibility(pinfo)

        assert answer.eligible is False
        assert answer.reason == SPENT
        assert answer.until == NOW + timedelta(minutes=5)

    def test_a_pending_credential_is_initializing(self):
        from llm_rosetta_gateway.transport.provider_info import TOKEN_PENDING_SENTINEL

        pinfo = _pinfo(keys=TOKEN_PENDING_SENTINEL)

        answer = provider_eligibility(pinfo)

        assert answer.eligible is False
        assert answer.reason == INITIALIZING

    def test_a_provider_block_is_reported_before_anything_else(self):
        pinfo = _pinfo()
        pinfo.quota_identity = "@eligibility-block"
        QUOTA_STATE.block_provider("@eligibility-block", 30, "usage limit")
        pinfo.quota_provider_blocked = lambda: True

        answer = provider_eligibility(pinfo)

        assert answer.eligible is False
        assert answer.reason == BLOCKED
        assert answer.until is not None

    def test_one_working_key_keeps_the_provider(self):
        """A disabled credential is not a disabled provider."""
        from llm_rosetta_gateway.quota.service import bind_provider_quota

        pinfo = _pinfo(keys="k1,k2")
        bind_provider_quota("a", pinfo)
        for _ in range(3):
            QUOTA_STATE.rate_limits.note(
                "a", credential_fingerprint("k1"), POLICY, now=NOW
            )

        answer = provider_eligibility(pinfo)

        assert answer.eligible is True
        assert answer.sunk is True

    def test_every_key_disabled_takes_the_provider_out(self):
        from llm_rosetta_gateway.quota.service import bind_provider_quota

        pinfo = _pinfo(keys="k1")
        bind_provider_quota("a", pinfo)
        for _ in range(3):
            QUOTA_STATE.rate_limits.note(
                "a", credential_fingerprint("k1"), POLICY, now=NOW
            )

        answer = provider_eligibility(pinfo)

        assert answer.eligible is False
        assert answer.reason == RATE_LIMITED


class TestTheStage:
    def test_the_stage_records_one_answer_per_candidate(self):
        config = _config("a", "b")
        ctx = config.selection_context("m", "chat_completions")
        for _ in range(3):
            QUOTA_STATE.rate_limits.note("a", None, POLICY, now=NOW)

        EligibilityStage().run(ctx)

        assert set(ctx.eligibility) == {"a", "b"}
        assert ctx.eligibility["a"].eligible is False
        assert ctx.eligibility["b"].eligible is True
        assert ctx.excluded_providers() == frozenset({"a"})


class TestTheRoute:
    def test_a_sunk_provider_is_tried_last(self):
        config = _config("a", "b")
        QUOTA_STATE.rate_limits.note("a", None, POLICY, now=NOW)

        served = set()
        for _ in range(6):
            selection = config.selection_context("m", "chat_completions")
            config.resolve(selection)
            assert selection.chosen is not None
            served.add(selection.chosen.name)

        assert served == {"b"}

    def test_a_route_of_nothing_but_sunk_providers_still_serves(self):
        config = _config("a")
        QUOTA_STATE.rate_limits.note("a", None, POLICY, now=NOW)

        selection = config.selection_context("m", "chat_completions")
        config.resolve(selection)

        assert selection.chosen is not None and selection.chosen.name == "a"

    def test_a_disabled_provider_leaves_the_sunk_one_to_serve(self):
        config = _config("a", "b")
        QUOTA_STATE.rate_limits.note("a", None, POLICY, now=NOW)
        for _ in range(3):
            QUOTA_STATE.rate_limits.note("b", None, POLICY, now=NOW)

        selection = config.selection_context("m", "chat_completions")
        config.resolve(selection)

        assert selection.chosen is not None and selection.chosen.name == "a"

    def test_a_route_with_only_disabled_providers_is_blocked(self):
        from llm_rosetta_gateway.maintenance.startup import ProviderNotReady

        config = _config("a")
        for _ in range(3):
            QUOTA_STATE.rate_limits.note("a", None, POLICY, now=NOW)

        selection = config.selection_context("m", "chat_completions")
        with pytest.raises(ProviderNotReady) as caught:
            config.resolve(selection)

        # The client is told when to come back, not that the gateway is booting.
        assert caught.value.retry_after is not None
        assert "quota block until" in str(caught.value)


class TestTheStrategySeesIt:
    def test_the_route_hands_a_sunk_provider_over_last(self):
        route = ModelRoute(
            providers=[ProviderEntry("a"), ProviderEntry("b")],
            route_id="r-sunk-order",
        )

        assert [route.select(sunk=frozenset({"a"})) for _ in range(4)] == ["b"] * 4

    def test_two_sunk_providers_keep_their_share(self):
        route = ModelRoute(
            providers=[ProviderEntry("a"), ProviderEntry("b")],
            route_id="r-sunk-share",
        )

        picks = [route.select(sunk=frozenset({"a", "b"})) for _ in range(10)]

        assert picks.count("a") == picks.count("b") == 5
