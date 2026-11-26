"""The 429 ladder, checked over every short event sequence.

``test_routing_rate_limit.py`` pins the ladder with one example per rule.  Here
the rules of ``quota/rate_limit.py`` are written out once as a small reference
model and ``RateLimitTracker`` is driven through *every* sequence of up to four
events (a refusal at several distances from the last event, a success, a read)
under five policies.  After each event the tracker and the model must agree on
the outcome of a refusal, on whether the lane is sunk, and on the live record
(count, strikes, window, disable).

The model is the module docstring restated, not a copy of the code: a record
lives until its window ends unless it is disabled, a refusal inside the window
adds to the count and one outside it restarts the count while the strikes
stay, ``after`` refusals disable for ``cooldown_s`` doubled per strike up to
the cap, and a success forgets the lane.  A separate test states the doubling
as a closed form.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from llm_rosetta_gateway.quota.rate_limit import RateLimitPolicy, RateLimitTracker

START = datetime(2026, 1, 1, tzinfo=timezone.utc)
PROVIDER, CREDENTIAL = "p", "c"

#: A policy is judged under the five shapes that reach different branches: the
#: default, a rung per refusal with a small cap, no sink, never disable, and a
#: cooldown longer than the window (a disabled lane outlives its window).
POLICIES = {
    "default": RateLimitPolicy(),
    "every_refusal": RateLimitPolicy(
        after=1, window_s=100, cooldown_s=10, cooldown_max_s=25
    ),
    "no_sink": RateLimitPolicy(
        sink=False, after=2, window_s=100, cooldown_s=10, cooldown_max_s=25
    ),
    "never_disable": RateLimitPolicy(after=0, window_s=100),
    "cooldown_over_window": RateLimitPolicy(
        after=2, window_s=100, cooldown_s=400, cooldown_max_s=900
    ),
}


def _events(policy: RateLimitPolicy) -> list[tuple[str, float]]:
    """``(kind, seconds after the previous event)``, placed at the policy's own
    boundaries: the cooldown and the window, each one second short, exact and
    one second over."""
    gaps = {
        0,
        1,
        policy.cooldown_s - 1,
        policy.cooldown_s,
        policy.cooldown_s + 1,
        policy.window_s - 1,
        policy.window_s,
        policy.window_s + 1,
    }
    refusals = [("refuse", g) for g in sorted(gaps) if g >= 0]
    return [
        *refusals,
        ("success", 0),
        ("read", policy.cooldown_s),
        ("read", policy.window_s),
    ]


@dataclass
class _Record:
    refusals: int
    window_end: datetime
    strikes: int
    disabled_until: datetime | None
    sink: bool

    def disabled(self, now: datetime) -> bool:
        return self.disabled_until is not None and now < self.disabled_until


class _Model:
    """One lane, from the rules in the module docstring."""

    def __init__(self, policy: RateLimitPolicy) -> None:
        self.policy = policy
        self.record: _Record | None = None

    def _expire(self, now: datetime) -> None:
        r = self.record
        if r is not None and now >= r.window_end and not r.disabled(now):
            self.record = None

    def refuse(self, now: datetime) -> str:
        p = self.policy
        self._expire(now)
        before = self.record
        if before is not None and now <= before.window_end:
            refusals, strikes = before.refusals + 1, before.strikes
        else:
            refusals, strikes = 1, (before.strikes if before else 0)
        window_end = now + timedelta(seconds=p.window_s)
        if p.after and refusals >= p.after:
            strikes += 1
            seconds = min(p.cooldown_s * 2 ** (strikes - 1), p.cooldown_max_s)
            until = now + timedelta(seconds=seconds)
            self.record = _Record(0, window_end, strikes, until, p.sink)
            return "disable"
        self.record = _Record(refusals, window_end, strikes, None, p.sink)
        return "sink" if p.sink else "off"

    def success(self) -> None:
        self.record = None

    def sunk(self, now: datetime) -> bool:
        r = self.record
        return bool(r is not None and r.sink and now < r.window_end)

    def live(self, now: datetime) -> _Record | None:
        self._expire(now)
        return self.record


def _view(record) -> tuple | None:
    if record is None:
        return None
    return (
        record.refusals,
        record.strikes,
        record.window_end,
        record.disabled_until,
        record.sink,
    )


def _sequences(policy: RateLimitPolicy, max_length: int = 4):
    events = _events(policy)
    for length in range(1, max_length + 1):
        yield from itertools.product(events, repeat=length)


class _Axis:
    """One of the tracker's two stores, behind the same five calls."""

    def __init__(self, tracker: RateLimitTracker, upstream: bool) -> None:
        self.t, self.upstream = tracker, upstream

    def note(self, policy, now):
        if self.upstream:
            return self.t.note_upstream(PROVIDER, "lane", policy, now=now)
        return self.t.note(PROVIDER, CREDENTIAL, policy, now=now)

    def clear(self):
        if self.upstream:
            self.t.clear_upstream(PROVIDER, "lane")
        else:
            self.t.clear_lane(PROVIDER, CREDENTIAL)

    def sunk(self, now):
        if self.upstream:
            return self.t.upstream_sunk(PROVIDER, "lane", now=now)
        return self.t.sunk(PROVIDER, now=now)

    def record(self, now):
        if self.upstream:
            return self.t.upstream_record(PROVIDER, "lane", now=now)
        return self.t.record(PROVIDER, CREDENTIAL, now=now)

    def disabled(self, now):
        if self.upstream:
            return self.t.upstream_disabled(PROVIDER, "lane", now=now)
        return self.t.disabled(PROVIDER, CREDENTIAL, now=now)


@pytest.mark.parametrize("upstream", [False, True], ids=["credential", "upstream"])
@pytest.mark.parametrize("name", POLICIES)
def test_the_tracker_agrees_with_the_model_on_every_short_sequence(
    name: str, upstream: bool
) -> None:
    policy = POLICIES[name]
    checked = 0
    for events in _sequences(policy):
        tracker, model = RateLimitTracker(), _Model(policy)
        axis = _Axis(tracker, upstream)
        now = START
        for step, (kind, gap) in enumerate(events):
            now += timedelta(seconds=gap)
            where = f"{name} {events[: step + 1]}"
            if kind == "refuse":
                assert axis.note(policy, now) == model.refuse(now), where
            elif kind == "success":
                axis.clear()
                model.success()
            assert axis.sunk(now) == model.sunk(now), where
            assert _view(axis.record(now)) == _view(model.live(now)), where
            record = model.record
            assert axis.disabled(now) == (
                record.disabled_until if record and record.disabled(now) else None
            ), where
            if not upstream:
                assert tracker.latest_disable(PROVIDER, now=now) == axis.disabled(
                    now
                ), where
        checked += 1
    assert checked == sum(len(_events(policy)) ** n for n in range(1, 5))


@pytest.mark.parametrize("name", ["every_refusal", "no_sink"])
def test_consecutive_disables_double_up_to_the_cap(name: str) -> None:
    """Closed form: the k-th disable lasts ``min(cooldown * 2**k, cap)``.

    The next refusals come while the previous disable's window still holds; a
    lane that waits the window out starts from the first rung again (the
    sequence test above covers that).
    """
    policy = POLICIES[name]
    tracker, now = RateLimitTracker(), START
    lengths = []
    for _ in range(5):
        for _ in range(max(policy.after, 1)):
            outcome = tracker.note(PROVIDER, CREDENTIAL, policy, now=now)
        assert outcome == "disable"
        until = tracker.disabled(PROVIDER, CREDENTIAL, now=now)
        assert until is not None
        lengths.append((until - now).total_seconds())
        now = until + timedelta(seconds=1)
    assert lengths == [
        min(policy.cooldown_s * 2**k, policy.cooldown_max_s) for k in range(5)
    ]


def test_a_policy_that_never_disables_never_does() -> None:
    policy = POLICIES["never_disable"]
    tracker, now = RateLimitTracker(), START
    for _ in range(50):
        assert tracker.note(PROVIDER, CREDENTIAL, policy, now=now) == "sink"
        assert tracker.disabled(PROVIDER, CREDENTIAL, now=now) is None
        now += timedelta(seconds=1)


def test_a_policy_survives_its_own_dictionary() -> None:
    for policy in POLICIES.values():
        assert RateLimitPolicy.from_config(policy.as_dict()) == policy
    assert RateLimitPolicy.from_config(None) == RateLimitPolicy()


def test_a_policy_accepts_its_lower_bounds_and_refuses_what_is_below() -> None:
    from_config = RateLimitPolicy.from_config
    assert from_config({"after": 0}).after == 0
    assert from_config({"window_s": 1}).window_s == 1.0
    assert from_config({"cooldown_s": 1, "cooldown_max_s": 1}).cooldown_s == 1.0
    for bad in (
        {"after": -1},
        {"window_s": 0.5},
        {"cooldown_s": 0.5},
        {"cooldown_max_s": 0.5},
        {"cooldown_s": 10, "cooldown_max_s": 9},
    ):
        with pytest.raises(ValueError):
            from_config(bad)


def test_a_policy_refuses_a_value_of_the_wrong_type_or_an_unknown_key() -> None:
    from_config = RateLimitPolicy.from_config
    bad_configs = [
        [],
        "x",
        {"unknown": 1},
        {"sink": 1},
        {"sink": "yes"},
        {"after": True},
        {"after": 1.5},
        {"after": "3"},
        {"window_s": True},
        {"window_s": "300"},
        {"cooldown_s": None},
    ]
    for bad in bad_configs:
        with pytest.raises(ValueError):
            from_config(bad)


def test_an_empty_config_is_the_default_policy_and_it_sinks() -> None:
    assert RateLimitPolicy.from_config({}) == RateLimitPolicy()
    assert RateLimitPolicy.from_config({}).sink is True
    assert RateLimitPolicy().sink is True


def test_provider_wide_reads_cover_every_credential_of_that_provider_only() -> None:
    policy = POLICIES["every_refusal"]
    tracker = RateLimitTracker()
    # ``snapshot`` reads the wall clock, so this timeline starts now.
    start = datetime.now(timezone.utc)
    tracker.note("p", "k1", policy, now=start)
    later = start + timedelta(seconds=5)
    tracker.note("p", "k2", policy, now=later)
    tracker.note("q", "k1", policy, now=later)
    tracker.note_upstream("p", "lane", policy, now=later)

    # The latest disable of p is the later credential's, and q is not in it.
    assert tracker.latest_disable("p", now=later) == later + timedelta(seconds=10)
    assert tracker.latest_disable("q", now=later) == later + timedelta(seconds=10)
    assert tracker.latest_disable("none", now=later) is None

    snapshot = tracker.snapshot("p")
    assert {lane["credential"] for lane in snapshot["lanes"]} == {"k1", "k2"}
    assert snapshot["disabled_until"] is not None
    assert snapshot["disabled_until"].endswith("Z")

    assert tracker.tracked() == 4
    tracker.clear_provider("p")
    assert tracker.tracked() == 1
    assert tracker.record("p", "k1", now=later) is None
    assert tracker.upstream_record("p", "lane", now=later) is None
    assert tracker.record("q", "k1", now=later) is not None
    assert tracker.snapshot("p") == {
        "sunk": False,
        "disabled_until": None,
        "lanes": [],
    }
