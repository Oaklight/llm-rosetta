"""Provider selection as laws (Hypothesis and bounded enumeration).

``test_routing_strategy.py`` pins each strategy with a handful of sequences
(``A A A A A B`` for weights 5 and 1).  The laws behind them are stated here
over generated weights, candidate sets and identities:

* weighted round-robin picks each candidate exactly its weight per cycle and
  repeats the cycle; rotate walks the candidates in order and around; least-used
  keeps every count within one of the others; ordered takes the first;
* affinity is a weighted rendezvous: the choice is stable for an identity,
  survives the removal of a candidate that did not win, moves only onto a
  candidate that is added, and cannot be lost by a heavier weight;
* ``ModelRoute.select`` never returns an excluded candidate and offers a sunk
  one only when nothing else is left (and only when the route sinks at all);
* ``select_ready_entry`` is checked on every combination of three candidates'
  answers and exclusions against an independent statement of the rule.

Each test names the product's own function: the strategy comes from
``create_strategy`` and the selector is the one ``AnchorStage`` calls.
"""

from __future__ import annotations

import itertools
from collections import Counter

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from llm_rosetta_gateway.routing import (  # noqa: E402
    AnchorStage,
    Eligibility,
    ModelRoute,
    ProviderEntry,
    create_strategy,
)
from llm_rosetta_gateway.routing import select as select_module  # noqa: E402
from llm_rosetta_gateway.routing.model_routing import ModelRouting  # noqa: E402
from llm_rosetta_gateway.routing.select import select_ready_entry  # noqa: E402

_SETTINGS = settings(max_examples=200, deadline=None, derandomize=True)
ROUTE = "r"

_NAMES = ["a", "b", "c", "d", "e"]


def _entries(weights: list[float]) -> list[ProviderEntry]:
    return [ProviderEntry(_NAMES[i], weight=w) for i, w in enumerate(weights)]


_INT_WEIGHTS = st.lists(st.integers(min_value=1, max_value=6), min_size=2, max_size=4)
_REAL_WEIGHTS = st.lists(
    st.floats(min_value=0.1, max_value=10.0, allow_nan=False), min_size=2, max_size=5
)


def test_the_selectors_under_test_are_the_ones_the_product_calls() -> None:
    assert AnchorStage.run.__globals__["select_ready_entry"] is select_ready_entry
    assert select_ready_entry.__module__ == select_module.__name__
    for name in ("weighted", "ordered", "rotate", "least_used", "pace"):
        strategy = create_strategy(name)
        assert type(strategy).__module__.endswith("routing.strategy"), name


@_SETTINGS
@given(weights=_INT_WEIGHTS, cycles=st.integers(min_value=1, max_value=3))
def test_weighted_round_robin_picks_each_candidate_its_weight_per_cycle(
    weights: list[int], cycles: int
) -> None:
    entries = _entries([float(w) for w in weights])
    strategy = create_strategy("weighted")
    cycle = sum(weights)
    picks = [
        strategy.select(entries, route_id=ROUTE).name for _ in range(cycles * cycle)
    ]
    for index, weight in enumerate(weights):
        assert picks.count(_NAMES[index]) == cycles * weight
    assert picks == picks[:cycle] * cycles


@_SETTINGS
@given(weights=_REAL_WEIGHTS, count=st.integers(min_value=1, max_value=12))
def test_rotate_takes_turns_in_order_whatever_the_weights(
    weights: list[float], count: int
) -> None:
    entries = _entries(weights)
    strategy = create_strategy("rotate")
    picks = [strategy.select(entries, route_id=ROUTE).name for _ in range(count)]
    assert picks == [_NAMES[i % len(entries)] for i in range(count)]


@_SETTINGS
@given(weights=_REAL_WEIGHTS, count=st.integers(min_value=1, max_value=12))
def test_ordered_always_takes_the_first(weights: list[float], count: int) -> None:
    entries = _entries(weights)
    strategy = create_strategy("ordered")
    assert {strategy.select(entries, route_id=ROUTE).name for _ in range(count)} == {
        "a"
    }


@_SETTINGS
@given(weights=_REAL_WEIGHTS, count=st.integers(min_value=1, max_value=40))
def test_least_used_keeps_every_count_within_one_of_the_others(
    weights: list[float], count: int
) -> None:
    entries = _entries(weights)
    strategy = create_strategy("least_used")
    picks = Counter(strategy.select(entries, route_id=ROUTE).name for _ in range(count))
    counts = [picks[e.name] for e in entries]
    assert max(counts) - min(counts) <= 1
    assert sum(counts) == count


@_SETTINGS
@given(weights=_REAL_WEIGHTS)
def test_routes_keep_separate_counters(
    weights: list[float],
) -> None:
    entries = _entries(weights)
    for name in ("weighted", "rotate", "least_used"):
        alone = create_strategy(name)
        first = [alone.select(entries, route_id="x").name for _ in range(5)]
        # Another route's traffic in between does not move this route's counter.
        shared = create_strategy(name)
        mixed = []
        for _ in range(5):
            shared.select(entries, route_id="y")
            mixed.append(shared.select(entries, route_id="x").name)
        assert mixed == first, name


@_SETTINGS
@given(
    weights=st.lists(st.integers(1, 5), min_size=3, max_size=5),
    warmup=st.integers(1, 7),
)
def test_a_route_whose_candidates_change_starts_its_counters_over(
    weights: list[int], warmup: int
) -> None:
    entries = _entries([float(w) for w in weights])
    fewer = entries[:-1]
    for name in ("weighted", "least_used"):
        used = create_strategy(name)
        for _ in range(warmup):
            used.select(entries, route_id="x")
        fresh = create_strategy(name)
        assert [used.select(fewer, route_id="x").name for _ in range(6)] == [
            fresh.select(fewer, route_id="x").name for _ in range(6)
        ], name


@pytest.mark.parametrize(
    "name", ["weighted", "ordered", "rotate", "least_used", "pace"]
)
def test_no_candidate_is_an_error(name: str) -> None:
    with pytest.raises(ValueError):
        create_strategy(name).select([], route_id=ROUTE)
    with pytest.raises(ValueError):
        create_strategy("affinity_round_robin").select([], route_id=ROUTE)


_IDENTITY = st.text(min_size=1, max_size=12)


@_SETTINGS
@given(weights=_REAL_WEIGHTS, identity=_IDENTITY)
def test_affinity_is_stable_and_survives_the_removal_of_a_loser(
    weights: list[float], identity: str
) -> None:
    entries = _entries(weights)
    chosen = create_strategy("affinity_round_robin").select(
        entries, route_id=ROUTE, identity=identity
    )
    again = create_strategy("affinity_round_robin").select(
        entries, route_id="other", identity=identity
    )
    assert again.name == chosen.name
    for loser in entries:
        if loser.name == chosen.name or len(entries) <= 2:
            continue
        rest = [e for e in entries if e.name != loser.name]
        kept = create_strategy("affinity_round_robin").select(
            rest, route_id=ROUTE, identity=identity
        )
        assert kept.name == chosen.name


@_SETTINGS
@given(weights=_REAL_WEIGHTS, identity=_IDENTITY, new_weight=st.floats(0.1, 10.0))
def test_affinity_moves_an_identity_only_onto_an_added_candidate(
    weights: list[float], identity: str, new_weight: float
) -> None:
    if len(weights) >= len(_NAMES):
        return
    entries = _entries(weights)
    added = ProviderEntry("z-new", weight=new_weight)
    strategy = create_strategy("affinity_round_robin")
    before = strategy.select(entries, route_id=ROUTE, identity=identity)
    after = strategy.select([*entries, added], route_id=ROUTE, identity=identity)
    assert after.name in (before.name, "z-new")


@_SETTINGS
@given(
    weights=_REAL_WEIGHTS,
    identity=_IDENTITY,
    boost=st.floats(min_value=1.0, max_value=20.0),
)
def test_a_heavier_weight_never_loses_an_identity_it_won(
    weights: list[float], identity: str, boost: float
) -> None:
    entries = _entries(weights)
    strategy = create_strategy("affinity_round_robin")
    winner = strategy.select(entries, route_id=ROUTE, identity=identity)
    heavier = [
        ProviderEntry(e.name, weight=e.weight * (boost if e.name == winner.name else 1))
        for e in entries
    ]
    assert (
        strategy.select(heavier, route_id=ROUTE, identity=identity).name == winner.name
    )


@_SETTINGS
@given(weights=_INT_WEIGHTS, count=st.integers(min_value=1, max_value=10))
def test_affinity_without_an_identity_is_weighted_round_robin(
    weights: list[int], count: int
) -> None:
    entries = _entries([float(w) for w in weights])
    affinity = create_strategy("affinity_round_robin")
    plain = create_strategy("weighted")
    for _ in range(count):
        assert (
            affinity.select(entries, route_id=ROUTE).name
            == plain.select(entries, route_id=ROUTE).name
        )


_STATES = st.lists(
    st.tuples(st.booleans(), st.booleans()), min_size=2, max_size=5
)  # (excluded, sunk)


@_SETTINGS
@given(
    weights=st.lists(st.integers(1, 4), min_size=2, max_size=5),
    marks=_STATES,
    sink=st.booleans(),
    strategy=st.sampled_from(["weighted", "ordered", "rotate", "least_used"]),
    calls=st.integers(min_value=1, max_value=6),
)
def test_a_route_never_returns_an_excluded_candidate_and_offers_a_sunk_one_last(
    weights: list[int],
    marks: list[tuple[bool, bool]],
    sink: bool,
    strategy: str,
    calls: int,
) -> None:
    size = min(len(weights), len(marks))
    entries = _entries([float(w) for w in weights[:size]])
    marks = marks[:size]
    excluded = frozenset(e.key for e, (out, _) in zip(entries, marks) if out)
    sunk = frozenset(e.key for e, (_, s) in zip(entries, marks) if s)
    route = ModelRoute(
        entries,
        ROUTE,
        strategy=create_strategy(strategy),
        routing=ModelRouting(strategy=strategy, sink=sink),
    )
    open_ = [e for e in entries if e.key not in excluded]
    if not open_:
        with pytest.raises(ValueError):
            route.select(excluded=excluded, sunk=sunk)
        return
    standing = [e for e in open_ if e.key not in sunk]
    for _ in range(calls):
        picked = route.select(excluded=excluded, sunk=sunk)
        assert picked.key not in excluded
        if sink and standing:
            assert picked.key not in sunk


_ANSWER = ["absent", "ineligible", "eligible", "sunk"]


@pytest.mark.parametrize("excluded_mask", range(8))
def test_select_ready_entry_follows_its_rule_on_every_combination(
    excluded_mask: int,
) -> None:
    entries = _entries([1.0, 1.0, 1.0])
    excluded = frozenset(e.key for i, e in enumerate(entries) if excluded_mask >> i & 1)
    for states in itertools.product(_ANSWER, repeat=3):
        eligibility = {}
        for entry, state in zip(entries, states):
            if state == "absent":
                continue
            eligibility[entry.key] = Eligibility(
                name=entry.name,
                eligible=state != "ineligible",
                sunk=state == "sunk",
            )
        route = ModelRoute(
            entries,
            ROUTE,
            strategy=create_strategy("ordered"),
            routing=ModelRouting(strategy="ordered"),
        )
        chosen = select_ready_entry(route, eligibility=eligibility, excluded=excluded)

        usable = [
            e
            for e, s in zip(entries, states)
            if s in ("eligible", "sunk") and e.key not in excluded
        ]
        standing = [
            e for e, s in zip(entries, states) if e in usable and s == "eligible"
        ]
        expected = (standing or usable or [None])[0]
        assert chosen == expected, (states, excluded_mask)


def test_an_empty_upstream_model_is_no_upstream_model() -> None:
    plain, empty = ProviderEntry("a"), ProviderEntry("a", upstream_model="")
    assert empty.upstream_model is None
    assert plain.key == empty.key == ("a", None)
    assert ProviderEntry("a", upstream_model="m").key == ("a", "m")


@_SETTINGS
@given(weights=_REAL_WEIGHTS)
def test_the_first_weighted_pick_goes_to_the_earliest_of_the_heaviest(
    weights: list[float],
) -> None:
    entries = _entries(weights)
    first = create_strategy("weighted").select(entries, route_id=ROUTE)
    assert first.name == _NAMES[weights.index(max(weights))]


@_SETTINGS
@given(weights=_REAL_WEIGHTS, identity=_IDENTITY)
def test_affinity_is_the_documented_weighted_rendezvous(
    weights: list[float], identity: str
) -> None:
    """The smallest ``-ln(u) / weight`` wins, ``u`` hashed from (identity, name)."""
    import hashlib
    import math

    entries = _entries(weights)

    def score(entry: ProviderEntry) -> float:
        digest = hashlib.sha256(f"{identity}\0{entry.name}".encode()).digest()
        draw = (int.from_bytes(digest[:8], "big") + 1) / 2**64
        return -math.log(draw) / entry.weight

    expected = min(entries, key=score)
    chosen = create_strategy("affinity_round_robin").select(
        entries, route_id=ROUTE, identity=identity
    )
    assert chosen.name == expected.name


def test_affinity_tells_two_lanes_of_one_provider_apart() -> None:
    lanes = [
        ProviderEntry("a", upstream_model="m1"),
        ProviderEntry("a", upstream_model="m2"),
    ]
    strategy = create_strategy("affinity_round_robin")
    won = {
        strategy.select(lanes, route_id=ROUTE, identity=f"client-{i}").upstream_model
        for i in range(200)
    }
    assert won == {"m1", "m2"}


@_SETTINGS
@given(
    weights=st.lists(st.floats(0.5, 5.0), min_size=1, max_size=3), identity=_IDENTITY
)
def test_a_zero_weight_candidate_never_beats_a_positive_one(
    weights: list[float], identity: str
) -> None:
    entries = [*_entries(weights), ProviderEntry("zero", weight=0.0)]
    chosen = create_strategy("affinity_round_robin").select(
        entries, route_id=ROUTE, identity=identity
    )
    assert chosen.name != "zero"


@pytest.mark.parametrize(
    "strategy", ["weighted", "ordered", "rotate", "least_used", "pace"]
)
@pytest.mark.parametrize("affinity", ["off", "session"])
def test_strategy_for_follows_the_affinity_and_then_the_strategy(
    strategy: str, affinity: str
) -> None:
    from llm_rosetta_gateway.routing.strategy import strategy_for

    chosen = strategy_for(ModelRouting(strategy=strategy, affinity=affinity))
    expected = create_strategy(
        "affinity_round_robin" if affinity == "session" else strategy
    )
    assert type(chosen) is type(expected)


def test_a_route_scales_weights_by_the_request_factors_and_never_to_zero() -> None:
    entries = _entries([1.0, 1.0])
    weighted = ModelRoute(entries, ROUTE, strategy=create_strategy("weighted"))
    factors = {("a", None): 0.5}
    picks = Counter(weighted.select(weights=factors).name for _ in range(3))
    assert picks == {"a": 1, "b": 2}

    spent = ModelRoute(entries, "spent", strategy=create_strategy("weighted"))
    picks = Counter(spent.select(weights={("a", None): 0.0}).name for _ in range(21))
    assert picks == {"a": 1, "b": 20}


def test_pace_ignores_configured_weights_and_scales_by_headroom_alone() -> None:
    entries = _entries([5.0, 1.0])
    pace = ModelRoute(entries, ROUTE, strategy=create_strategy("pace"))
    assert Counter(pace.select().name for _ in range(2)) == {"a": 1, "b": 1}

    scaled = ModelRoute(entries, "scaled", strategy=create_strategy("pace"))
    picks = Counter(scaled.select(weights={("b", None): 0.5}).name for _ in range(3))
    assert picks == {"a": 2, "b": 1}

    floored = ModelRoute(entries, "floored", strategy=create_strategy("pace"))
    picks = Counter(floored.select(weights={("a", None): 0.0}).name for _ in range(21))
    assert picks == {"a": 1, "b": 20}


@_SETTINGS
@given(weights=_INT_WEIGHTS)
def test_a_route_that_does_not_sink_still_offers_a_sunk_candidate(
    weights: list[int],
) -> None:
    entries = _entries([float(w) for w in weights])
    sunk = frozenset({entries[0].key})
    for sink, expected in ((True, set(_NAMES[1 : len(entries)])), (False, None)):
        route = ModelRoute(
            entries,
            f"sink-{sink}",
            strategy=create_strategy("rotate"),
            routing=ModelRouting(strategy="rotate", sink=sink),
        )
        seen = {route.select(sunk=sunk).name for _ in range(len(entries))}
        assert seen == (expected or {e.name for e in entries})


def test_a_route_reports_its_providers_and_whether_it_has_several() -> None:
    one = ModelRoute(_entries([1.0]), ROUTE)
    two = ModelRoute(_entries([1.0, 2.0]), ROUTE)
    assert (one.provider_names, one.is_multi) == (["a"], False)
    assert (two.provider_names, two.is_multi) == (["a", "b"], True)


@_SETTINGS
@given(
    heads=st.lists(st.sampled_from([None, 0.0, 0.2, 0.5, 1.0]), min_size=2, max_size=3),
    calls=st.integers(min_value=1, max_value=8),
)
def test_the_ready_selector_scales_each_candidate_by_its_headroom(
    heads: list[float | None], calls: int
) -> None:
    entries = _entries([1.0] * len(heads))
    eligibility = {
        e.key: Eligibility(name=e.name, eligible=True, headroom=h)
        for e, h in zip(entries, heads)
    }
    factors = {
        k: a.weight_factor for k, a in eligibility.items() if a.weight_factor != 1.0
    }
    via_selector = ModelRoute(entries, "sel", strategy=create_strategy("weighted"))
    direct = ModelRoute(entries, "dir", strategy=create_strategy("weighted"))
    for _ in range(calls):
        picked = select_ready_entry(via_selector, eligibility=eligibility)
        assert picked == direct.select(weights=factors or None)


def test_the_anchor_stage_chooses_what_the_ready_selector_chooses() -> None:
    from llm_rosetta_gateway.routing import SelectionContext

    entries = _entries([1.0, 1.0])
    route = ModelRoute(
        entries,
        ROUTE,
        strategy=create_strategy("ordered"),
        routing=ModelRouting(strategy="ordered"),
    )
    ctx = SelectionContext("m", "chat_completions")
    ctx.install_route(
        model_route=route,
        providers={},
        provider_types={},
        provider_shim_names={},
        provider_account_tokens={},
    )
    ctx.with_eligibility(
        {
            ("a", None): Eligibility(name="a", eligible=False),
            ("b", None): Eligibility(name="b", eligible=True),
        }
    )
    AnchorStage().run(ctx)
    assert ctx.chosen == entries[1]

    ctx.with_eligibility({("a", None): Eligibility(name="a", eligible=False)})
    ctx.chosen = None
    AnchorStage().run(ctx)
    assert ctx.chosen is None


@_SETTINGS
@given(weights=_INT_WEIGHTS)
def test_a_route_applies_the_configured_weights_when_the_request_has_none(
    weights: list[int],
) -> None:
    entries = _entries([float(w) for w in weights])
    route = ModelRoute(entries, ROUTE, strategy=create_strategy("weighted"))
    picks = Counter(route.select().name for _ in range(sum(weights)))
    assert [picks[e.name] for e in entries] == weights
