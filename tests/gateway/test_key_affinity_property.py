"""Key affinity as exact laws (Hypothesis).

``test_key_affinity.py`` checks rendezvous hashing with samples: 300
identities and "at most 40% moved" for an added key.  The laws hold for every
identity, so they are stated that way here:

* ``select`` returns a configured key, whatever the order of the keys;
* adding a key moves an identity only onto the new key;
* removing a key moves only the identities that were on it, onto a remaining key;
* a usable-key filter selects as a ring of just those keys would, and a ring
  with no usable key falls back to all of them;
* ``select`` never advances the round-robin counter, and ``next`` walks the
  usable keys in order and then around;
* ``refresh`` keeps the ring on an empty list, reports a changed key count and
  keeps the round-robin position in range;
* the prefix of a request depends on its system instruction and first
  messages only, and not on how its dicts are ordered.
"""

from __future__ import annotations

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from llm_rosetta_gateway.routing import (  # noqa: E402
    compute_affinity_identity,
    extract_prefix_from_ir,
)
from llm_rosetta_gateway.transport.provider_info import KeyRing  # noqa: E402

_SETTINGS = settings(max_examples=300, deadline=None, derandomize=True)

_KEY = st.text(alphabet="abcdefghijkl0123456789-_", min_size=1, max_size=8)
_KEYS = st.lists(_KEY, min_size=1, max_size=6, unique=True)
_IDENTITY = st.text(max_size=20)


def _ring(keys: list[str], usable=None) -> KeyRing:
    return KeyRing(",".join(keys), usable=usable)


@_SETTINGS
@given(keys=_KEYS, identity=_IDENTITY, seed=st.randoms())
def test_select_returns_a_configured_key_whatever_their_order(
    keys: list[str], identity: str, seed
) -> None:
    shuffled = list(keys)
    seed.shuffle(shuffled)
    chosen = _ring(keys).select(identity)
    assert chosen in keys
    assert _ring(shuffled).select(identity) == chosen
    assert _ring(keys).select(identity) == chosen


@_SETTINGS
@given(keys=_KEYS, new_key=_KEY, identity=_IDENTITY)
def test_adding_a_key_moves_an_identity_only_onto_the_new_key(
    keys: list[str], new_key: str, identity: str
) -> None:
    if new_key in keys:
        return
    before = _ring(keys).select(identity)
    after = _ring([*keys, new_key]).select(identity)
    assert after in (before, new_key)


@_SETTINGS
@given(keys=st.lists(_KEY, min_size=2, max_size=6, unique=True), identity=_IDENTITY)
def test_removing_a_key_moves_only_the_identities_that_were_on_it(
    keys: list[str], identity: str
) -> None:
    removed = keys[0]
    remaining = keys[1:]
    before = _ring(keys).select(identity)
    after = _ring(remaining).select(identity)
    assert after in remaining
    if before != removed:
        assert after == before


@_SETTINGS
@given(keys=_KEYS, identity=_IDENTITY, flags=st.lists(st.booleans(), min_size=6))
def test_a_usable_filter_selects_as_a_ring_of_the_usable_keys_would(
    keys: list[str], identity: str, flags: list[bool]
) -> None:
    ok = {key for key, flag in zip(keys, flags) if flag}
    filtered = _ring(keys, usable=lambda key: key in ok).select(identity)
    if ok:
        assert filtered == _ring([k for k in keys if k in ok]).select(identity)
    else:
        assert filtered == _ring(keys).select(identity)


@_SETTINGS
@given(keys=_KEYS, identities=st.lists(_IDENTITY, max_size=8))
def test_select_does_not_advance_the_round_robin(
    keys: list[str], identities: list[str]
) -> None:
    touched, untouched = _ring(keys), _ring(keys)
    for identity in identities:
        touched.select(identity)
    assert [touched.next() for _ in range(2 * len(keys))] == [
        untouched.next() for _ in range(2 * len(keys))
    ]


@_SETTINGS
@given(keys=_KEYS, rounds=st.integers(min_value=1, max_value=3))
def test_next_walks_the_keys_in_order_and_then_around(
    keys: list[str], rounds: int
) -> None:
    ring = _ring(keys)
    assert [ring.next() for _ in range(rounds * len(keys))] == keys * rounds


@_SETTINGS
@given(keys=_KEYS, flags=st.lists(st.booleans(), min_size=6))
def test_next_skips_unusable_keys_and_cycles_all_when_none_is_usable(
    keys: list[str], flags: list[bool]
) -> None:
    ok = [key for key, flag in zip(keys, flags) if flag]
    ring = _ring(keys, usable=lambda key: key in ok)
    walked = [ring.next() for _ in range(2 * len(keys))]
    if ok:
        assert set(walked) == set(ok)
        assert all(key in ok for key in walked)
    else:
        assert walked == keys * 2
    assert ring.any_usable() == bool(ok)


@_SETTINGS
@given(keys=_KEYS, other=_KEYS, advanced=st.integers(min_value=0, max_value=8))
def test_refresh_reports_a_changed_count_and_keeps_the_position_in_range(
    keys: list[str], other: list[str], advanced: int
) -> None:
    ring = _ring(keys)
    for _ in range(advanced):
        ring.next()
    previous = ring.refresh(",".join(other))
    assert previous == (len(keys) if len(keys) != len(other) else None)
    assert ring.keys() == tuple(other)
    assert ring.next() in other
    assert len(ring) == len(other)


@_SETTINGS
@given(keys=_KEYS, blanks=st.sampled_from(["", " ", ",", " , ,"]))
def test_refresh_with_no_key_leaves_the_ring_alone(
    keys: list[str], blanks: str
) -> None:
    ring = _ring(keys)
    assert ring.refresh(blanks) is None
    assert ring.keys() == tuple(keys)


@_SETTINGS
@given(keys=_KEYS, value=_KEY)
def test_contains_only_means_exactly_that_one_key(keys: list[str], value: str) -> None:
    assert _ring(keys).contains_only(value) == (keys == [value])


_LEAF = st.one_of(st.none(), st.booleans(), st.integers(), st.text(max_size=8))
_MESSAGE = st.dictionaries(
    st.text(min_size=1, max_size=4), _LEAF, min_size=1, max_size=3
)


@_SETTINGS
@given(
    system=st.one_of(st.none(), st.lists(_MESSAGE, min_size=1, max_size=2)),
    messages=st.lists(_MESSAGE, max_size=5),
    extra=st.lists(_MESSAGE, max_size=3),
    limit=st.integers(min_value=1, max_value=3),
)
def test_a_prefix_reads_only_the_system_instruction_and_the_leading_messages(
    system, messages: list[dict], extra: list[dict], limit: int
) -> None:
    ir: dict = {"messages": messages}
    if system is not None:
        ir["system_instruction"] = system
    prefix = extract_prefix_from_ir(ir, max_messages=limit)

    longer = {**ir, "messages": messages + extra}
    if len(messages) >= limit:
        assert extract_prefix_from_ir(longer, max_messages=limit) == prefix

    reordered = {
        **ir,
        "messages": [dict(reversed(list(m.items()))) for m in messages],
    }
    if system is not None:
        reordered["system_instruction"] = [
            dict(reversed(list(m.items()))) for m in system
        ]
    assert extract_prefix_from_ir(reordered, max_messages=limit) == prefix

    parts = len(messages[:limit]) + (1 if system else 0)
    assert prefix.count("\n") == max(parts - 1, 0)


@_SETTINGS
@given(client=st.text(max_size=8), prefix=st.text(max_size=8))
def test_an_identity_needs_both_halves(client: str, prefix: str) -> None:
    identity = compute_affinity_identity(client, prefix)
    if client and prefix:
        assert identity == f"{client}\n{prefix}"
    else:
        assert identity is None
