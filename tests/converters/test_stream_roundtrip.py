"""Streaming round trips on generated turns — epic #1223, properties 2 and 3.

The body legs have their own modules (#1310 same format, #1329 cross format).
This is the stream leg: the same two properties on event sequences.

The input is generated rather than hand-written.  Every turn in
:data:`~tests.converters.test_leg_agreement.CASE_TABLE` is rendered as the
event sequence a stream of it would emit — ``expand_response`` is that
rendering (epic #1238) — so the cases come from a table of turns, not from one
canonical stream.

Two properties:

- **Same format, fixed point.**  ``IR events → A → IR events`` is not the
  identity: a writer adds the wrapper events its wire needs, and a reader
  reports the block boundaries its own vocabulary has.  What must hold is that
  a second pass changes nothing — ``read(write(read(write(events))))`` equals
  ``read(write(events))``.  A dialect that has eaten its input by the first
  pass and a dialect that carries it both satisfy this; the cross-format test
  below is what stops that from being vacuous.
- **Cross format, the turn.**  ``A → IR → B → IR → A`` must give back the turn
  A produced.  Event sequences are not comparable across dialects — each one
  has its own block bookkeeping — so the comparison is the turn view the leg
  agreement harness already defines (:class:`TurnView`), and a field that
  changes is recorded with the reason it is still open.

A dialect that cannot express a turn refuses it, by design (#1234), and
``REFUSED_REASON`` names those cases; they are asserted here as refusals
rather than skipped silently.

What the dialects disagree about today is committed in
``stream_roundtrip_losses.json``: a fixed point that stops holding, a new
field that changes across a pair, or a baseline entry that no longer happens
all fail the case.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

import pytest

from llm_rosetta.converters.base.context import StreamContext
from llm_rosetta.converters.base.errors import UnrepresentableFinishReason
from llm_rosetta.ir.primitives import expand_response
from tests.converters.test_leg_agreement import (
    CASE_TABLE,
    DIALECTS,
    REFUSED_REASON,
    _ir_response,
    _view_of_events,
)

BASELINE_PATH = Path(__file__).with_name("stream_roundtrip_losses.json")

#: What :func:`write_baseline` writes for an entry nobody has reviewed yet.
_UNREVIEWED = "UNREVIEWED"

_DIALECT_NAMES = list(DIALECTS)
_PAIR_IDS = [f"{a}->{b}" for a in _DIALECT_NAMES for b in _DIALECT_NAMES if a != b]
_TURNS = {case_id: turn for case_id, turn, _reason in CASE_TABLE}

#: The view fields a cross-format detour can change.
_VIEW_FIELDS = ("reason", "text", "reasoning", "refusal", "calls", "usage")


def _events(case_id: str) -> list[dict[str, Any]]:
    """Return the generated IR event sequence for one turn."""
    response = cast(Any, _ir_response(_TURNS[case_id]))
    return cast(list[dict[str, Any]], expand_response(response))


def drive(format_name: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Write *events* through one dialect and read them back.

    Args:
        format_name: The dialect to drive.
        events: The IR event sequence to feed it.

    Returns:
        The IR events the reader produced.

    Raises:
        UnrepresentableFinishReason: When the dialect cannot express the
            turn's terminal state — its designed answer, not a failure.
    """
    converter = DIALECTS[format_name]()
    write_context = StreamContext()
    chunks: list[dict[str, Any]] = []
    for event in copy.deepcopy(events):
        produced = converter.stream_response_to_provider(event, context=write_context)
        items = produced if isinstance(produced, list) else [produced]
        chunks.extend(item for item in items if item)

    # ``iter_stream_from_provider`` closes the stream, which a dialect whose
    # terminal event arrives with the last chunk needs: Chat defers its
    # ``stream_end`` until a chunk that never comes, so feeding the chunks one
    # at a time measures the harness, not the reader.
    read_context = converter.create_stream_context()
    return list(converter.iter_stream_from_provider(chunks, context=read_context))


def _baseline() -> dict[str, Any]:
    """Return the committed record of what the dialects disagree about today."""
    if not BASELINE_PATH.exists():
        return {"not_fixed": {}, "cross": []}
    return json.loads(BASELINE_PATH.read_text())


def _refused(format_name: str, case_id: str) -> bool:
    """Whether the dialect is expected to refuse this turn."""
    return case_id in REFUSED_REASON.get(format_name, set())


@pytest.mark.parametrize("case_id", list(_TURNS))
@pytest.mark.parametrize("format_name", _DIALECT_NAMES)
def test_a_same_format_stream_round_trip_is_a_fixed_point(
    format_name: str, case_id: str
) -> None:
    """A second pass through the dialect changes nothing the first pass produced."""
    if _refused(format_name, case_id):
        with pytest.raises(UnrepresentableFinishReason):
            drive(format_name, _events(case_id))
        return

    first = drive(format_name, _events(case_id))
    second = drive(format_name, first)

    known = _baseline()["not_fixed"]
    key = f"{format_name}/{case_id}"
    if first != second:
        assert key in known, (
            f"{format_name}/{case_id}: a second pass through the dialect "
            "changes what the first pass produced, and the baseline has no "
            "entry saying why"
        )
        return
    assert key not in known, (
        f"{format_name}/{case_id}: the baseline records this as not a fixed "
        "point, but it is now one — drop the entry"
    )


def _views_differ(a_name: str, b_name: str, case_id: str) -> set[str]:
    """Return the turn-view fields the detour through *b_name* changed."""
    first = drive(a_name, _events(case_id))
    middle = drive(b_name, first)
    again = drive(a_name, middle)
    before = asdict(_view_of_events(first))
    after = asdict(_view_of_events(again))
    return {field for field in _VIEW_FIELDS if before[field] != after[field]}


@pytest.mark.parametrize(
    ("a_name", "b_name"),
    [(a, b) for a in _DIALECT_NAMES for b in _DIALECT_NAMES if a != b],
    ids=_PAIR_IDS,
)
def test_a_cross_format_stream_round_trip_keeps_the_turn(
    a_name: str, b_name: str
) -> None:
    """What B carries has to come back, and what it does not is recorded."""
    key = f"{a_name}->{b_name}"
    baseline = _baseline()["cross"]
    known = {
        (entry["case"], entry["field"]) for entry in baseline if entry["pair"] == key
    }

    found: list[tuple[str, str]] = []
    for case_id in _TURNS:
        if _refused(a_name, case_id) or _refused(b_name, case_id):
            continue
        try:
            changed = _views_differ(a_name, b_name, case_id)
        except UnrepresentableFinishReason:
            continue
        found.extend((case_id, field) for field in sorted(changed))

    new = sorted(set(found) - known)
    assert not new, (
        f"{key} changes {len(new)} turn field(s) across the detour, with no "
        "entry saying why: " + "; ".join(f"{case}:{field}" for case, field in new)
    )

    stale = sorted(known - set(found))
    assert not stale, (
        f"{key}: the baseline claims {len(stale)} change(s) that no longer "
        "happen: " + "; ".join(f"{case}:{field}" for case, field in stale)
    )


def test_every_baseline_entry_says_why() -> None:
    """An entry without a reason hides a defect nobody decided to accept."""
    for entry in _baseline()["cross"]:
        assert entry["reason"] and entry["reason"] != _UNREVIEWED, entry
        assert entry["pair"] in _PAIR_IDS, entry
        assert entry["field"] in _VIEW_FIELDS, entry
    for key, reason in _baseline()["not_fixed"].items():
        assert reason and reason != _UNREVIEWED, key
        dialect, _, case_id = key.partition("/")
        assert dialect in _DIALECT_NAMES, key
        assert case_id in _TURNS, key


def _not_fixed_now(previous: dict[str, Any]) -> dict[str, str]:
    """Return the dialect/case pairs whose second pass is not a fixed point."""
    found: dict[str, str] = {}
    for format_name in _DIALECT_NAMES:
        for case_id in _TURNS:
            if _refused(format_name, case_id):
                continue
            first = drive(format_name, _events(case_id))
            if first != drive(format_name, first):
                key = f"{format_name}/{case_id}"
                found[key] = previous["not_fixed"].get(key, _UNREVIEWED)
    return dict(sorted(found.items()))


def _cross_now(reasons: dict[str, str]) -> list[dict[str, Any]]:
    """Return the turn-view fields each ordered pair changes today."""
    found: list[dict[str, Any]] = []
    for a_name in _DIALECT_NAMES:
        for b_name in _DIALECT_NAMES:
            if a_name == b_name:
                continue
            pair = f"{a_name}->{b_name}"
            for case_id in _TURNS:
                if _refused(a_name, case_id) or _refused(b_name, case_id):
                    continue
                try:
                    changed = _views_differ(a_name, b_name, case_id)
                except UnrepresentableFinishReason:
                    continue
                for field in sorted(changed):
                    marker = "|".join((pair, case_id, field))
                    found.append(
                        {
                            "pair": pair,
                            "case": case_id,
                            "field": field,
                            "reason": reasons.get(marker, _UNREVIEWED),
                        }
                    )
    return found


def write_baseline() -> Path:
    """Regenerate the baseline from the current behaviour.

    Run it only after reviewing what changed: the file is the reviewed
    statement of what the dialects disagree about, not a cache.
    """
    previous = _baseline()
    reasons = {
        "|".join((e["pair"], e["case"], e["field"])): e["reason"]
        for e in previous["cross"]
    }
    data = {
        "schema_version": 1,
        "definition": (
            "not_fixed records the dialect/case pairs whose second stream pass "
            "differs from the first.  cross records the turn-view fields a "
            "detour through the other dialect changes.  Both carry the reason "
            "each entry is still open; an entry that no longer happens fails "
            "the suite."
        ),
        "not_fixed": _not_fixed_now(previous),
        "cross": _cross_now(reasons),
    }
    BASELINE_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return BASELINE_PATH
