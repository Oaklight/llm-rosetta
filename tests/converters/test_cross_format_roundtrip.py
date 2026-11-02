"""Cross-format round trips on generated bodies — epic #1223, property 3.

``A → IR → B → IR → A`` must preserve the ``A ∩ B`` common subset, and every
path outside it must be named by a conversion warning.  The wire suite checks
that on one canonical body per format; this module generates the bodies from
each dialect's own declared request fields, reusing the generator in
:mod:`tests.converters.test_roundtrip_property`.

``A ∩ B`` is not declared anywhere, so it is defined here operationally.  For
the body generated from field *f* of *A* and an ordered pair ``(A, B)``, a path
of the IR is in the common subset when B's own round trip gives it back — that
is, when ``A → IR → B → IR`` leaves it unchanged.  The two halves of the
property then read:

- **No corruption.**  A path in the subset has to survive the return leg.  If
  the detour through B changes a value B claims to carry, B is at fault, and
  the case fails naming the field and the path.
- **No silent loss.**  A path outside the subset has to be explained by a
  conversion warning whose ``subject`` names it, exactly as the wire suite
  decides it.

What the dialects disagree about today is committed in
``cross_format_losses.json``.  A path that starts going silently missing fails
the case, and the corruptions are listed one by one with the issue that owns
each, so an unreviewed change cannot hide behind the baseline.

Converter-internal markers (``provider_extensions._owner`` and any other
``_``-prefixed key) are excluded from the comparison: they are bookkeeping the
readers stamp, not request data, and a detour through another dialect
re-stamps them by design.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.conversion_warnings import WarningSink, recording_to
from tests.converters.test_roundtrip_property import CASES, FORMATS, Format, _body
from tools.conformance.suites.wire import differences, explained

BASELINE_PATH = Path(__file__).with_name("cross_format_losses.json")

#: What :func:`write_baseline` writes for a corruption nobody has reviewed yet.
#: :func:`test_every_corruption_says_why` refuses it, so a fresh entry cannot
#: land by accident.
_UNREVIEWED = "UNREVIEWED"

_NAMES = list(FORMATS)
_PAIRS = [(a, b) for a in _NAMES for b in _NAMES if a != b]


def _read(format: Format, body: dict[str, Any]) -> tuple[Any, list[Any]]:
    """Read a provider body into the IR, collecting what the reader reports."""
    converter = format.converter()
    ctx = ConversionContext()
    sink = WarningSink(leg=format.name)
    ctx.warnings = sink
    with recording_to(sink):
        ir = converter.request_from_provider(body, context=ctx)
    return ir, list(sink)


def _write(format: Format, ir: Any) -> tuple[dict[str, Any], list[Any]]:
    """Write an IR request as a provider body, collecting what the writer reports."""
    converter = format.converter()
    ctx = ConversionContext()
    sink = WarningSink(leg=format.name)
    with recording_to(sink):
        body, _ = converter.request_to_provider(ir, context=ctx)
    return body, list(sink)


def _comparable(ir: Any) -> Any:
    """Return *ir* without the converter-internal extension markers.

    ``provider_extensions`` carries an ``_owner`` stamp naming the dialect the
    set was read for, plus other ``_``-prefixed keys that carry state between a
    converter's two halves.  None of them is request data, and a detour through
    another dialect re-stamps the set by design, so comparing them would report
    a change the IR is meant to make.
    """
    if not isinstance(ir, dict):
        return ir
    result = dict(ir)
    extensions = result.get("provider_extensions")
    if isinstance(extensions, dict):
        kept = {k: v for k, v in extensions.items() if not k.startswith("_")}
        if kept:
            result["provider_extensions"] = kept
        else:
            result.pop("provider_extensions")
    return result


def measure(a_name: str, b_name: str) -> tuple[Counter[str], set[tuple[str, str]]]:
    """Run every generated case of *a_name* through *b_name* and back.

    Args:
        a_name: The source dialect.
        b_name: The dialect the request detours through.

    Returns:
        How many bodies lost each silent path, and the ``(field, loss)`` pairs
        where the detour changed a path it carried.
    """
    a, b = FORMATS[a_name], FORMATS[b_name]
    silent: Counter[str] = Counter()
    corrupt: set[tuple[str, str]] = set()
    for name, field, value in CASES:
        if name != a_name:
            continue
        ir_a, w1 = _read(a, _body(a, field, value))
        body_b, w2 = _write(b, ir_a)
        ir_b, w3 = _read(b, body_b)
        body_a, w4 = _write(a, ir_b)
        ir_a_again, w5 = _read(a, body_a)

        before = _comparable(ir_a)
        lost_by_b = {
            loss.split(":")[0] for loss in differences(before, _comparable(ir_b))
        }
        warnings = [*w1, *w2, *w3, *w4, *w5]
        for loss in differences(before, _comparable(ir_a_again)):
            path = loss.split(":")[0]
            if path in lost_by_b:
                if not explained(loss, warnings):
                    silent[path] += 1
            else:
                corrupt.add((field, loss))
    return silent, corrupt


def _baseline() -> dict[str, Any]:
    """Return the committed record of what the dialects disagree about today."""
    if not BASELINE_PATH.exists():
        return {"silent": {}, "corruptions": []}
    return json.loads(BASELINE_PATH.read_text())


@pytest.mark.parametrize(("a", "b"), _PAIRS, ids=[f"{a}->{b}" for a, b in _PAIRS])
def test_a_cross_format_round_trip_keeps_the_common_subset(a: str, b: str) -> None:
    """What B carries has to come back, and what it does not has to be reported."""
    silent, corrupt = measure(a, b)
    key = f"{a}->{b}"
    baseline = _baseline()

    known_silent: dict[str, int] = baseline["silent"].get(key, {})
    known_corrupt = {
        (entry["field"], entry["loss"])
        for entry in baseline["corruptions"]
        if entry["pair"] == key
    }

    # A count, not just a path: a loss that spreads to more bodies is a
    # regression even though the path is already known.
    new_silent = sorted(
        f"{path} ({count} bodies, was {known_silent.get(path, 0)})"
        for path, count in silent.items()
        if count > known_silent.get(path, 0)
    )
    assert not new_silent, (
        f"{key} loses {len(new_silent)} IR path(s) with no conversion warning "
        "naming them: " + "; ".join(new_silent)
    )

    stale = sorted(path for path in known_silent if path not in silent)
    assert not stale, (
        f"{key}: the baseline claims {len(stale)} loss(es) that no longer "
        "happen, so it no longer says what the dialects disagree about: "
        + "; ".join(stale)
    )

    new_corrupt = sorted(corrupt - known_corrupt)
    assert not new_corrupt, (
        f"{key} changes what it claims to carry, on "
        f"{len(new_corrupt)} path(s): "
        + "; ".join(f"{field}: {loss}" for field, loss in new_corrupt)
    )


def test_every_format_contributes_cases() -> None:
    """A pair whose source generated no body would pass without testing anything."""
    for name in FORMATS:
        assert any(case[0] == name for case in CASES), name


def test_every_corruption_says_why() -> None:
    """A corruption in the baseline without a reason hides an unreviewed defect."""
    for entry in _baseline()["corruptions"]:
        assert entry["reason"], entry
        assert entry["reason"] != _UNREVIEWED, (
            f"{entry['pair']} {entry['field']}: the corruption is recorded but "
            "nobody said why it is still open"
        )
        assert entry["pair"] in {f"{a}->{b}" for a, b in _PAIRS}, entry


def test_the_baseline_names_live_pairs_only() -> None:
    """A baseline key for a pair that no longer exists would never apply."""
    pairs = {f"{a}->{b}" for a, b in _PAIRS}
    for key in _baseline()["silent"]:
        assert key in pairs, key


def write_baseline() -> Path:
    """Regenerate the baseline from the current behaviour.

    Run it only after reviewing what changed: the file is the reviewed
    statement of what the dialects disagree about, not a cache.
    """
    silent: dict[str, dict[str, int]] = {}
    corruptions: list[dict[str, str]] = []
    known = {
        "|".join((entry["pair"], entry["field"], entry["loss"])): entry
        for entry in _baseline()["corruptions"]
    }
    for a, b in _PAIRS:
        key = f"{a}->{b}"
        lost, changed = measure(a, b)
        if lost:
            silent[key] = dict(sorted(lost.items()))
        for field, loss in sorted(changed):
            marker = f"{key}|{field}|{loss}"
            previous = known.get(marker, {})
            corruptions.append(
                {
                    "pair": key,
                    "field": field,
                    "loss": loss,
                    "reason": previous.get("reason", _UNREVIEWED),
                }
            )
    data = {
        "schema_version": 2,
        "definition": (
            "A path is in A ∩ B when A → IR → B → IR gives it back unchanged.  "
            "A path outside it must be named by a conversion warning; each "
            "silent entry counts the bodies that lost it, and the corruptions "
            "are paths B carries and then changes, each with the reason it is "
            "still open."
        ),
        "silent_issue": "https://github.com/Oaklight/llm-rosetta/issues/1328",
        "silent": silent,
        "corruptions": corruptions,
    }
    BASELINE_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n")
    return BASELINE_PATH
