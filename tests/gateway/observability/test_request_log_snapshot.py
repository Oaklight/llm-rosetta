"""The store an open leaves behind, frozen, so the next slices fail if it moves.

``scripts/dev/migration_differential.py`` proves that two trees open a store
identically, but it is a script: it runs when someone remembers to run it, and
its claim lives in a PR message.  This is the same observation kept where a test
enforces it — one JSON snapshot of what an open leaves behind for every shape a
store can be found in, re-derived on every run, with nothing but the snapshot to
compare against.

The snapshot was recorded from ``master`` before the maintenance refactor, so
the expectation is the *old* code's behaviour and the tree under test is what
has to reproduce it.  Regenerate it deliberately, never to make a red test
green:

    python scripts/dev/migration_differential.py --keep --old origin/master
    env PYTHONPATH=<the kept workdir>/tree-old/src \\
        python tests/observability/test_request_log_snapshot.py

The record mode runs against whichever tree is importable, which is the point:
freeze the behaviour you mean to keep, then assert the new code still has it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import pytest

HARNESS = Path(__file__).parents[3] / "scripts" / "dev" / "migration_differential.py"
SNAPSHOT = Path(__file__).with_name("request_log_snapshot.json")

#: One test per state, taken from the snapshot itself: a state the snapshot does
#: not know about is a difference too, and the test below catches it.  Read
#: defensively so that recording it for the first time is possible at all, and
#: empty when there is none, which fails the coverage test rather than passing.
STATES = sorted(json.loads(SNAPSHOT.read_text())) if SNAPSHOT.exists() else []


def _harness():
    """The differential harness as an importable module, without running it."""
    spec = importlib.util.spec_from_file_location("migration_differential", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def observed() -> dict[str, Any]:
    """What opening a store leaves behind today, in the tree under test."""
    with tempfile.TemporaryDirectory() as root:
        return _harness().observe_states(Path(root) / "stores")


def _first_difference(actual: Any, expected: Any, path: str = "") -> str | None:
    """Where two observations part company, as a path rather than a 200-line diff."""
    if isinstance(actual, dict) and isinstance(expected, dict):
        for key in sorted(set(actual) | set(expected)):
            found = _first_difference(
                actual.get(key), expected.get(key), f"{path}.{key}"
            )
            if found is not None:
                return found
        return None
    if isinstance(actual, list) and isinstance(expected, list):
        if len(actual) != len(expected):
            return (
                f"{path or '<root>'}: {len(actual)} entries, expected {len(expected)}"
            )
        for index in range(len(actual)):
            found = _first_difference(
                actual[index], expected[index], f"{path}[{index}]"
            )
            if found is not None:
                return found
        return None
    if actual != expected:
        return f"{path or '<root>'}: {actual!r}, expected {expected!r}"
    return None


def test_the_snapshot_covers_exactly_the_states_the_harness_observes(observed):
    """A new state — another migration — has to be frozen on purpose."""
    assert sorted(observed) == STATES, (
        "the harness observes a different set of states than the snapshot "
        "records, so part of what an open does is unasserted: regenerate it"
    )


@pytest.mark.parametrize("state", STATES)
def test_the_snapshot_records_the_two_things_it_compares(observed, state):
    """A snapshot that captured nothing compares equal to nothing.

    Both halves are load-bearing and one of them was silently empty when this
    file was first recorded: an INFO line is not emitted at all unless its
    logger's level allows it, so a test process (root at WARNING) captured empty
    log lists and the log comparison asserted nothing.  A recorded fixture is
    evidence only while it holds something, so this asserts the *fixture* has
    content rather than only that the run agrees with it.
    """
    recorded = json.loads(SNAPSHOT.read_text())[state]
    for name, value in (
        ("the fixture", recorded),
        (f"the run in {state}", observed[state]),
    ):
        assert value["schema"]["objects"], f"{name} recorded no schema objects"
        assert value["schema"]["columns"], f"{name} recorded no columns"
        assert value["log_first_open"] or value["log_second_open"], (
            f"{name} recorded no log lines, so the log comparison asserts nothing"
        )


@pytest.mark.parametrize("state", STATES)
def test_the_open_leaves_the_store_the_snapshot_records(observed, state):
    difference = _first_difference(
        observed[state], json.loads(SNAPSHOT.read_text())[state]
    )
    assert difference is None, (
        f"opening a store in the {state!r} state no longer leaves what it left: "
        f"{difference}"
    )


def main(argv: list[str]) -> int:
    """Record the snapshot from the tree that is importable right now."""
    out = Path(argv[0]) if argv else SNAPSHOT
    with tempfile.TemporaryDirectory() as root:
        observed = _harness().observe_states(Path(root) / "stores")
    out.write_text(json.dumps(observed, indent=1, sort_keys=True) + "\n")
    print(f"recorded {len(observed)} states from {_module_path()} to {out}")
    return 0


def _module_path() -> str:
    import llm_rosetta_gateway.observability.persistence as module

    return module.__file__ or "?"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
