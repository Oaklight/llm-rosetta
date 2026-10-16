"""Tests for the differential harness's comparison, not for its runs.

The harness proves something by running two trees against each other, and the
guard against a vacuous pass is its own same-tree refusal.  What is asserted
here is the part that decides *whether* two observations match — the masking
that keeps the run's clock and store path out of the comparison, and the
comparison itself — because a harness that quietly calls two different stores
identical is worse than no harness at all.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).parents[1] / "scripts" / "dev" / "migration_differential.py"
)


@pytest.fixture(scope="module")
def differential():
    """The harness as an importable module, without running its ``main``."""
    spec = importlib.util.spec_from_file_location("migration_differential", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state(*, columns: list[list], log_first: int = 2, log_second: int = 1) -> dict:
    return {
        "schema": {
            "objects": [["table", "request_log", "CREATE TABLE request_log (...)"]],
            "versions": [1, 2, 3, 4, 5, 6],
            "columns": columns,
            "indexes": ["idx_rl_status", "idx_rl_timestamp"],
        },
        "schema_version": 6,
        "log_first_open": [f"INFO line {index}" for index in range(log_first)],
        "log_second_open": [f"INFO line {index}" for index in range(log_second)],
    }


_COLUMNS = [["id", "TEXT", None], ["model", "TEXT", None]]
_WITH_EXTRA_COLUMN = [*_COLUMNS, ["smuggled", "TEXT", None]]


class TestTheMasking:
    """Two runs happen in different places at different times, on purpose."""

    def test_the_store_path_and_the_clock_are_masked(self, differential):
        masked = differential._normalise(
            {
                "log": [
                    "INFO Database backup created: "
                    "/tmp/run-abc/stores-old/fresh/backups/gateway-20261016-195051.db",
                    "INFO applied at 2026-10-16T19:50:51.482913+00:00",
                ]
            },
            Path("/tmp/run-abc/stores-old"),
        )
        assert masked == {
            "log": [
                "INFO Database backup created: "
                "<store-root>/fresh/backups/gateway-<ts>.db",
                "INFO applied at <ts>",
            ]
        }

    def test_what_is_not_the_environment_survives_masking(self, differential):
        """A missing mask would hide a difference; an over-eager one hides all."""
        masked = differential._normalise(
            _state(columns=_COLUMNS), Path("/tmp/run-abc/stores-old")
        )
        assert masked["schema"]["versions"] == [1, 2, 3, 4, 5, 6]
        assert masked["schema"]["columns"] == _COLUMNS


class TestTheComparison:
    """The verdict, and the report that has to be readable enough to trust."""

    def test_the_same_observation_is_a_match(self, differential, capsys):
        state = _state(columns=_COLUMNS)
        matched = differential._compare(
            {"states": {"fresh": state}}, {"states": {"fresh": state}}
        )
        assert matched is True
        assert "IDENTICAL" in capsys.readouterr().out

    def test_a_difference_in_one_column_is_a_difference(self, differential, capsys):
        """The whole point: a schema that moved one column must not pass."""
        matched = differential._compare(
            {"states": {"fresh": _state(columns=_WITH_EXTRA_COLUMN)}},
            {"states": {"fresh": _state(columns=_COLUMNS)}},
        )
        report = capsys.readouterr().out
        assert matched is False
        assert "DIFFERENT: fresh" in report
        assert "smuggled" in report, "the report has to say what changed"

    def test_a_difference_in_one_log_line_is_a_difference(self, differential, capsys):
        matched = differential._compare(
            {"states": {"at_3": _state(columns=_COLUMNS, log_second=4)}},
            {"states": {"at_3": _state(columns=_COLUMNS, log_second=2)}},
        )
        assert matched is False
        assert "DIFFERENT: at_3" in capsys.readouterr().out

    def test_a_state_only_one_tree_has_is_a_difference(self, differential, capsys):
        """Which is how a tree with another migration shows up."""
        matched = differential._compare(
            {"states": {"fresh": _state(columns=_COLUMNS)}},
            {
                "states": {
                    "fresh": _state(columns=_COLUMNS),
                    "at_6": _state(columns=_COLUMNS),
                }
            },
        )
        report = capsys.readouterr().out
        assert matched is False
        assert "at_6" in report
        assert "new tree only" in report

    def test_every_state_is_reported_not_only_the_first(self, differential, capsys):
        old = {"states": {name: _state(columns=_COLUMNS) for name in ("fresh", "at_1")}}
        new = {
            "states": {
                name: _state(columns=_WITH_EXTRA_COLUMN) for name in ("fresh", "at_1")
            }
        }
        assert differential._compare(old, new) is False
        report = capsys.readouterr().out
        assert "2 of 2 states" in report

    def test_a_state_summary_counts_both_opens(self, differential):
        """A fresh store works on the first open, a back-level one on the second."""
        summary = differential._summarise(_state(columns=_COLUMNS, log_first=12))
        assert "log=12/1" in summary
        assert "cols=2" in summary


class TestPickingTheTrees:
    """A tree is a checkout or an export; either way the package is in it."""

    def test_a_checkout_is_found_by_its_src_directory(self, differential, tmp_path):
        (tmp_path / "src" / "llm_rosetta").mkdir(parents=True)
        assert differential._source_root(tmp_path) == tmp_path / "src"

    def test_an_unpacked_package_is_found_too(self, differential, tmp_path):
        (tmp_path / "llm_rosetta").mkdir(parents=True)
        assert differential._source_root(tmp_path) == tmp_path

    def test_a_tree_without_the_package_is_not_a_tree(self, differential, tmp_path):
        assert differential._source_root(tmp_path) is None

    def test_a_tree_hashes_the_same_wherever_it_is_copied(self, differential, tmp_path):
        """The same-tree refusal must not be defeated by a different directory."""
        first = tmp_path / "one" / "observability"
        second = tmp_path / "two" / "elsewhere" / "observability"
        for root in (first, second):
            root.mkdir(parents=True)
            (root / "persistence.py").write_text("VERSION = 6\n")
            (root / "__pycache__").mkdir()
            (root / "__pycache__" / "persistence.cpython-310.pyc").write_bytes(b"junk")
        assert differential._hash_tree(first) == differential._hash_tree(second)

    def test_a_tree_that_changed_hashes_differently(self, differential, tmp_path):
        first = tmp_path / "one"
        second = tmp_path / "two"
        for root in (first, second):
            root.mkdir()
            (root / "persistence.py").write_text("VERSION = 6\n")
        (second / "persistence.py").write_text("VERSION = 7\n")
        assert differential._hash_tree(first) != differential._hash_tree(second)
