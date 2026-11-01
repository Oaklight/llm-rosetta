"""Tests for the ``request-fields`` conformance suite.

The suite's contract: every request field a format declares is in the ledger,
and a field the read leg drops says why.  Additions fail (an upstream field is
a reviewed change); removals are reported (a deprecation does not mean our
behaviour changed).  The class itself is a reviewed statement, not something
the suite can derive, so a new path starts ``unclassified`` and fails until a
human reads the reader.
"""

from __future__ import annotations

from typing import Any

import pytest

# The suite reads provider SDK types.  `dev-extras-test` installs the `[dev]`
# extra alone, so the SDKs are absent there; the `conformance` job installs
# `[all]` and is where this file actually runs.
pytest.importorskip("openai")
pytest.importorskip("anthropic")
pytest.importorskip("google.genai")

from tools.conformance import ProbeFailure
from tools.conformance.suites import SUITES
from tools.conformance.suites.request_fields import (
    _CLASSIFICATIONS,
    _INTERACTIONS_FIELDS,
    _REASON_REQUIRED,
    _SKIP,
    _RequestFieldsProbe,
    build,
    inventory,
    read_baseline,
)


def _baseline() -> dict[str, Any]:
    return read_baseline()


def _probe(
    current: dict[str, Any],
    baseline: dict[str, Any],
    format_name: str = "chat_completions",
) -> _RequestFieldsProbe:
    return _RequestFieldsProbe(format_name, current, baseline)


def _with_fields(baseline: dict[str, Any], name: str, fields: dict) -> dict[str, Any]:
    return {
        **baseline,
        "formats": {
            **baseline["formats"],
            name: {**baseline["formats"][name], "fields": fields},
        },
    }


class TestTheCommittedLedger:
    def test_every_format_is_present(self):
        assert set(_baseline()["formats"]) == set(inventory())

    def test_every_field_is_classified(self):
        for format_name, data in _baseline()["formats"].items():
            for path, entry in data["fields"].items():
                assert entry["class"] in _CLASSIFICATIONS, (format_name, path)
                if entry["class"] in _REASON_REQUIRED:
                    assert entry["reason"], (format_name, path)

    def test_the_suite_passes_on_the_committed_ledger(self):
        probes, _env = build()
        for probe in probes:
            if probe.name == "inventory":
                raise AssertionError("the inventory could not be built")
            probe.run()

    def test_the_ledger_records_both_outcomes(self):
        """A ledger with no ``dropped`` entry would not be recording losses."""
        for format_name, data in _baseline()["formats"].items():
            classes = {entry["class"] for entry in data["fields"].values()}
            assert "dropped" in classes, format_name


class TestTheSkipTable:
    def test_every_skip_path_is_walked(self):
        """A stale skip path silently stops applying when an SDK renames a field."""
        walked = inventory()
        for format_name, skipped in _SKIP.items():
            for path in skipped:
                assert path in walked[format_name]["fields"], (format_name, path)

    def test_every_skip_entry_carries_a_legal_class(self):
        for format_name, skipped in _SKIP.items():
            for path, classification in skipped.items():
                assert classification in _CLASSIFICATIONS, (format_name, path)


class TestDetectingChange:
    def test_a_new_field_fails(self):
        baseline = _baseline()
        current = inventory()["chat_completions"]
        current = {**current, "fields": {**current["fields"], "brand_new": "brand_new"}}

        with pytest.raises(ProbeFailure, match="new request fields"):
            _probe(current, baseline).run()

    def test_a_removed_field_is_reported_not_failed(self):
        baseline = _baseline()
        current = inventory()["chat_completions"]
        dropped = next(iter(current["fields"]))
        fields = {k: v for k, v in current["fields"].items() if k != dropped}
        current = {**current, "fields": fields}

        result = _probe(current, baseline).run()

        assert dropped in result["removed"]

    def test_an_unclassified_field_fails(self):
        baseline = _baseline()
        current = inventory()["chat_completions"]
        recorded = dict(baseline["formats"]["chat_completions"]["fields"])
        path = next(iter(recorded))
        recorded[path] = {"class": "unclassified", "reason": ""}

        with pytest.raises(ProbeFailure, match="unclassified"):
            _probe(current, _with_fields(baseline, "chat_completions", recorded)).run()

    def test_a_dropped_field_without_a_reason_fails(self):
        baseline = _baseline()
        current = inventory()["chat_completions"]
        recorded = dict(baseline["formats"]["chat_completions"]["fields"])
        path = next(
            path for path, entry in recorded.items() if entry["class"] == "dropped"
        )
        recorded[path] = {"class": "dropped", "reason": ""}

        with pytest.raises(ProbeFailure, match="needs a reason"):
            _probe(current, _with_fields(baseline, "chat_completions", recorded)).run()


def test_a_missing_ledger_fails():
    """A repository state, not an environment property: never a skip."""
    current = inventory()["chat_completions"]

    with pytest.raises(ProbeFailure, match="no ledger"):
        _RequestFieldsProbe("chat_completions", current, {}).run()


def test_a_format_missing_from_the_ledger_fails():
    current = inventory()["chat_completions"]
    baseline = read_baseline()
    baseline = {
        **baseline,
        "formats": {
            name: data
            for name, data in baseline["formats"].items()
            if name != "chat_completions"
        },
    }

    with pytest.raises(ProbeFailure, match="format missing"):
        _probe(current, baseline).run()


def test_the_cli_registers_the_suite():
    assert "request-fields" in SUITES


def test_the_interactions_surface_is_hand_authored():
    formats = inventory()

    assert formats["google_interactions"]["source"] == "no_sdk"
    assert "generation_config.tool_choice" in _INTERACTIONS_FIELDS
