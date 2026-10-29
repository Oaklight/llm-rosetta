"""Tests for the ``fields`` conformance suite.

The suite's contract: every response field the SDK declares is in the ledger,
and a field that is not comparable carries a reason.  Additions fail (an
upstream field is a reviewed change); removals are reported (a deprecation
does not mean our behaviour changed).
"""

from __future__ import annotations

from typing import Any

import pytest

from tools.conformance import ProbeFailure
from tools.conformance.suites import SUITES
from tools.conformance.suites.fields import (
    _FieldsProbe,
    _sdk_versions,
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
) -> _FieldsProbe:
    return _FieldsProbe(format_name, current, baseline, _sdk_versions())


class TestTheCommittedLedger:
    def test_every_format_is_present(self):
        assert set(_baseline()["formats"]) == set(inventory())

    def test_every_field_is_classified(self):
        for format_name, data in _baseline()["formats"].items():
            for path, entry in data["fields"].items():
                assert entry["class"] != "unclassified", (format_name, path)
                if entry["class"] in {"provider_only", "not_compared"}:
                    assert entry["reason"], (format_name, path)

    def test_every_variant_is_classified(self):
        for format_name, data in _baseline()["formats"].items():
            for path, entries in data["variants"].items():
                for name, entry in entries.items():
                    assert entry["class"] != "unclassified", (format_name, path, name)
                    if entry["class"] == "not_compared":
                        assert entry["reason"], (format_name, path, name)

    def test_the_suite_passes_on_the_committed_ledger(self):
        probes, _env = build()
        for probe in probes:
            if probe.name == "inventory":
                raise AssertionError("the inventory could not be built")
            probe.run()


class TestDetectingChange:
    def test_a_new_field_fails(self):
        baseline = _baseline()
        current = inventory()["chat_completions"]
        current = {**current, "fields": {**current["fields"], "brand_new": "brand_new"}}

        with pytest.raises(ProbeFailure, match="new fields"):
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
        path = next(iter(baseline["formats"]["chat_completions"]["fields"]))
        fields = dict(baseline["formats"]["chat_completions"]["fields"])
        fields[path] = {"class": "unclassified", "reason": ""}
        baseline = {
            **baseline,
            "formats": {
                **baseline["formats"],
                "chat_completions": {
                    **baseline["formats"]["chat_completions"],
                    "fields": fields,
                },
            },
        }

        with pytest.raises(ProbeFailure, match="unclassified"):
            _probe(current, baseline).run()

    def test_a_provider_only_field_without_a_reason_fails(self):
        baseline = _baseline()
        current = inventory()["chat_completions"]
        recorded = baseline["formats"]["chat_completions"]["fields"]
        path = next(
            path
            for path, entry in recorded.items()
            if entry["class"] == "provider_only"
        )
        fields = dict(recorded)
        fields[path] = {"class": "provider_only", "reason": ""}
        baseline = {
            **baseline,
            "formats": {
                **baseline["formats"],
                "chat_completions": {
                    **baseline["formats"]["chat_completions"],
                    "fields": fields,
                },
            },
        }

        with pytest.raises(ProbeFailure, match="needs a reason"):
            _probe(current, baseline).run()

    def test_a_new_variant_fails(self):
        baseline = _baseline()
        current = inventory()["open_responses"]
        variants = {
            **current["variants"],
            "output[]": [*current["variants"]["output[]"], "BrandNewItem"],
        }
        current = {**current, "variants": variants}

        with pytest.raises(ProbeFailure, match="new variants"):
            _probe(current, baseline, "open_responses").run()


def test_the_cli_registers_the_suite():
    assert "fields" in SUITES


def test_the_interactions_schema_is_hand_authored():
    formats = inventory()

    assert formats["google_interactions"]["source"] == "no_sdk"
    assert "steps[].type" in formats["google_interactions"]["fields"]
