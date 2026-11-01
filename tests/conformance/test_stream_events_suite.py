"""Tests for the ``stream-events`` conformance suite.

The suite's contract: one canonical IR stream goes through every chat dialect
and the ledger records what came back.  A cell that changes fails the case —
a recovered type that disappears is a loss, and one that appears is an
improvement worth recording.
"""

from __future__ import annotations

from typing import Any

import pytest

# The suite drives the converters, which import no provider SDK — but the
# package's `fields` module does, and `dev-extras-test` installs `[dev]` alone.
pytest.importorskip("openai")
pytest.importorskip("anthropic")
pytest.importorskip("google.genai")

from tools.conformance import ProbeFailure
from tools.conformance.suites import SUITES
from tools.conformance.suites.stream_events import (
    SEQUENCE,
    _StreamEventsProbe,
    inventory,
    observe,
    read_baseline,
)


def _baseline() -> dict[str, Any]:
    return read_baseline()


def _probe(
    current: dict[str, Any],
    baseline: dict[str, Any],
    format_name: str = "chat_completions",
) -> _StreamEventsProbe:
    return _StreamEventsProbe(format_name, current, baseline)


class TestTheCommittedLedger:
    def test_every_dialect_is_present(self):
        assert set(_baseline()["formats"]) == set(inventory())

    def test_the_suite_passes_on_the_committed_ledger(self):
        probes, _env = build_probes()
        for probe in probes:
            if probe.name == "inventory":
                raise AssertionError("the inventory could not be built")
            probe.run()

    def test_every_dialect_recovers_something(self):
        """A dialect that recovers nothing would make its row vacuous."""
        for name, data in _baseline()["formats"].items():
            assert data["recovered"], name


class TestTheSequence:
    def test_the_sequence_covers_every_ir_event_type(self):
        from llm_rosetta.types.ir import stream

        declared = {
            literal
            for name in dir(stream)
            if name.endswith("Event")
            for literal in _event_type_literals(getattr(stream, name))
        }
        assert {event["type"] for event in SEQUENCE} == declared - {
            "provider_passthrough"
        }

    def test_observe_is_deterministic(self):
        """The same dialect observed twice gives the same answer."""
        assert observe("messages") == observe("messages")


class TestDetectingChange:
    def test_a_lost_type_fails(self):
        baseline = _baseline()
        current = inventory()["chat_completions"]
        current = {
            **current,
            "recovered": [t for t in current["recovered"] if t != "text_delta"],
        }

        with pytest.raises(ProbeFailure, match="lost"):
            _probe(current, baseline).run()

    def test_an_added_type_fails(self):
        baseline = _baseline()
        current = inventory()["chat_completions"]
        current = {**current, "recovered": [*current["recovered"], "tool_result"]}

        with pytest.raises(ProbeFailure, match="added"):
            _probe(current, baseline).run()

    def test_a_changed_chunk_count_fails(self):
        baseline = _baseline()
        current = inventory()["chat_completions"]
        current = {**current, "chunks": current["chunks"] + 1}

        with pytest.raises(ProbeFailure, match="chunk count"):
            _probe(current, baseline).run()

    def test_a_reordered_sequence_fails(self):
        baseline = _baseline()
        current = inventory()["chat_completions"]
        current = {**current, "recovered": list(reversed(current["recovered"]))}

        with pytest.raises(ProbeFailure, match="order changed"):
            _probe(current, baseline).run()

    def test_a_missing_dialect_fails(self):
        with pytest.raises(ProbeFailure, match="format missing"):
            _probe(inventory()["chat_completions"], {}).run()


def test_the_cli_registers_the_suite():
    assert "stream-events" in SUITES


def _event_type_literals(cls: Any) -> set[str]:
    """Return the ``type`` literals a TypedDict declares, if any."""
    import typing

    annotation = getattr(cls, "__annotations__", {}).get("type")
    while annotation is not None:
        if typing.get_origin(annotation) is typing.Literal:
            return {arg for arg in typing.get_args(annotation) if isinstance(arg, str)}
        args = typing.get_args(annotation)
        if not args:
            return set()
        # Unwrap ``Required[Literal[...]]`` / ``NotRequired[Literal[...]]``.
        annotation = args[0]
    return set()


def build_probes():
    from tools.conformance.suites.stream_events import build

    return build()
