"""An interaction still in progress reads as no choice (#1739)."""

from typing import Any

import pytest

from llm_rosetta.conversion_warnings import WarningSink, recording_to
from llm_rosetta.converters.llm.google_interactions import GoogleInteractionsConverter

_STEPS: list[dict[str, Any]] = [
    {"type": "model_output", "content": [{"type": "text", "text": "partial"}]}
]


def _read(status: str, steps: list[dict[str, Any]]) -> Any:
    body = {"id": "i1", "model": "m", "status": status, "steps": steps}
    return GoogleInteractionsConverter().response_from_provider(body)


@pytest.mark.parametrize("steps", [[], _STEPS], ids=["no-steps", "with-steps"])
def test_an_in_progress_interaction_has_no_choice(steps) -> None:
    """It has no terminal condition, and the IR choice requires a finish reason."""
    assert _read("in_progress", steps)["choices"] == []


def test_an_in_progress_interaction_reports_the_steps_it_does_not_read() -> None:
    sink = WarningSink(leg="response")
    with recording_to(sink):
        _read("in_progress", _STEPS)
    assert [(w.subject, w.code) for w in sink] == [("steps", "dropped")]
    assert "1 step(s)" in sink[0]

    quiet = WarningSink(leg="response")
    with recording_to(quiet):
        _read("in_progress", [])
    assert list(quiet) == []


@pytest.mark.parametrize(
    ("status", "reason"),
    [
        ("completed", "stop"),
        ("requires_action", "tool_calls"),
        ("incomplete", "length"),
        ("failed", "error"),
        ("cancelled", "cancelled"),
    ],
)
def test_every_other_status_keeps_its_choice(status, reason) -> None:
    choices = _read(status, _STEPS)["choices"]
    assert [c["finish_reason"]["reason"] for c in choices] == [reason]


def test_an_in_progress_interaction_comes_back_in_progress() -> None:
    """The writer does not turn an in-flight source into ``completed`` (#1739)."""
    converter = GoogleInteractionsConverter()
    ir = _read("in_progress", _STEPS)
    assert ir["in_flight"] is True
    back = converter.response_to_provider(ir)
    assert (back["status"], back["steps"]) == ("in_progress", [])


def test_an_empty_choice_list_is_not_taken_for_an_in_flight_response() -> None:
    """Chat can answer with no choice; that is not a response still running."""
    ir: Any = {"id": "c", "object": "response", "created": 1, "model": "m"}
    ir["choices"] = []
    back = GoogleInteractionsConverter().response_to_provider(ir)
    assert back["status"] == "completed"
