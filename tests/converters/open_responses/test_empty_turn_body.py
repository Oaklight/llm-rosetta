"""A finished Open Responses turn with no output keeps its finish reason (#1721)."""

from typing import Any

import pytest

from llm_rosetta.converters.llm.chat_completions import ChatCompletionsConverter
from llm_rosetta.converters.llm.open_responses import OpenResponsesConverter


def _read(body: dict[str, Any]) -> Any:
    return OpenResponsesConverter().response_from_provider(body)


class TestEmptyTurnBody:
    def test_empty_ir_turn_round_trips_with_its_finish_reason(self):
        converter = OpenResponsesConverter()
        ir: Any = {
            "id": "turn1",
            "object": "response",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": []},
                    "finish_reason": {"reason": "stop"},
                }
            ],
        }
        back = converter.response_from_provider(converter.response_to_provider(ir))
        assert len(back["choices"]) == 1
        assert back["choices"][0]["message"]["content"] == []
        assert back["choices"][0]["finish_reason"] == {"reason": "stop"}

    @pytest.mark.parametrize(
        ("extra", "reason"),
        [
            ({"status": "completed"}, "stop"),
            (
                {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "max_output_tokens"},
                },
                "length",
            ),
            (
                {
                    "status": "incomplete",
                    "incomplete_details": {"reason": "content_filter"},
                },
                "content_filter",
            ),
            (
                {"status": "incomplete", "incomplete_details": {"reason": "other"}},
                "length",
            ),
            (
                {"status": "incomplete", "incomplete_details": {"reason": None}},
                "length",
            ),
            ({"status": "failed"}, "error"),
            ({"status": "cancelled"}, "cancelled"),
        ],
        ids=[
            "completed",
            "incomplete-max-output-tokens",
            "incomplete-content-filter",
            "incomplete-unknown-reason",
            "incomplete-null-reason",
            "failed",
            "cancelled",
        ],
    )
    def test_a_finished_body_with_empty_output_has_one_choice(self, extra, reason):
        back = _read({"id": "r", "model": "m", "output": [], **extra})
        assert len(back["choices"]) == 1
        assert back["choices"][0]["message"]["content"] == []
        assert back["choices"][0]["finish_reason"] == {"reason": reason}

    @pytest.mark.parametrize("status", ["in_progress", "queued"])
    def test_response_still_in_flight_has_no_choice(self, status):
        back = _read({"id": "r", "status": status, "output": [], "model": "m"})
        assert back["choices"] == []

    @pytest.mark.parametrize(
        "extra",
        [
            {},
            {"status": "no-such-status"},
            {"status": 0},
            {"status": True},
            {"status": ["completed"]},
            {"status": {"state": "completed"}},
        ],
        ids=["missing", "unknown", "zero", "true", "list", "dict"],
    )
    def test_pin_a_missing_or_unknown_status_reads_as_stop(self, extra):
        """PIN, not a rule: the spec makes ``status`` required, so a body
        without it, or with a value outside the spec (a non-string included), is
        outside the contract.  Today it reads as a finished ``stop`` turn and
        never raises."""
        back = _read({"id": "r", "model": "m", "output": [], **extra})
        assert [c["finish_reason"] for c in back["choices"]] == [{"reason": "stop"}]

    def test_the_empty_completed_turn_writes_as_chat_stop_with_empty_content(self):
        ir = _read({"id": "r", "status": "completed", "output": [], "model": "m"})
        out = ChatCompletionsConverter().response_to_provider(ir)
        (choice,) = out["choices"]
        assert choice["finish_reason"] == "stop"
        assert choice["message"]["content"] == ""

    @pytest.mark.parametrize("reason", ["tool_calls", "refusal"])
    def test_pin_an_empty_turn_with_a_reason_the_wire_cannot_carry_reads_as_stop(
        self, reason
    ):
        """PIN, not a rule: Open Responses has no status for ``tool_calls`` and
        no field for a refusal with no words, and an empty turn has neither a
        call nor a refusal part to carry them.  The IR reason is lost and the
        body reads back as ``stop``."""
        converter = OpenResponsesConverter()
        ir: Any = {
            "id": "turn1",
            "object": "response",
            "created": 1,
            "model": "m",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": []},
                    "finish_reason": {"reason": reason},
                }
            ],
        }
        back = converter.response_from_provider(converter.response_to_provider(ir))
        assert [c["finish_reason"] for c in back["choices"]] == [{"reason": "stop"}]
