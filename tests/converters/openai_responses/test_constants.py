"""Tests for OpenAI Responses converter constants."""

import pytest

from llm_rosetta.converters.openai_responses._constants import (
    RESPONSES_INCOMPLETE_REASON_TO_IR,
    RESPONSES_REASON_TO_INCOMPLETE_REASON,
    RESPONSES_REASON_TO_STATUS,
    RESPONSES_STATUS_TO_REASON,
    ResponsesEventType,
    generate_message_id,
)

# Valid IR finish reason values (from types/ir/response.py FinishReason)
VALID_IR_REASONS = {
    "stop",
    "length",
    "tool_calls",
    "content_filter",
    "refusal",
    "error",
    "cancelled",
}


class TestResponsesStatusMaps:
    def test_status_to_reason_values_are_valid_ir_reasons(self):
        for status, ir_val in RESPONSES_STATUS_TO_REASON.items():
            assert ir_val in VALID_IR_REASONS, (
                f"status '{status}' maps to '{ir_val}' which is not a valid IR reason"
            )

    def test_incomplete_reason_values_are_valid_ir_reasons(self):
        for inc_reason, ir_val in RESPONSES_INCOMPLETE_REASON_TO_IR.items():
            assert ir_val in VALID_IR_REASONS, (
                f"incomplete reason '{inc_reason}' maps to '{ir_val}' "
                "which is not a valid IR reason"
            )

    def test_reason_to_status_keys_are_valid_ir_reasons(self):
        for ir_val in RESPONSES_REASON_TO_STATUS:
            assert ir_val in VALID_IR_REASONS, (
                f"REASON_TO_STATUS key '{ir_val}' is not a valid IR reason"
            )

    def test_round_trip_status_to_reason_to_status(self):
        """Simple statuses round-trip correctly."""
        for status, ir_reason in RESPONSES_STATUS_TO_REASON.items():
            back = RESPONSES_REASON_TO_STATUS.get(ir_reason)
            assert back == status, (
                f"status '{status}' -> IR '{ir_reason}' -> status '{back}' "
                f"(expected '{status}')"
            )

    def test_round_trip_incomplete_reason(self):
        """Incomplete reasons round-trip: incomplete_details.reason -> IR -> status + incomplete_details.reason."""
        for inc_reason, ir_reason in RESPONSES_INCOMPLETE_REASON_TO_IR.items():
            status = RESPONSES_REASON_TO_STATUS.get(ir_reason)
            assert status == "incomplete", (
                f"IR reason '{ir_reason}' should map to 'incomplete' status, got '{status}'"
            )
            back = RESPONSES_REASON_TO_INCOMPLETE_REASON.get(ir_reason)
            assert back == inc_reason, (
                f"IR reason '{ir_reason}' -> incomplete_reason '{back}' "
                f"(expected '{inc_reason}')"
            )


class TestResponsesEventType:
    @pytest.mark.parametrize(
        "attr",
        [
            "RESPONSE_CREATED",
            "RESPONSE_COMPLETED",
            "RESPONSE_FAILED",
            "OUTPUT_ITEM_ADDED",
            "OUTPUT_ITEM_DONE",
            "CONTENT_PART_ADDED",
            "CONTENT_PART_DONE",
            "OUTPUT_TEXT_DELTA",
            "OUTPUT_TEXT_DONE",
            "REASONING_SUMMARY_TEXT_DELTA",
            "FUNCTION_CALL_ARGS_DELTA",
            "FUNCTION_CALL_ARGS_DONE",
        ],
    )
    def test_event_types_are_non_empty_strings(self, attr):
        value = getattr(ResponsesEventType, attr)
        assert isinstance(value, str) and value

    def test_all_event_types_start_with_response_prefix(self):
        for attr in dir(ResponsesEventType):
            if attr.startswith("_"):
                continue
            value = getattr(ResponsesEventType, attr)
            assert value.startswith("response."), (
                f"{attr} = '{value}' does not start with 'response.'"
            )


class TestGenerateMessageId:
    def test_with_response_id(self):
        assert generate_message_id("resp_123") == "msg_resp_123"

    def test_with_empty_response_id(self):
        assert generate_message_id("") == "msg_"
