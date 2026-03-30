"""Tests for Google GenAI converter constants."""

from llm_rosetta.converters.google_genai._constants import (
    GOOGLE_REASON_FROM_PROVIDER,
    GOOGLE_REASON_TO_PROVIDER,
    generate_tool_call_id,
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


class TestGoogleReasonMaps:
    def test_from_provider_values_are_valid_ir_reasons(self):
        for provider_val, ir_val in GOOGLE_REASON_FROM_PROVIDER.items():
            assert ir_val in VALID_IR_REASONS, (
                f"'{provider_val}' maps to '{ir_val}' which is not a valid IR reason"
            )

    def test_to_provider_keys_are_valid_ir_reasons(self):
        for ir_val in GOOGLE_REASON_TO_PROVIDER:
            assert ir_val in VALID_IR_REASONS, (
                f"TO_PROVIDER key '{ir_val}' is not a valid IR reason"
            )

    def test_to_provider_covers_from_provider_values(self):
        from_values = set(GOOGLE_REASON_FROM_PROVIDER.values())
        to_keys = set(GOOGLE_REASON_TO_PROVIDER.keys())
        missing = from_values - to_keys
        assert not missing, (
            f"FROM_PROVIDER produces {missing} but TO_PROVIDER has no mapping for them"
        )


class TestGenerateToolCallId:
    def test_starts_with_call_prefix(self):
        result = generate_tool_call_id()
        assert result.startswith("call_")

    def test_has_correct_length(self):
        result = generate_tool_call_id()
        # "call_" (5 chars) + 24 hex chars = 29 chars
        assert len(result) == 29

    def test_unique_ids(self):
        ids = {generate_tool_call_id() for _ in range(100)}
        assert len(ids) == 100
