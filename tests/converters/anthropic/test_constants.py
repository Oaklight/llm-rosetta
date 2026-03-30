"""Tests for Anthropic converter constants."""

import pytest

from llm_rosetta.converters.anthropic._constants import (
    ANTHROPIC_REASON_FROM_PROVIDER,
    ANTHROPIC_REASON_TO_PROVIDER,
    AnthropicEventType,
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


class TestAnthropicReasonMaps:
    def test_from_provider_values_are_valid_ir_reasons(self):
        for provider_val, ir_val in ANTHROPIC_REASON_FROM_PROVIDER.items():
            assert ir_val in VALID_IR_REASONS, (
                f"'{provider_val}' maps to '{ir_val}' which is not a valid IR reason"
            )

    def test_to_provider_keys_are_valid_ir_reasons(self):
        for ir_val in ANTHROPIC_REASON_TO_PROVIDER:
            assert ir_val in VALID_IR_REASONS, (
                f"TO_PROVIDER key '{ir_val}' is not a valid IR reason"
            )

    def test_to_provider_covers_from_provider_values(self):
        from_values = set(ANTHROPIC_REASON_FROM_PROVIDER.values())
        to_keys = set(ANTHROPIC_REASON_TO_PROVIDER.keys())
        missing = from_values - to_keys
        assert not missing, (
            f"FROM_PROVIDER produces {missing} but TO_PROVIDER has no mapping for them"
        )

    def test_refusal_present_in_both_directions(self):
        assert "refusal" in ANTHROPIC_REASON_FROM_PROVIDER
        assert "refusal" in ANTHROPIC_REASON_TO_PROVIDER


class TestAnthropicEventType:
    @pytest.mark.parametrize(
        "attr",
        [
            "MESSAGE_START",
            "CONTENT_BLOCK_START",
            "CONTENT_BLOCK_DELTA",
            "CONTENT_BLOCK_STOP",
            "MESSAGE_DELTA",
            "MESSAGE_STOP",
        ],
    )
    def test_event_types_are_non_empty_strings(self, attr):
        value = getattr(AnthropicEventType, attr)
        assert isinstance(value, str) and value
