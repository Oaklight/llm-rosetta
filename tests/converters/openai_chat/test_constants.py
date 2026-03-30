"""Tests for OpenAI Chat converter constants."""

from llm_rosetta.converters.openai_chat._constants import (
    OPENAI_CHAT_REASON_FROM_PROVIDER,
    OPENAI_CHAT_REASON_TO_PROVIDER,
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


class TestOpenAIChatReasonMaps:
    def test_from_provider_values_are_valid_ir_reasons(self):
        for provider_val, ir_val in OPENAI_CHAT_REASON_FROM_PROVIDER.items():
            assert ir_val in VALID_IR_REASONS, (
                f"'{provider_val}' maps to '{ir_val}' which is not a valid IR reason"
            )

    def test_to_provider_keys_are_valid_ir_reasons(self):
        for ir_val in OPENAI_CHAT_REASON_TO_PROVIDER:
            assert ir_val in VALID_IR_REASONS, (
                f"TO_PROVIDER key '{ir_val}' is not a valid IR reason"
            )

    def test_to_provider_covers_from_provider_values(self):
        from_values = set(OPENAI_CHAT_REASON_FROM_PROVIDER.values())
        to_keys = set(OPENAI_CHAT_REASON_TO_PROVIDER.keys())
        missing = from_values - to_keys
        assert not missing, (
            f"FROM_PROVIDER produces {missing} but TO_PROVIDER has no mapping for them"
        )

    def test_identity_mapping(self):
        """TO_PROVIDER is an identity mapping since IR reasons match OpenAI Chat."""
        for key, val in OPENAI_CHAT_REASON_TO_PROVIDER.items():
            assert key == val
