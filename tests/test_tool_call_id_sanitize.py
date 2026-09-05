"""Tests for tool_call_id sanitization."""

import re


from llm_rosetta.converters.base.helpers.tool_call_id import (
    MAX_TOOL_CALL_ID_LENGTH,
    sanitize_tool_call_id,
)


class TestSanitizeToolCallId:
    """Unit tests for the sanitize_tool_call_id function."""

    def test_clean_id_unchanged(self):
        assert sanitize_tool_call_id("call_abc123") == "call_abc123"

    def test_anthropic_id_unchanged(self):
        assert (
            sanitize_tool_call_id("toolu_vrtx_017Gnb9udfcY5kqjdAUkence")
            == "toolu_vrtx_017Gnb9udfcY5kqjdAUkence"
        )

    def test_openai_id_unchanged(self):
        assert (
            sanitize_tool_call_id("call_NTEBXqjyLLvYusbiGf1fXZv2")
            == "call_NTEBXqjyLLvYusbiGf1fXZv2"
        )

    def test_dashes_preserved(self):
        assert sanitize_tool_call_id("call-abc-123") == "call-abc-123"

    def test_newline_replaced(self):
        result = sanitize_tool_call_id("abc\ndef")
        assert "\n" not in result
        assert result == "abc_def"

    def test_spaces_replaced(self):
        assert sanitize_tool_call_id("abc def") == "abc_def"

    def test_dots_replaced(self):
        assert sanitize_tool_call_id("abc.def") == "abc_def"

    def test_empty_string(self):
        assert sanitize_tool_call_id("") == ""

    def test_all_invalid_chars(self):
        result = sanitize_tool_call_id("!@#$%")
        assert re.match(r"^[a-zA-Z0-9_-]+$", result)
        assert result == "_____"

    def test_exactly_64_chars(self):
        raw = "a" * 64
        assert sanitize_tool_call_id(raw) == raw

    def test_65_chars_truncated(self):
        raw = "a" * 65
        result = sanitize_tool_call_id(raw)
        assert len(result) == 64
        assert re.match(r"^[a-zA-Z0-9_-]+$", result)

    def test_long_id_truncated_with_hash(self):
        raw = "a" * 100
        result = sanitize_tool_call_id(raw)
        assert len(result) == MAX_TOOL_CALL_ID_LENGTH
        assert result.startswith("a" * 55)
        assert "_" in result[55:]

    def test_deterministic(self):
        raw = "call-3898c698-4c99-4803-87b7-217020159648-0\nfc_d9dc660e-7d55-9b7f-af1e-f62010e084f7_0"
        assert sanitize_tool_call_id(raw) == sanitize_tool_call_id(raw)

    def test_brettin_id_valid(self):
        """The real-world broken ID from the bug report."""
        raw = "call-3898c698-4c99-4803-87b7-217020159648-0\nfc_d9dc660e-7d55-9b7f-af1e-f62010e084f7_0"
        result = sanitize_tool_call_id(raw)
        assert len(result) <= 64
        assert re.match(r"^[a-zA-Z0-9_-]+$", result)

    def test_different_long_ids_produce_different_results(self):
        a = "a" * 100
        b = "b" * 100
        assert sanitize_tool_call_id(a) != sanitize_tool_call_id(b)

    def test_custom_max_length(self):
        raw = "a" * 50
        result = sanitize_tool_call_id(raw, max_length=30)
        assert len(result) == 30

    def test_tool_call_and_result_match(self):
        """Tool call and tool result with the same raw ID produce the same sanitized ID."""
        raw = "call-x\nfc_y"
        assert sanitize_tool_call_id(raw) == sanitize_tool_call_id(raw)


class TestSanitizationInConverters:
    """Integration tests: verify sanitized IDs flow through converter output."""

    def test_openai_chat_tool_call_sanitized(self):
        from llm_rosetta.converters.openai_chat.tool_ops import OpenAIChatToolOps

        ir_tool_call = {
            "type": "tool_call",
            "tool_call_id": "bad\nid_with_newline",
            "tool_name": "test_fn",
            "tool_input": {"x": 1},
            "tool_type": "function",
        }
        result = OpenAIChatToolOps.ir_tool_call_to_p(ir_tool_call)  # ty: ignore[invalid-argument-type]
        assert "\n" not in result["id"]
        assert re.match(r"^[a-zA-Z0-9_-]+$", result["id"])

    def test_openai_chat_tool_result_sanitized(self):
        from llm_rosetta.converters.openai_chat.tool_ops import OpenAIChatToolOps

        ir_tool_result = {
            "type": "tool_result",
            "tool_call_id": "bad\nid",
            "result": "ok",
        }
        result = OpenAIChatToolOps.ir_tool_result_to_p(ir_tool_result)  # ty: ignore[invalid-argument-type]
        assert "\n" not in result["tool_call_id"]

    def test_anthropic_tool_call_sanitized(self):
        from llm_rosetta.converters.anthropic.tool_ops import AnthropicToolOps

        ir_tool_call = {
            "type": "tool_call",
            "tool_call_id": "too-long-" * 10,
            "tool_name": "test_fn",
            "tool_input": {"x": 1},
            "tool_type": "function",
        }
        result = AnthropicToolOps.ir_tool_call_to_p(ir_tool_call)  # ty: ignore[invalid-argument-type]
        assert len(result["id"]) <= 64
        assert re.match(r"^[a-zA-Z0-9_-]+$", result["id"])

    def test_anthropic_tool_result_sanitized(self):
        from llm_rosetta.converters.anthropic.tool_ops import AnthropicToolOps

        ir_tool_result = {
            "type": "tool_result",
            "tool_call_id": "has spaces and.dots",
            "result": [{"type": "text", "text": "ok"}],
        }
        result = AnthropicToolOps.ir_tool_result_to_p(ir_tool_result)  # ty: ignore[invalid-argument-type]
        assert result["tool_use_id"] == "has_spaces_and_dots"

    def test_openai_chat_tool_call_result_ids_match(self):
        """Tool call and result must produce matching sanitized IDs."""
        from llm_rosetta.converters.openai_chat.tool_ops import OpenAIChatToolOps

        raw_id = "call-bad\nid-0\nfc_other_0"

        call_result = OpenAIChatToolOps.ir_tool_call_to_p(
            {
                "type": "tool_call",
                "tool_call_id": raw_id,
                "tool_name": "fn",
                "tool_input": {},
                "tool_type": "function",
            }
        )
        result_result = OpenAIChatToolOps.ir_tool_result_to_p(
            {
                "type": "tool_result",
                "tool_call_id": raw_id,
                "result": "ok",
            }
        )
        assert call_result["id"] == result_result["tool_call_id"]

    def test_google_generate_tool_call_sanitized(self):
        from llm_rosetta.converters.google_generate.tool_ops import (
            GoogleGenerateToolOps,
        )

        ir_tool_call = {
            "type": "tool_call",
            "tool_call_id": "bad\nid_with_newline",
            "tool_name": "test_fn",
            "tool_input": {"x": 1},
            "tool_type": "function",
        }
        result = GoogleGenerateToolOps.ir_tool_call_to_p(ir_tool_call)  # ty: ignore[invalid-argument-type]
        fc = result.get("functionCall", {})
        assert "\n" not in fc.get("id", "")
        assert re.match(r"^[a-zA-Z0-9_-]+$", fc["id"])

    def test_openai_responses_tool_call_sanitized(self):
        from llm_rosetta.converters.openai_responses.tool_ops import (
            OpenAIResponsesToolOps,
        )

        ir_tool_call = {
            "type": "tool_call",
            "tool_call_id": "bad\nid_with_newline",
            "tool_name": "test_fn",
            "tool_input": {"x": 1},
            "tool_type": "function",
        }
        result = OpenAIResponsesToolOps.ir_tool_call_to_p(ir_tool_call)  # ty: ignore[invalid-argument-type]
        assert "\n" not in result.get("call_id", "")
        assert re.match(r"^[a-zA-Z0-9_-]+$", result["call_id"])
