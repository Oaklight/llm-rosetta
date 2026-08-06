"""Tests for the OpenAI Chat custom-tool grammar format shape.

The Responses API flattens grammar fields into ``format``
(``{"type", "syntax", "definition"}``) while Chat Completions nests them
under ``format.grammar``. Emitting the flat shape to a Chat Completions
upstream is rejected with
``Missing required parameter: 'tools[N].custom.format.grammar'``.
"""

from llm_rosetta.converters.openai_chat.tool_ops import OpenAIChatToolOps
from llm_rosetta.converters.openai_responses.tool_ops import OpenAIResponsesToolOps

LARK_DEFINITION = 'start: "*** Begin Patch" /(.|\\n)+/ "*** End Patch"'

RESPONSES_CUSTOM_TOOL = {
    "type": "custom",
    "name": "apply_patch",
    "description": "Apply a V4A patch. FREEFORM — do not wrap in JSON.",
    "format": {
        "type": "grammar",
        "syntax": "lark",
        "definition": LARK_DEFINITION,
    },
}

CHAT_CUSTOM_TOOL = {
    "type": "custom",
    "custom": {
        "name": "apply_patch",
        "description": "Apply a V4A patch. FREEFORM — do not wrap in JSON.",
        "format": {
            "type": "grammar",
            "grammar": {"syntax": "lark", "definition": LARK_DEFINITION},
        },
    },
}


# ---------------------------------------------------------------------------
# Request path — grammar format shape
# ---------------------------------------------------------------------------


class TestCustomToolGrammarShape:
    """Grammar fields must nest under ``format.grammar`` for Chat Completions."""

    def test_responses_flat_format_becomes_nested_for_chat(self):
        """Responses → IR → Chat nests syntax/definition under ``grammar``."""
        ir_tool = OpenAIResponsesToolOps.p_tool_definition_to_ir(RESPONSES_CUSTOM_TOOL)
        chat_tool = OpenAIChatToolOps.ir_tool_definition_to_p(ir_tool)

        fmt = chat_tool["custom"]["format"]
        assert fmt["type"] == "grammar"
        assert fmt["grammar"] == {
            "syntax": "lark",
            "definition": LARK_DEFINITION,
        }
        # The flat spelling must not leak through alongside the nested one.
        assert "syntax" not in fmt
        assert "definition" not in fmt

    def test_chat_nested_format_becomes_flat_for_responses(self):
        """Chat → IR → Responses flattens ``grammar`` back into ``format``."""
        ir_tool = OpenAIChatToolOps.p_tool_definition_to_ir(CHAT_CUSTOM_TOOL)
        responses_tool = OpenAIResponsesToolOps.ir_tool_definition_to_p(ir_tool)

        assert responses_tool["format"] == {
            "type": "grammar",
            "syntax": "lark",
            "definition": LARK_DEFINITION,
        }

    def test_chat_round_trip_does_not_double_nest(self):
        """Chat → IR → Chat is stable (no ``grammar.grammar``)."""
        ir_tool = OpenAIChatToolOps.p_tool_definition_to_ir(CHAT_CUSTOM_TOOL)
        chat_tool = OpenAIChatToolOps.ir_tool_definition_to_p(ir_tool)

        expected_fmt = CHAT_CUSTOM_TOOL["custom"]
        assert isinstance(expected_fmt, dict)
        assert chat_tool["custom"]["format"] == expected_fmt["format"]

    def test_responses_round_trip_is_stable(self):
        """Responses → IR → Chat → IR → Responses preserves the flat shape."""
        ir_a = OpenAIResponsesToolOps.p_tool_definition_to_ir(RESPONSES_CUSTOM_TOOL)
        chat_tool = OpenAIChatToolOps.ir_tool_definition_to_p(ir_a)
        ir_b = OpenAIChatToolOps.p_tool_definition_to_ir(chat_tool)
        responses_tool = OpenAIResponsesToolOps.ir_tool_definition_to_p(ir_b)

        assert responses_tool["format"] == RESPONSES_CUSTOM_TOOL["format"]

    def test_text_format_is_untouched(self):
        """Non-grammar formats pass through both directions unchanged."""
        tool = dict(RESPONSES_CUSTOM_TOOL, format={"type": "text"})
        ir_tool = OpenAIResponsesToolOps.p_tool_definition_to_ir(tool)
        chat_tool = OpenAIChatToolOps.ir_tool_definition_to_p(ir_tool)

        assert chat_tool["custom"]["format"] == {"type": "text"}

    def test_custom_tool_without_format_is_untouched(self):
        """A custom tool with no ``format`` gains none."""
        tool = {k: v for k, v in RESPONSES_CUSTOM_TOOL.items() if k != "format"}
        ir_tool = OpenAIResponsesToolOps.p_tool_definition_to_ir(tool)
        chat_tool = OpenAIChatToolOps.ir_tool_definition_to_p(ir_tool)

        assert "format" not in chat_tool["custom"]
