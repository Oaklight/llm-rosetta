"""Unit tests for tool-result error marking."""

from llm_rosetta.converters.base.tools.error_marker import (
    TOOL_ERROR_MARKER,
    mark_tool_error,
    warn_tool_error,
    warn_tool_error_dropped,
)


class TestMarkToolError:
    def test_string_becomes_text_block_plus_marker(self):
        result = mark_tool_error("boom", text_block_type="text")
        assert result == [
            {"type": "text", "text": "boom"},
            {"type": "text", "text": TOOL_ERROR_MARKER},
        ]

    def test_empty_string_yields_marker_only(self):
        """An empty payload must not emit an empty content block."""
        result = mark_tool_error("", text_block_type="text")
        assert result == [{"type": "text", "text": TOOL_ERROR_MARKER}]

    def test_empty_list_yields_marker_only(self):
        """An empty payload reaches the marker either way it is shaped."""
        result = mark_tool_error([], text_block_type="text")
        assert result == [{"type": "text", "text": TOOL_ERROR_MARKER}]

    def test_list_appends_marker_without_mutating_input(self):
        original = [{"type": "input_text", "text": "boom"}]
        result = mark_tool_error(original, text_block_type="input_text")
        assert result == [
            {"type": "input_text", "text": "boom"},
            {"type": "input_text", "text": TOOL_ERROR_MARKER},
        ]
        assert original == [{"type": "input_text", "text": "boom"}]

    def test_block_type_is_provider_specific(self):
        result = mark_tool_error("x", text_block_type="input_text")
        assert all(block["type"] == "input_text" for block in result)


class TestWarnToolError:
    def test_appends_message_naming_the_call(self):
        warnings: list[str] = []
        warn_tool_error(
            warnings,
            format_name="OpenAI Chat Completions",
            tool_call_id="toolu_1",
        )
        assert len(warnings) == 1
        assert "OpenAI Chat Completions" in warnings[0]
        assert "toolu_1" in warnings[0]
        assert TOOL_ERROR_MARKER in warnings[0]

    def test_no_sink_is_a_noop(self):
        """The public dispatch helpers convert one part without a context."""
        warn_tool_error(None, format_name="OpenAI Responses", tool_call_id="c")


class TestWarnToolErrorDropped:
    def test_reports_a_drop_without_claiming_a_marker(self):
        warnings: list[str] = []
        warn_tool_error_dropped(
            warnings,
            format_name="OpenAI Responses",
            tool_call_id="ws_1",
        )
        assert len(warnings) == 1
        assert "ws_1" in warnings[0]
        assert "dropped" in warnings[0]
        assert TOOL_ERROR_MARKER not in warnings[0]

    def test_no_sink_is_a_noop(self):
        warn_tool_error_dropped(
            None, format_name="OpenAI Responses", tool_call_id="ws_1"
        )
