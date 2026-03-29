"""
Tests for strip_orphaned_tool_config — shared function and converter integration.

Covers the third symptom of Codex context compaction breaking request
structural integrity (see llm-rosetta#87):
  1. Orphaned tool_calls in messages  → fix_orphaned_tool_calls_ir  (v0.2.4)
  2. Tool messages in wrong position  → _reorder_tool_messages      (v0.2.6)
  3. tool_choice without tools        → strip_orphaned_tool_config   (this)
"""

from typing import cast

from llm_rosetta.converters.base.tools import strip_orphaned_tool_config
from llm_rosetta.converters.anthropic.converter import AnthropicConverter
from llm_rosetta.converters.google_genai.converter import GoogleGenAIConverter
from llm_rosetta.converters.openai_chat.converter import OpenAIChatConverter
from llm_rosetta.converters.openai_responses.converter import OpenAIResponsesConverter
from llm_rosetta.types.ir import ToolChoice
from llm_rosetta.types.ir.request import IRRequest


# ==================== Unit tests for shared function ====================


class TestStripOrphanedToolConfig:
    """Unit tests for strip_orphaned_tool_config."""

    def test_no_tools_strips_tool_choice(self):
        """tool_choice is stripped when no tools are present."""
        ir = cast(
            IRRequest,
            {
                "model": "test",
                "messages": [],
                "tool_choice": cast(ToolChoice, {"mode": "auto"}),
            },
        )
        warnings = strip_orphaned_tool_config(ir)
        assert "tool_choice" not in ir
        assert len(warnings) == 1
        assert "tool_choice" in warnings[0]

    def test_no_tools_strips_tool_config(self):
        """tool_config is stripped when no tools are present."""
        ir = cast(
            IRRequest,
            {
                "model": "test",
                "messages": [],
                "tool_config": {"max_calls": 5},
            },
        )
        warnings = strip_orphaned_tool_config(ir)
        assert "tool_config" not in ir
        assert len(warnings) == 1
        assert "tool_config" in warnings[0]

    def test_no_tools_strips_both(self):
        """Both tool_choice and tool_config stripped together."""
        ir = cast(
            IRRequest,
            {
                "model": "test",
                "messages": [],
                "tool_choice": cast(ToolChoice, {"mode": "auto"}),
                "tool_config": {"max_calls": 5},
            },
        )
        warnings = strip_orphaned_tool_config(ir)
        assert "tool_choice" not in ir
        assert "tool_config" not in ir
        assert len(warnings) == 2

    def test_with_tools_preserves_tool_choice(self):
        """tool_choice is kept when tools are present."""
        ir = cast(
            IRRequest,
            {
                "model": "test",
                "messages": [],
                "tools": [
                    {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {"type": "object"},
                    }
                ],
                "tool_choice": cast(ToolChoice, {"mode": "auto"}),
            },
        )
        warnings = strip_orphaned_tool_config(ir)
        assert "tool_choice" in ir
        assert warnings == []

    def test_no_tool_fields_noop(self):
        """No warnings when neither tool_choice nor tool_config present."""
        ir = cast(IRRequest, {"model": "test", "messages": []})
        warnings = strip_orphaned_tool_config(ir)
        assert warnings == []

    def test_empty_tools_list_strips(self):
        """Empty tools list is treated as no tools."""
        ir = cast(
            IRRequest,
            {
                "model": "test",
                "messages": [],
                "tools": [],
                "tool_choice": cast(ToolChoice, {"mode": "required"}),
            },
        )
        warnings = strip_orphaned_tool_config(ir)
        assert "tool_choice" not in ir
        assert len(warnings) == 1


# ==================== Converter integration tests ====================


def _make_ir_request_with_orphaned_tool_choice() -> IRRequest:
    """Create an IR request with tool_choice but no tools (Codex compaction)."""
    return cast(
        IRRequest,
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}],
                }
            ],
            "tool_choice": cast(ToolChoice, {"mode": "auto"}),
        },
    )


def _make_ir_request_with_tools_and_choice() -> IRRequest:
    """Create a valid IR request with tools and tool_choice."""
    return cast(
        IRRequest,
        {
            "model": "test-model",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}],
                }
            ],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
            "tool_choice": cast(ToolChoice, {"mode": "auto"}),
        },
    )


class TestOpenAIChatStripOrphanedToolConfig:
    """OpenAI Chat converter strips orphaned tool_choice."""

    def setup_method(self):
        self.converter = OpenAIChatConverter()

    def test_orphaned_tool_choice_stripped(self):
        ir = _make_ir_request_with_orphaned_tool_choice()
        result, warnings = self.converter.request_to_provider(ir)
        assert "tool_choice" not in result
        assert any("tool_choice" in w for w in warnings)

    def test_valid_tool_choice_preserved(self):
        ir = _make_ir_request_with_tools_and_choice()
        result, warnings = self.converter.request_to_provider(ir)
        assert "tool_choice" in result
        assert not any("Stripped" in w for w in warnings)


class TestOpenAIResponsesStripOrphanedToolConfig:
    """OpenAI Responses converter strips orphaned tool_choice."""

    def setup_method(self):
        self.converter = OpenAIResponsesConverter()

    def test_orphaned_tool_choice_stripped(self):
        ir = _make_ir_request_with_orphaned_tool_choice()
        result, warnings = self.converter.request_to_provider(ir)
        assert "tool_choice" not in result
        assert any("tool_choice" in w for w in warnings)

    def test_valid_tool_choice_preserved(self):
        ir = _make_ir_request_with_tools_and_choice()
        result, warnings = self.converter.request_to_provider(ir)
        assert "tool_choice" in result
        assert not any("Stripped" in w for w in warnings)


class TestAnthropicStripOrphanedToolConfig:
    """Anthropic converter strips orphaned tool_choice."""

    def setup_method(self):
        self.converter = AnthropicConverter()

    def test_orphaned_tool_choice_stripped(self):
        ir = _make_ir_request_with_orphaned_tool_choice()
        result, warnings = self.converter.request_to_provider(ir)
        assert "tool_choice" not in result
        assert any("tool_choice" in w for w in warnings)

    def test_valid_tool_choice_preserved(self):
        ir = _make_ir_request_with_tools_and_choice()
        result, warnings = self.converter.request_to_provider(ir)
        assert "tool_choice" in result
        assert not any("Stripped" in w for w in warnings)


class TestGoogleGenAIStripOrphanedToolConfig:
    """Google GenAI converter strips orphaned tool_choice."""

    def setup_method(self):
        self.converter = GoogleGenAIConverter()

    def test_orphaned_tool_choice_stripped(self):
        ir = _make_ir_request_with_orphaned_tool_choice()
        result, warnings = self.converter.request_to_provider(ir)
        # Google maps tool_choice to config.tool_config
        config = result.get("config", {})
        assert "tool_config" not in config
        assert any("tool_choice" in w for w in warnings)

    def test_valid_tool_choice_preserved(self):
        ir = _make_ir_request_with_tools_and_choice()
        result, warnings = self.converter.request_to_provider(ir)
        config = result.get("config", {})
        assert "tool_config" in config
        assert not any("Stripped" in w for w in warnings)
