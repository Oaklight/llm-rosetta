"""
OpenAI Chat ToolOps unit tests.
"""

import json

from llm_rosetta.converters.openai_chat.tool_ops import OpenAIChatToolOps
from typing import cast

from llm_rosetta.types.ir import (
    ToolCallConfig,
    ToolCallPart,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
)


class TestOpenAIChatToolOps:
    """Unit tests for OpenAIChatToolOps."""

    # ==================== Tool Definition ====================

    def test_ir_tool_definition_to_p(self):
        """Test IR ToolDefinition → OpenAI tool definition."""
        ir_tool = cast(
            ToolDefinition,
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
                "required_parameters": ["location"],
                "metadata": {},
            },
        )
        result = OpenAIChatToolOps.ir_tool_definition_to_p(ir_tool)
        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"
        assert result["function"]["description"] == "Get current weather"
        assert "parameters" in result["function"]

    def test_p_tool_definition_to_ir(self):
        """Test OpenAI tool definition → IR ToolDefinition."""
        provider_tool = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            },
        }
        result = OpenAIChatToolOps.p_tool_definition_to_ir(provider_tool)
        assert result["type"] == "function"
        assert result["name"] == "get_weather"
        assert result["description"] == "Get weather"
        assert result["parameters"]["type"] == "object"
        assert result["required_parameters"] == ["city"]

    def test_tool_definition_round_trip(self):
        """Test tool definition round-trip."""
        ir_tool = cast(
            ToolDefinition,
            {
                "type": "function",
                "name": "search",
                "description": "Search the web",
                "parameters": {"type": "object", "properties": {}},
                "required_parameters": [],
                "metadata": {},
            },
        )
        provider = OpenAIChatToolOps.ir_tool_definition_to_p(ir_tool)
        restored = OpenAIChatToolOps.p_tool_definition_to_ir(provider)
        assert restored["name"] == ir_tool["name"]
        assert restored["description"] == ir_tool["description"]

    # ==================== Tool Choice ====================

    def test_ir_tool_choice_none(self):
        """Test mode:none → 'none'."""
        result = OpenAIChatToolOps.ir_tool_choice_to_p(
            {"mode": "none", "tool_name": ""}
        )
        assert result == "none"

    def test_ir_tool_choice_auto(self):
        """Test mode:auto → 'auto'."""
        result = OpenAIChatToolOps.ir_tool_choice_to_p(
            {"mode": "auto", "tool_name": ""}
        )
        assert result == "auto"

    def test_ir_tool_choice_any(self):
        """Test mode:any → 'required'."""
        result = OpenAIChatToolOps.ir_tool_choice_to_p({"mode": "any", "tool_name": ""})
        assert result == "required"

    def test_ir_tool_choice_specific(self):
        """Test mode:tool → specific function."""
        result = OpenAIChatToolOps.ir_tool_choice_to_p(
            {"mode": "tool", "tool_name": "get_weather"}
        )
        assert result == {"type": "function", "function": {"name": "get_weather"}}

    def test_p_tool_choice_none(self):
        """Test 'none' → mode:none."""
        result = OpenAIChatToolOps.p_tool_choice_to_ir("none")
        assert result["mode"] == "none"

    def test_p_tool_choice_auto(self):
        """Test 'auto' → mode:auto."""
        result = OpenAIChatToolOps.p_tool_choice_to_ir("auto")
        assert result["mode"] == "auto"

    def test_p_tool_choice_required(self):
        """Test 'required' → mode:any."""
        result = OpenAIChatToolOps.p_tool_choice_to_ir("required")
        assert result["mode"] == "any"

    def test_p_tool_choice_specific(self):
        """Test specific function → mode:tool."""
        result = OpenAIChatToolOps.p_tool_choice_to_ir(
            {"type": "function", "function": {"name": "get_weather"}}
        )
        assert result["mode"] == "tool"
        assert result["tool_name"] == "get_weather"

    def test_tool_choice_round_trip(self):
        """Test tool choice round-trip."""
        for mode in ["none", "auto", "any"]:
            ir = cast(ToolChoice, {"mode": mode, "tool_name": ""})
            provider = OpenAIChatToolOps.ir_tool_choice_to_p(ir)
            restored = OpenAIChatToolOps.p_tool_choice_to_ir(provider)
            assert restored["mode"] == mode

    # ==================== Tool Call ====================

    def test_ir_tool_call_to_p(self):
        """Test IR ToolCallPart → OpenAI tool call."""
        ir_tc = ToolCallPart(
            type="tool_call",
            tool_call_id="call_123",
            tool_name="get_weather",
            tool_input={"city": "Beijing"},
        )
        result = OpenAIChatToolOps.ir_tool_call_to_p(ir_tc)
        assert result["id"] == "call_123"
        assert result["type"] == "function"
        assert result["function"]["name"] == "get_weather"
        assert json.loads(result["function"]["arguments"]) == {"city": "Beijing"}

    def test_p_tool_call_to_ir(self):
        """Test OpenAI tool call → IR ToolCallPart."""
        provider_tc = {
            "id": "call_456",
            "type": "function",
            "function": {
                "name": "search",
                "arguments": '{"query": "test"}',
            },
        }
        result = OpenAIChatToolOps.p_tool_call_to_ir(provider_tc)
        assert result["type"] == "tool_call"
        assert result["tool_call_id"] == "call_456"
        assert result["tool_name"] == "search"
        assert result["tool_input"] == {"query": "test"}

    def test_p_tool_call_to_ir_invalid_json(self):
        """Test p_tool_call_to_ir handles invalid JSON arguments."""
        provider_tc = {
            "id": "call_789",
            "type": "function",
            "function": {
                "name": "tool",
                "arguments": "not valid json",
            },
        }
        result = OpenAIChatToolOps.p_tool_call_to_ir(provider_tc)
        assert result["tool_input"] == {"raw_arguments": "not valid json"}

    def test_tool_call_round_trip(self):
        """Test tool call round-trip."""
        original = ToolCallPart(
            type="tool_call",
            tool_call_id="call_rt",
            tool_name="func",
            tool_input={"a": 1, "b": "two"},
        )
        provider = OpenAIChatToolOps.ir_tool_call_to_p(original)
        restored = OpenAIChatToolOps.p_tool_call_to_ir(provider)
        assert restored["tool_call_id"] == original["tool_call_id"]
        assert restored["tool_name"] == original["tool_name"]
        assert restored["tool_input"] == original["tool_input"]

    # ==================== Tool Result ====================

    def test_ir_tool_result_to_p(self):
        """Test IR ToolResultPart → OpenAI tool message."""
        ir_tr = cast(
            ToolResultPart,
            {
                "type": "tool_result",
                "tool_call_id": "call_123",
                "result": "Sunny, 25°C",
            },
        )
        result = OpenAIChatToolOps.ir_tool_result_to_p(ir_tr)
        assert result["role"] == "tool"
        assert result["tool_call_id"] == "call_123"
        assert result["content"] == "Sunny, 25°C"

    def test_p_tool_result_to_ir(self):
        """Test OpenAI tool message → IR ToolResultPart."""
        provider_tr = {
            "role": "tool",
            "tool_call_id": "call_456",
            "content": "Result data",
        }
        result = OpenAIChatToolOps.p_tool_result_to_ir(provider_tr)
        assert result["type"] == "tool_result"
        assert result["tool_call_id"] == "call_456"
        assert result["result"] == "Result data"

    def test_tool_result_round_trip(self):
        """Test tool result round-trip."""
        original = cast(
            ToolResultPart,
            {
                "type": "tool_result",
                "tool_call_id": "call_rt",
                "result": "42",
            },
        )
        provider = OpenAIChatToolOps.ir_tool_result_to_p(original)
        restored = OpenAIChatToolOps.p_tool_result_to_ir(provider)
        assert restored["tool_call_id"] == original["tool_call_id"]
        assert restored["result"] == original["result"]

    # ==================== Tool Config ====================

    def test_ir_tool_config_to_p(self):
        """Test IR ToolCallConfig → OpenAI parallel_tool_calls."""
        result = OpenAIChatToolOps.ir_tool_config_to_p({"disable_parallel": True})
        assert result["parallel_tool_calls"] is False

        result = OpenAIChatToolOps.ir_tool_config_to_p({"disable_parallel": False})
        assert result["parallel_tool_calls"] is True

    def test_p_tool_config_to_ir(self):
        """Test OpenAI parallel_tool_calls → IR ToolCallConfig."""
        result = OpenAIChatToolOps.p_tool_config_to_ir({"parallel_tool_calls": False})
        assert result["disable_parallel"] is True

        result = OpenAIChatToolOps.p_tool_config_to_ir({"parallel_tool_calls": True})
        assert result["disable_parallel"] is False

    def test_tool_config_round_trip(self):
        """Test tool config round-trip."""
        original = cast(ToolCallConfig, {"disable_parallel": True})
        provider = OpenAIChatToolOps.ir_tool_config_to_p(original)
        restored = OpenAIChatToolOps.p_tool_config_to_ir(provider)
        assert restored["disable_parallel"] == original["disable_parallel"]
