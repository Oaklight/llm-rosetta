"""
Anthropic ToolOps unit tests.
"""

from llm_rosetta.converters.anthropic.tool_ops import AnthropicToolOps
from typing import cast

from llm_rosetta.types.ir import (
    ToolCallPart,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
)


class TestAnthropicToolOps:
    """Unit tests for AnthropicToolOps."""

    # ==================== Tool Definition ====================

    def test_ir_tool_definition_to_p(self):
        """Test IR ToolDefinition → Anthropic tool definition."""
        ir_tool: ToolDefinition = {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather information",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            "required_parameters": ["city"],
            "metadata": {},
        }
        result = AnthropicToolOps.ir_tool_definition_to_p(ir_tool)
        assert result["name"] == "get_weather"
        assert result["description"] == "Get weather information"
        assert result["input_schema"]["type"] == "object"
        assert "city" in result["input_schema"]["properties"]

    def test_p_tool_definition_to_ir(self):
        """Test Anthropic tool definition → IR ToolDefinition."""
        provider_tool = {
            "name": "get_weather",
            "description": "Get weather info",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
        result = AnthropicToolOps.p_tool_definition_to_ir(provider_tool)
        assert result["type"] == "function"
        assert result["name"] == "get_weather"
        assert result["description"] == "Get weather info"
        assert result["parameters"]["type"] == "object"
        assert result["required_parameters"] == ["city"]

    def test_tool_definition_round_trip(self):
        """Test tool definition round-trip."""
        ir_tool: ToolDefinition = {
            "type": "function",
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {}},
            "required_parameters": [],
            "metadata": {},
        }
        provider = AnthropicToolOps.ir_tool_definition_to_p(ir_tool)
        restored = AnthropicToolOps.p_tool_definition_to_ir(provider)
        assert restored["name"] == ir_tool["name"]
        assert restored["description"] == ir_tool["description"]

    # ==================== Tool Choice ====================

    def test_ir_tool_choice_auto(self):
        """Test IR auto tool choice → Anthropic."""
        result = AnthropicToolOps.ir_tool_choice_to_p({"mode": "auto", "tool_name": ""})
        assert result["type"] == "auto"

    def test_ir_tool_choice_any(self):
        """Test IR any tool choice → Anthropic."""
        result = AnthropicToolOps.ir_tool_choice_to_p({"mode": "any", "tool_name": ""})
        assert result["type"] == "any"

    def test_ir_tool_choice_tool(self):
        """Test IR specific tool choice → Anthropic."""
        result = AnthropicToolOps.ir_tool_choice_to_p(
            {"mode": "tool", "tool_name": "get_weather"}
        )
        assert result["type"] == "tool"
        assert result["name"] == "get_weather"

    def test_ir_tool_choice_none(self):
        """Test IR none tool choice → Anthropic."""
        result = AnthropicToolOps.ir_tool_choice_to_p({"mode": "none", "tool_name": ""})
        assert result["type"] == "none"

    def test_p_tool_choice_auto(self):
        """Test Anthropic auto → IR tool choice."""
        result = AnthropicToolOps.p_tool_choice_to_ir({"type": "auto"})
        assert result["mode"] == "auto"

    def test_p_tool_choice_any(self):
        """Test Anthropic any → IR tool choice."""
        result = AnthropicToolOps.p_tool_choice_to_ir({"type": "any"})
        assert result["mode"] == "any"

    def test_p_tool_choice_tool(self):
        """Test Anthropic tool → IR tool choice."""
        result = AnthropicToolOps.p_tool_choice_to_ir(
            {"type": "tool", "name": "get_weather"}
        )
        assert result["mode"] == "tool"
        assert result["tool_name"] == "get_weather"

    def test_tool_choice_round_trip(self):
        """Test tool choice round-trip."""
        original = cast(ToolChoice, {"mode": "tool", "tool_name": "search"})
        provider = AnthropicToolOps.ir_tool_choice_to_p(original)
        restored = AnthropicToolOps.p_tool_choice_to_ir(provider)
        assert restored["mode"] == original["mode"]
        assert restored["tool_name"] == original["tool_name"]

    # ==================== Tool Call ====================

    def test_ir_tool_call_to_p(self):
        """Test IR ToolCallPart → Anthropic tool_use block."""
        ir_tc = ToolCallPart(
            type="tool_call",
            tool_call_id="call_123",
            tool_name="get_weather",
            tool_input={"city": "Beijing"},
            tool_type="function",
        )
        result = AnthropicToolOps.ir_tool_call_to_p(ir_tc)
        assert result["type"] == "tool_use"
        assert result["id"] == "call_123"
        assert result["name"] == "get_weather"
        assert result["input"] == {"city": "Beijing"}

    def test_ir_tool_call_to_p_web_search(self):
        """Test IR web_search ToolCallPart → Anthropic server_tool_use."""
        ir_tc = ToolCallPart(
            type="tool_call",
            tool_call_id="call_456",
            tool_name="web_search",
            tool_input={"query": "AI news"},
            tool_type="web_search",
        )
        result = AnthropicToolOps.ir_tool_call_to_p(ir_tc)
        assert result["type"] == "server_tool_use"
        assert result["name"] == "web_search"

    def test_p_tool_call_to_ir(self):
        """Test Anthropic tool_use → IR ToolCallPart."""
        provider = {
            "type": "tool_use",
            "id": "toolu_123",
            "name": "get_weather",
            "input": {"city": "SF"},
        }
        result = AnthropicToolOps.p_tool_call_to_ir(provider)
        assert result["type"] == "tool_call"
        assert result["tool_call_id"] == "toolu_123"
        assert result["tool_name"] == "get_weather"
        assert result["tool_input"] == {"city": "SF"}
        assert result["tool_type"] == "function"

    def test_p_tool_call_to_ir_server_tool(self):
        """Test Anthropic server_tool_use → IR ToolCallPart."""
        provider = {
            "type": "server_tool_use",
            "id": "server_456",
            "name": "web_search",
            "input": {"query": "test"},
        }
        result = AnthropicToolOps.p_tool_call_to_ir(provider)
        assert result["type"] == "tool_call"
        assert result["tool_type"] == "web_search"
        assert result["tool_call_id"] == "server_456"

    def test_tool_call_round_trip(self):
        """Test tool call round-trip."""
        original = ToolCallPart(
            type="tool_call",
            tool_call_id="call_rt",
            tool_name="search",
            tool_input={"q": "test"},
            tool_type="function",
        )
        provider = AnthropicToolOps.ir_tool_call_to_p(original)
        restored = AnthropicToolOps.p_tool_call_to_ir(provider)
        assert restored["tool_call_id"] == original["tool_call_id"]
        assert restored["tool_name"] == original["tool_name"]
        assert restored["tool_input"] == original["tool_input"]

    # ==================== Tool Result ====================

    def test_ir_tool_result_to_p(self):
        """Test IR ToolResultPart → Anthropic tool_result block."""
        ir_tr = ToolResultPart(
            type="tool_result",
            tool_call_id="call_123",
            result="Weather is sunny",
        )
        result = AnthropicToolOps.ir_tool_result_to_p(ir_tr)
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "call_123"
        assert result["content"] == "Weather is sunny"

    def test_ir_tool_result_to_p_with_error(self):
        """Test IR ToolResultPart with error → Anthropic tool_result."""
        ir_tr = ToolResultPart(
            type="tool_result",
            tool_call_id="call_err",
            result="Error occurred",
            is_error=True,
        )
        result = AnthropicToolOps.ir_tool_result_to_p(ir_tr)
        assert result["is_error"] is True

    def test_p_tool_result_to_ir(self):
        """Test Anthropic tool_result → IR ToolResultPart."""
        provider = {
            "type": "tool_result",
            "tool_use_id": "tool_789",
            "content": "Result data",
            "is_error": True,
        }
        result = AnthropicToolOps.p_tool_result_to_ir(provider)
        assert result["type"] == "tool_result"
        assert result["tool_call_id"] == "tool_789"
        assert result["result"] == "Result data"
        assert result["is_error"] is True

    def test_tool_result_round_trip(self):
        """Test tool result round-trip."""
        original = ToolResultPart(
            type="tool_result",
            tool_call_id="call_rt",
            result="success",
            is_error=False,
        )
        provider = AnthropicToolOps.ir_tool_result_to_p(original)
        restored = AnthropicToolOps.p_tool_result_to_ir(provider)
        assert restored["tool_call_id"] == original["tool_call_id"]
        assert restored["result"] == original["result"]

    def test_ir_tool_result_to_p_dict_content_serialized_to_json(self):
        """Test that dict result content is serialized to JSON string for Anthropic."""
        ir_tr = ToolResultPart(
            type="tool_result",
            tool_call_id="call_dict",
            result={"temperature": 72, "unit": "F"},
        )
        result = AnthropicToolOps.ir_tool_result_to_p(ir_tr)
        assert result["type"] == "tool_result"
        assert result["tool_use_id"] == "call_dict"
        assert isinstance(result["content"], str)
        assert result["content"] == '{"temperature": 72, "unit": "F"}'

    def test_ir_tool_result_to_p_list_content_converted(self):
        """Test that list IR content blocks are converted to Anthropic format."""
        ir_blocks = [{"type": "text", "text": "hello"}]
        ir_tr = ToolResultPart(
            type="tool_result",
            tool_call_id="call_list",
            result=ir_blocks,
        )
        result = AnthropicToolOps.ir_tool_result_to_p(ir_tr)
        assert isinstance(result["content"], list)
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "hello"

    def test_ir_tool_result_to_p_converts_ir_image_blocks(self):
        """Test IR ImagePart list → Anthropic image blocks with source."""
        ir_blocks = [
            {
                "type": "image",
                "image_data": {"data": "AAAA", "media_type": "image/png"},
            }
        ]
        ir_tr = ToolResultPart(
            type="tool_result",
            tool_call_id="call_img",
            result=ir_blocks,
        )
        result = AnthropicToolOps.ir_tool_result_to_p(ir_tr)
        assert isinstance(result["content"], list)
        assert len(result["content"]) == 1
        img_block = result["content"][0]
        assert img_block["type"] == "image"
        assert img_block["source"]["type"] == "base64"
        assert img_block["source"]["media_type"] == "image/png"
        assert img_block["source"]["data"] == "AAAA"

    def test_p_tool_result_to_ir_normalizes_image_blocks(self):
        """Test Anthropic image blocks in tool_result → IR ImagePart."""
        provider = {
            "type": "tool_result",
            "tool_use_id": "call_img",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "BBBB",
                    },
                }
            ],
        }
        result = AnthropicToolOps.p_tool_result_to_ir(provider)
        assert isinstance(result["result"], list)
        assert len(result["result"]) == 1
        assert result["result"][0]["type"] == "image"
        assert result["result"][0]["image_data"]["data"] == "BBBB"
        assert result["result"][0]["image_data"]["media_type"] == "image/png"

    # ==================== Tool Config ====================

    def test_ir_tool_config_to_p(self):
        """Test IR ToolCallConfig → Anthropic tool config fields."""
        result = AnthropicToolOps.ir_tool_config_to_p({"disable_parallel": True})
        assert result["disable_parallel_tool_use"] is True

    def test_ir_tool_config_to_p_empty(self):
        """Test empty IR ToolCallConfig → empty dict."""
        result = AnthropicToolOps.ir_tool_config_to_p({})
        assert result == {}

    def test_p_tool_config_to_ir(self):
        """Test Anthropic tool config → IR ToolCallConfig."""
        result = AnthropicToolOps.p_tool_config_to_ir(
            {"disable_parallel_tool_use": True}
        )
        assert result["disable_parallel"] is True

    def test_p_tool_config_to_ir_empty(self):
        """Test empty Anthropic tool config → empty IR."""
        result = AnthropicToolOps.p_tool_config_to_ir({})
        assert result == {}
