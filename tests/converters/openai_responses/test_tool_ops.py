"""
OpenAI Responses ToolOps unit tests.
"""

import json
from typing import Any, cast

from llm_rosetta.converters.openai_responses.tool_ops import OpenAIResponsesToolOps
from llm_rosetta.types.ir import (
    ToolCallConfig,
    ToolCallPart,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
)


class TestOpenAIResponsesToolOps:
    """Unit tests for OpenAIResponsesToolOps."""

    # ==================== Tool Definition ====================

    def test_ir_tool_definition_to_p(self):
        """Test IR ToolDefinition → OpenAI Responses flat tool definition."""
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
        result = OpenAIResponsesToolOps.ir_tool_definition_to_p(ir_tool)
        assert result["type"] == "function"
        assert result["name"] == "get_weather"
        assert result["description"] == "Get current weather"
        assert result["parameters"]["type"] == "object"
        assert result["strict"] is False

    def test_ir_tool_definition_to_p_non_function(self):
        """Test non-function tool type → custom wrapper."""
        ir_tool = cast(
            ToolDefinition,
            {
                "type": "web_search",
                "name": "search",
                "description": "Search the web",
                "parameters": {},
                "required_parameters": [],
                "metadata": {},
            },
        )
        result = OpenAIResponsesToolOps.ir_tool_definition_to_p(ir_tool)
        assert result["type"] == "custom"
        assert result["name"] == "search"

    def test_p_tool_definition_to_ir_flat(self):
        """Test OpenAI Responses flat tool definition → IR ToolDefinition."""
        provider_tool = {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        }
        result = OpenAIResponsesToolOps.p_tool_definition_to_ir(provider_tool)
        assert result["type"] == "function"
        assert result["name"] == "get_weather"
        assert result["description"] == "Get weather"
        assert result["parameters"]["type"] == "object"
        assert result["required_parameters"] == ["city"]

    def test_p_tool_definition_to_ir_nested(self):
        """Test OpenAI nested format (with function key) → IR ToolDefinition."""
        provider_tool = {
            "type": "function",
            "function": {
                "name": "search",
                "description": "Search",
                "parameters": {
                    "type": "object",
                    "properties": {"query": {"type": "string"}},
                    "required": ["query"],
                },
            },
        }
        result = OpenAIResponsesToolOps.p_tool_definition_to_ir(provider_tool)
        assert result["type"] == "function"
        assert result["name"] == "search"
        assert result["description"] == "Search"
        assert result["required_parameters"] == ["query"]

    def test_tool_definition_round_trip(self):
        """Test tool definition round-trip."""
        ir_tool = cast(
            ToolDefinition,
            {
                "type": "function",
                "name": "calculate",
                "description": "Calculate expression",
                "parameters": {"type": "object", "properties": {}},
                "required_parameters": [],
                "metadata": {},
            },
        )
        provider = OpenAIResponsesToolOps.ir_tool_definition_to_p(ir_tool)
        restored = OpenAIResponsesToolOps.p_tool_definition_to_ir(provider)
        assert restored["name"] == ir_tool["name"]
        assert restored["description"] == ir_tool["description"]

    # ==================== Tool Choice ====================

    def test_ir_tool_choice_none(self):
        """Test mode:none → 'none'."""
        result = OpenAIResponsesToolOps.ir_tool_choice_to_p(
            cast(ToolChoice, {"mode": "none", "tool_name": ""})
        )
        assert result == "none"

    def test_ir_tool_choice_auto(self):
        """Test mode:auto → 'auto'."""
        result = OpenAIResponsesToolOps.ir_tool_choice_to_p(
            cast(ToolChoice, {"mode": "auto", "tool_name": ""})
        )
        assert result == "auto"

    def test_ir_tool_choice_any(self):
        """Test mode:any → 'required'."""
        result = OpenAIResponsesToolOps.ir_tool_choice_to_p(
            cast(ToolChoice, {"mode": "any", "tool_name": ""})
        )
        assert result == "required"

    def test_ir_tool_choice_required(self):
        """Test mode:required → 'required'."""
        result = OpenAIResponsesToolOps.ir_tool_choice_to_p(
            cast(Any, {"mode": "required", "tool_name": ""})
        )
        assert result == "required"

    def test_ir_tool_choice_specific(self):
        """Test mode:tool → specific function dict."""
        result = OpenAIResponsesToolOps.ir_tool_choice_to_p(
            cast(ToolChoice, {"mode": "tool", "tool_name": "get_weather"})
        )
        assert result == {"type": "function", "name": "get_weather"}

    def test_ir_tool_choice_legacy_type_field(self):
        """Test legacy 'type' field support."""
        result = OpenAIResponsesToolOps.ir_tool_choice_to_p(
            cast(Any, {"type": "auto", "tool_name": ""})
        )
        assert result == "auto"

    def test_p_tool_choice_none(self):
        """Test 'none' → mode:none."""
        result = OpenAIResponsesToolOps.p_tool_choice_to_ir("none")
        assert result["mode"] == "none"

    def test_p_tool_choice_auto(self):
        """Test 'auto' → mode:auto."""
        result = OpenAIResponsesToolOps.p_tool_choice_to_ir("auto")
        assert result["mode"] == "auto"

    def test_p_tool_choice_required(self):
        """Test 'required' → mode:any."""
        result = OpenAIResponsesToolOps.p_tool_choice_to_ir("required")
        assert result["mode"] == "any"

    def test_p_tool_choice_specific(self):
        """Test specific function → mode:tool."""
        result = OpenAIResponsesToolOps.p_tool_choice_to_ir(
            {"type": "function", "function": {"name": "get_weather"}}
        )
        assert result["mode"] == "tool"
        assert result["tool_name"] == "get_weather"

    def test_p_tool_choice_unknown_string(self):
        """Test unknown string → mode:auto fallback."""
        result = OpenAIResponsesToolOps.p_tool_choice_to_ir("unknown")
        assert result["mode"] == "auto"

    def test_tool_choice_round_trip(self):
        """Test tool choice round-trip."""
        for mode in ["none", "auto"]:
            ir = cast(ToolChoice, {"mode": mode, "tool_name": ""})
            provider = OpenAIResponsesToolOps.ir_tool_choice_to_p(ir)
            restored = OpenAIResponsesToolOps.p_tool_choice_to_ir(provider)
            assert restored["mode"] == mode

    def test_tool_choice_any_round_trip(self):
        """Test mode:any round-trip (any → required → any)."""
        ir = cast(ToolChoice, {"mode": "any", "tool_name": ""})
        provider = OpenAIResponsesToolOps.ir_tool_choice_to_p(ir)
        assert provider == "required"
        restored = OpenAIResponsesToolOps.p_tool_choice_to_ir(provider)
        assert restored["mode"] == "any"

    # ==================== Tool Call ====================

    def test_ir_tool_call_to_p_function(self):
        """Test IR ToolCallPart → OpenAI Responses function_call item."""
        ir_tc = ToolCallPart(
            type="tool_call",
            tool_call_id="call_123",
            tool_name="get_weather",
            tool_input={"city": "Beijing"},
        )
        result = OpenAIResponsesToolOps.ir_tool_call_to_p(ir_tc)
        assert result["type"] == "function_call"
        assert result["id"] == "call_123"
        assert result["call_id"] == "call_123"
        assert result["name"] == "get_weather"
        assert json.loads(result["arguments"]) == {"city": "Beijing"}

    def test_ir_tool_call_to_p_mcp(self):
        """Test IR ToolCallPart with mcp tool_type → mcp_call item."""
        ir_tc = ToolCallPart(
            type="tool_call",
            tool_call_id="call_mcp",
            tool_name="mcp://server/tool",
            tool_input={"param": "value"},
            tool_type="mcp",
        )
        result = OpenAIResponsesToolOps.ir_tool_call_to_p(ir_tc)
        assert result["type"] == "mcp_call"
        assert result["id"] == "call_mcp"

    def test_ir_tool_call_to_p_mcp_by_name(self):
        """Test IR ToolCallPart with mcp:// prefix → mcp_call item."""
        ir_tc = ToolCallPart(
            type="tool_call",
            tool_call_id="call_mcp2",
            tool_name="mcp://myserver/mytool",
            tool_input={},
        )
        result = OpenAIResponsesToolOps.ir_tool_call_to_p(ir_tc)
        assert result["type"] == "mcp_call"

    def test_ir_tool_call_to_p_web_search(self):
        """Test IR ToolCallPart with web_search type."""
        ir_tc = ToolCallPart(
            type="tool_call",
            tool_call_id="call_ws",
            tool_name="web_search",
            tool_input={"query": "test"},
            tool_type="web_search",
        )
        result = OpenAIResponsesToolOps.ir_tool_call_to_p(ir_tc)
        assert result["type"] == "function_web_search"
        assert result["query"] == "test"

    def test_ir_tool_call_to_p_code_interpreter(self):
        """Test IR ToolCallPart with code_interpreter type."""
        ir_tc = ToolCallPart(
            type="tool_call",
            tool_call_id="call_ci",
            tool_name="code_interpreter",
            tool_input={"code": "print('hello')"},
            tool_type="code_interpreter",
        )
        result = OpenAIResponsesToolOps.ir_tool_call_to_p(ir_tc)
        assert result["type"] == "code_interpreter_call"
        assert result["code"] == "print('hello')"

    def test_p_tool_call_to_ir_function_call(self):
        """Test OpenAI Responses function_call → IR ToolCallPart."""
        provider_tc = {
            "type": "function_call",
            "call_id": "call_456",
            "name": "search",
            "arguments": '{"query": "test"}',
        }
        result = OpenAIResponsesToolOps.p_tool_call_to_ir(provider_tc)
        assert result["type"] == "tool_call"
        assert result["tool_call_id"] == "call_456"
        assert result["tool_name"] == "search"
        assert result["tool_input"] == {"query": "test"}
        assert result["tool_type"] == "function"

    def test_p_tool_call_to_ir_mcp_call(self):
        """Test OpenAI Responses mcp_call → IR ToolCallPart."""
        provider_tc = {
            "type": "mcp_call",
            "id": "call_mcp",
            "server": "myserver",
            "tool": "mytool",
            "arguments": '{"key": "val"}',
        }
        result = OpenAIResponsesToolOps.p_tool_call_to_ir(provider_tc)
        assert result["type"] == "tool_call"
        assert result["tool_type"] == "mcp"
        assert "mcp://" in result["tool_name"]

    def test_p_tool_call_to_ir_shell_call(self):
        """Test OpenAI Responses shell_call → IR ToolCallPart."""
        provider_tc = {
            "type": "shell_call",
            "call_id": "call_sh",
            "name": "shell",
            "arguments": '{"cmd": "ls"}',
        }
        result = OpenAIResponsesToolOps.p_tool_call_to_ir(provider_tc)
        assert result["type"] == "tool_call"
        assert result["tool_type"] == "code_interpreter"

    def test_p_tool_call_to_ir_invalid_json(self):
        """Test p_tool_call_to_ir handles invalid JSON arguments."""
        provider_tc = {
            "type": "function_call",
            "call_id": "call_789",
            "name": "tool",
            "arguments": "not valid json",
        }
        result = OpenAIResponsesToolOps.p_tool_call_to_ir(provider_tc)
        assert result["tool_input"] == {"input": "not valid json"}

    def test_p_tool_call_to_ir_empty_arguments(self):
        """Test p_tool_call_to_ir handles empty string arguments."""
        provider_tc = {
            "type": "function_call",
            "call_id": "call_empty",
            "name": "tool",
            "arguments": "",
        }
        result = OpenAIResponsesToolOps.p_tool_call_to_ir(provider_tc)
        assert result["tool_input"] == {}

    def test_p_tool_call_to_ir_dict_arguments(self):
        """Test p_tool_call_to_ir handles dict arguments directly."""
        provider_tc = {
            "type": "function_call",
            "call_id": "call_dict",
            "name": "tool",
            "arguments": {"key": "value"},
        }
        result = OpenAIResponsesToolOps.p_tool_call_to_ir(provider_tc)
        assert result["tool_input"] == {"key": "value"}

    def test_p_tool_call_to_ir_unsupported_type(self):
        """Test p_tool_call_to_ir raises on unsupported type."""
        import pytest

        with pytest.raises(ValueError, match="Unsupported"):
            OpenAIResponsesToolOps.p_tool_call_to_ir({"type": "unknown_call"})

    def test_tool_call_round_trip(self):
        """Test tool call round-trip."""
        original = ToolCallPart(
            type="tool_call",
            tool_call_id="call_rt",
            tool_name="func",
            tool_input={"a": 1, "b": "two"},
        )
        provider = OpenAIResponsesToolOps.ir_tool_call_to_p(original)
        restored = OpenAIResponsesToolOps.p_tool_call_to_ir(provider)
        assert restored["tool_call_id"] == original["tool_call_id"]
        assert restored["tool_name"] == original["tool_name"]
        assert restored["tool_input"] == original["tool_input"]

    # ==================== Tool Result ====================

    def test_ir_tool_result_to_p(self):
        """Test IR ToolResultPart → OpenAI Responses function_call_output."""
        ir_tr = ToolResultPart(
            type="tool_result",
            tool_call_id="call_123",
            result="Sunny, 25°C",
        )
        result = OpenAIResponsesToolOps.ir_tool_result_to_p(ir_tr)
        assert result["type"] == "function_call_output"
        assert result["call_id"] == "call_123"
        assert result["output"] == "Sunny, 25°C"

    def test_ir_tool_result_to_p_dict_result(self):
        """Test IR ToolResultPart with dict result → string output."""
        ir_tr = ToolResultPart(
            type="tool_result",
            tool_call_id="call_dict",
            result={"temp": 25, "condition": "sunny"},
        )
        result = OpenAIResponsesToolOps.ir_tool_result_to_p(ir_tr)
        assert result["type"] == "function_call_output"
        # Dict result is converted to string
        assert isinstance(result["output"], str)

    def test_p_tool_result_to_ir(self):
        """Test OpenAI Responses function_call_output → IR ToolResultPart."""
        provider_tr = {
            "type": "function_call_output",
            "call_id": "call_456",
            "output": "Result data",
        }
        result = OpenAIResponsesToolOps.p_tool_result_to_ir(provider_tr)
        assert result["type"] == "tool_result"
        assert result["tool_call_id"] == "call_456"
        assert result["result"] == "Result data"

    def test_p_tool_result_to_ir_json_output(self):
        """Test p_tool_result_to_ir parses JSON output."""
        provider_tr = {
            "type": "function_call_output",
            "call_id": "call_json",
            "output": '{"temp": 25}',
        }
        result = OpenAIResponsesToolOps.p_tool_result_to_ir(provider_tr)
        assert result["result"] == {"temp": 25}

    def test_p_tool_result_to_ir_with_error(self):
        """Test p_tool_result_to_ir with is_error flag."""
        provider_tr = {
            "type": "function_call_output",
            "call_id": "call_err",
            "output": "Error occurred",
            "is_error": True,
        }
        result = OpenAIResponsesToolOps.p_tool_result_to_ir(provider_tr)
        assert result["is_error"] is True

    def test_tool_result_round_trip(self):
        """Test tool result round-trip."""
        original = ToolResultPart(
            type="tool_result",
            tool_call_id="call_rt",
            result="Sunny, 25°C",
        )
        provider = OpenAIResponsesToolOps.ir_tool_result_to_p(original)
        restored = OpenAIResponsesToolOps.p_tool_result_to_ir(provider)
        assert restored["tool_call_id"] == original["tool_call_id"]
        assert restored["result"] == original["result"]

    def test_p_tool_result_to_ir_converts_input_image_to_ir(self):
        """Test input_image in parsed output → IR ImagePart."""
        provider_tr = {
            "type": "function_call_output",
            "call_id": "call_img",
            "output": json.dumps(
                [
                    {
                        "type": "input_image",
                        "image_url": "data:image/png;base64,AAAA",
                    }
                ]
            ),
        }
        result = OpenAIResponsesToolOps.p_tool_result_to_ir(provider_tr)
        assert isinstance(result["result"], list)
        assert len(result["result"]) == 1
        assert result["result"][0]["type"] == "image"
        assert result["result"][0]["image_data"]["data"] == "AAAA"
        assert result["result"][0]["image_data"]["media_type"] == "image/png"

    def test_ir_tool_result_to_p_multimodal_list(self):
        """Test IR blocks → output array format."""
        ir_tr = ToolResultPart(
            type="tool_result",
            tool_call_id="call_multi",
            result=[
                {"type": "text", "text": "Chart description"},
                {
                    "type": "image",
                    "image_data": {"data": "CCCC", "media_type": "image/png"},
                },
            ],
        )
        result = OpenAIResponsesToolOps.ir_tool_result_to_p(ir_tr)
        assert result["type"] == "function_call_output"
        output = result["output"]
        assert isinstance(output, list)
        assert len(output) == 2
        assert output[0]["type"] == "input_text"
        assert output[0]["text"] == "Chart description"
        assert output[1]["type"] == "input_image"
        assert "base64,CCCC" in output[1]["image_url"]

    # ==================== Tool Config ====================

    def test_ir_tool_config_to_p_disable_parallel(self):
        """Test IR ToolCallConfig disable_parallel → parallel_tool_calls."""
        result = OpenAIResponsesToolOps.ir_tool_config_to_p(
            cast(ToolCallConfig, {"disable_parallel": True})
        )
        assert result["parallel_tool_calls"] is False

        result = OpenAIResponsesToolOps.ir_tool_config_to_p(
            cast(ToolCallConfig, {"disable_parallel": False})
        )
        assert result["parallel_tool_calls"] is True

    def test_ir_tool_config_to_p_max_calls(self):
        """Test IR ToolCallConfig max_calls → max_tool_calls."""
        result = OpenAIResponsesToolOps.ir_tool_config_to_p(
            cast(ToolCallConfig, {"max_calls": 5})
        )
        assert result["max_tool_calls"] == 5

    def test_ir_tool_config_to_p_both(self):
        """Test IR ToolCallConfig with both fields."""
        result = OpenAIResponsesToolOps.ir_tool_config_to_p(
            cast(ToolCallConfig, {"disable_parallel": True, "max_calls": 3})
        )
        assert result["parallel_tool_calls"] is False
        assert result["max_tool_calls"] == 3

    def test_p_tool_config_to_ir(self):
        """Test OpenAI parallel_tool_calls → IR ToolCallConfig."""
        result = OpenAIResponsesToolOps.p_tool_config_to_ir(
            {"parallel_tool_calls": False}
        )
        assert result["disable_parallel"] is True

        result = OpenAIResponsesToolOps.p_tool_config_to_ir(
            {"parallel_tool_calls": True}
        )
        assert result["disable_parallel"] is False

    def test_p_tool_config_to_ir_max_calls(self):
        """Test OpenAI max_tool_calls → IR ToolCallConfig."""
        result = OpenAIResponsesToolOps.p_tool_config_to_ir({"max_tool_calls": 10})
        assert result["max_calls"] == 10

    def test_p_tool_config_to_ir_non_dict(self):
        """Test p_tool_config_to_ir with non-dict input."""
        result = OpenAIResponsesToolOps.p_tool_config_to_ir("invalid")
        assert result == {}

    def test_tool_config_round_trip(self):
        """Test tool config round-trip."""
        original = cast(ToolCallConfig, {"disable_parallel": True, "max_calls": 5})
        provider = OpenAIResponsesToolOps.ir_tool_config_to_p(original)
        restored = OpenAIResponsesToolOps.p_tool_config_to_ir(provider)
        assert restored["disable_parallel"] == original["disable_parallel"]
        assert restored["max_calls"] == original["max_calls"]
