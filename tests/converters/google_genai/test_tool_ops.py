"""
Google GenAI ToolOps unit tests.
"""

import pytest

from llm_rosetta.converters.google_genai.tool_ops import GoogleGenAIToolOps
from llm_rosetta.types.ir import ToolCallPart, ToolDefinition, ToolResultPart


class TestGoogleGenAIToolOps:
    """Unit tests for GoogleGenAIToolOps."""

    # ==================== Tool Definition ====================

    def test_ir_tool_definition_to_p(self):
        """Test IR ToolDefinition → Google FunctionDeclaration."""
        ir_tool: ToolDefinition = {
            "type": "function",
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
        result = GoogleGenAIToolOps.ir_tool_definition_to_p(ir_tool)
        assert "function_declarations" in result
        assert len(result["function_declarations"]) == 1
        func_decl = result["function_declarations"][0]
        assert func_decl["name"] == "get_weather"
        assert func_decl["description"] == "Get current weather"
        assert "parameters" in func_decl

    def test_p_tool_definition_to_ir(self):
        """Test Google FunctionDeclaration → IR ToolDefinition."""
        provider_tool = {
            "function_declarations": [
                {
                    "name": "get_weather",
                    "description": "Get weather info",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                }
            ]
        }
        result = GoogleGenAIToolOps.p_tool_definition_to_ir(provider_tool)
        assert result["type"] == "function"
        assert result["name"] == "get_weather"
        assert result["description"] == "Get weather info"
        assert result["required_parameters"] == ["city"]

    def test_tool_definition_round_trip(self):
        """Test tool definition round-trip."""
        ir_tool: ToolDefinition = {
            "type": "function",
            "name": "search",
            "description": "Search the web",
            "parameters": {"type": "object", "properties": {}},
        }
        provider = GoogleGenAIToolOps.ir_tool_definition_to_p(ir_tool)
        restored = GoogleGenAIToolOps.p_tool_definition_to_ir(provider)
        assert restored["name"] == ir_tool["name"]
        assert restored["description"] == ir_tool["description"]

    # ==================== Tool Choice ====================

    def test_ir_tool_choice_auto(self):
        """Test IR auto tool choice → Google AUTO."""
        result = GoogleGenAIToolOps.ir_tool_choice_to_p({"mode": "auto"})
        assert result["function_calling_config"]["mode"] == "AUTO"

    def test_ir_tool_choice_none(self):
        """Test IR none tool choice → Google NONE."""
        result = GoogleGenAIToolOps.ir_tool_choice_to_p({"mode": "none"})
        assert result["function_calling_config"]["mode"] == "NONE"

    def test_ir_tool_choice_any(self):
        """Test IR any tool choice → Google ANY."""
        result = GoogleGenAIToolOps.ir_tool_choice_to_p({"mode": "any"})
        assert result["function_calling_config"]["mode"] == "ANY"

    def test_ir_tool_choice_tool(self):
        """Test IR specific tool choice → Google ANY with allowed_function_names."""
        result = GoogleGenAIToolOps.ir_tool_choice_to_p(
            {"mode": "tool", "tool_name": "get_weather"}
        )
        config = result["function_calling_config"]
        assert config["mode"] == "ANY"
        assert config["allowed_function_names"] == ["get_weather"]

    def test_p_tool_choice_auto(self):
        """Test Google AUTO → IR auto."""
        result = GoogleGenAIToolOps.p_tool_choice_to_ir(
            {"function_calling_config": {"mode": "AUTO"}}
        )
        assert result["mode"] == "auto"

    def test_p_tool_choice_none(self):
        """Test Google NONE → IR none."""
        result = GoogleGenAIToolOps.p_tool_choice_to_ir(
            {"function_calling_config": {"mode": "NONE"}}
        )
        assert result["mode"] == "none"

    def test_p_tool_choice_any_with_names(self):
        """Test Google ANY with allowed names → IR tool."""
        result = GoogleGenAIToolOps.p_tool_choice_to_ir(
            {
                "function_calling_config": {
                    "mode": "ANY",
                    "allowed_function_names": ["get_weather"],
                }
            }
        )
        assert result["mode"] == "tool"
        assert result["tool_name"] == "get_weather"

    def test_tool_choice_round_trip(self):
        """Test tool choice round-trip."""
        original = {"mode": "auto"}
        provider = GoogleGenAIToolOps.ir_tool_choice_to_p(original)
        restored = GoogleGenAIToolOps.p_tool_choice_to_ir(provider)
        assert restored["mode"] == original["mode"]

    # ==================== Tool Call ====================

    def test_ir_tool_call_to_p(self):
        """Test IR ToolCallPart → Google function_call Part."""
        ir_tc = ToolCallPart(
            type="tool_call",
            tool_call_id="call_123",
            tool_name="get_weather",
            tool_input={"location": "NYC"},
            tool_type="function",
        )
        result = GoogleGenAIToolOps.ir_tool_call_to_p(ir_tc)
        assert "function_call" in result
        assert result["function_call"]["name"] == "get_weather"
        assert result["function_call"]["args"] == {"location": "NYC"}

    def test_ir_tool_call_to_p_with_thought_signature(self):
        """Test IR ToolCallPart with thought_signature → Google Part."""
        ir_tc = {
            "type": "tool_call",
            "tool_call_id": "call_123",
            "tool_name": "get_weather",
            "tool_input": {},
            "provider_metadata": {"google": {"thought_signature": "sig123"}},
        }
        result = GoogleGenAIToolOps.ir_tool_call_to_p(ir_tc)
        assert result["thoughtSignature"] == "sig123"

    def test_p_tool_call_to_ir(self):
        """Test Google function_call Part → IR ToolCallPart."""
        provider = {
            "function_call": {
                "name": "get_weather",
                "args": {"location": "NYC"},
            }
        }
        result = GoogleGenAIToolOps.p_tool_call_to_ir(provider)
        assert result["type"] == "tool_call"
        assert result["tool_name"] == "get_weather"
        assert result["tool_input"] == {"location": "NYC"}
        assert result["tool_call_id"].startswith("call_get_weather_")

    def test_p_tool_call_to_ir_rest_api_format(self):
        """Test Google functionCall (REST API) → IR ToolCallPart."""
        provider = {
            "functionCall": {
                "name": "search",
                "args": {"query": "test"},
            }
        }
        result = GoogleGenAIToolOps.p_tool_call_to_ir(provider)
        assert result["tool_name"] == "search"

    def test_p_tool_call_to_ir_with_thought_signature(self):
        """Test Google function_call with thoughtSignature → IR ToolCallPart."""
        provider = {
            "function_call": {"name": "search", "args": {}},
            "thoughtSignature": "sig456",
        }
        result = GoogleGenAIToolOps.p_tool_call_to_ir(provider)
        assert result["provider_metadata"]["google"]["thought_signature"] == "sig456"

    def test_tool_call_round_trip(self):
        """Test tool call round-trip (name and input preserved)."""
        original = ToolCallPart(
            type="tool_call",
            tool_call_id="call_rt",
            tool_name="search",
            tool_input={"q": "test"},
            tool_type="function",
        )
        provider = GoogleGenAIToolOps.ir_tool_call_to_p(original)
        restored = GoogleGenAIToolOps.p_tool_call_to_ir(provider)
        assert restored["tool_name"] == original["tool_name"]
        assert restored["tool_input"] == original["tool_input"]

    # ==================== Tool Result ====================

    def test_ir_tool_result_to_p(self):
        """Test IR ToolResultPart → Google function_response Part."""
        ir_tr = ToolResultPart(
            type="tool_result",
            tool_call_id="call_123",
            result="Sunny, 25°C",
        )
        result = GoogleGenAIToolOps.ir_tool_result_to_p(ir_tr)
        assert "function_response" in result
        assert result["function_response"]["name"] == "call_123"
        assert result["function_response"]["response"]["output"] == "Sunny, 25°C"

    def test_ir_tool_result_to_p_error(self):
        """Test IR ToolResultPart with error → Google function_response Part."""
        ir_tr = ToolResultPart(
            type="tool_result",
            tool_call_id="call_err",
            result="API Error",
            is_error=True,
        )
        result = GoogleGenAIToolOps.ir_tool_result_to_p(ir_tr)
        assert result["function_response"]["response"]["error"] == "API Error"

    def test_ir_tool_result_to_p_with_context(self):
        """Test IR ToolResultPart with context lookup."""
        ir_input = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_123",
                        "tool_name": "get_weather",
                        "tool_input": {},
                    }
                ],
            }
        ]
        ir_tr = ToolResultPart(
            type="tool_result",
            tool_call_id="call_123",
            result="Sunny",
        )
        result = GoogleGenAIToolOps.ir_tool_result_to_p_with_context(ir_tr, ir_input)
        assert result["function_response"]["name"] == "get_weather"

    def test_ir_tool_result_to_p_with_context_no_match(self):
        """Test IR ToolResultPart with context but no matching call."""
        ir_input = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        ir_tr = ToolResultPart(
            type="tool_result",
            tool_call_id="nonexistent",
            result="data",
        )
        with pytest.warns(UserWarning, match="Could not find corresponding tool call"):
            result = GoogleGenAIToolOps.ir_tool_result_to_p_with_context(
                ir_tr, ir_input
            )
        assert result["function_response"]["name"] == "nonexistent"

    def test_p_tool_result_to_ir(self):
        """Test Google function_response Part → IR ToolResultPart."""
        provider = {
            "function_response": {
                "name": "get_weather",
                "response": {"output": "Sunny"},
            }
        }
        result = GoogleGenAIToolOps.p_tool_result_to_ir(provider)
        assert result["type"] == "tool_result"
        assert result["tool_call_id"] == "get_weather"
        assert result["result"] == "Sunny"
        assert result["is_error"] is False

    def test_p_tool_result_to_ir_error(self):
        """Test Google function_response error → IR ToolResultPart."""
        provider = {
            "function_response": {
                "name": "get_weather",
                "response": {"error": "API Error"},
            }
        }
        result = GoogleGenAIToolOps.p_tool_result_to_ir(provider)
        assert result["is_error"] is True
        assert result["result"] == "API Error"

    def test_p_tool_result_to_ir_rest_format(self):
        """Test Google functionResponse (REST) → IR ToolResultPart."""
        provider = {
            "functionResponse": {
                "name": "search",
                "response": {"output": "results"},
            }
        }
        result = GoogleGenAIToolOps.p_tool_result_to_ir(provider)
        assert result["tool_call_id"] == "search"

    # ==================== Tool Config ====================

    def test_ir_tool_config_to_p(self):
        """Test IR ToolCallConfig → Google tool config (empty)."""
        result = GoogleGenAIToolOps.ir_tool_config_to_p({"disable_parallel": True})
        assert result == {}

    def test_p_tool_config_to_ir(self):
        """Test Google tool config → IR ToolCallConfig (empty)."""
        result = GoogleGenAIToolOps.p_tool_config_to_ir({})
        assert result == {}
