"""Tests for Google Interactions tool_ops."""

from llm_rosetta.converters.google_interactions.tool_ops import (
    GoogleInteractionsToolOps,
)


class TestToolDefinitionConversion:
    def test_ir_function_to_provider(self):
        ir = {
            "type": "function",
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
        result = GoogleInteractionsToolOps.ir_tool_to_p(ir)  # ty: ignore
        assert result["type"] == "function"
        assert result["name"] == "get_weather"
        assert result["description"] == "Get current weather"
        assert result["parameters"]["properties"]["location"]["type"] == "string"

    def test_provider_function_to_ir(self):
        p = {
            "type": "function",
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        }
        result = GoogleInteractionsToolOps.p_tool_to_ir(p)
        assert result["type"] == "function"
        assert result["name"] == "get_weather"
        assert result["description"] == "Get current weather"

    def test_provider_mcp_server_to_ir(self):
        p = {
            "type": "mcp_server",
            "name": "weather_service",
            "url": "https://example.com/mcp",
        }
        result = GoogleInteractionsToolOps.p_tool_to_ir(p)
        assert result["type"] == "mcp"
        assert result["name"] == "weather_service"
        assert result["metadata"]["url"] == "https://example.com/mcp"

    def test_roundtrip_function_tool(self):
        ir = {
            "type": "function",
            "name": "calc",
            "description": "Calculate",
            "parameters": {"type": "object", "properties": {}},
        }
        p = GoogleInteractionsToolOps.ir_tool_to_p(ir)  # ty: ignore
        back = GoogleInteractionsToolOps.p_tool_to_ir(p)
        assert back["name"] == ir["name"]
        assert back["description"] == ir["description"]


class TestFunctionCallConversion:
    def test_provider_function_call_to_ir(self):
        step = {
            "type": "function_call",
            "id": "call_123",
            "name": "get_weather",
            "arguments": {"location": "San Francisco"},
        }
        result = GoogleInteractionsToolOps.p_function_call_to_ir(step)
        assert result["type"] == "tool_call"
        assert result["tool_call_id"] == "call_123"
        assert result["tool_name"] == "get_weather"
        assert result["tool_input"] == {"location": "San Francisco"}

    def test_ir_tool_call_to_provider(self):
        ir = {
            "type": "tool_call",
            "tool_call_id": "call_456",
            "tool_name": "search",
            "tool_input": {"query": "hello"},
        }
        result = GoogleInteractionsToolOps.ir_tool_call_to_p(ir)  # ty: ignore
        assert result["type"] == "function_call"
        assert result["id"] == "call_456"
        assert result["name"] == "search"
        assert result["arguments"] == {"query": "hello"}

    def test_roundtrip_function_call(self):
        step = {
            "type": "function_call",
            "id": "call_rt",
            "name": "fn",
            "arguments": {"x": 1},
        }
        ir = GoogleInteractionsToolOps.p_function_call_to_ir(step)
        back = GoogleInteractionsToolOps.ir_tool_call_to_p(ir)
        assert back["id"] == step["id"]
        assert back["name"] == step["name"]
        assert back["arguments"] == step["arguments"]


class TestFunctionResultConversion:
    def test_provider_function_result_to_ir(self):
        step = {
            "type": "function_result",
            "call_id": "call_123",
            "result": [{"type": "text", "text": '{"weather":"sunny"}'}],
        }
        result = GoogleInteractionsToolOps.p_function_result_to_ir(step)
        assert result["type"] == "tool_result"
        assert result["tool_call_id"] == "call_123"
        assert "sunny" in result["result"]

    def test_provider_function_result_string(self):
        step = {
            "type": "function_result",
            "call_id": "call_456",
            "result": "success",
        }
        result = GoogleInteractionsToolOps.p_function_result_to_ir(step)
        assert result["result"] == "success"

    def test_provider_function_result_dict(self):
        step = {
            "type": "function_result",
            "call_id": "call_789",
            "result": {"status": "ok"},
        }
        result = GoogleInteractionsToolOps.p_function_result_to_ir(step)
        assert '"status"' in result["result"]

    def test_provider_function_result_error(self):
        step = {
            "type": "function_result",
            "call_id": "call_err",
            "result": "error occurred",
            "is_error": True,
        }
        result = GoogleInteractionsToolOps.p_function_result_to_ir(step)
        assert result["is_error"] is True

    def test_ir_tool_result_to_provider(self):
        ir = {
            "type": "tool_result",
            "tool_call_id": "call_123",
            "result": "42",
        }
        result = GoogleInteractionsToolOps.ir_tool_result_to_p(ir)  # ty: ignore
        assert result["type"] == "function_result"
        assert result["call_id"] == "call_123"
        assert result["result"] == "42"

    def test_roundtrip_function_result(self):
        ir = {
            "type": "tool_result",
            "tool_call_id": "call_rt",
            "result": "done",
        }
        p = GoogleInteractionsToolOps.ir_tool_result_to_p(ir)  # ty: ignore
        back = GoogleInteractionsToolOps.p_function_result_to_ir(p)
        assert back["tool_call_id"] == ir["tool_call_id"]
        assert back["result"] == ir["result"]


class TestToolChoiceConversion:
    def test_string_modes(self):
        for mode in ("auto", "any", "none"):
            ir = {"mode": mode}
            p = GoogleInteractionsToolOps.ir_tool_choice_to_p(ir)  # ty: ignore
            assert p == mode
            back = GoogleInteractionsToolOps.p_tool_choice_to_ir(p)
            assert back["mode"] == mode

    def test_specific_tool(self):
        ir = {"mode": "tool", "tool_name": "get_weather"}
        p = GoogleInteractionsToolOps.ir_tool_choice_to_p(ir)  # ty: ignore
        assert isinstance(p, dict)
        assert p["allowed_tools"]["tools"] == ["get_weather"]

    def test_provider_specific_tool_to_ir(self):
        p = {"allowed_tools": {"mode": "any", "tools": ["search"]}}
        result = GoogleInteractionsToolOps.p_tool_choice_to_ir(p)
        assert result["mode"] == "tool"
        assert result["tool_name"] == "search"
