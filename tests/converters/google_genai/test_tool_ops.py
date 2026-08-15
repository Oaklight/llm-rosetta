"""
Google GenAI ToolOps unit tests.
"""

import pytest

from llm_rosetta.converters.google_genai.tool_ops import GoogleGenAIToolOps
from typing import cast

from llm_rosetta.types.ir import (
    ToolCallPart,
    ToolChoice,
    ToolDefinition,
    ToolResultPart,
)


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

    def test_ir_tool_definition_to_p_strips_additional_properties(self):
        """Test additionalProperties is stripped for Google GenAI."""
        ir_tool: ToolDefinition = {
            "type": "function",
            "name": "create_item",
            "description": "Create an item",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string", "additionalProperties": False},
                    },
                },
                "required": ["name"],
                "additionalProperties": False,
            },
        }
        result = GoogleGenAIToolOps.ir_tool_definition_to_p(ir_tool)
        params = result["function_declarations"][0]["parameters"]
        assert "additionalProperties" not in params
        assert "additionalProperties" not in params["properties"]["tags"]["items"]

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
        assert result is not None
        assert not isinstance(result, list)
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
        assert restored is not None
        assert not isinstance(restored, list)
        assert restored["name"] == ir_tool["name"]
        assert restored["description"] == ir_tool["description"]

    def test_p_tool_definition_camelcase(self):
        """Test camelCase functionDeclarations (REST/Gemini CLI format)."""
        provider_tool = {
            "functionDeclarations": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                }
            ]
        }
        result = GoogleGenAIToolOps.p_tool_definition_to_ir(provider_tool)
        assert result is not None
        assert not isinstance(result, list)
        assert result["name"] == "get_weather"

    def test_p_tool_definition_multiple_declarations(self):
        """Test multiple function declarations in a single tool entry."""
        provider_tool = {
            "functionDeclarations": [
                {"name": "tool_a", "description": "A"},
                {"name": "tool_b", "description": "B"},
                {"name": "tool_c", "description": "C"},
            ]
        }
        result = GoogleGenAIToolOps.p_tool_definition_to_ir(provider_tool)
        assert isinstance(result, list)
        assert len(result) == 3
        assert result[0]["name"] == "tool_a"
        assert result[1]["name"] == "tool_b"
        assert result[2]["name"] == "tool_c"

    # ==================== Tool Choice ====================

    def test_ir_tool_choice_auto(self):
        """Test IR auto tool choice → Google AUTO."""
        result = GoogleGenAIToolOps.ir_tool_choice_to_p(
            cast(ToolChoice, {"mode": "auto"})
        )
        assert result is not None
        assert result["function_calling_config"]["mode"] == "AUTO"

    def test_ir_tool_choice_none(self):
        """Test IR none tool choice → Google NONE."""
        result = GoogleGenAIToolOps.ir_tool_choice_to_p(
            cast(ToolChoice, {"mode": "none"})
        )
        assert result is not None
        assert result["function_calling_config"]["mode"] == "NONE"

    def test_ir_tool_choice_any(self):
        """Test IR any tool choice → Google ANY."""
        result = GoogleGenAIToolOps.ir_tool_choice_to_p(
            cast(ToolChoice, {"mode": "any"})
        )
        assert result is not None
        assert result["function_calling_config"]["mode"] == "ANY"

    def test_ir_tool_choice_tool(self):
        """Test IR specific tool choice → Google ANY with allowed_function_names."""
        result = GoogleGenAIToolOps.ir_tool_choice_to_p(
            cast(ToolChoice, {"mode": "tool", "tool_name": "get_weather"})
        )
        assert result is not None
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
        original = cast(ToolChoice, {"mode": "auto"})
        provider = GoogleGenAIToolOps.ir_tool_choice_to_p(original)
        restored = GoogleGenAIToolOps.p_tool_choice_to_ir(provider)
        assert restored["mode"] == original["mode"]

    def test_p_tool_choice_camelcase(self):
        """Test camelCase functionCallingConfig (REST format)."""
        result = GoogleGenAIToolOps.p_tool_choice_to_ir(
            {"functionCallingConfig": {"mode": "ANY"}}
        )
        assert result["mode"] == "any"

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
        assert "functionCall" in result
        assert result["functionCall"]["name"] == "get_weather"
        assert result["functionCall"]["args"] == {"location": "NYC"}

    def test_ir_tool_call_to_p_with_thought_signature(self):
        """Test IR ToolCallPart with thought_signature → Google Part."""
        ir_tc = cast(
            ToolCallPart,
            {
                "type": "tool_call",
                "tool_call_id": "call_123",
                "tool_name": "get_weather",
                "tool_input": {},
                "provider_metadata": {"google": {"thought_signature": "sig123"}},
            },
        )
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
        assert result["tool_call_id"].startswith("call_")

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

    def test_ir_tool_call_to_p_includes_id(self):
        """Test that functionCall output includes the id field."""
        ir = ToolCallPart(
            type="tool_call",
            tool_call_id="fyh071wz",
            tool_name="get_weather",
            tool_input={"location": "Tokyo"},
            tool_type="function",
        )
        result = GoogleGenAIToolOps.ir_tool_call_to_p(ir)
        assert result["functionCall"]["id"] == "fyh071wz"
        assert result["functionCall"]["name"] == "get_weather"

    def test_p_tool_call_to_ir_with_provider_id(self):
        """Test that provider-supplied id (Gemini 3.x) is preserved."""
        provider_part = {
            "functionCall": {
                "name": "get_weather",
                "args": {"location": "Tokyo"},
                "id": "fyh071wz",
            }
        }
        ir = GoogleGenAIToolOps.p_tool_call_to_ir(provider_part)
        assert ir["tool_call_id"] == "fyh071wz"

    def test_tool_call_id_round_trip(self):
        """Test that functionCall.id round-trips through IR."""
        provider_part = {
            "functionCall": {
                "name": "get_weather",
                "args": {"location": "London"},
                "id": "wak1n9ou",
            }
        }
        ir = GoogleGenAIToolOps.p_tool_call_to_ir(provider_part)
        restored = GoogleGenAIToolOps.ir_tool_call_to_p(ir)
        assert restored["functionCall"]["id"] == "wak1n9ou"
        assert restored["functionCall"]["name"] == "get_weather"

    def test_tool_call_without_id_generates_one(self):
        """Test that missing id (Gemini 2.5) gets a generated call_ id."""
        provider_part = {
            "functionCall": {
                "name": "get_weather",
                "args": {"location": "Tokyo"},
            }
        }
        ir = GoogleGenAIToolOps.p_tool_call_to_ir(provider_part)
        assert ir["tool_call_id"].startswith("call_")

    # ==================== Tool Result ====================

    def test_ir_tool_result_to_p(self):
        """Test IR ToolResultPart → Google function_response Part."""
        ir_tr = ToolResultPart(
            type="tool_result",
            tool_call_id="call_123",
            result="Sunny, 25°C",
        )
        result = GoogleGenAIToolOps.ir_tool_result_to_p(ir_tr)
        assert "functionResponse" in result
        assert result["functionResponse"]["name"] == "call_123"
        assert result["functionResponse"]["response"]["output"] == "Sunny, 25°C"

    def test_ir_tool_result_to_p_error(self):
        """Test IR ToolResultPart with error → Google function_response Part."""
        ir_tr = ToolResultPart(
            type="tool_result",
            tool_call_id="call_err",
            result="API Error",
            is_error=True,
        )
        result = GoogleGenAIToolOps.ir_tool_result_to_p(ir_tr)
        assert result["functionResponse"]["response"]["error"] == "API Error"

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
        assert result["functionResponse"]["name"] == "get_weather"

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
        assert result["functionResponse"]["name"] == "nonexistent"

    def test_ir_tool_result_to_p_includes_id(self):
        """Test that functionResponse output includes the id field."""
        ir = ToolResultPart(
            type="tool_result",
            tool_call_id="fyh071wz",
            result="Sunny, 25C",
        )
        result = GoogleGenAIToolOps.ir_tool_result_to_p(ir)
        assert result["functionResponse"]["id"] == "fyh071wz"

    def test_ir_tool_result_to_p_with_context_includes_id(self):
        """Test that functionResponse with context includes the id field."""
        ir_input = [
            {
                "role": "assistant",
                "content": [
                    ToolCallPart(
                        type="tool_call",
                        tool_call_id="fyh071wz",
                        tool_name="get_weather",
                        tool_input={"location": "Tokyo"},
                        tool_type="function",
                    )
                ],
            }
        ]
        ir_tr = ToolResultPart(
            type="tool_result",
            tool_call_id="fyh071wz",
            result="Sunny, 25C",
        )
        result = GoogleGenAIToolOps.ir_tool_result_to_p_with_context(ir_tr, ir_input)
        assert result["functionResponse"]["name"] == "get_weather"
        assert result["functionResponse"]["id"] == "fyh071wz"

    def test_p_tool_result_to_ir_with_id(self):
        """Test that functionResponse.id is used as tool_call_id."""
        provider = {
            "functionResponse": {
                "name": "get_weather",
                "id": "fyh071wz",
                "response": {"output": "Sunny"},
            }
        }
        result = GoogleGenAIToolOps.p_tool_result_to_ir(provider)
        assert result["tool_call_id"] == "fyh071wz"

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

    def test_ir_tool_result_to_p_list_preserved(self):
        """Test list result is preserved as-is for Google Struct."""
        ir_tr = ToolResultPart(
            type="tool_result",
            tool_call_id="call_list",
            result=[{"type": "text", "text": "hello"}],
        )
        result = GoogleGenAIToolOps.ir_tool_result_to_p(ir_tr)
        output = result["functionResponse"]["response"]["output"]
        assert isinstance(output, list)
        assert output == [{"type": "text", "text": "hello"}]

    def test_ir_tool_result_to_p_with_context_list_preserved(self):
        """Test list result via context method is preserved as-is."""
        ir_input = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_list",
                        "tool_name": "plot",
                        "tool_input": {},
                    }
                ],
            }
        ]
        ir_tr = ToolResultPart(
            type="tool_result",
            tool_call_id="call_list",
            result=[{"type": "image", "image_url": "https://example.com/img.png"}],
        )
        result = GoogleGenAIToolOps.ir_tool_result_to_p_with_context(ir_tr, ir_input)
        output = result["functionResponse"]["response"]["output"]
        assert isinstance(output, list)
        assert output[0]["type"] == "image"

    def test_ir_tool_result_to_p_dict_json_serialized(self):
        """Test dict result is still serialized via json.dumps."""
        import json

        ir_tr = ToolResultPart(
            type="tool_result",
            tool_call_id="call_dict",
            result={"key": "value"},
        )
        result = GoogleGenAIToolOps.ir_tool_result_to_p(ir_tr)
        output = result["functionResponse"]["response"]["output"]
        assert isinstance(output, str)
        assert json.loads(output) == {"key": "value"}

    def test_p_tool_result_to_ir_preserves_list(self):
        """Test list content in function_response is preserved as-is in IR."""
        provider = {
            "functionResponse": {
                "name": "screenshot",
                "id": "call_mm",
                "response": {
                    "output": [
                        {"type": "text", "text": "captured"},
                        {"type": "image", "image_url": "https://example.com/img.png"},
                    ]
                },
            }
        }
        result = GoogleGenAIToolOps.p_tool_result_to_ir(provider)
        assert isinstance(result["result"], list)
        assert len(result["result"]) == 2
        assert result["result"][0]["type"] == "text"
        assert result["result"][1]["type"] == "image"

    def test_multimodal_tool_result_round_trip(self):
        """Test multimodal tool result round-trip: IR → Google → IR."""
        ir_tr = ToolResultPart(
            type="tool_result",
            tool_call_id="call_rt",
            result=[
                {"type": "text", "text": "chart output:"},
                {"type": "image", "image_url": "https://example.com/chart.png"},
            ],
        )
        google = GoogleGenAIToolOps.ir_tool_result_to_p(ir_tr)
        restored = GoogleGenAIToolOps.p_tool_result_to_ir(google)
        assert isinstance(restored["result"], list)
        assert len(restored["result"]) == 2
        assert restored["result"][0] == {"type": "text", "text": "chart output:"}
        assert restored["result"][1] == {
            "type": "image",
            "image_url": "https://example.com/chart.png",
        }

    # ==================== Tool Config ====================

    def test_ir_tool_config_to_p(self):
        """Test IR ToolCallConfig → Google tool config (empty)."""
        result = GoogleGenAIToolOps.ir_tool_config_to_p({"disable_parallel": True})
        assert result == {}

    def test_p_tool_config_to_ir(self):
        """Test Google tool config → IR ToolCallConfig (empty)."""
        result = GoogleGenAIToolOps.p_tool_config_to_ir({})
        assert result == {}

    # ==================== Schema Sanitization (#372) ====================

    def test_ir_tool_definition_strips_title(self):
        """title fields should be stripped for Google GenAI."""
        ir_tool: ToolDefinition = {
            "type": "function",
            "name": "run_cmd",
            "description": "Run a command",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"title": "Command", "type": "string"},
                    "cwd": {"title": "Cwd", "type": "string", "nullable": True},
                },
            },
        }
        result = GoogleGenAIToolOps.ir_tool_definition_to_p(ir_tool)
        params = result["function_declarations"][0]["parameters"]
        assert "title" not in params["properties"]["command"]
        assert "title" not in params["properties"]["cwd"]

    def test_ir_tool_definition_preserves_nullable(self):
        """Google GenAI supports nullable — it should be preserved."""
        ir_tool: ToolDefinition = {
            "type": "function",
            "name": "test",
            "description": "Test",
            "parameters": {
                "type": "object",
                "properties": {
                    "field": {"type": "string", "nullable": True},
                },
            },
        }
        result = GoogleGenAIToolOps.ir_tool_definition_to_p(ir_tool)
        params = result["function_declarations"][0]["parameters"]
        assert params["properties"]["field"]["nullable"] is True

    def test_p_tool_definition_uppercase_types_normalized(self):
        """Google-native uppercase types (STRING, OBJECT) are lowercased to IR."""
        provider_tool = {
            "functionDeclarations": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "location": {"type": "STRING"},
                            "days": {"type": "INTEGER"},
                            "detailed": {"type": "BOOLEAN"},
                        },
                        "required": ["location"],
                    },
                }
            ]
        }
        result = GoogleGenAIToolOps.p_tool_definition_to_ir(provider_tool)
        assert isinstance(result, dict)
        params = result["parameters"]
        assert params["type"] == "object"
        assert params["properties"]["location"]["type"] == "string"
        assert params["properties"]["days"]["type"] == "integer"
        assert params["properties"]["detailed"]["type"] == "boolean"

    def test_ir_tool_definition_uppercases_types_for_google(self):
        """IR lowercase types are uppercased when emitting to Google format."""
        ir_tool: ToolDefinition = {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string"},
                    "tags": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                },
                "required": ["location"],
            },
        }
        result = GoogleGenAIToolOps.ir_tool_definition_to_p(ir_tool)
        params = result["function_declarations"][0]["parameters"]
        assert params["type"] == "OBJECT"
        assert params["properties"]["location"]["type"] == "STRING"
        assert params["properties"]["tags"]["type"] == "ARRAY"
        assert params["properties"]["tags"]["items"]["type"] == "STRING"

    def test_schema_type_round_trip(self):
        """Uppercase Google types → IR lowercase → Google uppercase round-trip."""
        provider_tool = {
            "functionDeclarations": [
                {
                    "name": "search",
                    "description": "Search",
                    "parameters": {
                        "type": "OBJECT",
                        "properties": {
                            "query": {"type": "STRING"},
                            "limit": {"type": "NUMBER"},
                        },
                    },
                }
            ]
        }
        ir = GoogleGenAIToolOps.p_tool_definition_to_ir(provider_tool)
        assert isinstance(ir, dict)
        assert ir["parameters"]["properties"]["query"]["type"] == "string"
        restored = GoogleGenAIToolOps.ir_tool_definition_to_p(ir)
        params = restored["function_declarations"][0]["parameters"]
        assert params["properties"]["query"]["type"] == "STRING"
        assert params["properties"]["limit"]["type"] == "NUMBER"

    def test_p_tool_definition_bare_decl_uppercase_normalized(self):
        """Bare function declaration with uppercase types also normalized."""
        bare = {
            "name": "lookup",
            "description": "Look up",
            "parameters": {
                "type": "OBJECT",
                "properties": {"id": {"type": "INTEGER"}},
            },
        }
        result = GoogleGenAIToolOps.p_tool_definition_to_ir(bare)
        assert isinstance(result, dict)
        assert result["parameters"]["type"] == "object"
        assert result["parameters"]["properties"]["id"]["type"] == "integer"


class TestProviderMetadataPreservation:
    """Tests for generic provider_metadata round-trip through Google format.

    Covers issue #401: responses_item_id must survive Google converter boundary.
    """

    def test_ir_to_p_preserves_full_provider_metadata(self):
        """ir_tool_call_to_p stashes entire provider_metadata as _provider_metadata."""
        ir_part = cast(
            ToolCallPart,
            {
                "type": "tool_call",
                "tool_call_id": "call_xyz",
                "tool_name": "get_weather",
                "tool_input": {"city": "London"},
                "tool_type": "function",
                "provider_metadata": {"responses_item_id": "fc_abc123"},
            },
        )
        result = GoogleGenAIToolOps.ir_tool_call_to_p(ir_part)
        assert result["_provider_metadata"] == {"responses_item_id": "fc_abc123"}

    def test_p_to_ir_restores_provider_metadata(self):
        """p_tool_call_to_ir restores _provider_metadata to provider_metadata."""
        p_call = {
            "functionCall": {"name": "get_weather", "args": {"city": "London"}},
            "_provider_metadata": {"responses_item_id": "fc_abc123"},
        }
        result = GoogleGenAIToolOps.p_tool_call_to_ir(p_call)
        assert (
            result.get("provider_metadata", {}).get("responses_item_id") == "fc_abc123"
        )

    def test_thought_signature_merged_with_provider_metadata(self):
        """thought_signature is merged into existing provider_metadata."""
        p_call = {
            "functionCall": {"name": "get_weather", "args": {}},
            "thoughtSignature": "sig_123",
            "_provider_metadata": {"responses_item_id": "fc_abc123"},
        }
        result = GoogleGenAIToolOps.p_tool_call_to_ir(p_call)
        pm = result.get("provider_metadata", {})
        assert pm.get("responses_item_id") == "fc_abc123"
        assert pm.get("google", {}).get("thought_signature") == "sig_123"

    def test_provider_metadata_round_trip(self):
        """provider_metadata survives IR → Google → IR round-trip."""
        ir_part = cast(
            ToolCallPart,
            {
                "type": "tool_call",
                "tool_call_id": "call_xyz",
                "tool_name": "get_weather",
                "tool_input": {"city": "London"},
                "tool_type": "function",
                "provider_metadata": {"responses_item_id": "fc_abc123"},
            },
        )
        google_call = GoogleGenAIToolOps.ir_tool_call_to_p(ir_part)
        restored = GoogleGenAIToolOps.p_tool_call_to_ir(google_call)
        assert (
            restored.get("provider_metadata", {}).get("responses_item_id")
            == "fc_abc123"
        )
