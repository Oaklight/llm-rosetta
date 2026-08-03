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

    def test_ir_tool_definition_to_p_custom(self):
        """Custom tool type emits native Chat custom format."""
        ir_tool = cast(
            ToolDefinition,
            {
                "type": "custom",
                "name": "apply_patch",
                "description": "Apply a patch",
                "parameters": {},
                "required_parameters": [],
                "metadata": {
                    "format": {
                        "type": "grammar",
                        "grammar": {"definition": "start: /.+/s", "syntax": "lark"},
                    }
                },
            },
        )
        result = OpenAIChatToolOps.ir_tool_definition_to_p(ir_tool)
        assert result["type"] == "custom"
        assert result["custom"]["name"] == "apply_patch"
        assert result["custom"]["description"] == "Apply a patch"
        assert result["custom"]["format"]["type"] == "grammar"

    def test_ir_tool_definition_to_p_custom_via_metadata(self):
        """Custom tool detected from metadata.provider_type fallback."""
        ir_tool = cast(
            ToolDefinition,
            {
                "type": "function",
                "name": "apply_patch",
                "description": "",
                "parameters": {},
                "metadata": {"provider_type": "custom"},
            },
        )
        result = OpenAIChatToolOps.ir_tool_definition_to_p(ir_tool)
        assert result["type"] == "custom"
        assert result["custom"]["name"] == "apply_patch"

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

    def test_ir_tool_result_to_p_list_json_serialized(self):
        """Test list result is serialized via json.dumps, not str()."""
        ir_tr = cast(
            ToolResultPart,
            {
                "type": "tool_result",
                "tool_call_id": "call_list",
                "result": [{"type": "text", "text": "hello"}],
            },
        )
        result = OpenAIChatToolOps.ir_tool_result_to_p(ir_tr)
        assert result["content"] == json.dumps([{"type": "text", "text": "hello"}])
        # Verify it's valid JSON (not Python repr)
        parsed = json.loads(result["content"])
        assert parsed == [{"type": "text", "text": "hello"}]

    def test_ir_tool_result_to_p_dict_json_serialized(self):
        """Test dict result is serialized via json.dumps, not str()."""
        ir_tr = cast(
            ToolResultPart,
            {
                "type": "tool_result",
                "tool_call_id": "call_dict",
                "result": {"temperature": 72},
            },
        )
        result = OpenAIChatToolOps.ir_tool_result_to_p(ir_tr)
        assert result["content"] == '{"temperature": 72}'

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


class TestProviderMetadataPreservation:
    """Tests for generic provider_metadata round-trip through Chat format.

    Covers issue #401: responses_item_id must survive Chat converter boundary.
    """

    def test_ir_to_p_preserves_provider_metadata(self):
        """ir_tool_call_to_p stashes provider_metadata as _provider_metadata."""
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
        result = OpenAIChatToolOps.ir_tool_call_to_p(ir_part)
        assert result["_provider_metadata"] == {"responses_item_id": "fc_abc123"}

    def test_p_to_ir_restores_provider_metadata(self):
        """p_tool_call_to_ir restores _provider_metadata to provider_metadata."""
        p_call = {
            "id": "call_xyz",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "London"}'},
            "_provider_metadata": {"responses_item_id": "fc_abc123"},
        }
        result = OpenAIChatToolOps.p_tool_call_to_ir(p_call)
        assert result.get("provider_metadata") == {"responses_item_id": "fc_abc123"}

    def test_no_metadata_when_absent(self):
        """No _provider_metadata when IR part has no provider_metadata."""
        ir_part = cast(
            ToolCallPart,
            {
                "type": "tool_call",
                "tool_call_id": "call_xyz",
                "tool_name": "get_weather",
                "tool_input": {},
                "tool_type": "function",
            },
        )
        result = OpenAIChatToolOps.ir_tool_call_to_p(ir_part)
        assert "_provider_metadata" not in result

    def test_provider_metadata_round_trip(self):
        """provider_metadata survives IR → Chat → IR round-trip."""
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
        chat_call = OpenAIChatToolOps.ir_tool_call_to_p(ir_part)
        restored = OpenAIChatToolOps.p_tool_call_to_ir(chat_call)
        assert restored.get("provider_metadata") == {"responses_item_id": "fc_abc123"}


class TestCustomToolSupport:
    """Tests for native Chat Completions custom tool support."""

    # --- Definition ---

    def test_p_tool_definition_to_ir_custom(self):
        """Chat CustomToolChatCompletions → IR custom ToolDefinition."""
        provider_tool = {
            "type": "custom",
            "custom": {
                "name": "apply_patch",
                "description": "Apply a V4A patch",
                "format": {
                    "type": "grammar",
                    "grammar": {"definition": "start: /.+/s", "syntax": "lark"},
                },
            },
        }
        result = OpenAIChatToolOps.p_tool_definition_to_ir(provider_tool)
        assert result["type"] == "custom"
        assert result["name"] == "apply_patch"
        assert result["description"] == "Apply a V4A patch"
        assert result["metadata"]["format"]["type"] == "grammar"

    def test_p_tool_definition_to_ir_custom_minimal(self):
        """Custom tool with only name."""
        provider_tool = {"type": "custom", "custom": {"name": "my_tool"}}
        result = OpenAIChatToolOps.p_tool_definition_to_ir(provider_tool)
        assert result["type"] == "custom"
        assert result["name"] == "my_tool"
        assert result["parameters"] == {}

    def test_custom_tool_definition_round_trip(self):
        """Custom tool definition survives IR → Chat → IR."""
        ir_tool = cast(
            ToolDefinition,
            {
                "type": "custom",
                "name": "apply_patch",
                "description": "Apply a patch",
                "parameters": {},
                "metadata": {"format": {"type": "text"}},
            },
        )
        provider = OpenAIChatToolOps.ir_tool_definition_to_p(ir_tool)
        assert provider["type"] == "custom"
        restored = OpenAIChatToolOps.p_tool_definition_to_ir(provider)
        assert restored["type"] == "custom"
        assert restored["name"] == "apply_patch"
        assert restored["metadata"]["format"] == {"type": "text"}

    # --- Tool Call ---

    def test_ir_tool_call_to_p_custom(self):
        """IR custom tool call → Chat ChatCompletionMessageCustomToolCall."""
        ir_tc = ToolCallPart(
            type="tool_call",
            tool_call_id="call_1",
            tool_name="apply_patch",
            tool_input={"input": "*** Begin Patch\n+hi\n*** End Patch"},
            tool_type="custom",
        )
        result = OpenAIChatToolOps.ir_tool_call_to_p(ir_tc)
        assert result["type"] == "custom"
        assert result["custom"]["name"] == "apply_patch"
        assert result["custom"]["input"] == "*** Begin Patch\n+hi\n*** End Patch"
        assert result["id"] == "call_1"

    def test_p_tool_call_to_ir_custom(self):
        """Chat ChatCompletionMessageCustomToolCall → IR custom tool call."""
        provider_tc = {
            "id": "call_1",
            "type": "custom",
            "custom": {
                "name": "apply_patch",
                "input": "*** Begin Patch\n+hi\n*** End Patch",
            },
        }
        result = OpenAIChatToolOps.p_tool_call_to_ir(provider_tc)
        assert result["tool_type"] == "custom"
        assert result["tool_name"] == "apply_patch"
        assert result["tool_input"] == {"input": "*** Begin Patch\n+hi\n*** End Patch"}
        assert result["tool_call_id"] == "call_1"

    def test_custom_tool_call_round_trip(self):
        """Custom tool call survives IR → Chat → IR."""
        ir_tc = ToolCallPart(
            type="tool_call",
            tool_call_id="call_rt",
            tool_name="apply_patch",
            tool_input={"input": "patch content"},
            tool_type="custom",
        )
        provider = OpenAIChatToolOps.ir_tool_call_to_p(ir_tc)
        restored = OpenAIChatToolOps.p_tool_call_to_ir(provider)
        assert restored["tool_type"] == "custom"
        assert restored["tool_name"] == "apply_patch"
        assert restored["tool_input"] == {"input": "patch content"}

    # --- Tool Choice ---

    def test_ir_tool_choice_to_p_custom(self):
        """IR tool choice with tool_type=custom → Chat custom tool choice."""
        result = OpenAIChatToolOps.ir_tool_choice_to_p(
            {"mode": "tool", "tool_name": "apply_patch", "tool_type": "custom"}
        )
        assert result == {"type": "custom", "custom": {"name": "apply_patch"}}

    def test_p_tool_choice_to_ir_custom(self):
        """Chat custom tool choice → IR with tool_type=custom."""
        result = OpenAIChatToolOps.p_tool_choice_to_ir(
            {"type": "custom", "custom": {"name": "apply_patch"}}
        )
        assert result["mode"] == "tool"
        assert result["tool_name"] == "apply_patch"
        assert result["tool_type"] == "custom"

    def test_custom_tool_choice_round_trip(self):
        """Custom tool choice survives IR → Chat → IR."""
        ir = cast(
            ToolChoice,
            {"mode": "tool", "tool_name": "apply_patch", "tool_type": "custom"},
        )
        provider = OpenAIChatToolOps.ir_tool_choice_to_p(ir)
        restored = OpenAIChatToolOps.p_tool_choice_to_ir(provider)
        assert restored["mode"] == "tool"
        assert restored["tool_name"] == "apply_patch"
        assert restored["tool_type"] == "custom"
