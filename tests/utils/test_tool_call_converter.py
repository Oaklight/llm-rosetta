"""
Tests for ToolCallConverter
"""

import json

import pytest

from llmir.utils.tool_call_converter import ToolCallConverter


class TestToolCallConverter:
    """测试 ToolCallConverter 的各种转换功能"""

    # ==================== IR → OpenAI Chat ====================

    def test_to_openai_chat_function(self):
        """测试转换为 OpenAI Chat function 类型"""
        tool_call = {
            "type": "tool_call",
            "tool_call_id": "call_123",
            "tool_name": "get_weather",
            "tool_input": {"city": "Beijing"},
            "tool_type": "function",
        }
        result = ToolCallConverter.to_openai_chat(tool_call)
        assert result == {
            "id": "call_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
        }

    def test_to_openai_chat_custom(self):
        """测试转换为 OpenAI Chat custom 类型"""
        tool_call = {
            "type": "tool_call",
            "tool_call_id": "call_456",
            "tool_name": "search",
            "tool_input": {"query": "test"},
            "tool_type": "web_search",
        }
        result = ToolCallConverter.to_openai_chat(tool_call)
        assert result == {
            "id": "call_456",
            "type": "custom",
            "custom": {
                "name": "web_search_search",
                "input": '{"query": "test"}',
            },
        }

    # ==================== IR → OpenAI Responses ====================

    def test_to_openai_responses_mcp_by_name(self):
        """测试通过名称识别 MCP 调用"""
        tool_call = {
            "type": "tool_call",
            "tool_call_id": "call_mcp",
            "tool_name": "mcp://server/tool",
            "tool_input": {"query": "test"},
            "tool_type": "function",
            "server_name": "my_server",
        }
        result = ToolCallConverter.to_openai_responses(tool_call)
        assert result == {
            "type": "mcp_call",
            "id": "call_mcp",
            "name": "mcp://server/tool",
            "arguments": '{"query": "test"}',
            "server_label": "my_server",
            "status": "calling",
        }

    def test_to_openai_responses_mcp_by_type(self):
        """测试通过类型识别 MCP 调用"""
        tool_call = {
            "type": "tool_call",
            "tool_call_id": "call_mcp2",
            "tool_name": "some_tool",
            "tool_input": {"param": "value"},
            "tool_type": "mcp",
        }
        result = ToolCallConverter.to_openai_responses(tool_call)
        assert result == {
            "type": "mcp_call",
            "id": "call_mcp2",
            "name": "some_tool",
            "arguments": '{"param": "value"}',
            "server_label": "default",
            "status": "calling",
        }

    def test_to_openai_responses_function(self):
        """测试转换为 function_call"""
        tool_call = {
            "type": "tool_call",
            "tool_call_id": "call_func",
            "tool_name": "calculate",
            "tool_input": {"x": 1, "y": 2},
            "tool_type": "function",
        }
        result = ToolCallConverter.to_openai_responses(tool_call)
        assert result == {
            "type": "function_call",
            "call_id": "call_func",
            "name": "calculate",
            "arguments": '{"x": 1, "y": 2}',
        }

    def test_to_openai_responses_web_search(self):
        """测试转换为 web_search"""
        tool_call = {
            "type": "tool_call",
            "tool_call_id": "call_search",
            "tool_name": "search",
            "tool_input": {"query": "Python tutorial"},
            "tool_type": "web_search",
        }
        result = ToolCallConverter.to_openai_responses(tool_call)
        assert result == {
            "type": "function_web_search",
            "call_id": "call_search",
            "query": "Python tutorial",
            "arguments": '{"query": "Python tutorial"}',
        }

    def test_to_openai_responses_code_interpreter(self):
        """测试转换为 code_interpreter"""
        tool_call = {
            "type": "tool_call",
            "tool_call_id": "call_code",
            "tool_name": "execute",
            "tool_input": {"code": "print('hello')"},
            "tool_type": "code_interpreter",
        }
        result = ToolCallConverter.to_openai_responses(tool_call)
        assert result == {
            "type": "code_interpreter_call",
            "call_id": "call_code",
            "code": "print('hello')",
            "arguments": '{"code": "print(\'hello\')"}',
        }

    def test_to_openai_responses_file_search(self):
        """测试转换为 file_search"""
        tool_call = {
            "type": "tool_call",
            "tool_call_id": "call_file",
            "tool_name": "search_files",
            "tool_input": {"query": "documentation"},
            "tool_type": "file_search",
        }
        result = ToolCallConverter.to_openai_responses(tool_call)
        assert result == {
            "type": "file_search_call",
            "call_id": "call_file",
            "query": "documentation",
            "arguments": '{"query": "documentation"}',
        }

    def test_to_openai_responses_unknown_type(self):
        """测试未知类型转换为默认 function_call"""
        tool_call = {
            "type": "tool_call",
            "tool_call_id": "call_unknown",
            "tool_name": "custom_tool",
            "tool_input": {"data": "test"},
            "tool_type": "unknown_type",
        }
        result = ToolCallConverter.to_openai_responses(tool_call)
        assert result == {
            "type": "function_call",
            "call_id": "call_unknown",
            "name": "unknown_type_custom_tool",
            "arguments": '{"data": "test"}',
        }

    # ==================== IR → Anthropic ====================

    def test_to_anthropic_function(self):
        """测试转换为 Anthropic tool_use"""
        tool_call = {
            "type": "tool_call",
            "tool_call_id": "call_123",
            "tool_name": "get_weather",
            "tool_input": {"city": "Beijing"},
            "tool_type": "function",
        }
        result = ToolCallConverter.to_anthropic(tool_call)
        assert result == {
            "type": "tool_use",
            "id": "call_123",
            "name": "get_weather",
            "input": {"city": "Beijing"},
        }

    def test_to_anthropic_web_search(self):
        """测试转换为 Anthropic server_tool_use (web_search)"""
        tool_call = {
            "type": "tool_call",
            "tool_call_id": "call_search",
            "tool_name": "search",
            "tool_input": {"query": "test"},
            "tool_type": "web_search",
        }
        result = ToolCallConverter.to_anthropic(tool_call)
        assert result == {
            "type": "server_tool_use",
            "id": "call_search",
            "name": "web_search",
            "input": {"query": "test"},
        }

    def test_to_anthropic_other_type(self):
        """测试其他类型转换为 Anthropic tool_use"""
        tool_call = {
            "type": "tool_call",
            "tool_call_id": "call_custom",
            "tool_name": "custom_tool",
            "tool_input": {"param": "value"},
            "tool_type": "custom_type",
        }
        result = ToolCallConverter.to_anthropic(tool_call)
        assert result == {
            "type": "tool_use",
            "id": "call_custom",
            "name": "custom_type_custom_tool",
            "input": {"param": "value"},
        }

    # ==================== IR → Google ====================

    def test_to_google_basic(self):
        """测试基本的 Google 转换"""
        tool_call = {
            "type": "tool_call",
            "tool_name": "get_weather",
            "tool_input": {"city": "Beijing"},
        }
        result = ToolCallConverter.to_google(tool_call)
        assert result == {
            "function_call": {"name": "get_weather", "args": {"city": "Beijing"}}
        }

    def test_to_google_with_thought_signature(self):
        """测试保留 thought_signature"""
        tool_call = {
            "type": "tool_call",
            "tool_name": "calculate",
            "tool_input": {"x": 1},
            "provider_metadata": {"google": {"thought_signature": "abc123"}},
        }
        result = ToolCallConverter.to_google(tool_call, preserve_metadata=True)
        assert result == {
            "function_call": {"name": "calculate", "args": {"x": 1}},
            "thoughtSignature": "abc123",
        }

    def test_to_google_without_preserving_metadata(self):
        """测试不保留 metadata"""
        tool_call = {
            "type": "tool_call",
            "tool_name": "calculate",
            "tool_input": {"x": 1},
            "provider_metadata": {"google": {"thought_signature": "abc123"}},
        }
        result = ToolCallConverter.to_google(tool_call, preserve_metadata=False)
        assert result == {"function_call": {"name": "calculate", "args": {"x": 1}}}
        assert "thoughtSignature" not in result

    # ==================== OpenAI Chat → IR ====================

    def test_from_openai_chat_function(self):
        """测试从 OpenAI Chat function 转换"""
        tool_call = {
            "id": "call_123",
            "type": "function",
            "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'},
        }
        result = ToolCallConverter.from_openai_chat(tool_call)
        assert result == {
            "type": "tool_call",
            "tool_call_id": "call_123",
            "tool_name": "get_weather",
            "tool_input": {"city": "Beijing"},
            "tool_type": "function",
        }

    def test_from_openai_chat_custom_with_underscore(self):
        """测试从 OpenAI Chat custom 转换（名称包含下划线）"""
        tool_call = {
            "id": "call_456",
            "type": "custom",
            "custom": {"name": "web_search_query", "input": '{"q": "test"}'},
        }
        result = ToolCallConverter.from_openai_chat(tool_call)
        assert result == {
            "type": "tool_call",
            "tool_call_id": "call_456",
            "tool_name": "search_query",
            "tool_input": {"q": "test"},
            "tool_type": "web",
        }

    def test_from_openai_chat_custom_without_underscore(self):
        """测试从 OpenAI Chat custom 转换（名称不包含下划线）"""
        tool_call = {
            "id": "call_789",
            "type": "custom",
            "custom": {"name": "mytool", "input": '{"data": "value"}'},
        }
        result = ToolCallConverter.from_openai_chat(tool_call)
        assert result == {
            "type": "tool_call",
            "tool_call_id": "call_789",
            "tool_name": "mytool",
            "tool_input": {"data": "value"},
            "tool_type": "custom",
        }

    def test_from_openai_chat_unsupported_type(self):
        """测试不支持的类型抛出异常"""
        tool_call = {"id": "call_999", "type": "unsupported", "data": {}}
        with pytest.raises(ValueError, match="Unsupported tool call type"):
            ToolCallConverter.from_openai_chat(tool_call)

    # ==================== Google → IR ====================

    def test_from_google_basic(self):
        """测试基本的 Google 转换"""
        part = {"function_call": {"name": "get_weather", "args": {"city": "Beijing"}}}
        result = ToolCallConverter.from_google(part)
        assert result["type"] == "tool_call"
        assert result["tool_name"] == "get_weather"
        assert result["tool_input"] == {"city": "Beijing"}
        assert result["tool_type"] == "function"
        assert "tool_call_id" in result
        assert result["tool_call_id"].startswith("call_get_weather_")

    def test_from_google_with_thought_signature(self):
        """测试保留 thoughtSignature"""
        part = {
            "function_call": {"name": "calculate", "args": {"x": 1}},
            "thoughtSignature": "abc123",
        }
        result = ToolCallConverter.from_google(part, preserve_metadata=True)
        assert result["provider_metadata"] == {
            "google": {"thought_signature": "abc123"}
        }

    def test_from_google_with_thought_signature_snake_case(self):
        """测试保留 thought_signature (snake_case)"""
        part = {
            "function_call": {"name": "calculate", "args": {"x": 1}},
            "thought_signature": "def456",
        }
        result = ToolCallConverter.from_google(part, preserve_metadata=True)
        assert result["provider_metadata"] == {
            "google": {"thought_signature": "def456"}
        }

    def test_from_google_without_preserving_metadata(self):
        """测试不保留 metadata"""
        part = {
            "function_call": {"name": "calculate", "args": {"x": 1}},
            "thoughtSignature": "abc123",
        }
        result = ToolCallConverter.from_google(part, preserve_metadata=False)
        assert "provider_metadata" not in result

    def test_from_google_functionCall_camelCase(self):
        """测试 functionCall (camelCase) 格式"""
        part = {"functionCall": {"name": "get_data", "args": {"id": 1}}}
        result = ToolCallConverter.from_google(part)
        assert result["tool_name"] == "get_data"
        assert result["tool_input"] == {"id": 1}

    def test_from_google_missing_function_call(self):
        """测试缺少 function_call 抛出异常"""
        part = {"some_other_field": "value"}
        with pytest.raises(ValueError, match="does not contain function_call"):
            ToolCallConverter.from_google(part)

    def test_from_google_with_existing_id(self):
        """测试使用已有的 ID"""
        part = {
            "function_call": {
                "id": "existing_id",
                "name": "tool",
                "args": {},
            }
        }
        result = ToolCallConverter.from_google(part)
        assert result["tool_call_id"] == "existing_id"
