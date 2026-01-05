"""
Tests for ToolConverter
"""

import pytest

from llmir.utils.tool_converter import ToolConverter


class TestToolConverter:
    """测试 ToolConverter 的工具定义和选择转换功能"""

    # ==================== Tool Definition Conversion ====================

    def test_convert_tool_definition_openai_chat(self):
        """测试转换为 OpenAI Chat 格式"""
        tool = {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather info",
            "parameters": {"type": "object", "properties": {}},
        }
        result = ToolConverter.convert_tool_definition(tool, "openai_chat")
        assert result == {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get weather info",
                "parameters": {"type": "object", "properties": {}},
            },
        }

    def test_convert_tool_definition_openai_chat_already_formatted(self):
        """测试已经是 OpenAI 格式的工具定义"""
        tool = {
            "type": "function",
            "function": {
                "name": "calculate",
                "description": "Calculate",
                "parameters": {},
            },
        }
        result = ToolConverter.convert_tool_definition(tool, "openai_chat")
        assert result == tool

    def test_convert_tool_definition_openai_chat_custom(self):
        """测试转换非 function 类型为 OpenAI Chat custom"""
        tool = {
            "type": "web_search",
            "name": "search",
            "description": "Search the web",
            "parameters": {},
        }
        result = ToolConverter.convert_tool_definition(tool, "openai_chat")
        assert result == {
            "type": "custom",
            "custom": {
                "name": "web_search_search",
                "description": "Search the web",
                "parameters": {},
            },
        }

    def test_convert_tool_definition_openai_responses(self):
        """测试转换为 OpenAI Responses 格式"""
        tool = {
            "type": "function",
            "name": "get_data",
            "description": "Get data",
            "parameters": {"type": "object"},
        }
        result = ToolConverter.convert_tool_definition(tool, "openai_responses")
        assert result == {
            "type": "function",
            "name": "get_data",
            "description": "Get data",
            "parameters": {"type": "object"},
            "strict": False,
        }

    def test_convert_tool_definition_openai_responses_custom(self):
        """测试转换非 function 类型为 OpenAI Responses custom"""
        tool = {
            "type": "mcp",
            "name": "mcp_tool",
            "description": "MCP tool",
            "parameters": {},
        }
        result = ToolConverter.convert_tool_definition(tool, "openai_responses")
        assert result == {
            "type": "custom",
            "name": "mcp_mcp_tool",
            "description": "MCP tool",
            "parameters": {},
        }

    def test_convert_tool_definition_anthropic(self):
        """测试转换为 Anthropic 格式"""
        tool = {
            "type": "function",
            "name": "analyze",
            "description": "Analyze data",
            "parameters": {
                "type": "object",
                "properties": {"data": {"type": "string"}},
            },
        }
        result = ToolConverter.convert_tool_definition(tool, "anthropic")
        assert result == {
            "name": "analyze",
            "description": "Analyze data",
            "input_schema": {
                "type": "object",
                "properties": {"data": {"type": "string"}},
            },
        }

    def test_convert_tool_definition_google(self):
        """测试转换为 Google 格式"""
        tool = {
            "type": "function",
            "name": "translate",
            "description": "Translate text",
            "parameters": {"type": "object"},
        }
        result = ToolConverter.convert_tool_definition(tool, "google")
        assert result == {
            "function_declarations": [
                {
                    "name": "translate",
                    "description": "Translate text",
                    "parameters": {"type": "object"},
                }
            ]
        }

    def test_convert_tool_definition_unsupported_format(self):
        """测试不支持的格式抛出异常"""
        tool = {"type": "function", "name": "test"}
        with pytest.raises(ValueError, match="Unsupported target format"):
            ToolConverter.convert_tool_definition(tool, "unsupported")

    # ==================== Tool Choice Conversion ====================

    def test_convert_tool_choice_openai_none(self):
        """测试转换 none 模式到 OpenAI"""
        tool_choice = {"mode": "none"}
        result = ToolConverter.convert_tool_choice(tool_choice, "openai")
        assert result == "none"

    def test_convert_tool_choice_openai_auto(self):
        """测试转换 auto 模式到 OpenAI"""
        tool_choice = {"mode": "auto"}
        result = ToolConverter.convert_tool_choice(tool_choice, "openai_chat")
        assert result == "auto"

    def test_convert_tool_choice_openai_any(self):
        """测试转换 any 模式到 OpenAI (映射为 required)"""
        tool_choice = {"mode": "any"}
        result = ToolConverter.convert_tool_choice(tool_choice, "openai_responses")
        assert result == "required"

    def test_convert_tool_choice_openai_required(self):
        """测试转换 required 模式到 OpenAI"""
        tool_choice = {"mode": "required"}
        result = ToolConverter.convert_tool_choice(tool_choice, "openai")
        assert result == "required"

    def test_convert_tool_choice_openai_tool(self):
        """测试转换 tool 模式到 OpenAI"""
        tool_choice = {"mode": "tool", "tool_name": "get_weather"}
        result = ToolConverter.convert_tool_choice(tool_choice, "openai")
        assert result == {"type": "function", "function": {"name": "get_weather"}}

    def test_convert_tool_choice_openai_function(self):
        """测试转换 function 模式到 OpenAI"""
        tool_choice = {"mode": "function", "tool_name": "calculate"}
        result = ToolConverter.convert_tool_choice(tool_choice, "openai")
        assert result == {"type": "function", "function": {"name": "calculate"}}

    def test_convert_tool_choice_openai_tool_with_function_field(self):
        """测试使用 function 字段的工具选择"""
        tool_choice = {"mode": "tool", "function": {"name": "search"}}
        result = ToolConverter.convert_tool_choice(tool_choice, "openai")
        assert result == {"type": "function", "function": {"name": "search"}}

    def test_convert_tool_choice_openai_tool_without_name(self):
        """测试没有工具名称时返回 required"""
        tool_choice = {"mode": "tool"}
        result = ToolConverter.convert_tool_choice(tool_choice, "openai")
        assert result == "required"

    def test_convert_tool_choice_openai_with_type_field(self):
        """测试使用 type 字段而非 mode"""
        tool_choice = {"type": "auto"}
        result = ToolConverter.convert_tool_choice(tool_choice, "openai")
        assert result == "auto"

    def test_convert_tool_choice_openai_unsupported_mode(self):
        """测试不支持的模式抛出异常"""
        tool_choice = {"mode": "unsupported"}
        with pytest.raises(ValueError, match="Unsupported tool choice mode"):
            ToolConverter.convert_tool_choice(tool_choice, "openai")

    def test_convert_tool_choice_anthropic_none(self):
        """测试转换 none 模式到 Anthropic"""
        tool_choice = {"mode": "none"}
        result = ToolConverter.convert_tool_choice(tool_choice, "anthropic")
        assert result == {"type": "none"}

    def test_convert_tool_choice_anthropic_auto(self):
        """测试转换 auto 模式到 Anthropic"""
        tool_choice = {"mode": "auto"}
        result = ToolConverter.convert_tool_choice(tool_choice, "anthropic")
        assert result == {"type": "auto"}

    def test_convert_tool_choice_anthropic_any(self):
        """测试转换 any 模式到 Anthropic"""
        tool_choice = {"mode": "any"}
        result = ToolConverter.convert_tool_choice(tool_choice, "anthropic")
        assert result == {"type": "any"}

    def test_convert_tool_choice_anthropic_tool(self):
        """测试转换 tool 模式到 Anthropic"""
        tool_choice = {"mode": "tool", "tool_name": "get_weather"}
        result = ToolConverter.convert_tool_choice(tool_choice, "anthropic")
        assert result == {"type": "tool", "name": "get_weather"}

    def test_convert_tool_choice_anthropic_tool_without_name(self):
        """测试没有工具名称的 tool 模式"""
        tool_choice = {"mode": "tool"}
        result = ToolConverter.convert_tool_choice(tool_choice, "anthropic")
        assert result == {"type": "tool"}

    def test_convert_tool_choice_anthropic_with_disable_parallel(self):
        """测试禁用并行工具使用"""
        tool_choice = {"mode": "auto", "disable_parallel": True}
        result = ToolConverter.convert_tool_choice(tool_choice, "anthropic")
        assert result == {"type": "auto", "disable_parallel_tool_use": True}

    def test_convert_tool_choice_anthropic_unsupported_mode(self):
        """测试不支持的模式抛出异常"""
        tool_choice = {"mode": "invalid"}
        with pytest.raises(ValueError, match="Unsupported tool choice mode"):
            ToolConverter.convert_tool_choice(tool_choice, "anthropic")

    def test_convert_tool_choice_google_none(self):
        """测试转换 none 模式到 Google"""
        tool_choice = {"mode": "none"}
        result = ToolConverter.convert_tool_choice(tool_choice, "google")
        assert result == {"function_calling_config": {"mode": "NONE"}}

    def test_convert_tool_choice_google_auto(self):
        """测试转换 auto 模式到 Google"""
        tool_choice = {"mode": "auto"}
        result = ToolConverter.convert_tool_choice(tool_choice, "google")
        assert result == {"function_calling_config": {"mode": "AUTO"}}

    def test_convert_tool_choice_google_any(self):
        """测试转换 any 模式到 Google"""
        tool_choice = {"mode": "any"}
        result = ToolConverter.convert_tool_choice(tool_choice, "google")
        assert result == {"function_calling_config": {"mode": "ANY"}}

    def test_convert_tool_choice_google_tool(self):
        """测试转换 tool 模式到 Google"""
        tool_choice = {"mode": "tool", "tool_name": "get_weather"}
        result = ToolConverter.convert_tool_choice(tool_choice, "google")
        assert result == {
            "function_calling_config": {
                "mode": "ANY",
                "allowed_function_names": ["get_weather"],
            }
        }

    def test_convert_tool_choice_google_tool_without_name(self):
        """测试没有工具名称的 tool 模式"""
        tool_choice = {"mode": "tool"}
        result = ToolConverter.convert_tool_choice(tool_choice, "google")
        assert result == {"function_calling_config": {"mode": "ANY"}}

    def test_convert_tool_choice_google_unknown_mode(self):
        """测试未知模式返回 None"""
        tool_choice = {"mode": "unknown"}
        result = ToolConverter.convert_tool_choice(tool_choice, "google")
        assert result is None

    def test_convert_tool_choice_unsupported_format(self):
        """测试不支持的格式抛出异常"""
        tool_choice = {"mode": "auto"}
        with pytest.raises(ValueError, match="Unsupported target format"):
            ToolConverter.convert_tool_choice(tool_choice, "unsupported")

    # ==================== Batch Conversion ====================

    def test_batch_convert_tools(self):
        """测试批量转换工具定义"""
        tools = [
            {"type": "function", "name": "tool1", "description": "Tool 1"},
            {"type": "function", "name": "tool2", "description": "Tool 2"},
        ]
        results = ToolConverter.batch_convert_tools(tools, "anthropic")
        assert len(results) == 2
        assert results[0] == {
            "name": "tool1",
            "description": "Tool 1",
            "input_schema": {},
        }
        assert results[1] == {
            "name": "tool2",
            "description": "Tool 2",
            "input_schema": {},
        }
