"""
测试Anthropic转换器
"""

import pytest

from src.llm_provider_converter.converters import AnthropicConverter
from src.llm_provider_converter.types.ir import IRInput, ToolChoice, ToolDefinition


class TestAnthropicConverter:
    """Anthropic转换器测试"""

    def setup_method(self):
        """设置测试环境"""
        self.converter = AnthropicConverter()

    def test_simple_text_message(self):
        """测试简单文本消息转换"""
        ir_input: IRInput = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
        ]

        result, warnings = self.converter.to_provider(ir_input)

        # 验证结果结构
        assert "messages" in result
        assert len(result["messages"]) == 2
        assert len(warnings) == 0

        # 验证第一条消息
        msg1 = result["messages"][0]
        assert msg1["role"] == "user"
        assert len(msg1["content"]) == 1
        assert msg1["content"][0]["type"] == "text"
        assert msg1["content"][0]["text"] == "Hello"

        # 验证第二条消息
        msg2 = result["messages"][1]
        assert msg2["role"] == "assistant"
        assert len(msg2["content"]) == 1
        assert msg2["content"][0]["type"] == "text"
        assert msg2["content"][0]["text"] == "Hi there!"

    def test_system_message(self):
        """测试system消息转换"""
        ir_input: IRInput = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ]

        result, warnings = self.converter.to_provider(ir_input)

        # 验证system消息被正确处理
        assert "system" in result
        assert len(result["system"]) == 1
        assert result["system"][0]["type"] == "text"
        assert result["system"][0]["text"] == "You are a helpful assistant."

        # 验证普通消息
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"

    def test_tool_call_conversion(self):
        """测试工具调用转换"""
        ir_input: IRInput = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me search for that"},
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_123",
                        "tool_name": "web_search",
                        "tool_input": {"query": "AI news"},
                        "tool_type": "function",
                    },
                ],
            }
        ]

        result, warnings = self.converter.to_provider(ir_input)

        # 验证消息结构
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert len(msg["content"]) == 2

        # 验证文本部分
        text_block = msg["content"][0]
        assert text_block["type"] == "text"
        assert text_block["text"] == "Let me search for that"

        # 验证工具调用部分
        tool_block = msg["content"][1]
        assert tool_block["type"] == "tool_use"
        assert tool_block["id"] == "call_123"
        assert tool_block["name"] == "web_search"
        assert tool_block["input"] == {"query": "AI news"}

    def test_tool_result_conversion(self):
        """测试工具结果转换"""
        ir_input: IRInput = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call_123",
                        "result": "Latest AI news: ...",
                    }
                ],
            }
        ]

        result, warnings = self.converter.to_provider(ir_input)

        # 验证工具结果转换
        msg = result["messages"][0]
        assert msg["role"] == "user"
        tool_result = msg["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_use_id"] == "call_123"
        assert tool_result["content"] == "Latest AI news: ..."

    def test_tool_definitions(self):
        """测试工具定义转换"""
        tools: list[ToolDefinition] = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ]

        ir_input: IRInput = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What's the weather?"}],
            }
        ]

        result, warnings = self.converter.to_provider(ir_input, tools=tools)

        # 验证工具定义
        assert "tools" in result
        assert len(result["tools"]) == 1

        tool = result["tools"][0]
        assert tool["name"] == "get_weather"
        assert tool["description"] == "Get weather information"
        assert "input_schema" in tool

    def test_tool_choice_conversion(self):
        """测试工具选择转换"""
        tool_choice: ToolChoice = {"mode": "auto", "disable_parallel": True}

        ir_input: IRInput = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        ]

        result, warnings = self.converter.to_provider(ir_input, tool_choice=tool_choice)

        # 验证工具选择
        assert "tool_choice" in result
        choice = result["tool_choice"]
        assert choice["type"] == "auto"
        assert choice["disable_parallel_tool_use"] is True

    def test_round_trip_conversion(self):
        """测试往返转换"""
        original_ir: IRInput = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are helpful."}],
            },
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        ]

        # IR -> Anthropic
        anthropic_data, warnings = self.converter.to_provider(original_ir)

        # Anthropic -> IR
        converted_ir = self.converter.from_provider(anthropic_data)

        # 验证往返转换
        assert len(converted_ir) == len(original_ir)

        # 验证每条消息
        for i, (orig, conv) in enumerate(zip(original_ir, converted_ir)):
            assert orig["role"] == conv["role"]
            assert len(orig["content"]) == len(conv["content"])

            for j, (orig_part, conv_part) in enumerate(
                zip(orig["content"], conv["content"])
            ):
                assert orig_part["type"] == conv_part["type"]
                if orig_part["type"] == "text":
                    assert orig_part["text"] == conv_part["text"]

    def test_extension_item_handling(self):
        """测试扩展项处理"""
        ir_input: IRInput = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {
                "type": "system_event",
                "event_type": "session_start",
                "timestamp": "2024-01-01T00:00:00Z",
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        ]

        result, warnings = self.converter.to_provider(ir_input)

        # 验证扩展项被忽略但产生警告
        assert len(result["messages"]) == 2  # 只有普通消息
        assert len(warnings) > 0  # 应该有警告
        assert any("System event ignored" in w for w in warnings)

    def test_validation_errors(self):
        """测试输入验证"""
        # 测试无效输入
        invalid_ir = [
            {"invalid": "data"}  # 既没有role也没有type
        ]

        with pytest.raises(ValueError):
            self.converter.to_provider(invalid_ir)

        # 测试缺少必需字段的消息
        invalid_message = [
            {"role": "user"}  # 缺少content字段
        ]

        with pytest.raises(ValueError):
            self.converter.to_provider(invalid_message)
