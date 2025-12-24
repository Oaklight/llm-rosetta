"""
OpenAI Chat Completions转换器测试
"""

import pytest

from src.llm_provider_converter.converters.openai_chat_converter import (
    OpenAIChatConverter,
)


class TestOpenAIChatConverter:
    """OpenAI Chat转换器测试类"""

    def setup_method(self):
        """设置测试"""
        self.converter = OpenAIChatConverter()

    def test_simple_text_message_ir_to_provider(self):
        """测试简单文本消息IR到Provider转换"""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello, world!"}]}
        ]

        result, warnings = self.converter.to_provider(messages)

        assert "messages" in result
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"] == "Hello, world!"

    def test_simple_text_message_provider_to_ir(self):
        """测试简单文本消息Provider到IR转换"""
        provider_data = {"messages": [{"role": "user", "content": "Hello, world!"}]}

        result = self.converter.from_provider(provider_data)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Hello, world!"

    def test_system_message_conversion(self):
        """测试系统消息转换"""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant."}],
            },
            {"role": "user", "content": [{"type": "text", "text": "Hello!"}]},
        ]

        result, warnings = self.converter.to_provider(messages)

        assert len(result["messages"]) == 2
        assert result["messages"][0]["role"] == "system"
        assert result["messages"][0]["content"] == "You are a helpful assistant."
        assert result["messages"][1]["role"] == "user"
        assert result["messages"][1]["content"] == "Hello!"

    def test_multimodal_message_conversion(self):
        """测试多模态消息转换"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {
                        "type": "image",
                        "url": "https://example.com/image.jpg",
                        "media_type": "image/jpeg",
                    },
                ],
            }
        ]

        result, warnings = self.converter.to_provider(messages)

        assert len(result["messages"]) == 1
        message = result["messages"][0]
        assert message["role"] == "user"
        assert isinstance(message["content"], list)
        assert len(message["content"]) == 2

        # 检查文本部分
        text_part = message["content"][0]
        assert text_part["type"] == "text"
        assert text_part["text"] == "What's in this image?"

        # 检查图片部分
        image_part = message["content"][1]
        assert image_part["type"] == "image_url"
        assert image_part["image_url"]["url"] == "https://example.com/image.jpg"

    def test_tool_call_conversion(self):
        """测试工具调用转换"""
        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What's the weather?"}],
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "call_123",
                        "name": "get_weather",
                        "arguments": {"location": "New York"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call_123",
                        "content": "Sunny, 25°C",
                    }
                ],
            },
        ]

        result, warnings = self.converter.to_provider(messages)

        assert len(result["messages"]) == 3

        # 检查用户消息
        user_msg = result["messages"][0]
        assert user_msg["role"] == "user"
        assert user_msg["content"] == "What's the weather?"

        # 检查助手工具调用
        assistant_msg = result["messages"][1]
        assert assistant_msg["role"] == "assistant"
        assert assistant_msg["content"] is None
        assert "tool_calls" in assistant_msg
        assert len(assistant_msg["tool_calls"]) == 1

        tool_call = assistant_msg["tool_calls"][0]
        assert tool_call["id"] == "call_123"
        assert tool_call["type"] == "function"
        assert tool_call["function"]["name"] == "get_weather"
        assert tool_call["function"]["arguments"] == '{"location": "New York"}'

        # 检查工具结果（转换为tool角色消息）
        tool_msg = result["messages"][2]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_123"
        assert tool_msg["content"] == "Sunny, 25°C"

    def test_tool_definitions_conversion(self):
        """测试工具定义转换"""
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get current weather",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]

        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

        result, warnings = self.converter.to_provider(messages, tools=tools)

        assert "tools" in result
        assert len(result["tools"]) == 1

        tool = result["tools"][0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "get_weather"
        assert tool["function"]["description"] == "Get current weather"
        assert "parameters" in tool["function"]

    def test_tool_choice_conversion(self):
        """测试工具选择配置转换"""
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

        # 测试auto
        result, warnings = self.converter.to_provider(
            messages, tool_choice={"type": "auto"}
        )
        assert result["tool_choice"] == "auto"

        # 测试none
        result, warnings = self.converter.to_provider(
            messages, tool_choice={"type": "none"}
        )
        assert result["tool_choice"] == "none"

        # 测试required
        result, warnings = self.converter.to_provider(
            messages, tool_choice={"type": "required"}
        )
        assert result["tool_choice"] == "required"

        # 测试specific function
        result, warnings = self.converter.to_provider(
            messages,
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )
        assert result["tool_choice"]["type"] == "function"
        assert result["tool_choice"]["function"]["name"] == "get_weather"

    def test_round_trip_conversion(self):
        """测试往返转换"""
        original_messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are helpful."}],
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Hello"},
                    {"type": "image", "url": "https://example.com/image.jpg"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
        ]

        # IR -> Provider -> IR
        provider_data, warnings = self.converter.to_provider(original_messages)
        converted_back = self.converter.from_provider(provider_data)

        # 验证基本结构
        assert len(converted_back) == len(original_messages)

        # 验证系统消息
        assert converted_back[0]["role"] == "system"
        assert converted_back[0]["content"][0]["text"] == "You are helpful."

        # 验证用户消息
        assert converted_back[1]["role"] == "user"
        assert len(converted_back[1]["content"]) == 2
        assert converted_back[1]["content"][0]["text"] == "Hello"
        assert converted_back[1]["content"][1]["type"] == "image"

        # 验证助手消息
        assert converted_back[2]["role"] == "assistant"
        assert converted_back[2]["content"][0]["text"] == "Hi there!"

    def test_error_handling(self):
        """测试错误处理"""
        # 测试空消息列表
        result, warnings = self.converter.to_provider([])
        assert result["messages"] == []

        # 测试无效消息格式
        with pytest.raises(ValueError):
            self.converter.to_provider([{"invalid": "message"}])

    def test_provider_to_ir_with_tool_calls(self):
        """测试Provider到IR的工具调用转换"""
        provider_data = {
            "messages": [
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "NYC"}',
                            },
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call_123", "content": "Sunny, 20°C"},
            ]
        }

        result = self.converter.from_provider(provider_data)

        assert len(result) == 2

        # 检查助手消息
        assistant_msg = result[0]
        assert assistant_msg["role"] == "assistant"
        assert len(assistant_msg["content"]) == 1
        assert assistant_msg["content"][0]["type"] == "tool_call"
        assert assistant_msg["content"][0]["tool_call_id"] == "call_123"
        assert assistant_msg["content"][0]["tool_name"] == "get_weather"
        assert assistant_msg["content"][0]["tool_input"] == {"location": "NYC"}

        # 检查用户消息（工具结果）
        user_msg = result[1]
        assert user_msg["role"] == "user"
        assert len(user_msg["content"]) == 1
        assert user_msg["content"][0]["type"] == "tool_result"
        assert user_msg["content"][0]["tool_call_id"] == "call_123"
        assert user_msg["content"][0]["result"] == "Sunny, 20°C"
