"""
OpenAI Chat Completions转换器测试
"""

import pytest

from llmir.converters.openai_chat_converter import (
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

    def test_to_provider_with_extensions(self):
        """测试 to_provider 对扩展项的处理"""
        messages = [
            {"type": "system_event", "event_type": "start"},
            {
                "type": "tool_chain_node",
                "tool_call": {
                    "type": "tool_call",
                    "id": "tc_1",
                    "name": "tool1",
                    "arguments": {},
                },
            },
            {"type": "batch_marker"},
        ]
        result, warnings = self.converter.to_provider(messages)
        assert len(warnings) == 3
        assert "System event ignored" in warnings[0]
        assert "Tool chain converted" in warnings[1]
        assert "Extension item ignored" in warnings[2]

    def test_to_provider_with_reasoning_content(self):
        """测试 to_provider 对推理内容的处理"""
        messages = [
            {
                "role": "user",
                "content": [{"type": "reasoning", "reasoning": "user reasoning"}],
            },
            {
                "role": "assistant",
                "content": [{"type": "reasoning", "reasoning": "assistant reasoning"}],
            },
        ]
        result, warnings = self.converter.to_provider(messages)
        assert len(warnings) == 2
        assert all("Reasoning content not supported" in w for w in warnings)

    def test_to_provider_with_file_content(self):
        """测试 to_provider 对文件内容的处理"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file_data": {"data": "base64data"},
                        "file_name": "test.txt",
                    }
                ],
            }
        ]
        result, _ = self.converter.to_provider(messages)
        file_part = result["messages"][0]["content"][0]
        assert file_part["type"] == "file"
        assert file_part["file"]["file_data"] == "base64data"

    def test_to_provider_with_image_data(self):
        """测试 to_provider 对 image_data 的处理"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image_data": {"data": "img_data", "media_type": "image/png"},
                    }
                ],
            }
        ]
        result, _ = self.converter.to_provider(messages)
        img_part = result["messages"][0]["content"][0]
        assert img_part["type"] == "image_url"
        assert img_part["image_url"]["url"].startswith("data:image/png;base64,")

    def test_to_provider_assistant_with_text_and_tools(self):
        """测试 to_provider 对同时包含文本和工具调用的助手消息的处理"""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me check"},
                    {"type": "tool_call", "id": "c1", "name": "tool1", "arguments": {}},
                ],
            }
        ]
        result, _ = self.converter.to_provider(messages)
        msg = result["messages"][0]
        assert msg["content"] == "Let me check"
        assert len(msg["tool_calls"]) == 1

    def test_to_provider_assistant_empty_content(self):
        """测试 to_provider 对空内容助手消息的处理"""
        messages = [{"role": "assistant", "content": []}]
        result, _ = self.converter.to_provider(messages)
        assert result["messages"][0]["content"] == ""

    def test_from_provider_with_api_response(self):
        """测试 from_provider 对 API 响应格式的处理"""
        provider_data = {
            "choices": [{"message": {"role": "assistant", "content": "Hello from API"}}]
        }
        result = self.converter.from_provider(provider_data)
        assert result[0]["content"][0]["text"] == "Hello from API"

    def test_from_provider_with_streaming_response(self):
        """测试 from_provider 对流式响应的处理"""
        provider_data = {
            "choices": [{"delta": {"role": "assistant", "content": "Streaming"}}]
        }
        result = self.converter.from_provider(provider_data)
        assert result[0]["content"][0]["text"] == "Streaming"

    def test_from_provider_with_input_audio(self):
        """测试 from_provider 对音频输入的处理"""
        provider_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_audio",
                            "input_audio": {"data": "audio_data", "format": "wav"},
                        }
                    ],
                }
            ]
        }
        result = self.converter.from_provider(provider_data)
        assert result[0]["content"][0]["type"] == "file"
        assert result[0]["content"][0]["file_data"]["media_type"] == "audio/wav"

    def test_from_provider_with_function_role(self):
        """测试 from_provider 对已弃用的 function 角色的处理"""
        provider_data = {
            "messages": [{"role": "function", "name": "old_func", "content": "result"}]
        }
        result = self.converter.from_provider(provider_data)
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "tool_result"
        assert "legacy_function_old_func" in result[0]["content"][0]["tool_call_id"]

    def test_from_provider_with_base64_image(self):
        """测试 from_provider 对 base64 图像的处理"""
        provider_data = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,abc123",
                                "detail": "high",
                            },
                        }
                    ],
                }
            ]
        }
        result = self.converter.from_provider(provider_data)
        img = result[0]["content"][0]
        assert img["type"] == "image"
        assert img["image_data"]["data"] == "abc123"
        assert img["image_data"]["media_type"] == "image/jpeg"

    def test_from_provider_with_pydantic_and_invalid_data(self):
        """测试 from_provider 对 Pydantic 模型和无效数据的处理"""

        class MockPydantic:
            def model_dump(self):
                return {"messages": []}

        result = self.converter.from_provider(MockPydantic())
        assert result == []

        with pytest.raises(ValueError, match="OpenAI data must be a dictionary"):
            self.converter.from_provider("not a dict")

    def test_image_conversion_errors(self):
        """测试图像转换的错误处理"""
        with pytest.raises(ValueError, match="must have either"):
            self.converter._convert_image_to_openai({"type": "image"})

    def test_file_conversion_errors(self):
        """测试文件转换的错误处理"""
        with pytest.raises(ValueError, match="must have either"):
            self.converter._convert_file_to_openai({"type": "file"})
