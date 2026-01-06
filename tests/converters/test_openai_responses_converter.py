"""
OpenAI Responses API转换器测试
"""

import pytest

from llmir.converters.openai_responses_converter import (
    OpenAIResponsesConverter,
)
from llmir.types.ir import ToolCallPart


class TestOpenAIResponsesConverter:
    """OpenAI Responses转换器测试类"""

    def setup_method(self):
        """设置测试"""
        self.converter = OpenAIResponsesConverter()

    def test_simple_text_message_ir_to_provider(self):
        """测试简单文本消息IR到Provider转换"""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello, world!"}]}
        ]

        result, warnings = self.converter.to_provider(messages)

        assert "input" in result
        assert len(result["input"]) == 1

        item = result["input"][0]
        assert item["type"] == "message"
        assert item["role"] == "user"
        assert item["content"] == [{"type": "input_text", "text": "Hello, world!"}]

    def test_simple_text_message_provider_to_ir(self):
        """测试简单文本消息Provider到IR转换"""
        provider_data = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello, world!"}],
                }
            ]
        }

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

        assert len(result["input"]) == 2

        # 检查系统消息
        system_item = result["input"][0]
        assert system_item["type"] == "message"
        assert system_item["role"] == "system"
        assert system_item["content"] == [
            {"type": "input_text", "text": "You are a helpful assistant."}
        ]

        # 检查用户消息
        user_item = result["input"][1]
        assert user_item["type"] == "message"
        assert user_item["role"] == "user"
        assert user_item["content"] == [{"type": "input_text", "text": "Hello!"}]

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

        assert len(result["input"]) == 1
        item = result["input"][0]
        assert item["type"] == "message"
        assert item["role"] == "user"
        assert len(item["content"]) == 2

        # 检查文本部分
        text_part = item["content"][0]
        assert text_part["type"] == "input_text"
        assert text_part["text"] == "What's in this image?"

        # 检查图片部分
        image_part = item["content"][1]
        assert image_part["type"] == "input_image"
        assert image_part["image_url"] == "https://example.com/image.jpg"

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

        assert len(result["input"]) == 3

        # 检查用户消息
        user_item = result["input"][0]
        assert user_item["type"] == "message"
        assert user_item["role"] == "user"

        # 检查工具调用
        tool_call_item = result["input"][1]
        assert tool_call_item["type"] == "function_call"
        assert tool_call_item["call_id"] == "call_123"
        assert tool_call_item["name"] == "get_weather"
        assert tool_call_item["arguments"] == '{"location": "New York"}'

        # 检查工具结果
        tool_result_item = result["input"][2]
        assert tool_result_item["type"] == "function_call_output"
        assert tool_result_item["call_id"] == "call_123"
        assert tool_result_item["output"] == "Sunny, 25°C"

    def test_mcp_call_conversion(self):
        """测试MCP调用转换"""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "id": "mcp_123",
                        "name": "mcp://server/tool",
                        "arguments": {"param": "value"},
                    }
                ],
            }
        ]

        result, warnings = self.converter.to_provider(messages)

        assert len(result["input"]) == 1
        item = result["input"][0]
        assert item["type"] == "mcp_call"
        assert item["id"] == "mcp_123"
        assert item["name"] == "mcp://server/tool"
        assert item["arguments"] == '{"param": "value"}'

    def test_reasoning_content_conversion(self):
        """测试推理内容转换"""
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": "Let me think about this...",
                        "reasoning": True,
                    },
                    {"type": "text", "text": "The answer is 42."},
                ],
            }
        ]

        result, warnings = self.converter.to_provider(messages)

        assert len(result["input"]) == 2

        # 检查推理内容
        reasoning_item = result["input"][0]
        assert reasoning_item["type"] == "reasoning"
        assert reasoning_item["content"] == "Let me think about this..."

        # 检查普通消息 - assistant角色的文本使用output_text
        message_item = result["input"][1]
        assert message_item["type"] == "message"
        assert message_item["role"] == "assistant"
        assert message_item["content"] == [
            {"type": "output_text", "text": "The answer is 42."}
        ]

    def test_tool_definitions_conversion(self):
        """测试工具定义转换"""
        # 使用IR格式的工具定义（与examples一致）
        tools = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            }
        ]

        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

        result, warnings = self.converter.to_provider(messages, tools=tools)

        assert "tools" in result
        assert len(result["tools"]) == 1

        tool = result["tools"][0]
        assert tool["type"] == "function"
        assert tool["name"] == "get_weather"
        assert tool["description"] == "Get current weather"
        assert "parameters" in tool

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

    def test_provider_to_ir_with_function_calls(self):
        """测试Provider到IR的函数调用转换"""
        provider_data = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "text", "text": "What's the weather?"}],
                },
                {
                    "type": "function_call",
                    "id": "call_123",
                    "name": "get_weather",
                    "arguments": {"location": "NYC"},
                },
                {
                    "type": "function_call_output",
                    "call_id": "call_123",
                    "output": "Sunny, 20°C",
                },
            ]
        }

        result = self.converter.from_provider(provider_data)

        assert len(result) == 3

        # 检查用户消息
        user_msg = result[0]
        assert user_msg["role"] == "user"
        assert user_msg["content"][0]["text"] == "What's the weather?"

        # 检查助手工具调用
        assistant_msg = result[1]
        assert assistant_msg["role"] == "assistant"
        assert len(assistant_msg["content"]) == 1
        assert assistant_msg["content"][0]["type"] == "tool_call"
        assert assistant_msg["content"][0]["tool_call_id"] == "call_123"
        assert assistant_msg["content"][0]["tool_name"] == "get_weather"
        assert assistant_msg["content"][0]["tool_input"] == {"location": "NYC"}

        # 检查用户工具结果
        tool_result_msg = result[2]
        assert tool_result_msg["role"] == "user"
        assert len(tool_result_msg["content"]) == 1
        assert tool_result_msg["content"][0]["type"] == "tool_result"
        assert tool_result_msg["content"][0]["tool_call_id"] == "call_123"
        assert tool_result_msg["content"][0]["result"] == "Sunny, 20°C"

    def test_provider_to_ir_with_mcp_calls(self):
        """测试Provider到IR的MCP调用转换"""
        provider_data = {
            "input": [
                {
                    "type": "mcp_call",
                    "id": "mcp_123",
                    "server": "weather_server",
                    "tool": "get_weather",
                    "arguments": {"location": "NYC"},
                },
                {
                    "type": "mcp_call_output",
                    "call_id": "mcp_123",
                    "output": {"temperature": 20, "condition": "sunny"},
                },
            ]
        }

        result = self.converter.from_provider(provider_data)

        assert len(result) == 2

        # 检查MCP调用
        mcp_call_msg = result[0]
        assert mcp_call_msg["role"] == "assistant"
        assert len(mcp_call_msg["content"]) == 1
        assert mcp_call_msg["content"][0]["type"] == "tool_call"
        assert mcp_call_msg["content"][0]["tool_call_id"] == "mcp_123"
        assert (
            mcp_call_msg["content"][0]["tool_name"]
            == "mcp://weather_server/get_weather"
        )
        assert mcp_call_msg["content"][0]["tool_input"] == {"location": "NYC"}

        # 检查MCP结果
        mcp_result_msg = result[1]
        assert mcp_result_msg["role"] == "user"
        assert len(mcp_result_msg["content"]) == 1
        assert mcp_result_msg["content"][0]["type"] == "tool_result"
        assert mcp_result_msg["content"][0]["tool_call_id"] == "mcp_123"
        assert mcp_result_msg["content"][0]["result"] == {
            "temperature": 20,
            "condition": "sunny",
        }

    def test_provider_to_ir_with_reasoning(self):
        """测试Provider到IR的推理内容转换"""
        # 这应该是API的output格式，不是input格式
        provider_data = {
            "output": [
                {"type": "reasoning", "content": "Let me think step by step..."},
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "The answer is 42."}],
                },
            ]
        }

        result = self.converter.from_provider(provider_data)

        assert len(result) == 2  # 推理内容和普通消息分开

        # 检查推理消息 - reasoning类型的item会被转换为带reasoning标记的内容
        reasoning_msg = result[0]
        assert reasoning_msg["role"] == "assistant"
        assert len(reasoning_msg["content"]) == 1
        reasoning_part = reasoning_msg["content"][0]
        # 推理内容保持为"reasoning"类型，使用"reasoning"字段存储内容
        assert reasoning_part["type"] == "reasoning"
        assert reasoning_part["reasoning"] == "Let me think step by step..."

        # 检查普通消息
        text_msg = result[1]
        assert text_msg["role"] == "assistant"
        assert len(text_msg["content"]) == 1
        text_part = text_msg["content"][0]
        assert text_part["type"] == "text"
        assert text_part["text"] == "The answer is 42."

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
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Let me think...", "reasoning": True},
                    {"type": "text", "text": "Hi there!"},
                ],
            },
        ]

        # IR -> Provider -> IR
        provider_data, warnings = self.converter.to_provider(original_messages)
        converted_back = self.converter.from_provider(provider_data)

        # 验证基本结构 - 推理内容会被分离，所以消息数量会增加
        assert len(converted_back) >= len(original_messages)

        # 验证系统消息
        assert converted_back[0]["role"] == "system"
        assert converted_back[0]["content"][0]["text"] == "You are helpful."

        # 验证用户消息 - 可能包含推理内容，所以内容数量可能不同
        user_msg = None
        for msg in converted_back:
            if msg["role"] == "user" and any(
                part.get("text") == "Hello" for part in msg["content"]
            ):
                user_msg = msg
                break
        assert user_msg is not None

        # 验证助手消息存在
        assistant_msgs = [msg for msg in converted_back if msg["role"] == "assistant"]
        assert len(assistant_msgs) >= 1

    def test_error_handling(self):
        """测试错误处理"""
        # 测试空输入列表
        result, warnings = self.converter.to_provider([])
        assert result["input"] == []

        # 测试无效消息格式
        with pytest.raises(ValueError):
            self.converter.to_provider([{"invalid": "message"}])

    def test_to_provider_with_file_and_developer_role(self):
        """测试 to_provider 对文件和 developer 角色的处理"""
        messages = [
            {
                "role": "developer",
                "content": [
                    {"type": "text", "text": "Analyze this file."},
                    {
                        "type": "file",
                        "file_name": "test.py",
                        "file_data": {"data": "cHJpbnQoImhlbGxvIik="},
                    },
                ],
            }
        ]
        result, _ = self.converter.to_provider(messages)
        assert len(result["input"]) == 1
        item = result["input"][0]
        assert item["role"] == "developer"
        assert len(item["content"]) == 2
        assert item["content"][0]["type"] == "input_text"
        file_part = item["content"][1]
        assert file_part["type"] == "input_file"
        assert file_part["filename"] == "test.py"
        assert file_part["file_data"] == "cHJpbnQoImhlbGxvIik="

    def test_to_provider_with_extensions(self):
        """测试 to_provider 对扩展项的处理"""
        messages = [
            {"type": "system_event", "event_type": "start", "timestamp": "123"},
            {
                "type": "tool_chain_node",
                "tool_call": {
                    "type": "tool_call",
                    "id": "tc_1",
                    "name": "tool1",
                    "arguments": {},
                },
            },
            {"type": "batch_marker", "marker_type": "end"},
        ]
        result, warnings = self.converter.to_provider(messages)
        assert len(result["input"]) == 2
        assert result["input"][0]["type"] == "system_event"
        assert result["input"][0]["event_type"] == "start"
        assert result["input"][1]["type"] == "function_call"
        assert result["input"][1]["call_id"] == "tc_1"
        assert len(warnings) == 2
        assert "Tool chain converted" in warnings[0]
        assert "Extension item ignored: batch_marker" in warnings[1]

    def test_from_provider_with_various_parts(self):
        """测试 from_provider 对各种内容部分的处理"""
        provider_data = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_image", "image_url": "http://a.com/b.jpg"},
                        {
                            "type": "input_file",
                            "file_id": "file_123",
                            "filename": "test.txt",
                        },
                    ],
                },
                {"type": "reasoning", "content": None},  # 测试空的推理内容
            ]
        }
        result = self.converter.from_provider(provider_data)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "image"
        assert msg["content"][0]["image_url"] == "http://a.com/b.jpg"
        assert msg["content"][1]["type"] == "file"
        assert msg["content"][1]["file_id"] == "file_123"

    def test_from_provider_error_handling(self):
        """测试 from_provider 的错误处理"""
        with pytest.raises(ValueError, match="must be a dictionary"):
            self.converter.from_provider("not a dict")

        with pytest.raises(ValueError, match="must be a list"):
            self.converter.from_provider({"input": "not a list"})

    def test_to_provider_with_tool_result_and_reasoning(self):
        """测试 to_provider 对工具结果和推理的处理"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call_abc",
                        "result": "some result",
                    },
                    {"type": "reasoning", "reasoning": "some reasoning"},
                ],
            }
        ]
        result, _ = self.converter.to_provider(messages)
        assert len(result["input"]) == 2
        assert result["input"][0]["type"] == "function_call_output"
        assert result["input"][0]["call_id"] == "call_abc"
        assert result["input"][1]["type"] == "reasoning"
        assert result["input"][1]["content"] == "some reasoning"

    def test_from_provider_with_string_arguments(self):
        """测试 from_provider 对字符串格式的 arguments 的处理"""
        provider_data = {
            "input": [
                {
                    "type": "function_call",
                    "name": "test_func",
                    "arguments": '{"key": "value"}',
                }
            ]
        }
        result = self.converter.from_provider(provider_data)
        tool_call = result[0]["content"][0]
        assert tool_call["tool_input"] == {"key": "value"}

    def test_from_provider_with_other_tool_calls(self):
        """测试 from_provider 对 shell_call 等其他工具调用的处理"""
        provider_data = {
            "output": [
                {"type": "shell_call", "id": "sh_1", "arguments": "ls"},
                {"type": "computer_call", "id": "pc_1", "name": "click"},
            ]
        }
        result = self.converter.from_provider(provider_data)
        assert len(result) == 1
        assert len(result[0]["content"]) == 2
        shell_call = result[0]["content"][0]
        assert shell_call["type"] == "tool_call"
        assert shell_call["tool_type"] == "code_interpreter"
        assert shell_call["tool_call_id"] == "sh_1"
        computer_call = result[0]["content"][1]
        assert computer_call["type"] == "tool_call"
        assert computer_call["tool_type"] == "computer_use"
        assert computer_call["tool_name"] == "click"

    def test_from_provider_with_system_event(self):
        """测试 from_provider 对 system_event 的处理"""
        provider_data = {"output": [{"type": "system_event", "event_type": "reboot"}]}
        result = self.converter.from_provider(provider_data)
        assert len(result) == 1
        assert result[0]["type"] == "system_event"
        assert result[0]["event_type"] == "reboot"

    def test_image_and_file_conversion_errors(self):
        """测试图像和文件转换中的错误处理"""
        with pytest.raises(ValueError, match="must have either"):
            self.converter._convert_image_to_responses({"type": "image"})
        with pytest.raises(ValueError, match="must have either"):
            self.converter._convert_file_to_responses({"type": "file"})

    def test_from_provider_with_base64_image_and_file_id(self):
        """测试 from_provider 对 base64 图像和 file_id 的处理"""
        provider_data = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,iVBORw0KGgo=",
                        },
                        {"type": "input_image", "file_id": "file_xyz"},
                    ],
                }
            ]
        }
        result = self.converter.from_provider(provider_data)
        msg_content = result[0]["content"]
        assert msg_content[0]["type"] == "image"
        assert msg_content[0]["image_data"]["media_type"] == "image/png"
        assert msg_content[0]["image_data"]["data"] == "iVBORw0KGgo="
        assert msg_content[1]["type"] == "image"
        assert msg_content[1]["file_id"] == "file_xyz"

    def test_to_provider_with_user_tool_call_and_assistant_reasoning(self):
        """测试 to_provider 对用户工具调用和助手推理的处理"""
        messages = [
            {
                "role": "user",
                "content": [
                    ToolCallPart(
                        type="tool_call",
                        tool_call_id="user_call",
                        tool_name="user_tool",
                        tool_input={},
                    )
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "reasoning", "reasoning": "Assistant reasoning"}],
            },
        ]
        result, _ = self.converter.to_provider(messages)
        assert len(result["input"]) == 2
        assert result["input"][0]["type"] == "function_call"
        assert result["input"][0]["call_id"] == "user_call"
        assert result["input"][1]["type"] == "reasoning"
        assert result["input"][1]["content"] == "Assistant reasoning"

    def test_from_provider_with_pydantic_and_string_content(self):
        """测试 from_provider 对 Pydantic 模型和字符串内容的处理"""

        class MockPydantic:
            def model_dump(self):
                return {
                    "input": [
                        {"type": "message", "role": "user", "content": "Just a string"}
                    ]
                }

        result = self.converter.from_provider(MockPydantic())
        assert len(result) == 1
        assert result[0]["content"][0]["text"] == "Just a string"

    def test_from_provider_with_appended_tool_calls(self):
        """测试 from_provider 对附加工具调用的处理"""
        provider_data = {
            "output": [
                {"type": "message", "role": "assistant", "content": []},
                {"type": "function_call", "id": "func_1", "name": "tool1"},
                {"type": "mcp_call", "id": "mcp_1", "name": "tool2"},
            ]
        }
        result = self.converter.from_provider(provider_data)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["type"] == "tool_call"
        assert result[0]["content"][1]["type"] == "tool_call"

    def test_from_provider_with_system_event_after_message(self):
        """测试 from_provider 在消息后处理 system_event"""
        provider_data = {
            "output": [
                {"type": "message", "role": "user", "content": "Hi"},
                {"type": "system_event", "event_type": "disconnect"},
            ]
        }
        result = self.converter.from_provider(provider_data)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[1]["type"] == "system_event"

    def test_from_provider_with_empty_input(self):
        """测试 from_provider 处理空输入"""
        result = self.converter.from_provider({"input": []})
        assert result == []

    def test_file_conversion_edge_cases(self):
        """测试文件转换的边缘情况"""
        # 无 filename 的 to_provider
        file_part = {"type": "file", "file_url": "http://a.com/b.txt"}
        result = self.converter._convert_file_to_responses(file_part)
        assert result["filename"] == "unknown"

        # 带 file_data 的 from_provider
        provider_part = {"type": "input_file", "file_data": "base64data"}
        ir_part = self.converter._convert_file_from_responses(provider_part)
        assert ir_part["file_data"]["data"] == "base64data"

    def test_image_from_provider_bad_data_url(self):
        """测试 from_provider 处理损坏的 data URL"""
        provider_part = {"type": "input_image", "image_url": "data:bad_url"}
        result = self.converter._convert_image_from_responses(provider_part)
        assert result["image_url"] == "data:bad_url"
        assert "image_data" not in result
