"""
Google GenAI转换器测试
"""

import pytest

from src.llm_provider_converter.converters.google_converter import GoogleConverter
from src.llm_provider_converter.types.ir import IRInput, ToolChoice, ToolDefinition


class TestGoogleConverter:
    """Google转换器测试类"""

    def setup_method(self):
        """设置测试"""
        self.converter = GoogleConverter()

    def test_simple_text_message_ir_to_provider(self):
        """测试简单文本消息IR到Provider转换"""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello, world!"}]}
        ]

        result, warnings = self.converter.to_provider(messages)

        assert "contents" in result
        assert len(result["contents"]) == 1
        assert result["contents"][0]["role"] == "user"
        assert result["contents"][0]["parts"] == [{"text": "Hello, world!"}]

    def test_simple_text_message_provider_to_ir(self):
        """测试简单文本消息Provider到IR转换"""
        provider_data = {
            "contents": [{"role": "user", "parts": [{"text": "Hello, world!"}]}]
        }

        result = self.converter.from_provider(provider_data)

        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Hello, world!"

    def test_role_mapping(self):
        """测试角色映射"""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
        ]

        result, warnings = self.converter.to_provider(messages)

        assert len(result["contents"]) == 2
        assert result["contents"][0]["role"] == "user"
        assert result["contents"][1]["role"] == "model"  # assistant -> model

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

        # 系统消息应该转换为system_instruction
        assert "system_instruction" in result
        assert result["system_instruction"]["parts"] == [
            {"text": "You are a helpful assistant."}
        ]

        # contents中只有用户消息
        assert len(result["contents"]) == 1
        assert result["contents"][0]["role"] == "user"

    def test_multiple_system_messages(self):
        """测试多个系统消息合并"""
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are helpful."}],
            },
            {"role": "system", "content": [{"type": "text", "text": "Be concise."}]},
            {"role": "user", "content": [{"type": "text", "text": "Hello!"}]},
        ]

        result, warnings = self.converter.to_provider(messages)

        # 系统消息应该合并
        assert "system_instruction" in result
        assert len(result["system_instruction"]["parts"]) == 2
        assert result["system_instruction"]["parts"][0]["text"] == "You are helpful."
        assert result["system_instruction"]["parts"][1]["text"] == "Be concise."

    def test_multimodal_message_conversion(self):
        """测试多模态消息转换"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image", "data": "base64data", "media_type": "image/jpeg"},
                ],
            }
        ]

        result, warnings = self.converter.to_provider(messages)

        assert len(result["contents"]) == 1
        content = result["contents"][0]
        assert content["role"] == "user"
        assert len(content["parts"]) == 2

        # 检查文本部分
        text_part = content["parts"][0]
        assert text_part["text"] == "What's in this image?"

        # 检查图片部分
        image_part = content["parts"][1]
        assert "inline_data" in image_part
        assert image_part["inline_data"]["mime_type"] == "image/jpeg"
        assert image_part["inline_data"]["data"] == "base64data"

    def test_audio_content_conversion(self):
        """测试音频内容转换"""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "audio", "data": "audiodata", "media_type": "audio/wav"}
                ],
            }
        ]

        result, warnings = self.converter.to_provider(messages)

        content = result["contents"][0]
        audio_part = content["parts"][0]
        assert "inline_data" in audio_part
        assert audio_part["inline_data"]["mime_type"] == "audio/wav"
        assert audio_part["inline_data"]["data"] == "audiodata"

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

        assert len(result["contents"]) == 3

        # 检查用户消息
        user_content = result["contents"][0]
        assert user_content["role"] == "user"
        assert user_content["parts"][0]["text"] == "What's the weather?"

        # 检查助手工具调用
        assistant_content = result["contents"][1]
        assert assistant_content["role"] == "model"
        function_call_part = assistant_content["parts"][0]
        assert "function_call" in function_call_part
        assert function_call_part["function_call"]["name"] == "get_weather"
        assert function_call_part["function_call"]["args"] == {"location": "New York"}

        # 检查工具结果
        tool_result_content = result["contents"][2]
        assert tool_result_content["role"] == "user"
        function_response_part = tool_result_content["parts"][0]
        assert "function_response" in function_response_part
        assert function_response_part["function_response"]["name"] == "call_123"
        assert (
            function_response_part["function_response"]["response"]["output"]
            == "Sunny, 25°C"
        )

    def test_tool_result_error_conversion(self):
        """测试工具结果错误转换"""
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call_123",
                        "content": "API Error",
                        "is_error": True,
                    }
                ],
            }
        ]

        result, warnings = self.converter.to_provider(messages)

        content = result["contents"][0]
        function_response_part = content["parts"][0]
        assert "function_response" in function_response_part
        assert (
            function_response_part["function_response"]["response"]["error"]
            == "API Error"
        )

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
        assert "function_declarations" in tool
        assert len(tool["function_declarations"]) == 1

        func_decl = tool["function_declarations"][0]
        assert func_decl["name"] == "get_weather"
        assert func_decl["description"] == "Get current weather"
        assert "parameters" in func_decl

    def test_tool_choice_conversion(self):
        """测试工具选择配置转换"""
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

        # 测试auto
        result, warnings = self.converter.to_provider(
            messages, tool_choice={"type": "auto"}
        )
        assert result["tool_config"]["function_calling_config"]["mode"] == "AUTO"

        # 测试none
        result, warnings = self.converter.to_provider(
            messages, tool_choice={"type": "none"}
        )
        assert result["tool_config"]["function_calling_config"]["mode"] == "NONE"

        # 测试required
        result, warnings = self.converter.to_provider(
            messages, tool_choice={"type": "required"}
        )
        assert result["tool_config"]["function_calling_config"]["mode"] == "ANY"

        # 测试specific function
        result, warnings = self.converter.to_provider(
            messages,
            tool_choice={"type": "function", "function": {"name": "get_weather"}},
        )
        config = result["tool_config"]["function_calling_config"]
        assert config["mode"] == "ANY"
        assert config["allowed_function_names"] == ["get_weather"]

    def test_provider_to_ir_with_function_calls(self):
        """测试Provider到IR的函数调用转换"""
        provider_data = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "function_call": {
                                "name": "get_weather",
                                "args": {"location": "NYC"},
                            }
                        }
                    ],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "name": "get_weather",
                                "response": {"output": "Sunny, 20°C"},
                            }
                        }
                    ],
                },
            ]
        }

        result = self.converter.from_provider(provider_data)

        assert len(result) == 2

        # 检查助手消息（角色映射：model -> assistant）
        assistant_msg = result[0]
        assert assistant_msg["role"] == "assistant"
        assert len(assistant_msg["content"]) == 1
        assert assistant_msg["content"][0]["type"] == "tool_call"
        assert assistant_msg["content"][0]["name"] == "get_weather"
        assert assistant_msg["content"][0]["arguments"] == {"location": "NYC"}

        # 检查用户消息
        user_msg = result[1]
        assert user_msg["role"] == "user"
        assert len(user_msg["content"]) == 1
        assert user_msg["content"][0]["type"] == "tool_result"
        assert user_msg["content"][0]["content"] == "Sunny, 20°C"
        assert user_msg["content"][0].get("is_error") is not True

    def test_provider_to_ir_with_function_error(self):
        """测试Provider到IR的函数错误转换"""
        provider_data = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "function_response": {
                                "name": "get_weather",
                                "response": {"error": "API Error"},
                            }
                        }
                    ],
                }
            ]
        }

        result = self.converter.from_provider(provider_data)

        assert len(result) == 1
        user_msg = result[0]
        assert user_msg["content"][0]["type"] == "tool_result"
        assert user_msg["content"][0]["content"] == "API Error"
        assert user_msg["content"][0]["is_error"] is True

    def test_provider_to_ir_with_system_instruction(self):
        """测试Provider到IR的系统指令转换"""
        provider_data = {
            "system_instruction": {"parts": [{"text": "You are helpful."}]},
            "contents": [{"role": "user", "parts": [{"text": "Hello!"}]}],
        }

        result = self.converter.from_provider(provider_data)

        assert len(result) == 2

        # 检查系统消息
        system_msg = result[0]
        assert system_msg["role"] == "system"
        assert system_msg["content"][0]["text"] == "You are helpful."

        # 检查用户消息
        user_msg = result[1]
        assert user_msg["role"] == "user"
        assert user_msg["content"][0]["text"] == "Hello!"

    def test_provider_to_ir_with_multimodal(self):
        """测试Provider到IR的多模态转换"""
        provider_data = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": "What's this?"},
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": "base64data",
                            }
                        },
                        {
                            "file_data": {
                                "file_uri": "gs://bucket/audio.wav",
                                "mime_type": "audio/wav",
                            }
                        },
                    ],
                }
            ]
        }

        result = self.converter.from_provider(provider_data)

        assert len(result) == 1
        user_msg = result[0]
        assert len(user_msg["content"]) == 3

        # 检查文本
        assert user_msg["content"][0]["type"] == "text"
        assert user_msg["content"][0]["text"] == "What's this?"

        # 检查内联图片
        assert user_msg["content"][1]["type"] == "image"
        assert user_msg["content"][1]["data"] == "base64data"
        assert user_msg["content"][1]["media_type"] == "image/jpeg"

        # 检查文件音频
        assert user_msg["content"][2]["type"] == "audio"
        assert user_msg["content"][2]["url"] == "gs://bucket/audio.wav"
        assert user_msg["content"][2]["media_type"] == "audio/wav"

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
                    {"type": "image", "data": "imagedata", "media_type": "image/jpeg"},
                ],
            },
            {"role": "assistant", "content": [{"type": "text", "text": "Hi there!"}]},
        ]

        # IR -> Provider -> IR
        provider_data = self.converter.to_provider(original_messages)
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
        assert converted_back[1]["content"][1]["data"] == "imagedata"

        # 验证助手消息
        assert converted_back[2]["role"] == "assistant"
        assert converted_back[2]["content"][0]["text"] == "Hi there!"

    def test_error_handling(self):
        """测试错误处理"""
        # 测试空消息列表
        result, warnings = self.converter.to_provider([])
        assert result["contents"] == []

        # 测试无效消息格式
        with pytest.raises((KeyError, TypeError)):
            self.converter.to_provider([{"invalid": "message"}])

    def test_unsupported_content_warnings(self):
        """测试不支持内容的警告"""
        import warnings

        # 测试图片URL（应该产生警告）
        messages = [
            {
                "role": "user",
                "content": [{"type": "image", "url": "https://example.com/image.jpg"}],
            }
        ]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result, warnings = self.converter.to_provider(messages)

            # 应该有警告
            assert len(w) > 0
            assert "不直接支持图片URL" in str(w[0].message)

        # 内容应该被跳过
        assert len(result["contents"][0]["parts"]) == 0

    def test_mcp_tool_warning(self):
        """测试MCP工具警告"""
        import warnings

        tools = [{"type": "mcp", "mcp": {"server": "test_server", "tool": "test_tool"}}]

        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            result, warnings = self.converter.to_provider(messages, tools=tools)

            # 应该有MCP警告
            assert len(w) > 0
            assert "MCP工具需要通过Google GenAI的MCP适配器处理" in str(w[0].message)
