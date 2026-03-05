"""
Tests for auto-detection and conversion functionality
"""

import pytest

from llmir.auto_detect import convert, detect_provider, get_converter_for_provider


class TestDetectProvider:
    """测试 provider 自动检测功能"""

    def test_detect_openai_chat_simple(self):
        """测试检测简单的 OpenAI Chat 格式"""
        body = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }
        assert detect_provider(body) == "openai_chat"

    def test_detect_openai_chat_with_multimodal(self):
        """测试检测带多模态内容的 OpenAI Chat 格式"""
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {
                            "type": "image_url",
                            "image_url": {"url": "https://example.com/image.jpg"},
                        },
                    ],
                }
            ]
        }
        assert detect_provider(body) == "openai_chat"

    def test_detect_openai_chat_with_tools(self):
        """测试检测带工具调用的 OpenAI Chat 格式"""
        body = {
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_123",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": '{"location": "SF"}',
                            },
                        }
                    ],
                },
            ]
        }
        assert detect_provider(body) == "openai_chat"

    def test_detect_openai_responses_with_input(self):
        """测试检测 OpenAI Responses API 格式（input 字段）"""
        body = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                }
            ]
        }
        assert detect_provider(body) == "openai_responses"

    def test_detect_openai_responses_with_output(self):
        """测试检测 OpenAI Responses API 格式（output 字段）"""
        body = {
            "output": [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hi!"}],
                }
            ]
        }
        assert detect_provider(body) == "openai_responses"

    def test_detect_openai_responses_with_function_call(self):
        """测试检测带函数调用的 Responses API 格式"""
        body = {
            "input": [
                {"type": "function_call", "call_id": "123", "name": "get_weather"}
            ]
        }
        assert detect_provider(body) == "openai_responses"

    def test_detect_openai_responses_with_reasoning(self):
        """测试检测带推理的 Responses API 格式"""
        body = {"output": [{"type": "reasoning", "reasoning": "Let me think..."}]}
        assert detect_provider(body) == "openai_responses"

    def test_detect_anthropic_simple(self):
        """测试检测简单的 Anthropic 格式"""
        # Anthropic 格式通常包含 max_tokens 或其他特征
        body = {
            "model": "claude-3-sonnet-20240229",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            ],
        }
        # 注意：仅有 messages 和 text 类型的内容无法明确区分，会默认为 openai_chat
        # 需要添加 Anthropic 特有字段
        assert detect_provider(body) == "openai_chat"  # 默认行为

        # 添加 system 字段后可以识别为 Anthropic
        body_with_system = {
            "system": "You are helpful",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            ],
        }
        assert detect_provider(body_with_system) == "anthropic"

    def test_detect_anthropic_with_system(self):
        """测试检测带 system 的 Anthropic 格式"""
        body = {
            "system": "You are a helpful assistant",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            ],
        }
        assert detect_provider(body) == "anthropic"

    def test_detect_anthropic_with_tool_use(self):
        """测试检测带工具使用的 Anthropic 格式"""
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "123",
                            "name": "get_weather",
                            "input": {},
                        }
                    ],
                }
            ]
        }
        assert detect_provider(body) == "anthropic"

    def test_detect_anthropic_with_image(self):
        """测试检测带图像的 Anthropic 格式"""
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's this?"},
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://example.com/image.jpg",
                            },
                        },
                    ],
                }
            ]
        }
        assert detect_provider(body) == "anthropic"

    def test_detect_google_simple(self):
        """测试检测简单的 Google GenAI 格式"""
        body = {"contents": [{"role": "user", "parts": [{"text": "Hello"}]}]}
        assert detect_provider(body) == "google"

    def test_detect_google_with_system_instruction(self):
        """测试检测带 system_instruction 的 Google 格式"""
        body = {
            "system_instruction": {"parts": [{"text": "You are helpful"}]},
            "contents": [{"role": "user", "parts": [{"text": "Hello"}]}],
        }
        assert detect_provider(body) == "google"

    def test_detect_google_with_function_call(self):
        """测试检测带函数调用的 Google 格式"""
        body = {
            "contents": [
                {
                    "role": "model",
                    "parts": [
                        {
                            "function_call": {
                                "name": "get_weather",
                                "args": {"location": "SF"},
                            }
                        }
                    ],
                }
            ]
        }
        assert detect_provider(body) == "google"

    def test_detect_google_with_inline_data(self):
        """测试检测带 inline_data 的 Google 格式"""
        body = {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {
                            "inline_data": {
                                "mime_type": "image/jpeg",
                                "data": "base64data",
                            }
                        }
                    ],
                }
            ]
        }
        assert detect_provider(body) == "google"

    def test_detect_invalid_input(self):
        """测试无效输入"""
        assert detect_provider({}) is None
        assert detect_provider({"unknown": "field"}) is None
        assert detect_provider("not a dict") is None
        assert detect_provider(None) is None

    def test_detect_empty_messages(self):
        """测试空消息列表"""
        body = {"messages": []}
        # 空消息列表应该返回 openai_chat（默认）
        assert detect_provider(body) == "openai_chat"


class TestGetConverterForProvider:
    """测试获取转换器功能"""

    def test_get_openai_chat_converter(self):
        """测试获取 OpenAI Chat 转换器"""
        from llmir.converters import OpenAIChatConverter

        converter = get_converter_for_provider("openai_chat")
        assert isinstance(converter, OpenAIChatConverter)

    def test_get_openai_responses_converter(self):
        """测试获取 OpenAI Responses 转换器"""
        from llmir.converters import OpenAIResponsesConverter

        converter = get_converter_for_provider("openai_responses")
        assert isinstance(converter, OpenAIResponsesConverter)

    def test_get_anthropic_converter(self):
        """测试获取 Anthropic 转换器"""
        from llmir.converters import AnthropicConverter

        converter = get_converter_for_provider("anthropic")
        assert isinstance(converter, AnthropicConverter)

    def test_get_google_converter(self):
        """测试获取 Google 转换器"""
        from llmir.converters import GoogleConverter

        converter = get_converter_for_provider("google")
        assert isinstance(converter, GoogleConverter)

    def test_get_invalid_provider(self):
        """测试无效的 provider"""
        with pytest.raises(ValueError, match="Unsupported provider"):
            get_converter_for_provider("invalid_provider")


class TestConvert:
    """测试自动转换功能"""

    def test_convert_openai_to_google(self):
        """测试从 OpenAI Chat 转换到 Google"""
        openai_body = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }

        google_body = convert(openai_body, "google")

        assert "contents" in google_body
        assert len(google_body["contents"]) == 2
        assert google_body["contents"][0]["role"] == "user"
        assert google_body["contents"][1]["role"] == "model"

    def test_convert_anthropic_to_openai(self):
        """测试从 Anthropic 转换到 OpenAI Chat"""
        # 使用明确的 Anthropic 格式（带 system 字段）
        anthropic_body = {
            "system": "You are helpful",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
            ],
        }

        openai_body = convert(anthropic_body, "openai_chat")

        assert "messages" in openai_body
        assert len(openai_body["messages"]) >= 2
        # 验证消息存在（可能包含 system 消息）
        assert any(msg.get("role") == "user" for msg in openai_body["messages"])
        assert any(msg.get("role") == "assistant" for msg in openai_body["messages"])

    def test_convert_google_to_anthropic(self):
        """测试从 Google 转换到 Anthropic"""
        # Google 请求格式（包含 contents）
        google_body = {
            "model": "gemini-2.0-flash",
            "contents": [{"role": "user", "parts": [{"text": "Hi!"}]}],
            "config": {},
        }

        anthropic_body = convert(google_body, "anthropic", source_provider="google")

        assert "messages" in anthropic_body
        assert len(anthropic_body["messages"]) == 1
        assert anthropic_body["messages"][0]["role"] == "user"
        assert anthropic_body["messages"][0]["content"][0]["text"] == "Hi!"

    def test_convert_with_explicit_source(self):
        """测试显式指定源 provider"""
        body = {"messages": [{"role": "user", "content": "Hello"}]}

        # 显式指定为 OpenAI
        result = convert(body, "google", source_provider="openai_chat")
        assert "contents" in result

    def test_convert_same_provider(self):
        """测试源和目标相同时直接返回"""
        openai_body = {"messages": [{"role": "user", "content": "Hello"}]}

        result = convert(openai_body, "openai_chat")
        assert result == openai_body

    def test_convert_with_tools(self):
        """测试带工具定义的转换"""
        openai_body = {
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Get weather",
                        "parameters": {
                            "type": "object",
                            "properties": {"location": {"type": "string"}},
                        },
                    },
                }
            ],
        }

        google_body = convert(openai_body, "google")

        # Google puts tools inside config
        assert "config" in google_body
        assert "tools" in google_body["config"]
        assert len(google_body["config"]["tools"]) == 1

    def test_convert_undetectable_source(self):
        """测试无法检测源格式时抛出错误"""
        invalid_body = {"unknown": "format"}

        with pytest.raises(ValueError, match="Unable to detect source provider"):
            convert(invalid_body, "google")

    def test_convert_openai_responses_to_chat(self):
        """测试从 OpenAI Responses 转换到 Chat"""
        responses_body = {
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                }
            ]
        }

        chat_body = convert(responses_body, "openai_chat")

        assert "messages" in chat_body
        assert chat_body["messages"][0]["content"] == "Hello"

    def test_convert_with_multimodal_content(self):
        """测试多模态内容的转换"""
        openai_body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's this?"},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "data:image/jpeg;base64,/9j/4AAQ",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ]
        }

        anthropic_body = convert(openai_body, "anthropic")

        assert "messages" in anthropic_body
        assert len(anthropic_body["messages"][0]["content"]) == 2
        assert anthropic_body["messages"][0]["content"][0]["type"] == "text"
        assert anthropic_body["messages"][0]["content"][1]["type"] == "image"

    def test_convert_roundtrip(self):
        """测试往返转换的一致性"""
        original = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there!"},
            ]
        }

        # OpenAI -> Anthropic -> OpenAI (更简单的往返路径)
        anthropic_body = convert(original, "anthropic")
        back_to_openai = convert(
            anthropic_body, "openai_chat", source_provider="anthropic"
        )

        assert "messages" in back_to_openai
        assert len(back_to_openai["messages"]) >= 2
        # 验证消息存在
        assert any(msg.get("role") == "user" for msg in back_to_openai["messages"])
        assert any(msg.get("role") == "assistant" for msg in back_to_openai["messages"])


class TestEdgeCases:
    """测试边缘情况"""

    def test_detect_ambiguous_messages_format(self):
        """测试模糊的消息格式（可能是 OpenAI 或 Anthropic）"""
        # 这种格式可能被识别为任一种，但应该有一致的行为
        body = {"messages": [{"role": "user", "content": []}]}
        result = detect_provider(body)
        # 应该返回某个有效的 provider，不应该是 None
        assert result in ["openai_chat", "anthropic"]

    def test_convert_empty_messages(self):
        """测试空消息列表的转换"""
        body = {"messages": []}
        result = convert(body, "google")
        assert "contents" in result
        assert result["contents"] == []

    def test_detect_with_nested_content(self):
        """测试嵌套内容的检测"""
        # 仅有 text 类型无法明确区分，默认为 openai_chat
        body = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                        {"type": "text", "text": "World"},
                    ],
                }
            ]
        }
        assert detect_provider(body) == "openai_chat"  # 默认行为

        # 添加 Anthropic 特有内容类型可以识别
        body_with_image = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's this?"},
                        {
                            "type": "image",
                            "source": {
                                "type": "url",
                                "url": "https://example.com/img.jpg",
                            },
                        },
                    ],
                }
            ]
        }
        assert detect_provider(body_with_image) == "anthropic"
