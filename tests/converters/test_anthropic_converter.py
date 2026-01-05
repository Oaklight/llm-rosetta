"""
测试Anthropic转换器
"""

import pytest

from llmir.converters.anthropic_converter import AnthropicConverter
from llmir.types.ir import IRInput, ToolChoice, ToolDefinition


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

    def test_from_provider_with_tool_use(self):
        """测试从Provider转换工具使用"""
        provider_data = {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "tool_123",
                    "name": "get_weather",
                    "input": {"location": "SF"},
                }
            ],
        }
        ir_output = self.converter.from_provider(provider_data)
        assert len(ir_output) == 1
        msg = ir_output[0]
        assert msg["role"] == "assistant"
        assert len(msg["content"]) == 1
        tool_call = msg["content"][0]
        assert tool_call["type"] == "tool_call"
        assert tool_call["tool_call_id"] == "tool_123"
        assert tool_call["tool_name"] == "get_weather"
        assert tool_call["tool_input"] == {"location": "SF"}

    def test_from_provider_with_image(self):
        """测试从Provider转换图片"""
        provider_data = {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": "base64_encoded_data",
                    },
                }
            ],
        }
        ir_output = self.converter.from_provider(provider_data)
        assert len(ir_output) == 1
        msg = ir_output[0]
        assert msg["role"] == "user"
        assert len(msg["content"]) == 1
        image_part = msg["content"][0]
        assert image_part["type"] == "image"
        assert image_part["image_data"]["media_type"] == "image/png"
        assert image_part["image_data"]["data"] == "base64_encoded_data"

    def test_from_provider_with_thinking(self):
        """测试从Provider转换思考过程"""
        provider_data = {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "I should use a tool."},
                {"type": "text", "text": "Okay, I will search."},
            ],
        }
        ir_output = self.converter.from_provider(provider_data)
        assert len(ir_output) == 1
        msg = ir_output[0]
        assert msg["role"] == "assistant"
        assert len(msg["content"]) == 2
        reasoning_part = msg["content"][0]
        assert reasoning_part["type"] == "reasoning"
        assert reasoning_part["reasoning"] == "I should use a tool."
        text_part = msg["content"][1]
        assert text_part["type"] == "text"
        assert text_part["text"] == "Okay, I will search."

    def test_to_provider_with_tool_chain_node(self):
        """测试 to_provider 对 tool_chain_node 的处理"""
        ir_input: IRInput = [
            {
                "type": "tool_chain_node",
                "tool_call": {
                    "type": "tool_call",
                    "tool_call_id": "call_456",
                    "tool_name": "another_tool",
                    "tool_input": {"param": "value"},
                    "tool_type": "function",
                },
            }
        ]
        result, warnings = self.converter.to_provider(ir_input)
        assert len(warnings) == 1
        assert "Tool chain converted to sequential calls" in warnings[0]
        assert len(result["messages"]) == 1
        msg = result["messages"][0]
        assert msg["role"] == "assistant"
        assert len(msg["content"]) == 1
        tool_call = msg["content"][0]
        assert tool_call["type"] == "tool_use"
        assert tool_call["id"] == "call_456"

    def test_to_provider_with_file_content(self):
        """测试 to_provider 对 file 类型的处理"""
        ir_input: IRInput = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "file",
                        "file_data": {
                            "media_type": "application/pdf",
                            "data": "pdf_base64_data",
                        },
                    }
                ],
            }
        ]
        result, _ = self.converter.to_provider(ir_input)
        msg = result["messages"][0]
        assert len(msg["content"]) == 1
        doc_part = msg["content"][0]
        assert doc_part["type"] == "document"
        assert doc_part["source"]["type"] == "base64"
        assert doc_part["source"]["media_type"] == "application/pdf"
        assert doc_part["source"]["data"] == "pdf_base64_data"

    def test_from_provider_with_document(self):
        """测试 from_provider 对 document 类型的处理"""
        provider_data = {
            "role": "user",
            "content": [
                {
                    "type": "document",
                    "source": {
                        "type": "base64",
                        "media_type": "application/xml",
                        "data": "xml_base64_data",
                    },
                }
            ],
        }
        ir_output = self.converter.from_provider(provider_data)
        file_part = ir_output[0]["content"][0]
        assert file_part["type"] == "file"
        assert file_part["file_data"]["media_type"] == "application/xml"
        assert file_part["file_data"]["data"] == "xml_base64_data"

    def test_from_provider_with_pydantic_model(self):
        """测试 from_provider 对 Pydantic 模型的处理"""

        class MockPydanticModel:
            def model_dump(self):
                return {
                    "role": "user",
                    "content": "Hello from Pydantic",
                }

        provider_data = MockPydanticModel()
        ir_output = self.converter.from_provider(provider_data)
        assert len(ir_output) == 1
        assert ir_output[0]["role"] == "user"
        assert ir_output[0]["content"][0]["text"] == "Hello from Pydantic"

    def test_from_provider_with_invalid_data(self):
        """测试 from_provider 对无效数据的处理"""
        with pytest.raises(ValueError, match="Anthropic data must be a dictionary"):
            self.converter.from_provider("just a string")

    def test_from_provider_with_single_message(self):
        """测试 from_provider 对单个消息字典的处理"""
        provider_data = {"role": "user", "content": "This is a single message"}
        ir_output = self.converter.from_provider(provider_data)
        assert len(ir_output) == 1
        assert ir_output[0]["role"] == "user"
        assert ir_output[0]["content"][0]["text"] == "This is a single message"

    def test_from_provider_with_server_tool_use(self):
        """测试 from_provider 对 server_tool_use 的处理"""
        provider_data = {
            "role": "assistant",
            "content": [
                {
                    "type": "server_tool_use",
                    "id": "server_tool_456",
                    "name": "web_search",
                    "input": {"query": "llm-bridge"},
                }
            ],
        }
        ir_output = self.converter.from_provider(provider_data)
        tool_call = ir_output[0]["content"][0]
        assert tool_call["type"] == "tool_call"
        assert tool_call["tool_type"] == "web_search"
        assert tool_call["tool_call_id"] == "server_tool_456"
        assert tool_call["tool_name"] == "web_search"

    def test_to_provider_with_ignored_extensions(self):
        """测试 to_provider 忽略 batch_marker 和 session_control"""
        ir_input: IRInput = [
            {"type": "batch_marker", "marker_type": "start"},
            {"type": "session_control", "control_type": "end"},
        ]
        _, warnings = self.converter.to_provider(ir_input)
        assert len(warnings) == 2
        assert "Extension item ignored: batch_marker" in warnings[0]
        assert "Extension item ignored: session_control" in warnings[1]

    def test_from_provider_with_string_system_message(self):
        """测试 from_provider 处理字符串格式的 system 消息"""
        provider_data = {"system": "You are a test assistant."}
        ir_output = self.converter.from_provider(provider_data)
        assert len(ir_output) == 1
        system_msg = ir_output[0]
        assert system_msg["role"] == "system"
        assert system_msg["content"][0]["text"] == "You are a test assistant."

    def test_image_url_conversion_round_trip(self):
        """测试 to_provider 和 from_provider 对 image_url 的双向转换"""
        # to_provider
        ir_input: IRInput = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image_url": "http://example.com/image.png"}
                ],
            }
        ]
        result, _ = self.converter.to_provider(ir_input)
        anthropic_content = result["messages"][0]["content"][0]
        assert anthropic_content["type"] == "image"
        assert anthropic_content["source"]["type"] == "url"
        assert anthropic_content["source"]["url"] == "http://example.com/image.png"

        # from_provider
        provider_data = {"role": "user", "content": [anthropic_content]}
        ir_output = self.converter.from_provider(provider_data)
        ir_part = ir_output[0]["content"][0]
        assert ir_part["type"] == "image"
        assert ir_part["image_url"] == "http://example.com/image.png"

    def test_file_url_conversion_round_trip(self):
        """测试 to_provider 和 from_provider 对 file_url 的双向转换"""
        # to_provider
        ir_input: IRInput = [
            {
                "role": "user",
                "content": [{"type": "file", "file_url": "http://example.com/doc.pdf"}],
            }
        ]
        result, _ = self.converter.to_provider(ir_input)
        anthropic_content = result["messages"][0]["content"][0]
        assert anthropic_content["type"] == "document"
        assert anthropic_content["source"]["type"] == "url"
        assert anthropic_content["source"]["url"] == "http://example.com/doc.pdf"

        # from_provider
        provider_data = {"role": "user", "content": [anthropic_content]}
        ir_output = self.converter.from_provider(provider_data)
        ir_part = ir_output[0]["content"][0]
        assert ir_part["type"] == "file"
        assert ir_part["file_url"] == "http://example.com/doc.pdf"

    def test_from_provider_with_tool_result(self):
        """测试 from_provider 对 tool_result 的处理"""
        provider_data = {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": "tool_789",
                    "content": "Tool execution result",
                    "is_error": True,
                }
            ],
        }
        ir_output = self.converter.from_provider(provider_data)
        tool_result = ir_output[0]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert tool_result["tool_call_id"] == "tool_789"
        assert tool_result["result"] == "Tool execution result"
        assert tool_result["is_error"] is True

    def test_to_provider_with_image_data(self):
        """测试 to_provider 对 image_data 的处理"""
        ir_input: IRInput = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image_data": {
                            "media_type": "image/jpeg",
                            "data": "jpeg_base64_data",
                        },
                    }
                ],
            }
        ]
        result, _ = self.converter.to_provider(ir_input)
        anthropic_content = result["messages"][0]["content"][0]
        assert anthropic_content["type"] == "image"
        assert anthropic_content["source"]["type"] == "base64"
        assert anthropic_content["source"]["media_type"] == "image/jpeg"
        assert anthropic_content["source"]["data"] == "jpeg_base64_data"

    def test_to_provider_with_reasoning(self):
        """测试 to_provider 对 reasoning 类型的处理"""
        ir_input: IRInput = [
            {
                "role": "assistant",
                "content": [{"type": "reasoning", "reasoning": "I need to think."}],
            }
        ]
        result, _ = self.converter.to_provider(ir_input)
        anthropic_content = result["messages"][0]["content"][0]
        assert anthropic_content["type"] == "thinking"
        assert anthropic_content["thinking"] == "I need to think."
