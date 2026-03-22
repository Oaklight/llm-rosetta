"""
Anthropic MessageOps unit tests.
"""

from typing import Any, Union, cast

from llm_rosetta.converters.anthropic.content_ops import AnthropicContentOps
from llm_rosetta.converters.anthropic.message_ops import AnthropicMessageOps
from llm_rosetta.converters.anthropic.tool_ops import AnthropicToolOps
from llm_rosetta.types.ir import Message
from llm_rosetta.types.ir.extensions import ExtensionItem


class TestAnthropicMessageOps:
    """Unit tests for AnthropicMessageOps."""

    def setup_method(self):
        """Set up test fixtures."""
        content_ops = AnthropicContentOps()
        tool_ops = AnthropicToolOps()
        self.message_ops = AnthropicMessageOps(content_ops, tool_ops)

    # ==================== IR → Provider ====================

    def test_ir_user_message_to_p(self):
        """Test IR user message → Anthropic user message."""
        ir_messages = cast(
            list[Message],
            [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        )
        result, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Hello"
        assert len(warnings) == 0

    def test_ir_assistant_message_to_p(self):
        """Test IR assistant message → Anthropic assistant message."""
        ir_messages = cast(
            list[Message],
            [{"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]}],
        )
        result, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"][0]["text"] == "Hi!"

    def test_ir_system_message_skipped(self):
        """Test IR system message is skipped (handled at converter level)."""
        ir_messages = cast(
            list[Message],
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are helpful."}],
                },
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_ir_assistant_with_tool_call(self):
        """Test IR assistant message with tool call → Anthropic."""
        ir_messages = cast(
            list[Message],
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me search"},
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_123",
                            "tool_name": "search",
                            "tool_input": {"q": "test"},
                            "tool_type": "function",
                        },
                    ],
                }
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "text"
        assert msg["content"][1]["type"] == "tool_use"
        assert msg["content"][1]["id"] == "call_123"

    def test_ir_user_with_tool_result(self):
        """Test IR user message with tool result → Anthropic."""
        ir_messages = cast(
            list[Message],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": "call_123",
                            "result": "Search results...",
                        }
                    ],
                }
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "user"
        assert msg["content"][0]["type"] == "tool_result"
        assert msg["content"][0]["tool_use_id"] == "call_123"

    def test_ir_tool_message_to_p(self):
        """Test IR tool message → Anthropic user message with tool_result."""
        ir_messages = cast(
            list[Message],
            [
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": "call_456",
                            "result": "Tool output",
                        }
                    ],
                }
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "user"
        assert msg["content"][0]["type"] == "tool_result"

    def test_ir_user_with_image(self):
        """Test IR user message with image → Anthropic."""
        ir_messages = cast(
            list[Message],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image_url": "https://example.com/img.jpg",
                        }
                    ],
                }
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(result) == 1
        msg = result[0]
        assert msg["content"][0]["type"] == "image"
        assert msg["content"][0]["source"]["url"] == "https://example.com/img.jpg"

    def test_ir_user_with_file(self):
        """Test IR user message with file → Anthropic."""
        ir_messages = cast(
            list[Message],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "file",
                            "file_data": {
                                "data": "pdf_data",
                                "media_type": "application/pdf",
                            },
                        }
                    ],
                }
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(result) == 1
        msg = result[0]
        assert msg["content"][0]["type"] == "document"

    def test_ir_assistant_with_reasoning(self):
        """Test IR assistant message with reasoning → Anthropic."""
        ir_messages = cast(
            list[Message],
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "reasoning", "reasoning": "I need to think."},
                        {"type": "text", "text": "Here is my answer."},
                    ],
                }
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(result) == 1
        msg = result[0]
        assert msg["content"][0]["type"] == "thinking"
        assert msg["content"][0]["thinking"] == "I need to think."
        assert msg["content"][1]["type"] == "text"

    def test_extension_item_system_event(self):
        """Test extension item system_event produces warning."""
        ir_messages = cast(
            list[Union[Message, ExtensionItem]],
            [
                {
                    "type": "system_event",
                    "event_type": "session_start",
                    "timestamp": "2024-01-01T00:00:00Z",
                },
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(result) == 0
        assert any("System event ignored" in w for w in warnings)

    def test_extension_item_tool_chain_node(self):
        """Test extension item tool_chain_node → assistant message."""
        ir_messages = cast(
            list[Union[Message, ExtensionItem]],
            [
                {
                    "type": "tool_chain_node",
                    "node_id": "node_1",
                    "tool_call": {
                        "type": "tool_call",
                        "tool_call_id": "call_chain",
                        "tool_name": "tool_a",
                        "tool_input": {"x": 1},
                        "tool_type": "function",
                    },
                },
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"][0]["type"] == "tool_use"
        assert any("Tool chain" in w for w in warnings)

    def test_extension_item_batch_marker(self):
        """Test extension item batch_marker produces warning."""
        ir_messages = cast(
            list[Union[Message, ExtensionItem]],
            [
                {"type": "batch_marker", "batch_id": "b1", "batch_type": "start"},
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(result) == 0
        assert any("batch_marker" in w for w in warnings)

    # ==================== Provider → IR ====================

    def test_p_user_message_to_ir(self):
        """Test Anthropic user message → IR."""
        provider_messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
        ]
        result = cast(list[Any], self.message_ops.p_messages_to_ir(provider_messages))
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Hello"

    def test_p_assistant_message_to_ir(self):
        """Test Anthropic assistant message → IR."""
        provider_messages = [
            {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]}
        ]
        result = cast(list[Any], self.message_ops.p_messages_to_ir(provider_messages))
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"][0]["text"] == "Hi!"

    def test_p_string_content_to_ir(self):
        """Test Anthropic message with string content → IR."""
        provider_messages = [{"role": "user", "content": "Hello string"}]
        result = cast(list[Any], self.message_ops.p_messages_to_ir(provider_messages))
        assert len(result) == 1
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Hello string"

    def test_p_tool_use_to_ir(self):
        """Test Anthropic tool_use block → IR ToolCallPart."""
        provider_messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_123",
                        "name": "get_weather",
                        "input": {"city": "SF"},
                    }
                ],
            }
        ]
        result = cast(list[Any], self.message_ops.p_messages_to_ir(provider_messages))
        tc = result[0]["content"][0]
        assert tc["type"] == "tool_call"
        assert tc["tool_call_id"] == "toolu_123"
        assert tc["tool_name"] == "get_weather"

    def test_p_server_tool_use_to_ir(self):
        """Test Anthropic server_tool_use → IR ToolCallPart."""
        provider_messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "server_456",
                        "name": "web_search",
                        "input": {"query": "test"},
                    }
                ],
            }
        ]
        result = cast(list[Any], self.message_ops.p_messages_to_ir(provider_messages))
        tc = result[0]["content"][0]
        assert tc["type"] == "tool_call"
        assert tc["tool_type"] == "web_search"

    def test_p_tool_result_to_ir(self):
        """Test Anthropic tool_result block → IR ToolResultPart.

        Anthropic places tool_result in user messages, but IR normalizes
        the role to "tool".
        """
        provider_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_789",
                        "content": "Result data",
                        "is_error": True,
                    }
                ],
            }
        ]
        result = cast(list[Any], self.message_ops.p_messages_to_ir(provider_messages))
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        tr = result[0]["content"][0]
        assert tr["type"] == "tool_result"
        assert tr["tool_call_id"] == "tool_789"
        assert tr["is_error"] is True

    def test_p_mixed_tool_result_and_text_to_ir(self):
        """Test Anthropic user message with tool_result + text splits into two IR messages."""
        provider_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_abc",
                        "content": "Result data",
                    },
                    {"type": "text", "text": "Now do something else"},
                ],
            }
        ]
        result = cast(list[Any], self.message_ops.p_messages_to_ir(provider_messages))
        assert len(result) == 2
        # First: tool message with tool_result
        assert result[0]["role"] == "tool"
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "tool_result"
        assert result[0]["content"][0]["tool_call_id"] == "tool_abc"
        # Second: user message with text
        assert result[1]["role"] == "user"
        assert len(result[1]["content"]) == 1
        assert result[1]["content"][0]["type"] == "text"
        assert result[1]["content"][0]["text"] == "Now do something else"

    def test_p_thinking_to_ir(self):
        """Test Anthropic thinking block → IR ReasoningPart."""
        provider_messages = [
            {
                "role": "assistant",
                "content": [
                    {"type": "thinking", "thinking": "Let me think..."},
                    {"type": "text", "text": "Here is my answer."},
                ],
            }
        ]
        result = cast(list[Any], self.message_ops.p_messages_to_ir(provider_messages))
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["type"] == "reasoning"
        assert result[0]["content"][0]["reasoning"] == "Let me think..."
        assert result[0]["content"][1]["type"] == "text"

    def test_p_image_to_ir(self):
        """Test Anthropic image block → IR ImagePart."""
        provider_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "base64data",
                        },
                    }
                ],
            }
        ]
        result = cast(list[Any], self.message_ops.p_messages_to_ir(provider_messages))
        img = result[0]["content"][0]
        assert img["type"] == "image"
        assert img["image_data"]["media_type"] == "image/png"

    def test_p_document_to_ir(self):
        """Test Anthropic document block → IR FilePart."""
        provider_messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "document",
                        "source": {
                            "type": "base64",
                            "media_type": "application/pdf",
                            "data": "pdf_data",
                        },
                    }
                ],
            }
        ]
        result = cast(list[Any], self.message_ops.p_messages_to_ir(provider_messages))
        file_part = result[0]["content"][0]
        assert file_part["type"] == "file"
        assert file_part["file_data"]["media_type"] == "application/pdf"

    def test_round_trip_messages(self):
        """Test message round-trip: IR → Provider → IR."""
        original = cast(
            list[Message],
            [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
            ],
        )
        provider, _ = self.message_ops.ir_messages_to_p(original)
        restored = cast(list[Any], self.message_ops.p_messages_to_ir(provider))
        assert len(restored) == 2
        assert restored[0]["role"] == "user"
        assert restored[0]["content"][0]["text"] == "Hello"
        assert restored[1]["role"] == "assistant"
        assert restored[1]["content"][0]["text"] == "Hi!"
