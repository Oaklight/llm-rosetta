"""
OpenAI Chat MessageOps unit tests.
"""

from typing import Any, Union, cast

from llm_rosetta.converters.openai_chat.content_ops import OpenAIChatContentOps
from llm_rosetta.converters.openai_chat.message_ops import OpenAIChatMessageOps
from llm_rosetta.converters.openai_chat.tool_ops import OpenAIChatToolOps
from llm_rosetta.types.ir import Message, ToolCallPart, ToolResultPart
from llm_rosetta.types.ir.extensions import ExtensionItem


class TestOpenAIChatMessageOps:
    """Unit tests for OpenAIChatMessageOps."""

    def setup_method(self):
        """Set up test fixtures."""
        self.content_ops = OpenAIChatContentOps()
        self.tool_ops = OpenAIChatToolOps()
        self.message_ops = OpenAIChatMessageOps(self.content_ops, self.tool_ops)

    # ==================== IR → Provider ====================

    def test_system_message_to_p(self):
        """Test IR system message → OpenAI system message."""
        messages = cast(
            list[Message],
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "You are helpful."}],
                }
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."
        assert warnings == []

    def test_user_text_message_to_p(self):
        """Test IR user text message → OpenAI user message (string content)."""
        messages = cast(
            list[Message],
            [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}],
        )
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Hello!"

    def test_user_multimodal_message_to_p(self):
        """Test IR user multimodal message → OpenAI user message (list content)."""
        messages = cast(
            list[Message],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's this?"},
                        {
                            "type": "image",
                            "image_url": "https://example.com/img.jpg",
                        },
                    ],
                }
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert isinstance(result[0]["content"], list)
        assert len(result[0]["content"]) == 2

    def test_user_message_with_tool_result_split(self):
        """Test user message with ToolResultPart splits into tool role message."""
        messages = cast(
            list[Message],
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Here's the result"},
                        {
                            "type": "tool_result",
                            "tool_call_id": "call_1",
                            "result": "42",
                        },
                    ],
                }
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        assert len(result) == 2
        assert result[0]["role"] == "user"
        assert result[0]["content"] == "Here's the result"
        assert result[1]["role"] == "tool"
        assert result[1]["tool_call_id"] == "call_1"

    def test_user_message_only_tool_result(self):
        """Test user message with only ToolResultPart → only tool message."""
        messages = cast(
            list[Message],
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": "call_1",
                            "result": "done",
                        }
                    ],
                }
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        assert len(result) == 1
        assert result[0]["role"] == "tool"

    def test_assistant_text_message_to_p(self):
        """Test IR assistant text message → OpenAI assistant message."""
        messages = cast(
            list[Message],
            [
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hi there!"}],
                }
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"] == "Hi there!"

    def test_assistant_tool_call_message_to_p(self):
        """Test IR assistant with tool calls → OpenAI assistant with tool_calls."""
        messages = cast(
            list[Message],
            [
                {
                    "role": "assistant",
                    "content": [
                        ToolCallPart(
                            type="tool_call",
                            tool_call_id="call_1",
                            tool_name="get_weather",
                            tool_input={"city": "NYC"},
                        )
                    ],
                }
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "assistant"
        assert msg["content"] is None
        assert len(msg["tool_calls"]) == 1
        assert msg["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_assistant_text_and_tool_calls_to_p(self):
        """Test assistant with both text and tool calls."""
        messages = cast(
            list[Message],
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check"},
                        ToolCallPart(
                            type="tool_call",
                            tool_call_id="c1",
                            tool_name="search",
                            tool_input={},
                        ),
                    ],
                }
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        msg = result[0]
        assert msg["content"] == "Let me check"
        assert len(msg["tool_calls"]) == 1

    def test_assistant_empty_content_to_p(self):
        """Test assistant with empty content."""
        messages = cast(list[Message], [{"role": "assistant", "content": []}])
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        assert result[0]["content"] == ""

    def test_assistant_refusal_to_p(self):
        """Test assistant with refusal part."""
        messages = cast(
            list[Message],
            [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "refusal", "refusal": "I cannot do that"},
                    ],
                }
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        assert result[0]["refusal"] == "I cannot do that"

    def test_tool_message_to_p(self):
        """Test IR tool message → OpenAI tool role message."""
        messages = cast(
            list[Message],
            [
                {
                    "role": "tool",
                    "content": [
                        ToolResultPart(
                            type="tool_result",
                            tool_call_id="call_1",
                            result="Result data",
                        )
                    ],
                }
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        assert len(result) == 1
        assert result[0]["role"] == "tool"
        assert result[0]["tool_call_id"] == "call_1"
        assert result[0]["content"] == "Result data"

    def test_extension_items_handling(self):
        """Test extension items produce warnings."""
        items = cast(
            list[Union[Message, ExtensionItem]],
            [
                {
                    "type": "system_event",
                    "event_type": "session_start",
                    "timestamp": "2024-01-01T00:00:00Z",
                },
                {
                    "type": "batch_marker",
                    "batch_id": "batch_1",
                    "batch_type": "start",
                },
                {
                    "type": "session_control",
                    "control_type": "cancel_tool",
                    "target_id": "call_1",
                },
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(items)
        assert len(warnings) == 3
        assert "System event ignored" in warnings[0]
        assert "Extension item ignored: batch_marker" in warnings[1]
        assert "Extension item ignored: session_control" in warnings[2]

    def test_file_content_warning(self):
        """Test file content in user message produces warning."""
        messages = cast(
            list[Message],
            [
                {
                    "role": "user",
                    "content": [{"type": "file", "file_data": {"data": "x"}}],
                }
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        assert len(warnings) == 1
        assert "File content not supported" in warnings[0]

    def test_reasoning_content_warning(self):
        """Test reasoning content produces warning."""
        messages = cast(
            list[Message],
            [
                {
                    "role": "assistant",
                    "content": [{"type": "reasoning", "reasoning": "thinking"}],
                }
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        assert len(warnings) == 1
        assert "Reasoning content not supported" in warnings[0]

    # ==================== Tool message reordering ====================

    def test_reorder_tool_messages_after_interleaved_user(self):
        """Tool messages are moved next to their assistant tool_calls."""
        messages = cast(
            list[Message],
            [
                {
                    "role": "assistant",
                    "content": [
                        ToolCallPart(
                            type="tool_call",
                            tool_call_id="call_1",
                            tool_name="exec_command",
                            tool_input={"cmd": "ls"},
                        )
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Warning: use apply_patch"}],
                },
                {
                    "role": "tool",
                    "content": [
                        ToolResultPart(
                            type="tool_result",
                            tool_call_id="call_1",
                            result="file.txt",
                        )
                    ],
                },
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(messages)

        assert [m["role"] for m in result] == ["assistant", "tool", "user"]
        assert result[1]["tool_call_id"] == "call_1"
        assert any("Reordered tool messages" in w for w in warnings)

    def test_reorder_preserves_correct_order(self):
        """No reorder warning when tool messages already follow assistant."""
        messages = cast(
            list[Message],
            [
                {
                    "role": "assistant",
                    "content": [
                        ToolCallPart(
                            type="tool_call",
                            tool_call_id="call_1",
                            tool_name="search",
                            tool_input={"q": "test"},
                        )
                    ],
                },
                {
                    "role": "tool",
                    "content": [
                        ToolResultPart(
                            type="tool_result",
                            tool_call_id="call_1",
                            result="found it",
                        )
                    ],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "thanks"}],
                },
            ],
        )
        result, warnings = self.message_ops.ir_messages_to_p(messages)

        assert [m["role"] for m in result] == ["assistant", "tool", "user"]
        assert not any("Reordered" in w for w in warnings)

    # ==================== Provider → IR ====================

    def test_p_system_to_ir(self):
        """Test OpenAI system message → IR SystemMessage."""
        result = cast(
            list[Any],
            self.message_ops.p_messages_to_ir(
                [{"role": "system", "content": "Be helpful"}]
            ),
        )
        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"][0]["text"] == "Be helpful"

    def test_p_user_string_to_ir(self):
        """Test OpenAI user message with string content → IR UserMessage."""
        result = cast(
            list[Any],
            self.message_ops.p_messages_to_ir([{"role": "user", "content": "Hello"}]),
        )
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Hello"

    def test_p_user_multimodal_to_ir(self):
        """Test OpenAI user multimodal message → IR UserMessage."""
        result = cast(
            list[Any],
            self.message_ops.p_messages_to_ir(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Look at this"},
                            {
                                "type": "image_url",
                                "image_url": {"url": "https://example.com/img.jpg"},
                            },
                        ],
                    }
                ]
            ),
        )
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "image"

    def test_p_assistant_with_tool_calls_to_ir(self):
        """Test OpenAI assistant with tool_calls → IR AssistantMessage."""
        result = cast(
            list[Any],
            self.message_ops.p_messages_to_ir(
                [
                    {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": "get_weather",
                                    "arguments": '{"city": "NYC"}',
                                },
                            }
                        ],
                    }
                ]
            ),
        )
        msg = result[0]
        assert msg["role"] == "assistant"
        assert len(msg["content"]) == 1
        assert msg["content"][0]["type"] == "tool_call"
        assert msg["content"][0]["tool_name"] == "get_weather"

    def test_p_assistant_with_refusal_to_ir(self):
        """Test OpenAI assistant with refusal → IR AssistantMessage."""
        result = cast(
            list[Any],
            self.message_ops.p_messages_to_ir(
                [{"role": "assistant", "content": None, "refusal": "Cannot do that"}]
            ),
        )
        msg = result[0]
        assert any(p.get("type") == "refusal" for p in msg["content"])

    def test_p_tool_to_ir(self):
        """Test OpenAI tool role message → IR ToolMessage."""
        result = cast(
            list[Any],
            self.message_ops.p_messages_to_ir(
                [{"role": "tool", "tool_call_id": "call_1", "content": "42"}]
            ),
        )
        assert result[0]["role"] == "tool"
        assert result[0]["content"][0]["type"] == "tool_result"
        assert result[0]["content"][0]["tool_call_id"] == "call_1"
        assert result[0]["content"][0]["result"] == "42"

    def test_p_function_to_ir(self):
        """Test OpenAI deprecated function role → IR ToolMessage."""
        result = cast(
            list[Any],
            self.message_ops.p_messages_to_ir(
                [{"role": "function", "name": "old_func", "content": "result"}]
            ),
        )
        assert result[0]["role"] == "tool"
        assert "legacy_function_old_func" in result[0]["content"][0]["tool_call_id"]

    def test_p_audio_input_to_ir(self):
        """Test OpenAI input_audio → IR FilePart."""
        result = cast(
            list[Any],
            self.message_ops.p_messages_to_ir(
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_audio",
                                "input_audio": {"data": "audio_data", "format": "mp3"},
                            }
                        ],
                    }
                ]
            ),
        )
        assert result[0]["content"][0]["type"] == "file"
        assert result[0]["content"][0]["file_data"]["media_type"] == "audio/mp3"

    # ==================== Round-trip ====================

    def test_messages_round_trip(self):
        """Test messages round-trip: IR → Provider → IR."""
        ir_messages = cast(
            list[Message],
            [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Be helpful"}],
                },
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello"}],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Hi!"}],
                },
            ],
        )
        provider_msgs, _ = self.message_ops.ir_messages_to_p(ir_messages)
        restored = cast(list[Any], self.message_ops.p_messages_to_ir(provider_msgs))

        assert len(restored) == 3
        assert restored[0]["role"] == "system"
        assert restored[0]["content"][0]["text"] == "Be helpful"
        assert restored[1]["role"] == "user"
        assert restored[1]["content"][0]["text"] == "Hello"
        assert restored[2]["role"] == "assistant"
        assert restored[2]["content"][0]["text"] == "Hi!"
