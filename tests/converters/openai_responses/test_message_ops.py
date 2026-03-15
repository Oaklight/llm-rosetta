"""
OpenAI Responses MessageOps unit tests.
"""

from typing import Any, Union, cast

from llm_rosetta.converters.openai_responses.content_ops import (
    OpenAIResponsesContentOps,
)
from llm_rosetta.converters.openai_responses.message_ops import (
    OpenAIResponsesMessageOps,
)
from llm_rosetta.converters.openai_responses.tool_ops import OpenAIResponsesToolOps
from llm_rosetta.types.ir import Message, ToolCallPart, ToolResultPart
from llm_rosetta.types.ir.extensions import ExtensionItem


class TestOpenAIResponsesMessageOps:
    """Unit tests for OpenAIResponsesMessageOps."""

    def setup_method(self):
        """Set up test fixtures."""
        self.content_ops = OpenAIResponsesContentOps()
        self.tool_ops = OpenAIResponsesToolOps()
        self.message_ops = OpenAIResponsesMessageOps(self.content_ops, self.tool_ops)

    # ==================== IR → Provider ====================

    def test_system_message_to_p(self):
        """Test IR system message → OpenAI Responses message item."""
        messages = cast(list[Message], [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are helpful."}],
            }
        ])
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        assert len(result) == 1
        assert result[0]["type"] == "message"
        assert result[0]["role"] == "system"
        assert result[0]["content"][0]["type"] == "input_text"
        assert result[0]["content"][0]["text"] == "You are helpful."
        assert warnings == []

    def test_user_text_message_to_p(self):
        """Test IR user text message → OpenAI Responses message item."""
        messages = cast(list[Message], [{"role": "user", "content": [{"type": "text", "text": "Hello!"}]}])
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        assert len(result) == 1
        assert result[0]["type"] == "message"
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "input_text"
        assert result[0]["content"][0]["text"] == "Hello!"

    def test_user_multimodal_message_to_p(self):
        """Test IR user multimodal message → OpenAI Responses items."""
        messages = cast(list[Message], [
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
        ])
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        assert len(result) == 1
        msg = result[0]
        assert msg["type"] == "message"
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "input_text"
        assert msg["content"][1]["type"] == "input_image"

    def test_user_message_with_file_to_p(self):
        """Test IR user message with file → OpenAI Responses items."""
        messages = cast(list[Message], [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Read this file"},
                    {
                        "type": "file",
                        "file_name": "doc.pdf",
                        "file_url": "https://example.com/doc.pdf",
                    },
                ],
            }
        ])
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        assert len(result) == 1
        msg = result[0]
        assert len(msg["content"]) == 2
        assert msg["content"][1]["type"] == "input_file"

    def test_developer_message_to_p(self):
        """Test IR developer message → OpenAI Responses developer message item."""
        messages = cast(list[Message], [
            {
                "role": "developer",
                "content": [{"type": "text", "text": "Developer instructions"}],
            }
        ])
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        assert len(result) == 1
        assert result[0]["role"] == "developer"

    def test_assistant_text_message_to_p(self):
        """Test IR assistant text message → OpenAI Responses output items."""
        messages = cast(list[Message], [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi there!"}],
            }
        ])
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        assert len(result) == 1
        msg = result[0]
        assert msg["type"] == "message"
        assert msg["role"] == "assistant"
        assert msg["content"][0]["type"] == "output_text"
        assert msg["content"][0]["text"] == "Hi there!"

    def test_assistant_tool_call_message_to_p(self):
        """Test IR assistant with tool calls → function_call items."""
        messages = cast(list[Message], [
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
        ])
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        assert len(result) == 1
        assert result[0]["type"] == "function_call"
        assert result[0]["name"] == "get_weather"
        assert result[0]["call_id"] == "call_1"

    def test_assistant_text_and_tool_calls_to_p(self):
        """Test assistant with both text and tool calls → multiple items."""
        messages = cast(list[Message], [
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
        ])
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        # Should produce a message item + a function_call item
        assert len(result) == 2
        assert result[0]["type"] == "message"
        assert result[0]["content"][0]["type"] == "output_text"
        assert result[1]["type"] == "function_call"

    def test_assistant_reasoning_to_p(self):
        """Test assistant with reasoning → reasoning item."""
        messages = cast(list[Message], [
            {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "reasoning": "thinking..."},
                    {"type": "text", "text": "The answer is 42"},
                ],
            }
        ])
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        # Reasoning items come first, then message
        assert len(result) == 2
        assert result[0]["type"] == "reasoning"
        assert result[1]["type"] == "message"

    def test_tool_message_to_p(self):
        """Test IR tool message → function_call_output items."""
        messages = cast(list[Message], [
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
        ])
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        assert len(result) == 1
        assert result[0]["type"] == "function_call_output"
        assert result[0]["call_id"] == "call_1"
        assert result[0]["output"] == "Result data"

    def test_user_message_with_tool_result_to_p(self):
        """Test user message with ToolResultPart → function_call_output items."""
        messages = cast(list[Message], [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call_1",
                        "result": "42",
                    },
                ],
            }
        ])
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        # Tool results become separate function_call_output items
        assert any(item["type"] == "function_call_output" for item in result)

    def test_extension_items_handling(self):
        """Test extension items produce warnings or are handled."""
        items = cast(list[Union[Message, ExtensionItem]], [
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
        ])
        result, warnings = self.message_ops.ir_messages_to_p(items)
        # system_event is converted, batch_marker and session_control produce warnings
        assert len(warnings) >= 2

    def test_multi_turn_to_p(self):
        """Test multi-turn conversation → multiple items."""
        messages = cast(list[Message], [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "Hi!"}],
            },
            {"role": "user", "content": [{"type": "text", "text": "How are you?"}]},
        ])
        result, warnings = self.message_ops.ir_messages_to_p(messages)
        assert len(result) == 3
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"
        assert result[2]["role"] == "user"

    # ==================== Provider → IR ====================

    def test_p_message_to_ir_user(self):
        """Test OpenAI Responses user message → IR UserMessage."""
        result = cast(list[Any], self.message_ops.p_messages_to_ir(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "Hello"}],
                }
            ]
        ))
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Hello"

    def test_p_message_to_ir_system(self):
        """Test OpenAI Responses system message → IR SystemMessage."""
        result = cast(list[Any], self.message_ops.p_messages_to_ir(
            [
                {
                    "type": "message",
                    "role": "system",
                    "content": [{"type": "input_text", "text": "Be helpful"}],
                }
            ]
        ))
        assert len(result) == 1
        assert result[0]["role"] == "system"
        assert result[0]["content"][0]["text"] == "Be helpful"

    def test_p_message_to_ir_assistant(self):
        """Test OpenAI Responses assistant message → IR AssistantMessage."""
        result = cast(list[Any], self.message_ops.p_messages_to_ir(
            [
                {
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "Hello!"}],
                }
            ]
        ))
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Hello!"

    def test_p_message_to_ir_string_content(self):
        """Test OpenAI Responses message with string content → IR."""
        result = cast(list[Any], self.message_ops.p_messages_to_ir(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": "Hello string",
                }
            ]
        ))
        assert len(result) == 1
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Hello string"

    def test_p_function_call_to_ir(self):
        """Test OpenAI Responses function_call → IR assistant with ToolCallPart."""
        result = cast(list[Any], self.message_ops.p_messages_to_ir(
            [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city": "NYC"}',
                }
            ]
        ))
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        tc = result[0]["content"][0]
        assert tc["type"] == "tool_call"
        assert tc["tool_call_id"] == "call_1"
        assert tc["tool_name"] == "get_weather"
        assert tc["tool_input"] == {"city": "NYC"}

    def test_p_function_call_output_to_ir(self):
        """Test OpenAI Responses function_call_output → IR user with ToolResultPart."""
        result = cast(list[Any], self.message_ops.p_messages_to_ir(
            [
                {
                    "type": "function_call_output",
                    "call_id": "call_1",
                    "output": "Sunny, 25°C",
                }
            ]
        ))
        assert len(result) == 1
        assert result[0]["role"] == "user"
        tr = result[0]["content"][0]
        assert tr["type"] == "tool_result"
        assert tr["tool_call_id"] == "call_1"
        assert tr["result"] == "Sunny, 25°C"

    def test_p_reasoning_to_ir(self):
        """Test OpenAI Responses reasoning item → IR assistant with ReasoningPart."""
        result = cast(list[Any], self.message_ops.p_messages_to_ir(
            [{"type": "reasoning", "content": "Let me think..."}]
        ))
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"][0]["type"] == "reasoning"
        assert result[0]["content"][0]["reasoning"] == "Let me think..."

    def test_p_reasoning_empty_to_ir(self):
        """Test OpenAI Responses reasoning with empty content → skipped."""
        result = cast(list[Any], self.message_ops.p_messages_to_ir(
            [{"type": "reasoning", "content": None}]
        ))
        # Empty reasoning returns None, so no content is added
        assert len(result) == 0

    def test_p_system_event_to_ir(self):
        """Test OpenAI Responses system_event → IR extension item."""
        result = cast(list[Any], self.message_ops.p_messages_to_ir(
            [
                {
                    "type": "system_event",
                    "event_type": "session_start",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "message": "Session started",
                }
            ]
        ))
        assert len(result) == 1
        assert result[0]["type"] == "system_event"
        assert result[0]["event_type"] == "session_start"

    def test_p_consecutive_tool_calls_grouped(self):
        """Test consecutive function_call items are grouped into one assistant message."""
        result = cast(list[Any], self.message_ops.p_messages_to_ir(
            [
                {
                    "type": "function_call",
                    "call_id": "call_1",
                    "name": "get_weather",
                    "arguments": '{"city": "NYC"}',
                },
                {
                    "type": "function_call",
                    "call_id": "call_2",
                    "name": "get_time",
                    "arguments": '{"tz": "EST"}',
                },
            ]
        ))
        # Both should be in the same assistant message
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert len(result[0]["content"]) == 2

    def test_p_multimodal_message_to_ir(self):
        """Test OpenAI Responses multimodal message → IR with multiple parts."""
        result = cast(list[Any], self.message_ops.p_messages_to_ir(
            [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "What's this?"},
                        {
                            "type": "input_image",
                            "image_url": "https://example.com/img.jpg",
                            "detail": "auto",
                        },
                    ],
                }
            ]
        ))
        assert len(result) == 1
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "image"

    def test_p_full_conversation_to_ir(self):
        """Test full conversation with message + function_call + output → IR."""
        items = [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "What's the weather?"}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "get_weather",
                "arguments": '{"city": "NYC"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "Sunny, 25°C",
            },
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "The weather in NYC is sunny."}
                ],
            },
        ]
        result = cast(list[Any], self.message_ops.p_messages_to_ir(items))
        assert len(result) == 4
        assert result[0]["role"] == "user"
        assert result[1]["role"] == "assistant"  # function_call
        assert result[1]["content"][0]["type"] == "tool_call"
        assert result[2]["role"] == "user"  # function_call_output
        assert result[2]["content"][0]["type"] == "tool_result"
        assert result[3]["role"] == "assistant"  # final message

    # ==================== Round-trip ====================

    def test_messages_round_trip_simple(self):
        """Test simple messages round-trip: IR → Provider → IR."""
        ir_messages = cast(list[Message], [
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
        ])
        provider_items, _ = self.message_ops.ir_messages_to_p(ir_messages)
        restored = cast(list[Any], self.message_ops.p_messages_to_ir(provider_items))

        assert len(restored) == 3
        assert restored[0]["role"] == "system"
        assert restored[0]["content"][0]["text"] == "Be helpful"
        assert restored[1]["role"] == "user"
        assert restored[1]["content"][0]["text"] == "Hello"
        assert restored[2]["role"] == "assistant"
        assert restored[2]["content"][0]["text"] == "Hi!"

    def test_messages_round_trip_with_tool_calls(self):
        """Test messages round-trip with tool calls."""
        ir_messages = cast(list[Message], [
            {
                "role": "user",
                "content": [{"type": "text", "text": "What's the weather?"}],
            },
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
            },
        ])
        provider_items, _ = self.message_ops.ir_messages_to_p(ir_messages)
        restored = cast(list[Any], self.message_ops.p_messages_to_ir(provider_items))

        assert len(restored) == 2
        assert restored[0]["role"] == "user"
        assert restored[1]["role"] == "assistant"
        tc = restored[1]["content"][0]
        assert tc["type"] == "tool_call"
        assert tc["tool_name"] == "get_weather"
        assert tc["tool_input"] == {"city": "NYC"}
