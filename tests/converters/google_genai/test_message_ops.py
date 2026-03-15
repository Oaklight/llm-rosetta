"""
Google GenAI MessageOps unit tests.
"""

from typing import Any, Union, cast

import pytest

from llm_rosetta.converters.google_genai.content_ops import GoogleGenAIContentOps
from llm_rosetta.converters.google_genai.message_ops import GoogleGenAIMessageOps
from llm_rosetta.converters.google_genai.tool_ops import GoogleGenAIToolOps
from llm_rosetta.types.ir import Message
from llm_rosetta.types.ir.extensions import ExtensionItem


class TestGoogleGenAIMessageOps:
    """Unit tests for GoogleGenAIMessageOps."""

    def setup_method(self):
        """Set up test fixtures."""
        content_ops = GoogleGenAIContentOps()
        tool_ops = GoogleGenAIToolOps()
        self.message_ops = GoogleGenAIMessageOps(content_ops, tool_ops)

    # ==================== IR → Provider ====================

    def test_ir_user_message_to_p(self):
        """Test IR user message → Google Content."""
        ir_messages = cast(list[Message], [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}])
        result, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["parts"][0]["text"] == "Hello"
        assert len(warnings) == 0

    def test_ir_assistant_message_to_p(self):
        """Test IR assistant message → Google Content with model role."""
        ir_messages = cast(list[Message], [
            {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]}
        ])
        result, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(result) == 1
        assert result[0]["role"] == "model"
        assert result[0]["parts"][0]["text"] == "Hi!"

    def test_ir_system_message_skipped(self):
        """Test IR system message is skipped (handled at converter level)."""
        ir_messages = cast(list[Message], [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are helpful."}],
            },
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ])
        result, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_ir_assistant_with_tool_call(self):
        """Test IR assistant message with tool call → Google Content."""
        ir_messages = cast(list[Message], [
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
        ])
        result, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "model"
        assert len(msg["parts"]) == 2
        assert msg["parts"][0]["text"] == "Let me search"
        assert "function_call" in msg["parts"][1]
        assert msg["parts"][1]["function_call"]["name"] == "search"

    def test_ir_user_with_tool_result(self):
        """Test IR user message with tool result → Google Content."""
        ir_messages = cast(list[Message], [
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
        ])
        result, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(result) == 1
        msg = result[0]
        assert msg["role"] == "user"
        assert "function_response" in msg["parts"][0]

    def test_ir_user_with_image(self):
        """Test IR user message with image → Google Content."""
        ir_messages = cast(list[Message], [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "data": "base64data",
                        "media_type": "image/jpeg",
                    }
                ],
            }
        ])
        result, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(result) == 1
        msg = result[0]
        assert "inline_data" in msg["parts"][0]
        assert msg["parts"][0]["inline_data"]["mime_type"] == "image/jpeg"

    def test_ir_assistant_with_reasoning(self):
        """Test IR assistant message with reasoning → Google Content."""
        ir_messages = cast(list[Message], [
            {
                "role": "assistant",
                "content": [
                    {"type": "reasoning", "reasoning": "I need to think."},
                    {"type": "text", "text": "Here is my answer."},
                ],
            }
        ])
        result, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(result) == 1
        msg = result[0]
        assert msg["parts"][0]["thought"] is True
        assert msg["parts"][0]["text"] == "I need to think."
        assert msg["parts"][1]["text"] == "Here is my answer."

    def test_ir_multi_part_message(self):
        """Test IR message with multiple content parts."""
        ir_messages = cast(list[Message], [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What is this?"},
                    {
                        "type": "image",
                        "data": "imgdata",
                        "media_type": "image/png",
                    },
                    {"type": "text", "text": "Please describe it."},
                ],
            }
        ])
        result, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(result) == 1
        msg = result[0]
        assert len(msg["parts"]) == 3
        assert msg["parts"][0]["text"] == "What is this?"
        assert "inline_data" in msg["parts"][1]
        assert msg["parts"][2]["text"] == "Please describe it."

    def test_ir_unsupported_content_type_warns(self):
        """Test unsupported content type emits warning."""
        ir_messages = cast(list[Message], [
            {
                "role": "user",
                "content": [{"type": "unknown_type", "data": "something"}],
            }
        ])
        with pytest.warns(UserWarning, match="不支持的内容类型"):
            result, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(result) == 1
        # The unsupported part should be skipped (None filtered out)
        assert len(result[0]["parts"]) == 0

    def test_extension_item_produces_warning(self):
        """Test extension item produces warning."""
        ir_messages = cast(list[Union[Message, ExtensionItem]], [
            {
                "type": "system_event",
                "event_type": "session_start",
                "timestamp": "2024-01-01T00:00:00Z",
            },
        ])
        result, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(result) == 0
        assert any("不支持扩展项类型" in w for w in warnings)

    def test_ir_tool_result_with_context(self):
        """Test IR tool result with context lookup for function name."""
        ir_messages = cast(list[Message], [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_123",
                        "tool_name": "get_weather",
                        "tool_input": {"city": "NYC"},
                        "tool_type": "function",
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_call_id": "call_123",
                        "result": "Sunny, 25°C",
                    }
                ],
            },
        ])
        result, warnings = self.message_ops.ir_messages_to_p(ir_messages)
        assert len(result) == 2
        # The tool result should use the function name from context
        tool_result_part = result[1]["parts"][0]
        assert tool_result_part["function_response"]["name"] == "get_weather"

    # ==================== Provider → IR ====================

    def test_p_user_message_to_ir(self):
        """Test Google user Content → IR."""
        provider_messages = [{"role": "user", "parts": [{"text": "Hello"}]}]
        result = cast(list[Any], self.message_ops.p_messages_to_ir(provider_messages))
        assert len(result) == 1
        assert result[0]["role"] == "user"
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][0]["text"] == "Hello"

    def test_p_model_message_to_ir(self):
        """Test Google model Content → IR assistant message."""
        provider_messages = [{"role": "model", "parts": [{"text": "Hi!"}]}]
        result = cast(list[Any], self.message_ops.p_messages_to_ir(provider_messages))
        assert len(result) == 1
        assert result[0]["role"] == "assistant"
        assert result[0]["content"][0]["text"] == "Hi!"

    def test_p_function_call_to_ir(self):
        """Test Google function_call Part → IR ToolCallPart."""
        provider_messages = [
            {
                "role": "model",
                "parts": [
                    {
                        "function_call": {
                            "name": "get_weather",
                            "args": {"city": "SF"},
                        }
                    }
                ],
            }
        ]
        result = cast(list[Any], self.message_ops.p_messages_to_ir(provider_messages))
        tc = result[0]["content"][0]
        assert tc["type"] == "tool_call"
        assert tc["tool_name"] == "get_weather"
        assert tc["tool_input"] == {"city": "SF"}

    def test_p_function_response_to_ir(self):
        """Test Google function_response Part → IR ToolResultPart."""
        provider_messages = [
            {
                "role": "user",
                "parts": [
                    {
                        "function_response": {
                            "name": "get_weather",
                            "response": {"output": "Sunny"},
                        }
                    }
                ],
            }
        ]
        result = cast(list[Any], self.message_ops.p_messages_to_ir(provider_messages))
        tr = result[0]["content"][0]
        assert tr["type"] == "tool_result"
        assert tr["result"] == "Sunny"

    def test_p_thought_to_ir(self):
        """Test Google thought Part → IR ReasoningPart."""
        provider_messages = [
            {
                "role": "model",
                "parts": [
                    {"thought": True, "text": "Let me think..."},
                    {"text": "Here is my answer."},
                ],
            }
        ]
        result = cast(list[Any], self.message_ops.p_messages_to_ir(provider_messages))
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["type"] == "reasoning"
        assert result[0]["content"][0]["reasoning"] == "Let me think..."
        assert result[0]["content"][1]["type"] == "text"

    def test_p_multi_part_message_to_ir(self):
        """Test Google Content with multiple parts → IR."""
        provider_messages = [
            {
                "role": "user",
                "parts": [
                    {"text": "Look at this:"},
                    {"inline_data": {"mime_type": "image/png", "data": "imgdata"}},
                ],
            }
        ]
        result = cast(list[Any], self.message_ops.p_messages_to_ir(provider_messages))
        assert len(result) == 1
        assert len(result[0]["content"]) == 2
        assert result[0]["content"][0]["type"] == "text"
        assert result[0]["content"][1]["type"] == "image"

    def test_p_rest_api_function_call_to_ir(self):
        """Test Google REST API functionCall format → IR."""
        provider_messages = [
            {
                "role": "model",
                "parts": [
                    {
                        "functionCall": {
                            "name": "search",
                            "args": {"query": "test"},
                        }
                    }
                ],
            }
        ]
        result = cast(list[Any], self.message_ops.p_messages_to_ir(provider_messages))
        tc = result[0]["content"][0]
        assert tc["type"] == "tool_call"
        assert tc["tool_name"] == "search"

    def test_p_rest_api_function_response_to_ir(self):
        """Test Google REST API functionResponse format → IR."""
        provider_messages = [
            {
                "role": "user",
                "parts": [
                    {
                        "functionResponse": {
                            "name": "search",
                            "response": {"output": "results"},
                        }
                    }
                ],
            }
        ]
        result = cast(list[Any], self.message_ops.p_messages_to_ir(provider_messages))
        tr = result[0]["content"][0]
        assert tr["type"] == "tool_result"

    def test_p_empty_parts_returns_none(self):
        """Test Google Content with no convertible parts returns None."""
        provider_messages = [{"role": "model", "parts": [{"text": ""}]}]
        result = cast(list[Any], self.message_ops.p_messages_to_ir(provider_messages))
        # Empty text is ignored, so no content parts → message is None
        assert len(result) == 0

    def test_p_non_dict_message_skipped(self):
        """Test non-dict message is skipped."""
        provider_messages = ["not a dict", {"role": "user", "parts": [{"text": "Hi"}]}]
        result = cast(list[Any], self.message_ops.p_messages_to_ir(provider_messages))
        assert len(result) == 1
        assert result[0]["content"][0]["text"] == "Hi"

    def test_p_thought_signature_only_ignored(self):
        """Test Part with only thoughtSignature is silently ignored."""
        provider_messages = [
            {
                "role": "model",
                "parts": [
                    {"text": "Hello"},
                    {"thoughtSignature": "sig123"},
                ],
            }
        ]
        # Should not warn about unknown part types for thoughtSignature
        result = cast(list[Any], self.message_ops.p_messages_to_ir(provider_messages))
        assert len(result) == 1
        assert result[0]["content"][0]["type"] == "text"

    # ==================== System Instruction Helpers ====================

    def test_extract_system_instruction_single(self):
        """Test extracting a single system message."""
        ir_messages = cast(list[Message], [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are helpful."}],
            },
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ])
        system_instruction, remaining = self.message_ops.extract_system_instruction(
            ir_messages
        )
        remaining = cast(list[Any], remaining)
        assert system_instruction is not None
        assert system_instruction["role"] == "user"
        assert len(system_instruction["parts"]) == 1
        assert system_instruction["parts"][0]["text"] == "You are helpful."
        assert len(remaining) == 1
        assert remaining[0]["role"] == "user"

    def test_extract_system_instruction_multiple(self):
        """Test extracting multiple system messages (merged)."""
        ir_messages = cast(list[Message], [
            {
                "role": "system",
                "content": [{"type": "text", "text": "Be helpful."}],
            },
            {
                "role": "system",
                "content": [{"type": "text", "text": "Be concise."}],
            },
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ])
        system_instruction, remaining = self.message_ops.extract_system_instruction(
            ir_messages
        )
        assert system_instruction is not None
        assert len(system_instruction["parts"]) == 2
        assert system_instruction["parts"][0]["text"] == "Be helpful."
        assert system_instruction["parts"][1]["text"] == "Be concise."
        assert len(remaining) == 1

    def test_extract_system_instruction_none(self):
        """Test no system messages returns None."""
        ir_messages = cast(list[Message], [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ])
        system_instruction, remaining = self.message_ops.extract_system_instruction(
            ir_messages
        )
        assert system_instruction is None
        assert len(remaining) == 1

    def test_extract_system_instruction_non_text_parts_ignored(self):
        """Test non-text parts in system messages are ignored."""
        ir_messages = cast(list[Message], [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": "Be helpful."},
                    {"type": "image", "data": "imgdata", "media_type": "image/png"},
                ],
            },
        ])
        system_instruction, remaining = self.message_ops.extract_system_instruction(
            ir_messages
        )
        assert system_instruction is not None
        assert len(system_instruction["parts"]) == 1
        assert system_instruction["parts"][0]["text"] == "Be helpful."

    # ==================== Round Trip ====================

    def test_round_trip_messages(self):
        """Test message round-trip: IR → Provider → IR."""
        original = cast(list[Message], [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "Hi!"}]},
        ])
        provider, _ = self.message_ops.ir_messages_to_p(original)
        restored = cast(list[Any], self.message_ops.p_messages_to_ir(provider))
        assert len(restored) == 2
        assert restored[0]["role"] == "user"
        assert restored[0]["content"][0]["text"] == "Hello"
        assert restored[1]["role"] == "assistant"
        assert restored[1]["content"][0]["text"] == "Hi!"

    def test_round_trip_with_tool_call(self):
        """Test tool call round-trip: IR → Provider → IR."""
        original = cast(list[Message], [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "call_123",
                        "tool_name": "get_weather",
                        "tool_input": {"city": "NYC"},
                        "tool_type": "function",
                    }
                ],
            }
        ])
        provider, _ = self.message_ops.ir_messages_to_p(original)
        restored = cast(list[Any], self.message_ops.p_messages_to_ir(provider))
        assert len(restored) == 1
        tc = restored[0]["content"][0]
        assert tc["type"] == "tool_call"
        assert tc["tool_name"] == "get_weather"
        assert tc["tool_input"] == {"city": "NYC"}
