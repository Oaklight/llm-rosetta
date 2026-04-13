"""
OpenAI Chat MessageOps unit tests.
"""

from typing import Any, Union, cast

from llm_rosetta.converters.openai_chat.content_ops import OpenAIChatContentOps
from llm_rosetta.converters.openai_chat.message_ops import (
    OpenAIChatMessageOps,
    _has_multimodal_content,
)
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


class TestMultimodalToolResultPacking:
    """Tests for Phase 2 multimodal tool result dual encoding (packing/unpacking)."""

    def setup_method(self):
        """Set up test fixtures."""
        self.content_ops = OpenAIChatContentOps()
        self.tool_ops = OpenAIChatToolOps()
        self.message_ops = OpenAIChatMessageOps(self.content_ops, self.tool_ops)

    # --- Helper builders ---

    @staticmethod
    def _make_ir_conversation(
        tool_results: list[ToolResultPart],
    ) -> list[Message]:
        """Build a minimal IR conversation with an assistant tool_calls + tool results.

        Creates: [user "run tools"] → [assistant with tool_calls] → [tool results].
        """
        tool_calls = [
            ToolCallPart(
                type="tool_call",
                tool_call_id=tr["tool_call_id"],
                tool_name=f"fn_{tr['tool_call_id']}",
                tool_input={},
                tool_type="function",
            )
            for tr in tool_results
        ]
        return cast(
            list[Message],
            [
                {"role": "user", "content": [{"type": "text", "text": "run tools"}]},
                {"role": "assistant", "content": tool_calls},
                {"role": "tool", "content": list(tool_results)},
            ],
        )

    # ==================== Packing Tests ====================

    def test_text_only_no_packing(self):
        """Text-only tool result → no synthetic user message."""
        tr = ToolResultPart(
            type="tool_result",
            tool_call_id="call_1",
            result="plain text result",
        )
        ir_msgs = self._make_ir_conversation([tr])
        result, warnings = self.message_ops.ir_messages_to_p(ir_msgs)

        # Should be: user, assistant, tool — no synthetic user msg
        roles = [m["role"] for m in result]
        assert roles == ["user", "assistant", "tool"]
        assert result[2]["content"] == "plain text result"

    def test_image_only_packing(self):
        """Image-only tool result → json.dumps in tool msg + synthetic user msg."""
        tr = ToolResultPart(
            type="tool_result",
            tool_call_id="call_img",
            result=[
                {"type": "image", "image_url": "https://example.com/chart.png"},
            ],
        )
        ir_msgs = self._make_ir_conversation([tr])
        result, _ = self.message_ops.ir_messages_to_p(ir_msgs)

        roles = [m["role"] for m in result]
        assert roles == ["user", "assistant", "tool", "user"]

        # Tool message has json.dumps fallback
        import json

        tool_content = json.loads(result[2]["content"])
        assert tool_content[0]["type"] == "image"

        # Synthetic user message has tagged image_url
        synthetic = result[3]
        parts = synthetic["content"]
        assert parts[0]["type"] == "text"
        assert '<tool-content call-id="call_img">' in parts[0]["text"]
        assert any(p["type"] == "image_url" for p in parts)
        assert parts[-1]["type"] == "text"
        assert parts[-1]["text"] == "</tool-content>"

    def test_mixed_text_image_packing(self):
        """Text+image tool result → both packed in synthetic user msg."""
        tr = ToolResultPart(
            type="tool_result",
            tool_call_id="call_mix",
            result=[
                {"type": "text", "text": "Here is the chart:"},
                {"type": "image", "image_url": "https://example.com/img.png"},
            ],
        )
        ir_msgs = self._make_ir_conversation([tr])
        result, _ = self.message_ops.ir_messages_to_p(ir_msgs)

        roles = [m["role"] for m in result]
        assert roles == ["user", "assistant", "tool", "user"]

        synthetic = result[3]["content"]
        # open tag, text, image_url, close tag
        assert len(synthetic) == 4
        assert synthetic[0]["text"].startswith("<tool-content")
        assert synthetic[1]["type"] == "text"
        assert synthetic[1]["text"] == "Here is the chart:"
        assert synthetic[2]["type"] == "image_url"
        assert synthetic[3]["text"] == "</tool-content>"

    def test_multiple_tools_partial_packing(self):
        """Two tools: one text-only, one image → only image tool gets packed."""
        tr_text = ToolResultPart(
            type="tool_result",
            tool_call_id="call_t",
            result="just text",
        )
        tr_img = ToolResultPart(
            type="tool_result",
            tool_call_id="call_i",
            result=[{"type": "image", "image_url": "https://example.com/x.png"}],
        )
        ir_msgs = self._make_ir_conversation([tr_text, tr_img])
        result, _ = self.message_ops.ir_messages_to_p(ir_msgs)

        roles = [m["role"] for m in result]
        # user, assistant, tool(text), tool(img), user(synthetic)
        assert roles == ["user", "assistant", "tool", "tool", "user"]

        # Only one <tool-content> section in synthetic msg
        synthetic = result[4]["content"]
        open_tags = [
            p
            for p in synthetic
            if p.get("type") == "text" and "<tool-content" in p.get("text", "")
        ]
        assert len(open_tags) == 1
        assert 'call-id="call_i"' in open_tags[0]["text"]

    def test_multiple_multimodal_one_user_msg(self):
        """Two multimodal tools → one synthetic user msg with two sections."""
        tr_a = ToolResultPart(
            type="tool_result",
            tool_call_id="call_a",
            result=[{"type": "image", "image_url": "https://example.com/a.png"}],
        )
        tr_b = ToolResultPart(
            type="tool_result",
            tool_call_id="call_b",
            result=[{"type": "image", "image_url": "https://example.com/b.png"}],
        )
        ir_msgs = self._make_ir_conversation([tr_a, tr_b])
        result, _ = self.message_ops.ir_messages_to_p(ir_msgs)

        roles = [m["role"] for m in result]
        assert roles == ["user", "assistant", "tool", "tool", "user"]

        synthetic = result[4]["content"]
        open_tags = [
            p
            for p in synthetic
            if p.get("type") == "text" and "<tool-content" in p.get("text", "")
        ]
        assert len(open_tags) == 2
        close_tags = [
            p
            for p in synthetic
            if p.get("type") == "text" and p.get("text") == "</tool-content>"
        ]
        assert len(close_tags) == 2

    # ==================== Unpacking Tests ====================

    def test_unpack_synthetic_user_message(self):
        """Provider messages with synthetic msg → IR tool result recovers multimodal."""
        provider_msgs = [
            {"role": "user", "content": "run tools"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_u1",
                        "type": "function",
                        "function": {"name": "plot", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_u1",
                "content": '[{"type":"image","image_url":"https://example.com/plot.png"}]',
            },
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": '<tool-content call-id="call_u1">'},
                    {
                        "type": "image_url",
                        "image_url": {"url": "https://example.com/plot.png"},
                    },
                    {"type": "text", "text": "</tool-content>"},
                ],
            },
        ]
        ir_msgs = cast(list[Message], self.message_ops.p_messages_to_ir(provider_msgs))

        # Should have: user, assistant, tool — synthetic user removed
        roles = [m["role"] for m in ir_msgs]
        assert roles == ["user", "assistant", "tool"]

        # Tool result should have multimodal content (from synthetic msg)
        tool_result = ir_msgs[2]["content"][0]
        assert tool_result["type"] == "tool_result"
        assert isinstance(tool_result["result"], list)
        assert tool_result["result"][0]["type"] == "image"

    def test_unpack_preserves_normal_user(self):
        """Normal user message is not affected by unpacking."""
        provider_msgs = [
            {"role": "user", "content": "Hello, how are you?"},
            {"role": "assistant", "content": "I'm fine!"},
        ]
        ir_msgs = cast(list[Message], self.message_ops.p_messages_to_ir(provider_msgs))

        assert len(ir_msgs) == 2
        assert ir_msgs[0]["role"] == "user"
        assert cast(dict, ir_msgs[0]["content"][0])["text"] == "Hello, how are you?"

    # ==================== Roundtrip Tests ====================

    def test_roundtrip_multimodal(self):
        """IR → Chat → IR roundtrip preserves image content."""
        tr = ToolResultPart(
            type="tool_result",
            tool_call_id="call_rt",
            result=[
                {"type": "text", "text": "chart output:"},
                {"type": "image", "image_url": "https://example.com/chart.png"},
            ],
        )
        ir_msgs = self._make_ir_conversation([tr])
        provider_msgs, _ = self.message_ops.ir_messages_to_p(ir_msgs)
        restored = cast(list[Message], self.message_ops.p_messages_to_ir(provider_msgs))

        # Find the tool message
        tool_msg = [m for m in restored if m["role"] == "tool"][0]
        result = tool_msg["content"][0]["result"]
        assert isinstance(result, list)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "chart output:"
        assert result[1]["type"] == "image"

    def test_roundtrip_text_only(self):
        """IR → Chat → IR roundtrip for text-only (no packing occurs)."""
        tr = ToolResultPart(
            type="tool_result",
            tool_call_id="call_txt",
            result="just text",
        )
        ir_msgs = self._make_ir_conversation([tr])
        provider_msgs, _ = self.message_ops.ir_messages_to_p(ir_msgs)
        restored = cast(list[Message], self.message_ops.p_messages_to_ir(provider_msgs))

        tool_msg = [m for m in restored if m["role"] == "tool"][0]
        result = tool_msg["content"][0]["result"]
        assert result == "just text"

    def test_roundtrip_mixed_tools(self):
        """IR → Chat → IR with mixed text/multimodal tools."""
        tr_text = ToolResultPart(
            type="tool_result",
            tool_call_id="call_m1",
            result="plain result",
        )
        tr_img = ToolResultPart(
            type="tool_result",
            tool_call_id="call_m2",
            result=[
                {"type": "image", "image_url": "https://example.com/img.png"},
            ],
        )
        ir_msgs = self._make_ir_conversation([tr_text, tr_img])
        provider_msgs, _ = self.message_ops.ir_messages_to_p(ir_msgs)
        restored = cast(list[Message], self.message_ops.p_messages_to_ir(provider_msgs))

        tool_msgs = [m for m in restored if m["role"] == "tool"]
        # Text tool result preserved as string
        text_tr = [
            m for m in tool_msgs if m["content"][0]["tool_call_id"] == "call_m1"
        ][0]
        assert isinstance(text_tr["content"][0]["result"], str)
        # Image tool result preserved as list with ImagePart
        img_tr = [m for m in tool_msgs if m["content"][0]["tool_call_id"] == "call_m2"][
            0
        ]
        assert isinstance(img_tr["content"][0]["result"], list)
        assert img_tr["content"][0]["result"][0]["type"] == "image"

    # ==================== Detection & Edge Cases ====================

    def test_is_synthetic_detection(self):
        """_is_synthetic_tool_content_msg correctly identifies synthetic vs normal."""
        synthetic = {
            "role": "user",
            "content": [
                {"type": "text", "text": '<tool-content call-id="call_1">'},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/x.png"},
                },
                {"type": "text", "text": "</tool-content>"},
            ],
        }
        normal_user = {"role": "user", "content": "Hello!"}
        normal_user_list = {
            "role": "user",
            "content": [{"type": "text", "text": "Hello!"}],
        }
        assistant = {"role": "assistant", "content": "Hi!"}

        assert OpenAIChatMessageOps._is_synthetic_tool_content_msg(synthetic) is True
        assert OpenAIChatMessageOps._is_synthetic_tool_content_msg(normal_user) is False
        assert (
            OpenAIChatMessageOps._is_synthetic_tool_content_msg(normal_user_list)
            is False
        )
        assert OpenAIChatMessageOps._is_synthetic_tool_content_msg(assistant) is False

    def test_file_part_skipped_with_warning(self):
        """File parts in multimodal tool result produce warning, skipped during packing."""
        tr = ToolResultPart(
            type="tool_result",
            tool_call_id="call_f",
            result=[
                {"type": "text", "text": "some data"},
                {
                    "type": "file",
                    "file_data": {"data": "abc", "media_type": "text/plain"},
                },
            ],
        )
        ir_msgs = self._make_ir_conversation([tr])
        result, warnings = self.message_ops.ir_messages_to_p(ir_msgs)

        # Should still pack (text part survives)
        roles = [m["role"] for m in result]
        assert roles == ["user", "assistant", "tool", "user"]
        assert any("File content not supported" in w for w in warnings)

        # Synthetic msg should have text but not file
        synthetic = result[3]["content"]
        types = [p["type"] for p in synthetic]
        assert "text" in types
        # No file-related part type in synthetic
        assert "file" not in types

    def test_has_multimodal_content_helper(self):
        """_has_multimodal_content correctly detects multimodal vs text-only."""
        assert _has_multimodal_content("just a string") is False
        assert _has_multimodal_content([{"type": "text", "text": "hi"}]) is False
        assert _has_multimodal_content([{"type": "image", "image_url": "x"}]) is True
        assert (
            _has_multimodal_content(
                [{"type": "text", "text": "hi"}, {"type": "image", "image_url": "x"}]
            )
            is True
        )
        assert _has_multimodal_content([]) is False
        assert _has_multimodal_content(None) is False
