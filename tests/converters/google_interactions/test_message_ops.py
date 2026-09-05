"""Tests for Google Interactions message_ops."""

import pytest
from llm_rosetta.converters.google_interactions.message_ops import (
    GoogleInteractionsMessageOps,
)


@pytest.fixture
def ops():
    return GoogleInteractionsMessageOps()


class TestStepsToIRMessages:
    def test_simple_user_input(self, ops):
        steps = [
            {
                "type": "user_input",
                "content": [{"type": "text", "text": "Hello"}],
            }
        ]
        messages = ops.p_steps_to_ir_messages(steps)
        assert len(messages) == 1
        assert messages[0]["role"] == "user"
        assert messages[0]["content"][0]["text"] == "Hello"

    def test_simple_model_output(self, ops):
        steps = [
            {
                "type": "model_output",
                "content": [{"type": "text", "text": "Hi there!"}],
            }
        ]
        messages = ops.p_steps_to_ir_messages(steps)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"][0]["text"] == "Hi there!"

    def test_multi_turn(self, ops):
        steps = [
            {
                "type": "user_input",
                "content": [{"type": "text", "text": "My name is Alice."}],
            },
            {
                "type": "model_output",
                "content": [{"type": "text", "text": "Hello Alice!"}],
            },
            {
                "type": "user_input",
                "content": [{"type": "text", "text": "What is my name?"}],
            },
        ]
        messages = ops.p_steps_to_ir_messages(steps)
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"

    def test_thought_and_model_output_merge(self, ops):
        """Consecutive thought + model_output should merge into one assistant message."""
        steps = [
            {
                "type": "thought",
                "signature": "sig_abc",
                "summary": [{"type": "text", "text": "Let me think..."}],
            },
            {
                "type": "model_output",
                "content": [{"type": "text", "text": "The answer is 42."}],
            },
        ]
        messages = ops.p_steps_to_ir_messages(steps)
        assert len(messages) == 1
        msg = messages[0]
        assert msg["role"] == "assistant"
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "reasoning"
        assert msg["content"][0]["signature"] == "sig_abc"
        assert msg["content"][1]["type"] == "text"
        assert msg["content"][1]["text"] == "The answer is 42."

    def test_function_call(self, ops):
        steps = [
            {
                "type": "function_call",
                "id": "call_123",
                "name": "get_weather",
                "arguments": {"location": "SF"},
            }
        ]
        messages = ops.p_steps_to_ir_messages(steps)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        assert messages[0]["content"][0]["type"] == "tool_call"
        assert messages[0]["content"][0]["tool_name"] == "get_weather"

    def test_function_result(self, ops):
        steps = [
            {
                "type": "function_result",
                "call_id": "call_123",
                "result": "sunny",
            }
        ]
        messages = ops.p_steps_to_ir_messages(steps)
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        assert messages[0]["content"][0]["type"] == "tool_result"
        assert messages[0]["content"][0]["tool_call_id"] == "call_123"

    def test_tool_use_flow(self, ops):
        """Full tool use flow: model_output + function_call + function_result + model_output."""
        steps = [
            {
                "type": "user_input",
                "content": [{"type": "text", "text": "What's the weather?"}],
            },
            {
                "type": "function_call",
                "id": "call_1",
                "name": "get_weather",
                "arguments": {"location": "SF"},
            },
            {"type": "function_result", "call_id": "call_1", "result": "sunny"},
            {
                "type": "model_output",
                "content": [{"type": "text", "text": "It's sunny in SF."}],
            },
        ]
        messages = ops.p_steps_to_ir_messages(steps)
        assert len(messages) == 4
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"  # function_call
        assert messages[1]["content"][0]["type"] == "tool_call"
        assert messages[2]["role"] == "tool"  # function_result
        assert messages[3]["role"] == "assistant"  # model_output

    def test_unknown_step_type_skipped(self, ops):
        steps = [
            {"type": "unknown_step", "data": "whatever"},
            {"type": "model_output", "content": [{"type": "text", "text": "ok"}]},
        ]
        messages = ops.p_steps_to_ir_messages(steps)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"


class TestIRMessagesToSteps:
    def test_user_message(self, ops):
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        steps = ops.ir_messages_to_p_steps(messages)
        assert len(steps) == 1
        assert steps[0]["type"] == "user_input"
        assert steps[0]["content"][0]["text"] == "Hello"

    def test_assistant_text(self, ops):
        messages = [{"role": "assistant", "content": [{"type": "text", "text": "Hi"}]}]
        steps = ops.ir_messages_to_p_steps(messages)
        assert len(steps) == 1
        assert steps[0]["type"] == "model_output"

    def test_assistant_with_reasoning_and_text(self, ops):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "reasoning",
                        "reasoning": "Thinking...",
                        "signature": "sig",
                    },
                    {"type": "text", "text": "Answer"},
                ],
            }
        ]
        steps = ops.ir_messages_to_p_steps(messages)
        assert len(steps) == 2
        assert steps[0]["type"] == "thought"
        assert steps[0]["signature"] == "sig"
        assert steps[1]["type"] == "model_output"
        assert steps[1]["content"][0]["text"] == "Answer"

    def test_assistant_with_tool_call(self, ops):
        messages = [
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "c1",
                        "tool_name": "fn",
                        "tool_input": {"x": 1},
                    }
                ],
            }
        ]
        steps = ops.ir_messages_to_p_steps(messages)
        assert len(steps) == 1
        assert steps[0]["type"] == "function_call"
        assert steps[0]["id"] == "c1"

    def test_tool_message(self, ops):
        messages = [
            {
                "role": "tool",
                "content": [
                    {"type": "tool_result", "tool_call_id": "c1", "result": "ok"}
                ],
            }
        ]
        steps = ops.ir_messages_to_p_steps(messages)
        assert len(steps) == 1
        assert steps[0]["type"] == "function_result"
        assert steps[0]["call_id"] == "c1"

    def test_roundtrip_multi_turn(self, ops):
        """Steps → IR → Steps preserves structure."""
        original_steps = [
            {"type": "user_input", "content": [{"type": "text", "text": "Hello"}]},
            {"type": "model_output", "content": [{"type": "text", "text": "Hi"}]},
            {"type": "user_input", "content": [{"type": "text", "text": "Bye"}]},
        ]
        ir_messages = ops.p_steps_to_ir_messages(original_steps)
        back_steps = ops.ir_messages_to_p_steps(ir_messages)
        assert len(back_steps) == 3
        assert back_steps[0]["type"] == "user_input"
        assert back_steps[0]["content"][0]["text"] == "Hello"
        assert back_steps[1]["type"] == "model_output"
        assert back_steps[1]["content"][0]["text"] == "Hi"
        assert back_steps[2]["type"] == "user_input"
        assert back_steps[2]["content"][0]["text"] == "Bye"
