"""Tests for Anthropic TypedDict type definitions.

This module tests that the TypedDict replicas correctly represent
the Anthropic SDK types and can be used for type checking.
"""

from llmir.types.anthropic import (
    # Request types
    MessageCreateParams,
    MessageParam,
    ToolParam,
    ToolChoiceParam,
    ThinkingConfigParam,
    # Response types
    Message,
    TextBlock,
    ThinkingBlock,
    ToolUseBlock,
    Usage,
    StopReason,
)


class TestRequestTypes:
    """Test request type definitions."""

    def test_message_param_structure(self):
        """Test MessageParam has correct structure."""
        msg: MessageParam = {"role": "user", "content": "Hello, Claude!"}
        assert msg["role"] in ["user", "assistant"]
        assert isinstance(msg["content"], str)

    def test_message_param_with_blocks(self):
        """Test MessageParam with content blocks."""
        msg: MessageParam = {
            "role": "user",
            "content": [{"type": "text", "text": "Hello!"}],
        }
        assert msg["role"] == "user"
        assert isinstance(msg["content"], list)

    def test_tool_param_structure(self):
        """Test ToolParam has correct structure."""
        tool: ToolParam = {
            "name": "get_weather",
            "description": "Get weather information",
            "input_schema": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
        assert tool["name"] == "get_weather"
        assert "input_schema" in tool

    def test_message_create_params_minimal(self):
        """Test minimal MessageCreateParams."""
        params: MessageCreateParams = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "Hello!"}],
        }
        assert params["model"] == "claude-3-5-sonnet-20241022"
        assert params["max_tokens"] == 1024
        assert len(params["messages"]) == 1


class TestResponseTypes:
    """Test response type definitions."""

    def test_text_block_structure(self):
        """Test TextBlock has correct structure."""
        block: TextBlock = {"type": "text", "text": "Hello, world!"}
        assert block["type"] == "text"
        assert block["text"] == "Hello, world!"

    def test_thinking_block_structure(self):
        """Test ThinkingBlock has correct structure."""
        block: ThinkingBlock = {
            "type": "thinking",
            "signature": "sig_123",
            "thinking": "Let me think about this...",
        }
        assert block["type"] == "thinking"
        assert "signature" in block
        assert "thinking" in block

    def test_tool_use_block_structure(self):
        """Test ToolUseBlock has correct structure."""
        block: ToolUseBlock = {
            "type": "tool_use",
            "id": "toolu_123",
            "name": "get_weather",
            "input": {"location": "San Francisco"},
        }
        assert block["type"] == "tool_use"
        assert block["id"] == "toolu_123"
        assert block["name"] == "get_weather"
        assert isinstance(block["input"], dict)

    def test_usage_structure(self):
        """Test Usage has correct structure."""
        usage: Usage = {"input_tokens": 100, "output_tokens": 50}
        assert usage["input_tokens"] == 100
        assert usage["output_tokens"] == 50

    def test_message_structure(self):
        """Test Message has correct structure."""
        message: Message = {
            "id": "msg_123",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": "Hello!"}],
            "model": "claude-3-5-sonnet-20241022",
            "usage": {"input_tokens": 10, "output_tokens": 5},
        }
        assert message["id"] == "msg_123"
        assert message["type"] == "message"
        assert message["role"] == "assistant"
        assert len(message["content"]) == 1
        assert message["usage"]["input_tokens"] == 10

    def test_stop_reason_values(self):
        """Test StopReason has all expected values."""
        valid_reasons: list[StopReason] = [
            "end_turn",
            "max_tokens",
            "stop_sequence",
            "tool_use",
            "pause_turn",
            "refusal",
        ]
        # Just verify these are valid type annotations
        for reason in valid_reasons:
            assert reason in [
                "end_turn",
                "max_tokens",
                "stop_sequence",
                "tool_use",
                "pause_turn",
                "refusal",
            ]


class TestToolChoiceTypes:
    """Test tool choice type definitions."""

    def test_tool_choice_auto(self):
        """Test auto tool choice."""
        choice: ToolChoiceParam = {"type": "auto"}
        assert choice["type"] == "auto"

    def test_tool_choice_any(self):
        """Test any tool choice."""
        choice: ToolChoiceParam = {"type": "any"}
        assert choice["type"] == "any"

    def test_tool_choice_tool(self):
        """Test specific tool choice."""
        choice: ToolChoiceParam = {"type": "tool", "name": "get_weather"}
        assert choice["type"] == "tool"
        assert choice["name"] == "get_weather"

    def test_tool_choice_none(self):
        """Test none tool choice."""
        choice: ToolChoiceParam = {"type": "none"}
        assert choice["type"] == "none"


class TestThinkingConfigTypes:
    """Test thinking configuration type definitions."""

    def test_thinking_enabled(self):
        """Test enabled thinking config."""
        config: ThinkingConfigParam = {"type": "enabled", "budget_tokens": 2048}
        assert config["type"] == "enabled"
        assert config["budget_tokens"] == 2048

    def test_thinking_disabled(self):
        """Test disabled thinking config."""
        config: ThinkingConfigParam = {"type": "disabled"}
        assert config["type"] == "disabled"


class TestTypeImports:
    """Test that all types can be imported."""

    def test_all_request_types_importable(self):
        """Test all request types can be imported."""
        # If we get here, imports succeeded
        assert True

    def test_all_response_types_importable(self):
        """Test all response types can be imported."""
        # If we get here, imports succeeded
        assert True
