"""
Tests for cache_hint preservation in Anthropic converter.

Covers the round-trip of Anthropic's block-level `cache_control` through
the IR pipeline via the `cache_hint` field.

Scope: PR 1 of #362 — preservation only, no auto-injection.
"""

from llm_rosetta.converters.anthropic import AnthropicConverter
from llm_rosetta.converters.anthropic.content_ops import AnthropicContentOps
from llm_rosetta.converters.anthropic.tool_ops import AnthropicToolOps
from llm_rosetta.types.ir import (
    FilePart,
    IRRequest,
    ImagePart,
    ReasoningPart,
    TextPart,
    ToolCallPart,
    ToolDefinition,
    ToolResultPart,
)


CACHE_CONTROL_EPHEMERAL = {"type": "ephemeral"}


# ============================================================================
# ContentOps — block-level round-trip
# ============================================================================


class TestTextCacheHint:
    """cache_control preservation on text blocks."""

    def test_p_text_to_ir_reads_cache_control(self):
        provider_text = {
            "type": "text",
            "text": "Hello",
            "cache_control": CACHE_CONTROL_EPHEMERAL,
        }
        ir = AnthropicContentOps.p_text_to_ir(provider_text)
        assert ir["cache_hint"] == CACHE_CONTROL_EPHEMERAL

    def test_p_text_to_ir_no_cache_control(self):
        provider_text = {"type": "text", "text": "Hello"}
        ir = AnthropicContentOps.p_text_to_ir(provider_text)
        assert "cache_hint" not in ir

    def test_ir_text_to_p_writes_cache_control(self):
        ir_text = TextPart(type="text", text="Hello")
        ir_text["cache_hint"] = CACHE_CONTROL_EPHEMERAL
        result = AnthropicContentOps.ir_text_to_p(ir_text)
        assert result["cache_control"] == CACHE_CONTROL_EPHEMERAL

    def test_ir_text_to_p_no_cache_hint(self):
        ir_text = TextPart(type="text", text="Hello")
        result = AnthropicContentOps.ir_text_to_p(ir_text)
        assert "cache_control" not in result

    def test_text_round_trip(self):
        provider_text = {
            "type": "text",
            "text": "Context",
            "cache_control": CACHE_CONTROL_EPHEMERAL,
        }
        ir = AnthropicContentOps.p_text_to_ir(provider_text)
        result = AnthropicContentOps.ir_text_to_p(ir)
        assert result["cache_control"] == CACHE_CONTROL_EPHEMERAL
        assert result["text"] == "Context"


class TestImageCacheHint:
    """cache_control preservation on image blocks."""

    def test_p_image_to_ir_reads_cache_control(self):
        provider_image = {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/img.png"},
            "cache_control": CACHE_CONTROL_EPHEMERAL,
        }
        ir = AnthropicContentOps.p_image_to_ir(provider_image)
        assert ir["cache_hint"] == CACHE_CONTROL_EPHEMERAL

    def test_p_image_to_ir_base64_with_cache_control(self):
        provider_image = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "abc123",
            },
            "cache_control": CACHE_CONTROL_EPHEMERAL,
        }
        ir = AnthropicContentOps.p_image_to_ir(provider_image)
        assert ir["cache_hint"] == CACHE_CONTROL_EPHEMERAL

    def test_ir_image_to_p_writes_cache_control(self):
        ir_image = ImagePart(type="image", image_url="https://example.com/img.png")
        ir_image["cache_hint"] = CACHE_CONTROL_EPHEMERAL
        result = AnthropicContentOps.ir_image_to_p(ir_image)
        assert result["cache_control"] == CACHE_CONTROL_EPHEMERAL

    def test_ir_image_to_p_base64_writes_cache_control(self):
        ir_image = ImagePart(
            type="image",
            image_data={"data": "abc123", "media_type": "image/png"},
        )
        ir_image["cache_hint"] = CACHE_CONTROL_EPHEMERAL
        result = AnthropicContentOps.ir_image_to_p(ir_image)
        assert result["cache_control"] == CACHE_CONTROL_EPHEMERAL

    def test_image_round_trip(self):
        provider_image = {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/img.png"},
            "cache_control": CACHE_CONTROL_EPHEMERAL,
        }
        ir = AnthropicContentOps.p_image_to_ir(provider_image)
        result = AnthropicContentOps.ir_image_to_p(ir)
        assert result["cache_control"] == CACHE_CONTROL_EPHEMERAL


class TestFileCacheHint:
    """cache_control preservation on file/document blocks."""

    def test_p_file_to_ir_reads_cache_control(self):
        provider_file = {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "abc123",
            },
            "cache_control": CACHE_CONTROL_EPHEMERAL,
        }
        ir = AnthropicContentOps.p_file_to_ir(provider_file)
        assert ir["cache_hint"] == CACHE_CONTROL_EPHEMERAL

    def test_ir_file_to_p_writes_cache_control(self):
        ir_file = FilePart(
            type="file",
            file_data={"data": "abc123", "media_type": "application/pdf"},
        )
        ir_file["cache_hint"] = CACHE_CONTROL_EPHEMERAL
        result = AnthropicContentOps.ir_file_to_p(ir_file)
        assert result["cache_control"] == CACHE_CONTROL_EPHEMERAL

    def test_file_round_trip(self):
        provider_file = {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/pdf",
                "data": "abc123",
            },
            "cache_control": CACHE_CONTROL_EPHEMERAL,
        }
        ir = AnthropicContentOps.p_file_to_ir(provider_file)
        result = AnthropicContentOps.ir_file_to_p(ir)
        assert result["cache_control"] == CACHE_CONTROL_EPHEMERAL


class TestReasoningCacheHint:
    """cache_control preservation on thinking blocks."""

    def test_p_reasoning_to_ir_reads_cache_control(self):
        provider_reasoning = {
            "type": "thinking",
            "thinking": "Let me think...",
            "signature": "sig123",
            "cache_control": CACHE_CONTROL_EPHEMERAL,
        }
        ir = AnthropicContentOps.p_reasoning_to_ir(provider_reasoning)
        assert ir["cache_hint"] == CACHE_CONTROL_EPHEMERAL

    def test_ir_reasoning_to_p_writes_cache_control(self):
        ir_reasoning = ReasoningPart(
            type="reasoning", reasoning="Let me think...", signature="sig123"
        )
        ir_reasoning["cache_hint"] = CACHE_CONTROL_EPHEMERAL
        result = AnthropicContentOps.ir_reasoning_to_p(ir_reasoning)
        assert result["cache_control"] == CACHE_CONTROL_EPHEMERAL

    def test_reasoning_round_trip(self):
        provider_reasoning = {
            "type": "thinking",
            "thinking": "Let me think...",
            "signature": "sig123",
            "cache_control": CACHE_CONTROL_EPHEMERAL,
        }
        ir = AnthropicContentOps.p_reasoning_to_ir(provider_reasoning)
        result = AnthropicContentOps.ir_reasoning_to_p(ir)
        assert result["cache_control"] == CACHE_CONTROL_EPHEMERAL


# ============================================================================
# ToolOps — tool definition and tool call/result round-trip
# ============================================================================


class TestToolDefinitionCacheHint:
    """cache_control preservation on tool definitions."""

    def test_p_tool_definition_to_ir_reads_cache_control(self):
        provider_tool = {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {}},
            "cache_control": CACHE_CONTROL_EPHEMERAL,
        }
        ir = AnthropicToolOps.p_tool_definition_to_ir(provider_tool)
        assert ir["cache_hint"] == CACHE_CONTROL_EPHEMERAL

    def test_p_tool_definition_to_ir_no_cache_control(self):
        provider_tool = {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {}},
        }
        ir = AnthropicToolOps.p_tool_definition_to_ir(provider_tool)
        assert "cache_hint" not in ir

    def test_ir_tool_definition_to_p_writes_cache_control(self):
        ir_tool: ToolDefinition = {
            "type": "function",
            "name": "get_weather",
            "description": "Get weather",
            "parameters": {"type": "object", "properties": {}},
            "cache_hint": CACHE_CONTROL_EPHEMERAL,
        }
        result = AnthropicToolOps.ir_tool_definition_to_p(ir_tool)
        assert result["cache_control"] == CACHE_CONTROL_EPHEMERAL

    def test_tool_definition_round_trip(self):
        provider_tool = {
            "name": "get_weather",
            "description": "Get weather",
            "input_schema": {"type": "object", "properties": {}},
            "cache_control": CACHE_CONTROL_EPHEMERAL,
        }
        ir = AnthropicToolOps.p_tool_definition_to_ir(provider_tool)
        result = AnthropicToolOps.ir_tool_definition_to_p(ir)
        assert result["cache_control"] == CACHE_CONTROL_EPHEMERAL
        assert result["name"] == "get_weather"


class TestToolCallCacheHint:
    """cache_control preservation on tool_use blocks."""

    def test_p_tool_call_to_ir_reads_cache_control(self):
        provider_tool_call = {
            "type": "tool_use",
            "id": "call_1",
            "name": "get_weather",
            "input": {"location": "Tokyo"},
            "cache_control": CACHE_CONTROL_EPHEMERAL,
        }
        ir = AnthropicToolOps.p_tool_call_to_ir(provider_tool_call)
        assert ir["cache_hint"] == CACHE_CONTROL_EPHEMERAL

    def test_ir_tool_call_to_p_writes_cache_control(self):
        ir_tool_call = ToolCallPart(
            type="tool_call",
            tool_call_id="call_1",
            tool_name="get_weather",
            tool_input={"location": "Tokyo"},
        )
        ir_tool_call["cache_hint"] = CACHE_CONTROL_EPHEMERAL
        result = AnthropicToolOps.ir_tool_call_to_p(ir_tool_call)
        assert result["cache_control"] == CACHE_CONTROL_EPHEMERAL

    def test_tool_call_round_trip(self):
        provider_tool_call = {
            "type": "tool_use",
            "id": "call_1",
            "name": "get_weather",
            "input": {"location": "Tokyo"},
            "cache_control": CACHE_CONTROL_EPHEMERAL,
        }
        ir = AnthropicToolOps.p_tool_call_to_ir(provider_tool_call)
        result = AnthropicToolOps.ir_tool_call_to_p(ir)
        assert result["cache_control"] == CACHE_CONTROL_EPHEMERAL


class TestToolResultCacheHint:
    """cache_control preservation on tool_result blocks."""

    def test_p_tool_result_to_ir_reads_cache_control(self):
        provider_tool_result = {
            "type": "tool_result",
            "tool_use_id": "call_1",
            "content": "72°F",
            "cache_control": CACHE_CONTROL_EPHEMERAL,
        }
        ir = AnthropicToolOps.p_tool_result_to_ir(provider_tool_result)
        assert ir["cache_hint"] == CACHE_CONTROL_EPHEMERAL

    def test_ir_tool_result_to_p_writes_cache_control(self):
        ir_tool_result = ToolResultPart(
            type="tool_result",
            tool_call_id="call_1",
            result="72°F",
        )
        ir_tool_result["cache_hint"] = CACHE_CONTROL_EPHEMERAL
        result = AnthropicToolOps.ir_tool_result_to_p(ir_tool_result)
        assert result["cache_control"] == CACHE_CONTROL_EPHEMERAL

    def test_tool_result_round_trip(self):
        provider_tool_result = {
            "type": "tool_result",
            "tool_use_id": "call_1",
            "content": "72°F",
            "cache_control": CACHE_CONTROL_EPHEMERAL,
        }
        ir = AnthropicToolOps.p_tool_result_to_ir(provider_tool_result)
        result = AnthropicToolOps.ir_tool_result_to_p(ir)
        assert result["cache_control"] == CACHE_CONTROL_EPHEMERAL


# ============================================================================
# Full request round-trip via AnthropicConverter
# ============================================================================


class TestConverterCacheHintRoundTrip:
    """End-to-end cache_hint preservation through AnthropicConverter."""

    def setup_method(self):
        self.converter = AnthropicConverter()

    def test_message_content_cache_control_round_trip(self):
        """cache_control on message content blocks survives request round-trip."""
        provider_request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Long context...",
                            "cache_control": CACHE_CONTROL_EPHEMERAL,
                        },
                        {"type": "text", "text": "Question?"},
                    ],
                }
            ],
        }
        ir = self.converter.request_from_provider(provider_request)
        result, warnings = self.converter.request_to_provider(ir)

        user_msg = result["messages"][0]
        assert user_msg["content"][0]["cache_control"] == CACHE_CONTROL_EPHEMERAL
        assert "cache_control" not in user_msg["content"][1]

    def test_tool_definition_cache_control_round_trip(self):
        """cache_control on tool definitions survives request round-trip."""
        provider_request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "What's the weather?"}],
            "tools": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                    },
                    "cache_control": CACHE_CONTROL_EPHEMERAL,
                },
                {
                    "name": "search",
                    "description": "Search",
                    "input_schema": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                    },
                },
            ],
        }
        ir = self.converter.request_from_provider(provider_request)
        result, warnings = self.converter.request_to_provider(ir)

        assert result["tools"][0]["cache_control"] == CACHE_CONTROL_EPHEMERAL
        assert "cache_control" not in result["tools"][1]

    def test_system_blocks_cache_control_round_trip(self):
        """cache_control on system content blocks survives request round-trip."""
        provider_request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": [
                {
                    "type": "text",
                    "text": "You are a helpful assistant.",
                    "cache_control": CACHE_CONTROL_EPHEMERAL,
                },
                {"type": "text", "text": "Additional context."},
            ],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        ir = self.converter.request_from_provider(provider_request)
        result, warnings = self.converter.request_to_provider(ir)

        # System should be structured blocks (not a joined string)
        assert isinstance(result["system"], list)
        assert result["system"][0]["cache_control"] == CACHE_CONTROL_EPHEMERAL
        assert "cache_control" not in result["system"][1]

    def test_system_string_round_trip(self):
        """Plain string system instruction round-trips as structured blocks."""
        provider_request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": "Hello"}],
        }
        ir = self.converter.request_from_provider(provider_request)
        assert ir["system_instruction"] == [
            {"type": "text", "text": "You are helpful."}
        ]
        result, warnings = self.converter.request_to_provider(ir)
        assert isinstance(result["system"], list)
        assert result["system"][0]["text"] == "You are helpful."
        assert "cache_control" not in result["system"][0]

    def test_system_blocks_without_cache_control(self):
        """System blocks without cache_control become list[TextPart] in IR."""
        provider_request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": [
                {"type": "text", "text": "You are helpful."},
                {"type": "text", "text": "Be concise."},
            ],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        ir = self.converter.request_from_provider(provider_request)
        assert len(ir["system_instruction"]) == 2
        assert ir["system_instruction"][0]["text"] == "You are helpful."
        assert ir["system_instruction"][1]["text"] == "Be concise."
        assert "cache_hint" not in ir["system_instruction"][0]

    def test_no_cache_control_no_change(self):
        """Requests without cache_control are unaffected (no regression)."""
        provider_request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Hello"},
                    ],
                }
            ],
        }
        ir = self.converter.request_from_provider(provider_request)
        result, warnings = self.converter.request_to_provider(ir)

        user_msg = result["messages"][0]
        assert "cache_control" not in user_msg["content"][0]

    def test_assistant_tool_use_cache_control_round_trip(self):
        """cache_control on tool_use blocks in assistant messages survives."""
        provider_request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "call_1",
                            "name": "get_weather",
                            "input": {"location": "Tokyo"},
                            "cache_control": CACHE_CONTROL_EPHEMERAL,
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "call_1",
                            "content": "72°F",
                            "cache_control": CACHE_CONTROL_EPHEMERAL,
                        }
                    ],
                },
            ],
        }
        ir = self.converter.request_from_provider(provider_request)
        result, warnings = self.converter.request_to_provider(ir)

        # Assistant message with tool_use
        assistant_msg = result["messages"][1]
        assert assistant_msg["content"][0]["cache_control"] == CACHE_CONTROL_EPHEMERAL

        # Tool result (now in a user message in Anthropic format)
        tool_result_msg = result["messages"][2]
        assert tool_result_msg["content"][0]["cache_control"] == CACHE_CONTROL_EPHEMERAL

    def test_mixed_cache_control_preserved_selectively(self):
        """Only blocks with cache_control get it on output; others stay clean."""
        provider_request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": [
                {
                    "type": "text",
                    "text": "System prompt part 1",
                    "cache_control": CACHE_CONTROL_EPHEMERAL,
                },
                {"type": "text", "text": "System prompt part 2"},
            ],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Long document...",
                            "cache_control": CACHE_CONTROL_EPHEMERAL,
                        },
                        {"type": "text", "text": "Short question"},
                    ],
                }
            ],
            "tools": [
                {
                    "name": "tool_a",
                    "description": "Tool A",
                    "input_schema": {"type": "object", "properties": {}},
                    "cache_control": CACHE_CONTROL_EPHEMERAL,
                },
                {
                    "name": "tool_b",
                    "description": "Tool B",
                    "input_schema": {"type": "object", "properties": {}},
                },
            ],
        }
        ir = self.converter.request_from_provider(provider_request)
        result, warnings = self.converter.request_to_provider(ir)

        # System blocks
        assert isinstance(result["system"], list)
        assert result["system"][0]["cache_control"] == CACHE_CONTROL_EPHEMERAL
        assert "cache_control" not in result["system"][1]

        # User message blocks
        user_content = result["messages"][0]["content"]
        assert user_content[0]["cache_control"] == CACHE_CONTROL_EPHEMERAL
        assert "cache_control" not in user_content[1]

        # Tool definitions
        assert result["tools"][0]["cache_control"] == CACHE_CONTROL_EPHEMERAL
        assert "cache_control" not in result["tools"][1]


# ============================================================================
# Cross-format: cache_hint on IR ignored by non-Anthropic output
# ============================================================================


class TestCacheHintIgnoredByOtherConverters:
    """cache_hint on IR parts is harmless for non-Anthropic converters."""

    def test_openai_ignores_cache_hint_on_text(self):
        """OpenAI converter output does not include cache_control."""
        from llm_rosetta.converters.openai_chat import OpenAIChatConverter

        converter = OpenAIChatConverter()
        # Create an IR request with cache_hint on text
        ir_request: IRRequest = {
            "model": "gpt-4",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Hello",
                            "cache_hint": CACHE_CONTROL_EPHEMERAL,
                        }
                    ],
                }
            ],
        }
        result, warnings = converter.request_to_provider(ir_request)
        # OpenAI output should not contain cache_control
        messages = result.get("messages", [])
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for part in content:
                    if isinstance(part, dict):
                        assert "cache_control" not in part


class TestCrossFormatSystemCacheHint:
    """System blocks with cache_control are flattened for non-Anthropic output."""

    def test_anthropic_system_cache_to_openai_flattens(self):
        """Anthropic system with cache_control → IR → OpenAI: string, no cache_control."""
        from llm_rosetta.converters.openai_chat import OpenAIChatConverter

        anthropic_converter = AnthropicConverter()
        openai_converter = OpenAIChatConverter()

        provider_request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": [
                {
                    "type": "text",
                    "text": "You are helpful.",
                    "cache_control": CACHE_CONTROL_EPHEMERAL,
                },
                {"type": "text", "text": "Be concise."},
            ],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        ir = anthropic_converter.request_from_provider(provider_request)
        result, warnings = openai_converter.request_to_provider(ir)

        # System should be a structured content array (no cache_control)
        system_msgs = [m for m in result["messages"] if m["role"] == "system"]
        assert len(system_msgs) == 1
        content = system_msgs[0]["content"]
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["text"] == "You are helpful."
        assert content[1]["text"] == "Be concise."
        assert "cache_control" not in str(result)

    def test_anthropic_system_cache_to_google_flattens(self):
        """Anthropic system with cache_control → IR → Google: string, no cache_control."""
        from llm_rosetta.converters.google_genai import GoogleGenAIConverter

        anthropic_converter = AnthropicConverter()
        google_converter = GoogleGenAIConverter()

        provider_request = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1024,
            "system": [
                {
                    "type": "text",
                    "text": "You are helpful.",
                    "cache_control": CACHE_CONTROL_EPHEMERAL,
                },
            ],
            "messages": [{"role": "user", "content": "Hello"}],
        }
        ir = anthropic_converter.request_from_provider(provider_request)
        result, warnings = google_converter.request_to_provider(ir)

        # Google system_instruction should be structured but without cache_control
        system_inst = result.get("system_instruction")
        assert system_inst is not None
        # Should contain the text, flattened
        system_text = str(system_inst)
        assert "helpful" in system_text
        assert "cache_control" not in str(result)
