"""
Tests for the shared tool content conversion helpers.
"""

from llm_rosetta.converters.base.tool_content import (
    convert_content_blocks_to_ir,
    convert_ir_content_blocks_to_p,
)
from llm_rosetta.converters.anthropic.content_ops import AnthropicContentOps
from llm_rosetta.converters.openai_responses.content_ops import (
    OpenAIResponsesContentOps,
)


class TestConvertContentBlocksToIR:
    """Tests for convert_content_blocks_to_ir."""

    def test_text_block(self):
        """Test provider text block → IR TextPart."""
        blocks = [{"type": "text", "text": "hello"}]
        result = convert_content_blocks_to_ir(blocks, AnthropicContentOps)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "hello"

    def test_input_text_block(self):
        """Test OpenAI input_text → IR TextPart."""
        blocks = [{"type": "input_text", "text": "hello"}]
        result = convert_content_blocks_to_ir(blocks, OpenAIResponsesContentOps)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "hello"

    def test_image_block_anthropic(self):
        """Test Anthropic image block → IR ImagePart."""
        blocks = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "AAAA",
                },
            }
        ]
        result = convert_content_blocks_to_ir(blocks, AnthropicContentOps)
        assert len(result) == 1
        assert result[0]["type"] == "image"
        assert result[0]["image_data"]["data"] == "AAAA"

    def test_input_image_block_openai(self):
        """Test OpenAI input_image → IR ImagePart."""
        blocks = [
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,BBBB",
            }
        ]
        result = convert_content_blocks_to_ir(blocks, OpenAIResponsesContentOps)
        assert len(result) == 1
        assert result[0]["type"] == "image"
        assert result[0]["image_data"]["data"] == "BBBB"
        assert result[0]["image_data"]["media_type"] == "image/png"

    def test_string_block(self):
        """Test plain string block → IR TextPart."""
        blocks = ["just a string"]
        result = convert_content_blocks_to_ir(blocks, AnthropicContentOps)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "just a string"

    def test_unknown_type_passthrough(self):
        """Test unknown block type passes through unchanged."""
        blocks = [{"type": "custom_thing", "data": "xyz"}]
        result = convert_content_blocks_to_ir(blocks, AnthropicContentOps)
        assert len(result) == 1
        assert result[0] == {"type": "custom_thing", "data": "xyz"}

    def test_mixed_blocks(self):
        """Test mixed text + image blocks."""
        blocks = [
            {"type": "text", "text": "Here is the chart:"},
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": "CCCC",
                },
            },
        ]
        result = convert_content_blocks_to_ir(blocks, AnthropicContentOps)
        assert len(result) == 2
        assert result[0]["type"] == "text"
        assert result[1]["type"] == "image"

    def test_empty_list(self):
        """Test empty list returns empty list."""
        result = convert_content_blocks_to_ir([], AnthropicContentOps)
        assert result == []


class TestConvertIRContentBlocksToP:
    """Tests for convert_ir_content_blocks_to_p."""

    def test_text_to_anthropic(self):
        """Test IR TextPart → Anthropic text block."""
        blocks = [{"type": "text", "text": "hello"}]
        result = convert_ir_content_blocks_to_p(blocks, AnthropicContentOps)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "hello"

    def test_text_to_openai_responses(self):
        """Test IR TextPart → OpenAI input_text block."""
        blocks = [{"type": "text", "text": "hello"}]
        result = convert_ir_content_blocks_to_p(blocks, OpenAIResponsesContentOps)
        assert len(result) == 1
        assert result[0]["type"] == "input_text"
        assert result[0]["text"] == "hello"

    def test_image_to_anthropic(self):
        """Test IR ImagePart → Anthropic image block."""
        blocks = [
            {
                "type": "image",
                "image_data": {"data": "DDDD", "media_type": "image/png"},
            }
        ]
        result = convert_ir_content_blocks_to_p(blocks, AnthropicContentOps)
        assert len(result) == 1
        assert result[0]["type"] == "image"
        assert result[0]["source"]["type"] == "base64"
        assert result[0]["source"]["data"] == "DDDD"

    def test_image_to_openai_responses(self):
        """Test IR ImagePart → OpenAI input_image block."""
        blocks = [
            {
                "type": "image",
                "image_data": {"data": "EEEE", "media_type": "image/jpeg"},
            }
        ]
        result = convert_ir_content_blocks_to_p(blocks, OpenAIResponsesContentOps)
        assert len(result) == 1
        assert result[0]["type"] == "input_image"
        assert "base64,EEEE" in result[0]["image_url"]

    def test_string_block(self):
        """Test plain string → provider text block."""
        blocks = ["a string"]
        result = convert_ir_content_blocks_to_p(blocks, AnthropicContentOps)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert result[0]["text"] == "a string"

    def test_unknown_type_passthrough(self):
        """Test unknown IR block type passes through unchanged."""
        blocks = [{"type": "audio", "data": "xyz"}]
        result = convert_ir_content_blocks_to_p(blocks, AnthropicContentOps)
        assert len(result) == 1
        assert result[0] == {"type": "audio", "data": "xyz"}

    def test_empty_list(self):
        """Test empty list returns empty list."""
        result = convert_ir_content_blocks_to_p([], AnthropicContentOps)
        assert result == []


class TestCrossProviderRoundTrip:
    """End-to-end cross-provider tool result content conversion tests.

    These test the exact bug path: OpenAI Responses input_image → IR → Anthropic image.
    """

    def test_openai_responses_input_image_to_anthropic(self):
        """The P0 bug path: input_image from OpenAI → IR → Anthropic image block."""
        # Step 1: OpenAI Responses provider blocks → IR
        openai_blocks = [
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,iVBOR",
            }
        ]
        ir_blocks = convert_content_blocks_to_ir(
            openai_blocks, OpenAIResponsesContentOps
        )
        assert len(ir_blocks) == 1
        assert ir_blocks[0]["type"] == "image"

        # Step 2: IR → Anthropic provider blocks
        anthropic_blocks = convert_ir_content_blocks_to_p(
            ir_blocks, AnthropicContentOps
        )
        assert len(anthropic_blocks) == 1
        assert anthropic_blocks[0]["type"] == "image"
        assert anthropic_blocks[0]["source"]["type"] == "base64"
        assert anthropic_blocks[0]["source"]["data"] == "iVBOR"
        assert anthropic_blocks[0]["source"]["media_type"] == "image/png"

    def test_anthropic_image_to_openai_responses(self):
        """Reverse path: Anthropic image → IR → OpenAI input_image."""
        anthropic_blocks = [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": "FFFF",
                },
            }
        ]
        ir_blocks = convert_content_blocks_to_ir(anthropic_blocks, AnthropicContentOps)
        assert ir_blocks[0]["type"] == "image"

        openai_blocks = convert_ir_content_blocks_to_p(
            ir_blocks, OpenAIResponsesContentOps
        )
        assert openai_blocks[0]["type"] == "input_image"
        assert "base64,FFFF" in openai_blocks[0]["image_url"]

    def test_mixed_text_and_image_cross_provider(self):
        """Mixed text + image from OpenAI → IR → Anthropic."""
        openai_blocks = [
            {"type": "input_text", "text": "Here is the chart:"},
            {
                "type": "input_image",
                "image_url": "data:image/png;base64,CHART",
            },
        ]
        ir_blocks = convert_content_blocks_to_ir(
            openai_blocks, OpenAIResponsesContentOps
        )
        assert len(ir_blocks) == 2

        anthropic_blocks = convert_ir_content_blocks_to_p(
            ir_blocks, AnthropicContentOps
        )
        assert len(anthropic_blocks) == 2
        assert anthropic_blocks[0]["type"] == "text"
        assert anthropic_blocks[0]["text"] == "Here is the chart:"
        assert anthropic_blocks[1]["type"] == "image"
        assert anthropic_blocks[1]["source"]["data"] == "CHART"
