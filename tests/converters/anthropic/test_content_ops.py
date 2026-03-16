"""
Anthropic ContentOps unit tests.
"""

import pytest

from llm_rosetta.converters.anthropic.content_ops import AnthropicContentOps

# Note: ir_refusal_to_p returns dict | None; tests assert result is not None before subscript
from llm_rosetta.types.ir import FilePart, ImagePart, ReasoningPart, TextPart


class TestAnthropicContentOps:
    """Unit tests for AnthropicContentOps."""

    # ==================== Text ====================

    def test_ir_text_to_p(self):
        """Test IR TextPart → Anthropic text content block."""
        ir_text = TextPart(type="text", text="Hello, world!")
        result = AnthropicContentOps.ir_text_to_p(ir_text)
        assert result == {"type": "text", "text": "Hello, world!"}

    def test_p_text_to_ir_from_string(self):
        """Test Anthropic string → IR TextPart."""
        result = AnthropicContentOps.p_text_to_ir("Hello!")
        assert result["type"] == "text"
        assert result["text"] == "Hello!"

    def test_p_text_to_ir_from_dict(self):
        """Test Anthropic text dict → IR TextPart."""
        result = AnthropicContentOps.p_text_to_ir({"type": "text", "text": "Hi"})
        assert result["type"] == "text"
        assert result["text"] == "Hi"

    def test_p_text_to_ir_invalid(self):
        """Test p_text_to_ir raises on invalid input."""
        with pytest.raises(ValueError, match="Cannot convert"):
            AnthropicContentOps.p_text_to_ir(42)

    def test_text_round_trip(self):
        """Test text round-trip: IR → Provider → IR."""
        original = TextPart(type="text", text="Round trip test")
        provider = AnthropicContentOps.ir_text_to_p(original)
        restored = AnthropicContentOps.p_text_to_ir(provider)
        assert restored["text"] == original["text"]

    # ==================== Image ====================

    def test_ir_image_to_p_with_base64(self):
        """Test IR ImagePart with base64 → Anthropic image block."""
        ir_image = ImagePart(
            type="image",
            image_data={"data": "abc123", "media_type": "image/png"},
        )
        result = AnthropicContentOps.ir_image_to_p(ir_image)
        assert result["type"] == "image"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "image/png"
        assert result["source"]["data"] == "abc123"

    def test_ir_image_to_p_with_url(self):
        """Test IR ImagePart with URL → Anthropic image block."""
        ir_image = ImagePart(type="image", image_url="https://example.com/img.jpg")
        result = AnthropicContentOps.ir_image_to_p(ir_image)
        assert result["type"] == "image"
        assert result["source"]["type"] == "url"
        assert result["source"]["url"] == "https://example.com/img.jpg"

    def test_ir_image_to_p_no_data(self):
        """Test ir_image_to_p raises when no image data."""
        with pytest.raises(ValueError, match="image_url or image_data"):
            AnthropicContentOps.ir_image_to_p({"type": "image"})

    def test_p_image_to_ir_base64(self):
        """Test Anthropic base64 image → IR ImagePart."""
        provider = {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/png",
                "data": "base64data",
            },
        }
        result = AnthropicContentOps.p_image_to_ir(provider)
        assert result["type"] == "image"
        assert result["image_data"]["data"] == "base64data"
        assert result["image_data"]["media_type"] == "image/png"

    def test_p_image_to_ir_url(self):
        """Test Anthropic URL image → IR ImagePart."""
        provider = {
            "type": "image",
            "source": {"type": "url", "url": "https://example.com/img.jpg"},
        }
        result = AnthropicContentOps.p_image_to_ir(provider)
        assert result["type"] == "image"
        assert result["image_url"] == "https://example.com/img.jpg"

    def test_image_base64_round_trip(self):
        """Test image base64 round-trip."""
        original = ImagePart(
            type="image",
            image_data={"data": "base64data", "media_type": "image/png"},
        )
        provider = AnthropicContentOps.ir_image_to_p(original)
        restored = AnthropicContentOps.p_image_to_ir(provider)
        assert restored["image_data"]["data"] == "base64data"
        assert restored["image_data"]["media_type"] == "image/png"

    def test_image_url_round_trip(self):
        """Test image URL round-trip."""
        original = ImagePart(type="image", image_url="https://example.com/img.jpg")
        provider = AnthropicContentOps.ir_image_to_p(original)
        restored = AnthropicContentOps.p_image_to_ir(provider)
        assert restored["image_url"] == "https://example.com/img.jpg"

    # ==================== File ====================

    def test_ir_file_to_p_with_base64(self):
        """Test IR FilePart with base64 → Anthropic document block."""
        ir_file = FilePart(
            type="file",
            file_data={"data": "pdf_data", "media_type": "application/pdf"},
        )
        result = AnthropicContentOps.ir_file_to_p(ir_file)
        assert result["type"] == "document"
        assert result["source"]["type"] == "base64"
        assert result["source"]["media_type"] == "application/pdf"
        assert result["source"]["data"] == "pdf_data"

    def test_ir_file_to_p_with_url(self):
        """Test IR FilePart with URL → Anthropic document block."""
        ir_file = FilePart(type="file", file_url="https://example.com/doc.pdf")
        result = AnthropicContentOps.ir_file_to_p(ir_file)
        assert result["type"] == "document"
        assert result["source"]["type"] == "url"
        assert result["source"]["url"] == "https://example.com/doc.pdf"

    def test_ir_file_to_p_no_data(self):
        """Test ir_file_to_p raises when no file data."""
        with pytest.raises(ValueError, match="file_data or file_url"):
            AnthropicContentOps.ir_file_to_p({"type": "file"})

    def test_p_file_to_ir_base64(self):
        """Test Anthropic document base64 → IR FilePart."""
        provider = {
            "type": "document",
            "source": {
                "type": "base64",
                "media_type": "application/xml",
                "data": "xml_data",
            },
        }
        result = AnthropicContentOps.p_file_to_ir(provider)
        assert result["type"] == "file"
        assert result["file_data"]["media_type"] == "application/xml"
        assert result["file_data"]["data"] == "xml_data"

    def test_p_file_to_ir_url(self):
        """Test Anthropic document URL → IR FilePart."""
        provider = {
            "type": "document",
            "source": {"type": "url", "url": "https://example.com/doc.pdf"},
        }
        result = AnthropicContentOps.p_file_to_ir(provider)
        assert result["type"] == "file"
        assert result["file_url"] == "https://example.com/doc.pdf"

    def test_file_round_trip(self):
        """Test file round-trip."""
        original = FilePart(
            type="file",
            file_data={"data": "pdf_data", "media_type": "application/pdf"},
        )
        provider = AnthropicContentOps.ir_file_to_p(original)
        restored = AnthropicContentOps.p_file_to_ir(provider)
        assert restored["file_data"]["data"] == "pdf_data"
        assert restored["file_data"]["media_type"] == "application/pdf"

    # ==================== Audio (not supported) ====================

    def test_ir_audio_to_p_raises(self):
        """Test ir_audio_to_p raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="does not support audio"):
            AnthropicContentOps.ir_audio_to_p({"type": "audio", "audio_id": "a1"})

    def test_p_audio_to_ir_raises(self):
        """Test p_audio_to_ir raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="does not support audio"):
            AnthropicContentOps.p_audio_to_ir({})

    # ==================== Reasoning ====================

    def test_ir_reasoning_to_p(self):
        """Test IR ReasoningPart → Anthropic thinking block."""
        ir_reasoning = ReasoningPart(type="reasoning", reasoning="I need to think.")
        result = AnthropicContentOps.ir_reasoning_to_p(ir_reasoning)
        assert result["type"] == "thinking"
        assert result["thinking"] == "I need to think."

    def test_ir_reasoning_to_p_with_signature(self):
        """Test IR ReasoningPart with signature → Anthropic thinking block."""
        ir_reasoning = ReasoningPart(
            type="reasoning", reasoning="thinking...", signature="sig123"
        )
        result = AnthropicContentOps.ir_reasoning_to_p(ir_reasoning)
        assert result["type"] == "thinking"
        assert result["thinking"] == "thinking..."
        assert result["signature"] == "sig123"

    def test_p_reasoning_to_ir(self):
        """Test Anthropic thinking block → IR ReasoningPart."""
        provider = {"type": "thinking", "thinking": "I should use a tool."}
        result = AnthropicContentOps.p_reasoning_to_ir(provider)
        assert result["type"] == "reasoning"
        assert result["reasoning"] == "I should use a tool."

    def test_p_reasoning_to_ir_with_signature(self):
        """Test Anthropic thinking block with signature → IR ReasoningPart."""
        provider = {
            "type": "thinking",
            "thinking": "thinking...",
            "signature": "sig456",
        }
        result = AnthropicContentOps.p_reasoning_to_ir(provider)
        assert result["type"] == "reasoning"
        assert result["reasoning"] == "thinking..."
        assert result["signature"] == "sig456"

    def test_reasoning_round_trip(self):
        """Test reasoning round-trip."""
        original = ReasoningPart(type="reasoning", reasoning="deep thought")
        provider = AnthropicContentOps.ir_reasoning_to_p(original)
        restored = AnthropicContentOps.p_reasoning_to_ir(provider)
        assert restored["reasoning"] == original["reasoning"]

    # ==================== Refusal ====================

    def test_ir_refusal_to_p(self):
        """Test IR RefusalPart → Anthropic text block (with warning)."""
        with pytest.warns(UserWarning, match="does not have a dedicated refusal"):
            result = AnthropicContentOps.ir_refusal_to_p(
                {"type": "refusal", "refusal": "I cannot do that"}
            )
        assert result is not None
        assert result["type"] == "text"
        assert "I cannot do that" in result["text"]

    def test_p_refusal_to_ir(self):
        """Test refusal string → IR RefusalPart."""
        result = AnthropicContentOps.p_refusal_to_ir("I cannot do that")
        assert result["type"] == "refusal"
        assert result["refusal"] == "I cannot do that"

    # ==================== Citation ====================

    def test_ir_citation_to_p_returns_none(self):
        """Test ir_citation_to_p returns None with warning."""
        with pytest.warns(UserWarning, match="part of TextBlock"):
            result = AnthropicContentOps.ir_citation_to_p({"type": "citation"})
        assert result is None

    def test_p_citation_to_ir_char_location(self):
        """Test Anthropic char_location citation → IR CitationPart."""
        provider = {
            "type": "char_location",
            "cited_text": "some cited text",
        }
        result = AnthropicContentOps.p_citation_to_ir(provider)
        assert result["type"] == "citation"
        assert result["text_citation"]["cited_text"] == "some cited text"

    def test_p_citation_to_ir_url(self):
        """Test Anthropic url_citation → IR CitationPart."""
        provider = {
            "type": "url_citation",
            "url": "https://example.com",
            "title": "Example",
            "start_index": 0,
            "end_index": 10,
        }
        result = AnthropicContentOps.p_citation_to_ir(provider)
        assert result["type"] == "citation"
        assert result["url_citation"]["url"] == "https://example.com"

    def test_p_citation_to_ir_fallback(self):
        """Test unknown citation type → fallback CitationPart."""
        result = AnthropicContentOps.p_citation_to_ir({"type": "unknown"})
        assert result["type"] == "citation"
