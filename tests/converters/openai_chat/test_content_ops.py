"""
OpenAI Chat ContentOps unit tests.
"""


import pytest

from llm_rosetta.converters.openai_chat.content_ops import OpenAIChatContentOps
from typing import cast

from llm_rosetta.types.ir import CitationPart, ImagePart, TextPart


class TestOpenAIChatContentOps:
    """Unit tests for OpenAIChatContentOps."""

    # ==================== Text ====================

    def test_ir_text_to_p(self):
        """Test IR TextPart → OpenAI text content."""
        ir_text = TextPart(type="text", text="Hello, world!")
        result = OpenAIChatContentOps.ir_text_to_p(ir_text)
        assert result == {"type": "text", "text": "Hello, world!"}

    def test_p_text_to_ir_from_string(self):
        """Test OpenAI string → IR TextPart."""
        result = OpenAIChatContentOps.p_text_to_ir("Hello!")
        assert result["type"] == "text"
        assert result["text"] == "Hello!"

    def test_p_text_to_ir_from_dict(self):
        """Test OpenAI text dict → IR TextPart."""
        result = OpenAIChatContentOps.p_text_to_ir({"type": "text", "text": "Hi"})
        assert result["type"] == "text"
        assert result["text"] == "Hi"

    def test_p_text_to_ir_invalid(self):
        """Test p_text_to_ir raises on invalid input."""
        with pytest.raises(ValueError, match="Cannot convert"):
            OpenAIChatContentOps.p_text_to_ir(42)

    def test_text_round_trip(self):
        """Test text round-trip: IR → Provider → IR."""
        original = TextPart(type="text", text="Round trip test")
        provider = OpenAIChatContentOps.ir_text_to_p(original)
        restored = OpenAIChatContentOps.p_text_to_ir(provider)
        assert restored["text"] == original["text"]

    # ==================== Image ====================

    def test_ir_image_to_p_with_url(self):
        """Test IR ImagePart with URL → OpenAI image_url."""
        ir_image = ImagePart(
            type="image", image_url="https://example.com/img.jpg", detail="high"
        )
        result = OpenAIChatContentOps.ir_image_to_p(ir_image)
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == "https://example.com/img.jpg"
        assert result["image_url"]["detail"] == "high"

    def test_ir_image_to_p_with_base64(self):
        """Test IR ImagePart with base64 → OpenAI data URI."""
        ir_image = ImagePart(
            type="image",
            image_data={"data": "abc123", "media_type": "image/png"},
        )
        result = OpenAIChatContentOps.ir_image_to_p(ir_image)
        assert result["type"] == "image_url"
        assert result["image_url"]["url"] == "data:image/png;base64,abc123"

    def test_ir_image_to_p_no_data(self):
        """Test ir_image_to_p raises when no image data."""
        with pytest.raises(ValueError, match="image_url or image_data"):
            OpenAIChatContentOps.ir_image_to_p({"type": "image"})

    def test_p_image_to_ir_with_url(self):
        """Test OpenAI image_url with URL → IR ImagePart."""
        provider = {
            "type": "image_url",
            "image_url": {"url": "https://example.com/img.jpg", "detail": "low"},
        }
        result = OpenAIChatContentOps.p_image_to_ir(provider)
        assert result["type"] == "image"
        assert result["image_url"] == "https://example.com/img.jpg"
        assert result["detail"] == "low"

    def test_p_image_to_ir_with_data_uri(self):
        """Test OpenAI data URI → IR ImagePart with image_data."""
        provider = {
            "type": "image_url",
            "image_url": {"url": "data:image/jpeg;base64,xyz789", "detail": "high"},
        }
        result = OpenAIChatContentOps.p_image_to_ir(provider)
        assert result["type"] == "image"
        assert result["image_data"]["data"] == "xyz789"
        assert result["image_data"]["media_type"] == "image/jpeg"
        assert result["detail"] == "high"

    def test_image_url_round_trip(self):
        """Test image URL round-trip."""
        original = ImagePart(
            type="image", image_url="https://example.com/img.jpg", detail="auto"
        )
        provider = OpenAIChatContentOps.ir_image_to_p(original)
        restored = OpenAIChatContentOps.p_image_to_ir(provider)
        assert restored["image_url"] == original["image_url"]

    def test_image_base64_round_trip(self):
        """Test image base64 round-trip."""
        original = ImagePart(
            type="image",
            image_data={"data": "base64data", "media_type": "image/png"},
            detail="high",
        )
        provider = OpenAIChatContentOps.ir_image_to_p(original)
        restored = OpenAIChatContentOps.p_image_to_ir(provider)
        assert restored["image_data"]["data"] == "base64data"
        assert restored["image_data"]["media_type"] == "image/png"

    # ==================== File (not supported) ====================

    def test_ir_file_to_p_raises(self):
        """Test ir_file_to_p raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="does not support file"):
            OpenAIChatContentOps.ir_file_to_p({"type": "file"})

    def test_p_file_to_ir_raises(self):
        """Test p_file_to_ir raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="does not support file"):
            OpenAIChatContentOps.p_file_to_ir({})

    # ==================== Audio (not supported) ====================

    def test_ir_audio_to_p_raises(self):
        """Test ir_audio_to_p raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="does not support audio"):
            OpenAIChatContentOps.ir_audio_to_p({"type": "audio", "audio_id": "a1"})

    def test_p_audio_to_ir_raises(self):
        """Test p_audio_to_ir raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="does not support audio"):
            OpenAIChatContentOps.p_audio_to_ir({})

    # ==================== Reasoning ====================

    def test_ir_reasoning_to_p_returns_none(self):
        """Test ir_reasoning_to_p returns None with warning."""
        with pytest.warns(UserWarning, match="Reasoning content not supported"):
            result = OpenAIChatContentOps.ir_reasoning_to_p(
                {"type": "reasoning", "reasoning": "thinking..."}
            )
        assert result is None

    # ==================== Refusal ====================

    def test_ir_refusal_to_p(self):
        """Test IR RefusalPart → refusal dict."""
        result = OpenAIChatContentOps.ir_refusal_to_p(
            {"type": "refusal", "refusal": "I cannot do that"}
        )
        assert result == {"refusal": "I cannot do that"}

    def test_p_refusal_to_ir(self):
        """Test refusal string → IR RefusalPart."""
        result = OpenAIChatContentOps.p_refusal_to_ir("I cannot do that")
        assert result["type"] == "refusal"
        assert result["refusal"] == "I cannot do that"

    # ==================== Citation ====================

    def test_ir_citation_to_p_url(self):
        """Test IR CitationPart with url_citation → OpenAI annotation."""
        ir_citation = cast(CitationPart, {
            "type": "citation",
            "url_citation": {
                "start_index": 0,
                "end_index": 10,
                "title": "Test",
                "url": "https://example.com",
            },
        })
        result = OpenAIChatContentOps.ir_citation_to_p(ir_citation)
        assert result is not None
        assert result["type"] == "url_citation"
        assert result["url"] == "https://example.com"

    def test_ir_citation_to_p_no_url(self):
        """Test ir_citation_to_p returns None when no url_citation."""
        result = OpenAIChatContentOps.ir_citation_to_p({"type": "citation"})
        assert result is None

    def test_p_citation_to_ir(self):
        """Test OpenAI annotation → IR CitationPart."""
        provider = {
            "type": "url_citation",
            "start_index": 5,
            "end_index": 15,
            "title": "Source",
            "url": "https://example.com/source",
        }
        result = OpenAIChatContentOps.p_citation_to_ir(provider)
        assert result["type"] == "citation"
        assert result["url_citation"]["url"] == "https://example.com/source"
