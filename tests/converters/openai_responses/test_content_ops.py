"""
OpenAI Responses ContentOps unit tests.
"""

import pytest

from llm_rosetta.converters.openai_responses.content_ops import (
    OpenAIResponsesContentOps,
)
from llm_rosetta.types.ir import CitationPart, ImagePart, ReasoningPart, TextPart


class TestOpenAIResponsesContentOps:
    """Unit tests for OpenAIResponsesContentOps."""

    # ==================== Text ====================

    def test_ir_text_to_p_input(self):
        """Test IR TextPart → OpenAI Responses input_text (default context)."""
        ir_text = TextPart(type="text", text="Hello, world!")
        result = OpenAIResponsesContentOps.ir_text_to_p(ir_text)
        assert result == {"type": "input_text", "text": "Hello, world!"}

    def test_ir_text_to_p_output(self):
        """Test IR TextPart → OpenAI Responses output_text (output context)."""
        ir_text = TextPart(type="text", text="Response text")
        result = OpenAIResponsesContentOps.ir_text_to_p(ir_text, context="output")
        assert result == {"type": "output_text", "text": "Response text"}

    def test_p_text_to_ir_from_string(self):
        """Test OpenAI string → IR TextPart."""
        result = OpenAIResponsesContentOps.p_text_to_ir("Hello!")
        assert result["type"] == "text"
        assert result["text"] == "Hello!"

    def test_p_text_to_ir_from_input_text(self):
        """Test OpenAI input_text dict → IR TextPart."""
        result = OpenAIResponsesContentOps.p_text_to_ir(
            {"type": "input_text", "text": "Hi"}
        )
        assert result["type"] == "text"
        assert result["text"] == "Hi"

    def test_p_text_to_ir_from_output_text(self):
        """Test OpenAI output_text dict → IR TextPart."""
        result = OpenAIResponsesContentOps.p_text_to_ir(
            {"type": "output_text", "text": "Response"}
        )
        assert result["type"] == "text"
        assert result["text"] == "Response"

    def test_p_text_to_ir_from_text_dict(self):
        """Test OpenAI plain text dict → IR TextPart."""
        result = OpenAIResponsesContentOps.p_text_to_ir(
            {"type": "text", "text": "Plain"}
        )
        assert result["type"] == "text"
        assert result["text"] == "Plain"

    def test_p_text_to_ir_invalid(self):
        """Test p_text_to_ir raises on invalid input."""
        with pytest.raises(ValueError, match="Cannot convert"):
            OpenAIResponsesContentOps.p_text_to_ir(42)

    def test_text_round_trip_input(self):
        """Test text round-trip: IR → Provider (input) → IR."""
        original = TextPart(type="text", text="Round trip test")
        provider = OpenAIResponsesContentOps.ir_text_to_p(original, context="input")
        restored = OpenAIResponsesContentOps.p_text_to_ir(provider)
        assert restored["text"] == original["text"]

    def test_text_round_trip_output(self):
        """Test text round-trip: IR → Provider (output) → IR."""
        original = TextPart(type="text", text="Output round trip")
        provider = OpenAIResponsesContentOps.ir_text_to_p(original, context="output")
        restored = OpenAIResponsesContentOps.p_text_to_ir(provider)
        assert restored["text"] == original["text"]

    # ==================== Image ====================

    def test_ir_image_to_p_with_url(self):
        """Test IR ImagePart with URL → OpenAI Responses input_image."""
        ir_image = ImagePart(
            type="image", image_url="https://example.com/img.jpg", detail="high"
        )
        result = OpenAIResponsesContentOps.ir_image_to_p(ir_image)
        assert result["type"] == "input_image"
        assert result["image_url"] == "https://example.com/img.jpg"
        assert result["detail"] == "high"

    def test_ir_image_to_p_with_base64(self):
        """Test IR ImagePart with base64 → OpenAI Responses data URI."""
        ir_image = ImagePart(
            type="image",
            image_data={"data": "abc123", "media_type": "image/png"},
        )
        result = OpenAIResponsesContentOps.ir_image_to_p(ir_image)
        assert result["type"] == "input_image"
        assert result["image_url"] == "data:image/png;base64,abc123"

    def test_ir_image_to_p_no_data(self):
        """Test ir_image_to_p raises when no image data."""
        with pytest.raises(ValueError, match="image_url"):
            OpenAIResponsesContentOps.ir_image_to_p({"type": "image"})

    def test_p_image_to_ir_with_url(self):
        """Test OpenAI Responses input_image with URL → IR ImagePart."""
        provider = {
            "type": "input_image",
            "image_url": "https://example.com/img.jpg",
            "detail": "low",
        }
        result = OpenAIResponsesContentOps.p_image_to_ir(provider)
        assert result["type"] == "image"
        assert result["image_url"] == "https://example.com/img.jpg"
        assert result["detail"] == "low"

    def test_p_image_to_ir_with_data_uri(self):
        """Test OpenAI Responses data URI → IR ImagePart with image_data."""
        provider = {
            "type": "input_image",
            "image_url": "data:image/jpeg;base64,xyz789",
            "detail": "high",
        }
        result = OpenAIResponsesContentOps.p_image_to_ir(provider)
        assert result["type"] == "image"
        assert result["image_data"]["data"] == "xyz789"
        assert result["image_data"]["media_type"] == "image/jpeg"
        assert result["detail"] == "high"

    def test_p_image_to_ir_with_file_id(self):
        """Test OpenAI Responses file_id image → IR dict with file_id."""
        provider = {
            "type": "input_image",
            "file_id": "file-abc123",
            "detail": "auto",
        }
        result = OpenAIResponsesContentOps.p_image_to_ir(provider)
        assert result["type"] == "image"
        assert result["file_id"] == "file-abc123"

    def test_image_url_round_trip(self):
        """Test image URL round-trip."""
        original = ImagePart(
            type="image", image_url="https://example.com/img.jpg", detail="auto"
        )
        provider = OpenAIResponsesContentOps.ir_image_to_p(original)
        restored = OpenAIResponsesContentOps.p_image_to_ir(provider)
        assert restored["image_url"] == original["image_url"]

    def test_image_base64_round_trip(self):
        """Test image base64 round-trip."""
        original = ImagePart(
            type="image",
            image_data={"data": "base64data", "media_type": "image/png"},
            detail="high",
        )
        provider = OpenAIResponsesContentOps.ir_image_to_p(original)
        restored = OpenAIResponsesContentOps.p_image_to_ir(provider)
        assert restored["image_data"]["data"] == "base64data"
        assert restored["image_data"]["media_type"] == "image/png"

    # ==================== File ====================

    def test_ir_file_to_p_with_data(self):
        """Test IR FilePart with file_data → OpenAI Responses input_file."""
        ir_file = {
            "type": "file",
            "file_name": "doc.pdf",
            "file_data": {"data": "filedata123", "media_type": "application/pdf"},
        }
        result = OpenAIResponsesContentOps.ir_file_to_p(ir_file)
        assert result["type"] == "input_file"
        assert result["filename"] == "doc.pdf"
        assert result["file_data"] == "filedata123"

    def test_ir_file_to_p_with_url(self):
        """Test IR FilePart with file_url → OpenAI Responses input_file."""
        ir_file = {
            "type": "file",
            "file_name": "doc.pdf",
            "file_url": "https://example.com/doc.pdf",
        }
        result = OpenAIResponsesContentOps.ir_file_to_p(ir_file)
        assert result["type"] == "input_file"
        assert result["file_url"] == "https://example.com/doc.pdf"

    def test_ir_file_to_p_no_data(self):
        """Test ir_file_to_p raises when no file data."""
        with pytest.raises(ValueError, match="file_data or file_url"):
            OpenAIResponsesContentOps.ir_file_to_p({"type": "file"})

    def test_p_file_to_ir_with_data(self):
        """Test OpenAI Responses input_file with data → IR FilePart."""
        provider = {
            "type": "input_file",
            "filename": "test.txt",
            "file_data": "rawdata",
        }
        result = OpenAIResponsesContentOps.p_file_to_ir(provider)
        assert result["type"] == "file"
        assert result["file_name"] == "test.txt"
        assert result["file_data"]["data"] == "rawdata"

    def test_p_file_to_ir_with_url(self):
        """Test OpenAI Responses input_file with URL → IR FilePart."""
        provider = {
            "type": "input_file",
            "filename": "test.txt",
            "file_url": "https://example.com/test.txt",
        }
        result = OpenAIResponsesContentOps.p_file_to_ir(provider)
        assert result["type"] == "file"
        assert result["file_url"] == "https://example.com/test.txt"

    def test_p_file_to_ir_with_file_id(self):
        """Test OpenAI Responses input_file with file_id → IR dict."""
        provider = {
            "type": "input_file",
            "filename": "test.txt",
            "file_id": "file-xyz",
        }
        result = OpenAIResponsesContentOps.p_file_to_ir(provider)
        assert result["type"] == "file"
        assert result["file_id"] == "file-xyz"

    # ==================== Audio (not supported) ====================

    def test_ir_audio_to_p_raises(self):
        """Test ir_audio_to_p raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="does not support audio"):
            OpenAIResponsesContentOps.ir_audio_to_p({"type": "audio", "audio_id": "a1"})

    def test_p_audio_to_ir_raises(self):
        """Test p_audio_to_ir raises NotImplementedError."""
        with pytest.raises(NotImplementedError, match="does not support audio"):
            OpenAIResponsesContentOps.p_audio_to_ir({})

    # ==================== Reasoning ====================

    def test_ir_reasoning_to_p(self):
        """Test IR ReasoningPart → OpenAI Responses reasoning item."""
        ir_reasoning = ReasoningPart(type="reasoning", reasoning="thinking...")
        result = OpenAIResponsesContentOps.ir_reasoning_to_p(ir_reasoning)
        assert result["type"] == "reasoning"
        assert result["content"] == "thinking..."

    def test_ir_reasoning_to_p_empty(self):
        """Test IR ReasoningPart with no reasoning → empty content."""
        ir_reasoning = {"type": "reasoning"}
        result = OpenAIResponsesContentOps.ir_reasoning_to_p(ir_reasoning)
        assert result["type"] == "reasoning"
        assert result["content"] == ""

    def test_p_reasoning_to_ir_with_content(self):
        """Test OpenAI Responses reasoning with content → IR ReasoningPart."""
        provider = {"type": "reasoning", "content": "I need to think about this..."}
        result = OpenAIResponsesContentOps.p_reasoning_to_ir(provider)
        assert result is not None
        assert result["type"] == "reasoning"
        assert result["reasoning"] == "I need to think about this..."

    def test_p_reasoning_to_ir_with_reasoning_field(self):
        """Test OpenAI Responses reasoning with 'reasoning' field → IR ReasoningPart."""
        provider = {"type": "reasoning", "reasoning": "Deep thought"}
        result = OpenAIResponsesContentOps.p_reasoning_to_ir(provider)
        assert result is not None
        assert result["reasoning"] == "Deep thought"

    def test_p_reasoning_to_ir_empty(self):
        """Test OpenAI Responses reasoning with empty content → None."""
        provider = {"type": "reasoning", "content": None}
        result = OpenAIResponsesContentOps.p_reasoning_to_ir(provider)
        assert result is None

    def test_reasoning_round_trip(self):
        """Test reasoning round-trip: IR → Provider → IR."""
        original = ReasoningPart(type="reasoning", reasoning="Step by step analysis")
        provider = OpenAIResponsesContentOps.ir_reasoning_to_p(original)
        restored = OpenAIResponsesContentOps.p_reasoning_to_ir(provider)
        assert restored is not None
        assert restored["reasoning"] == original["reasoning"]

    # ==================== Refusal ====================

    def test_ir_refusal_to_p_returns_none(self):
        """Test ir_refusal_to_p returns None with warning."""
        with pytest.warns(UserWarning, match="Refusal content not directly supported"):
            result = OpenAIResponsesContentOps.ir_refusal_to_p(
                {"type": "refusal", "refusal": "I cannot do that"}
            )
        assert result is None

    def test_p_refusal_to_ir_raises(self):
        """Test p_refusal_to_ir raises NotImplementedError."""
        with pytest.raises(
            NotImplementedError, match="does not produce refusal content"
        ):
            OpenAIResponsesContentOps.p_refusal_to_ir("I cannot do that")

    # ==================== Citation ====================

    def test_ir_citation_to_p_url(self):
        """Test IR CitationPart with url_citation → OpenAI annotation."""
        ir_citation = CitationPart(
            type="citation",
            url_citation={
                "start_index": 0,
                "end_index": 10,
                "title": "Test",
                "url": "https://example.com",
            },
        )
        result = OpenAIResponsesContentOps.ir_citation_to_p(ir_citation)
        assert result is not None
        assert result["type"] == "url_citation"
        assert result["url"] == "https://example.com"
        assert result["start_index"] == 0
        assert result["end_index"] == 10
        assert result["title"] == "Test"

    def test_ir_citation_to_p_no_url(self):
        """Test ir_citation_to_p returns None when no url_citation."""
        result = OpenAIResponsesContentOps.ir_citation_to_p({"type": "citation"})
        assert result is None

    def test_p_citation_to_ir_url(self):
        """Test OpenAI url_citation annotation → IR CitationPart."""
        provider = {
            "type": "url_citation",
            "start_index": 5,
            "end_index": 15,
            "title": "Source",
            "url": "https://example.com/source",
        }
        result = OpenAIResponsesContentOps.p_citation_to_ir(provider)
        assert result["type"] == "citation"
        assert result["url_citation"]["url"] == "https://example.com/source"
        assert result["url_citation"]["start_index"] == 5
        assert result["url_citation"]["end_index"] == 15

    def test_p_citation_to_ir_unknown_type(self):
        """Test OpenAI unknown citation type → fallback CitationPart."""
        provider = {"type": "file_citation", "file_id": "file-123"}
        result = OpenAIResponsesContentOps.p_citation_to_ir(provider)
        assert result["type"] == "citation"

    def test_citation_round_trip(self):
        """Test citation round-trip: IR → Provider → IR."""
        original = CitationPart(
            type="citation",
            url_citation={
                "start_index": 10,
                "end_index": 20,
                "title": "Reference",
                "url": "https://example.com/ref",
            },
        )
        provider = OpenAIResponsesContentOps.ir_citation_to_p(original)
        assert provider is not None
        restored = OpenAIResponsesContentOps.p_citation_to_ir(provider)
        assert restored["url_citation"]["url"] == original["url_citation"]["url"]
        assert (
            restored["url_citation"]["start_index"]
            == original["url_citation"]["start_index"]
        )
