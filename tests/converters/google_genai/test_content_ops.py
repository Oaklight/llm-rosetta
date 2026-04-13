"""
Google GenAI ContentOps unit tests.
"""

from typing import cast

import pytest

from llm_rosetta.converters.google_genai.content_ops import GoogleGenAIContentOps
from llm_rosetta.types.ir import AudioPart, FilePart, ImagePart, ReasoningPart, TextPart


class TestGoogleGenAIContentOps:
    """Unit tests for GoogleGenAIContentOps."""

    # ==================== Text ====================

    def test_ir_text_to_p(self):
        """Test IR TextPart → Google text Part."""
        ir_text = TextPart(type="text", text="Hello, world!")
        result = GoogleGenAIContentOps.ir_text_to_p(ir_text)
        assert result == {"text": "Hello, world!"}

    def test_ir_text_to_p_with_thought_signature(self):
        """Test IR TextPart with thought_signature → Google text Part."""
        ir_text = cast(
            TextPart,
            {
                "type": "text",
                "text": "Hello",
                "provider_metadata": {"google": {"thought_signature": "sig123"}},
            },
        )
        result = GoogleGenAIContentOps.ir_text_to_p(ir_text)
        assert result["text"] == "Hello"
        assert result["thoughtSignature"] == "sig123"

    def test_p_text_to_ir(self):
        """Test Google text Part → IR TextPart."""
        result = GoogleGenAIContentOps.p_text_to_ir({"text": "Hi there"})
        assert result["type"] == "text"
        assert result["text"] == "Hi there"

    def test_text_round_trip(self):
        """Test text round-trip: IR → Provider → IR."""
        original = TextPart(type="text", text="Round trip test")
        provider = GoogleGenAIContentOps.ir_text_to_p(original)
        restored = GoogleGenAIContentOps.p_text_to_ir(provider)
        assert restored["text"] == original["text"]

    # ==================== Image ====================

    def test_ir_image_to_p_with_image_data_structured(self):
        """Test IR ImagePart with image_data → Google inline_data Part."""
        ir_image = ImagePart(
            type="image",
            image_data={"data": "base64data", "media_type": "image/jpeg"},
        )
        result = GoogleGenAIContentOps.ir_image_to_p(ir_image)
        assert result is not None
        assert result["inlineData"]["mimeType"] == "image/jpeg"
        assert result["inlineData"]["data"] == "base64data"

    def test_ir_image_to_p_with_image_data(self):
        """Test IR ImagePart with image_data → Google inline_data Part."""
        ir_image = ImagePart(
            type="image",
            image_data={"data": "abc123", "media_type": "image/png"},
        )
        result = GoogleGenAIContentOps.ir_image_to_p(ir_image)
        assert result is not None
        assert result["inlineData"]["mimeType"] == "image/png"
        assert result["inlineData"]["data"] == "abc123"

    def test_ir_image_to_p_with_url_downloads(self):
        """Test IR ImagePart with URL downloads and converts to inline base64."""
        from unittest.mock import MagicMock, patch

        ir_image = ImagePart(type="image", image_url="https://example.com/img.jpg")
        mock_resp = MagicMock()
        mock_resp.content = b"\x89PNG\r\n"
        mock_resp.headers = {"content-type": "image/png"}
        mock_resp.raise_for_status = MagicMock()

        with patch(
            "llm_rosetta.converters.google_genai.content_ops.httpx.get",
            return_value=mock_resp,
        ):
            result = GoogleGenAIContentOps.ir_image_to_p(ir_image)

        assert result is not None
        assert result["inlineData"]["mimeType"] == "image/png"
        import base64

        assert result["inlineData"]["data"] == base64.b64encode(b"\x89PNG\r\n").decode()

    def test_ir_image_to_p_with_url_download_failure(self):
        """Test IR ImagePart with URL returns None on download failure."""
        from unittest.mock import patch

        ir_image = ImagePart(type="image", image_url="https://example.com/img.jpg")
        with patch(
            "llm_rosetta.converters.google_genai.content_ops.httpx.get",
            side_effect=Exception("timeout"),
        ):
            result = GoogleGenAIContentOps.ir_image_to_p(ir_image)
        assert result is None

    def test_ir_image_to_p_with_data_uri(self):
        """Test IR ImagePart with data URI is parsed directly."""
        data_uri = "data:image/png;base64,aWNvbg=="
        ir_image = ImagePart(type="image", image_url=data_uri)
        result = GoogleGenAIContentOps.ir_image_to_p(ir_image)
        assert result is not None
        assert result["inlineData"]["mimeType"] == "image/png"
        assert result["inlineData"]["data"] == "aWNvbg=="

    def test_p_image_to_ir(self):
        """Test Google inline_data image Part → IR ImagePart."""
        provider = {"inline_data": {"mime_type": "image/jpeg", "data": "base64data"}}
        result = GoogleGenAIContentOps.p_image_to_ir(provider)
        assert result["type"] == "image"
        assert result["image_data"]["data"] == "base64data"
        assert result["image_data"]["media_type"] == "image/jpeg"

    def test_image_round_trip(self):
        """Test image round-trip with image_data."""
        original = ImagePart(
            type="image",
            image_data={"data": "imgdata", "media_type": "image/gif"},
        )
        provider = GoogleGenAIContentOps.ir_image_to_p(original)
        assert provider is not None
        restored = GoogleGenAIContentOps.p_image_to_ir(provider)
        assert restored["image_data"]["data"] == "imgdata"
        assert restored["image_data"]["media_type"] == "image/gif"

    # ==================== File ====================

    def test_ir_file_to_p_with_file_data_structured(self):
        """Test IR FilePart with file_data → Google inline_data Part."""
        ir_file = FilePart(
            type="file",
            file_data={"data": "filedata", "media_type": "text/csv"},
        )
        result = GoogleGenAIContentOps.ir_file_to_p(ir_file)
        assert result is not None
        assert result["inlineData"]["data"] == "filedata"
        assert result["inlineData"]["mimeType"] == "text/csv"

    def test_ir_file_to_p_with_file_data(self):
        """Test IR FilePart with file_data → Google inline_data Part."""
        ir_file = cast(
            FilePart,
            {
                "type": "file",
                "file_data": {"data": "pdfdata", "media_type": "application/pdf"},
            },
        )
        result = GoogleGenAIContentOps.ir_file_to_p(ir_file)
        assert result is not None
        assert result["inlineData"]["data"] == "pdfdata"

    def test_ir_file_to_p_with_url_warns(self):
        """Test IR FilePart with URL emits warning and returns None."""
        ir_file = cast(
            FilePart, {"type": "file", "file_url": "https://example.com/doc.pdf"}
        )
        with pytest.warns(UserWarning, match="不直接支持文件URL"):
            result = GoogleGenAIContentOps.ir_file_to_p(ir_file)
        assert result is None

    def test_p_file_to_ir(self):
        """Test Google inline_data file Part → IR FilePart."""
        provider = {"inline_data": {"data": "filedata", "mime_type": "application/pdf"}}
        result = GoogleGenAIContentOps.p_file_to_ir(provider)
        assert result["type"] == "file"
        assert result["file_data"]["data"] == "filedata"
        assert result["file_data"]["media_type"] == "application/pdf"

    # ==================== Audio ====================

    def test_ir_audio_to_p_with_audio_data_structured(self):
        """Test IR AudioPart with audio_data → Google inline_data Part."""
        ir_audio = AudioPart(
            type="audio",
            audio_data={"data": "audiodata", "media_type": "audio/wav"},
        )
        result = GoogleGenAIContentOps.ir_audio_to_p(ir_audio)
        assert result is not None
        assert result["inlineData"]["data"] == "audiodata"
        assert result["inlineData"]["mimeType"] == "audio/wav"

    def test_ir_audio_to_p_unsupported_warns(self):
        """Test IR AudioPart without data emits warning."""
        ir_audio = cast(AudioPart, {"type": "audio"})
        with pytest.warns(UserWarning, match="不支持的音频格式"):
            result = GoogleGenAIContentOps.ir_audio_to_p(ir_audio)
        assert result is None

    def test_p_audio_to_ir_inline(self):
        """Test Google inline_data audio Part → IR AudioPart."""
        provider = {"inline_data": {"mime_type": "audio/wav", "data": "audiodata"}}
        result = GoogleGenAIContentOps.p_audio_to_ir(provider)
        assert result["type"] == "audio"
        assert result["audio_data"]["data"] == "audiodata"
        assert result["audio_data"]["media_type"] == "audio/wav"

    def test_p_audio_to_ir_file_data(self):
        """Test Google file_data audio Part → IR AudioPart."""
        provider = {
            "file_data": {"file_uri": "gs://bucket/audio.wav", "mime_type": "audio/wav"}
        }
        result = GoogleGenAIContentOps.p_audio_to_ir(provider)
        assert result["type"] == "audio"
        assert result["url"] == "gs://bucket/audio.wav"

    # ==================== Reasoning (Thought) ====================

    def test_ir_reasoning_to_p(self):
        """Test IR ReasoningPart → Google thought Part."""
        ir_reasoning = ReasoningPart(type="reasoning", reasoning="Thinking...")
        result = GoogleGenAIContentOps.ir_reasoning_to_p(ir_reasoning)
        assert result["thought"] is True
        assert result["text"] == "Thinking..."

    def test_p_reasoning_to_ir(self):
        """Test Google thought Part → IR ReasoningPart."""
        provider = {"thought": True, "text": "I should use a tool."}
        result = GoogleGenAIContentOps.p_reasoning_to_ir(provider)
        assert result["type"] == "reasoning"
        assert result["reasoning"] == "I should use a tool."

    def test_reasoning_round_trip(self):
        """Test reasoning round-trip."""
        original = ReasoningPart(type="reasoning", reasoning="deep thought")
        provider = GoogleGenAIContentOps.ir_reasoning_to_p(original)
        restored = GoogleGenAIContentOps.p_reasoning_to_ir(provider)
        assert restored["reasoning"] == original["reasoning"]

    # ==================== Composite Part Dispatch ====================

    def test_p_part_to_ir_text(self):
        """Test p_part_to_ir dispatches text correctly."""
        parts = GoogleGenAIContentOps.p_part_to_ir({"text": "Hello"})
        assert len(parts) == 1
        assert parts[0]["type"] == "text"
        assert parts[0]["text"] == "Hello"

    def test_p_part_to_ir_inline_image(self):
        """Test p_part_to_ir dispatches inline image correctly."""
        parts = GoogleGenAIContentOps.p_part_to_ir(
            {"inline_data": {"mime_type": "image/png", "data": "imgdata"}}
        )
        assert len(parts) == 1
        assert parts[0]["type"] == "image"

    def test_p_part_to_ir_inline_audio(self):
        """Test p_part_to_ir dispatches inline audio correctly."""
        parts = GoogleGenAIContentOps.p_part_to_ir(
            {"inline_data": {"mime_type": "audio/wav", "data": "audiodata"}}
        )
        assert len(parts) == 1
        assert parts[0]["type"] == "audio"

    def test_p_part_to_ir_inline_file(self):
        """Test p_part_to_ir dispatches inline file correctly."""
        parts = GoogleGenAIContentOps.p_part_to_ir(
            {"inline_data": {"mime_type": "application/pdf", "data": "pdfdata"}}
        )
        assert len(parts) == 1
        assert parts[0]["type"] == "file"

    def test_p_part_to_ir_file_data_image(self):
        """Test p_part_to_ir dispatches file_data image correctly."""
        parts = GoogleGenAIContentOps.p_part_to_ir(
            {"file_data": {"file_uri": "gs://a/b.png", "mime_type": "image/png"}}
        )
        assert len(parts) == 1
        assert parts[0]["type"] == "image"
        assert parts[0]["image_url"] == "gs://a/b.png"

    def test_p_part_to_ir_file_data_audio(self):
        """Test p_part_to_ir dispatches file_data audio correctly."""
        parts = GoogleGenAIContentOps.p_part_to_ir(
            {"file_data": {"file_uri": "gs://a/b.wav", "mime_type": "audio/wav"}}
        )
        assert len(parts) == 1
        assert parts[0]["type"] == "audio"
        assert parts[0]["url"] == "gs://a/b.wav"

    def test_p_part_to_ir_file_data_generic(self):
        """Test p_part_to_ir dispatches file_data generic file correctly."""
        parts = GoogleGenAIContentOps.p_part_to_ir(
            {"file_data": {"file_uri": "gs://a/b.txt", "mime_type": "text/plain"}}
        )
        assert len(parts) == 1
        assert parts[0]["type"] == "file"
        assert parts[0]["file_url"] == "gs://a/b.txt"

    def test_p_part_to_ir_with_thought_signature(self):
        """Test p_part_to_ir preserves thoughtSignature."""
        parts = GoogleGenAIContentOps.p_part_to_ir(
            {"text": "Hello", "thoughtSignature": "sig789"}
        )
        assert len(parts) == 1
        assert parts[0]["provider_metadata"]["google"]["thought_signature"] == "sig789"

    def test_p_part_to_ir_camelcase_inline_image(self):
        """Test p_part_to_ir handles camelCase inlineData for image."""
        parts = GoogleGenAIContentOps.p_part_to_ir(
            {"inlineData": {"mimeType": "image/png", "data": "imgdata"}}
        )
        assert len(parts) == 1
        assert parts[0]["type"] == "image"
        assert parts[0]["image_data"]["data"] == "imgdata"
        assert parts[0]["image_data"]["media_type"] == "image/png"

    def test_p_part_to_ir_camelcase_file_data(self):
        """Test p_part_to_ir handles camelCase fileData."""
        parts = GoogleGenAIContentOps.p_part_to_ir(
            {"fileData": {"fileUri": "gs://a/b.png", "mimeType": "image/png"}}
        )
        assert len(parts) == 1
        assert parts[0]["type"] == "image"
        assert parts[0]["image_url"] == "gs://a/b.png"

    def test_p_audio_to_ir_camelcase(self):
        """Test p_audio_to_ir handles camelCase inlineData."""
        provider = {"inlineData": {"mimeType": "audio/wav", "data": "audiodata"}}
        result = GoogleGenAIContentOps.p_audio_to_ir(provider)
        assert result["type"] == "audio"
        assert result["audio_data"]["data"] == "audiodata"
        assert result["audio_data"]["media_type"] == "audio/wav"

    def test_p_part_to_ir_empty_text_ignored(self):
        """Test p_part_to_ir ignores empty text."""
        parts = GoogleGenAIContentOps.p_part_to_ir({"text": ""})
        assert len(parts) == 0

    def test_p_part_to_ir_none_text_ignored(self):
        """Test p_part_to_ir ignores None text."""
        parts = GoogleGenAIContentOps.p_part_to_ir({"text": None})
        assert len(parts) == 0
