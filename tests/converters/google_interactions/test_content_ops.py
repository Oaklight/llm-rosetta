"""Tests for Google Interactions content_ops."""

from llm_rosetta.converters.google_interactions.content_ops import (
    GoogleInteractionsContentOps,
)


class TestTextConversion:
    def test_ir_text_to_provider(self):
        ir = {"type": "text", "text": "Hello world"}
        result = GoogleInteractionsContentOps.ir_text_to_p(ir)  # ty: ignore
        assert result == {"type": "text", "text": "Hello world"}

    def test_provider_text_to_ir(self):
        p = {"type": "text", "text": "Hello world"}
        result = GoogleInteractionsContentOps.p_text_to_ir(p)
        assert result == {"type": "text", "text": "Hello world"}

    def test_roundtrip_text(self):
        ir = {"type": "text", "text": "Round trip test"}
        p = GoogleInteractionsContentOps.ir_text_to_p(ir)  # ty: ignore
        back = GoogleInteractionsContentOps.p_text_to_ir(p)
        assert back["text"] == ir["text"]


class TestImageConversion:
    def test_ir_image_data_to_provider(self):
        ir = {
            "type": "image",
            "image_data": {"data": "abc123", "media_type": "image/jpeg"},
        }
        result = GoogleInteractionsContentOps.ir_image_to_p(ir)  # ty: ignore
        assert result["type"] == "image"
        assert result["data"] == "abc123"
        assert result["mime_type"] == "image/jpeg"

    def test_ir_image_url_to_provider(self):
        ir = {"type": "image", "image_url": "https://example.com/img.png"}
        result = GoogleInteractionsContentOps.ir_image_to_p(ir)  # ty: ignore
        assert result["type"] == "image"
        assert result["uri"] == "https://example.com/img.png"

    def test_provider_image_data_to_ir(self):
        p = {"type": "image", "data": "abc123", "mime_type": "image/jpeg"}
        result = GoogleInteractionsContentOps.p_image_to_ir(p)
        assert result["type"] == "image"
        assert result["image_data"]["data"] == "abc123"
        assert result["image_data"]["media_type"] == "image/jpeg"

    def test_provider_image_uri_to_ir(self):
        p = {"type": "image", "uri": "https://example.com/img.png"}
        result = GoogleInteractionsContentOps.p_image_to_ir(p)
        assert result["type"] == "image"
        assert result["image_url"] == "https://example.com/img.png"

    def test_roundtrip_image_data(self):
        ir = {
            "type": "image",
            "image_data": {"data": "base64data", "media_type": "image/png"},
        }
        p = GoogleInteractionsContentOps.ir_image_to_p(ir)  # ty: ignore
        back = GoogleInteractionsContentOps.p_image_to_ir(p)
        assert back["image_data"]["data"] == ir["image_data"]["data"]
        assert back["image_data"]["media_type"] == ir["image_data"]["media_type"]


class TestThoughtConversion:
    def test_thought_step_to_ir(self):
        step = {
            "type": "thought",
            "signature": "sig_abc",
            "summary": [{"type": "text", "text": "I need to think about this."}],
        }
        result = GoogleInteractionsContentOps.p_thought_to_ir(step)
        assert result["type"] == "reasoning"
        assert result["signature"] == "sig_abc"
        assert result["reasoning"] == "I need to think about this."

    def test_thought_step_no_summary(self):
        step = {"type": "thought", "signature": "sig_abc"}
        result = GoogleInteractionsContentOps.p_thought_to_ir(step)
        assert result["type"] == "reasoning"
        assert result["signature"] == "sig_abc"
        assert "reasoning" not in result

    def test_ir_reasoning_to_thought_step(self):
        ir = {
            "type": "reasoning",
            "reasoning": "Let me think...",
            "signature": "sig_xyz",
        }
        result = GoogleInteractionsContentOps.ir_reasoning_to_p(ir)  # ty: ignore
        assert result["type"] == "thought"
        assert result["signature"] == "sig_xyz"
        assert result["summary"] == [{"type": "text", "text": "Let me think..."}]

    def test_roundtrip_thought(self):
        step = {
            "type": "thought",
            "signature": "sig_rt",
            "summary": [{"type": "text", "text": "Thinking..."}],
        }
        ir = GoogleInteractionsContentOps.p_thought_to_ir(step)
        back = GoogleInteractionsContentOps.ir_reasoning_to_p(ir)
        assert back["type"] == "thought"
        assert back["signature"] == "sig_rt"
        assert back["summary"][0]["text"] == "Thinking..."


class TestAnnotations:
    def test_annotations_to_citations(self):
        annotations = [
            {
                "url": "https://example.com",
                "title": "Example",
                "start_index": 0,
                "end_index": 10,
            }
        ]
        result = GoogleInteractionsContentOps.p_annotations_to_ir(annotations)
        assert len(result) == 1
        assert result[0]["type"] == "citation"
        assert result[0]["url_citation"]["url"] == "https://example.com"
        assert result[0]["url_citation"]["title"] == "Example"

    def test_empty_annotations(self):
        result = GoogleInteractionsContentOps.p_annotations_to_ir([])
        assert result == []


class TestContentDispatch:
    def test_dispatch_text(self):
        content = {"type": "text", "text": "hello"}
        result = GoogleInteractionsContentOps.p_content_to_ir(content)
        assert result["type"] == "text"  # ty: ignore
        assert result["text"] == "hello"  # ty: ignore

    def test_dispatch_image(self):
        content = {"type": "image", "data": "abc", "mime_type": "image/png"}
        result = GoogleInteractionsContentOps.p_content_to_ir(content)
        assert result["type"] == "image"  # ty: ignore

    def test_dispatch_unknown(self):
        content = {"type": "video", "data": "abc"}
        result = GoogleInteractionsContentOps.p_content_to_ir(content)
        assert result is None
