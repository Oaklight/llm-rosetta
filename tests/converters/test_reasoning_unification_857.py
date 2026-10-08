"""Regression tests for #857: unified opaque reasoning data mapping.

Verifies the two semantic categories map to the correct IR fields:
  - signature: verification data coexisting with visible reasoning text
  - redacted_data: opaque content blob (encrypted_content, redacted_thinking)

Covers same-format round-trips and cross-format conversions.
"""

from typing import cast

from llm_rosetta.converters.anthropic.content_ops import AnthropicContentOps
from llm_rosetta.converters.google_generate.content_ops import GoogleGenerateContentOps
from llm_rosetta.converters.openai_chat import OpenAIChatConverter
from llm_rosetta.converters.openai_responses.content_ops import (
    OpenAIResponsesContentOps,
)
from llm_rosetta.types.ir import ReasoningPart


class TestOpenAIResponsesEncryptedContent:
    """encrypted_content maps to redacted_data, not signature."""

    def test_provider_to_ir(self):
        provider = {
            "type": "reasoning",
            "id": "rs_001",
            "encrypted_content": "opaque-blob-abc",
            "summary": [{"type": "summary_text", "text": "visible summary"}],
        }
        ir = OpenAIResponsesContentOps.p_reasoning_to_ir(provider)
        assert ir is not None
        assert ir["redacted_data"] == "opaque-blob-abc"
        assert "signature" not in ir

    def test_ir_to_provider(self):
        ir = ReasoningPart(type="reasoning", reasoning="visible text")
        ir["redacted_data"] = "opaque-blob-xyz"
        result = OpenAIResponsesContentOps.ir_reasoning_to_p(ir, output_item=True)
        assert result is not None
        assert result["encrypted_content"] == "opaque-blob-xyz"

    def test_round_trip(self):
        provider = {
            "type": "reasoning",
            "id": "rs_002",
            "encrypted_content": "round-trip-blob",
            "summary": [],
        }
        ir = OpenAIResponsesContentOps.p_reasoning_to_ir(provider)
        assert ir is not None
        restored = OpenAIResponsesContentOps.ir_reasoning_to_p(ir, output_item=True)
        assert restored is not None
        assert restored["encrypted_content"] == "round-trip-blob"

    def test_signature_field_independent(self):
        """signature and redacted_data are independent — signature is for verification."""
        ir = ReasoningPart(type="reasoning", reasoning="visible")
        ir["signature"] = "verification-sig"
        ir["redacted_data"] = "encrypted-blob"
        result = OpenAIResponsesContentOps.ir_reasoning_to_p(ir, output_item=True)
        assert result is not None
        assert result["encrypted_content"] == "encrypted-blob"


class TestOpenAIChatEncryptedContent:
    """Chat encrypted_content maps to redacted_data, not provider_metadata."""

    def test_round_trip(self):
        conv = OpenAIChatConverter()
        resp = {
            "id": "c1",
            "object": "chat.completion",
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Answer.",
                        "reasoning_content": "Thinking...",
                        "encrypted_content": "chat-encrypted-blob",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
        ir = conv.response_from_provider(resp)
        # Verify IR uses redacted_data, not provider_metadata
        reasoning_part = cast(ReasoningPart, ir["choices"][0]["message"]["content"][0])
        assert reasoning_part["redacted_data"] == "chat-encrypted-blob"
        assert "encrypted_content" not in reasoning_part.get(
            "provider_metadata", {}
        ).get("openai_chat", {})

        out = conv.response_to_provider(ir)
        msg = out["choices"][0]["message"]
        assert msg["encrypted_content"] == "chat-encrypted-blob"

    def test_reasoning_details_still_in_provider_metadata(self):
        """reasoning_details stays in provider_metadata (structured metadata)."""
        conv = OpenAIChatConverter()
        resp = {
            "id": "c2",
            "object": "chat.completion",
            "model": "test",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": "Answer.",
                        "reasoning_content": "Thinking...",
                        "reasoning_details": [{"type": "reasoning.text", "text": "x"}],
                        "encrypted_content": "blob",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
        ir = conv.response_from_provider(resp)
        reasoning_part = cast(ReasoningPart, ir["choices"][0]["message"]["content"][0])
        assert reasoning_part["redacted_data"] == "blob"
        assert (
            reasoning_part["provider_metadata"]["openai_chat"]["reasoning_details"][0][
                "type"
            ]
            == "reasoning.text"
        )

        out = conv.response_to_provider(ir)
        msg = out["choices"][0]["message"]
        assert msg["encrypted_content"] == "blob"
        assert msg["reasoning_details"][0]["type"] == "reasoning.text"


class TestGoogleGenerateThoughtSignature:
    """Reasoning thoughtSignature maps to signature, not provider_metadata."""

    def test_provider_to_ir(self):
        provider = {
            "thought": True,
            "text": "Deep thinking...",
            "thoughtSignature": "google-sig-123",
        }
        ir = GoogleGenerateContentOps.p_reasoning_to_ir(provider)
        assert ir["signature"] == "google-sig-123"
        assert "provider_metadata" not in ir

    def test_ir_to_provider(self):
        ir = ReasoningPart(type="reasoning", reasoning="Deep thinking...")
        ir["signature"] = "google-sig-456"
        result = GoogleGenerateContentOps.ir_reasoning_to_p(ir)
        assert result["thoughtSignature"] == "google-sig-456"
        assert result["thought"] is True

    def test_round_trip(self):
        provider = {"thought": True, "text": "Hmm...", "thoughtSignature": "sig-rt"}
        ir = GoogleGenerateContentOps.p_reasoning_to_ir(provider)
        restored = GoogleGenerateContentOps.ir_reasoning_to_p(ir)
        assert restored["thoughtSignature"] == "sig-rt"
        assert restored["text"] == "Hmm..."

    def test_text_part_still_uses_provider_metadata(self):
        """TextPart thoughtSignature still uses provider_metadata (no signature field on TextPart)."""
        from typing import cast

        from llm_rosetta.types.ir import TextPart

        ir_text = cast(
            TextPart,
            {
                "type": "text",
                "text": "Hello",
                "provider_metadata": {"google": {"thought_signature": "text-sig"}},
            },
        )
        result = GoogleGenerateContentOps.ir_text_to_p(ir_text)
        assert result["thoughtSignature"] == "text-sig"


class TestCrossFormatConversion:
    """Cross-format reasoning conversions after #857 unification."""

    def test_responses_to_anthropic_encrypted_content_becomes_redacted_thinking(self):
        """OpenAI Responses encrypted_content → Anthropic redacted_thinking."""
        responses_item = {
            "type": "reasoning",
            "id": "rs_cross",
            "encrypted_content": "cross-format-blob",
            "summary": [],
        }
        ir = OpenAIResponsesContentOps.p_reasoning_to_ir(responses_item)
        assert ir is not None
        assert ir["redacted_data"] == "cross-format-blob"

        anthropic_block = AnthropicContentOps.ir_reasoning_to_p(ir)
        assert anthropic_block["type"] == "redacted_thinking"
        assert anthropic_block["data"] == "cross-format-blob"

    def test_anthropic_redacted_to_responses_encrypted_content(self):
        """Anthropic redacted_thinking → OpenAI Responses encrypted_content."""
        ir = ReasoningPart(type="reasoning")
        ir["redacted_data"] = "anthropic-redacted-blob"

        result = OpenAIResponsesContentOps.ir_reasoning_to_p(ir, output_item=True)
        assert result is not None
        assert result["encrypted_content"] == "anthropic-redacted-blob"

    def test_google_signature_to_anthropic_signature(self):
        """Google thoughtSignature → Anthropic thinking.signature."""
        google_part = {
            "thought": True,
            "text": "Let me think...",
            "thoughtSignature": "google-verification-sig",
        }
        ir = GoogleGenerateContentOps.p_reasoning_to_ir(google_part)
        anthropic = AnthropicContentOps.ir_reasoning_to_p(ir)
        assert anthropic["type"] == "thinking"
        assert anthropic["signature"] == "google-verification-sig"
        assert anthropic["thinking"] == "Let me think..."

    def test_signature_and_redacted_data_are_independent_concepts(self):
        """signature (verification) and redacted_data (opaque content) don't interfere."""
        ir = ReasoningPart(type="reasoning", reasoning="visible text")
        ir["signature"] = "verification-only"

        anthropic = AnthropicContentOps.ir_reasoning_to_p(ir)
        assert anthropic["type"] == "thinking"
        assert anthropic["signature"] == "verification-only"
        assert anthropic["thinking"] == "visible text"

        ir2 = ReasoningPart(type="reasoning")
        ir2["redacted_data"] = "opaque-content"

        anthropic2 = AnthropicContentOps.ir_reasoning_to_p(ir2)
        assert anthropic2["type"] == "redacted_thinking"
        assert anthropic2["data"] == "opaque-content"
