"""
Cross-provider round-trip tests for provider_metadata preservation.

Verifies that Google thought_signature survives:
  Google → IR → Anthropic → IR → Google

Covers tool_call, tool_result, and reasoning parts.
Refs: https://github.com/Oaklight/llm-rosetta/issues/225
"""

from llm_rosetta.converters.anthropic.content_ops import AnthropicContentOps
from llm_rosetta.converters.anthropic.tool_ops import AnthropicToolOps
from llm_rosetta.converters.google_generate.content_ops import GoogleGenerateContentOps
from llm_rosetta.converters.google_generate.tool_ops import GoogleGenerateToolOps
from llm_rosetta.types.ir import ToolCallPart, ToolResultPart, ReasoningPart


class TestGoogleAnthropicToolCallRoundTrip:
    """Google → Anthropic → Google round-trip for tool calls with thought_signature."""

    THOUGHT_SIG = "eyJhbGciOiAiUlMyNTYiLCAidHlwIjogIkpXVCJ9.test-signature"

    def _google_provider_tool_call(self) -> dict:
        """A Google functionCall Part with thoughtSignature."""
        return {
            "functionCall": {
                "name": "Glob",
                "args": {"pattern": "*.py"},
            },
            "thoughtSignature": self.THOUGHT_SIG,
        }

    def test_tool_call_google_to_anthropic_to_google(self):
        """thought_signature survives Google → IR → Anthropic → IR → Google."""
        google_part = self._google_provider_tool_call()

        # Google → IR
        ir = GoogleGenerateToolOps.p_tool_call_to_ir(google_part)
        assert (
            ir["provider_metadata"]["google"]["thought_signature"] == self.THOUGHT_SIG
        )

        # IR → Anthropic
        anthropic_block = AnthropicToolOps.ir_tool_call_to_p(ir)
        assert anthropic_block["type"] == "tool_use"
        assert (
            anthropic_block["_provider_metadata"]["google"]["thought_signature"]
            == self.THOUGHT_SIG
        )

        # Anthropic → IR (simulating client sending it back)
        ir2 = AnthropicToolOps.p_tool_call_to_ir(anthropic_block)
        assert (
            ir2["provider_metadata"]["google"]["thought_signature"] == self.THOUGHT_SIG
        )

        # IR → Google (outbound to upstream)
        google_out = GoogleGenerateToolOps.ir_tool_call_to_p(ir2)
        assert google_out["thoughtSignature"] == self.THOUGHT_SIG

    def test_tool_call_no_metadata_unaffected(self):
        """Tool calls without provider_metadata still work normally."""
        ir = ToolCallPart(
            type="tool_call",
            tool_call_id="call_1",
            tool_name="test",
            tool_input={"x": 1},
            tool_type="function",
        )
        anthropic = AnthropicToolOps.ir_tool_call_to_p(ir)
        assert "_provider_metadata" not in anthropic

        ir2 = AnthropicToolOps.p_tool_call_to_ir(anthropic)
        assert "provider_metadata" not in ir2


class TestGoogleAnthropicToolResultRoundTrip:
    """Google → Anthropic → Google round-trip for tool results with provider_metadata."""

    def test_tool_result_provider_metadata_roundtrip(self):
        """provider_metadata on tool_result survives Anthropic round-trip."""
        ir = ToolResultPart(
            type="tool_result",
            tool_call_id="call_1",
            result="ok",
            is_error=False,
        )
        ir["provider_metadata"] = {"google": {"some_field": "value"}}

        anthropic = AnthropicToolOps.ir_tool_result_to_p(ir)
        assert anthropic["_provider_metadata"]["google"]["some_field"] == "value"

        ir2 = AnthropicToolOps.p_tool_result_to_ir(anthropic)
        assert ir2["provider_metadata"]["google"]["some_field"] == "value"

    def test_tool_result_no_metadata_unaffected(self):
        """Tool results without provider_metadata still work normally."""
        ir = ToolResultPart(
            type="tool_result",
            tool_call_id="call_2",
            result="data",
            is_error=False,
        )
        anthropic = AnthropicToolOps.ir_tool_result_to_p(ir)
        assert "_provider_metadata" not in anthropic


class TestGoogleAnthropicReasoningRoundTrip:
    """Google → Anthropic → Google round-trip for reasoning with thought_signature."""

    THOUGHT_SIG = "eyJ0aGlua2luZyI6IHRydWV9.reasoning-sig"

    def _google_thought_part(self) -> dict:
        """A Google thought Part with thoughtSignature."""
        return {
            "thought": True,
            "text": "Let me think about this...",
            "thoughtSignature": self.THOUGHT_SIG,
        }

    def test_reasoning_google_to_anthropic_to_google(self):
        """thought_signature on reasoning survives Google → IR → Anthropic → IR → Google.

        After #857, Google generate maps thoughtSignature to ReasoningPart.signature
        (same as Google Interactions and Anthropic). The signature field is shared,
        so it round-trips through Anthropic naturally.
        """
        google_part = self._google_thought_part()

        # Google → IR: thoughtSignature → signature
        ir = GoogleGenerateContentOps.p_reasoning_to_ir(google_part)
        assert ir["reasoning"] == "Let me think about this..."
        assert ir["signature"] == self.THOUGHT_SIG

        # IR → Anthropic: signature → thinking.signature
        anthropic_block = AnthropicContentOps.ir_reasoning_to_p(ir)
        assert anthropic_block["type"] == "thinking"
        assert anthropic_block["signature"] == self.THOUGHT_SIG

        # Anthropic → IR: thinking.signature → signature
        ir2 = AnthropicContentOps.p_reasoning_to_ir(anthropic_block)
        assert ir2["signature"] == self.THOUGHT_SIG

        # IR → Google: signature → thoughtSignature
        google_out = GoogleGenerateContentOps.ir_reasoning_to_p(ir2)
        assert google_out["thoughtSignature"] == self.THOUGHT_SIG
        assert google_out["thought"] is True

    def test_reasoning_signature_round_trip_single_provider(self):
        """Signature round-trips within a single provider."""
        ir = ReasoningPart(
            type="reasoning",
            reasoning="thinking...",
        )
        ir["signature"] = "some-verification-sig"

        anthropic = AnthropicContentOps.ir_reasoning_to_p(ir)
        assert anthropic["signature"] == "some-verification-sig"

        ir2 = AnthropicContentOps.p_reasoning_to_ir(anthropic)
        assert ir2["signature"] == "some-verification-sig"

    def test_reasoning_no_metadata_unaffected(self):
        """Reasoning blocks without signature still work normally."""
        ir = ReasoningPart(type="reasoning", reasoning="hmm")
        anthropic = AnthropicContentOps.ir_reasoning_to_p(ir)
        assert "signature" not in anthropic
        assert "_provider_metadata" not in anthropic
        assert anthropic["type"] == "thinking"
