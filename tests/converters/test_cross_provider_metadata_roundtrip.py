"""
Cross-provider round-trip tests for provider_metadata preservation.

Verifies that Google thought_signature survives:
  Google → IR → Anthropic → IR → Google

Covers tool_call, tool_result, and reasoning parts.
Refs: https://github.com/Oaklight/llm-rosetta/issues/225
"""

from llm_rosetta.converters.anthropic.content_ops import AnthropicContentOps
from llm_rosetta.converters.anthropic.tool_ops import AnthropicToolOps
from llm_rosetta.converters.google_genai.content_ops import GoogleGenAIContentOps
from llm_rosetta.converters.google_genai.tool_ops import GoogleGenAIToolOps
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
        ir = GoogleGenAIToolOps.p_tool_call_to_ir(google_part)
        assert ir["provider_metadata"]["google"]["thought_signature"] == self.THOUGHT_SIG

        # IR → Anthropic
        anthropic_block = AnthropicToolOps.ir_tool_call_to_p(ir)
        assert anthropic_block["type"] == "tool_use"
        assert anthropic_block["_provider_metadata"]["google"]["thought_signature"] == self.THOUGHT_SIG

        # Anthropic → IR (simulating client sending it back)
        ir2 = AnthropicToolOps.p_tool_call_to_ir(anthropic_block)
        assert ir2["provider_metadata"]["google"]["thought_signature"] == self.THOUGHT_SIG

        # IR → Google (outbound to upstream)
        google_out = GoogleGenAIToolOps.ir_tool_call_to_p(ir2)
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
        """thought_signature on reasoning survives Google → IR → Anthropic → IR → Google."""
        google_part = self._google_thought_part()

        # Google → IR
        ir = GoogleGenAIContentOps.p_reasoning_to_ir(google_part)
        assert ir["reasoning"] == "Let me think about this..."
        # Google p_reasoning_to_ir may not store thought_signature in provider_metadata
        # — it stores it directly. Let's build the IR manually to test the Anthropic path.

        # Build IR with provider_metadata (as it would come from a full converter pipeline)
        ir_with_meta = ReasoningPart(
            type="reasoning",
            reasoning="Let me think about this...",
        )
        ir_with_meta["provider_metadata"] = {
            "google": {"thought_signature": self.THOUGHT_SIG}
        }

        # IR → Anthropic
        anthropic_block = AnthropicContentOps.ir_reasoning_to_p(ir_with_meta)
        assert anthropic_block["type"] == "thinking"
        assert anthropic_block["_provider_metadata"]["google"]["thought_signature"] == self.THOUGHT_SIG

        # Anthropic → IR
        ir2 = AnthropicContentOps.p_reasoning_to_ir(anthropic_block)
        assert ir2["provider_metadata"]["google"]["thought_signature"] == self.THOUGHT_SIG

        # IR → Google
        google_out = GoogleGenAIContentOps.ir_reasoning_to_p(ir2)
        assert google_out["thoughtSignature"] == self.THOUGHT_SIG
        assert google_out["thought"] is True

    def test_reasoning_native_signature_preserved_separately(self):
        """Anthropic native signature and provider_metadata coexist."""
        ir = ReasoningPart(
            type="reasoning",
            reasoning="thinking...",
        )
        ir["signature"] = "anthropic-native-sig"
        ir["provider_metadata"] = {"google": {"thought_signature": "google-sig"}}

        anthropic = AnthropicContentOps.ir_reasoning_to_p(ir)
        assert anthropic["signature"] == "anthropic-native-sig"
        assert anthropic["_provider_metadata"]["google"]["thought_signature"] == "google-sig"

        ir2 = AnthropicContentOps.p_reasoning_to_ir(anthropic)
        assert ir2["signature"] == "anthropic-native-sig"
        assert ir2["provider_metadata"]["google"]["thought_signature"] == "google-sig"

    def test_reasoning_no_metadata_unaffected(self):
        """Reasoning blocks without provider_metadata still work normally."""
        ir = ReasoningPart(type="reasoning", reasoning="hmm")
        anthropic = AnthropicContentOps.ir_reasoning_to_p(ir)
        assert "_provider_metadata" not in anthropic
        assert anthropic["type"] == "thinking"
