"""
Cross-provider round-trip tests for provider_metadata preservation.

Verifies that Google thought_signature survives:
  Google → IR → Anthropic → IR → Google

Covers tool_call, tool_result, and reasoning parts.
Refs: https://github.com/Oaklight/llm-rosetta/issues/225
"""

from llm_rosetta.converters.messages.content_ops import MessagesContentOps
from llm_rosetta.converters.messages.tool_ops import MessagesToolOps
from llm_rosetta.converters.google_generate.content_ops import GoogleGenerateContentOps
from llm_rosetta.converters.google_generate.tool_ops import GoogleGenerateToolOps
from llm_rosetta.converters.google_interactions.tool_ops import (
    GoogleInteractionsToolOps,
)
from llm_rosetta.converters.open_responses.tool_ops import OpenResponsesToolOps
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
        anthropic_block = MessagesToolOps.ir_tool_call_to_p(ir)
        assert anthropic_block["type"] == "tool_use"
        assert (
            anthropic_block["_llm_rosetta_provider_metadata"]["google"][
                "thought_signature"
            ]
            == self.THOUGHT_SIG
        )

        # Anthropic → IR (simulating client sending it back)
        ir2 = MessagesToolOps.p_tool_call_to_ir(anthropic_block)
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
        anthropic = MessagesToolOps.ir_tool_call_to_p(ir)
        assert "_llm_rosetta_provider_metadata" not in anthropic

        ir2 = MessagesToolOps.p_tool_call_to_ir(anthropic)
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

        anthropic = MessagesToolOps.ir_tool_result_to_p(ir)
        assert (
            anthropic["_llm_rosetta_provider_metadata"]["google"]["some_field"]
            == "value"
        )

        ir2 = MessagesToolOps.p_tool_result_to_ir(anthropic)
        assert ir2["provider_metadata"]["google"]["some_field"] == "value"

    def test_tool_result_no_metadata_unaffected(self):
        """Tool results without provider_metadata still work normally."""
        ir = ToolResultPart(
            type="tool_result",
            tool_call_id="call_2",
            result="data",
            is_error=False,
        )
        anthropic = MessagesToolOps.ir_tool_result_to_p(ir)
        assert "_llm_rosetta_provider_metadata" not in anthropic


class TestGoogleInteractionsToolCallSignatureRoundTrip:
    """GI ↔ GG round-trip for function_call signature via provider_metadata.google."""

    THOUGHT_SIG = "eyJhbGciOiAiUlMyNTYifQ.gi-function-call-sig"

    def test_gi_to_gg_signature_survives(self):
        """GI function_call signature → IR → GG functionCall thoughtSignature."""
        gi_step = {
            "type": "function_call",
            "id": "call_1",
            "name": "get_weather",
            "arguments": {"loc": "SF"},
            "signature": self.THOUGHT_SIG,
        }
        ir = GoogleInteractionsToolOps.p_function_call_to_ir(gi_step)
        assert (
            ir["provider_metadata"]["google"]["thought_signature"] == self.THOUGHT_SIG
        )

        gg_part = GoogleGenerateToolOps.ir_tool_call_to_p(ir)
        assert gg_part["thoughtSignature"] == self.THOUGHT_SIG

    def test_gg_to_gi_signature_survives(self):
        """GG functionCall thoughtSignature → IR → GI function_call signature."""
        gg_part = {
            "functionCall": {"name": "search", "args": {"q": "test"}},
            "thoughtSignature": self.THOUGHT_SIG,
        }
        ir = GoogleGenerateToolOps.p_tool_call_to_ir(gg_part)
        assert (
            ir["provider_metadata"]["google"]["thought_signature"] == self.THOUGHT_SIG
        )

        gi_step = GoogleInteractionsToolOps.ir_function_call_to_p(ir)
        assert gi_step["signature"] == self.THOUGHT_SIG

    def test_gi_to_messages_to_gi_signature_survives(self):
        """GI → IR → Messages → IR → GI: signature carried via _llm_rosetta_provider_metadata."""
        gi_step = {
            "type": "function_call",
            "id": "call_cross",
            "name": "fn",
            "arguments": {},
            "signature": self.THOUGHT_SIG,
        }
        ir = GoogleInteractionsToolOps.p_function_call_to_ir(gi_step)

        messages_block = MessagesToolOps.ir_tool_call_to_p(ir)
        assert (
            messages_block["_llm_rosetta_provider_metadata"]["google"][
                "thought_signature"
            ]
            == self.THOUGHT_SIG
        )

        ir2 = MessagesToolOps.p_tool_call_to_ir(messages_block)
        assert (
            ir2["provider_metadata"]["google"]["thought_signature"] == self.THOUGHT_SIG
        )

        gi_back = GoogleInteractionsToolOps.ir_function_call_to_p(ir2)
        assert gi_back["signature"] == self.THOUGHT_SIG

    def test_no_signature_no_metadata(self):
        gi_step = {
            "type": "function_call",
            "id": "call_plain",
            "name": "fn",
            "arguments": {},
        }
        ir = GoogleInteractionsToolOps.p_function_call_to_ir(gi_step)
        assert "provider_metadata" not in ir

        gi_back = GoogleInteractionsToolOps.ir_function_call_to_p(ir)
        assert "signature" not in gi_back


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

        # IR → Anthropic: the foreign signature is dropped from the flat
        # field (#881) — Anthropic would reject a Google signature — but it
        # is ferried through provider_metadata so the value survives.
        anthropic_block = MessagesContentOps.ir_reasoning_to_p(ir)
        assert anthropic_block is not None
        assert anthropic_block["type"] == "thinking"
        assert "signature" not in anthropic_block
        assert (
            anthropic_block["_llm_rosetta_provider_metadata"]["google"][
                "thought_signature"
            ]
            == self.THOUGHT_SIG
        )

        # Anthropic → IR: the signature comes back from the Google namespace.
        ir2 = MessagesContentOps.p_reasoning_to_ir(anthropic_block)
        assert (
            ir2["provider_metadata"]["google"]["thought_signature"] == self.THOUGHT_SIG
        )

        # IR → Google: thoughtSignature restored from the Google namespace.
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

        anthropic = MessagesContentOps.ir_reasoning_to_p(ir)

        assert anthropic is not None
        assert anthropic["signature"] == "some-verification-sig"

        ir2 = MessagesContentOps.p_reasoning_to_ir(anthropic)
        assert ir2["signature"] == "some-verification-sig"

    def test_reasoning_no_metadata_unaffected(self):
        """Reasoning blocks without signature still work normally."""
        ir = ReasoningPart(type="reasoning", reasoning="hmm")
        anthropic = MessagesContentOps.ir_reasoning_to_p(ir)
        assert anthropic is not None
        assert "signature" not in anthropic
        assert "_llm_rosetta_provider_metadata" not in anthropic
        assert anthropic["type"] == "thinking"


class TestToolCallCarrierRoundTrip:
    """The ``open_responses`` tool-call carrier now ferries provider_metadata.

    Formerly an ``xfail`` for the missing carrier; fixed in #930 (folded into
    #881), so this is a plain regression test.
    """

    def test_open_responses_preserves_tool_call_provider_metadata(self):
        ir = ToolCallPart(
            type="tool_call",
            tool_call_id="c1",
            tool_name="fn",
            tool_input={},
            provider_metadata={"google": {"thought_signature": "sig"}},
        )
        block = OpenResponsesToolOps.ir_tool_call_to_p(ir)
        back = OpenResponsesToolOps.p_tool_call_to_ir(block)
        assert (
            back.get("provider_metadata", {}).get("google", {}).get("thought_signature")
            == "sig"
        )
