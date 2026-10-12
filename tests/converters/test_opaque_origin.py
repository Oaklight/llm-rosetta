"""Cross-provider opaque reasoning/signature handling (#881, #930).

Opaque reasoning values (``signature``, ``redacted_data``, Google
``thought_signature``) are only valid for the provider that issued them.  These
tests cover the origin tracking that lets a converter drop a foreign value
(Anthropic / OpenAI) or replace it with Google's documented dummy signature.
"""

from typing import Any, cast

from llm_rosetta.converters.base.context import StreamContext
from llm_rosetta.converters.base.helpers.opaque import (
    DUMMY_THOUGHT_SIGNATURE,
    carrier_metadata,
    get_opaque_origin,
    google_thought_signature,
    inject_google_function_call_signature,
    inject_interactions_function_call_signature,
    is_own_origin,
    set_google_thought_signature,
    set_opaque_origin,
)
from llm_rosetta.converters.google_generate import GoogleGenerateConverter
from llm_rosetta.converters.google_generate.content_ops import GoogleGenerateContentOps
from llm_rosetta.converters.google_interactions import GoogleInteractionsConverter
from llm_rosetta.converters.google_interactions.content_ops import (
    GoogleInteractionsContentOps,
)
from llm_rosetta.converters.messages import MessagesConverter
from llm_rosetta.converters.messages.content_ops import MessagesContentOps
from llm_rosetta.converters.open_responses.content_ops import (
    OpenAIResponsesContentOps,
)
from llm_rosetta.types.ir import IRRequest, ToolCallPart


class TestOpaqueOriginHelper:
    def test_set_and_get(self):
        part: dict[str, Any] = {"type": "reasoning"}
        set_opaque_origin(part, signature="google_generate")
        assert get_opaque_origin(part, "signature") == "google_generate"
        assert get_opaque_origin(part, "redacted_data") is None

    def test_unknown_origin_is_own(self):
        assert is_own_origin(None, "messages") is True
        assert is_own_origin("messages", "messages") is True
        assert is_own_origin("google_generate", "messages") is False

    def test_google_family_shares_origin(self):
        assert is_own_origin("google_interactions", "google_generate") is True

    def test_existing_origin_not_overwritten(self):
        part: dict[str, Any] = {"type": "reasoning"}
        set_opaque_origin(part, signature="google_generate")
        set_opaque_origin(part, signature="messages")
        assert get_opaque_origin(part, "signature") == "google_generate"

    def test_provider_metadata_none_tolerated(self):
        part: dict[str, Any] = {"type": "reasoning", "provider_metadata": None}
        set_opaque_origin(part, signature="messages")
        assert get_opaque_origin(part, "signature") == "messages"

    def test_google_namespace_mirror(self):
        part: dict[str, Any] = {"type": "reasoning"}
        set_google_thought_signature(part, "sig")
        assert google_thought_signature(part) == "sig"

    def test_inject_google_first_function_call_only(self):
        parts = [
            {"text": "hi"},
            {"functionCall": {"name": "f"}},
            {"functionCall": {"name": "g"}},
        ]
        assert inject_google_function_call_signature(parts) is True
        assert parts[1]["thoughtSignature"] == DUMMY_THOUGHT_SIGNATURE
        assert "thoughtSignature" not in parts[2]

    def test_inject_google_skips_real_signature(self):
        parts = [{"functionCall": {"name": "f"}, "thoughtSignature": "real"}]
        assert inject_google_function_call_signature(parts) is False
        assert parts[0]["thoughtSignature"] == "real"

    def test_inject_google_no_function_call(self):
        parts = [{"text": "hi"}]
        assert inject_google_function_call_signature(parts) is False

    def test_carrier_metadata_filters_reserved_key(self):
        part = {
            "type": "reasoning",
            "provider_metadata": {
                "_opaque_origin": {"signature": "messages"},
                "google": {"thought_signature": "GSIG"},
            },
        }
        assert carrier_metadata(part) == {"google": {"thought_signature": "GSIG"}}
        assert carrier_metadata({"type": "reasoning"}) is None
        assert (
            carrier_metadata(
                {"type": "reasoning", "provider_metadata": {"_opaque_origin": {}}}
            )
            is None
        )

    def test_inject_interactions_first_step_only(self):
        steps = [
            {"type": "thought"},
            {"type": "function_call", "id": "a"},
            {"type": "function_call", "id": "b"},
        ]
        assert inject_interactions_function_call_signature(steps) is True
        assert steps[1]["signature"] == DUMMY_THOUGHT_SIGNATURE
        assert "signature" not in steps[2]


class TestCrossProviderDrop:
    def test_responses_encrypted_dropped_for_anthropic(self):
        ir = OpenAIResponsesContentOps.p_reasoning_to_ir(
            {
                "type": "reasoning",
                "id": "rs_1",
                "encrypted_content": "blob",
                "summary": [],
            }
        )
        assert ir is not None
        assert MessagesContentOps.ir_reasoning_to_p(ir) is None

    def test_anthropic_redacted_dropped_for_responses(self):
        ir = MessagesContentOps.p_redacted_thinking_to_ir(
            {"type": "redacted_thinking", "data": "blob"}
        )
        out = OpenAIResponsesContentOps.ir_reasoning_to_p(ir, output_item=True)
        assert out is not None
        assert "encrypted_content" not in out

    def test_google_signature_dropped_for_anthropic(self):
        ir = GoogleGenerateContentOps.p_reasoning_to_ir(
            {"thought": True, "text": "t", "thoughtSignature": "gsig"}
        )
        out = MessagesContentOps.ir_reasoning_to_p(ir)
        assert out is not None
        assert "signature" not in out

    def test_google_signature_dropped_for_interactions_thought(self):
        ir = GoogleGenerateContentOps.p_reasoning_to_ir(
            {"thought": True, "text": "t", "thoughtSignature": "gsig"}
        )
        out = GoogleInteractionsContentOps.ir_reasoning_to_p(ir)
        # Origin is google_generate, which the Interactions family owns.
        assert out["signature"] == "gsig"

    def test_anthropic_signature_dropped_for_google(self):
        ir = MessagesContentOps.p_reasoning_to_ir(
            {"type": "thinking", "thinking": "t", "signature": "asig"}
        )
        out = GoogleGenerateContentOps.ir_reasoning_to_p(ir)
        assert "thoughtSignature" not in out

    def test_same_format_preserved(self):
        ir = MessagesContentOps.p_reasoning_to_ir(
            {"type": "thinking", "thinking": "t", "signature": "asig"}
        )
        out = MessagesContentOps.ir_reasoning_to_p(ir)
        assert out is not None
        assert out["signature"] == "asig"

    def test_bare_ir_unknown_origin_preserved(self):
        out = MessagesContentOps.ir_reasoning_to_p(
            {"type": "reasoning", "reasoning": "t", "signature": "bare"}
        )
        assert out is not None
        assert out["signature"] == "bare"


class TestGoogleDummyInjection:
    def _ir_with_tool_call(self, tool_call: Any) -> IRRequest:
        return cast(
            IRRequest,
            {
                "model": "gemini-3-pro",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                    {"role": "assistant", "content": [tool_call]},
                ],
            },
        )

    def test_gg_injects_dummy_on_foreign_history(self):
        conv = GoogleGenerateConverter()
        ir = self._ir_with_tool_call(
            {
                "type": "tool_call",
                "tool_call_id": "c1",
                "tool_name": "f",
                "tool_input": {},
                "tool_type": "function",
            }
        )
        out, _ = conv.request_to_provider(ir)
        part = out["contents"][1]["parts"][0]
        assert part["thoughtSignature"] == DUMMY_THOUGHT_SIGNATURE

    def test_gg_preserves_real_google_signature(self):
        conv = GoogleGenerateConverter()
        ir = self._ir_with_tool_call(
            cast(
                ToolCallPart,
                {
                    "type": "tool_call",
                    "tool_call_id": "c1",
                    "tool_name": "f",
                    "tool_input": {},
                    "tool_type": "function",
                    "provider_metadata": {"google": {"thought_signature": "REAL"}},
                },
            )
        )
        out, _ = conv.request_to_provider(ir)
        part = out["contents"][1]["parts"][0]
        assert part["thoughtSignature"] == "REAL"

    def test_gi_injects_dummy_on_foreign_history(self):
        conv = GoogleInteractionsConverter()
        ir = self._ir_with_tool_call(
            {
                "type": "tool_call",
                "tool_call_id": "c1",
                "tool_name": "f",
                "tool_input": {},
                "tool_type": "function",
            }
        )
        out, _ = conv.request_to_provider(ir)
        step = out["input"][1]
        assert step["signature"] == DUMMY_THOUGHT_SIGNATURE

    def test_no_dummy_on_turn_without_tool_call(self):
        conv = GoogleGenerateConverter()
        ir = cast(
            IRRequest,
            {
                "model": "gemini-3-pro",
                "messages": [
                    {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "hello"}],
                    },
                ],
            },
        )
        out, _ = conv.request_to_provider(ir)
        parts = out["contents"][1]["parts"]
        assert all("thoughtSignature" not in p for p in parts)


class TestStreamingOpaqueGating:
    def test_anthropic_inbound_stamps_origin(self):
        conv = MessagesConverter()
        events = cast(
            list[Any],
            conv.stream_response_from_provider(
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "signature_delta", "signature": "sig"},
                }
            ),
        )
        assert get_opaque_origin(events[0], "signature") == "messages"

    def test_anthropic_reasoning_block_does_not_leak_origin(self):
        ir = GoogleGenerateContentOps.p_reasoning_to_ir(
            {"thought": True, "text": "t", "thoughtSignature": "GSIG"}
        )
        block = MessagesContentOps.ir_reasoning_to_p(ir)
        assert block is not None
        assert "_opaque_origin" not in block["_llm_rosetta_provider_metadata"]
        assert (
            block["_llm_rosetta_provider_metadata"]["google"]["thought_signature"]
            == "GSIG"
        )

    def test_anthropic_outbound_keeps_text_when_dropping_foreign_signature(self):
        conv = MessagesConverter()
        result = conv.stream_response_to_provider(
            cast(
                Any,
                {
                    "type": "reasoning_delta",
                    "reasoning": "visible text",
                    "signature": "foreign",
                    "provider_metadata": {
                        "_opaque_origin": {"signature": "google_generate"}
                    },
                },
            )
        )
        chunks = result if isinstance(result, list) else [result]
        deltas = [c["delta"] for c in chunks if isinstance(c, dict) and "delta" in c]
        assert deltas, f"no delta emitted: {chunks}"
        assert deltas[-1]["type"] == "thinking_delta"
        assert deltas[-1]["thinking"] == "visible text"
        assert all(d.get("type") != "signature_delta" for d in deltas)

    def test_anthropic_outbound_drops_foreign_signature(self):
        conv = MessagesConverter()
        result = conv.stream_response_to_provider(
            cast(
                Any,
                {
                    "type": "reasoning_delta",
                    "reasoning": "",
                    "signature": "foreign",
                    "provider_metadata": {
                        "_opaque_origin": {"signature": "google_generate"}
                    },
                },
            )
        )
        assert result == {}

    def test_gi_outbound_drops_foreign_signature(self):
        conv = GoogleInteractionsConverter()
        result = conv.stream_response_to_provider(
            cast(
                Any,
                {
                    "type": "reasoning_delta",
                    "reasoning": "",
                    "signature": "foreign",
                    "block_index": 0,
                    "provider_metadata": {"_opaque_origin": {"signature": "messages"}},
                },
            )
        )
        chunks = result if isinstance(result, list) else [result]
        # The foreign signature delta is never emitted (the step.start
        # preamble the GI override emits for any delta is unrelated).
        assert all(
            chunk.get("delta", {}).get("type") != "thought_signature"
            for chunk in chunks
        )

    def test_gg_outbound_carries_google_signature(self):
        conv = GoogleGenerateConverter()
        ctx = StreamContext()
        conv.stream_response_to_provider(
            cast(
                Any,
                {
                    "type": "tool_call_start",
                    "tool_call_id": "c1",
                    "tool_name": "f",
                    "choice_index": 0,
                    "provider_metadata": {"google": {"thought_signature": "GS"}},
                },
            ),
            context=ctx,
        )
        conv.stream_response_to_provider(
            cast(
                Any,
                {
                    "type": "tool_call_delta",
                    "tool_call_id": "c1",
                    "arguments_delta": "{}",
                    "choice_index": 0,
                },
            ),
            context=ctx,
        )
        chunks = conv.stream_response_to_provider(
            cast(
                Any,
                {
                    "type": "finish",
                    "finish_reason": {"reason": "tool_calls"},
                    "choice_index": 0,
                },
            ),
            context=ctx,
        )
        parts: list[dict[str, Any]] = []
        for chunk in chunks if isinstance(chunks, list) else [chunks]:
            for cand in chunk.get("candidates", []):
                parts.extend(cand.get("content", {}).get("parts", []))
        assert parts[0]["thoughtSignature"] == "GS"

    def test_gg_outbound_injects_dummy_without_signature(self):
        conv = GoogleGenerateConverter()
        ctx = StreamContext()
        conv.stream_response_to_provider(
            cast(
                Any,
                {
                    "type": "tool_call_start",
                    "tool_call_id": "c1",
                    "tool_name": "f",
                    "choice_index": 0,
                },
            ),
            context=ctx,
        )
        conv.stream_response_to_provider(
            cast(
                Any,
                {
                    "type": "tool_call_delta",
                    "tool_call_id": "c1",
                    "arguments_delta": "{}",
                    "choice_index": 0,
                },
            ),
            context=ctx,
        )
        chunks = conv.stream_response_to_provider(
            cast(
                Any,
                {
                    "type": "finish",
                    "finish_reason": {"reason": "tool_calls"},
                    "choice_index": 0,
                },
            ),
            context=ctx,
        )
        parts: list[dict[str, Any]] = []
        for chunk in chunks if isinstance(chunks, list) else [chunks]:
            for cand in chunk.get("candidates", []):
                parts.extend(cand.get("content", {}).get("parts", []))
        assert parts[0]["thoughtSignature"] == DUMMY_THOUGHT_SIGNATURE
