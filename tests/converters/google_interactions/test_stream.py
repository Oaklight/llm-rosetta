"""Tests for Google Interactions streaming conversion."""

from llm_rosetta.converters.google_interactions import GoogleInteractionsConverter
from llm_rosetta.converters.base.context import StreamContext
from typing import cast

from llm_rosetta.types.ir.stream import (
    ContentBlockEndEvent,
    ContentBlockStartEvent,
    FinishEvent,
    ReasoningDeltaEvent,
    TextDeltaEvent,
    ToolCallDeltaEvent,
    ToolCallStartEvent,
)


class TestStreamChunkToIR:
    def setup_method(self):
        self.converter = GoogleInteractionsConverter()

    def test_interaction_created(self):
        chunk = {
            "data": {
                "event_type": "interaction.created",
                "interaction": {
                    "id": "int_123",
                    "model": "gemini-3.6-flash",
                    "status": "in_progress",
                },
            }
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert len(events) == 1
        assert events[0]["type"] == "stream_start"
        assert events[0]["response_id"] == "int_123"
        assert events[0]["model"] == "gemini-3.6-flash"

    def test_step_start_model_output(self):
        chunk = {
            "data": {
                "event_type": "step.start",
                "index": 0,
                "step": {"type": "model_output"},
            }
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert len(events) == 1
        assert events[0]["type"] == "content_block_start"
        assert events[0]["block_index"] == 0
        assert events[0]["block_type"] == "text"

    def test_step_start_thought(self):
        chunk = {
            "data": {
                "event_type": "step.start",
                "index": 0,
                "step": {"type": "thought"},
            }
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert events[0]["block_type"] == "thinking"  # ty: ignore

    def test_step_start_function_call(self):
        chunk = {
            "data": {
                "event_type": "step.start",
                "index": 1,
                "step": {
                    "type": "function_call",
                    "id": "call_123",
                    "name": "get_weather",
                },
            }
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert len(events) == 2
        assert events[0]["type"] == "content_block_start"
        assert events[0]["block_type"] == "tool_use"
        assert events[1]["type"] == "tool_call_start"
        assert events[1]["tool_call_id"] == "call_123"
        assert events[1]["tool_name"] == "get_weather"

    def test_step_start_function_call_with_signature(self):
        chunk = {
            "data": {
                "event_type": "step.start",
                "index": 1,
                "step": {
                    "type": "function_call",
                    "id": "call_sig",
                    "name": "get_weather",
                    "signature": "sig_stream",
                },
            }
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert len(events) == 2
        tc_start = events[1]
        assert tc_start["type"] == "tool_call_start"
        assert (
            tc_start["provider_metadata"]["google"]["thought_signature"] == "sig_stream"
        )

    def test_step_start_function_call_no_signature(self):
        chunk = {
            "data": {
                "event_type": "step.start",
                "index": 1,
                "step": {
                    "type": "function_call",
                    "id": "call_no",
                    "name": "fn",
                },
            }
        }
        events = self.converter.stream_response_from_provider(chunk)
        tc_start = events[1]
        assert "provider_metadata" not in tc_start

    def test_step_delta_text(self):
        chunk = {
            "data": {
                "event_type": "step.delta",
                "index": 0,
                "delta": {"type": "text", "text": "Hello"},
            }
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert len(events) == 1
        assert events[0]["type"] == "text_delta"
        assert events[0]["text"] == "Hello"
        assert events[0]["block_index"] == 0

    def test_step_delta_thought_summary(self):
        chunk = {
            "data": {
                "event_type": "step.delta",
                "index": 0,
                "delta": {"type": "thought_summary", "text": "Thinking..."},
            }
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert len(events) == 1
        assert events[0]["type"] == "reasoning_delta"
        assert events[0]["reasoning"] == "Thinking..."

    def test_step_delta_thought_signature(self):
        chunk = {
            "data": {
                "event_type": "step.delta",
                "index": 0,
                "delta": {"type": "thought_signature", "signature": "sig_abc"},
            }
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert len(events) == 1
        assert events[0]["type"] == "reasoning_delta"
        assert events[0]["signature"] == "sig_abc"

    def test_step_delta_server_call_signature(self):
        context = StreamContext()
        self.converter.stream_response_from_provider(
            {
                "event_type": "step.start",
                "index": 1,
                "step": {"type": "google_search_call", "id": "call_sig"},
            },
            context,
        )
        events = self.converter.stream_response_from_provider(
            {
                "event_type": "step.delta",
                "index": 1,
                "delta": {
                    "type": "google_search_call",
                    "signature": "sig_delta",
                    "arguments": {"queries": ["mars"]},
                },
            },
            context,
        )
        # The delta carries the signature; step.start's empty signature must
        # not be required for replay.
        delta = cast(ToolCallDeltaEvent, events[0])
        assert delta["signature"] == "sig_delta"
        assert delta["provider_metadata"]["google"]["thought_signature"] == "sig_delta"
        assert (
            delta["provider_metadata"]["_opaque_origin"]["signature"]
            == "google_interactions"
        )

    def test_step_delta_arguments(self):
        chunk = {
            "data": {
                "event_type": "step.delta",
                "index": 1,
                "delta": {
                    "type": "arguments",
                    "call_id": "call_123",
                    "arguments": '{"loc',
                },
            }
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert len(events) == 1
        assert events[0]["type"] == "tool_call_delta"
        assert events[0]["arguments_delta"] == '{"loc'

    def test_step_stop(self):
        chunk = {
            "data": {
                "event_type": "step.stop",
                "index": 0,
            }
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert len(events) == 1
        assert events[0]["type"] == "content_block_end"
        assert events[0]["block_index"] == 0

    def test_step_stop_with_usage(self):
        chunk = {
            "data": {
                "event_type": "step.stop",
                "index": 0,
                "step_usage": {
                    "total_input_tokens": 10,
                    "total_output_tokens": 5,
                    "total_tokens": 15,
                },
            }
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert len(events) == 2
        assert events[0]["type"] == "content_block_end"
        assert events[1]["type"] == "usage"
        assert events[1]["usage"]["prompt_tokens"] == 10

    def test_interaction_completed(self):
        chunk = {
            "data": {
                "event_type": "interaction.completed",
                "interaction": {
                    "id": "int_123",
                    "model": "gemini-3.6-flash",
                    "status": "completed",
                    "usage": {
                        "total_input_tokens": 20,
                        "total_output_tokens": 10,
                        "total_tokens": 30,
                    },
                },
            }
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert len(events) == 3
        assert events[0]["type"] == "finish"
        assert events[0]["finish_reason"]["reason"] == "stop"
        assert events[1]["type"] == "usage"
        assert events[1]["usage"]["total_tokens"] == 30
        assert events[2]["type"] == "stream_end"

    def test_ir_tool_call_start_to_p_with_signature(self):
        """ToolCallStartEvent with provider_metadata → exactly one enriched
        step.start carrying the signature."""
        event = ToolCallStartEvent(
            type="tool_call_start",
            tool_call_id="call_out",
            tool_name="fn",
            tool_call_index=0,
            provider_metadata={"google": {"thought_signature": "sig_outbound"}},
        )
        result = self.converter.stream_response_to_provider(event, StreamContext())
        steps = result if isinstance(result, list) else [result]
        signed = [
            s
            for s in steps
            if s.get("event_type") == "step.start"
            and s.get("step", {}).get("signature")
        ]
        assert len(signed) == 1
        assert signed[0]["step"]["signature"] == "sig_outbound"
        assert signed[0]["step"]["id"] == "call_out"
        assert signed[0]["step"]["name"] == "fn"

    def test_ir_tool_call_start_to_p_no_signature(self):
        """ToolCallStartEvent without provider_metadata → no step.start
        carries a signature."""
        event = ToolCallStartEvent(
            type="tool_call_start",
            tool_call_id="call_plain",
            tool_name="fn",
            tool_call_index=0,
        )
        result = self.converter.stream_response_to_provider(event, StreamContext())
        steps = result if isinstance(result, list) else [result]
        assert all(
            "signature" not in s.get("step", {})
            for s in steps
            if s.get("event_type") == "step.start"
        )

    def test_function_call_roundtrip_signature_on_enriched_step_only(self):
        """Inbound step.start → IR → outbound: the signature lands on the
        enriched step.start (with id/name) and on no other.

        GI emits two same-index ``step.start`` for one function_call — a bare
        one from ``ContentBlockStartEvent`` plus the enriched one from
        ``ToolCallStartEvent``. Asserting the signature is present on exactly
        one (the enriched) guards against it silently moving to the bare one.
        """
        chunk = {
            "data": {
                "event_type": "step.start",
                "index": 1,
                "step": {
                    "type": "function_call",
                    "id": "call_rt",
                    "name": "fn",
                    "signature": "sig_rt",
                },
            }
        }
        ir_events = self.converter.stream_response_from_provider(chunk)
        ctx = StreamContext()
        out: list = []
        for e in ir_events:
            r = self.converter.stream_response_to_provider(e, ctx)
            out.extend(r if isinstance(r, list) else [r])
        step_starts = [s for s in out if s.get("event_type") == "step.start"]
        assert len(step_starts) == 1
        signed = [s for s in step_starts if s.get("step", {}).get("signature")]
        assert len(signed) == 1
        assert signed[0]["step"]["signature"] == "sig_rt"
        assert signed[0]["step"]["id"] == "call_rt"
        assert signed[0]["step"]["name"] == "fn"

    def test_interaction_completed_requires_action(self):
        chunk = {
            "data": {
                "event_type": "interaction.completed",
                "interaction": {
                    "status": "requires_action",
                },
            }
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert events[0]["type"] == "finish"
        assert events[0]["finish_reason"]["reason"] == "tool_calls"

    def test_full_streaming_sequence(self):
        """Simulate a complete streaming sequence."""
        chunks = [
            {
                "data": {
                    "event_type": "interaction.created",
                    "interaction": {"id": "i1", "model": "m"},
                }
            },
            {
                "data": {
                    "event_type": "step.start",
                    "index": 0,
                    "step": {"type": "model_output"},
                }
            },
            {
                "data": {
                    "event_type": "step.delta",
                    "index": 0,
                    "delta": {"type": "text", "text": "Hi"},
                }
            },
            {
                "data": {
                    "event_type": "step.delta",
                    "index": 0,
                    "delta": {"type": "text", "text": " there"},
                }
            },
            {"data": {"event_type": "step.stop", "index": 0}},
            {
                "data": {
                    "event_type": "interaction.completed",
                    "interaction": {
                        "status": "completed",
                        "usage": {
                            "total_input_tokens": 5,
                            "total_output_tokens": 3,
                            "total_tokens": 8,
                        },
                    },
                }
            },
        ]
        all_events = []
        for chunk in chunks:
            events = self.converter.stream_response_from_provider(chunk)
            all_events.extend(events)

        types = [e["type"] for e in all_events]
        assert types == [
            "stream_start",
            "content_block_start",
            "text_delta",
            "text_delta",
            "content_block_end",
            "finish",
            "usage",
            "stream_end",
        ]


class TestStreamIRToProvider:
    """Outbound IR → GI streaming: one step.start per step (no duplicates).

    Regression tests for #931 — the ContentBlockStart handler did not mark
    the step as opened, so the delta fallback synthesized a second, wrong
    index/type ``step.start``, and a ``tool_use`` block emitted a bare one
    on top of the enriched ``ToolCallStartEvent``.
    """

    def setup_method(self):
        self.converter = GoogleInteractionsConverter()

    def _run(self, events):
        ctx = StreamContext()
        out: list = []
        for e in events:
            r = self.converter.stream_response_to_provider(e, ctx)
            out.extend(r if isinstance(r, list) else ([r] if r else []))
        return out

    def test_text_stream_single_step_start(self):
        out = self._run(
            [
                ContentBlockStartEvent(
                    type="content_block_start", block_index=0, block_type="text"
                ),
                TextDeltaEvent(type="text_delta", text="hi", block_index=0),
                TextDeltaEvent(type="text_delta", text="!", block_index=0),
                ContentBlockEndEvent(type="content_block_end", block_index=0),
            ]
        )
        starts = [c for c in out if c.get("event_type") == "step.start"]
        assert len(starts) == 1
        assert starts[0] == {
            "event_type": "step.start",
            "index": 0,
            "step": {"type": "model_output"},
        }
        assert [c["event_type"] for c in out] == [
            "step.start",
            "step.delta",
            "step.delta",
            "step.stop",
        ]

    def test_tool_stream_single_enriched_step_start(self):
        out = self._run(
            [
                ContentBlockStartEvent(
                    type="content_block_start", block_index=1, block_type="tool_use"
                ),
                ToolCallStartEvent(
                    type="tool_call_start",
                    tool_call_id="c1",
                    tool_name="fn",
                    tool_call_index=1,
                    provider_metadata={"google": {"thought_signature": "sig"}},
                ),
                ToolCallDeltaEvent(
                    type="tool_call_delta",
                    tool_call_id="c1",
                    arguments_delta="{}",
                    block_index=1,
                ),
                ContentBlockEndEvent(type="content_block_end", block_index=1),
            ]
        )
        starts = [c for c in out if c.get("event_type") == "step.start"]
        assert len(starts) == 1
        assert starts[0]["index"] == 1
        assert starts[0]["step"] == {
            "type": "function_call",
            "id": "c1",
            "name": "fn",
            "signature": "sig",
        }
        assert [c["event_type"] for c in out] == [
            "step.start",
            "step.delta",
            "step.stop",
        ]

    def test_delta_without_block_start_synthesizes_once(self):
        """The fallback fires exactly once when no ContentBlockStart is seen.

        Characterisation of behaviour this fix does not change (the fallback
        guard is untouched); kept to catch a future refactor that drops the
        fallback entirely, not as #931 regression evidence.
        """
        out = self._run(
            [
                TextDeltaEvent(type="text_delta", text="a", block_index=0),
                TextDeltaEvent(type="text_delta", text="b", block_index=0),
            ]
        )
        starts = [c for c in out if c.get("event_type") == "step.start"]
        assert len(starts) == 1
        assert starts[0] == {
            "event_type": "step.start",
            "index": 0,
            "step": {"type": "model_output"},
        }

    def test_tool_opener_reuses_block_index(self):
        """Opener uses the block index, not the 0-based tool_call_index.

        ``messages`` (Anthropic) emits ``ContentBlockStart(block_index=N)`` +
        ``ToolCallStartEvent(tool_call_index=0-based)`` while the deltas carry
        ``block_index``; the opener must match the deltas' index.
        """
        out = self._run(
            [
                ContentBlockStartEvent(
                    type="content_block_start", block_index=2, block_type="tool_use"
                ),
                ToolCallStartEvent(
                    type="tool_call_start",
                    tool_call_id="c1",
                    tool_name="fn",
                    tool_call_index=0,
                ),
                ToolCallDeltaEvent(
                    type="tool_call_delta",
                    tool_call_id="c1",
                    arguments_delta="{}",
                    block_index=2,
                ),
                ContentBlockEndEvent(type="content_block_end", block_index=2),
            ]
        )
        starts = [c for c in out if c.get("event_type") == "step.start"]
        deltas = [c for c in out if c.get("event_type") == "step.delta"]
        assert len(starts) == 1
        assert starts[0]["index"] == 2
        assert deltas[0]["index"] == 2

    def test_content_block_end_suppresses_finish_step_stop(self):
        """ContentBlockEnd + Finish emit exactly one step.stop (not two)."""
        out = self._run(
            [
                ContentBlockStartEvent(
                    type="content_block_start", block_index=0, block_type="text"
                ),
                TextDeltaEvent(type="text_delta", text="hi", block_index=0),
                ContentBlockEndEvent(type="content_block_end", block_index=0),
                FinishEvent(type="finish", finish_reason={"reason": "stop"}),
            ]
        )
        stops = [c for c in out if c.get("event_type") == "step.stop"]
        assert len(stops) == 1
        assert stops[0]["index"] == 0

    def test_finish_synthesizes_step_stop_when_no_block_end(self):
        """Finish still synthesizes a step.stop when no ContentBlockEnd came."""
        out = self._run(
            [
                TextDeltaEvent(type="text_delta", text="hi", block_index=0),
                FinishEvent(type="finish", finish_reason={"reason": "stop"}),
            ]
        )
        stops = [c for c in out if c.get("event_type") == "step.stop"]
        assert len(stops) == 1


class TestReasoningSignatureOrigin:
    """The reasoning-signature arm keeps its own origin and drops a foreign one."""

    def setup_method(self):
        self.converter = GoogleInteractionsConverter()

    def _signed(self, origin: str | None) -> ReasoningDeltaEvent:
        event = ReasoningDeltaEvent(
            type="reasoning_delta", reasoning="", signature="SIG-REASON", block_index=0
        )
        if origin is not None:
            event["provider_metadata"] = {"_opaque_origin": {"signature": origin}}
        return event

    def _deltas(self, event: ReasoningDeltaEvent) -> list:
        out = self.converter.stream_response_to_provider(event, StreamContext())
        out = out if isinstance(out, list) else ([out] if out else [])
        return [o["delta"] for o in out if o.get("event_type") == "step.delta"]

    def test_reader_stamps_the_converter_tag(self):
        chunk = {
            "data": {
                "event_type": "step.delta",
                "index": 0,
                "delta": {"type": "thought_signature", "signature": "SIG-REASON"},
            }
        }
        (event,) = self.converter.stream_response_from_provider(chunk)
        reasoning = cast(ReasoningDeltaEvent, event)
        origin = reasoning["provider_metadata"]["_opaque_origin"]["signature"]
        assert origin == self.converter._CONVERTER_TAG

    def test_own_origin_signature_is_written(self):
        assert self._deltas(self._signed("google_interactions")) == [
            {"type": "thought_signature", "signature": "SIG-REASON"}
        ]

    def test_untagged_signature_is_written(self):
        assert self._deltas(self._signed(None)) == [
            {"type": "thought_signature", "signature": "SIG-REASON"}
        ]

    def test_foreign_origin_signature_is_dropped(self):
        assert self._deltas(self._signed("messages")) == []
