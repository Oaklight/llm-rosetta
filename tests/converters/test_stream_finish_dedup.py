"""Tests for duplicate finish event suppression (issue #589).

When upstream repeats ``finish_reason`` on the same choice (e.g. to attach
late ``usage`` data), the converter must emit exactly one ``finish`` IR
event per choice.  The cross-format Chat→Responses path must likewise
produce exactly one set of ``*.done`` / ``output_item.done`` SSE events.
"""

from llm_rosetta.converters.base.context import StreamContext
from llm_rosetta.converters.openai_chat.converter import OpenAIChatConverter
from llm_rosetta.converters.openai_responses.converter import OpenAIResponsesConverter
from llm_rosetta.converters.openai_responses.stream_context import (
    OpenAIResponsesStreamContext,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tool_call_chunks():
    """Upstream Chat chunks: tool call with repeated finish_reason + late usage."""
    base = {
        "id": "chatcmpl-dedup",
        "object": "chat.completion.chunk",
        "model": "gpt-4",
        "created": 1700000000,
    }
    return [
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "index": 0,
                                "id": "call_abc",
                                "type": "function",
                                "function": {"name": "get_weather", "arguments": ""},
                            }
                        ],
                    },
                    "finish_reason": None,
                }
            ],
        },
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {"index": 0, "function": {"arguments": '{"city":"NYC"}'}}
                        ]
                    },
                    "finish_reason": None,
                }
            ],
        },
        # First finish
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}]},
        # Repeated finish with late usage
        {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        },
    ]


def _make_text_chunks():
    """Upstream Chat chunks: text response with repeated finish_reason + late usage."""
    base = {
        "id": "chatcmpl-dedup-text",
        "object": "chat.completion.chunk",
        "model": "gpt-4",
        "created": 1700000000,
    }
    return [
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "Hello"},
                    "finish_reason": None,
                }
            ],
        },
        # First finish
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        # Repeated finish with late usage
        {
            **base,
            "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6},
        },
    ]


def _collect_ir(chunks):
    """Parse Chat chunks into IR events."""
    conv = OpenAIChatConverter()
    ctx = StreamContext()
    ir_events = []
    for chunk in chunks:
        ir_events.extend(conv.stream_response_from_provider(chunk, context=ctx))
    return ir_events, ctx


def _chat_to_responses(ir_events, chat_ctx):
    """Convert IR events through the Responses converter."""
    resp_conv = OpenAIResponsesConverter()
    resp_ctx = OpenAIResponsesStreamContext.from_base(chat_ctx)
    sse_events = []
    for ev in ir_events:
        result = resp_conv.stream_response_to_provider(ev, context=resp_ctx)
        if isinstance(result, list):
            sse_events.extend(
                r for r in result if isinstance(r, dict) and r.get("type")
            )
        elif isinstance(result, dict) and result.get("type"):
            sse_events.append(result)
    return sse_events, resp_ctx


# ---------------------------------------------------------------------------
# Chat converter IR-level dedup
# ---------------------------------------------------------------------------


class TestChatFinishDedup:
    """Repeated finish_reason in upstream Chat chunks → single IR finish."""

    def test_tool_call_single_finish_ir(self):
        ir_events, _ = _collect_ir(_make_tool_call_chunks())
        finish_events = [e for e in ir_events if e.get("type") == "finish"]
        assert len(finish_events) == 1

    def test_text_single_finish_ir(self):
        ir_events, _ = _collect_ir(_make_text_chunks())
        finish_events = [e for e in ir_events if e.get("type") == "finish"]
        assert len(finish_events) == 1

    def test_usage_still_emitted(self):
        ir_events, _ = _collect_ir(_make_tool_call_chunks())
        usage_events = [e for e in ir_events if e.get("type") == "usage"]
        assert len(usage_events) == 1
        assert usage_events[0]["usage"]["prompt_tokens"] == 10

    def test_stream_end_emitted(self):
        ir_events, _ = _collect_ir(_make_tool_call_chunks())
        end_events = [e for e in ir_events if e.get("type") == "stream_end"]
        assert len(end_events) == 1

    def test_context_marks_choice_finished(self):
        _, ctx = _collect_ir(_make_tool_call_chunks())
        assert ctx.is_choice_finished(0)
        assert ctx.is_ended


# ---------------------------------------------------------------------------
# Cross-format Chat → Responses dedup
# ---------------------------------------------------------------------------


class TestChatToResponsesFinishDedup:
    """Cross-format path must not duplicate *.done SSE events."""

    def test_tool_call_single_done_events(self):
        ir_events, chat_ctx = _collect_ir(_make_tool_call_chunks())
        sse, _ = _chat_to_responses(ir_events, chat_ctx)
        types = [e["type"] for e in sse]
        assert types.count("response.function_call_arguments.done") == 1
        assert types.count("response.output_item.done") == 1

    def test_tool_call_response_completed_once(self):
        ir_events, chat_ctx = _collect_ir(_make_tool_call_chunks())
        sse, _ = _chat_to_responses(ir_events, chat_ctx)
        types = [e["type"] for e in sse]
        assert types.count("response.completed") == 1

    def test_text_single_done_events(self):
        ir_events, chat_ctx = _collect_ir(_make_text_chunks())
        sse, _ = _chat_to_responses(ir_events, chat_ctx)
        types = [e["type"] for e in sse]
        assert types.count("response.output_item.done") == 1
        assert types.count("response.completed") == 1

    def test_tool_call_order_no_duplicates(self):
        """_tool_call_order must not contain duplicate IDs after from_base."""
        ir_events, chat_ctx = _collect_ir(_make_tool_call_chunks())
        _, resp_ctx = _chat_to_responses(ir_events, chat_ctx)
        assert resp_ctx._tool_call_order == ["call_abc"]

    def test_usage_merged_into_completed(self):
        ir_events, chat_ctx = _collect_ir(_make_tool_call_chunks())
        sse, _ = _chat_to_responses(ir_events, chat_ctx)
        completed = [e for e in sse if e["type"] == "response.completed"]
        assert len(completed) == 1
        usage = completed[0].get("response", {}).get("usage", {})
        assert usage.get("input_tokens", 0) > 0 or usage.get("output_tokens", 0) > 0


# ---------------------------------------------------------------------------
# StreamContext.from_base field consistency
# ---------------------------------------------------------------------------


class TestStreamContextFromBase:
    """from_base must copy all tool call tracking fields consistently."""

    def test_tool_call_index_copied(self):
        base = StreamContext()
        base.register_tool_call("call_1", "fn_a")
        base.register_tool_call("call_2", "fn_b")

        derived = OpenAIResponsesStreamContext.from_base(base)
        assert derived._tool_call_index == {"call_1": 0, "call_2": 1}
        assert derived._tool_call_order == ["call_1", "call_2"]

    def test_no_duplicate_after_re_register(self):
        """Re-registering the same call_id in derived context is a no-op."""
        base = StreamContext()
        base.register_tool_call("call_1", "fn_a")

        derived = OpenAIResponsesStreamContext.from_base(base)
        derived.register_tool_call("call_1", "fn_a")
        assert derived._tool_call_order == ["call_1"]


# ---------------------------------------------------------------------------
# Multi-choice (n>1) dedup
# ---------------------------------------------------------------------------


class TestMultiChoiceFinishDedup:
    """With n>1, each choice finishes independently — no cross-contamination."""

    @staticmethod
    def _make_multi_choice_chunks():
        base = {
            "id": "chatcmpl-multi",
            "object": "chat.completion.chunk",
            "model": "gpt-4",
            "created": 1700000000,
        }
        return [
            # Both choices start with text
            {
                **base,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "A"},
                        "finish_reason": None,
                    },
                    {
                        "index": 1,
                        "delta": {"role": "assistant", "content": "B"},
                        "finish_reason": None,
                    },
                ],
            },
            # Choice 0 finishes first
            {
                **base,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"},
                ],
            },
            # Choice 1 finishes
            {
                **base,
                "choices": [
                    {"index": 1, "delta": {}, "finish_reason": "stop"},
                ],
            },
            # Repeated finish on BOTH choices with late usage
            {
                **base,
                "choices": [
                    {"index": 0, "delta": {}, "finish_reason": "stop"},
                    {"index": 1, "delta": {}, "finish_reason": "stop"},
                ],
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 4,
                    "total_tokens": 14,
                },
            },
        ]

    def test_each_choice_finishes_exactly_once(self):
        ir_events, _ = _collect_ir(self._make_multi_choice_chunks())
        finish_events = [e for e in ir_events if e.get("type") == "finish"]
        assert len(finish_events) == 2

    def test_choice_indexes_tracked(self):
        _, ctx = _collect_ir(self._make_multi_choice_chunks())
        assert ctx.is_choice_finished(0)
        assert ctx.is_choice_finished(1)

    def test_usage_still_captured(self):
        ir_events, _ = _collect_ir(self._make_multi_choice_chunks())
        usage_events = [e for e in ir_events if e.get("type") == "usage"]
        assert len(usage_events) == 1
        assert usage_events[0]["usage"]["total_tokens"] == 14

    def test_stream_end_emitted_once(self):
        ir_events, _ = _collect_ir(self._make_multi_choice_chunks())
        end_events = [e for e in ir_events if e.get("type") == "stream_end"]
        assert len(end_events) == 1
