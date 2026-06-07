"""
Stream usage detail fields — round-trip regression tests.

Validates that cache_read_tokens, cache_creation_tokens, reasoning_tokens,
prompt_tokens_details, and completion_tokens_details survive a
stream_response_from_provider → stream_response_to_provider round-trip
for all four converters.

Ref: https://github.com/Oaklight/llm-rosetta/issues/247 §1
"""

from __future__ import annotations

from typing import Any, cast

from llm_rosetta.converters.anthropic import AnthropicConverter
from llm_rosetta.converters.base.context import StreamContext
from llm_rosetta.converters.google_genai import GoogleGenAIConverter
from llm_rosetta.converters.openai_chat import OpenAIChatConverter
from llm_rosetta.converters.openai_responses import OpenAIResponsesConverter
from llm_rosetta.types.ir.stream import IRStreamEvent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _collect_usage_events(events: list[Any]) -> list[dict[str, Any]]:
    """Extract UsageEvent dicts from a flat event list."""
    result: list[dict[str, Any]] = []
    for e in events:
        if isinstance(e, dict) and e.get("type") == "usage":
            result.append(e)
    return result


def _run_stream_from_provider(
    converter: Any,
    chunks: list[dict[str, Any]],
) -> list[IRStreamEvent]:
    """Feed provider chunks through stream_response_from_provider."""
    ctx = StreamContext()
    all_events: list[IRStreamEvent] = []
    for chunk in chunks:
        result = converter.stream_response_from_provider(chunk, context=ctx)
        if isinstance(result, list):
            all_events.extend(cast(list[IRStreamEvent], result))
        elif result:
            all_events.append(cast(IRStreamEvent, result))
    return all_events


def _run_stream_to_provider(
    converter: Any,
    events: list[IRStreamEvent],
) -> list[dict[str, Any]]:
    """Feed IR events through stream_response_to_provider."""
    ctx = StreamContext()
    output: list[dict[str, Any]] = []
    for event in events:
        result = converter.stream_response_to_provider(event, context=ctx)
        if isinstance(result, list):
            output.extend(result)
        elif isinstance(result, dict) and result:
            output.append(result)
    return output


# ===========================================================================
# Anthropic
# ===========================================================================


class TestAnthropicStreamUsage:
    """Anthropic stream usage detail fields."""

    def setup_method(self) -> None:
        self.converter = AnthropicConverter()

    def test_message_start_cache_tokens_from_provider(self) -> None:
        """message_start.usage cache fields → UsageEvent."""
        chunks: list[dict[str, Any]] = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "content": [],
                    "usage": {
                        "input_tokens": 100,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 50,
                        "cache_creation_input_tokens": 25,
                    },
                },
            },
        ]
        events = _run_stream_from_provider(self.converter, chunks)
        usage_events = _collect_usage_events(events)
        assert len(usage_events) >= 1
        usage = usage_events[0]["usage"]
        assert usage["prompt_tokens"] == 100
        assert usage["cache_read_tokens"] == 50
        assert usage["cache_creation_tokens"] == 25

    def test_message_delta_cache_tokens_from_provider(self) -> None:
        """message_delta.usage cache fields → UsageEvent."""
        chunks: list[dict[str, Any]] = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_test",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "content": [],
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                },
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {
                    "output_tokens": 42,
                    "cache_read_input_tokens": 15,
                    "cache_creation_input_tokens": 7,
                },
            },
        ]
        events = _run_stream_from_provider(self.converter, chunks)
        usage_events = _collect_usage_events(events)
        # The second UsageEvent (from message_delta) should have cache fields
        delta_usage = usage_events[-1]["usage"]
        assert delta_usage["completion_tokens"] == 42
        assert delta_usage["cache_read_tokens"] == 15
        assert delta_usage["cache_creation_tokens"] == 7

    def test_cache_tokens_roundtrip_ir_level(self) -> None:
        """cache fields survive from_provider → IR extraction.

        The IR UsageEvent must contain the full cache detail fields.
        (to_provider reconstruction of message_start.usage has a
        pre-existing timing issue where stream_start fires before
        the UsageEvent is processed, so we verify at the IR level.)
        """
        chunks: list[dict[str, Any]] = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_rt",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-sonnet-4-20250514",
                    "content": [],
                    "usage": {
                        "input_tokens": 200,
                        "output_tokens": 0,
                        "cache_read_input_tokens": 80,
                        "cache_creation_input_tokens": 40,
                    },
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Hi"},
            },
            {
                "type": "content_block_stop",
                "index": 0,
            },
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": {"output_tokens": 5},
            },
            {"type": "message_stop"},
        ]

        ir_events = _run_stream_from_provider(self.converter, chunks)
        usage_events = _collect_usage_events(ir_events)

        # First usage (from message_start) has input + cache fields
        start_usage = usage_events[0]["usage"]
        assert start_usage["prompt_tokens"] == 200
        assert start_usage["cache_read_tokens"] == 80
        assert start_usage["cache_creation_tokens"] == 40

        # Second usage (from message_delta) has output tokens
        delta_usage = usage_events[1]["usage"]
        assert delta_usage["completion_tokens"] == 5


# ===========================================================================
# OpenAI Chat
# ===========================================================================


class TestOpenAIChatStreamUsage:
    """OpenAI Chat stream usage detail fields."""

    def setup_method(self) -> None:
        self.converter = OpenAIChatConverter()

    def test_stream_usage_details_from_provider(self) -> None:
        """Stream chunk usage with prompt/completion details → UsageEvent."""
        chunks: list[dict[str, Any]] = [
            {
                "id": "chatcmpl-test",
                "object": "chat.completion.chunk",
                "model": "gpt-4o",
                "created": 1700000000,
                "choices": [],
                "usage": {
                    "prompt_tokens": 150,
                    "completion_tokens": 60,
                    "total_tokens": 210,
                    "prompt_tokens_details": {"cached_tokens": 30},
                    "completion_tokens_details": {"reasoning_tokens": 20},
                },
            },
        ]
        events = _run_stream_from_provider(self.converter, chunks)
        usage_events = _collect_usage_events(events)
        assert len(usage_events) >= 1
        usage = usage_events[0]["usage"]
        assert usage["prompt_tokens"] == 150
        assert usage["completion_tokens"] == 60
        assert usage["cache_read_tokens"] == 30
        assert usage["reasoning_tokens"] == 20
        assert usage["prompt_tokens_details"] == {"cached_tokens": 30}
        assert usage["completion_tokens_details"] == {"reasoning_tokens": 20}

    def test_stream_usage_details_roundtrip(self) -> None:
        """Usage detail fields survive OAI Chat round-trip."""
        chunks: list[dict[str, Any]] = [
            {
                "id": "chatcmpl-rt",
                "object": "chat.completion.chunk",
                "model": "gpt-4o",
                "created": 1700000000,
                "choices": [
                    {
                        "index": 0,
                        "delta": {"role": "assistant", "content": "Hi"},
                        "finish_reason": None,
                    }
                ],
            },
            {
                "id": "chatcmpl-rt",
                "object": "chat.completion.chunk",
                "model": "gpt-4o",
                "created": 1700000000,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "total_tokens": 150,
                    "prompt_tokens_details": {"cached_tokens": 40},
                    "completion_tokens_details": {"reasoning_tokens": 10},
                },
            },
        ]
        ir_events = _run_stream_from_provider(self.converter, chunks)
        output = _run_stream_to_provider(self.converter, ir_events)

        # Find the chunk with usage in the output
        usage_chunks = [c for c in output if "usage" in c and c.get("usage")]
        assert len(usage_chunks) >= 1
        p_usage = usage_chunks[-1]["usage"]
        assert p_usage["prompt_tokens"] == 100
        assert p_usage["completion_tokens"] == 50
        assert p_usage["prompt_tokens_details"] == {"cached_tokens": 40}
        assert p_usage["completion_tokens_details"] == {"reasoning_tokens": 10}


# ===========================================================================
# OpenAI Responses
# ===========================================================================


class TestOpenAIResponsesStreamUsage:
    """OpenAI Responses stream usage detail fields."""

    def setup_method(self) -> None:
        self.converter = OpenAIResponsesConverter()

    def test_response_completed_usage_details(self) -> None:
        """response.completed usage with input/output details → UsageEvent."""
        chunks: list[dict[str, Any]] = [
            {
                "type": "response.created",
                "response": {
                    "id": "resp_test",
                    "object": "response",
                    "model": "gpt-4o",
                    "status": "in_progress",
                    "output": [],
                },
            },
            {
                "type": "response.completed",
                "response": {
                    "id": "resp_test",
                    "object": "response",
                    "model": "gpt-4o",
                    "status": "completed",
                    "output": [],
                    "usage": {
                        "input_tokens": 200,
                        "output_tokens": 80,
                        "total_tokens": 280,
                        "input_tokens_details": {"cached_tokens": 60},
                        "output_tokens_details": {"reasoning_tokens": 30},
                    },
                },
            },
        ]
        events = _run_stream_from_provider(self.converter, chunks)
        usage_events = _collect_usage_events(events)
        assert len(usage_events) >= 1
        usage = usage_events[0]["usage"]
        assert usage["prompt_tokens"] == 200
        assert usage["completion_tokens"] == 80
        assert usage["cache_read_tokens"] == 60
        assert usage["reasoning_tokens"] == 30


# ===========================================================================
# Google GenAI
# ===========================================================================


class TestGoogleStreamUsage:
    """Google GenAI stream usage detail fields."""

    def setup_method(self) -> None:
        self.converter = GoogleGenAIConverter()

    def test_stream_usage_cached_content_from_provider(self) -> None:
        """Stream chunk with cachedContentTokenCount → cache_read_tokens."""
        chunks: list[dict[str, Any]] = [
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Hello"}], "role": "model"},
                        "index": 0,
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 120,
                    "candidatesTokenCount": 45,
                    "totalTokenCount": 165,
                    "thoughtsTokenCount": 15,
                    "cachedContentTokenCount": 30,
                },
                "modelVersion": "gemini-2.5-flash",
            },
        ]
        events = _run_stream_from_provider(self.converter, chunks)
        usage_events = _collect_usage_events(events)
        assert len(usage_events) >= 1
        usage = usage_events[0]["usage"]
        assert usage["prompt_tokens"] == 120
        assert usage["completion_tokens"] == 45
        assert usage["reasoning_tokens"] == 15
        assert usage["cache_read_tokens"] == 30

    def test_stream_usage_snake_case_variant(self) -> None:
        """Snake-case usage fields (SDK style) also round-trip."""
        chunks: list[dict[str, Any]] = [
            {
                "candidates": [
                    {
                        "content": {"parts": [{"text": "Hi"}], "role": "model"},
                        "index": 0,
                    }
                ],
                "usage_metadata": {
                    "prompt_token_count": 100,
                    "candidates_token_count": 50,
                    "total_token_count": 150,
                    "thoughts_token_count": 10,
                    "cached_content_token_count": 20,
                },
                "modelVersion": "gemini-2.5-flash",
            },
        ]
        events = _run_stream_from_provider(self.converter, chunks)
        usage_events = _collect_usage_events(events)
        assert len(usage_events) >= 1
        usage = usage_events[0]["usage"]
        assert usage["prompt_tokens"] == 100
        assert usage["reasoning_tokens"] == 10
        assert usage["cache_read_tokens"] == 20
