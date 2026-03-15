"""
Google GenAI stream converter unit tests.
"""

import json

from llm_rosetta.converters.base.stream_context import StreamContext
from llm_rosetta.converters.google_genai import GoogleGenAIConverter


class TestStreamResponseFromProvider:
    """Tests for stream_response_from_provider."""

    def setup_method(self):
        self.converter = GoogleGenAIConverter()

    # --- Text delta ---

    def test_text_delta(self):
        """Text part produces TextDeltaEvent."""
        chunk = {
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello"}],
                    },
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert len(events) == 1
        assert events[0]["type"] == "text_delta"
        assert events[0]["text"] == "Hello"
        assert events[0]["choice_index"] == 0

    def test_text_delta_empty_string(self):
        """Empty text part still produces TextDeltaEvent."""
        chunk = {
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{"text": ""}],
                    },
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert len(events) == 1
        assert events[0]["type"] == "text_delta"
        assert events[0]["text"] == ""

    def test_text_delta_choice_index(self):
        """Choice index is preserved from candidate index."""
        chunk = {
            "candidates": [
                {
                    "index": 1,
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hi"}],
                    },
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert events[0]["choice_index"] == 1

    # --- Reasoning delta (thought) ---

    def test_thought_text_delta(self):
        """Text part with thought: true produces ReasoningDeltaEvent."""
        chunk = {
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Let me think...", "thought": True}],
                    },
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert len(events) == 1
        assert events[0]["type"] == "reasoning_delta"
        assert events[0]["reasoning"] == "Let me think..."
        assert events[0]["choice_index"] == 0

    def test_thought_false_is_text(self):
        """Text part with thought: false produces TextDeltaEvent."""
        chunk = {
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{"text": "regular text", "thought": False}],
                    },
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert len(events) == 1
        assert events[0]["type"] == "text_delta"

    # --- Tool call (functionCall) ---

    def test_function_call_produces_start_and_delta(self):
        """functionCall part produces both ToolCallStartEvent and ToolCallDeltaEvent."""
        chunk = {
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "function_call": {
                                    "name": "get_weather",
                                    "args": {"city": "NYC"},
                                }
                            }
                        ],
                    },
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        types = [e["type"] for e in events]
        assert "tool_call_start" in types
        assert "tool_call_delta" in types

        start = [e for e in events if e["type"] == "tool_call_start"][0]
        assert start["tool_name"] == "get_weather"
        assert start["tool_call_id"].startswith("call_get_weather_")

        delta = [e for e in events if e["type"] == "tool_call_delta"][0]
        args = json.loads(delta["arguments_delta"])
        assert args == {"city": "NYC"}

    def test_function_call_camel_case(self):
        """functionCall (camelCase) key is also handled."""
        chunk = {
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "search",
                                    "args": {"q": "test"},
                                }
                            }
                        ],
                    },
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        start_events = [e for e in events if e["type"] == "tool_call_start"]
        assert len(start_events) == 1
        assert start_events[0]["tool_name"] == "search"

    def test_function_call_with_id(self):
        """functionCall with explicit id uses that id."""
        chunk = {
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "function_call": {
                                    "id": "custom_id",
                                    "name": "func",
                                    "args": {},
                                }
                            }
                        ],
                    },
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        start = [e for e in events if e["type"] == "tool_call_start"][0]
        assert start["tool_call_id"] == "custom_id"

    def test_function_call_empty_args(self):
        """functionCall with empty args produces valid JSON."""
        chunk = {
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "function_call": {
                                    "name": "no_args_func",
                                    "args": {},
                                }
                            }
                        ],
                    },
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        delta = [e for e in events if e["type"] == "tool_call_delta"][0]
        assert json.loads(delta["arguments_delta"]) == {}

    # --- Finish event ---

    def test_finish_stop(self):
        """finish_reason STOP maps to 'stop'."""
        chunk = {
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": []},
                    "finish_reason": "STOP",
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        finish_events = [e for e in events if e["type"] == "finish"]
        assert len(finish_events) == 1
        assert finish_events[0]["finish_reason"]["reason"] == "stop"

    def test_finish_max_tokens(self):
        """finish_reason MAX_TOKENS maps to 'length'."""
        chunk = {
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": []},
                    "finish_reason": "MAX_TOKENS",
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        finish_events = [e for e in events if e["type"] == "finish"]
        assert finish_events[0]["finish_reason"]["reason"] == "length"

    def test_finish_safety(self):
        """finish_reason SAFETY maps to 'content_filter'."""
        chunk = {
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": []},
                    "finish_reason": "SAFETY",
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        finish_events = [e for e in events if e["type"] == "finish"]
        assert finish_events[0]["finish_reason"]["reason"] == "content_filter"

    def test_finish_recitation(self):
        """finish_reason RECITATION maps to 'content_filter'."""
        chunk = {
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": []},
                    "finish_reason": "RECITATION",
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        finish_events = [e for e in events if e["type"] == "finish"]
        assert finish_events[0]["finish_reason"]["reason"] == "content_filter"

    def test_finish_malformed_function_call(self):
        """finish_reason MALFORMED_FUNCTION_CALL maps to 'error'."""
        chunk = {
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": []},
                    "finish_reason": "MALFORMED_FUNCTION_CALL",
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        finish_events = [e for e in events if e["type"] == "finish"]
        assert finish_events[0]["finish_reason"]["reason"] == "error"

    def test_finish_other(self):
        """finish_reason OTHER maps to 'error'."""
        chunk = {
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": []},
                    "finish_reason": "OTHER",
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        finish_events = [e for e in events if e["type"] == "finish"]
        assert finish_events[0]["finish_reason"]["reason"] == "error"

    # --- Usage event ---

    def test_usage_metadata(self):
        """usage_metadata produces UsageEvent."""
        chunk = {
            "candidates": [],
            "usage_metadata": {
                "prompt_token_count": 10,
                "candidates_token_count": 5,
                "total_token_count": 15,
            },
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert len(events) == 1
        assert events[0]["type"] == "usage"
        assert events[0]["usage"]["prompt_tokens"] == 10
        assert events[0]["usage"]["completion_tokens"] == 5
        assert events[0]["usage"]["total_tokens"] == 15

    def test_usage_with_reasoning_tokens(self):
        """usage_metadata with thoughts_token_count includes reasoning_tokens."""
        chunk = {
            "candidates": [],
            "usage_metadata": {
                "prompt_token_count": 10,
                "candidates_token_count": 20,
                "total_token_count": 30,
                "thoughts_token_count": 8,
            },
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert events[0]["usage"]["reasoning_tokens"] == 8

    # --- Empty / irrelevant ---

    def test_empty_candidates(self):
        """Chunk with empty candidates and no usage produces no events."""
        chunk = {"candidates": []}
        events = self.converter.stream_response_from_provider(chunk)
        assert events == []

    def test_empty_parts(self):
        """Candidate with empty parts and no finish_reason produces no events."""
        chunk = {
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": []},
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert events == []

    def test_text_none_not_emitted(self):
        """Part with text: None should NOT produce an event."""
        chunk = {
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{"text": None}],
                    },
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        assert events == []

    # --- SDK object normalization ---

    def test_normalize_sdk_object(self):
        """SDK objects with model_dump() are normalized."""

        class MockChunk:
            def model_dump(self):
                return {
                    "candidates": [
                        {
                            "index": 0,
                            "content": {
                                "role": "model",
                                "parts": [{"text": "sdk"}],
                            },
                        }
                    ]
                }

        events = self.converter.stream_response_from_provider(MockChunk())
        assert len(events) == 1
        assert events[0]["text"] == "sdk"

    def test_normalize_tuple(self):
        """Tuple input is unwrapped (first element)."""
        inner = {
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{"text": "tuple"}],
                    },
                }
            ]
        }
        events = self.converter.stream_response_from_provider((inner,))
        assert len(events) == 1
        assert events[0]["text"] == "tuple"


class TestStreamResponseToProvider:
    """Tests for stream_response_to_provider."""

    def setup_method(self):
        self.converter = GoogleGenAIConverter()

    def test_text_delta(self):
        """TextDeltaEvent → Google chunk."""
        event = {"type": "text_delta", "text": "Hello", "choice_index": 0}
        result = self.converter.stream_response_to_provider(event)
        candidate = result["candidates"][0]
        assert candidate["index"] == 0
        assert candidate["content"]["role"] == "model"
        assert candidate["content"]["parts"][0]["text"] == "Hello"

    def test_reasoning_delta(self):
        """ReasoningDeltaEvent → Google chunk with thought: true."""
        event = {
            "type": "reasoning_delta",
            "reasoning": "thinking...",
            "choice_index": 0,
        }
        result = self.converter.stream_response_to_provider(event)
        part = result["candidates"][0]["content"]["parts"][0]
        assert part["thought"] is True
        assert part["text"] == "thinking..."

    def test_tool_call_start(self):
        """ToolCallStartEvent → Google chunk with function_call."""
        event = {
            "type": "tool_call_start",
            "tool_call_id": "call_1",
            "tool_name": "search",
            "choice_index": 0,
        }
        result = self.converter.stream_response_to_provider(event)
        fc = result["candidates"][0]["content"]["parts"][0]["function_call"]
        assert fc["name"] == "search"
        assert fc["args"] == {}

    def test_tool_call_delta(self):
        """ToolCallDeltaEvent → Google chunk with function_call args."""
        event = {
            "type": "tool_call_delta",
            "tool_call_id": "call_1",
            "arguments_delta": '{"q": "test"}',
            "choice_index": 0,
        }
        result = self.converter.stream_response_to_provider(event)
        fc = result["candidates"][0]["content"]["parts"][0]["function_call"]
        assert fc["args"] == {"q": "test"}

    def test_tool_call_delta_invalid_json(self):
        """ToolCallDeltaEvent with invalid JSON falls back to empty args."""
        event = {
            "type": "tool_call_delta",
            "tool_call_id": "call_1",
            "arguments_delta": "not json",
            "choice_index": 0,
        }
        result = self.converter.stream_response_to_provider(event)
        fc = result["candidates"][0]["content"]["parts"][0]["function_call"]
        assert fc["args"] == {}

    def test_finish_event_stop(self):
        """FinishEvent with 'stop' → STOP."""
        event = {
            "type": "finish",
            "finish_reason": {"reason": "stop"},
            "choice_index": 0,
        }
        result = self.converter.stream_response_to_provider(event)
        assert result["candidates"][0]["finish_reason"] == "STOP"

    def test_finish_event_length(self):
        """FinishEvent with 'length' → MAX_TOKENS."""
        event = {
            "type": "finish",
            "finish_reason": {"reason": "length"},
            "choice_index": 0,
        }
        result = self.converter.stream_response_to_provider(event)
        assert result["candidates"][0]["finish_reason"] == "MAX_TOKENS"

    def test_finish_event_content_filter(self):
        """FinishEvent with 'content_filter' → SAFETY."""
        event = {
            "type": "finish",
            "finish_reason": {"reason": "content_filter"},
            "choice_index": 0,
        }
        result = self.converter.stream_response_to_provider(event)
        assert result["candidates"][0]["finish_reason"] == "SAFETY"

    def test_finish_event_error(self):
        """FinishEvent with 'error' → OTHER."""
        event = {
            "type": "finish",
            "finish_reason": {"reason": "error"},
            "choice_index": 0,
        }
        result = self.converter.stream_response_to_provider(event)
        assert result["candidates"][0]["finish_reason"] == "OTHER"

    def test_usage_event(self):
        """UsageEvent → Google usage_metadata."""
        event = {
            "type": "usage",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        result = self.converter.stream_response_to_provider(event)
        assert result["usage_metadata"]["prompt_token_count"] == 10
        assert result["usage_metadata"]["candidates_token_count"] == 5
        assert result["usage_metadata"]["total_token_count"] == 15

    def test_usage_event_with_reasoning_tokens(self):
        """UsageEvent with reasoning_tokens includes thoughts_token_count."""
        event = {
            "type": "usage",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "reasoning_tokens": 8,
            },
        }
        result = self.converter.stream_response_to_provider(event)
        assert result["usage_metadata"]["thoughts_token_count"] == 8

    def test_unknown_event_type(self):
        """Unknown event type returns empty dict."""
        event = {"type": "unknown_event"}
        result = self.converter.stream_response_to_provider(event)
        assert result == {}

    def test_default_choice_index(self):
        """Default choice_index is 0 when not specified."""
        event = {"type": "text_delta", "text": "Hi"}
        result = self.converter.stream_response_to_provider(event)
        assert result["candidates"][0]["index"] == 0


class TestStreamRoundTrip:
    """Round-trip tests: provider → IR → provider."""

    def setup_method(self):
        self.converter = GoogleGenAIConverter()

    def test_text_delta_round_trip(self):
        """Text delta round-trip preserves content."""
        chunk = {
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello"}],
                    },
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        restored = self.converter.stream_response_to_provider(events[0])
        assert restored["candidates"][0]["content"]["parts"][0]["text"] == "Hello"

    def test_reasoning_delta_round_trip(self):
        """Reasoning delta round-trip preserves content."""
        chunk = {
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{"text": "step 1", "thought": True}],
                    },
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        restored = self.converter.stream_response_to_provider(events[0])
        part = restored["candidates"][0]["content"]["parts"][0]
        assert part["text"] == "step 1"
        assert part["thought"] is True

    def test_finish_round_trip(self):
        """Finish event round-trip preserves reason mapping."""
        chunk = {
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": []},
                    "finish_reason": "STOP",
                }
            ]
        }
        events = self.converter.stream_response_from_provider(chunk)
        finish = [e for e in events if e["type"] == "finish"][0]
        restored = self.converter.stream_response_to_provider(finish)
        assert restored["candidates"][0]["finish_reason"] == "STOP"

    def test_usage_round_trip(self):
        """Usage event round-trip preserves token counts."""
        chunk = {
            "candidates": [],
            "usage_metadata": {
                "prompt_token_count": 20,
                "candidates_token_count": 10,
                "total_token_count": 30,
            },
        }
        events = self.converter.stream_response_from_provider(chunk)
        restored = self.converter.stream_response_to_provider(events[0])
        assert restored["usage_metadata"]["total_token_count"] == 30


class TestStreamResponseFromProviderWithContext:
    """Tests for stream_response_from_provider with StreamContext."""

    def setup_method(self):
        self.converter = GoogleGenAIConverter()

    def test_first_chunk_emits_stream_start(self):
        """First chunk with context emits StreamStartEvent."""
        context = StreamContext()
        chunk = {
            "model_version": "gemini-2.0-flash",
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello"}],
                    },
                }
            ],
        }
        events = self.converter.stream_response_from_provider(chunk, context=context)
        assert events[0]["type"] == "stream_start"
        assert events[0]["response_id"] == ""
        assert events[0]["model"] == "gemini-2.0-flash"
        assert context.is_started
        # Text delta should follow
        assert events[1]["type"] == "text_delta"
        assert events[1]["text"] == "Hello"

    def test_stream_start_with_response_id(self):
        """StreamStartEvent captures response_id from chunk."""
        context = StreamContext()
        chunk = {
            "response_id": "resp_abc123",
            "model_version": "gemini-2.0-flash",
            "candidates": [],
        }
        events = self.converter.stream_response_from_provider(chunk, context=context)
        assert events[0]["type"] == "stream_start"
        assert events[0]["response_id"] == "resp_abc123"
        assert context.response_id == "resp_abc123"
        assert context.model == "gemini-2.0-flash"

    def test_stream_start_only_on_first_chunk(self):
        """StreamStartEvent is only emitted for the first chunk."""
        context = StreamContext()
        chunk1 = {
            "model_version": "gemini-2.0-flash",
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello"}],
                    },
                }
            ],
        }
        chunk2 = {
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{"text": " world"}],
                    },
                }
            ],
        }
        events1 = self.converter.stream_response_from_provider(chunk1, context=context)
        events2 = self.converter.stream_response_from_provider(chunk2, context=context)
        assert any(e["type"] == "stream_start" for e in events1)
        assert not any(e["type"] == "stream_start" for e in events2)

    def test_finish_reason_emits_stream_end(self):
        """Chunk with finish_reason emits StreamEndEvent when context provided."""
        context = StreamContext()
        # First chunk to start the stream
        chunk1 = {
            "model_version": "gemini-2.0-flash",
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hi"}],
                    },
                }
            ],
        }
        self.converter.stream_response_from_provider(chunk1, context=context)

        # Final chunk with finish_reason
        chunk2 = {
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": []},
                    "finish_reason": "STOP",
                }
            ],
        }
        events = self.converter.stream_response_from_provider(chunk2, context=context)
        types = [e["type"] for e in events]
        assert "finish" in types
        assert "stream_end" in types
        # stream_end should be the last event
        assert events[-1]["type"] == "stream_end"
        assert context.is_ended

    def test_no_stream_end_without_finish_reason(self):
        """No StreamEndEvent when chunk has no finish_reason."""
        context = StreamContext()
        chunk = {
            "model_version": "gemini-2.0-flash",
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{"text": "partial"}],
                    },
                }
            ],
        }
        events = self.converter.stream_response_from_provider(chunk, context=context)
        assert not any(e["type"] == "stream_end" for e in events)
        assert not context.is_ended

    def test_tool_call_registered_in_context(self):
        """Tool calls are registered in context."""
        context = StreamContext()
        chunk = {
            "model_version": "gemini-2.0-flash",
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "function_call": {
                                    "name": "get_weather",
                                    "args": {"city": "NYC"},
                                }
                            }
                        ],
                    },
                }
            ],
        }
        events = self.converter.stream_response_from_provider(chunk, context=context)
        start = [e for e in events if e["type"] == "tool_call_start"][0]
        tool_call_id = start["tool_call_id"]
        assert context.get_tool_name(tool_call_id) == "get_weather"

    def test_without_context_no_lifecycle_events(self):
        """Without context, no StreamStartEvent or StreamEndEvent."""
        chunk = {
            "model_version": "gemini-2.0-flash",
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [{"text": "Hello"}],
                    },
                    "finish_reason": "STOP",
                }
            ],
        }
        events = self.converter.stream_response_from_provider(chunk)
        types = [e["type"] for e in events]
        assert "stream_start" not in types
        assert "stream_end" not in types
        # Original events should still be present
        assert "text_delta" in types
        assert "finish" in types

    def test_without_context_tool_call_still_works(self):
        """Without context, tool calls still produce start+delta events."""
        chunk = {
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "function_call": {
                                    "name": "search",
                                    "args": {"q": "test"},
                                }
                            }
                        ],
                    },
                }
            ],
        }
        events = self.converter.stream_response_from_provider(chunk)
        types = [e["type"] for e in events]
        assert "tool_call_start" in types
        assert "tool_call_delta" in types


class TestStreamResponseToProviderWithContext:
    """Tests for stream_response_to_provider with StreamContext."""

    def setup_method(self):
        self.converter = GoogleGenAIConverter()

    def test_stream_start_produces_initial_chunk(self):
        """StreamStartEvent produces initial chunk with model_version."""
        event = {
            "type": "stream_start",
            "response_id": "resp_123",
            "model": "gemini-2.0-flash",
        }
        result = self.converter.stream_response_to_provider(event)
        assert result["candidates"] == []
        assert result["model_version"] == "gemini-2.0-flash"

    def test_stream_start_stores_context(self):
        """StreamStartEvent stores metadata in context."""
        context = StreamContext()
        event = {
            "type": "stream_start",
            "response_id": "resp_123",
            "model": "gemini-2.0-flash",
        }
        self.converter.stream_response_to_provider(event, context=context)
        assert context.response_id == "resp_123"
        assert context.model == "gemini-2.0-flash"
        assert context.is_started

    def test_stream_end_returns_empty_dict(self):
        """StreamEndEvent returns empty dict."""
        event = {"type": "stream_end"}
        result = self.converter.stream_response_to_provider(event)
        assert result == {}

    def test_stream_end_marks_context_ended(self):
        """StreamEndEvent marks context as ended."""
        context = StreamContext()
        context.mark_started()
        event = {"type": "stream_end"}
        self.converter.stream_response_to_provider(event, context=context)
        assert context.is_ended

    def test_content_block_start_returns_empty_dict(self):
        """ContentBlockStartEvent returns empty dict."""
        event = {
            "type": "content_block_start",
            "block_index": 0,
            "block_type": "text",
        }
        result = self.converter.stream_response_to_provider(event)
        assert result == {}

    def test_content_block_end_returns_empty_dict(self):
        """ContentBlockEndEvent returns empty dict."""
        event = {
            "type": "content_block_end",
            "block_index": 0,
        }
        result = self.converter.stream_response_to_provider(event)
        assert result == {}

    def test_tool_call_delta_with_context_recovers_name(self):
        """tool_call_delta with context recovers tool name (P0 fix)."""
        context = StreamContext()
        context.register_tool_call("call_1", "get_weather")

        event = {
            "type": "tool_call_delta",
            "tool_call_id": "call_1",
            "arguments_delta": '{"city": "NYC"}',
            "choice_index": 0,
        }
        result = self.converter.stream_response_to_provider(event, context=context)
        fc = result["candidates"][0]["content"]["parts"][0]["function_call"]
        assert fc["name"] == "get_weather"
        assert fc["args"] == {"city": "NYC"}

    def test_tool_call_delta_without_context_name_empty(self):
        """tool_call_delta without context has empty name."""
        event = {
            "type": "tool_call_delta",
            "tool_call_id": "call_1",
            "arguments_delta": '{"city": "NYC"}',
            "choice_index": 0,
        }
        result = self.converter.stream_response_to_provider(event)
        fc = result["candidates"][0]["content"]["parts"][0]["function_call"]
        assert fc["name"] == ""

    def test_tool_call_delta_context_unknown_id_name_empty(self):
        """tool_call_delta with context but unknown ID has empty name."""
        context = StreamContext()
        context.register_tool_call("call_other", "other_func")

        event = {
            "type": "tool_call_delta",
            "tool_call_id": "call_unknown",
            "arguments_delta": "{}",
            "choice_index": 0,
        }
        result = self.converter.stream_response_to_provider(event, context=context)
        fc = result["candidates"][0]["content"]["parts"][0]["function_call"]
        assert fc["name"] == ""

    def test_stream_start_without_context(self):
        """StreamStartEvent without context still produces chunk."""
        event = {
            "type": "stream_start",
            "response_id": "resp_123",
            "model": "gemini-2.0-flash",
        }
        result = self.converter.stream_response_to_provider(event)
        assert result["candidates"] == []
        assert result["model_version"] == "gemini-2.0-flash"

    def test_stream_end_without_context(self):
        """StreamEndEvent without context returns empty dict."""
        event = {"type": "stream_end"}
        result = self.converter.stream_response_to_provider(event)
        assert result == {}
