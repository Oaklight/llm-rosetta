"""Tests for Google Interactions streaming conversion."""

from llm_rosetta.converters.google_interactions import GoogleInteractionsConverter


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
