"""Tests for Google Interactions Converter (full request/response conversion)."""

from llm_rosetta.converters.google_interactions import GoogleInteractionsConverter
from llm_rosetta.types.ir import IRRequest, IRResponse


class TestRequestToProvider:
    def setup_method(self):
        self.converter = GoogleInteractionsConverter()

    def test_simple_request(self):
        ir: IRRequest = {
            "model": "gemini-3.6-flash",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
            ],
        }
        result, warnings = self.converter.request_to_provider(ir)
        assert result["model"] == "gemini-3.6-flash"
        assert result["input"] == "Hello!"

    def test_request_with_system_instruction(self):
        ir: IRRequest = {
            "model": "gemini-3.6-flash",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
            "system_instruction": [
                {"type": "text", "text": "You are a helpful assistant."}
            ],
        }
        result, _ = self.converter.request_to_provider(ir)
        assert result["system_instruction"] == "You are a helpful assistant."

    def test_request_with_system_message_in_messages(self):
        ir: IRRequest = {
            "model": "gemini-3.6-flash",
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Be helpful."}],
                },
                {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            ],
        }
        result, _ = self.converter.request_to_provider(ir)
        assert result["system_instruction"] == "Be helpful."
        if isinstance(result["input"], str):
            assert result["input"] == "Hi"

    def test_request_with_tools(self):
        ir: IRRequest = {
            "model": "gemini-3.6-flash",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Weather?"}]}
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        }
        result, _ = self.converter.request_to_provider(ir)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "get_weather"

    def test_request_with_generation_config(self):
        ir: IRRequest = {
            "model": "gemini-3.6-flash",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
            "generation": {"max_tokens": 1024, "seed": 42},
        }
        result, _ = self.converter.request_to_provider(ir)
        assert result["generation_config"]["max_output_tokens"] == 1024
        assert result["generation_config"]["seed"] == 42

    def test_request_with_reasoning(self):
        ir: IRRequest = {
            "model": "gemini-3.6-flash",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Think"}]}
            ],
            "reasoning": {"effort": "medium", "summary": "auto"},
        }
        result, _ = self.converter.request_to_provider(ir)
        gc = result["generation_config"]
        assert gc["thinking_level"] == "medium"
        assert gc["thinking_summaries"] == "auto"

    def test_request_with_streaming(self):
        ir: IRRequest = {
            "model": "gemini-3.6-flash",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
            "stream": {"enabled": True},
        }
        result, _ = self.converter.request_to_provider(ir)
        assert result["stream"] is True

    def test_request_with_multi_turn(self):
        ir: IRRequest = {
            "model": "gemini-3.6-flash",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "Hello!"}]},
                {"role": "user", "content": [{"type": "text", "text": "Bye"}]},
            ],
        }
        result, _ = self.converter.request_to_provider(ir)
        assert isinstance(result["input"], list)
        assert len(result["input"]) == 3

    def test_request_with_provider_extensions(self):
        ir: IRRequest = {
            "model": "gemini-3.6-flash",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
            "provider_extensions": {
                "store": False,
                "previous_interaction_id": "prev_123",
            },
        }
        result, _ = self.converter.request_to_provider(ir)
        assert result["store"] is False
        assert result["previous_interaction_id"] == "prev_123"


class TestRequestFromProvider:
    def setup_method(self):
        self.converter = GoogleInteractionsConverter()

    def test_simple_string_input(self):
        p = {"model": "gemini-3.6-flash", "input": "Hello!"}
        ir = self.converter.request_from_provider(p)
        assert ir["model"] == "gemini-3.6-flash"
        assert len(ir["messages"]) == 1
        assert ir["messages"][0]["role"] == "user"  # ty: ignore
        assert ir["messages"][0]["content"][0]["text"] == "Hello!"  # ty: ignore

    def test_steps_input(self):
        p = {
            "model": "gemini-3.6-flash",
            "input": [
                {"type": "user_input", "content": [{"type": "text", "text": "Hi"}]},
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": "Hello"}],
                },
            ],
        }
        ir = self.converter.request_from_provider(p)
        assert len(ir["messages"]) == 2

    def test_system_instruction(self):
        p = {
            "model": "gemini-3.6-flash",
            "system_instruction": "Be concise.",
            "input": "Hi",
        }
        ir = self.converter.request_from_provider(p)
        assert ir["system_instruction"] == [{"type": "text", "text": "Be concise."}]

    def test_thinking_config(self):
        p = {
            "model": "gemini-3.6-flash",
            "input": "Think",
            "generation_config": {"thinking_level": "low"},
        }
        ir = self.converter.request_from_provider(p)
        assert ir["reasoning"]["effort"] == "low"

    def test_roundtrip_request(self):
        original: IRRequest = {
            "model": "gemini-3.6-flash",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
            ],
            "generation": {"max_tokens": 512},
            "reasoning": {"effort": "medium"},
        }
        p, _ = self.converter.request_to_provider(original)
        back = self.converter.request_from_provider(p)
        assert back["model"] == "gemini-3.6-flash"
        assert back["messages"][0]["content"][0]["text"] == "Hello!"  # ty: ignore
        assert back["generation"]["max_tokens"] == 512
        assert back["reasoning"]["effort"] == "medium"


class TestResponseFromProvider:
    def setup_method(self):
        self.converter = GoogleInteractionsConverter()

    def test_simple_response(self):
        p = {
            "id": "interaction_123",
            "object": "interaction",
            "model": "gemini-3.6-flash",
            "status": "completed",
            "created": "2025-12-04T15:01:45Z",
            "steps": [
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": "Hello!"}],
                }
            ],
            "usage": {
                "total_input_tokens": 10,
                "total_output_tokens": 5,
                "total_tokens": 15,
            },
        }
        ir = self.converter.response_from_provider(p)
        assert ir["id"] == "interaction_123"
        assert ir["object"] == "response"
        assert ir["model"] == "gemini-3.6-flash"
        assert ir["choices"][0]["message"]["content"][0]["text"] == "Hello!"  # ty: ignore
        assert ir["choices"][0]["finish_reason"]["reason"] == "stop"
        assert ir["usage"]["prompt_tokens"] == 10

    def test_tool_call_response(self):
        p = {
            "id": "int_456",
            "model": "gemini-3.6-flash",
            "status": "requires_action",
            "steps": [
                {
                    "type": "function_call",
                    "id": "call_1",
                    "name": "get_weather",
                    "arguments": {"location": "SF"},
                }
            ],
        }
        ir = self.converter.response_from_provider(p)
        assert ir["choices"][0]["finish_reason"]["reason"] == "tool_calls"
        tc = ir["choices"][0]["message"]["content"][0]
        assert tc["type"] == "tool_call"
        assert tc["tool_name"] == "get_weather"

    def test_thinking_response(self):
        p = {
            "id": "int_789",
            "model": "gemini-3.6-flash",
            "status": "completed",
            "steps": [
                {
                    "type": "thought",
                    "signature": "sig_abc",
                    "summary": [{"type": "text", "text": "Thinking..."}],
                },
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": "42"}],
                },
            ],
            "usage": {
                "total_input_tokens": 20,
                "total_output_tokens": 10,
                "total_tokens": 30,
                "total_thought_tokens": 50,
            },
        }
        ir = self.converter.response_from_provider(p)
        content = ir["choices"][0]["message"]["content"]
        assert content[0]["type"] == "reasoning"
        assert content[1]["type"] == "text"
        assert ir["usage"]["reasoning_tokens"] == 50

    def test_status_mappings(self):
        for status, expected in [
            ("completed", "stop"),
            ("requires_action", "tool_calls"),
            ("incomplete", "length"),
            ("budget_exceeded", "length"),
            ("failed", "error"),
            ("cancelled", "cancelled"),
        ]:
            p = {"id": "x", "model": "m", "status": status, "steps": []}
            ir = self.converter.response_from_provider(p)
            assert ir["choices"][0]["finish_reason"]["reason"] == expected


class TestResponseToProvider:
    def setup_method(self):
        self.converter = GoogleInteractionsConverter()

    def test_simple_response(self):
        ir: IRResponse = {
            "id": "resp_123",
            "object": "response",
            "created": 1733324505,
            "model": "gemini-3.6-flash",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "Hello!"}],
                    },
                    "finish_reason": {"reason": "stop"},
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        }
        p = self.converter.response_to_provider(ir)
        assert p["id"] == "resp_123"
        assert p["object"] == "interaction"
        assert p["status"] == "completed"
        assert p["steps"][0]["type"] == "model_output"
        assert p["usage"]["total_input_tokens"] == 10

    def test_roundtrip_response(self):
        original = {
            "id": "int_rt",
            "object": "interaction",
            "model": "gemini-3.6-flash",
            "status": "completed",
            "created": "2025-12-04T15:01:45Z",
            "steps": [
                {
                    "type": "model_output",
                    "content": [{"type": "text", "text": "Hello"}],
                }
            ],
            "usage": {
                "total_input_tokens": 10,
                "total_output_tokens": 5,
                "total_tokens": 15,
            },
        }
        ir = self.converter.response_from_provider(original)
        back = self.converter.response_to_provider(ir)
        assert back["id"] == "int_rt"
        assert back["status"] == "completed"
        assert back["steps"][0]["content"][0]["text"] == "Hello"
        assert back["usage"]["total_input_tokens"] == 10
