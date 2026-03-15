"""
Anthropic Converter integration tests.

Tests the top-level AnthropicConverter with full request/response conversion.
"""

from llm_rosetta.converters.anthropic import AnthropicConverter
from llm_rosetta.types.ir.request import IRRequest


class TestAnthropicConverter:
    """Integration tests for AnthropicConverter."""

    def setup_method(self):
        """Set up test fixtures."""
        self.converter = AnthropicConverter()

    # ==================== request_to_provider ====================

    def test_simple_request(self):
        """Test simple request conversion."""
        ir_request: IRRequest = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
            ],
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        assert result["model"] == "claude-3-5-sonnet-20241022"
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"][0]["text"] == "Hello!"
        assert result["max_tokens"] == 4096  # default

    def test_request_with_system_instruction(self):
        """Test request with system instruction."""
        ir_request: IRRequest = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
            ],
            "system_instruction": "You are a helpful assistant.",
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        assert result["system"] == "You are a helpful assistant."

    def test_request_with_system_message_in_messages(self):
        """Test system message in messages list is extracted."""
        ir_request: IRRequest = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": "Be helpful."}],
                },
                {"role": "user", "content": [{"type": "text", "text": "Hi"}]},
            ],
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        assert result["system"] == "Be helpful."
        # System message should not be in messages list
        assert all(m["role"] != "system" for m in result["messages"])

    def test_request_with_generation_config(self):
        """Test request with generation config."""
        ir_request: IRRequest = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
            ],
            "generation": {
                "temperature": 0.7,
                "max_tokens": 1024,
                "top_p": 0.9,
                "top_k": 50,
                "stop_sequences": ["\n\nHuman:"],
            },
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 1024
        assert result["top_p"] == 0.9
        assert result["top_k"] == 50
        assert result["stop_sequences"] == ["\n\nHuman:"]

    def test_request_with_tools(self):
        """Test request with tools."""
        ir_request: IRRequest = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Weather?"}]}
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                    "required_parameters": [],
                    "metadata": {},
                }
            ],
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "get_weather"
        assert "input_schema" in result["tools"][0]

    def test_request_with_tool_choice(self):
        """Test request with tool choice."""
        ir_request: IRRequest = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Weather?"}]}
            ],
            "tool_choice": {"mode": "tool", "tool_name": "get_weather"},
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        assert result["tool_choice"]["type"] == "tool"
        assert result["tool_choice"]["name"] == "get_weather"

    def test_request_with_tool_config(self):
        """Test request with tool config (disable_parallel)."""
        ir_request: IRRequest = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            ],
            "tool_choice": {"mode": "auto", "tool_name": ""},
            "tool_config": {"disable_parallel": True},
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        assert result["tool_choice"]["disable_parallel_tool_use"] is True

    def test_request_with_reasoning(self):
        """Test request with reasoning config."""
        ir_request: IRRequest = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Think!"}]}
            ],
            "reasoning": {"type": "enabled", "budget_tokens": 2048},
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        assert result["thinking"]["type"] == "enabled"
        assert result["thinking"]["budget_tokens"] == 2048

    def test_request_with_stream(self):
        """Test request with stream config."""
        ir_request: IRRequest = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            ],
            "stream": {"enabled": True},
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        assert result["stream"] is True

    def test_request_with_provider_extensions(self):
        """Test request with provider extensions pass-through."""
        ir_request: IRRequest = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            ],
            "provider_extensions": {"metadata": {"user_id": "123"}},
        }
        result, warnings = self.converter.request_to_provider(ir_request)
        assert result["metadata"] == {"user_id": "123"}

    # ==================== request_from_provider ====================

    def test_request_from_provider_basic(self):
        """Test basic request from provider."""
        provider_request = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Hello"}]}
            ],
        }
        ir_request = self.converter.request_from_provider(provider_request)
        assert ir_request["model"] == "claude-3-5-sonnet-20241022"
        assert len(ir_request["messages"]) == 1
        assert ir_request["generation"]["max_tokens"] == 1024

    def test_request_from_provider_with_system(self):
        """Test request from provider with system."""
        provider_request = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "system": "You are helpful.",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
        }
        ir_request = self.converter.request_from_provider(provider_request)
        assert ir_request["system_instruction"] == "You are helpful."

    def test_request_from_provider_with_thinking(self):
        """Test request from provider with thinking."""
        provider_request = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 1024,
            "thinking": {"type": "enabled", "budget_tokens": 4096},
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "Think"}]}
            ],
        }
        ir_request = self.converter.request_from_provider(provider_request)
        assert ir_request["reasoning"]["type"] == "enabled"
        assert ir_request["reasoning"]["budget_tokens"] == 4096

    def test_request_from_provider_pydantic(self):
        """Test request from provider with Pydantic model."""

        class MockPydanticModel:
            def model_dump(self):
                return {
                    "model": "claude-3-5-sonnet-20241022",
                    "max_tokens": 1024,
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "Hello"}],
                        }
                    ],
                }

        ir_request = self.converter.request_from_provider(MockPydanticModel())
        assert ir_request["model"] == "claude-3-5-sonnet-20241022"

    # ==================== response_from_provider ====================

    def test_response_from_provider_basic(self):
        """Test basic response from provider."""
        provider_response = {
            "id": "msg_01XFD67890",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [{"type": "text", "text": "Hello! How can I help?"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 15,
                "output_tokens": 25,
            },
        }
        result = self.converter.response_from_provider(provider_response)
        assert result["id"] == "msg_01XFD67890"
        assert result["object"] == "response"
        assert result["model"] == "claude-3-5-sonnet-20241022"
        assert len(result["choices"]) == 1

        choice = result["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"][0]["text"] == "Hello! How can I help?"
        assert choice["finish_reason"]["reason"] == "stop"

        assert result["usage"]["prompt_tokens"] == 15
        assert result["usage"]["completion_tokens"] == 25
        assert result["usage"]["total_tokens"] == 40

    def test_response_from_provider_with_tool_use(self):
        """Test response with tool use."""
        provider_response = {
            "id": "msg_tool",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_123",
                    "name": "get_weather",
                    "input": {"city": "SF"},
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 10, "output_tokens": 20},
        }
        result = self.converter.response_from_provider(provider_response)
        choice = result["choices"][0]
        assert choice["finish_reason"]["reason"] == "tool_calls"
        tc = choice["message"]["content"][0]
        assert tc["type"] == "tool_call"
        assert tc["tool_name"] == "get_weather"

    def test_response_from_provider_with_cache(self):
        """Test response with cache usage."""
        provider_response = {
            "id": "msg_cache",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [{"type": "text", "text": "Cached response"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 15,
                "output_tokens": 25,
                "cache_read_input_tokens": 10,
            },
        }
        result = self.converter.response_from_provider(provider_response)
        assert result["usage"]["cache_read_tokens"] == 10

    def test_response_from_provider_pydantic(self):
        """Test response from provider with Pydantic model."""

        class MockResponse:
            def model_dump(self):
                return {
                    "id": "msg_pydantic",
                    "type": "message",
                    "role": "assistant",
                    "model": "claude-3-5-sonnet-20241022",
                    "content": [{"type": "text", "text": "Hi"}],
                    "stop_reason": "end_turn",
                    "usage": {"input_tokens": 5, "output_tokens": 10},
                }

        result = self.converter.response_from_provider(MockResponse())
        assert result["id"] == "msg_pydantic"

    # ==================== response_to_provider ====================

    def test_response_to_provider_basic(self):
        """Test basic response to provider."""
        ir_response = {
            "id": "resp_123",
            "object": "response",
            "created": 1700000000,
            "model": "claude-3-5-sonnet-20241022",
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
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
        result = self.converter.response_to_provider(ir_response)
        assert result["id"] == "resp_123"
        assert result["type"] == "message"
        assert result["role"] == "assistant"
        assert result["content"][0]["type"] == "text"
        assert result["content"][0]["text"] == "Hello!"
        assert result["stop_reason"] == "end_turn"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 20

    def test_response_to_provider_with_tool_calls(self):
        """Test response to provider with tool calls."""
        ir_response = {
            "id": "resp_tc",
            "object": "response",
            "created": 1700000000,
            "model": "claude-3-5-sonnet-20241022",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": [
                            {
                                "type": "tool_call",
                                "tool_call_id": "call_123",
                                "tool_name": "search",
                                "tool_input": {"q": "test"},
                                "tool_type": "function",
                            }
                        ],
                    },
                    "finish_reason": {"reason": "tool_calls"},
                }
            ],
        }
        result = self.converter.response_to_provider(ir_response)
        assert result["stop_reason"] == "tool_use"
        assert result["content"][0]["type"] == "tool_use"
        assert result["content"][0]["name"] == "search"

    # ==================== messages_to_provider / messages_from_provider ====================

    def test_messages_to_provider(self):
        """Test messages_to_provider delegates to message_ops."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ]
        result, warnings = self.converter.messages_to_provider(messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    def test_messages_from_provider(self):
        """Test messages_from_provider delegates to message_ops."""
        provider_messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ]
        result = self.converter.messages_from_provider(provider_messages)
        assert len(result) == 1
        assert result[0]["role"] == "user"

    # ==================== Stream Support ====================

    def test_stream_text_delta(self):
        """Test stream text delta event conversion."""
        event = {
            "type": "content_block_delta",
            "delta": {"type": "text_delta", "text": "Hello"},
        }
        events = self.converter.stream_response_from_provider(event)
        assert len(events) == 1
        assert events[0]["type"] == "text_delta"
        assert events[0]["text"] == "Hello"

    def test_stream_tool_call_start(self):
        """Test stream tool call start event conversion."""
        event = {
            "type": "content_block_start",
            "content_block": {
                "type": "tool_use",
                "id": "toolu_123",
                "name": "search",
            },
        }
        events = self.converter.stream_response_from_provider(event)
        assert len(events) == 1
        assert events[0]["type"] == "tool_call_start"
        assert events[0]["tool_call_id"] == "toolu_123"
        assert events[0]["tool_name"] == "search"

    def test_stream_tool_call_delta(self):
        """Test stream tool call delta event conversion."""
        event = {
            "type": "content_block_delta",
            "delta": {
                "type": "input_json_delta",
                "partial_json": '{"city":',
            },
        }
        events = self.converter.stream_response_from_provider(event)
        assert len(events) == 1
        assert events[0]["type"] == "tool_call_delta"
        assert events[0]["arguments_delta"] == '{"city":'

    def test_stream_finish_event(self):
        """Test stream finish event conversion."""
        event = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
        }
        events = self.converter.stream_response_from_provider(event)
        assert len(events) >= 1
        finish = [e for e in events if e["type"] == "finish"]
        assert len(finish) == 1
        assert finish[0]["finish_reason"]["reason"] == "stop"

    def test_stream_usage_event(self):
        """Test stream usage event from message_start."""
        event = {
            "type": "message_start",
            "message": {
                "usage": {"input_tokens": 100},
            },
        }
        events = self.converter.stream_response_from_provider(event)
        assert len(events) == 1
        assert events[0]["type"] == "usage"
        assert events[0]["usage"]["prompt_tokens"] == 100

    def test_stream_ping_ignored(self):
        """Test ping event is ignored."""
        event = {"type": "ping"}
        events = self.converter.stream_response_from_provider(event)
        assert len(events) == 0

    def test_stream_content_block_stop_ignored(self):
        """Test content_block_stop is ignored."""
        event = {"type": "content_block_stop"}
        events = self.converter.stream_response_from_provider(event)
        assert len(events) == 0

    def test_stream_message_stop_ignored(self):
        """Test message_stop is ignored."""
        event = {"type": "message_stop"}
        events = self.converter.stream_response_from_provider(event)
        assert len(events) == 0

    # ==================== stream_response_to_provider ====================

    def test_stream_to_provider_text_delta(self):
        """Test IR text delta → Anthropic SSE event."""
        ir_event = {"type": "text_delta", "text": "Hello"}
        result = self.converter.stream_response_to_provider(ir_event)
        assert result["type"] == "content_block_delta"
        assert result["delta"]["type"] == "text_delta"
        assert result["delta"]["text"] == "Hello"

    def test_stream_to_provider_tool_call_start(self):
        """Test IR tool call start → Anthropic SSE event."""
        ir_event = {
            "type": "tool_call_start",
            "tool_call_id": "tc_123",
            "tool_name": "search",
        }
        result = self.converter.stream_response_to_provider(ir_event)
        assert result["type"] == "content_block_start"
        assert result["content_block"]["type"] == "tool_use"
        assert result["content_block"]["id"] == "tc_123"

    def test_stream_to_provider_tool_call_delta(self):
        """Test IR tool call delta → Anthropic SSE event."""
        ir_event = {
            "type": "tool_call_delta",
            "tool_call_id": "tc_123",
            "arguments_delta": '{"q":',
        }
        result = self.converter.stream_response_to_provider(ir_event)
        assert result["type"] == "content_block_delta"
        assert result["delta"]["type"] == "input_json_delta"

    def test_stream_to_provider_finish(self):
        """Test IR finish → Anthropic SSE event."""
        ir_event = {
            "type": "finish",
            "finish_reason": {"reason": "stop"},
        }
        result = self.converter.stream_response_to_provider(ir_event)
        assert result["type"] == "message_delta"
        assert result["delta"]["stop_reason"] == "end_turn"

    def test_stream_to_provider_usage(self):
        """Test IR usage → Anthropic SSE event."""
        ir_event = {
            "type": "usage",
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
        }
        result = self.converter.stream_response_to_provider(ir_event)
        assert result["type"] == "message_delta"
        assert result["usage"]["input_tokens"] == 10
        assert result["usage"]["output_tokens"] == 20
