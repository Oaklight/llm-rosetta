"""
Anthropic Messages API Full Request/Response Conversion Tests
"""

import pytest
from llmir.converters.anthropic import AnthropicConverter
from llmir.types.ir_request import IRRequest
from llmir.types.ir_response import Message, TextPart

class TestAnthropicFullConversion:
    """Anthropic full conversion test class"""

    def setup_method(self):
        """Set up tests"""
        self.converter = AnthropicConverter()

    def test_full_request_conversion(self):
        """Test full IRRequest to Anthropic conversion"""
        ir_request: IRRequest = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello!"}]
                }
            ],
            "system_instruction": "You are a helpful assistant.",
            "generation": {
                "temperature": 0.7,
                "max_tokens": 1024,
                "top_p": 0.9,
                "top_k": 50,
                "stop_sequences": ["\n\nHuman:"],
            },
            "reasoning": {
                "type": "enabled",
                "budget_tokens": 2048
            },
            "stream": {"enabled": True}
        }

        result, warnings = self.converter.to_provider(ir_request)

        assert result["model"] == "claude-3-5-sonnet-20241022"
        assert result["system"] == "You are a helpful assistant."
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "user"
        assert result["messages"][0]["content"][0]["text"] == "Hello!"
        
        assert result["temperature"] == 0.7
        assert result["max_tokens"] == 1024
        assert result["top_p"] == 0.9
        assert result["top_k"] == 50
        assert result["stop_sequences"] == ["\n\nHuman:"]
        assert result["thinking"] == {"type": "enabled", "budget_tokens": 2048}
        assert result["stream"] is True

    def test_full_response_conversion(self):
        """Test full Anthropic response to IRResponse conversion"""
        provider_response = {
            "id": "msg_01XFD67890",
            "type": "message",
            "role": "assistant",
            "model": "claude-3-5-sonnet-20241022",
            "content": [
                {
                    "type": "text",
                    "text": "Hello! How can I help you today?"
                }
            ],
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": 15,
                "output_tokens": 25,
                "cache_read_input_tokens": 10
            }
        }

        result = self.converter.from_provider(provider_response)

        assert result["id"] == "msg_01XFD67890"
        assert result["object"] == "response"
        assert result["model"] == "claude-3-5-sonnet-20241022"
        assert len(result["choices"]) == 1
        
        choice = result["choices"][0]
        assert choice["index"] == 0
        assert choice["message"]["role"] == "assistant"
        assert choice["message"]["content"][0]["text"] == "Hello! How can I help you today?"
        assert choice["finish_reason"]["reason"] == "stop"
        
        usage = result["usage"]
        assert usage["prompt_tokens"] == 15
        assert usage["completion_tokens"] == 25
        assert usage["total_tokens"] == 40
        assert usage["cache_read_tokens"] == 10

    def test_request_with_tools(self):
        """Test request with tools and tool choice"""
        ir_request: IRRequest = {
            "model": "claude-3-5-sonnet-20241022",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Weather?"}]}],
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "parameters": {"type": "object", "properties": {}}
                }
            ],
            "tool_choice": {"mode": "tool", "tool_name": "get_weather"},
            "tool_config": {"disable_parallel": True}
        }

        result, _ = self.converter.to_provider(ir_request)

        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "get_weather"
        assert result["tool_choice"] == {
            "type": "tool", 
            "name": "get_weather",
            "disable_parallel_tool_use": True
        }