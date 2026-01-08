import pytest
from llmir.converters.google import GoogleConverter
from llmir.types.ir_request import IRRequest
from llmir.types.ir_response import IRResponse


def test_google_request_conversion():
    converter = GoogleConverter()
    ir_request: IRRequest = {
        "model": "gemini-1.5-flash",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        "generation": {"temperature": 0.7, "max_tokens": 100},
    }

    provider_payload, warnings = converter.to_provider(ir_request)

    assert provider_payload["model"] == "gemini-1.5-flash"
    assert len(provider_payload["contents"]) == 1
    assert provider_payload["contents"][0]["role"] == "user"
    assert provider_payload["contents"][0]["parts"][0]["text"] == "Hello"
    assert provider_payload["config"]["temperature"] == 0.7
    assert provider_payload["config"]["max_output_tokens"] == 100


def test_google_response_conversion():
    converter = GoogleConverter()
    provider_response = {
        "response_id": "resp_123",
        "model_version": "gemini-1.5-flash",
        "candidates": [
            {
                "index": 0,
                "content": {
                    "role": "model",
                    "parts": [
                        {"thought": True, "text": "Thinking..."},
                        {"text": "Hello! How can I help you?"},
                    ],
                },
                "finish_reason": "STOP",
            }
        ],
        "usage_metadata": {
            "prompt_token_count": 10,
            "candidates_token_count": 20,
            "total_token_count": 30,
            "thoughts_token_count": 5,
        },
    }

    ir_response = converter.from_provider(provider_response)

    assert isinstance(ir_response, dict)
    assert ir_response["id"] == "resp_123"
    assert ir_response["model"] == "gemini-1.5-flash"
    assert len(ir_response["choices"]) == 1

    message = ir_response["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert message["content"][0]["type"] == "reasoning"
    assert message["content"][0]["reasoning"] == "Thinking..."
    assert message["content"][1]["type"] == "text"
    assert message["content"][1]["text"] == "Hello! How can I help you?"

    assert ir_response["usage"]["prompt_tokens"] == 10
    assert ir_response["usage"]["completion_tokens"] == 20
    assert ir_response["usage"]["reasoning_tokens"] == 5


def test_google_system_instruction_conversion():
    converter = GoogleConverter()
    ir_request: IRRequest = {
        "model": "gemini-1.5-flash",
        "system_instruction": "You are a helpful assistant.",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
    }

    provider_payload, _ = converter.to_provider(ir_request)

    assert "system_instruction" in provider_payload
    assert (
        provider_payload["system_instruction"]["parts"][0]["text"]
        == "You are a helpful assistant."
    )
