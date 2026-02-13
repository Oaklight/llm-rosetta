from llmir.converters.openai_responses import OpenAIResponsesConverter
from llmir.types.ir_request import IRRequest


def test_openai_responses_request_conversion():
    converter = OpenAIResponsesConverter()
    ir_request: IRRequest = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}],
        "generation": {"temperature": 0.7, "max_tokens": 100},
    }

    provider_payload, warnings = converter.to_provider(ir_request)

    assert provider_payload["model"] == "gpt-4o"
    assert len(provider_payload["input"]) == 1
    assert provider_payload["input"][0]["type"] == "message"
    assert provider_payload["input"][0]["role"] == "user"
    assert provider_payload["temperature"] == 0.7
    assert provider_payload["max_output_tokens"] == 100


def test_openai_responses_response_conversion():
    converter = OpenAIResponsesConverter()
    provider_response = {
        "id": "resp_123",
        "object": "response",
        "created_at": 1700000000,
        "model": "gpt-4o",
        "status": "completed",
        "output": [
            {"type": "reasoning", "reasoning": "Thinking about the greeting..."},
            {
                "type": "message",
                "role": "assistant",
                "content": [
                    {"type": "output_text", "text": "Hello! How can I help you today?"}
                ],
            },
        ],
        "usage": {
            "input_tokens": 10,
            "output_tokens": 20,
            "total_tokens": 30,
            "output_tokens_details": {"reasoning_tokens": 5},
        },
    }

    ir_response = converter.from_provider(provider_response)

    assert isinstance(ir_response, dict)
    assert ir_response["id"] == "resp_123"
    assert ir_response["model"] == "gpt-4o"
    assert len(ir_response["choices"]) == 1

    message = ir_response["choices"][0]["message"]
    assert message["role"] == "assistant"
    assert message["content"][0]["type"] == "reasoning"
    assert message["content"][0]["reasoning"] == "Thinking about the greeting..."
    assert message["content"][1]["type"] == "text"
    assert message["content"][1]["text"] == "Hello! How can I help you today?"

    assert ir_response["usage"]["prompt_tokens"] == 10
    assert ir_response["usage"]["completion_tokens"] == 20
    assert ir_response["usage"]["reasoning_tokens"] == 5


def test_openai_responses_system_instruction_conversion():
    converter = OpenAIResponsesConverter()
    ir_request: IRRequest = {
        "model": "gpt-4o",
        "system_instruction": "You are a helpful assistant.",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "Hi"}]}],
    }

    provider_payload, _ = converter.to_provider(ir_request)

    assert provider_payload["instructions"] == "You are a helpful assistant."
