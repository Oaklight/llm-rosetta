"""
OpenAI Chat Converter End-to-End Integration Test (SDK Version)

Tests the full conversion pipeline with real OpenAI API calls using the
official openai Python SDK and the refactored converter interfaces:
- request_to_provider / request_from_provider
- response_from_provider / response_to_provider
- stream_response_from_provider / stream_response_to_provider

Requires:
- OPENAI_API_KEY in .env
- openai Python SDK installed

Usage:
    conda activate llm_rosetta
    python tests/integration/test_openai_chat_sdk_e2e.py
"""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import dotenv
from openai import OpenAI

from examples.tools import available_tools, tools_spec
from llm_rosetta.converters.openai_chat import OpenAIChatConverter
from llm_rosetta.types.ir import (
    IRRequest,
    ToolCallPart,
    create_tool_result_message,
    extract_text_content,
    extract_tool_calls,
)

dotenv.load_dotenv()

# ============================================================================
# Setup
# ============================================================================

IMAGE_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/thumb/0/02/"
    "La_Libert%C3%A9_guidant_le_peuple_-_Eug%C3%A8ne_Delacroix_-_"
    "Mus%C3%A9e_du_Louvre_Peintures_RF_129_-_apr%C3%A8s_restauration_2024.jpg/"
    "3840px-La_Libert%C3%A9_guidant_le_peuple_-_Eug%C3%A8ne_Delacroix_-_"
    "Mus%C3%A9e_du_Louvre_Peintures_RF_129_-_apr%C3%A8s_restauration_2024.jpg"
)

openai_model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
openai_api_key = os.getenv("OPENAI_API_KEY")
openai_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")

if not openai_api_key:
    print("ERROR: OPENAI_API_KEY not set in .env")
    sys.exit(1)

client = OpenAI(api_key=openai_api_key, base_url=openai_base_url)
converter = OpenAIChatConverter()


# ============================================================================
# Helpers
# ============================================================================


def section(title: str) -> None:
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}")


def ok(name: str) -> None:
    print(f"  ✓ PASS: {name}")


def fail(name: str, err: str) -> None:
    print(f"  ✗ FAIL: {name} — {err}")


def execute_tool(tc: ToolCallPart) -> str:
    """Execute a tool call and return the result string."""
    fn = available_tools.get(tc["tool_name"])
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {tc['tool_name']}"})
    return fn(**tc["tool_input"])


# ============================================================================
# Test 1: Non-stream basic text
# ============================================================================


def test_non_stream_basic():
    """IRRequest → provider → API → response_from_provider → IRResponse."""
    section("Test 1: Non-stream basic text")

    ir_request: IRRequest = {
        "model": openai_model,
        "system_instruction": "You are a helpful assistant. Reply concisely.",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]},
        ],
    }

    # IR → provider
    provider_req, warnings = converter.request_to_provider(ir_request)
    print(f"  Warnings: {warnings}")
    print(f"  Provider request keys: {list(provider_req.keys())}")
    assert "model" in provider_req
    assert "messages" in provider_req
    assert len(provider_req["messages"]) >= 2  # system + user

    # Call API
    response = client.chat.completions.create(**provider_req)
    print(f"  API finish_reason: {response.choices[0].finish_reason}")

    # Provider response → IR
    ir_response = converter.response_from_provider(response)
    assert "choices" in ir_response
    assert len(ir_response["choices"]) >= 1

    msg = ir_response["choices"][0]["message"]
    text = extract_text_content(msg)
    print(f"  Response text: {text}")
    assert "4" in text

    # Verify usage
    assert "usage" in ir_response
    print(f"  Usage: {ir_response['usage']}")

    ok("Non-stream basic text")
    return True


# ============================================================================
# Test 2: Non-stream with image
# ============================================================================


def test_non_stream_image():
    """Test multimodal (image URL) request."""
    section("Test 2: Non-stream with image")

    ir_request: IRRequest = {
        "model": openai_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Describe this painting in one sentence.",
                    },
                    {
                        "type": "image",
                        "image_url": IMAGE_URL,
                    },
                ],
            },
        ],
    }

    provider_req, warnings = converter.request_to_provider(ir_request)
    print(f"  Warnings: {warnings}")

    # Verify image_url structure
    user_msg = provider_req["messages"][0]
    assert isinstance(user_msg["content"], list)
    img_part = user_msg["content"][1]
    assert img_part["type"] == "image_url"
    assert "url" in img_part["image_url"]

    response = client.chat.completions.create(**provider_req)
    ir_response = converter.response_from_provider(response)

    msg = ir_response["choices"][0]["message"]
    text = extract_text_content(msg)
    print(f"  Response text: {text[:120]}...")
    assert len(text) > 10

    ok("Non-stream with image")
    return True


# ============================================================================
# Test 3: Non-stream with tool calls
# ============================================================================


def test_non_stream_tool_calls():
    """Test tool call flow: request → tool_calls → tool_result → final."""
    section("Test 3: Non-stream with tool calls")

    # Round 1: Ask about weather
    ir_request: IRRequest = {
        "model": openai_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's the weather in San Francisco?",
                    }
                ],
            },
        ],
        "tools": tools_spec,
        "tool_choice": {"mode": "auto"},
    }

    provider_req, warnings = converter.request_to_provider(ir_request)
    print(f"  Round 1 warnings: {warnings}")
    assert "tools" in provider_req

    response = client.chat.completions.create(**provider_req)
    ir_response = converter.response_from_provider(response)

    assistant_msg = ir_response["choices"][0]["message"]
    tool_calls = extract_tool_calls(assistant_msg)
    print(f"  Tool calls: {len(tool_calls)}")
    assert len(tool_calls) >= 1, "Expected at least one tool call"

    tc = tool_calls[0]
    print(f"  Tool: {tc['tool_name']}({json.dumps(tc['tool_input'])})")

    # Execute tool
    result = execute_tool(tc)
    print(f"  Tool result: {result}")

    # Round 2: Send tool result
    tool_result_msg = create_tool_result_message(tc["tool_call_id"], result)

    ir_request_r2: IRRequest = {
        "model": openai_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's the weather in San Francisco?",
                    }
                ],
            },
            assistant_msg,
            tool_result_msg,
        ],
        "tools": tools_spec,
    }

    provider_req_r2, warnings_r2 = converter.request_to_provider(ir_request_r2)
    print(f"  Round 2 warnings: {warnings_r2}")

    response_r2 = client.chat.completions.create(**provider_req_r2)
    ir_response_r2 = converter.response_from_provider(response_r2)

    final_msg = ir_response_r2["choices"][0]["message"]
    final_text = extract_text_content(final_msg)
    print(f"  Final response: {final_text[:120]}...")
    assert len(final_text) > 5

    ok("Non-stream with tool calls")
    return True


# ============================================================================
# Test 4: Streaming text
# ============================================================================


def test_stream_text():
    """Test streaming with stream_response_from_provider."""
    section("Test 4: Streaming text")

    ir_request: IRRequest = {
        "model": openai_model,
        "system_instruction": "Reply concisely.",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Count from 1 to 5."}],
            },
        ],
        "stream": {"enabled": True, "include_usage": True},
    }

    provider_req, warnings = converter.request_to_provider(ir_request)
    print(f"  Warnings: {warnings}")
    assert provider_req.get("stream") is True

    stream = client.chat.completions.create(**provider_req)

    collected_text = []
    event_types_seen = set()
    print("  Stream output: ", end="")

    for chunk in stream:
        ir_events = converter.stream_response_from_provider(chunk)
        for event in ir_events:
            event_types_seen.add(event["type"])

            if event["type"] == "text_delta":
                print(event["text"], end="", flush=True)
                collected_text.append(event["text"])
            elif event["type"] == "finish":
                print(f"\n  Finish reason: {event['finish_reason']}")
            elif event["type"] == "usage":
                print(f"  Stream usage: {event['usage']}")

    full_text = "".join(collected_text)
    print(f"  Full text: {full_text}")
    assert len(full_text) > 0
    assert "text_delta" in event_types_seen
    assert "finish" in event_types_seen

    ok("Streaming text")
    return True


# ============================================================================
# Test 5: Streaming with tool calls
# ============================================================================


def test_stream_tool_calls():
    """Test streaming tool call chunks."""
    section("Test 5: Streaming with tool calls")

    ir_request: IRRequest = {
        "model": openai_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "What's the weather in Tokyo?",
                    }
                ],
            },
        ],
        "tools": tools_spec,
        "tool_choice": {"mode": "auto"},
        "stream": {"enabled": True, "include_usage": True},
    }

    provider_req, warnings = converter.request_to_provider(ir_request)
    assert provider_req.get("stream") is True

    stream = client.chat.completions.create(**provider_req)

    event_types_seen = set()
    tool_call_ids = []
    tool_names = []
    arg_fragments = []

    for chunk in stream:
        ir_events = converter.stream_response_from_provider(chunk)
        for event in ir_events:
            event_types_seen.add(event["type"])

            if event["type"] == "tool_call_start":
                tool_call_ids.append(event["tool_call_id"])
                tool_names.append(event["tool_name"])
                print(
                    f"  Tool call start: {event['tool_name']} (id={event['tool_call_id']})"
                )
            elif event["type"] == "tool_call_delta":
                arg_fragments.append(event["arguments_delta"])
            elif event["type"] == "finish":
                print(f"  Finish reason: {event['finish_reason']}")
            elif event["type"] == "usage":
                print(f"  Stream usage: {event['usage']}")

    print(f"  Event types seen: {event_types_seen}")
    print(f"  Tool names: {tool_names}")
    print(f"  Arguments assembled: {''.join(arg_fragments)}")

    assert "tool_call_start" in event_types_seen, "Expected tool_call_start event"
    assert len(tool_names) >= 1
    assert "finish" in event_types_seen

    ok("Streaming with tool calls")
    return True


# ============================================================================
# Test 6: Round-trip request conversion
# ============================================================================


def test_request_round_trip():
    """Test request_to_provider → request_from_provider round-trip."""
    section("Test 6: Request round-trip conversion")

    ir_request: IRRequest = {
        "model": openai_model,
        "system_instruction": "Be helpful.",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
        ],
        "tools": tools_spec,
        "tool_choice": {"mode": "auto"},
        "generation": {
            "temperature": 0.7,
            "max_output_tokens": 100,
        },
    }

    # IR → provider
    provider_req, warnings = converter.request_to_provider(ir_request)
    print(f"  Warnings: {warnings}")
    print(f"  Provider keys: {list(provider_req.keys())}")

    # provider → IR
    restored = converter.request_from_provider(provider_req)
    print(f"  Restored model: {restored['model']}")
    print(f"  Restored messages count: {len(restored['messages'])}")

    assert restored["model"] == openai_model
    assert len(restored["messages"]) >= 1
    assert "system_instruction" in restored
    assert "tools" in restored

    ok("Request round-trip conversion")
    return True


# ============================================================================
# Test 7: Response round-trip conversion
# ============================================================================


def test_response_round_trip():
    """Test response_from_provider → response_to_provider round-trip."""
    section("Test 7: Response round-trip conversion")

    ir_request: IRRequest = {
        "model": openai_model,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Say hi."}]},
        ],
    }

    provider_req, _ = converter.request_to_provider(ir_request)
    response = client.chat.completions.create(**provider_req)

    # Provider → IR
    ir_response = converter.response_from_provider(response)
    print(f"  IR response id: {ir_response['id']}")
    print(f"  IR response model: {ir_response['model']}")

    # IR → Provider
    restored_provider = converter.response_to_provider(ir_response)
    print(f"  Restored object: {restored_provider.get('object')}")
    assert restored_provider["object"] == "chat.completion"
    assert len(restored_provider["choices"]) >= 1

    # Verify content preserved
    original_text = extract_text_content(ir_response["choices"][0]["message"])
    restored_msg = restored_provider["choices"][0]["message"]
    assert original_text == restored_msg.get("content", "")

    ok("Response round-trip conversion")
    return True


# ============================================================================
# Main
# ============================================================================


def run_all():
    print("\nOpenAI Chat Converter E2E Tests (SDK Version)")
    print(f"Model: {openai_model}")
    print(f"Base URL: {openai_base_url}")

    tests = [
        ("Non-stream basic text", test_non_stream_basic),
        ("Non-stream with image", test_non_stream_image),
        ("Non-stream with tool calls", test_non_stream_tool_calls),
        ("Streaming text", test_stream_text),
        ("Streaming with tool calls", test_stream_tool_calls),
        ("Request round-trip", test_request_round_trip),
        ("Response round-trip", test_response_round_trip),
    ]

    results = []
    for name, fn in tests:
        try:
            success = fn()
            results.append((name, success))
        except Exception as e:
            fail(name, str(e))
            import traceback

            traceback.print_exc()
            results.append((name, False))

    # Summary
    section("SUMMARY")
    passed = sum(1 for _, s in results if s)
    total = len(results)
    for name, success in results:
        status = "✓" if success else "✗"
        print(f"  {status} {name}")
    print(f"\n  {passed}/{total} passed")

    return passed == total


if __name__ == "__main__":
    success = run_all()
    sys.exit(0 if success else 1)
