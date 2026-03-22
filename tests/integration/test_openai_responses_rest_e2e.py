"""
OpenAI Responses Converter End-to-End Integration Test (REST Version)

Tests the full conversion pipeline with real OpenAI Responses API calls
using raw HTTP requests (requests library) and the refactored converter
interfaces:
- request_to_provider / request_from_provider
- response_from_provider / response_to_provider

Requires:
- OPENAI_RESPONSES_API_KEY in .env
- Network access

Usage:
    conda activate llm_rosetta
    python tests/integration/test_openai_responses_rest_e2e.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import dotenv
import requests

from typing import cast

from examples.tools import available_tools, tools_spec
from llm_rosetta.converters.openai_responses import OpenAIResponsesConverter
from llm_rosetta.types.ir import (
    IRRequest,
    Message,
    ToolCallPart,
    ToolDefinition,
    extract_text_content,
    extract_tool_calls,
)

dotenv.load_dotenv(override=True)

# ============================================================================
# Setup
# ============================================================================

openai_model = os.getenv("OPENAI_RESPONSES_MODEL", "gpt-4o-mini")
openai_api_key = os.getenv("OPENAI_RESPONSES_API_KEY")
openai_base_url = os.getenv("OPENAI_RESPONSES_BASE_URL", "https://api.openai.com/v1")

if not openai_api_key:
    print("ERROR: OPENAI_RESPONSES_API_KEY not set in .env")
    sys.exit(1)

API_URL = f"{openai_base_url}/responses"
HEADERS = {
    "Authorization": f"Bearer {openai_api_key}",
    "Content-Type": "application/json",
}

converter = OpenAIResponsesConverter()


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


def call_api(provider_req: dict, max_retries: int = 3) -> dict:
    """Call the OpenAI Responses API and return the response dict.

    Retries on 429 (rate limit) with exponential backoff.
    """
    for attempt in range(max_retries):
        response = requests.post(
            API_URL, headers=HEADERS, json=provider_req, timeout=60
        )
        if response.status_code == 429:
            wait = 2 ** (attempt + 1)
            print(f"  [Rate limited, retrying in {wait}s...]")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()
    # Final attempt without catching
    response = requests.post(API_URL, headers=HEADERS, json=provider_req, timeout=60)
    response.raise_for_status()
    return response.json()


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
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is 2+2?"}],
            },
        ],
    }

    # IR → provider
    provider_req, warnings = converter.request_to_provider(ir_request)
    print(f"  Warnings: {warnings}")
    print(f"  Provider request keys: {list(provider_req.keys())}")
    assert "model" in provider_req
    assert "input" in provider_req
    assert "instructions" in provider_req

    # Call API
    response_data = call_api(provider_req)
    print(f"  API status: {response_data.get('status')}")

    # Provider response → IR
    ir_response = converter.response_from_provider(response_data)
    assert "choices" in ir_response
    assert len(ir_response["choices"]) >= 1

    msg = ir_response["choices"][0]["message"]
    text = extract_text_content(msg)
    print(f"  Response text: {text}")
    assert "4" in text

    # Verify usage
    if "usage" in ir_response:
        print(f"  Usage: {ir_response['usage']}")

    ok("Non-stream basic text")
    return True


# ============================================================================
# Test 2: Non-stream with tool calls
# ============================================================================


def test_non_stream_tool_calls():
    """Test tool call flow: request → tool_calls → tool_result → final."""
    section("Test 2: Non-stream with tool calls")

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
        "tools": cast(list[ToolDefinition], tools_spec),
        "tool_choice": {"mode": "auto"},
    }

    provider_req, warnings = converter.request_to_provider(ir_request)
    print(f"  Round 1 warnings: {warnings}")
    assert "tools" in provider_req

    response_data = call_api(provider_req)
    ir_response = converter.response_from_provider(response_data)

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
    ir_request_r2: IRRequest = {
        "model": openai_model,
        "messages": cast(
            list[Message],
            [
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
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": tc["tool_call_id"],
                            "result": result,
                        }
                    ],
                },
            ],
        ),
        "tools": cast(list[ToolDefinition], tools_spec),
    }

    provider_req_r2, warnings_r2 = converter.request_to_provider(ir_request_r2)
    print(f"  Round 2 warnings: {warnings_r2}")

    response_data_r2 = call_api(provider_req_r2)
    ir_response_r2 = converter.response_from_provider(response_data_r2)

    final_msg = ir_response_r2["choices"][0]["message"]
    final_text = extract_text_content(final_msg)
    print(f"  Final response: {final_text[:120]}...")
    assert len(final_text) > 5

    ok("Non-stream with tool calls")
    return True


# ============================================================================
# Test 3: Request round-trip conversion
# ============================================================================


def test_request_round_trip():
    """Test request_to_provider → request_from_provider round-trip."""
    section("Test 3: Request round-trip conversion")

    ir_request: IRRequest = {
        "model": openai_model,
        "system_instruction": "Be helpful.",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Hello"}],
            },
        ],
        "tools": cast(list[ToolDefinition], tools_spec),
        "tool_choice": {"mode": "auto"},
        "generation": {
            "temperature": 0.7,
            "max_tokens": 100,
        },
    }

    # IR → provider
    provider_req, warnings = converter.request_to_provider(ir_request)
    print(f"  Warnings: {warnings}")
    print(f"  Provider keys: {list(provider_req.keys())}")

    # provider → IR
    restored = converter.request_from_provider(provider_req)
    print(f"  Restored model: {restored['model']}")
    messages = list(restored["messages"])
    print(f"  Restored messages count: {len(messages)}")

    assert restored["model"] == openai_model
    assert len(messages) >= 1
    assert "system_instruction" in restored
    assert "tools" in restored

    ok("Request round-trip conversion")
    return True


# ============================================================================
# Test 4: Response round-trip conversion
# ============================================================================


def test_response_round_trip():
    """Test response_from_provider → response_to_provider round-trip."""
    section("Test 4: Response round-trip conversion")

    ir_request: IRRequest = {
        "model": openai_model,
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Say hi."}],
            },
        ],
    }

    provider_req, _ = converter.request_to_provider(ir_request)
    response_data = call_api(provider_req)

    # Provider → IR
    ir_response = converter.response_from_provider(response_data)
    print(f"  IR response id: {ir_response['id']}")
    print(f"  IR response model: {ir_response['model']}")

    # IR → Provider
    restored_provider = converter.response_to_provider(ir_response)
    print(f"  Restored object: {restored_provider.get('object')}")
    assert restored_provider["object"] == "response"
    assert len(restored_provider["output"]) >= 1

    # Verify content preserved
    original_text = extract_text_content(ir_response["choices"][0]["message"])
    restored_output = restored_provider["output"]
    # Find the message item
    message_items = [item for item in restored_output if item.get("type") == "message"]
    assert len(message_items) >= 1
    restored_text = message_items[0]["content"][0]["text"]
    assert original_text == restored_text

    ok("Response round-trip conversion")
    return True


# ============================================================================
# Main
# ============================================================================


def run_all():
    print("\nOpenAI Responses Converter E2E Tests (REST Version)")
    print(f"Model: {openai_model}")
    print(f"Base URL: {openai_base_url}")

    tests = [
        ("Non-stream basic text", test_non_stream_basic),
        ("Non-stream with tool calls", test_non_stream_tool_calls),
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
        # Brief pause between tests to avoid rate limiting
        time.sleep(3)

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
