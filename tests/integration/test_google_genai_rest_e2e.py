"""
Google GenAI Converter End-to-End Integration Test (REST Version)

Tests the full conversion pipeline with real Google Gemini API calls
using raw HTTP requests (requests library) and the refactored converter
interfaces:
- request_to_provider / request_from_provider
- response_from_provider / response_to_provider

Requires:
- GOOGLE_API_KEY in .env
- Network access (uses proxychains if direct access fails)

Usage:
    conda activate llm_rosetta
    proxychains -q python tests/integration/test_google_genai_rest_e2e.py
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
from llm_rosetta.converters.google_genai import GoogleGenAIConverter
from llm_rosetta.types.ir import (
    IRRequest,
    ToolCallPart,
    ToolDefinition,
    create_tool_result_message,
    extract_text_content,
    extract_tool_calls,
)

dotenv.load_dotenv(override=True)

# ============================================================================
# Setup
# ============================================================================

google_model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
google_api_key = os.getenv("GOOGLE_API_KEY")
google_base_url = os.getenv(
    "GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com"
)

if not google_api_key:
    print("ERROR: GOOGLE_API_KEY not set in .env")
    sys.exit(1)

API_URL = f"{google_base_url}/v1beta/models/{google_model}:generateContent"
HEADERS = {
    "x-goog-api-key": google_api_key,
    "Content-Type": "application/json",
}

converter = GoogleGenAIConverter()


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
    """Call the Google Gemini REST API and return the response dict.

    Retries on 429 (rate limit) with exponential backoff.
    """
    # Build REST API request body from converter output
    request_body = {"contents": provider_req.get("contents", [])}

    if "system_instruction" in provider_req:
        request_body["system_instruction"] = provider_req["system_instruction"]

    config = provider_req.get("config", {})
    if config.get("tools"):
        request_body["tools"] = config["tools"]
    if config.get("tool_config"):
        request_body["tool_config"] = config["tool_config"]

    # Pass generation config fields at top level for REST API
    gen_fields = [
        "temperature",
        "top_p",
        "top_k",
        "max_output_tokens",
        "stop_sequences",
        "candidate_count",
    ]
    generation_config = {}
    for field in gen_fields:
        if field in config:
            generation_config[field] = config[field]
    if generation_config:
        request_body["generation_config"] = generation_config

    for attempt in range(max_retries):
        response = requests.post(
            API_URL, headers=HEADERS, json=request_body, timeout=60
        )
        if response.status_code == 429:
            wait = 2 ** (attempt + 1)
            print(f"  [Rate limited, retrying in {wait}s...]")
            time.sleep(wait)
            continue
        response.raise_for_status()
        return response.json()
    # Final attempt without catching
    response = requests.post(API_URL, headers=HEADERS, json=request_body, timeout=60)
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
        "model": google_model,
        "system_instruction": "You are a helpful assistant. Reply concisely.",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "What is 2+2?"}]},
        ],
    }

    # IR → provider
    provider_req, warnings = converter.request_to_provider(ir_request)
    print(f"  Warnings: {warnings}")
    print(f"  Provider request keys: {list(provider_req.keys())}")
    assert "contents" in provider_req

    # Call API
    response_data = call_api(provider_req)
    print(f"  Response keys: {list(response_data.keys())}")

    # Provider response → IR
    ir_response = converter.response_from_provider(response_data)
    assert "choices" in ir_response
    assert len(ir_response["choices"]) >= 1

    msg = ir_response["choices"][0]["message"]
    text = extract_text_content(msg)
    print(f"  Response text: {text}")
    assert "4" in text

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
        "model": google_model,
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

    response_data = call_api(provider_req)
    ir_response = converter.response_from_provider(response_data)
    assert len(ir_response["choices"]) >= 1

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
        "model": google_model,
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
        "model": google_model,
        "system_instruction": "Be helpful.",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}]},
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

    assert restored["model"] == google_model
    assert len(messages) >= 1
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
        "model": google_model,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Say hi."}]},
        ],
    }

    provider_req, _ = converter.request_to_provider(ir_request)
    response_data = call_api(provider_req)

    # Add response_id to make it a full response for response_from_provider
    response_data["response_id"] = response_data.get("response_id", "rest-test-id")
    response_data["model_version"] = response_data.get("model_version", google_model)

    # Provider → IR
    ir_response = converter.response_from_provider(response_data)
    print(f"  IR response id: {ir_response['id']}")
    print(f"  IR choices: {len(ir_response['choices'])}")

    # IR → Provider
    restored_provider = converter.response_to_provider(ir_response)
    print(f"  Restored candidates: {len(restored_provider['candidates'])}")
    assert len(restored_provider["candidates"]) >= 1

    # Verify content preserved
    original_text = extract_text_content(ir_response["choices"][0]["message"])
    restored_parts = restored_provider["candidates"][0]["content"]["parts"]
    restored_text = ""
    for part in restored_parts:
        if "text" in part:
            restored_text += part["text"]
    assert original_text == restored_text

    ok("Response round-trip conversion")
    return True


# ============================================================================
# Test 5: Multi-turn conversation
# ============================================================================


def test_multi_turn():
    """Test multi-turn conversation flow."""
    section("Test 5: Multi-turn conversation")

    # Turn 1
    ir_request: IRRequest = {
        "model": google_model,
        "system_instruction": "You are a helpful assistant. Reply concisely.",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "My name is Alice."}],
            },
        ],
    }

    provider_req, _ = converter.request_to_provider(ir_request)
    response_data = call_api(provider_req)
    ir_response = converter.response_from_provider(response_data)
    assistant_msg_1 = ir_response["choices"][0]["message"]
    text_1 = extract_text_content(assistant_msg_1)
    print(f"  Turn 1 response: {text_1[:80]}...")

    # Turn 2
    ir_request_2: IRRequest = {
        "model": google_model,
        "system_instruction": "You are a helpful assistant. Reply concisely.",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "My name is Alice."}],
            },
            assistant_msg_1,
            {
                "role": "user",
                "content": [{"type": "text", "text": "What is my name?"}],
            },
        ],
    }

    provider_req_2, _ = converter.request_to_provider(ir_request_2)
    response_data_2 = call_api(provider_req_2)
    ir_response_2 = converter.response_from_provider(response_data_2)
    text_2 = extract_text_content(ir_response_2["choices"][0]["message"])
    print(f"  Turn 2 response: {text_2[:80]}...")
    assert "alice" in text_2.lower()

    ok("Multi-turn conversation")
    return True


# ============================================================================
# Main
# ============================================================================


def run_all():
    print("\nGoogle GenAI Converter E2E Tests (REST Version)")
    print(f"Model: {google_model}")
    print(f"Base URL: {google_base_url}")

    tests = [
        ("Non-stream basic text", test_non_stream_basic),
        ("Non-stream with tool calls", test_non_stream_tool_calls),
        ("Request round-trip", test_request_round_trip),
        ("Response round-trip", test_response_round_trip),
        ("Multi-turn conversation", test_multi_turn),
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
