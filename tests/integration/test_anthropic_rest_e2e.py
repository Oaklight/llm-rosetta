"""
Anthropic Converter End-to-End Integration Test

Tests the full conversion pipeline with real Anthropic-compatible API calls
using the new refactored converter interfaces:
- request_to_provider / request_from_provider
- response_from_provider / response_to_provider
- stream_response_from_provider / stream_response_to_provider

Requires:
- ANTHROPIC_API_KEY in .env
- Network access (uses proxychains if direct access fails)

Usage:
    conda activate llm_rosetta
    python tests/integration/test_anthropic_e2e.py
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
from llm_rosetta.converters.anthropic import AnthropicConverter
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

anthropic_model = os.getenv("ANTHROPIC_MODEL", "claude-3-5-sonnet-20241022")
anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
anthropic_base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

if not anthropic_api_key:
    print("ERROR: ANTHROPIC_API_KEY not set in .env")
    sys.exit(1)

API_URL = f"{anthropic_base_url}/v1/messages"
HEADERS = {
    "x-api-key": anthropic_api_key,
    "anthropic-version": "2023-06-01",
    "Content-Type": "application/json",
}

converter = AnthropicConverter()


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
    """Call the Anthropic Messages API and return the response dict.

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


def call_api_stream(provider_req: dict, max_retries: int = 3):
    """Call the Anthropic Messages API with streaming and yield SSE events.

    Retries on 429 (rate limit) with exponential backoff.
    """
    provider_req["stream"] = True
    for attempt in range(max_retries):
        response = requests.post(
            API_URL, headers=HEADERS, json=provider_req, timeout=60, stream=True
        )
        if response.status_code == 429:
            wait = 2 ** (attempt + 1)
            print(f"  [Rate limited, retrying in {wait}s...]")
            time.sleep(wait)
            continue
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8")
            if decoded.startswith("data: "):
                data_str = decoded[6:]
                if data_str.strip() == "[DONE]":
                    return
                try:
                    yield json.loads(data_str)
                except json.JSONDecodeError:
                    continue
        return

    # Final attempt
    response = requests.post(
        API_URL, headers=HEADERS, json=provider_req, timeout=60, stream=True
    )
    response.raise_for_status()
    for line in response.iter_lines():
        if not line:
            continue
        decoded = line.decode("utf-8")
        if decoded.startswith("data: "):
            data_str = decoded[6:]
            if data_str.strip() == "[DONE]":
                return
            try:
                yield json.loads(data_str)
            except json.JSONDecodeError:
                continue


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
        "model": anthropic_model,
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
    assert "system" in provider_req

    # Call API
    response_data = call_api(provider_req)
    print(f"  API stop_reason: {response_data.get('stop_reason')}")

    # Provider response → IR
    ir_response = converter.response_from_provider(response_data)
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
# Test 2: Non-stream with tool calls
# ============================================================================


def test_non_stream_tool_calls():
    """Test tool call flow: request → tool_calls → tool_result → final."""
    section("Test 2: Non-stream with tool calls")

    # Round 1: Ask about weather
    ir_request: IRRequest = {
        "model": anthropic_model,
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
    tool_result_msg = create_tool_result_message(tc["tool_call_id"], result)

    ir_request_r2: IRRequest = {
        "model": anthropic_model,
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
# Test 3: Streaming text
# ============================================================================


def test_stream_text():
    """Test streaming with stream_response_from_provider."""
    section("Test 3: Streaming text")

    ir_request: IRRequest = {
        "model": anthropic_model,
        "system_instruction": "Reply concisely.",
        "messages": [
            {
                "role": "user",
                "content": [{"type": "text", "text": "Count from 1 to 5."}],
            },
        ],
        "stream": {"enabled": True},
    }

    provider_req, warnings = converter.request_to_provider(ir_request)
    print(f"  Warnings: {warnings}")

    # Remove stream from provider_req since call_api_stream adds it
    provider_req.pop("stream", None)

    collected_text = []
    event_types_seen = set()
    print("  Stream output: ", end="")

    for sse_event in call_api_stream(provider_req):
        ir_events = converter.stream_response_from_provider(sse_event)
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
# Test 4: Streaming with tool calls
# ============================================================================


def test_stream_tool_calls():
    """Test streaming tool call chunks."""
    section("Test 4: Streaming with tool calls")

    ir_request: IRRequest = {
        "model": anthropic_model,
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
        "tools": cast(list[ToolDefinition], tools_spec),
        "tool_choice": {"mode": "auto"},
        "stream": {"enabled": True},
    }

    provider_req, warnings = converter.request_to_provider(ir_request)
    provider_req.pop("stream", None)

    event_types_seen = set()
    tool_call_ids = []
    tool_names = []
    arg_fragments = []

    for sse_event in call_api_stream(provider_req):
        ir_events = converter.stream_response_from_provider(sse_event)
        for event in ir_events:
            event_types_seen.add(event["type"])

            if event["type"] == "tool_call_start":
                tool_call_ids.append(event["tool_call_id"])
                tool_names.append(event["tool_name"])
                print(
                    f"  Tool call start: {event['tool_name']} "
                    f"(id={event['tool_call_id']})"
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
# Test 5: Round-trip request conversion
# ============================================================================


def test_request_round_trip():
    """Test request_to_provider → request_from_provider round-trip."""
    section("Test 5: Request round-trip conversion")

    ir_request: IRRequest = {
        "model": anthropic_model,
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

    assert restored["model"] == anthropic_model
    assert len(messages) >= 1
    assert "system_instruction" in restored
    assert "tools" in restored

    ok("Request round-trip conversion")
    return True


# ============================================================================
# Test 6: Response round-trip conversion
# ============================================================================


def test_response_round_trip():
    """Test response_from_provider → response_to_provider round-trip."""
    section("Test 6: Response round-trip conversion")

    ir_request: IRRequest = {
        "model": anthropic_model,
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Say hi."}]},
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
    print(f"  Restored type: {restored_provider.get('type')}")
    assert restored_provider["type"] == "message"
    assert len(restored_provider["content"]) >= 1

    # Verify content preserved
    original_text = extract_text_content(ir_response["choices"][0]["message"])
    restored_content = restored_provider["content"]
    restored_text = ""
    for block in restored_content:
        if block.get("type") == "text":
            restored_text += block.get("text", "")
    assert original_text == restored_text

    ok("Response round-trip conversion")
    return True


# ============================================================================
# Main
# ============================================================================


def run_all():
    print("\nAnthropic Converter E2E Tests")
    print(f"Model: {anthropic_model}")
    print(f"Base URL: {anthropic_base_url}")

    tests = [
        ("Non-stream basic text", test_non_stream_basic),
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
