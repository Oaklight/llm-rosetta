"""
Google GenAI Converter End-to-End Integration Test (SDK Version)

Tests the full conversion pipeline with real Google Gemini API calls using the
official google-genai Python SDK and the refactored converter interfaces:
- request_to_provider / request_from_provider
- response_from_provider / response_to_provider

Requires:
- GOOGLE_API_KEY in .env
- google-genai Python SDK installed
- Network access (uses proxychains if direct access fails)

Usage:
    conda activate llm_rosetta
    proxychains -q python tests/integration/test_google_genai_sdk_e2e.py
"""

import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import dotenv
from google import genai
from google.genai import types

from typing import Any, cast

from examples.tools import (
    available_tools,
    generate_chart,
    multimodal_tools_spec,
    tools_spec,
)
from llm_rosetta.converters.google_genai import GoogleGenAIConverter
from llm_rosetta.types.ir import (
    IRRequest,
    Message,
    ToolCallPart,
    ToolDefinition,
    create_tool_result_message,
    extract_all_text,
    extract_text_content,
    extract_tool_calls,
)

dotenv.load_dotenv(override=True)

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

google_model = os.getenv("GOOGLE_MODEL", "gemini-2.0-flash")
google_api_key = os.getenv("GOOGLE_API_KEY")

if not google_api_key:
    print("ERROR: GOOGLE_API_KEY not set in .env")
    sys.exit(1)

client = genai.Client(api_key=google_api_key)
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


def get_text(msg: Message) -> str:
    """Extract text from message, falling back to reasoning content for thinking models."""
    text = extract_text_content(msg)
    return text if text else extract_all_text(msg)


def execute_tool(tc: ToolCallPart) -> str:
    """Execute a tool call and return the result string."""
    fn = available_tools.get(tc["tool_name"])
    if fn is None:
        return json.dumps({"error": f"Unknown tool: {tc['tool_name']}"})
    return fn(**tc["tool_input"])


def build_sdk_tools(ir_tools):
    """Build Google SDK Tool objects from IR tool definitions."""
    declarations = []
    for tool in ir_tools:
        declarations.append(
            types.FunctionDeclaration(
                name=tool["name"],
                description=tool.get("description", ""),
                parameters=tool.get("parameters"),
            )
        )
    return [types.Tool(function_declarations=declarations)]


def build_sdk_config(provider_req):
    """Build a GenerateContentConfig from the converter's provider request."""
    config = provider_req.get("config", {})
    config_kwargs = {}

    # Generation params
    if "temperature" in config:
        config_kwargs["temperature"] = config["temperature"]
    if "max_output_tokens" in config:
        config_kwargs["max_output_tokens"] = config["max_output_tokens"]
    if "top_p" in config:
        config_kwargs["top_p"] = config["top_p"]
    if "top_k" in config:
        config_kwargs["top_k"] = config["top_k"]
    if "stop_sequences" in config:
        config_kwargs["stop_sequences"] = config["stop_sequences"]

    # System instruction
    system_instruction = provider_req.get("system_instruction")
    if system_instruction:
        if isinstance(system_instruction, dict):
            parts = system_instruction.get("parts", [])
            text_parts = [p["text"] for p in parts if "text" in p]
            config_kwargs["system_instruction"] = "\n".join(text_parts)
        elif isinstance(system_instruction, str):
            config_kwargs["system_instruction"] = system_instruction

    # Tools
    if config.get("tools"):
        sdk_tools = []
        for tool_group in config["tools"]:
            func_decls = tool_group.get("function_declarations", [])
            declarations = []
            for fd in func_decls:
                declarations.append(
                    types.FunctionDeclaration(
                        name=fd["name"],
                        description=fd.get("description", ""),
                        parameters=fd.get("parameters"),
                    )
                )
            sdk_tools.append(types.Tool(function_declarations=declarations))
        config_kwargs["tools"] = sdk_tools

    # Tool config
    if config.get("tool_config"):
        tc = config["tool_config"]
        fcc = tc.get("function_calling_config", {})
        config_kwargs["tool_config"] = types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(
                mode=fcc.get("mode", "AUTO"),
                allowed_function_names=fcc.get("allowed_function_names"),
            )
        )

    return types.GenerateContentConfig(**config_kwargs) if config_kwargs else None


def build_sdk_contents(provider_req):
    """Build SDK Content objects from the converter's provider request.

    Preserves thought and thoughtSignature fields which are required by
    thinking models (e.g. gemini-3-flash-preview) for tool call round-trips.
    """
    sdk_contents = []
    for content in provider_req.get("contents", []):
        sdk_parts = []
        for part in content.get("parts", []):
            if part.get("thought") is True:
                # Thought part: use Part with thought=True
                sdk_part = types.Part(
                    text=part.get("text", ""),
                    thought=True,
                )
                sdk_parts.append(sdk_part)
            elif "function_call" in part:
                fc = part["function_call"]
                sdk_part = types.Part.from_function_call(
                    name=fc["name"], args=fc.get("args", {})
                )
                # Preserve thoughtSignature if present
                if "thoughtSignature" in part:
                    sdk_part.thought_signature = part["thoughtSignature"]
                sdk_parts.append(sdk_part)
            elif "function_response" in part:
                fr = part["function_response"]
                sdk_parts.append(
                    types.Part.from_function_response(
                        name=fr["name"], response=fr.get("response", {})
                    )
                )
            elif "text" in part:
                sdk_part = types.Part.from_text(text=part["text"])
                # Preserve thoughtSignature if present
                if "thoughtSignature" in part:
                    sdk_part.thought_signature = part["thoughtSignature"]
                sdk_parts.append(sdk_part)
        sdk_contents.append(
            types.Content(role=content.get("role", "user"), parts=sdk_parts)
        )
    return sdk_contents


# ============================================================================
# Test 1: Non-stream basic text
# ============================================================================


def test_non_stream_basic():
    """IRRequest → provider → SDK API → response_from_provider → IRResponse."""
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

    # Build SDK objects
    sdk_contents = build_sdk_contents(provider_req)
    sdk_config = build_sdk_config(provider_req)

    # Call SDK API
    response = client.models.generate_content(
        model=google_model,
        contents=sdk_contents,
        config=sdk_config,
    )
    print(f"  SDK response type: {type(response).__name__}")

    # Provider response → IR (SDK returns Pydantic model)
    ir_response = converter.response_from_provider(cast(dict[str, Any], response))
    assert "choices" in ir_response
    assert len(ir_response["choices"]) >= 1

    msg = ir_response["choices"][0]["message"]
    text = get_text(msg)
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

    sdk_contents = build_sdk_contents(provider_req)
    sdk_config = build_sdk_config(provider_req)

    response = client.models.generate_content(
        model=google_model,
        contents=sdk_contents,
        config=sdk_config,
    )
    ir_response = converter.response_from_provider(cast(dict[str, Any], response))

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
        "tool_choice": {"mode": "none"},
    }

    provider_req_r2, warnings_r2 = converter.request_to_provider(ir_request_r2)
    print(f"  Round 2 warnings: {warnings_r2}")

    sdk_contents_r2 = build_sdk_contents(provider_req_r2)
    sdk_config_r2 = build_sdk_config(provider_req_r2)

    response_r2 = client.models.generate_content(
        model=google_model,
        contents=sdk_contents_r2,
        config=sdk_config_r2,
    )
    ir_response_r2 = converter.response_from_provider(cast(dict[str, Any], response_r2))

    final_msg = ir_response_r2["choices"][0]["message"]
    final_text = get_text(final_msg)
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
    sdk_contents = build_sdk_contents(provider_req)
    sdk_config = build_sdk_config(provider_req)

    response = client.models.generate_content(
        model=google_model,
        contents=sdk_contents,
        config=sdk_config,
    )

    # Provider → IR (SDK Pydantic model)
    ir_response = converter.response_from_provider(cast(dict[str, Any], response))
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
# Test 5: Multimodal tool result
# ============================================================================


def test_multimodal_tool_result():
    """Test tool returning multimodal content (text + image)."""
    section("Test 5: Multimodal tool result")

    ir_request: IRRequest = {
        "model": google_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Generate a bar chart for me."},
                ],
            },
        ],
        "tools": cast(list[ToolDefinition], multimodal_tools_spec),
        "tool_choice": {"mode": "tool", "tool_name": "generate_chart"},
    }

    provider_req, warnings = converter.request_to_provider(ir_request)
    print(f"  Round 1 warnings: {warnings}")

    sdk_contents = build_sdk_contents(provider_req)
    sdk_config = build_sdk_config(provider_req)

    response = client.models.generate_content(
        model=google_model,
        contents=sdk_contents,
        config=sdk_config,
    )
    ir_response = converter.response_from_provider(cast(dict[str, Any], response))

    assistant_msg = ir_response["choices"][0]["message"]
    tool_calls = extract_tool_calls(assistant_msg)
    text = get_text(assistant_msg)
    print(f"  Tool calls: {len(tool_calls)}")

    if tool_calls:
        tc = tool_calls[0]
        print(f"  Tool: {tc['tool_name']}({json.dumps(tc['tool_input'])})")
        assert tc["tool_name"] == "generate_chart"

        result = generate_chart(**tc["tool_input"])
        print(f"  Tool result type: {type(result).__name__}, length: {len(result)}")

        tool_result_msg = create_tool_result_message(tc["tool_call_id"], result)

        ir_request_r2: IRRequest = {
            "model": google_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Generate a bar chart for me."},
                    ],
                },
                assistant_msg,
                tool_result_msg,
            ],
            "tools": cast(list[ToolDefinition], multimodal_tools_spec),
            "tool_choice": {"mode": "none"},
        }

        provider_req_r2, warnings_r2 = converter.request_to_provider(ir_request_r2)
        print(f"  Round 2 warnings: {warnings_r2}")

        sdk_contents_r2 = build_sdk_contents(provider_req_r2)
        sdk_config_r2 = build_sdk_config(provider_req_r2)

        response_r2 = client.models.generate_content(
            model=google_model,
            contents=sdk_contents_r2,
            config=sdk_config_r2,
        )
        ir_response_r2 = converter.response_from_provider(
            cast(dict[str, Any], response_r2)
        )

        final_text = get_text(ir_response_r2["choices"][0]["message"])
        print(f"  Final response: {final_text[:120]}...")
        assert len(final_text) > 5
    else:
        print(
            f"  Model ignored forced tool_choice, responded with text: {text[:120]}..."
        )
        assert len(text) > 5

    ok("Multimodal tool result")
    return True


# ============================================================================
# Test 6: Image input with tool calls
# ============================================================================


def test_image_with_tool_calls():
    """Test image input combined with tool call capability."""
    section("Test 6: Image input with tool calls")

    ir_request: IRRequest = {
        "model": google_model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "This painting depicts a famous scene in Paris, France. "
                            "What is the current weather in Paris?"
                        ),
                    },
                    {"type": "image", "image_url": IMAGE_URL},
                ],
            },
        ],
        "tools": cast(list[ToolDefinition], tools_spec),
        "tool_choice": {"mode": "auto"},
    }

    provider_req, warnings = converter.request_to_provider(ir_request)
    print(f"  Warnings: {warnings}")

    sdk_contents = build_sdk_contents(provider_req)
    sdk_config = build_sdk_config(provider_req)

    response = client.models.generate_content(
        model=google_model,
        contents=sdk_contents,
        config=sdk_config,
    )
    ir_response = converter.response_from_provider(cast(dict[str, Any], response))

    assistant_msg = ir_response["choices"][0]["message"]
    tool_calls = extract_tool_calls(assistant_msg)
    text = get_text(assistant_msg)

    if tool_calls:
        print(f"  Tool calls: {len(tool_calls)}")
        tc = tool_calls[0]
        print(f"  Tool: {tc['tool_name']}({json.dumps(tc['tool_input'])})")

        result = execute_tool(tc)
        tool_result_msg = create_tool_result_message(tc["tool_call_id"], result)

        ir_request_r2: IRRequest = {
            "model": google_model,
            "messages": cast(
                list[Message],
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "This painting depicts a famous scene in Paris. "
                                    "What is the current weather in Paris?"
                                ),
                            },
                            {"type": "image", "image_url": IMAGE_URL},
                        ],
                    },
                    assistant_msg,
                    tool_result_msg,
                ],
            ),
            "tools": cast(list[ToolDefinition], tools_spec),
            "tool_choice": {"mode": "none"},
        }

        provider_req_r2, _ = converter.request_to_provider(ir_request_r2)
        sdk_contents_r2 = build_sdk_contents(provider_req_r2)
        sdk_config_r2 = build_sdk_config(provider_req_r2)

        response_r2 = client.models.generate_content(
            model=google_model,
            contents=sdk_contents_r2,
            config=sdk_config_r2,
        )
        ir_response_r2 = converter.response_from_provider(
            cast(dict[str, Any], response_r2)
        )
        final_text = get_text(ir_response_r2["choices"][0]["message"])
        print(f"  Final response: {final_text[:120]}...")
        assert len(final_text) > 5
    else:
        print(f"  Model responded with text (no tool call): {text[:120]}...")
        assert len(text) > 5

    ok("Image input with tool calls")
    return True


# ============================================================================
# Test 7: Multi-turn conversation
# ============================================================================


def test_multi_turn():
    """Test multi-turn conversation flow."""
    section("Test 7: Multi-turn conversation")

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
    sdk_contents = build_sdk_contents(provider_req)
    sdk_config = build_sdk_config(provider_req)

    response = client.models.generate_content(
        model=google_model,
        contents=sdk_contents,
        config=sdk_config,
    )
    ir_response = converter.response_from_provider(cast(dict[str, Any], response))
    assistant_msg_1 = ir_response["choices"][0]["message"]
    text_1 = get_text(assistant_msg_1)
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
    sdk_contents_2 = build_sdk_contents(provider_req_2)
    sdk_config_2 = build_sdk_config(provider_req_2)

    response_2 = client.models.generate_content(
        model=google_model,
        contents=sdk_contents_2,
        config=sdk_config_2,
    )
    ir_response_2 = converter.response_from_provider(cast(dict[str, Any], response_2))
    text_2 = get_text(ir_response_2["choices"][0]["message"])
    print(f"  Turn 2 response: {text_2[:80]}...")
    assert "alice" in text_2.lower()

    ok("Multi-turn conversation")
    return True


# ============================================================================
# Main
# ============================================================================


def run_all():
    print("\nGoogle GenAI Converter E2E Tests (SDK Version)")
    print(f"Model: {google_model}")
    print(f"SDK version: {genai.__version__}")

    tests = [
        ("Non-stream basic text", test_non_stream_basic),
        ("Non-stream with tool calls", test_non_stream_tool_calls),
        ("Multimodal tool result", test_multimodal_tool_result),
        ("Image with tool calls", test_image_with_tool_calls),
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
