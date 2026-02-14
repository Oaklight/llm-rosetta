#!/usr/bin/env python3
"""Cross-provider multi-turn conversation: Anthropic <-> Google GenAI (REST).

Demonstrates LLMIR's ability to maintain conversation context across
different LLM providers using raw HTTP requests via httpx.
Odd turns use Anthropic, even turns use Google GenAI.

Conversation covers: text, images, and tool calls.

Usage:
    # Google GenAI requires proxy in restricted networks;
    # Anthropic does not need proxy but proxychains won't hurt.
    proxychains -q python examples/rest_based/cross_an_gg_rest.py

    # Or if Google GenAI is directly accessible
    python examples/rest_based/cross_an_gg_rest.py

Environment variables:
    ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, ANTHROPIC_MODEL
    GOOGLE_API_KEY, GOOGLE_MODEL
"""

import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common import (  # noqa: E402
    CONVERSATION_TURNS,
    TOOLS_SPEC,
    build_user_message,
    convert_image_urls_to_inline,
    get_anthropic_config,
    get_google_config,
    print_assistant_response,
    print_tool_calls,
    print_turn_header,
    process_tool_calls,
)

from llmir import AnthropicConverter, GoogleGenAIConverter  # noqa: E402

# Initialize converters
an_converter = AnthropicConverter()
gg_converter = GoogleGenAIConverter()


def send_anthropic_rest(ir_messages: list, model: str, config: dict) -> dict:
    """Send request to Anthropic via REST API.

    Args:
        ir_messages: List of IR messages representing the conversation history.
        model: Anthropic model name.
        config: Dictionary with api_key and base_url.

    Returns:
        IR assistant message dict extracted from the response.
    """
    ir_request = {
        "model": model,
        "messages": ir_messages,
        "tools": TOOLS_SPEC,
        "tool_choice": {"mode": "auto"},
    }
    provider_request, warnings = an_converter.request_to_provider(ir_request)
    if warnings:
        print(f"  Warnings: {warnings}")

    headers = {
        "x-api-key": config["api_key"],
        "anthropic-version": "2023-06-01",
        "Content-Type": "application/json",
    }
    url = f"{config['base_url']}/v1/messages"

    response = httpx.post(url, json=provider_request, headers=headers, timeout=60)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        print(f"  HTTP Error {response.status_code}: {response.text}")
        raise

    ir_response = an_converter.response_from_provider(response.json())
    return ir_response["choices"][0]["message"]


def send_google_rest(ir_messages: list, model: str, config: dict) -> dict:
    """Send request to Google GenAI via REST API.

    Image URLs in the IR messages are converted to inline base64 data
    before conversion, since Google GenAI does not support image URLs.

    Args:
        ir_messages: List of IR messages representing the conversation history.
        model: Google GenAI model name.
        config: Dictionary with api_key.

    Returns:
        IR assistant message dict extracted from the response.
    """
    # Convert image URLs to inline base64 for Google compatibility
    safe_messages = convert_image_urls_to_inline(ir_messages)

    ir_request = {
        "model": model,
        "messages": safe_messages,
        "tools": TOOLS_SPEC,
        "tool_choice": {"mode": "auto"},
    }
    provider_request, warnings = gg_converter.request_to_provider(ir_request)
    if warnings:
        print(f"  Warnings: {warnings}")

    # Build REST API request body from converter output.
    # request_to_provider puts tools/tool_config inside "config", but the
    # REST API expects them at the top level.
    request_body: dict = {"contents": provider_request["contents"]}

    provider_config = provider_request.get("config", {})
    if provider_config.get("tools"):
        request_body["tools"] = provider_config["tools"]
    if provider_config.get("tool_config"):
        request_body["tool_config"] = provider_config["tool_config"]

    if "system_instruction" in provider_request:
        request_body["system_instruction"] = provider_request["system_instruction"]

    headers = {
        "x-goog-api-key": config["api_key"],
        "Content-Type": "application/json",
    }
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    response = httpx.post(url, json=request_body, headers=headers, timeout=60)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        print(f"  HTTP Error {response.status_code}: {response.text}")
        raise

    ir_response = gg_converter.response_from_provider(response.json())
    return ir_response["choices"][0]["message"]


def main():
    """Run cross-provider multi-turn conversation between Anthropic and Google GenAI via REST."""
    print("=" * 60)
    print("Cross-Provider Multi-Turn Conversation")
    print("Anthropic <-> Google GenAI (REST)")
    print("=" * 60)

    # Load configurations
    an_config = get_anthropic_config()
    gg_config = get_google_config()
    an_model = an_config["model"]
    gg_model = gg_config["model"]

    print(f"Anthropic:    model={an_model}, base_url={an_config['base_url']}")
    print(f"Google GenAI: model={gg_model}")

    providers = [
        (
            "Anthropic",
            lambda msgs: send_anthropic_rest(msgs, an_model, an_config),
        ),
        (
            "Google GenAI",
            lambda msgs: send_google_rest(msgs, gg_model, gg_config),
        ),
    ]

    # Shared IR message history
    ir_messages: list = []

    for turn_info in CONVERSATION_TURNS:
        turn = turn_info["turn"]
        provider_idx = turn_info["provider_index"]
        provider_name, send_fn = providers[provider_idx]

        # Build and append user message
        user_msg = build_user_message(turn_info)
        ir_messages.append(user_msg)

        description = "Image + Text" if turn_info.get("has_image") else "Text"
        if turn_info.get("expects_tool_call"):
            description += f" (expects {turn_info.get('expected_tool', 'tool call')})"
        print_turn_header(turn, provider_name, description)
        print(f"  User: {turn_info['user_message'][:100]}...")

        # Send to provider
        assistant_msg = send_fn(ir_messages)
        ir_messages.append(assistant_msg)

        # Print response
        print_assistant_response(assistant_msg)
        print_tool_calls(assistant_msg)

        # Handle tool calls
        if process_tool_calls(ir_messages, assistant_msg):
            print("  [Tool results added, sending follow-up...]")
            # Send again to get final response after tool execution
            assistant_msg = send_fn(ir_messages)
            ir_messages.append(assistant_msg)
            print_assistant_response(assistant_msg)

    print(f"\n{'=' * 60}")
    print(f"Conversation complete! Total IR messages: {len(ir_messages)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
