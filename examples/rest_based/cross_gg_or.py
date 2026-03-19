#!/usr/bin/env python3
"""Cross-provider multi-turn conversation: Google GenAI <-> OpenAI Responses (REST).

Demonstrates LLM-Rosetta's ability to maintain conversation context across
different LLM providers using raw HTTP requests via httpx.
Odd turns use Google GenAI, even turns use OpenAI Responses.

Conversation covers: text, images, and tool calls.

Usage:
    # Both providers require proxy in restricted networks
    proxychains -q python examples/rest_based/cross_gg_or_rest.py

    # Or if both are directly accessible
    python examples/rest_based/cross_gg_or_rest.py

Environment variables:
    GOOGLE_API_KEY, GOOGLE_MODEL
    OPENAI_RESPONSES_API_KEY (fallback: OPENAI_API_KEY)
    OPENAI_RESPONSES_BASE_URL (fallback: OPENAI_BASE_URL)
    OPENAI_RESPONSES_MODEL (fallback: OPENAI_MODEL)
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
    get_google_config,
    get_openai_responses_config,
    print_assistant_response,
    print_tool_calls,
    print_turn_header,
    process_tool_calls,
)

from llm_rosetta import GoogleGenAIConverter, OpenAIResponsesConverter  # noqa: E402

# Initialize converters
gg_converter = GoogleGenAIConverter()
or_converter = OpenAIResponsesConverter()


def _strip_images(ir_messages: list) -> list:
    """Strip image parts from IR messages to avoid provider download failures.

    Some providers (e.g. OpenAI) may fail to download certain image URLs
    present in conversation history. This helper removes image parts while
    preserving all other content.

    Args:
        ir_messages: List of IR messages.

    Returns:
        New list of IR messages with image parts removed.
    """
    cleaned = []
    for msg in ir_messages:
        content = msg.get("content")
        if not isinstance(content, list):
            cleaned.append(msg)
            continue
        new_content = [p for p in content if p.get("type") != "image"]
        if new_content:
            cleaned.append({**msg, "content": new_content})
        else:
            # Keep message with empty text to preserve conversation structure
            cleaned.append({**msg, "content": [{"type": "text", "text": ""}]})
    return cleaned


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
    request_body, warnings = gg_converter.request_to_provider(
        ir_request, output_format="rest"
    )
    if warnings:
        print(f"  Warnings: {warnings}")

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


def send_openai_responses_rest(ir_messages: list, model: str, config: dict) -> dict:
    """Send request to OpenAI Responses API via REST.

    Image parts are stripped from history to avoid OpenAI image download
    failures with certain URLs.

    Args:
        ir_messages: List of IR messages representing the conversation history.
        model: OpenAI Responses model name.
        config: Dictionary with api_key and base_url.

    Returns:
        IR assistant message dict extracted from the response.
    """
    # Strip images from history to avoid OpenAI download failures
    safe_messages = _strip_images(ir_messages)
    ir_request = {
        "model": model,
        "messages": safe_messages,
        "tools": TOOLS_SPEC,
        "tool_choice": {"mode": "auto"},
    }
    provider_request, warnings = or_converter.request_to_provider(ir_request)
    if warnings:
        print(f"  Warnings: {warnings}")

    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }
    url = f"{config['base_url']}/responses"

    response = httpx.post(url, json=provider_request, headers=headers, timeout=60)
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError:
        print(f"  HTTP Error {response.status_code}: {response.text}")
        raise

    ir_response = or_converter.response_from_provider(response.json())
    return ir_response["choices"][0]["message"]


def main():
    """Run cross-provider multi-turn conversation between Google GenAI and OpenAI Responses via REST."""
    print("=" * 60)
    print("Cross-Provider Multi-Turn Conversation")
    print("Google GenAI <-> OpenAI Responses (REST)")
    print("=" * 60)

    # Load configurations
    gg_config = get_google_config()
    or_config = get_openai_responses_config()
    gg_model = gg_config["model"]
    or_model = or_config["model"]

    print(f"Google GenAI:     model={gg_model}")
    print(f"OpenAI Responses: model={or_model}, base_url={or_config['base_url']}")

    providers = [
        (
            "Google GenAI",
            lambda msgs: send_google_rest(msgs, gg_model, gg_config),
        ),
        (
            "OpenAI Responses",
            lambda msgs: send_openai_responses_rest(msgs, or_model, or_config),
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

        # Handle tool calls (loop to handle consecutive tool calls)
        while process_tool_calls(ir_messages, assistant_msg):
            print("  [Tool results added, sending follow-up...]")
            # Send again to get final response after tool execution
            assistant_msg = send_fn(ir_messages)
            ir_messages.append(assistant_msg)
            print_assistant_response(assistant_msg)
            print_tool_calls(assistant_msg)

    print(f"\n{'=' * 60}")
    print(f"Conversation complete! Total IR messages: {len(ir_messages)}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
