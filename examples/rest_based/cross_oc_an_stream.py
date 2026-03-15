#!/usr/bin/env python3
"""Cross-provider multi-turn conversation: OpenAI Chat <-> Anthropic (REST, Stream).

Demonstrates LLM-Rosetta's ability to maintain conversation context across
different LLM providers using raw HTTP requests via httpx with streaming.
Odd turns use OpenAI Chat, even turns use Anthropic.

Conversation covers: text, images, and tool calls.

Usage:
    # OpenAI requires proxy in restricted networks
    proxychains -q python examples/rest_based/cross_oc_an_rest_stream.py

    # Or if OpenAI is directly accessible
    python examples/rest_based/cross_oc_an_rest_stream.py

Environment variables:
    OPENAI_API_KEY, OPENAI_BASE_URL, OPENAI_MODEL
    ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, ANTHROPIC_MODEL
"""

import json
import os
import sys

import httpx
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common import (  # noqa: E402
    CONVERSATION_TURNS,
    TOOLS_SPEC,
    accumulate_stream_to_assistant_message,
    build_user_message,
    get_anthropic_config,
    get_openai_chat_config,
    print_assistant_response,
    print_stream_event,
    print_tool_calls,
    print_turn_header,
    process_tool_calls,
)

from llm_rosetta import AnthropicConverter, OpenAIChatConverter  # noqa: E402
from llm_rosetta.converters.base.stream_context import StreamContext  # noqa: E402

# Initialize converters
oc_converter = OpenAIChatConverter()
an_converter = AnthropicConverter()


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


def send_openai_chat_rest_stream(ir_messages: list, model: str, config: dict) -> dict:
    """Send streaming request to OpenAI Chat via REST API.

    Image parts are stripped from history to avoid OpenAI image download
    failures with certain URLs.

    Args:
        ir_messages: List of IR messages representing the conversation history.
        model: OpenAI model name.
        config: Dictionary with api_key and base_url.

    Returns:
        IR assistant message dict accumulated from stream events.
    """
    # Strip images from history to avoid OpenAI download failures
    safe_messages = _strip_images(ir_messages)
    ir_request = {
        "model": model,
        "messages": safe_messages,
        "tools": TOOLS_SPEC,
        "tool_choice": {"mode": "auto"},
    }
    provider_request, warnings = oc_converter.request_to_provider(ir_request)
    if warnings:
        print(f"  Warnings: {warnings}")

    headers = {
        "Authorization": f"Bearer {config['api_key']}",
        "Content-Type": "application/json",
    }
    url = f"{config['base_url']}/chat/completions"
    body = {
        **provider_request,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    ctx = StreamContext()
    all_events = []

    with httpx.stream(
        "POST", url, json=body, headers=headers, timeout=60.0
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: ") :]
            if data.strip() == "[DONE]":
                break
            chunk = json.loads(data)
            ir_events = oc_converter.stream_response_from_provider(chunk, context=ctx)
            for event in ir_events:
                print_stream_event(event)
                all_events.append(event)

    return accumulate_stream_to_assistant_message(all_events)


def send_anthropic_rest_stream(ir_messages: list, model: str, config: dict) -> dict:
    """Send streaming request to Anthropic via REST API.

    Args:
        ir_messages: List of IR messages representing the conversation history.
        model: Anthropic model name.
        config: Dictionary with api_key and base_url.

    Returns:
        IR assistant message dict accumulated from stream events.
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
    body = {**provider_request, "stream": True}

    ctx = StreamContext()
    all_events = []

    with httpx.stream(
        "POST", url, json=body, headers=headers, timeout=60.0
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue
            if line.startswith("event: "):
                continue  # Skip event lines; type is in the data JSON
            if line.startswith("data: "):
                data = line[len("data: ") :]
                event_data = json.loads(data)
                ir_events = an_converter.stream_response_from_provider(
                    event_data, context=ctx
                )
                for ir_event in ir_events:
                    print_stream_event(ir_event)
                    all_events.append(ir_event)

    return accumulate_stream_to_assistant_message(all_events)


def main():
    """Run cross-provider multi-turn streaming conversation between OpenAI Chat and Anthropic via REST."""
    print("=" * 60)
    print("Cross-Provider Multi-Turn Conversation (Stream)")
    print("OpenAI Chat <-> Anthropic (REST)")
    print("=" * 60)

    # Load configurations
    oc_config = get_openai_chat_config()
    an_config = get_anthropic_config()
    oc_model = oc_config["model"]
    an_model = an_config["model"]

    print(f"OpenAI Chat: model={oc_model}, base_url={oc_config['base_url']}")
    print(f"Anthropic:   model={an_model}, base_url={an_config['base_url']}")

    providers = [
        (
            "OpenAI Chat",
            lambda msgs: send_openai_chat_rest_stream(msgs, oc_model, oc_config),
        ),
        (
            "Anthropic",
            lambda msgs: send_anthropic_rest_stream(msgs, an_model, an_config),
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

        # Send to provider (streaming)
        assistant_msg = send_fn(ir_messages)
        ir_messages.append(assistant_msg)

        # Print response summary
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
