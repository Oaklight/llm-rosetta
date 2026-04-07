#!/usr/bin/env python3
"""Cross-provider multi-turn conversation: Anthropic <-> OpenAI Responses (SDK, Stream).

Demonstrates LLM-Rosetta's ability to maintain conversation context across
different LLM providers using SDK clients with streaming responses.
Odd turns use Anthropic, even turns use OpenAI Responses.

Conversation covers: text, images, and tool calls.

Usage:
    # OpenAI Responses requires proxy in restricted networks;
    # Anthropic does not need proxy.
    proxychains -q python examples/sdk_based/cross_an_or_stream.py

    # Or if both are directly accessible
    python examples/sdk_based/cross_an_or_stream.py

Environment variables:
    ANTHROPIC_API_KEY, ANTHROPIC_BASE_URL, ANTHROPIC_MODEL
    OPENAI_RESPONSES_API_KEY (fallback: OPENAI_API_KEY)
    OPENAI_RESPONSES_BASE_URL (fallback: OPENAI_BASE_URL)
    OPENAI_RESPONSES_MODEL (fallback: OPENAI_MODEL)
"""

import os
import sys

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from common import (  # noqa: E402
    CONVERSATION_TURNS,
    TOOLS_SPEC,
    accumulate_stream_to_assistant_message,
    build_user_message,
    get_anthropic_config,
    get_openai_responses_config,
    print_assistant_response,
    print_stream_event,
    print_tool_calls,
    print_turn_header,
    process_tool_calls,
)

from llm_rosetta import AnthropicConverter, OpenAIResponsesConverter  # noqa: E402
from llm_rosetta.converters.base.context import StreamContext  # noqa: E402

# Initialize converters
an_converter = AnthropicConverter()
or_converter = OpenAIResponsesConverter()


def init_anthropic_client():
    """Initialize Anthropic SDK client and return (client, model).

    Returns:
        Tuple of (Anthropic client, model name string).
    """
    import anthropic

    config = get_anthropic_config()
    return anthropic.Anthropic(api_key=config["api_key"]), config["model"]


def init_openai_responses_client():
    """Initialize OpenAI Responses SDK client and return (client, model).

    Returns:
        Tuple of (OpenAI client, model name string).
    """
    from openai import OpenAI

    config = get_openai_responses_config()
    return OpenAI(api_key=config["api_key"], base_url=config["base_url"]), config[
        "model"
    ]


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


def send_anthropic_stream(ir_messages, model, client):
    """Convert IR messages to Anthropic format, send with streaming, and return IR assistant message.

    Args:
        ir_messages: List of IR messages representing the conversation history.
        model: Anthropic model name.
        client: Anthropic SDK client instance.

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

    # SDK stream call using create(..., stream=True)
    stream = client.messages.create(**provider_request, stream=True)

    ctx = StreamContext()
    all_events = []
    for event in stream:
        ir_events = an_converter.stream_response_from_provider(
            event.model_dump(), context=ctx
        )
        for ir_event in ir_events:
            print_stream_event(ir_event)
            all_events.append(ir_event)

    # Accumulate into complete assistant message
    return accumulate_stream_to_assistant_message(all_events)


def send_openai_responses_stream(ir_messages, model, client):
    """Convert IR messages to OpenAI Responses format, send with streaming, and return IR assistant message.

    Image parts are stripped from history to avoid OpenAI image download
    failures with certain URLs.

    Args:
        ir_messages: List of IR messages representing the conversation history.
        model: OpenAI Responses model name.
        client: OpenAI SDK client instance.

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
    provider_request, warnings = or_converter.request_to_provider(ir_request)
    if warnings:
        print(f"  Warnings: {warnings}")

    # SDK stream call using responses.create with stream=True
    stream = client.responses.create(
        model=provider_request["model"],
        input=provider_request["input"],
        tools=provider_request.get("tools"),
        tool_choice=provider_request.get("tool_choice"),
        stream=True,
    )

    # Iterate SSE events and convert to IR events
    ctx = StreamContext()
    all_events = []
    for event in stream:
        ir_events = or_converter.stream_response_from_provider(
            event.model_dump(), context=ctx
        )
        for ir_event in ir_events:
            print_stream_event(ir_event)
            all_events.append(ir_event)

    # Accumulate into complete assistant message
    return accumulate_stream_to_assistant_message(all_events)


def main():
    """Run cross-provider multi-turn streaming conversation between Anthropic and OpenAI Responses."""
    print("=" * 60)
    print("Cross-Provider Multi-Turn Conversation (Stream)")
    print("Anthropic <-> OpenAI Responses (SDK)")
    print("=" * 60)

    # Initialize clients
    an_client, an_model = init_anthropic_client()
    or_client, or_model = init_openai_responses_client()

    providers = [
        (
            "Anthropic",
            lambda msgs: send_anthropic_stream(msgs, an_model, an_client),
        ),
        (
            "OpenAI Responses",
            lambda msgs: send_openai_responses_stream(msgs, or_model, or_client),
        ),
    ]

    # Shared IR message history
    ir_messages = []

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
