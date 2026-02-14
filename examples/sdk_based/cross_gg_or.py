#!/usr/bin/env python3
"""Cross-provider multi-turn conversation: Google GenAI <-> OpenAI Responses (SDK).

Demonstrates LLMIR's ability to maintain conversation context across
different LLM providers using SDK clients. Odd turns use Google GenAI,
even turns use OpenAI Responses.

Conversation covers: text, images, and tool calls.

Usage:
    # Both providers require proxy in restricted networks.
    proxychains -q python examples/sdk_based/cross_gg_or.py

    # Or if both are directly accessible
    python examples/sdk_based/cross_gg_or.py

Environment variables:
    GOOGLE_API_KEY, GOOGLE_MODEL
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
    build_user_message,
    convert_image_urls_to_inline,
    get_google_config,
    get_openai_responses_config,
    print_assistant_response,
    print_tool_calls,
    print_turn_header,
    process_tool_calls,
)

from llmir import GoogleGenAIConverter, OpenAIResponsesConverter  # noqa: E402

# Initialize converters
gg_converter = GoogleGenAIConverter()
or_converter = OpenAIResponsesConverter()


def init_google_client():
    """Initialize Google GenAI SDK client and return (client, model).

    Returns:
        Tuple of (Google GenAI client, model name string).
    """
    from google import genai

    config = get_google_config()
    return genai.Client(api_key=config["api_key"]), config["model"]


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


def _build_sdk_config(provider_request: dict):
    """Build a Google SDK GenerateContentConfig from the converter's provider request.

    Extracts tools, tool_config, and generation parameters from the
    converter output and constructs the appropriate SDK config object.

    Args:
        provider_request: Provider request dict from request_to_provider.

    Returns:
        GenerateContentConfig instance, or None if no config needed.
    """
    from google.genai import types

    config = provider_request.get("config", {})
    config_kwargs = {}

    # Generation params
    for key in ("temperature", "max_output_tokens", "top_p", "top_k", "stop_sequences"):
        if key in config:
            config_kwargs[key] = config[key]

    # System instruction
    system_instruction = provider_request.get("system_instruction")
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


def _build_sdk_contents(provider_request: dict) -> list:
    """Build Google SDK Content objects from the converter's provider request.

    Args:
        provider_request: Provider request dict from request_to_provider.

    Returns:
        List of google.genai.types.Content objects.
    """
    from google.genai import types

    sdk_contents = []
    for content in provider_request.get("contents", []):
        sdk_parts = []
        for part in content.get("parts", []):
            if "function_call" in part:
                fc = part["function_call"]
                sdk_parts.append(
                    types.Part.from_function_call(
                        name=fc["name"], args=fc.get("args", {})
                    )
                )
            elif "function_response" in part:
                fr = part["function_response"]
                sdk_parts.append(
                    types.Part.from_function_response(
                        name=fr["name"], response=fr.get("response", {})
                    )
                )
            elif "inline_data" in part:
                # Image data
                inline = part["inline_data"]
                sdk_parts.append(
                    types.Part.from_bytes(
                        data=inline["data"],
                        mime_type=inline.get("mime_type", "image/jpeg"),
                    )
                )
            elif "file_data" in part:
                # File URI reference
                fd = part["file_data"]
                sdk_parts.append(
                    types.Part.from_uri(
                        file_uri=fd["file_uri"],
                        mime_type=fd.get("mime_type", "image/jpeg"),
                    )
                )
            elif "text" in part:
                sdk_parts.append(types.Part.from_text(text=part["text"]))
        sdk_contents.append(
            types.Content(role=content.get("role", "user"), parts=sdk_parts)
        )
    return sdk_contents


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


def send_google_genai(ir_messages, model, client):
    """Convert IR messages to Google GenAI format, send, and return IR assistant message.

    Image URLs in the IR messages are converted to inline base64 data
    before conversion, since Google GenAI SDK does not support image URLs.

    Args:
        ir_messages: List of IR messages representing the conversation history.
        model: Google GenAI model name.
        client: Google GenAI SDK client instance.

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

    # Build SDK objects from converter output
    sdk_contents = _build_sdk_contents(provider_request)
    sdk_config = _build_sdk_config(provider_request)

    response = client.models.generate_content(
        model=model,
        contents=sdk_contents,
        config=sdk_config,
    )

    # Google SDK response is a Pydantic model; converter._normalize handles it
    ir_response = gg_converter.response_from_provider(response)
    assistant_msg = ir_response["choices"][0]["message"]
    return assistant_msg


def send_openai_responses(ir_messages, model, client):
    """Convert IR messages to OpenAI Responses format, send, and return IR assistant message.

    Image parts are stripped from history to avoid OpenAI image download
    failures with certain URLs.

    Args:
        ir_messages: List of IR messages representing the conversation history.
        model: OpenAI Responses model name.
        client: OpenAI SDK client instance.

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

    response = client.responses.create(
        model=provider_request["model"],
        input=provider_request["input"],
        tools=provider_request.get("tools"),
        tool_choice=provider_request.get("tool_choice"),
    )

    ir_response = or_converter.response_from_provider(response.model_dump())
    assistant_msg = ir_response["choices"][0]["message"]
    return assistant_msg


def main():
    """Run cross-provider multi-turn conversation between Google GenAI and OpenAI Responses."""
    print("=" * 60)
    print("Cross-Provider Multi-Turn Conversation")
    print("Google GenAI <-> OpenAI Responses (SDK)")
    print("=" * 60)

    # Initialize clients
    gg_client, gg_model = init_google_client()
    or_client, or_model = init_openai_responses_client()

    providers = [
        ("Google GenAI", lambda msgs: send_google_genai(msgs, gg_model, gg_client)),
        (
            "OpenAI Responses",
            lambda msgs: send_openai_responses(msgs, or_model, or_client),
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
