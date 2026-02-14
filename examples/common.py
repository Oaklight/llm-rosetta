"""Common resources for cross-provider multi-turn conversation examples.

This module provides shared tool definitions, mock tool execution,
conversation turn definitions, and helper functions used by all
cross-provider example scripts.
"""

import json
import os
from typing import Dict, List

from llmir.types.ir import (
    UserMessage,
    create_tool_result_message,
    extract_text_content,
    extract_tool_calls,
)

# ============================================================================
# Tool definitions (IR ToolDefinition format)
# ============================================================================

TOOLS_SPEC = [
    {
        "type": "function",
        "name": "get_current_weather",
        "description": "Get the current weather in a given location",
        "parameters": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city and state, e.g. San Francisco, CA",
                },
                "unit": {
                    "type": "string",
                    "enum": ["celsius", "fahrenheit"],
                    "description": "Temperature unit",
                },
            },
            "required": ["location"],
        },
    },
    {
        "type": "function",
        "name": "get_flight_info",
        "description": "Get flight information between two cities",
        "parameters": {
            "type": "object",
            "properties": {
                "origin": {
                    "type": "string",
                    "description": "Departure city",
                },
                "destination": {
                    "type": "string",
                    "description": "Arrival city",
                },
                "date": {
                    "type": "string",
                    "description": "Travel date in YYYY-MM-DD format",
                },
            },
            "required": ["origin", "destination"],
        },
    },
]

# ============================================================================
# Image URLs
# ============================================================================

IMAGE_GOLDEN_GATE = "https://upload.wikimedia.org/wikipedia/commons/thumb/0/0c/GoldenGateBridge-001.jpg/1280px-GoldenGateBridge-001.jpg"
IMAGE_TOKYO_TOWER = "https://www.japan-guide.com/g18/3009_01.jpg"

# ============================================================================
# Mock tool execution
# ============================================================================


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a mock tool and return result as string.

    Args:
        tool_name: Name of the tool to execute.
        tool_input: Input parameters for the tool.

    Returns:
        JSON string containing the mock tool result.
    """
    if tool_name == "get_current_weather":
        location = tool_input.get("location", "Unknown")
        unit = tool_input.get("unit", "fahrenheit")
        return json.dumps(
            {
                "location": location,
                "temperature": 72 if unit == "fahrenheit" else 22,
                "unit": unit,
                "condition": "sunny",
                "humidity": "45%",
            }
        )
    elif tool_name == "get_flight_info":
        origin = tool_input.get("origin", "Unknown")
        destination = tool_input.get("destination", "Unknown")
        date = tool_input.get("date", "2025-03-15")
        return json.dumps(
            {
                "flights": [
                    {
                        "airline": "United",
                        "flight": "UA123",
                        "departure": "08:00",
                        "arrival": "11:30",
                        "price": "$350",
                    },
                    {
                        "airline": "Delta",
                        "flight": "DL456",
                        "departure": "14:00",
                        "arrival": "17:30",
                        "price": "$420",
                    },
                ],
                "origin": origin,
                "destination": destination,
                "date": date,
            }
        )
    else:
        return json.dumps({"error": f"Unknown tool: {tool_name}"})


# ============================================================================
# Conversation turn definitions
# ============================================================================

CONVERSATION_TURNS: List[Dict] = [
    {
        "turn": 1,
        "provider_index": 0,  # Provider A
        "user_message": (
            "I'm planning a trip and need help comparing destinations. "
            "Let's start with San Francisco. "
            "What do you know about it as a travel destination?"
        ),
        "has_image": False,
        "expects_tool_call": False,
    },
    {
        "turn": 2,
        "provider_index": 1,  # Provider B
        "user_message": (
            "I found this photo from my last visit. "
            "Can you tell me what landmark this is "
            "and share some interesting facts about it?"
        ),
        "has_image": True,
        "image_url": IMAGE_GOLDEN_GATE,
        "expects_tool_call": False,
    },
    {
        "turn": 3,
        "provider_index": 0,  # Provider A
        "user_message": (
            "Great! Can you check the current weather in San Francisco for me?"
        ),
        "has_image": False,
        "expects_tool_call": True,
        "expected_tool": "get_current_weather",
    },
    {
        "turn": 4,
        "provider_index": 1,  # Provider B
        "user_message": (
            "Now let's look at flights. "
            "Can you find flights from New York to San Francisco?"
        ),
        "has_image": False,
        "expects_tool_call": True,
        "expected_tool": "get_flight_info",
    },
    {
        "turn": 5,
        "provider_index": 0,  # Provider A
        "user_message": (
            "Based on the weather and flight info, "
            "give me a brief summary of what we know so far "
            "about visiting San Francisco."
        ),
        "has_image": False,
        "expects_tool_call": False,
    },
    {
        "turn": 6,
        "provider_index": 1,  # Provider B
        "user_message": (
            "Now let's consider Tokyo as an alternative. "
            "Here's a photo from Tokyo. What landmark is this, "
            "and how does Tokyo compare to San Francisco "
            "as a travel destination?"
        ),
        "has_image": True,
        "image_url": IMAGE_TOKYO_TOWER,
        "expects_tool_call": False,
    },
    {
        "turn": 7,
        "provider_index": 0,  # Provider A
        "user_message": (
            "Can you check the current weather in Tokyo so we can compare?"
        ),
        "has_image": False,
        "expects_tool_call": True,
        "expected_tool": "get_current_weather",
    },
    {
        "turn": 8,
        "provider_index": 1,  # Provider B
        "user_message": (
            "Based on everything we've discussed - landmarks, weather, "
            "flights, and your knowledge of both cities - "
            "which destination would you recommend and why?"
        ),
        "has_image": False,
        "expects_tool_call": False,
    },
]

# ============================================================================
# Helper functions
# ============================================================================


def build_user_message(turn_info: dict) -> UserMessage:
    """Build an IR UserMessage from turn info.

    Args:
        turn_info: A dictionary from CONVERSATION_TURNS describing the turn.

    Returns:
        An IR UserMessage with appropriate content parts.
    """
    parts: list = []
    if turn_info.get("has_image"):
        parts.append({"type": "image", "image_url": turn_info["image_url"]})
    parts.append({"type": "text", "text": turn_info["user_message"]})
    return {"role": "user", "content": parts}


def process_tool_calls(ir_messages: list, assistant_message: dict) -> bool:
    """Check for tool calls in assistant message, execute them, and append results.

    Args:
        ir_messages: The conversation message list to append tool results to.
        assistant_message: The assistant's response message to check for tool calls.

    Returns:
        True if tool calls were found and processed, False otherwise.
    """
    tool_calls = extract_tool_calls(assistant_message)
    if not tool_calls:
        return False

    for tc in tool_calls:
        result = execute_tool(tc["tool_name"], tc["tool_input"])
        tool_msg = create_tool_result_message(tc["tool_call_id"], result)
        ir_messages.append(tool_msg)

    return True


def print_turn_header(turn: int, provider_name: str, description: str) -> None:
    """Print a formatted turn header.

    Args:
        turn: The turn number.
        provider_name: Name of the provider handling this turn.
        description: Brief description of the turn.
    """
    print(f"\n{'=' * 60}")
    print(f"Turn {turn}: {provider_name}")
    print(f"  {description}")
    print(f"{'=' * 60}")


def print_assistant_response(message: dict) -> None:
    """Print the assistant's response text.

    Args:
        message: The assistant message to extract and print text from.
    """
    text = extract_text_content(message)
    if text:
        print(f"  Assistant: {text[:200]}{'...' if len(text) > 200 else ''}")


def print_tool_calls(message: dict) -> None:
    """Print tool calls from assistant message.

    Args:
        message: The assistant message to extract and print tool calls from.
    """
    tool_calls = extract_tool_calls(message)
    for tc in tool_calls:
        print(f"  Tool Call: {tc['tool_name']}({json.dumps(tc['tool_input'])})")


# ============================================================================
# Provider configuration loaders
# ============================================================================


def get_openai_chat_config() -> dict:
    """Get OpenAI Chat API configuration from environment.

    Returns:
        Dictionary with api_key, base_url, and model settings.
    """
    return {
        "api_key": os.environ.get("OPENAI_API_KEY", ""),
        "base_url": os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        "model": os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
    }


def get_openai_responses_config() -> dict:
    """Get OpenAI Responses API configuration from environment.

    Returns:
        Dictionary with api_key, base_url, and model settings.
    """
    return {
        "api_key": os.environ.get(
            "OPENAI_RESPONSES_API_KEY",
            os.environ.get("OPENAI_API_KEY", ""),
        ),
        "base_url": os.environ.get(
            "OPENAI_RESPONSES_BASE_URL",
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        ),
        "model": os.environ.get(
            "OPENAI_RESPONSES_MODEL",
            os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
        ),
    }


def get_anthropic_config() -> dict:
    """Get Anthropic API configuration from environment.

    Returns:
        Dictionary with api_key, base_url, and model settings.
    """
    return {
        "api_key": os.environ.get("ANTHROPIC_API_KEY", ""),
        "base_url": os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        "model": os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
    }


def get_google_config() -> dict:
    """Get Google GenAI API configuration from environment.

    Returns:
        Dictionary with api_key and model settings.
    """
    return {
        "api_key": os.environ.get("GOOGLE_API_KEY", ""),
        "model": os.environ.get("GOOGLE_MODEL", "gemini-2.0-flash"),
    }
