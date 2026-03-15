"""Common resources for cross-provider multi-turn conversation examples.

This module provides shared tool definitions, mock tool execution,
conversation turn definitions, and helper functions used by all
cross-provider example scripts.
"""

import base64
import copy
import hashlib
import json
import logging
import os
import tempfile
import time
from pathlib import Path

import httpx

from llm_rosetta.types.ir import (
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

CONVERSATION_TURNS: list[dict] = [
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


# ============================================================================
# Google GenAI image URL workaround
# ============================================================================

logger = logging.getLogger(__name__)

# Persistent cache directory under system temp
_IMAGE_CACHE_DIR = Path(tempfile.gettempdir()) / "llm_rosetta_image_cache"

# User-Agent per Wikimedia policy: include project URL for contact
_USER_AGENT = (
    "LLM-Rosetta-Example/1.0 (https://github.com/llm-rosetta/llm-rosetta; bot-contact) "
    "python-httpx/" + httpx.__version__
)

# Retry settings for HTTP 429
_MAX_RETRIES = 3
_INITIAL_BACKOFF = 2.0  # seconds


def _get_cache_path(url: str) -> Path:
    """Return the cache file path for a given URL.

    Uses SHA-256 hash of the URL as the filename to avoid filesystem issues.

    Args:
        url: The image URL.

    Returns:
        Path to the cache file.
    """
    url_hash = hashlib.sha256(url.encode()).hexdigest()
    return _IMAGE_CACHE_DIR / url_hash


def _load_from_cache(url: str) -> tuple[str, str] | None:
    """Try to load image data from local file cache.

    The cache stores raw image bytes in ``<hash>.data`` and the MIME type
    in ``<hash>.meta``.

    Args:
        url: The image URL to look up.

    Returns:
        Tuple of (base64-encoded data, mime_type) if cached, None otherwise.
    """
    cache_path = _get_cache_path(url)
    data_file = cache_path.with_suffix(".data")
    meta_file = cache_path.with_suffix(".meta")

    if data_file.exists() and meta_file.exists():
        try:
            raw_bytes = data_file.read_bytes()
            mime_type = meta_file.read_text().strip()
            b64 = base64.b64encode(raw_bytes).decode("utf-8")
            logger.debug("Cache hit for %s", url)
            return b64, mime_type
        except OSError:
            # Corrupted cache entry; fall through to download
            pass
    return None


def _save_to_cache(url: str, raw_bytes: bytes, mime_type: str) -> None:
    """Save downloaded image data to local file cache.

    Args:
        url: The image URL (used as cache key).
        raw_bytes: Raw image bytes.
        mime_type: MIME type of the image.
    """
    try:
        _IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache_path = _get_cache_path(url)
        cache_path.with_suffix(".data").write_bytes(raw_bytes)
        cache_path.with_suffix(".meta").write_text(mime_type)
        logger.debug("Cached image for %s", url)
    except OSError as e:
        logger.warning("Failed to cache image for %s: %s", url, e)


def download_image_as_base64(url: str) -> tuple[str, str]:
    """Download image from URL and return (base64_data, mime_type).

    Features:
        - Local file cache (under system temp dir) to avoid redundant downloads.
        - Proper User-Agent header per Wikimedia policy.
        - Exponential backoff retry on HTTP 429 (rate limit) errors.

    Args:
        url: The image URL to download.

    Returns:
        Tuple of (base64-encoded image data, MIME type string).

    Raises:
        httpx.HTTPStatusError: If the request fails after all retries.
    """
    # Check local file cache first
    cached = _load_from_cache(url)
    if cached is not None:
        return cached

    # Download with retry logic for 429 errors
    last_exc: httpx.HTTPStatusError | None = None
    for attempt in range(_MAX_RETRIES):
        response = httpx.get(
            url,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT},
            timeout=60.0,
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as e:
            if response.status_code == 429 and attempt < _MAX_RETRIES - 1:
                # Use Retry-After header if available, otherwise exponential backoff
                retry_after = response.headers.get("retry-after")
                if retry_after and retry_after.isdigit():
                    wait = float(retry_after)
                else:
                    wait = _INITIAL_BACKOFF * (2**attempt)
                logger.warning(
                    "HTTP 429 for %s, retrying in %.1fs (attempt %d/%d)",
                    url,
                    wait,
                    attempt + 1,
                    _MAX_RETRIES,
                )
                time.sleep(wait)
                last_exc = e
                continue
            raise
        else:
            # Success
            content_type = response.headers.get("content-type", "image/jpeg")
            raw_bytes = response.content

            # Save to cache for future use
            _save_to_cache(url, raw_bytes, content_type)

            b64 = base64.b64encode(raw_bytes).decode("utf-8")
            return b64, content_type

    # All retries exhausted (should not normally reach here)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Failed to download image from {url}")


def convert_image_urls_to_inline(ir_messages: list) -> list:
    """Convert image URL parts to inline base64 data for Google GenAI compatibility.

    Google GenAI SDK does not directly support image URLs. This function
    creates a copy of the messages and replaces any ImagePart with
    ``image_url`` to use ``image_data`` (base64) instead.

    The original ``ir_messages`` list is NOT modified.

    Args:
        ir_messages: List of IR messages (not modified).

    Returns:
        New list of IR messages with image URLs replaced by inline data.
    """
    converted = []
    for msg in ir_messages:
        content = msg.get("content")
        if not isinstance(content, list):
            converted.append(msg)
            continue

        needs_conversion = any(
            p.get("type") == "image" and p.get("image_url") for p in content
        )
        if not needs_conversion:
            converted.append(msg)
            continue

        new_content = []
        for part in content:
            if part.get("type") == "image" and part.get("image_url"):
                b64, mime = download_image_as_base64(part["image_url"])
                new_content.append(
                    {
                        "type": "image",
                        "image_data": {
                            "data": b64,
                            "media_type": mime,
                        },
                    }
                )
            else:
                new_content.append(copy.deepcopy(part))
        converted.append({**msg, "content": new_content})

    return converted


# ============================================================================
# Stream helper functions
# ============================================================================


def accumulate_stream_to_assistant_message(ir_events: list) -> dict:
    """Accumulate IR stream events into a complete IR assistant message.

    Processes TextDeltaEvent, ToolCallStartEvent, and ToolCallDeltaEvent
    to reconstruct the full assistant message from streamed chunks.

    Args:
        ir_events: List of IR stream event dicts.

    Returns:
        An IR assistant message dict with role and content parts.
    """
    # Accumulate text by choice_index
    text_buffers: dict[int, list[str]] = {}
    # Accumulate tool calls in order: list of (tool_call_id, tool_name, args_buffer)
    tool_calls: list[tuple[str, str, list[str]]] = {}
    # Track tool_call_id -> index in tool_calls for delta appending
    tool_call_index_map: dict[str, int] = {}
    # Track the last registered tool_call_id for Anthropic compatibility
    last_tool_call_id: str = ""

    tool_calls = []

    for event in ir_events:
        event_type = event.get("type", "")

        if event_type == "text_delta":
            choice_idx = event.get("choice_index", 0)
            if choice_idx not in text_buffers:
                text_buffers[choice_idx] = []
            text_buffers[choice_idx].append(event["text"])

        elif event_type == "tool_call_start":
            tc_id = event["tool_call_id"]
            tc_name = event["tool_name"]
            idx = len(tool_calls)
            tool_calls.append((tc_id, tc_name, []))
            tool_call_index_map[tc_id] = idx
            last_tool_call_id = tc_id

        elif event_type == "tool_call_delta":
            tc_id = event["tool_call_id"]
            # Anthropic may send empty tool_call_id in deltas
            if not tc_id:
                tc_id = last_tool_call_id
            if tc_id in tool_call_index_map:
                idx = tool_call_index_map[tc_id]
                tool_calls[idx][2].append(event["arguments_delta"])

    # Assemble content parts
    content: list[dict] = []

    # Add text parts (sorted by choice_index)
    for choice_idx in sorted(text_buffers.keys()):
        full_text = "".join(text_buffers[choice_idx])
        if full_text:
            content.append({"type": "text", "text": full_text})

    # Add tool call parts
    for tc_id, tc_name, args_fragments in tool_calls:
        args_str = "".join(args_fragments)
        try:
            tool_input = json.loads(args_str) if args_str else {}
        except json.JSONDecodeError:
            tool_input = {}
        content.append(
            {
                "type": "tool_call",
                "tool_call_id": tc_id,
                "tool_name": tc_name,
                "tool_input": tool_input,
            }
        )

    return {"role": "assistant", "content": content}


def print_stream_event(event: dict) -> None:
    """Print a stream event in real-time for demonstration purposes.

    Handles different event types with appropriate formatting:
    - stream_start: prints model info
    - text_delta: prints text fragment inline
    - tool_call_start: prints tool name
    - tool_call_delta: prints argument fragment inline
    - finish: prints finish reason
    - stream_end: prints end marker

    Args:
        event: An IR stream event dict.
    """
    event_type = event.get("type", "")

    if event_type == "stream_start":
        print(f"[Stream started: model={event.get('model', '')}]")
    elif event_type == "text_delta":
        print(event["text"], end="", flush=True)
    elif event_type == "tool_call_start":
        print(f"\n[Tool call: {event['tool_name']}]")
    elif event_type == "tool_call_delta":
        print(event["arguments_delta"], end="", flush=True)
    elif event_type == "finish":
        finish_reason = event.get("finish_reason", {})
        reason = (
            finish_reason.get("reason", "unknown")
            if isinstance(finish_reason, dict)
            else str(finish_reason)
        )
        print(f"\n[Finished: {reason}]")
    elif event_type == "stream_end":
        print("[Stream ended]")
