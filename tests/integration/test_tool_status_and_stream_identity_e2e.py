"""E2E checks for the incomplete tool-status mapping and streamed tool identity.

Two fixes are exercised against real upstreams:

* an OpenAI Responses ``function_call_output`` with ``status="incomplete"``
  must reach an Anthropic target as ``is_error=True`` (and stay ``incomplete``
  on a same-format Responses round-trip);
* a streamed tool call must keep its ID, name and complete arguments when a
  real Chat Completions or Anthropic upstream is converted to Messages.

Requires credentials in the project ``.env``:

- ``ANTHROPIC_API_KEY`` / ``ANTHROPIC_BASE_URL`` / ``ANTHROPIC_MODEL``
- ``OPENAI_API_KEY`` / ``OPENAI_BASE_URL`` / ``OPENAI_MODEL``
- ``OPENAI_RESPONSES_API_KEY`` / ``OPENAI_RESPONSES_BASE_URL`` /
  ``OPENAI_RESPONSES_MODEL``

Usage:
    conda activate llm-rosetta
    pytest tests/integration/test_tool_status_and_stream_identity_e2e.py -v -s
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from typing import Any

import dotenv
import httpx
import pytest

dotenv.load_dotenv(
    os.path.join(os.path.dirname(__file__), "..", "..", ".env"), override=True
)

from llm_rosetta import ConversionPipeline

TIMEOUT = 120.0


def _env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name) or default


def _anthropic_headers() -> dict[str, str]:
    return {
        "x-api-key": _env("ANTHROPIC_API_KEY", ""),
        "anthropic-version": "2023-06-01",
        "content-type": "application/json",
    }


def _iter_sse(
    url: str, headers: dict[str, str], body: dict[str, Any]
) -> Iterator[dict[str, Any]]:
    with httpx.stream(
        "POST", url, headers=headers, json=body, timeout=TIMEOUT
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith("data:"):
                continue
            data = line[len("data:") :].strip()
            if not data or data == "[DONE]":
                continue
            yield json.loads(data)


def _tool_events(events: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str]:
    """Collect tool blocks and the terminal stop reason from Messages events."""
    calls: dict[int, dict[str, Any]] = {}
    stop_reason = ""
    for event in events:
        if event.get("type") == "content_block_start":
            block = event.get("content_block", {})
            if block.get("type") == "tool_use":
                calls[event["index"]] = {
                    "id": block.get("id", ""),
                    "name": block.get("name", ""),
                    "arguments": "",
                }
        elif event.get("type") == "content_block_delta":
            delta = event.get("delta", {})
            if delta.get("type") == "input_json_delta":
                calls[event["index"]]["arguments"] += delta.get("partial_json", "")
        elif event.get("type") == "message_delta":
            stop_reason = event.get("delta", {}).get("stop_reason") or stop_reason
    return list(calls.values()), stop_reason


def _tool_request(model: str) -> dict[str, Any]:
    return {
        "model": model,
        "max_tokens": 1024,
        "stream": True,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get weather for a city",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        "tool_choice": {"type": "tool", "name": "get_weather"},
        "messages": [{"role": "user", "content": "Call get_weather for Paris."}],
    }


def _assert_single_paris_call(events: list[dict[str, Any]]) -> None:
    calls, stop_reason = _tool_events(events)
    assert calls, events
    assert len(calls) == 1, calls
    call = calls[0]
    assert call["id"], call
    assert call["name"] == "get_weather", call
    assert json.loads(call["arguments"]) == {"city": "Paris"}, call
    assert stop_reason == "tool_use", stop_reason


def _rows(status: str) -> dict[str, Any]:
    """The #1006 reproducer: an incomplete function output in history."""
    return {
        "model": "gpt-4o",
        "tools": [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather for a city",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
        "input": [
            {"type": "message", "role": "user", "content": "Weather in Atlantis?"},
            {
                "type": "function_call",
                "call_id": "call_err1",
                "name": "get_weather",
                "arguments": '{"city":"Atlantis"}',
            },
            {
                "type": "function_call_output",
                "call_id": "call_err1",
                "status": status,
                "output": "City not found",
            },
            {
                "type": "message",
                "role": "user",
                "content": "Did the get_weather tool call fail? Answer yes or no.",
            },
        ],
        "max_output_tokens": 256,
    }


def test_incomplete_tool_output_reaches_anthropic_as_error() -> None:
    if not _env("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    converted = ConversionPipeline("openai_responses", "messages").convert_request(
        _rows("incomplete")
    )
    converted["model"] = _env("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")

    tool_results = [
        block
        for message in converted["messages"]
        if isinstance(message["content"], list)
        for block in message["content"]
        if block.get("type") == "tool_result"
    ]
    assert tool_results, converted["messages"]
    assert tool_results[0]["tool_use_id"] == "call_err1"
    assert tool_results[0]["content"] == "City not found"
    assert tool_results[0]["is_error"] is True

    url = _env("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    response = httpx.post(
        url + "/v1/messages",
        headers=_anthropic_headers(),
        json=converted,
        timeout=TIMEOUT,
    )
    assert response.status_code == 200, response.text
    assert response.json().get("content"), response.text


def test_same_format_responses_keeps_incomplete_status() -> None:
    if not _env("OPENAI_RESPONSES_API_KEY"):
        pytest.skip("OPENAI_RESPONSES_API_KEY not set")

    converted = ConversionPipeline(
        "openai_responses", "openai_responses"
    ).convert_request(_rows("incomplete"))
    converted["model"] = _env("OPENAI_RESPONSES_MODEL", "gpt-4o-mini")

    item = next(
        item
        for item in converted["input"]
        if item["type"] == "function_call_output"
    )
    assert item == {
        "type": "function_call_output",
        "call_id": "call_err1",
        "output": "City not found",
        "status": "incomplete",
    }

    base = _env("OPENAI_RESPONSES_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    response = httpx.post(
        base + "/responses",
        headers={
            "Authorization": "Bearer " + _env("OPENAI_RESPONSES_API_KEY", ""),
            "content-type": "application/json",
        },
        json=converted,
        timeout=TIMEOUT,
    )
    assert response.status_code == 200, response.text


def test_chat_stream_preserves_tool_call_identity() -> None:
    if not _env("OPENAI_API_KEY"):
        pytest.skip("OPENAI_API_KEY not set")

    model = _env("OPENAI_MODEL", "gpt-4o-mini")
    pipeline = ConversionPipeline("messages", "chat_completions")
    body = pipeline.convert_request(_tool_request(model))
    body["model"] = model
    body["stream"] = True

    base = _env("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    chunks = _iter_sse(
        base + "/chat/completions",
        {
            "Authorization": "Bearer " + _env("OPENAI_API_KEY", ""),
            "content-type": "application/json",
        },
        body,
    )
    processor = pipeline.create_stream_processor()
    events: list[dict[str, Any]] = []
    for chunk in chunks:
        events.extend(processor.process_chunk(chunk))
    events.extend(processor.process_chunk(None))

    _assert_single_paris_call(events)


def test_anthropic_stream_preserves_tool_call_identity() -> None:
    if not _env("ANTHROPIC_API_KEY"):
        pytest.skip("ANTHROPIC_API_KEY not set")

    model = _env("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
    pipeline = ConversionPipeline("messages", "messages")
    body = pipeline.convert_request(_tool_request(model))
    body["model"] = model
    body["stream"] = True

    base = _env("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    chunks = _iter_sse(base + "/v1/messages", _anthropic_headers(), body)
    processor = pipeline.create_stream_processor()
    events: list[dict[str, Any]] = []
    for chunk in chunks:
        events.extend(processor.process_chunk(chunk))
    events.extend(processor.process_chunk(None))

    _assert_single_paris_call(events)
