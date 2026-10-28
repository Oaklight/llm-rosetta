"""One rule for a turn that finished normally while owing the client a result.

Anthropic and the Interactions API report a tool call in their own stop reason;
Responses and Google generateContent do not, so the IR decides from the parts.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from llm_rosetta.converters.base import StreamContext
from llm_rosetta.converters.llm.chat_completions import ChatCompletionsConverter
from llm_rosetta.converters.llm.google_generate import GoogleGenerateConverter
from llm_rosetta.converters.llm.google_interactions import GoogleInteractionsConverter
from llm_rosetta.converters.llm.open_responses import OpenResponsesConverter
from llm_rosetta.ir.primitives.finish_reason import (
    client_tool_calls_in,
    reason_for_completed_turn,
)

CLIENT_CALL = {
    "type": "tool_call",
    "tool_call_id": "c1",
    "tool_name": "f",
    "tool_input": {},
}
SERVER_CALL = {**CLIENT_CALL, "tool_type": "intrinsic"}


@pytest.mark.parametrize(
    ("reason", "parts", "expected"),
    [
        ("stop", [CLIENT_CALL], "tool_calls"),
        ("stop", [SERVER_CALL], "stop"),
        ("stop", [{"type": "text", "text": "hi"}], "stop"),
        ("length", [CLIENT_CALL], "length"),
        ("content_filter", [CLIENT_CALL], "content_filter"),
        ("error", [CLIENT_CALL], "error"),
    ],
)
def test_the_rule_only_touches_a_normal_stop(reason, parts, expected):
    assert reason_for_completed_turn(reason, client_tool_calls_in(parts)) == expected


def _google_response(*parts: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": list(parts)},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": 2,
            "totalTokenCount": 3,
        },
    }


def _google_response_with_reason(reason: str) -> dict[str, Any]:
    response = _google_response({"functionCall": {"name": "f", "args": {}}})
    response["candidates"][0]["finishReason"] = reason
    return response


def test_a_google_function_call_is_a_tool_call_turn():
    ir = GoogleGenerateConverter().response_from_provider(
        cast(Any, _google_response({"functionCall": {"name": "f", "args": {}}}))
    )

    assert ir["choices"][0]["finish_reason"]["reason"] == "tool_calls"
    chat = ChatCompletionsConverter().response_to_provider(ir)
    assert chat["choices"][0]["finish_reason"] == "tool_calls"


def test_a_google_text_turn_still_finishes_normally():
    ir = GoogleGenerateConverter().response_from_provider(
        cast(Any, _google_response({"text": "done"}))
    )

    assert ir["choices"][0]["finish_reason"]["reason"] == "stop"


def test_a_responses_function_call_is_a_tool_call_turn():
    response = {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "m",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "status": "completed",
                "call_id": "c1",
                "name": "f",
                "arguments": "{}",
            }
        ],
    }

    ir = OpenResponsesConverter().response_from_provider(cast(Any, response))

    assert ir["choices"][0]["finish_reason"]["reason"] == "tool_calls"


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        ("MAX_TOKENS", "length"),
        ("SAFETY", "content_filter"),
        ("MALFORMED_FUNCTION_CALL", "error"),
        ("LANGUAGE", "error"),
    ],
)
def test_a_google_call_does_not_overwrite_a_non_stop_reason(finish_reason, expected):
    """A malformed call is still an error: the client must not be asked to run it."""
    ir = GoogleGenerateConverter().response_from_provider(
        cast(Any, _google_response_with_reason(finish_reason))
    )

    assert ir["choices"][0]["finish_reason"]["reason"] == expected


def _run_google_stream(*chunks: dict[str, Any]) -> list[Any]:
    converter = GoogleGenerateConverter()
    context = StreamContext()
    events: list[Any] = []
    for chunk in chunks:
        events.extend(converter.stream_response_from_provider(chunk, context))
    return events


def test_a_google_stream_finish_carries_the_same_reason():
    """Google finishes in a chunk with no content, after the call's own chunk."""
    events = _run_google_stream(
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "checking"}]},
                    "index": 0,
                }
            ]
        },
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"functionCall": {"name": "f", "args": {}}}],
                    },
                    "index": 0,
                }
            ]
        },
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": []},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ]
        },
    )

    finishes = [event for event in events if event["type"] == "finish"]
    assert finishes and finishes[-1]["finish_reason"]["reason"] == "tool_calls"


def test_a_google_stream_of_text_only_still_finishes_normally():
    events = _run_google_stream(
        {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "hi"}]}, "index": 0}
            ]
        },
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": []},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ]
        },
    )

    finishes = [event for event in events if event["type"] == "finish"]
    assert finishes and finishes[-1]["finish_reason"]["reason"] == "stop"


def test_an_interactions_function_call_is_a_tool_call_turn():
    """The API normally says ``requires_action``; when it says ``completed``
    the steps still make it a tool-call turn."""
    response = {
        "id": "int_1",
        "object": "interaction",
        "created": "2026-10-26T00:00:00Z",
        "model": "m",
        "status": "completed",
        "steps": [
            {"type": "function_call", "id": "c1", "name": "f", "arguments": {}},
        ],
    }

    ir = GoogleInteractionsConverter().response_from_provider(cast(Any, response))

    assert ir["choices"][0]["finish_reason"]["reason"] == "tool_calls"


def test_an_interactions_requires_action_is_unchanged():
    response = {
        "id": "int_2",
        "object": "interaction",
        "created": "2026-10-26T00:00:00Z",
        "model": "m",
        "status": "requires_action",
        "steps": [
            {"type": "function_call", "id": "c1", "name": "f", "arguments": {}},
        ],
    }

    ir = GoogleInteractionsConverter().response_from_provider(cast(Any, response))

    assert ir["choices"][0]["finish_reason"]["reason"] == "tool_calls"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # A hosted call the provider ran itself completes the turn; the same
        # exchange without request context falls back to the name table.
        ("googleSearch", "stop"),
        ("f", "tool_calls"),
    ],
    ids=["hosted-call", "client-call"],
)
def test_a_google_exchange_finishes_the_same_way_in_json_and_stream(name, expected):
    """The same logical exchange, read from either Google dialect.

    generateContent answers in one JSON body and as a chunk stream; the two
    readers must not disagree about a turn, which is the defect family here
    (#1227).  No request leg is recorded, so both take the same name-table
    fallback.
    """
    ir = GoogleGenerateConverter().response_from_provider(
        cast(Any, _google_response({"functionCall": {"name": name, "args": {}}}))
    )
    assert ir["choices"][0]["finish_reason"]["reason"] == expected

    events = _run_google_stream(
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"functionCall": {"name": name, "args": {}}}],
                    },
                    "index": 0,
                }
            ]
        },
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": []},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ]
        },
    )

    finishes = [event for event in events if event["type"] == "finish"]
    assert finishes and finishes[-1]["finish_reason"]["reason"] == expected
