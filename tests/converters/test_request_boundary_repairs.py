"""The request-level repairs that run once, at the converter boundary.

A repair in one converter is not a repair: the fifth one forgets it.  These
are the two that used to be called by four of the five chat converters —
stripping a ``tool_choice`` left without tools (Codex context compaction) and
assigning the batch ids that group parallel tool results.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm_rosetta.auto_detect import get_converter_for_provider

CHAT_FORMATS = (
    "chat_completions",
    "messages",
    "open_responses",
    "google_generate",
    "google_interactions",
)


def _request_without_tools() -> dict[str, Any]:
    return {
        "model": "m",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        "tool_choice": {"mode": "auto"},
    }


def _parallel_call_request() -> dict[str, Any]:
    return {
        "model": "m",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "go"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "a",
                        "tool_name": "f",
                        "tool_input": {},
                    },
                    {
                        "type": "tool_call",
                        "tool_call_id": "b",
                        "tool_name": "g",
                        "tool_input": {},
                    },
                ],
            },
            {
                "role": "tool",
                "content": [
                    {"type": "tool_result", "tool_call_id": "a", "result": "1"}
                ],
            },
            {
                "role": "tool",
                "content": [
                    {"type": "tool_result", "tool_call_id": "b", "result": "2"}
                ],
            },
        ],
    }


@pytest.mark.parametrize("fmt", CHAT_FORMATS)
def test_a_tool_choice_without_tools_never_reaches_the_upstream(fmt):
    ir = _request_without_tools()

    body, warnings = get_converter_for_provider(fmt).request_to_provider(ir)

    assert "tool_choice" not in json.dumps(body)
    assert sum("tool_choice" in str(warning) for warning in warnings) == 1


@pytest.mark.parametrize("fmt", CHAT_FORMATS)
def test_parallel_results_get_their_batch_ids(fmt):
    ir = _parallel_call_request()

    get_converter_for_provider(fmt).request_to_provider(ir)

    tool_messages = [
        message for message in ir["messages"] if message.get("role") == "tool"
    ]
    batch_ids = {message.get("batch_id") for message in tool_messages}
    assert len(batch_ids) == 1
    assert None not in batch_ids
