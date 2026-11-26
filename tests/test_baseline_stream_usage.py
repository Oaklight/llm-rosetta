"""The gateway's same-format usage record matches the cross-format one (#1757).

``BaselineStreamProcessor`` is the record of a stream that is not converted.  It
reads the raw chunk, whatever its dialect: Messages keeps the input half in
``message_start`` (``message.usage``) and the output half, cumulative, in
``message_delta``; Responses keeps its usage under ``response``; Interactions
under ``interaction`` with ``total_*`` names.  Whatever the dialect, the record
is what the converting path reports for the same upstream.
"""

from __future__ import annotations

import copy
from typing import Any

import pytest

from llm_rosetta.pipeline.chat.streaming import BaselineStreamProcessor
from test_usage_live_streams import _CAPTURES, _chunks, _run


def _record(chunks: list[dict[str, Any]]) -> dict[str, int]:
    processor = BaselineStreamProcessor()
    for chunk in chunks:
        processor.process_chunk(copy.deepcopy(chunk))
    usage = processor.get_accumulated_usage()
    assert usage is not None
    return usage


def _triple(usage: dict[str, Any]) -> tuple[Any, Any, Any]:
    return (
        usage.get("prompt_tokens"),
        usage.get("completion_tokens"),
        usage.get("total_tokens"),
    )


#: A Messages stream the way the API documents it: the input half in
#: ``message_start``, the cumulative output count alone in ``message_delta``.
_MESSAGES_STREAM = [
    {
        "type": "message_start",
        "message": {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": "m",
            "content": [],
            "usage": {"input_tokens": 12, "output_tokens": 1},
        },
    },
    {
        "type": "content_block_start",
        "index": 0,
        "content_block": {"type": "text", "text": ""},
    },
    {
        "type": "content_block_delta",
        "index": 0,
        "delta": {"type": "text_delta", "text": "hi"},
    },
    {"type": "content_block_stop", "index": 0},
    {
        "type": "message_delta",
        "delta": {"stop_reason": "end_turn"},
        "usage": {"output_tokens": 30},
    },
    {"type": "message_stop"},
]


def test_a_messages_stream_records_the_input_half_of_message_start():
    assert _triple(_record(_MESSAGES_STREAM)) == (12, 30, 42)


def test_the_messages_record_equals_what_a_conversion_reports():
    acc, _events, _out = _run("messages", "chat_completions", _MESSAGES_STREAM)

    assert acc is not None
    assert _triple(_record(_MESSAGES_STREAM)) == _triple(acc) == (12, 30, 42)


@pytest.mark.parametrize("name", sorted(_CAPTURES))
def test_a_same_format_record_is_the_total_the_upstream_reported(name):
    _upstream, wrapped, prompt, completion, total = _CAPTURES[name]

    assert _triple(_record(_chunks(name, wrapped))) == (prompt, completion, total)


def test_a_responses_stream_records_the_usage_of_response_completed():
    chunks = [
        {"type": "response.created", "response": {"id": "r", "usage": None}},
        {
            "type": "response.completed",
            "response": {
                "id": "r",
                "usage": {"input_tokens": 5, "output_tokens": 7, "total_tokens": 12},
            },
        },
    ]

    assert _triple(_record(chunks)) == (5, 7, 12)


def test_a_later_snapshot_replaces_a_field_and_keeps_the_others():
    chunks = [
        {"usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12}},
        {"usage": {"completion_tokens": 9}},
    ]

    assert _triple(_record(chunks)) == (10, 9, 19)
