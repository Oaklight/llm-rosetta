"""An orphan tool result is removed wherever it rides.

Some dialects carry a result in a ``role: "tool"`` message, others in a user
turn.  Providers reject a result with no matching declaration either way — the
Messages API answers ``400 unexpected `tool_use_id` found in `tool_result`
blocks`` — so the IR repair has to look at the parts, not the role.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from llm_rosetta.converters.llm.chat_completions import ChatCompletionsConverter
from llm_rosetta.converters.llm.messages import MessagesConverter
from llm_rosetta.ir.tools.orphan_fix import (
    UnrepairableToolHistory,
    fix_orphaned_tool_calls_ir,
)

ORPHAN_TURN = {
    "role": "user",
    "content": [{"type": "tool_result", "tool_call_id": "ghost", "result": "r"}],
}


@pytest.mark.parametrize(
    ("converter", "drop_key"),
    [(MessagesConverter(), "tool_result"), (ChatCompletionsConverter(), "tool")],
)
def test_a_result_in_a_user_turn_reaches_no_wire(converter, drop_key):
    ir = {
        "model": "m",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            ORPHAN_TURN,
        ],
    }

    body, _ = converter.request_to_provider(cast(Any, ir))

    assert "ghost" not in json.dumps(body)
    assert drop_key not in json.dumps(body)


def test_the_rest_of_the_user_turn_survives():
    ir = {
        "model": "m",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "here it is"},
                    {"type": "tool_result", "tool_call_id": "ghost", "result": "r"},
                ],
            },
        ],
    }

    body, _ = MessagesConverter().request_to_provider(cast(Any, ir))

    assert body["messages"][1]["content"] == [{"type": "text", "text": "here it is"}]


def test_a_result_that_answers_a_declaration_is_kept():
    ir = {
        "model": "m",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "c1",
                        "tool_name": "f",
                        "tool_input": {},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_call_id": "c1", "result": "ok"},
                ],
            },
        ],
    }

    body, _ = MessagesConverter().request_to_provider(cast(Any, ir))

    assert body["messages"][2]["content"][0]["type"] == "tool_result"
    assert body["messages"][2]["content"][0]["tool_use_id"] == "c1"


def test_an_orphan_only_history_fails_instead_of_sending_nothing():
    """Dropping the lone result would leave an empty ``messages`` array."""
    with pytest.raises(UnrepairableToolHistory) as raised:
        fix_orphaned_tool_calls_ir(cast(Any, [ORPHAN_TURN]))

    assert "ghost" in str(raised.value)


def test_a_real_answer_beside_an_orphan_keeps_only_the_answer():
    """The shape a client history actually has: one answer, one orphan, same turn."""
    ir = {
        "model": "m",
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "hi"}]},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_call",
                        "tool_call_id": "c1",
                        "tool_name": "f",
                        "tool_input": {},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_call_id": "g1", "result": "BAD1"},
                    {"type": "tool_result", "tool_call_id": "c1", "result": "GOOD"},
                    {"type": "tool_result", "tool_call_id": "g2", "result": "BAD2"},
                ],
            },
        ],
    }

    body, _ = MessagesConverter().request_to_provider(cast(Any, ir))

    results = [
        part
        for message in body["messages"]
        for part in message.get("content", [])
        if isinstance(part, dict) and part.get("type") == "tool_result"
    ]
    assert [part["tool_use_id"] for part in results] == ["c1"]
    assert results[0]["content"] == "GOOD"


def test_a_message_with_no_content_does_not_raise():
    patched = fix_orphaned_tool_calls_ir(
        cast(
            Any,
            [
                {"role": "assistant", "content": None},
                {"role": "user", "content": None},
            ],
        )
    )

    assert patched == [
        {"role": "assistant", "content": None},
        {"role": "user", "content": None},
    ]
