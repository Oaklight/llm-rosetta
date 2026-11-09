"""An assistant turn's parts keep the order they came in (fork ``12129169ad``).

A multi-step turn interleaves text, tool calls and reasoning, and the position
of a reasoning item is what ties it to the call it explains.  The writer used
to bucket the parts by kind and emit reasoning, then the message, then the
calls, which rewrote the history the client recorded.
"""

from typing import Any, cast

import pytest

from llm_rosetta.converters.llm.open_responses import OpenAIResponsesConverter
from llm_rosetta.converters.llm.open_responses.content_ops import (
    OpenAIResponsesContentOps,
)
from llm_rosetta.converters.llm.open_responses.message_ops import (
    OpenAIResponsesMessageOps,
)
from llm_rosetta.converters.llm.open_responses.tool_ops import OpenAIResponsesToolOps
from llm_rosetta.types.ir import Message

_REASONING_ID = "rs_order_1"


def _reasoning(text: str = "thinking") -> dict[str, Any]:
    """A reasoning part a Responses history can prove the origin of."""
    return {
        "type": "reasoning",
        "reasoning": text,
        "provider_metadata": {"responses_reasoning_id": _REASONING_ID},
    }


def _text(text: str = "answer") -> dict[str, Any]:
    return {"type": "text", "text": text}


def _call(call_id: str = "c1") -> dict[str, Any]:
    return {
        "type": "tool_call",
        "tool_call_id": call_id,
        "tool_name": "search",
        "tool_input": {},
    }


@pytest.fixture()
def message_ops() -> OpenAIResponsesMessageOps:
    return OpenAIResponsesMessageOps(
        OpenAIResponsesContentOps(), OpenAIResponsesToolOps()
    )


#: Each case is (IR parts, the item types the writer must emit, in order).
_CASES = [
    pytest.param(
        [_reasoning(), _text()],
        ["reasoning", "message"],
        id="reasoning-then-text",
    ),
    pytest.param(
        [_text(), _reasoning()],
        ["message", "reasoning"],
        id="text-then-reasoning",
    ),
    pytest.param(
        [_text(), _call("c1"), _reasoning(), _call("c2")],
        ["message", "function_call", "reasoning", "function_call"],
        id="text-call-reasoning-call",
    ),
    pytest.param(
        [_text(), _reasoning(), _call("c1")],
        ["message", "reasoning", "function_call"],
        id="text-reasoning-call",
    ),
    pytest.param(
        [_call("c1"), _reasoning()],
        ["function_call", "reasoning"],
        id="call-then-reasoning",
    ),
    pytest.param(
        [_reasoning(), _call("c1")],
        ["reasoning", "function_call"],
        id="reasoning-then-call",
    ),
]


@pytest.mark.parametrize(("parts", "expected"), _CASES)
def test_the_writer_keeps_the_part_order(message_ops, parts, expected):
    """Every part is emitted where it sits, not where its kind would sort."""
    messages = cast(list[Message], [{"role": "assistant", "content": parts}])

    items, _warnings = message_ops.ir_messages_to_p(messages)

    assert [item["type"] for item in items] == expected


def test_the_text_parts_stay_one_message_at_the_first_text(message_ops):
    """The one concession: several text parts become one item, at the first."""
    messages = cast(
        list[Message],
        [
            {
                "role": "assistant",
                "content": [_text("one "), _call("c1"), _text("two")],
            }
        ],
    )

    items, _warnings = message_ops.ir_messages_to_p(messages)

    assert [item["type"] for item in items] == ["message", "function_call"]
    assert [part["text"] for part in items[0]["content"]] == ["one ", "two"]


def test_a_two_round_tool_history_round_trips_in_order():
    """The shape a real client replays: two calls, each with its reasoning."""
    history = [
        {
            "type": "message",
            "role": "user",
            "content": [{"type": "input_text", "text": "go"}],
        },
        {
            "type": "reasoning",
            "id": "rs_1",
            "summary": [{"type": "summary_text", "text": "think 1"}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "calling"}],
        },
        {
            "type": "function_call",
            "id": "fc_1",
            "call_id": "c1",
            "name": "search",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "c1",
            "output": "r1",
        },
        {
            "type": "reasoning",
            "id": "rs_2",
            "summary": [{"type": "summary_text", "text": "think 2"}],
        },
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": "again"}],
        },
        {
            "type": "function_call",
            "id": "fc_2",
            "call_id": "c2",
            "name": "search",
            "arguments": "{}",
        },
        {
            "type": "function_call_output",
            "call_id": "c2",
            "output": "r2",
        },
    ]
    converter = OpenAIResponsesConverter()

    ir_request = converter.request_from_provider({"model": "m", "input": history})
    body, _warnings = converter.request_to_provider(ir_request)

    assert [item["type"] for item in body["input"]] == [
        item["type"] for item in history
    ]
