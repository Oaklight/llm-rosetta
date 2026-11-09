"""An assistant turn's part order is declared, and its loss is reported (#1491).

Chat Completions has no carrier for a part's position: ``reasoning_content``,
``content`` and ``tool_calls`` are independent message fields.  The reader
builds the canonical order ``[reasoning, text, call]``; the writer reproduces
that shape and reports a loss whenever the IR input deviates from it, or when
several text or reasoning parts must be merged into one field.

The shapes below are the eleven of Appendix A in #1491.
"""

from typing import Any, cast

import pytest

from llm_rosetta.conversion_warnings import DOWNGRADED, DROPPED, ConversionWarning
from llm_rosetta.converters.llm.chat_completions.content_ops import (
    ChatCompletionsContentOps,
)
from llm_rosetta.converters.llm.chat_completions.message_ops import (
    ChatCompletionsMessageOps,
)
from llm_rosetta.converters.llm.chat_completions.tool_ops import ChatCompletionsToolOps
from llm_rosetta.types.ir import Message


def _text(text: str) -> dict[str, Any]:
    return {"type": "text", "text": text}


def _reasoning(text: str) -> dict[str, Any]:
    return {"type": "reasoning", "reasoning": text}


def _call(call_id: str) -> dict[str, Any]:
    return {
        "type": "tool_call",
        "tool_call_id": call_id,
        "tool_name": "search",
        "tool_input": {},
    }


@pytest.fixture()
def message_ops() -> ChatCompletionsMessageOps:
    return ChatCompletionsMessageOps(
        ChatCompletionsContentOps(), ChatCompletionsToolOps()
    )


def _write(message_ops, parts: list[Any]) -> tuple[dict[str, Any], list[Any]]:
    """Write one assistant turn and return (the chat message, warnings)."""
    messages = cast(list[Message], [{"role": "assistant", "content": parts}])
    written, warnings = message_ops.ir_messages_to_p(messages)
    return written[0], warnings


def _of_subject(warnings: list[Any], subject: str) -> list[ConversionWarning]:
    return [
        warning
        for warning in warnings
        if isinstance(warning, ConversionWarning) and warning.subject == subject
    ]


#: (parts, the first deviation the writer must name, or None) — Appendix A.
#: Six shapes deviate from ``[reasoning, text, call]``; five keep it.
_SHAPES = [
    pytest.param([_text("A"), _call("c1")], None, id="text-call"),
    pytest.param(
        [_reasoning("R"), _text("A"), _call("c1")], None, id="reasoning-text-call"
    ),
    pytest.param([_reasoning("R"), _call("c1")], None, id="reasoning-call"),
    pytest.param([_call("c1"), _call("c2")], None, id="call-call"),
    pytest.param([_reasoning("R"), _text("A")], None, id="reasoning-text"),
    pytest.param(
        [_call("c1"), _text("A")], "a text after a tool call", id="text-after-call"
    ),
    pytest.param(
        [_text("A"), _reasoning("R"), _call("c1")],
        "reasoning after text",
        id="reasoning-after-text",
    ),
    pytest.param(
        [_call("c1"), _reasoning("R")],
        "reasoning after a tool call",
        id="reasoning-after-call",
    ),
    pytest.param(
        [_text("A"), _call("c1"), _text("B")],
        "a text after a tool call",
        id="text-call-text",
    ),
    pytest.param([_text("A"), _text("B")], "a second text part", id="two-text-parts"),
    pytest.param(
        [_reasoning("R1"), _reasoning("R2"), _text("A")],
        "a second reasoning part",
        id="two-reasoning-parts",
    ),
]


@pytest.mark.parametrize(("parts", "first_deviation"), _SHAPES)
def test_the_part_order_loss_is_reported(message_ops, parts, first_deviation):
    """Every deviation from [reasoning, text, call] says so; the rest stay quiet."""
    _written, warnings = _write(message_ops, parts)

    order_warnings = _of_subject(warnings, "part_order")

    assert len(order_warnings) == (0 if first_deviation is None else 1)
    if first_deviation is not None:
        assert order_warnings[0].code == DOWNGRADED
        assert "cannot be expressed in Chat Completions" in order_warnings[0]
        assert "[reasoning, text, call]" in order_warnings[0]
        assert f"(first: {first_deviation})" in order_warnings[0]


#: (parts, the indices of calls that were not written, expected deviation).
_DROPPED_CALL_CASES = [
    pytest.param(
        [_text("A"), _call("d1"), _text("B")],
        {1},
        "a second text part",
        id="text-dropped-text",
    ),
    pytest.param(
        [_text("A"), _call("d1"), _reasoning("R")],
        {1},
        "reasoning after text",
        id="text-dropped-reasoning",
    ),
    pytest.param([_call("d1"), _text("A")], {0}, None, id="dropped-text"),
    pytest.param(
        [_call("c1"), _call("d2"), _text("A")],
        {1},
        "a text after a tool call",
        id="call-dropped-text",
    ),
]


@pytest.mark.parametrize(("parts", "dropped", "expected"), _DROPPED_CALL_CASES)
def test_a_dropped_call_neither_closes_nor_moves_the_text(parts, dropped, expected):
    """A call that was not written must not look like a deviation.

    Defensive: this dialect writes every intrinsic call as a plain function
    call, so the writer never passes a non-empty set today.  These cases pin
    the pure scan's exclusion for a dialect that would really drop one.
    """
    assert (
        ChatCompletionsMessageOps._first_part_order_deviation(parts, dropped)
        == expected
    )


def test_the_ordinary_replay_shape_is_silent(message_ops):
    """``[text…, call…]`` is every historical turn and must warn about nothing."""
    _written, warnings = _write(message_ops, [_text("A"), _call("c1"), _call("c2")])

    assert warnings == []


def test_text_after_a_call_is_written_before_it(message_ops):
    """The reported loss matches the write: the text lands in ``content``."""
    written, _warnings = _write(message_ops, [_call("c1"), _text("A")])

    assert written["content"] == "A"
    assert [call["id"] for call in written["tool_calls"]] == ["c1"]


def test_two_reasoning_parts_lose_the_earlier_content(message_ops):
    """Content loss is its own report: one field, the last reasoning wins."""
    written, warnings = _write(
        message_ops, [_reasoning("R1"), _reasoning("R2"), _text("A")]
    )

    assert written["reasoning_content"] == "R2"
    content_warnings = _of_subject(warnings, "reasoning")
    assert len(content_warnings) == 1
    assert content_warnings[0].code == DROPPED
    assert "only the last one is kept" in content_warnings[0]
    assert "the earlier reasoning content is lost" in content_warnings[0]


def test_one_report_per_subject_per_request(message_ops):
    """A history that deviates on every turn reports the subject once.

    The one report describes the first turn's deviation, because that is the
    turn the deduplicated subject was raised for.
    """
    messages = cast(
        list[Message],
        [
            {"role": "assistant", "content": [_call("c1"), _text("A")]},
            {"role": "assistant", "content": [_reasoning("R1"), _reasoning("R2")]},
        ],
    )

    written, warnings = message_ops.ir_messages_to_p(messages)

    order_warnings = _of_subject(warnings, "part_order")
    assert len(order_warnings) == 1
    assert "a text after a tool call" in order_warnings[0]
    assert "a second reasoning part" not in order_warnings[0]
    assert [message["content"] for message in written] == ["A", ""]
    assert written[1]["reasoning_content"] == "R2"


def test_the_reader_builds_the_canonical_order(message_ops):
    """Provider → IR: the three fields come out as [reasoning, text, call]."""
    provider_messages = [
        {
            "role": "assistant",
            "reasoning_content": "T",
            "content": "A",
            "tool_calls": [
                {
                    "id": "c1",
                    "type": "function",
                    "function": {"name": "search", "arguments": "{}"},
                }
            ],
        }
    ]

    ir_messages = message_ops.p_messages_to_ir(provider_messages)

    assert [part["type"] for part in ir_messages[0]["content"]] == [
        "reasoning",
        "text",
        "tool_call",
    ]
