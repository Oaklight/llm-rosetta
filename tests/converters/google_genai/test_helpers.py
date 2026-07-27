"""Unit tests for extracted helpers in the Google GenAI converter/message ops."""

from typing import Any, cast

from llm_rosetta.converters.google_genai.message_ops import GoogleGenAIMessageOps
from llm_rosetta.types.ir import Message, is_tool_result_part


def _msg(role: str, parts: list[dict[str, Any]]) -> Message:
    return cast(Message, {"role": role, "content": parts})


def test_build_tool_call_queue_orders_by_appearance() -> None:
    messages = [
        _msg(
            "assistant",
            [
                {"type": "tool_call", "tool_call_id": "a1", "tool_name": "search"},
                {"type": "tool_call", "tool_call_id": "a2", "tool_name": "search"},
            ],
        ),
        _msg(
            "assistant",
            [{"type": "tool_call", "tool_call_id": "a3", "tool_name": "lookup"}],
        ),
    ]
    q = GoogleGenAIMessageOps._build_tool_call_queue(messages)
    assert q == {"search": ["a1", "a2"], "lookup": ["a3"]}


def test_resolve_tool_name_prefix_and_exact_match() -> None:
    q = {"search": ["a1"], "lookup": ["a2"]}
    assert (
        GoogleGenAIMessageOps._resolve_tool_name_for_result("search_123_0", q)
        == "search"
    )
    assert GoogleGenAIMessageOps._resolve_tool_name_for_result("lookup", q) == "lookup"
    # Unknown result_id → default guess = itself
    assert (
        GoogleGenAIMessageOps._resolve_tool_name_for_result("mystery", q) == "mystery"
    )


def test_apply_reconciliation_fifo_pairs_ids() -> None:
    messages = [
        _msg(
            "assistant",
            [
                {"type": "tool_call", "tool_call_id": "call_1", "tool_name": "search"},
                {"type": "tool_call", "tool_call_id": "call_2", "tool_name": "search"},
            ],
        ),
        _msg(
            "tool",
            [
                {
                    "type": "tool_result",
                    "tool_call_id": "search_A",
                    "result": "r1",
                },
                {
                    "type": "tool_result",
                    "tool_call_id": "search_B",
                    "result": "r2",
                },
            ],
        ),
    ]
    GoogleGenAIMessageOps._reconcile_tool_call_ids(messages)
    result_parts = messages[1]["content"]
    assert all(is_tool_result_part(part) for part in result_parts)
    result_ids = [
        part["tool_call_id"] for part in result_parts if is_tool_result_part(part)
    ]
    assert result_ids == ["call_1", "call_2"]


def test_reconcile_noop_when_no_tool_calls() -> None:
    messages = [
        _msg(
            "tool",
            [{"type": "tool_result", "tool_call_id": "x", "result": "r"}],
        ),
    ]
    # Should not raise / rewrite anything
    GoogleGenAIMessageOps._reconcile_tool_call_ids(messages)
    part = messages[0]["content"][0]
    assert is_tool_result_part(part)
    assert part["tool_call_id"] == "x"
