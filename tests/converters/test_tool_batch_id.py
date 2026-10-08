"""Tests for tool result batch tracking (issue #837).

Covers:
- assign_tool_batch_ids utility
- merge_tool_messages utility
- End-to-end cross-format grouping (Chat → Anthropic, Chat → Google Generate)
- Backward compatibility (IR without batch_id)
"""

from __future__ import annotations

import pytest

from llm_rosetta.converters.base.tools.batch import (
    assign_tool_batch_ids,
    merge_tool_messages,
)
from llm_rosetta.pipeline import ConversionPipeline


# ── Helpers ────────────────────────────────────────────────────


def _assistant_with_tools(*call_ids: str) -> dict:
    return {
        "role": "assistant",
        "content": [
            {
                "type": "tool_call",
                "tool_call_id": cid,
                "tool_name": "fn",
                "tool_input": {},
            }
            for cid in call_ids
        ],
    }


def _tool_result(call_id: str, result: str = "ok", **extra) -> dict:
    msg: dict = {
        "role": "tool",
        "content": [{"type": "tool_result", "tool_call_id": call_id, "result": result}],
    }
    msg.update(extra)
    return msg


def _user(text: str = "hello") -> dict:
    return {"role": "user", "content": [{"type": "text", "text": text}]}


def _tool_msg_count(msgs: list) -> int:
    return sum(1 for m in msgs if isinstance(m, dict) and m.get("role") == "tool")


def _tool_parts_per_msg(msgs: list) -> list[int]:
    return [
        len(m["content"])
        for m in msgs
        if isinstance(m, dict) and m.get("role") == "tool"
    ]


# ── assign_tool_batch_ids ──────────────────────────────────────


class TestAssignToolBatchIds:
    def test_single_turn(self):
        msgs = [
            _assistant_with_tools("c1", "c2"),
            _tool_result("c1"),
            _tool_result("c2"),
        ]
        assign_tool_batch_ids(msgs)
        assert msgs[1]["batch_id"] == "0"
        assert msgs[2]["batch_id"] == "0"

    def test_two_turns(self):
        msgs = [
            _assistant_with_tools("c1"),
            _tool_result("c1"),
            _assistant_with_tools("c2"),
            _tool_result("c2"),
        ]
        assign_tool_batch_ids(msgs)
        assert msgs[1]["batch_id"] == "0"
        assert msgs[3]["batch_id"] == "1"

    def test_assistant_without_tools_resets(self):
        msgs = [
            _assistant_with_tools("c1"),
            _tool_result("c1"),
            {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
            _tool_result("orphan"),
        ]
        assign_tool_batch_ids(msgs)
        assert msgs[1]["batch_id"] == "0"
        assert "batch_id" not in msgs[3]

    def test_non_tool_message_resets(self):
        msgs = [
            _assistant_with_tools("c1", "c2"),
            _tool_result("c1"),
            _user("interruption"),
            _tool_result("c2"),
        ]
        assign_tool_batch_ids(msgs)
        assert msgs[1]["batch_id"] == "0"
        assert "batch_id" not in msgs[3]

    def test_preserves_existing_batch_id(self):
        msgs = [
            _assistant_with_tools("c1"),
            _tool_result("c1", batch_id="custom"),
        ]
        assign_tool_batch_ids(msgs)
        assert msgs[1]["batch_id"] == "custom"

    def test_passthrough_items_transparent(self):
        msgs = [
            _assistant_with_tools("c1", "c2"),
            _tool_result("c1"),
            {"type": "passthrough", "data": "x"},
            _tool_result("c2"),
        ]
        assign_tool_batch_ids(msgs)
        assert msgs[1]["batch_id"] == "0"
        assert msgs[3]["batch_id"] == "0"

    def test_no_assistant_no_batch(self):
        msgs = [_tool_result("c1"), _tool_result("c2")]
        assign_tool_batch_ids(msgs)
        assert "batch_id" not in msgs[0]
        assert "batch_id" not in msgs[1]


# ── merge_tool_messages ────────────────────────────────────────


class TestMergeToolMessages:
    def test_merge_same_batch(self):
        msgs = [
            _assistant_with_tools("c1", "c2"),
            _tool_result("c1", batch_id="0"),
            _tool_result("c2", batch_id="0"),
        ]
        merged = merge_tool_messages(msgs)
        assert _tool_msg_count(merged) == 1
        assert _tool_parts_per_msg(merged) == [2]

    def test_split_different_batches(self):
        msgs = [
            _tool_result("c1", batch_id="0"),
            _tool_result("c2", batch_id="1"),
        ]
        merged = merge_tool_messages(msgs)
        assert _tool_msg_count(merged) == 2

    def test_adjacency_fallback_no_batch_id(self):
        msgs = [
            _tool_result("c1"),
            _tool_result("c2"),
        ]
        merged = merge_tool_messages(msgs)
        assert _tool_msg_count(merged) == 1

    def test_mixed_batch_and_no_batch(self):
        msgs = [
            _tool_result("c1", batch_id="X"),
            _tool_result("c2"),
        ]
        merged = merge_tool_messages(msgs)
        assert _tool_msg_count(merged) == 2

    def test_non_tool_breaks_merge(self):
        msgs = [
            _tool_result("c1", batch_id="0"),
            _user("mid"),
            _tool_result("c2", batch_id="0"),
        ]
        merged = merge_tool_messages(msgs)
        assert _tool_msg_count(merged) == 2

    def test_full_pipeline_assign_then_merge(self):
        msgs = [
            _assistant_with_tools("c1", "c2"),
            _tool_result("c1"),
            _tool_result("c2"),
            _assistant_with_tools("c3"),
            _tool_result("c3"),
        ]
        assign_tool_batch_ids(msgs)
        merged = merge_tool_messages(msgs)
        assert _tool_msg_count(merged) == 2
        assert _tool_parts_per_msg(merged) == [2, 1]

    def test_empty_input(self):
        assert merge_tool_messages([]) == []

    def test_large_batch(self):
        msgs = [
            _assistant_with_tools(*[f"c{i}" for i in range(10)]),
            *[_tool_result(f"c{i}", batch_id="0") for i in range(10)],
        ]
        merged = merge_tool_messages(msgs)
        assert _tool_msg_count(merged) == 1
        assert _tool_parts_per_msg(merged) == [10]


# ── End-to-end cross-format ────────────────────────────────────


class TestCrossFormatGrouping:
    def _chat_request_with_parallel_tools(self, n_calls: int = 2) -> dict:
        calls = [
            {
                "id": f"call_{i}",
                "type": "function",
                "function": {"name": "read", "arguments": "{}"},
            }
            for i in range(n_calls)
        ]
        return {
            "model": "test-model",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "Read files."},
                {"role": "assistant", "content": None, "tool_calls": calls},
                *[
                    {"role": "tool", "tool_call_id": c["id"], "content": f"result_{i}"}
                    for i, c in enumerate(calls)
                ],
            ],
        }

    def test_chat_to_anthropic_groups_tool_results(self):
        body = self._chat_request_with_parallel_tools(3)
        result = ConversionPipeline("openai_chat", "anthropic").convert_request(body)
        messages = result["messages"]
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant", "user"]
        tool_blocks = [
            b for b in messages[-1]["content"] if b.get("type") == "tool_result"
        ]
        assert len(tool_blocks) == 3

    def test_chat_to_google_generate_groups_tool_results(self):
        body = self._chat_request_with_parallel_tools(2)
        result = ConversionPipeline("openai_chat", "google_generate").convert_request(
            body
        )
        contents = result["contents"]
        fn_response_contents = [
            c
            for c in contents
            if any("functionResponse" in p for p in c.get("parts", []))
        ]
        assert len(fn_response_contents) == 1
        fr_parts = [
            p for p in fn_response_contents[0]["parts"] if "functionResponse" in p
        ]
        assert len(fr_parts) == 2

    def test_chat_to_anthropic_two_turns_stay_separate(self):
        body = {
            "model": "test-model",
            "max_tokens": 64,
            "messages": [
                {"role": "user", "content": "Do two things."},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_a",
                            "type": "function",
                            "function": {"name": "fn", "arguments": "{}"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_a", "content": "result_a"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_b",
                            "type": "function",
                            "function": {"name": "fn", "arguments": "{}"},
                        },
                    ],
                },
                {"role": "tool", "tool_call_id": "call_b", "content": "result_b"},
            ],
        }
        result = ConversionPipeline("openai_chat", "anthropic").convert_request(body)
        messages = result["messages"]
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant", "user", "assistant", "user"]
        tool_user_msgs = [
            m
            for m in messages
            if m["role"] == "user"
            and any(b.get("type") == "tool_result" for b in m["content"])
        ]
        assert len(tool_user_msgs) == 2
        assert len(tool_user_msgs[0]["content"]) == 1
        assert len(tool_user_msgs[1]["content"]) == 1

    def test_backward_compat_ir_without_batch_id(self):
        """IR messages without batch_id should still merge by adjacency."""
        from llm_rosetta.converters.anthropic import AnthropicConverter

        converter = AnthropicConverter()
        ir_messages = [
            _user("hi"),
            _assistant_with_tools("c1", "c2"),
            _tool_result("c1"),
            _tool_result("c2"),
        ]
        messages, warnings = converter.message_ops.ir_messages_to_p(ir_messages)
        tool_user_msgs = [
            m
            for m in messages
            if m.get("role") == "user"
            and any(b.get("type") == "tool_result" for b in m.get("content", []))
        ]
        assert len(tool_user_msgs) == 1
        assert len(tool_user_msgs[0]["content"]) == 2

    @pytest.mark.parametrize("n_calls", [1, 2, 3, 5])
    def test_chat_to_anthropic_various_counts(self, n_calls):
        body = self._chat_request_with_parallel_tools(n_calls)
        result = ConversionPipeline("openai_chat", "anthropic").convert_request(body)
        messages = result["messages"]
        roles = [m["role"] for m in messages]
        assert roles == ["user", "assistant", "user"]
        tool_blocks = [
            b for b in messages[-1]["content"] if b.get("type") == "tool_result"
        ]
        assert len(tool_blocks) == n_calls
