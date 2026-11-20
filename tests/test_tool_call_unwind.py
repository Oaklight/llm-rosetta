"""Tests for Argo parallel tool call unwind utility."""

from typing import Any

from llm_rosetta.conversion_warnings import DOWNGRADED, WarningSink, recording_to
from llm_rosetta.ir.primitives.call_unwind import (
    unwind_parallel_tool_calls_ir,
)


def _tc(tool_call_id: str, name: str = "get_weather", **kwargs: Any) -> dict:
    """Build an IR tool_call content part."""
    return {
        "type": "tool_call",
        "tool_call_id": tool_call_id,
        "tool_name": name,
        "tool_input": kwargs or {"city": "NYC"},
        "tool_type": "function",
    }


def _tr(tool_call_id: str, result: str = "Sunny") -> dict:
    """Build an IR tool_result content part."""
    return {
        "type": "tool_result",
        "tool_call_id": tool_call_id,
        "result": result,
    }


class TestUnwindNoOp:
    """Cases where unwind should NOT modify messages."""

    def test_no_messages(self):
        req = {"model": "test", "messages": []}
        result = unwind_parallel_tool_calls_ir(req)
        assert result is req

    def test_no_tool_calls(self):
        req = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
            ]
        }
        result = unwind_parallel_tool_calls_ir(req)
        assert result is req

    def test_single_tool_call(self):
        """Single tool call should NOT be unwound."""
        req = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
                {"role": "assistant", "content": [_tc("c1")]},
                {"role": "tool", "content": [_tr("c1")]},
            ]
        }
        result = unwind_parallel_tool_calls_ir(req)
        assert result is req


class TestUnwindParallel:
    """Cases where parallel tool calls should be unwound."""

    def test_two_parallel_calls(self):
        req = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
                {"role": "assistant", "content": [_tc("c1"), _tc("c2", city="LA")]},
                {"role": "tool", "content": [_tr("c1", "Sunny")]},
                {"role": "tool", "content": [_tr("c2", "Cloudy")]},
            ]
        }
        result = unwind_parallel_tool_calls_ir(req)
        assert result is not req
        msgs = result["messages"]
        assert len(msgs) == 5  # user + 2*(assistant + tool)
        # First pair
        assert msgs[1]["role"] == "assistant"
        tc_parts = [p for p in msgs[1]["content"] if p["type"] == "tool_call"]
        assert len(tc_parts) == 1
        assert tc_parts[0]["tool_call_id"] == "c1"
        assert msgs[2]["role"] == "tool"
        # Second pair
        assert msgs[3]["role"] == "assistant"
        tc_parts = [p for p in msgs[3]["content"] if p["type"] == "tool_call"]
        assert len(tc_parts) == 1
        assert tc_parts[0]["tool_call_id"] == "c2"
        assert msgs[4]["role"] == "tool"

    def test_three_parallel_calls(self):
        req = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
                {
                    "role": "assistant",
                    "content": [
                        _tc("c1", city="NYC"),
                        _tc("c2", city="LA"),
                        _tc("c3", city="SF"),
                    ],
                },
                {"role": "tool", "content": [_tr("c1", "Sunny")]},
                {"role": "tool", "content": [_tr("c2", "Cloudy")]},
                {"role": "tool", "content": [_tr("c3", "Rainy")]},
            ]
        }
        result = unwind_parallel_tool_calls_ir(req)
        msgs = result["messages"]
        assert len(msgs) == 7  # user + 3*(assistant + tool)
        for i in range(3):
            assert msgs[1 + i * 2]["role"] == "assistant"
            assert msgs[2 + i * 2]["role"] == "tool"

    def test_preserves_non_tool_call_parts_on_first_pair(self):
        """Text/reasoning parts go on the first unwound assistant message."""
        req = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me check both cities"},
                        _tc("c1"),
                        _tc("c2", city="LA"),
                    ],
                },
                {"role": "tool", "content": [_tr("c1")]},
                {"role": "tool", "content": [_tr("c2")]},
            ]
        }
        result = unwind_parallel_tool_calls_ir(req)
        msgs = result["messages"]
        # First assistant has text + tool_call
        first_assistant = msgs[1]
        assert any(p.get("type") == "text" for p in first_assistant["content"])
        assert any(p.get("type") == "tool_call" for p in first_assistant["content"])
        # Second assistant has only tool_call
        second_assistant = msgs[3]
        assert all(p.get("type") == "tool_call" for p in second_assistant["content"])


class TestUnwindMatching:
    """Test ID-based vs positional matching."""

    def test_id_based_matching(self):
        """Results matched by tool_call_id regardless of order."""
        req = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
                {"role": "assistant", "content": [_tc("c1"), _tc("c2", city="LA")]},
                # Results in reverse order
                {"role": "tool", "content": [_tr("c2", "Cloudy")]},
                {"role": "tool", "content": [_tr("c1", "Sunny")]},
            ]
        }
        result = unwind_parallel_tool_calls_ir(req)
        msgs = result["messages"]
        # c1 should be matched with its result regardless of order
        tool_1 = msgs[2]
        assert tool_1["content"][0]["tool_call_id"] == "c1"
        assert tool_1["content"][0]["result"] == "Sunny"

    def test_positional_fallback(self):
        """When IDs don't match, fall back to position."""
        req = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
                {"role": "assistant", "content": [_tc("c1"), _tc("c2", city="LA")]},
                # Results with different IDs
                {"role": "tool", "content": [_tr("x1", "Sunny")]},
                {"role": "tool", "content": [_tr("x2", "Cloudy")]},
            ]
        }
        result = unwind_parallel_tool_calls_ir(req)
        msgs = result["messages"]
        assert len(msgs) == 5


class TestUnwindSafety:
    """Safety: mismatch between call count and result count."""

    def test_mismatch_skips_reorder(self):
        """When counts don't match, leave messages unchanged."""
        req = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
                {"role": "assistant", "content": [_tc("c1"), _tc("c2", city="LA")]},
                # Only 1 result for 2 calls
                {"role": "tool", "content": [_tr("c1")]},
            ]
        }
        result = unwind_parallel_tool_calls_ir(req)
        # Should not be modified (mismatch)
        assert result is req

    def test_extra_results_skips_reorder(self):
        """More results than calls — skip reorder."""
        req = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
                {"role": "assistant", "content": [_tc("c1"), _tc("c2", city="LA")]},
                {"role": "tool", "content": [_tr("c1")]},
                {"role": "tool", "content": [_tr("c2")]},
                {"role": "tool", "content": [_tr("c3")]},  # extra
            ]
        }
        result = unwind_parallel_tool_calls_ir(req)
        assert result is req


class TestUnwindMultiTurn:
    """Multi-turn conversations with mixed tool call patterns."""

    def test_multi_turn_mixed(self):
        """Conversation with both single and parallel tool calls."""
        req = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "weather NYC?"}]},
                # Turn 1: single call (no change)
                {"role": "assistant", "content": [_tc("c1")]},
                {"role": "tool", "content": [_tr("c1", "Sunny")]},
                # Turn 2: model responds
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "NYC is sunny"}],
                },
                # Turn 3: user asks for more
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "now LA and SF?"}],
                },
                # Turn 4: parallel calls (should be unwound)
                {
                    "role": "assistant",
                    "content": [_tc("c2", city="LA"), _tc("c3", city="SF")],
                },
                {"role": "tool", "content": [_tr("c2", "Cloudy")]},
                {"role": "tool", "content": [_tr("c3", "Rainy")]},
            ]
        }
        result = unwind_parallel_tool_calls_ir(req)
        msgs = result["messages"]
        # user(0), assistant(1,c1), tool(2), assistant(3,text), user(4),
        # assistant(5,c2), tool(6), assistant(7,c3), tool(8)
        assert len(msgs) == 9
        # First 5 unchanged
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        assert len([p for p in msgs[1]["content"] if p["type"] == "tool_call"]) == 1
        assert msgs[2]["role"] == "tool"
        assert msgs[3]["role"] == "assistant"
        assert msgs[4]["role"] == "user"
        # Unwound parallel calls
        assert msgs[5]["role"] == "assistant"
        assert msgs[6]["role"] == "tool"
        assert msgs[7]["role"] == "assistant"
        assert msgs[8]["role"] == "tool"

    def test_preserves_other_request_fields(self):
        """Non-messages fields are preserved in the returned request."""
        req = {
            "model": "gemini25flash",
            "tools": [{"type": "function", "name": "test"}],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": [_tc("c1"), _tc("c2", city="LA")]},
                {"role": "tool", "content": [_tr("c1")]},
                {"role": "tool", "content": [_tr("c2")]},
            ],
        }
        result = unwind_parallel_tool_calls_ir(req)
        assert result["model"] == "gemini25flash"
        assert result["tools"] == [{"type": "function", "name": "test"}]


class TestUnwindReportsTheReorder:
    """A moved non-call part is reported, not silently relocated (#1492)."""

    @staticmethod
    def _two_call_request(first_content: list[dict]) -> dict:
        """One assistant turn with two calls, answered in order."""
        return {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "go"}]},
                {"role": "assistant", "content": first_content},
                {"role": "tool", "content": [_tr("c1")]},
                {"role": "tool", "content": [_tr("c2")]},
            ]
        }

    def test_a_part_after_the_first_call_is_reported(self):
        """``[c1, text A, c2]``: text A moves ahead of c1, and says so."""
        sink = WarningSink()
        with recording_to(sink):
            result = unwind_parallel_tool_calls_ir(
                self._two_call_request(
                    [
                        _tc("c1"),
                        {"type": "text", "text": "between the calls"},
                        _tc("c2", city="LA"),
                    ]
                )
            )

        # The move itself: all non-call parts are ahead of the first call.
        first_assistant = result["messages"][1]["content"]
        assert [p["type"] for p in first_assistant] == ["text", "tool_call"]

        assert len(sink) == 1
        assert sink[0].code == DOWNGRADED
        assert sink[0].subject == "content_order"
        assert "moved 1 non-tool-call part(s) (text)" in sink[0]
        assert "2 tool calls" in sink[0]

    def test_a_reasoning_part_after_the_first_call_is_reported(self):
        """Reasoning moves the same way, and the report names it."""
        sink = WarningSink()
        with recording_to(sink):
            unwind_parallel_tool_calls_ir(
                self._two_call_request(
                    [
                        _tc("c1"),
                        {"type": "reasoning", "reasoning": "thinking"},
                        _tc("c2", city="LA"),
                    ]
                ),
                warnings=sink,
            )

        assert len(sink) == 1
        assert "(reasoning)" in sink[0]

    def test_parts_already_ahead_of_the_calls_are_not_reported(self):
        """``[text A, c1, c2]`` keeps its order, so there is nothing to say."""
        sink = WarningSink()
        with recording_to(sink):
            unwind_parallel_tool_calls_ir(
                self._two_call_request(
                    [
                        {"type": "text", "text": "Let me check both"},
                        _tc("c1"),
                        _tc("c2", city="LA"),
                    ]
                ),
                warnings=sink,
            )

        assert list(sink) == []

    def test_several_moved_parts_are_counted_in_one_report(self):
        """``[text A, c1, text B, reasoning R, c2]``: one report, two parts."""
        sink = WarningSink()
        with recording_to(sink):
            unwind_parallel_tool_calls_ir(
                self._two_call_request(
                    [
                        {"type": "text", "text": "A"},
                        _tc("c1"),
                        {"type": "text", "text": "B"},
                        {"type": "reasoning", "reasoning": "R"},
                        _tc("c2", city="LA"),
                    ]
                )
            )

        assert len(sink) == 1
        assert "moved 2 non-tool-call part(s) (text, reasoning)" in sink[0]

    def test_two_unwound_turns_are_told_apart_by_message_index(self):
        """Two turns in one request name different messages, so both logs differ.

        Without the index the two reports are byte-identical, and a log with
        two entries reads as one entry written twice (#1492 review).
        """
        sink = WarningSink()
        with recording_to(sink):
            unwind_parallel_tool_calls_ir(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "go"}],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                _tc("c1"),
                                {"type": "text", "text": "first"},
                                _tc("c2", city="LA"),
                            ],
                        },
                        {"role": "tool", "content": [_tr("c1")]},
                        {"role": "tool", "content": [_tr("c2")]},
                        {
                            "role": "user",
                            "content": [{"type": "text", "text": "again"}],
                        },
                        {
                            "role": "assistant",
                            "content": [
                                _tc("c3"),
                                {"type": "text", "text": "second"},
                                _tc("c4", city="LA"),
                            ],
                        },
                        {"role": "tool", "content": [_tr("c3")]},
                        {"role": "tool", "content": [_tr("c4")]},
                    ]
                }
            )

        assert len(sink) == 2
        assert "message index 1" in sink[0]
        assert "message index 5" in sink[1]
        assert str(sink[0]) != str(sink[1])
