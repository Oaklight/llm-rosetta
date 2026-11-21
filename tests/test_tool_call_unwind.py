"""Tests for Argo parallel tool call unwind utility."""

import copy
import json
from typing import Any

import pytest

from llm_rosetta import ChatCompletionsConverter, MessagesConverter
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


class TestUnwindBatchedResults:
    """Results batched in one ``role: tool`` message (#1611).

    Anthropic sends every result of a parallel turn in one ``user`` message;
    in IR that is one tool message holding several ``tool_result`` parts.
    The unwind counts results, not messages, and gives each call exactly its
    own result.
    """

    @staticmethod
    def _batched(*results: dict) -> dict:
        return {"role": "tool", "content": list(results), "batch_id": "0"}

    def test_two_batched_results_are_split_and_paired(self):
        req = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
                {"role": "assistant", "content": [_tc("c1"), _tc("c2", city="LA")]},
                self._batched(_tr("c1", "Sunny"), _tr("c2", "Cloudy")),
            ]
        }
        before = copy.deepcopy(req)

        result = unwind_parallel_tool_calls_ir(req)

        assert result["messages"] == [
            before["messages"][0],
            {"role": "assistant", "content": [_tc("c1")]},
            {"role": "tool", "content": [_tr("c1", "Sunny")], "batch_id": "0"},
            {"role": "assistant", "content": [_tc("c2", city="LA")]},
            {"role": "tool", "content": [_tr("c2", "Cloudy")], "batch_id": "0"},
        ]
        assert req == before  # the input is not mutated

    def test_batched_results_in_reverse_order_pair_by_id(self):
        req = {
            "messages": [
                {"role": "assistant", "content": [_tc("c1"), _tc("c2", city="LA")]},
                self._batched(_tr("c2", "Cloudy"), _tr("c1", "Sunny")),
            ]
        }

        msgs = unwind_parallel_tool_calls_ir(req)["messages"]

        assert [m["role"] for m in msgs] == ["assistant", "tool"] * 2
        assert msgs[1]["content"] == [_tr("c1", "Sunny")]
        assert msgs[3]["content"] == [_tr("c2", "Cloudy")]

    def test_a_batch_and_a_single_result_together_count_as_three(self):
        req = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [_tc("c1"), _tc("c2", city="LA"), _tc("c3", city="SF")],
                },
                self._batched(_tr("c1", "a"), _tr("c2", "b")),
                {"role": "tool", "content": [_tr("c3", "c")]},
            ]
        }

        msgs = unwind_parallel_tool_calls_ir(req)["messages"]

        assert [m["role"] for m in msgs] == ["assistant", "tool"] * 3
        assert [m["content"] for m in msgs[1::2]] == [
            [_tr("c1", "a")],
            [_tr("c2", "b")],
            [_tr("c3", "c")],
        ]
        # The single-result message is passed through as the same object.
        assert msgs[5] is req["messages"][2]

    def test_batched_results_fewer_than_calls_skip_reorder(self):
        sink = WarningSink()
        req = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [_tc("c1"), _tc("c2", city="LA"), _tc("c3", city="SF")],
                },
                self._batched(_tr("c1"), _tr("c2")),
            ]
        }
        with recording_to(sink):
            result = unwind_parallel_tool_calls_ir(req)

        assert result is req
        assert len(sink) == 1
        assert (
            "3 tool calls but 2 result unit(s) (2 tool_result part(s) "
            "in 1 tool message(s))" in sink[0]
        )

    def test_batched_results_more_than_calls_skip_reorder(self):
        req = {
            "messages": [
                {"role": "assistant", "content": [_tc("c1"), _tc("c2", city="LA")]},
                self._batched(_tr("c1"), _tr("c2"), _tr("c3")),
            ]
        }
        assert unwind_parallel_tool_calls_ir(req) is req

    def test_a_batch_carrying_a_non_result_part_is_left_unchanged(self):
        """A split cannot say which result a stray part belongs to."""
        sink = WarningSink()
        req = {
            "messages": [
                {"role": "assistant", "content": [_tc("c1"), _tc("c2", city="LA")]},
                {
                    "role": "tool",
                    "content": [_tr("c1"), {"type": "text", "text": "note"}, _tr("c2")],
                },
            ]
        }
        with recording_to(sink):
            result = unwind_parallel_tool_calls_ir(req)

        assert result is req
        assert len(sink) == 1
        assert sink[0].code == DOWNGRADED
        assert "non-result part" in sink[0]

    @pytest.mark.parametrize("content", [None, "done", {"text": "x"}])
    def test_a_tool_message_without_list_content_is_not_a_batch(self, content):
        """Non-list content passes through as one message, as before #1611."""
        odd = {"role": "tool", "content": content}
        second = {"role": "tool", "content": [_tr("c2", "Cloudy")]}
        req = {
            "messages": [
                {"role": "assistant", "content": [_tc("c1"), _tc("c2", city="LA")]},
                odd,
                second,
            ]
        }

        msgs = unwind_parallel_tool_calls_ir(req)["messages"]

        assert [m["role"] for m in msgs] == ["assistant", "tool"] * 2
        assert msgs[1] is odd  # positional pairing, the object untouched
        assert msgs[3] is second

    def test_a_lone_message_without_list_content_is_a_mismatch(self):
        sink = WarningSink()
        req = {
            "messages": [
                {"role": "assistant", "content": [_tc("c1"), _tc("c2", city="LA")]},
                {"role": "tool", "content": None},
            ]
        }
        with recording_to(sink):
            result = unwind_parallel_tool_calls_ir(req)

        assert result is req
        assert len(sink) == 1

    def test_the_mismatch_report_says_what_it_counted(self):
        """Pairing units, results and messages are named apart."""
        sink = WarningSink()
        req = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [_tc("c1"), _tc("c2", city="LA"), _tc("c3", city="SF")],
                },
                {"role": "tool", "content": "ok"},
                {"role": "tool", "content": [_tr("c2")]},
            ]
        }
        with recording_to(sink):
            unwind_parallel_tool_calls_ir(req)

        assert len(sink) == 1
        assert (
            "3 tool calls but 2 result unit(s) (1 tool_result part(s) "
            "in 2 tool message(s))" in sink[0]
        )

    @pytest.mark.parametrize(
        "result_ids",
        [("c1", "c1"), ("cX", "cY"), ("c1", "c3")],
        ids=["duplicate", "foreign", "one_missing"],
    )
    def test_a_batch_whose_ids_do_not_match_the_calls_is_left_unchanged(
        self, result_ids
    ):
        """Counts match after the split, ids do not: no positional guess."""
        sink = WarningSink()
        req = {
            "messages": [
                {"role": "assistant", "content": [_tc("c1"), _tc("c2", city="LA")]},
                self._batched(*(_tr(rid) for rid in result_ids)),
            ]
        }
        with recording_to(sink):
            result = unwind_parallel_tool_calls_ir(req)

        assert result is req
        assert len(sink) == 1
        assert sink[0].code == DOWNGRADED
        assert "result ids do not match the call ids" in sink[0]

    def test_the_id_check_counts_duplicates_not_just_names(self):
        """Same id set, different multiplicity: still not the calls' answers."""
        req = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [_tc("c1"), _tc("c1", city="LA"), _tc("c2", city="SF")],
                },
                self._batched(_tr("c1"), _tr("c2"), _tr("c2")),
            ]
        }
        assert unwind_parallel_tool_calls_ir(req) is req

    def test_a_reordered_batch_with_the_right_ids_still_unwinds(self):
        req = {
            "messages": [
                {"role": "assistant", "content": [_tc("c1"), _tc("c2", city="LA")]},
                self._batched(_tr("c2", "Cloudy"), _tr("c1", "Sunny")),
            ]
        }
        sink = WarningSink()
        with recording_to(sink):
            msgs = unwind_parallel_tool_calls_ir(req)["messages"]

        assert len(sink) == 0
        assert [m["content"][0]["tool_call_id"] for m in msgs] == [
            "c1",
            "c1",
            "c2",
            "c2",
        ]

    def test_one_result_per_message_output_is_unchanged(self):
        """The pre-#1611 shape keeps its exact output, objects included."""
        tool_1 = {"role": "tool", "content": [_tr("c1", "Sunny")], "batch_id": "0"}
        tool_2 = {"role": "tool", "content": [_tr("c2", "Cloudy")], "batch_id": "0"}
        req = {
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "weather?"}]},
                {"role": "assistant", "content": [_tc("c1"), _tc("c2", city="LA")]},
                tool_2,
                tool_1,
            ]
        }

        msgs = unwind_parallel_tool_calls_ir(req)["messages"]

        assert json.dumps(msgs, sort_keys=True) == json.dumps(
            [
                req["messages"][0],
                {"role": "assistant", "content": [_tc("c1")]},
                tool_1,
                {"role": "assistant", "content": [_tc("c2", city="LA")]},
                tool_2,
            ],
            sort_keys=True,
        )
        assert msgs[2] is tool_1
        assert msgs[4] is tool_2


# ---------------------------------------------------------------------------
# Round trips through the converters (#1611)
# ---------------------------------------------------------------------------


def _anthropic_body(tool_results: list[dict]) -> dict:
    """An Anthropic request whose parallel results follow in ``tool_results``."""
    return {
        "model": "m",
        "max_tokens": 64,
        "messages": [
            {"role": "user", "content": "weather in NYC and LA?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Checking both."},
                    {
                        "type": "tool_use",
                        "id": "t1",
                        "name": "w",
                        "input": {"c": "NYC"},
                    },
                    {"type": "tool_use", "id": "t2", "name": "w", "input": {"c": "LA"}},
                ],
            },
            *tool_results,
        ],
    }


_BATCHED = [
    {
        "role": "user",
        "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": "Sunny"},
            {"type": "tool_result", "tool_use_id": "t2", "content": "Cloudy"},
        ],
    }
]

_ONE_PER_MESSAGE = [
    {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Sunny"}],
    },
    {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "Cloudy"}],
    },
]

_UNWOUND_ANTHROPIC = [
    {"role": "user", "content": [{"type": "text", "text": "weather in NYC and LA?"}]},
    {
        "role": "assistant",
        "content": [
            {"type": "text", "text": "Checking both."},
            {"type": "tool_use", "id": "t1", "name": "w", "input": {"c": "NYC"}},
        ],
    },
    {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "Sunny"}],
    },
    {
        "role": "assistant",
        "content": [
            {"type": "tool_use", "id": "t2", "name": "w", "input": {"c": "LA"}}
        ],
    },
    {
        "role": "user",
        "content": [{"type": "tool_result", "tool_use_id": "t2", "content": "Cloudy"}],
    },
]


class TestUnwindRoundTrips:
    """A = Anthropic Messages, B = Chat Completions; the unwind runs on IR."""

    @staticmethod
    def _unwound_ir(body: dict) -> Any:
        ir: Any = MessagesConverter().request_from_provider(body)
        return unwind_parallel_tool_calls_ir(ir)

    def test_batched_results_are_one_tool_message_in_ir(self):
        """Precondition: the batched shape really is one multi-result message."""
        ir: Any = MessagesConverter().request_from_provider(_anthropic_body(_BATCHED))
        tool_msgs = [m for m in ir["messages"] if m.get("role") == "tool"]
        assert len(tool_msgs) == 1
        assert len(tool_msgs[0]["content"]) == 2

    @pytest.mark.parametrize(
        "results", [_BATCHED, _ONE_PER_MESSAGE], ids=["batched", "one_per_message"]
    )
    def test_a_ir_a(self, results):
        body, _ = MessagesConverter().request_to_provider(
            self._unwound_ir(_anthropic_body(results))
        )
        assert body["messages"] == _UNWOUND_ANTHROPIC

    @pytest.mark.parametrize(
        "results", [_BATCHED, _ONE_PER_MESSAGE], ids=["batched", "one_per_message"]
    )
    def test_a_ir_b(self, results):
        body, _ = ChatCompletionsConverter().request_to_provider(
            self._unwound_ir(_anthropic_body(results))
        )
        msgs = body["messages"]
        assert [m["role"] for m in msgs] == [
            "user",
            "assistant",
            "tool",
            "assistant",
            "tool",
        ]
        assert [len(m["tool_calls"]) for m in msgs[1::2]] == [1, 1]
        assert [m["tool_calls"][0]["id"] for m in msgs[1::2]] == ["t1", "t2"]
        assert [(m["tool_call_id"], m["content"]) for m in msgs[2::2]] == [
            ("t1", "Sunny"),
            ("t2", "Cloudy"),
        ]

    @pytest.mark.parametrize(
        "results", [_BATCHED, _ONE_PER_MESSAGE], ids=["batched", "one_per_message"]
    )
    def test_a_ir_b_ir_a(self, results):
        chat_body, _ = ChatCompletionsConverter().request_to_provider(
            self._unwound_ir(_anthropic_body(results))
        )
        ir_again = ChatCompletionsConverter().request_from_provider(chat_body)
        body, _ = MessagesConverter().request_to_provider(ir_again)
        assert body["messages"] == _UNWOUND_ANTHROPIC

    def test_the_unwind_is_idempotent_on_its_own_output(self):
        ir = self._unwound_ir(_anthropic_body(_BATCHED))
        assert unwind_parallel_tool_calls_ir(ir) is ir
