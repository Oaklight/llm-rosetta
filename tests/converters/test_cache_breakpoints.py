"""Tests for auto cache breakpoint injection.

Covers the helper function directly (unit tests) and an end-to-end
round-trip through OpenAI Chat → IR → Anthropic with cache_control.
"""

from __future__ import annotations

import copy
from typing import Any, cast

from llm_rosetta.converters.base.helpers.cache_breakpoints import (
    inject_cache_breakpoints,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

_EPHEMERAL = {"type": "ephemeral"}


def _tool(name: str, **extra: Any) -> dict[str, Any]:
    return {
        "type": "function",
        "name": name,
        "description": f"Tool {name}",
        "parameters": {"type": "object"},
        **extra,
    }


def _user_msg(text: str, **extra: Any) -> dict[str, Any]:
    return {
        "role": "user",
        "content": [{"type": "text", "text": text, **extra}],
    }


def _assistant_msg(text: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": [{"type": "text", "text": text}],
    }


def _full_ir_request() -> dict[str, Any]:
    """IR request with all segments: tools, system, 3 user messages."""
    return {
        "model": "claude-opus-4",
        "system_instruction": [
            {"type": "text", "text": "You are helpful."},
        ],
        "tools": [_tool("read_file"), _tool("write_file"), _tool("search")],
        "messages": [
            _user_msg("first question"),
            _assistant_msg("answer 1"),
            _user_msg("second question"),
            _assistant_msg("answer 2"),
            _user_msg("third question"),
        ],
    }


def _count_cache_hints(ir: dict[str, Any]) -> int:
    """Count total cache_hint markers across all segments."""
    count = 0
    for part in ir.get("system_instruction", []):
        if part.get("cache_hint") is not None:
            count += 1
    for tool in ir.get("tools", []):
        if tool.get("cache_hint") is not None:
            count += 1
    for msg in ir.get("messages", []):
        for part in msg.get("content", []):
            if isinstance(part, dict) and part.get("cache_hint") is not None:
                count += 1
    return count


def _hint_locations(ir: dict[str, Any]) -> list[str]:
    """Return human-readable locations of cache_hint markers."""
    locs: list[str] = []
    for i, tool in enumerate(ir.get("tools", [])):
        if tool.get("cache_hint") is not None:
            locs.append(f"tools[{i}]")
    for i, part in enumerate(ir.get("system_instruction", [])):
        if part.get("cache_hint") is not None:
            locs.append(f"system[{i}]")
    for i, msg in enumerate(ir.get("messages", [])):
        for j, part in enumerate(msg.get("content", [])):
            if isinstance(part, dict) and part.get("cache_hint") is not None:
                locs.append(f"msg[{i}].content[{j}] role={msg['role']}")
    return locs


# ---------------------------------------------------------------------------
# none_only mode (default)
# ---------------------------------------------------------------------------


class TestNoneOnlyMode:
    def test_full_request_places_4_breakpoints(self):
        ir = _full_ir_request()
        inject_cache_breakpoints(ir)
        assert _count_cache_hints(ir) == 4
        locs = _hint_locations(ir)
        assert "tools[2]" in locs  # last tool
        assert "system[0]" in locs  # system tail
        # Last two user messages (indices 2 and 4 in messages list)
        assert any("msg[4]" in loc and "user" in loc for loc in locs)
        assert any("msg[2]" in loc and "user" in loc for loc in locs)

    def test_no_tools(self):
        ir = _full_ir_request()
        del ir["tools"]
        inject_cache_breakpoints(ir)
        assert _count_cache_hints(ir) == 3  # system + 2 user msgs

    def test_no_system(self):
        ir = _full_ir_request()
        del ir["system_instruction"]
        inject_cache_breakpoints(ir)
        assert _count_cache_hints(ir) == 3  # tools + 2 user msgs

    def test_single_user_message(self):
        ir = {
            "model": "claude-opus-4",
            "system_instruction": [{"type": "text", "text": "sys"}],
            "tools": [_tool("t1")],
            "messages": [_user_msg("only question")],
        }
        inject_cache_breakpoints(ir)
        assert _count_cache_hints(ir) == 3  # tools + system + 1 user msg

    def test_no_user_messages(self):
        ir = {
            "model": "claude-opus-4",
            "system_instruction": [{"type": "text", "text": "sys"}],
            "tools": [_tool("t1")],
            "messages": [_assistant_msg("hi")],
        }
        inject_cache_breakpoints(ir)
        assert _count_cache_hints(ir) == 2  # tools + system only

    def test_empty_messages(self):
        ir = {
            "model": "claude-opus-4",
            "messages": [],
        }
        inject_cache_breakpoints(ir)
        assert _count_cache_hints(ir) == 0

    def test_minimal_request(self):
        ir = {"model": "claude-opus-4", "messages": [_user_msg("hi")]}
        inject_cache_breakpoints(ir)
        assert _count_cache_hints(ir) == 1  # just the user msg

    def test_existing_cache_hint_skips(self):
        ir = _full_ir_request()
        ir["tools"][0]["cache_hint"] = {"type": "ephemeral"}
        original = copy.deepcopy(ir)
        inject_cache_breakpoints(ir)
        assert ir == original  # no changes

    def test_existing_hint_on_system_skips(self):
        ir = _full_ir_request()
        ir["system_instruction"][0]["cache_hint"] = {"type": "ephemeral"}
        original = copy.deepcopy(ir)
        inject_cache_breakpoints(ir)
        assert ir == original

    def test_existing_hint_on_message_skips(self):
        ir = _full_ir_request()
        ir["messages"][0]["content"][0]["cache_hint"] = {"type": "ephemeral"}
        original = copy.deepcopy(ir)
        inject_cache_breakpoints(ir)
        assert ir == original

    def test_idempotent(self):
        ir = _full_ir_request()
        inject_cache_breakpoints(ir)
        snapshot = copy.deepcopy(ir)
        # Second call sees existing hints → no-op
        inject_cache_breakpoints(ir)
        assert ir == snapshot

    def test_cache_hint_value(self):
        ir = _full_ir_request()
        inject_cache_breakpoints(ir)
        assert ir["tools"][-1]["cache_hint"] == _EPHEMERAL
        assert ir["system_instruction"][-1]["cache_hint"] == _EPHEMERAL

    def test_no_tools_no_system_two_users(self):
        ir = {
            "model": "claude-opus-4",
            "messages": [
                _user_msg("q1"),
                _assistant_msg("a1"),
                _user_msg("q2"),
            ],
        }
        inject_cache_breakpoints(ir)
        assert _count_cache_hints(ir) == 2  # both user messages


# ---------------------------------------------------------------------------
# fill_gaps mode
# ---------------------------------------------------------------------------


class TestFillGapsMode:
    def test_fills_missing_segments(self):
        ir = _full_ir_request()
        # Pre-mark tools only
        ir["tools"][-1]["cache_hint"] = dict(_EPHEMERAL)
        inject_cache_breakpoints(ir, mode="fill_gaps")
        # Tools already had hint → not double-marked; system + 2 msgs filled
        assert _count_cache_hints(ir) == 4  # 1 existing + 3 new

    def test_all_segments_hinted_is_noop(self):
        ir = _full_ir_request()
        ir["tools"][-1]["cache_hint"] = dict(_EPHEMERAL)
        ir["system_instruction"][0]["cache_hint"] = dict(_EPHEMERAL)
        ir["messages"][-1]["content"][0]["cache_hint"] = dict(_EPHEMERAL)
        original = copy.deepcopy(ir)
        inject_cache_breakpoints(ir, mode="fill_gaps")
        assert ir == original

    def test_no_existing_hints_fills_all(self):
        ir = _full_ir_request()
        inject_cache_breakpoints(ir, mode="fill_gaps")
        assert _count_cache_hints(ir) == 4

    def test_existing_hints_count_against_budget(self):
        """Pre-existing hints must not push the total past Anthropic's limit."""
        ir = _full_ir_request()
        # Two hints already present in tools — budget drops to 2.
        ir["tools"][0]["cache_hint"] = dict(_EPHEMERAL)
        ir["tools"][1]["cache_hint"] = dict(_EPHEMERAL)
        inject_cache_breakpoints(ir, mode="fill_gaps")
        assert _count_cache_hints(ir) <= 4

    def test_budget_exhausted_by_existing_hints(self):
        """Four pre-existing hints leave no budget for injection."""
        ir = _full_ir_request()
        for i in range(3):
            ir["tools"][i]["cache_hint"] = dict(_EPHEMERAL)
        ir["system_instruction"][0]["cache_hint"] = dict(_EPHEMERAL)
        original = copy.deepcopy(ir)
        inject_cache_breakpoints(ir, mode="fill_gaps")
        assert ir == original
        assert _count_cache_hints(ir) == 4


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_tools_list(self):
        ir = {
            "model": "claude-opus-4",
            "tools": [],
            "system_instruction": [{"type": "text", "text": "sys"}],
            "messages": [_user_msg("q")],
        }
        inject_cache_breakpoints(ir)
        assert _count_cache_hints(ir) == 2  # system + user

    def test_empty_system_list(self):
        ir = {
            "model": "claude-opus-4",
            "system_instruction": [],
            "tools": [_tool("t1")],
            "messages": [_user_msg("q")],
        }
        inject_cache_breakpoints(ir)
        assert _count_cache_hints(ir) == 2  # tools + user

    def test_user_message_with_empty_content(self):
        ir = {
            "model": "claude-opus-4",
            "messages": [{"role": "user", "content": []}],
        }
        inject_cache_breakpoints(ir)
        assert _count_cache_hints(ir) == 0

    def test_multipart_user_message_marks_last_part(self):
        ir = {
            "model": "claude-opus-4",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "part1"},
                        {"type": "text", "text": "part2"},
                        {"type": "text", "text": "part3"},
                    ],
                }
            ],
        }
        inject_cache_breakpoints(ir)
        assert ir["messages"][0]["content"][2].get("cache_hint") == _EPHEMERAL
        assert ir["messages"][0]["content"][0].get("cache_hint") is None
        assert ir["messages"][0]["content"][1].get("cache_hint") is None


# ---------------------------------------------------------------------------
# Round-trip: OpenAI Chat → IR → auto_cache → Anthropic body
# ---------------------------------------------------------------------------


class TestRoundTrip:
    def test_openai_to_anthropic_has_cache_control(self):
        """End-to-end: an OpenAI Chat request converted to Anthropic
        body should have cache_control on the expected blocks after
        auto_cache_breakpoints fires."""
        from llm_rosetta.converters.anthropic.converter import AnthropicConverter
        from llm_rosetta.converters.openai_chat.converter import OpenAIChatConverter

        openai_request = {
            "model": "claude-opus-4",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"},
                {"role": "assistant", "content": "Hi there!"},
                {"role": "user", "content": "What is 2+2?"},
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "calculator",
                        "description": "Calculate math",
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }

        # OpenAI → IR
        oai = OpenAIChatConverter()
        ir = oai.request_from_provider(openai_request)
        ir_dict = cast(dict[str, Any], ir)

        # Inject cache breakpoints
        inject_cache_breakpoints(ir_dict)

        # Verify IR has cache_hint
        assert _count_cache_hints(ir_dict) == 4

        # IR → Anthropic (returns (body, warnings) tuple)
        anthropic = AnthropicConverter()
        anthropic_body, _warnings = anthropic.request_to_provider(ir)

        # Verify Anthropic body has cache_control
        # On tools
        tools = anthropic_body.get("tools", [])
        assert any(t.get("cache_control") is not None for t in tools), (
            "Expected cache_control on at least one tool"
        )

        # On system
        system = anthropic_body.get("system", [])
        if isinstance(system, list):
            assert any(s.get("cache_control") is not None for s in system), (
                "Expected cache_control on system block"
            )

        # On messages
        messages = anthropic_body.get("messages", [])
        user_msgs_with_cc = [
            m
            for m in messages
            if m.get("role") == "user"
            and isinstance(m.get("content"), list)
            and any(
                p.get("cache_control") is not None
                for p in m["content"]
                if isinstance(p, dict)
            )
        ]
        assert len(user_msgs_with_cc) >= 1, "Expected cache_control on user messages"
