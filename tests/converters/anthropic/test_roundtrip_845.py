"""Regression tests for issue #845: Anthropic round-trip fidelity.

Three bugs:
1. Mid-conversation system messages dropped/hoisted
2. redacted_thinking blocks dropped
3. Extra caller/is_error fields added
"""

import json


from llm_rosetta import ConversionPipeline


# ---------------------------------------------------------------------------
# Bug 1: Mid-conversation system messages
# ---------------------------------------------------------------------------


class TestSystemMessagePreservation:
    """Mid-conversation system messages must not be silently dropped."""

    TOOLS = [
        {
            "name": "run_cmd",
            "description": "run a shell command",
            "input_schema": {
                "type": "object",
                "properties": {"cmd": {"type": "string"}},
            },
        }
    ]

    def _base_messages(self):
        return [
            {"role": "user", "content": "list the files"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "run_cmd",
                        "input": {"cmd": "ls"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "toolu_1",
                        "content": "a.txt b.txt",
                    }
                ],
            },
            {
                "role": "system",
                "content": "[harness] run_cmd is DISABLED from now on.",
            },
            {"role": "user", "content": "now delete b.txt"},
        ]

    def test_late_system_preserved_with_top_level_system(self):
        """Late system message content preserved when top-level system exists."""
        req = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "system": "You are a careful assistant.",
            "tools": self.TOOLS,
            "messages": self._base_messages(),
        }
        pipe = ConversionPipeline("anthropic", "anthropic")
        out = pipe.convert_request(req)

        out_json = json.dumps(out)
        assert "[harness]" in out_json, "Late system message text was dropped"

        # Should be in a user message (as <system> envelope), not lost
        found_in_user = any(
            m["role"] == "user" and "[harness]" in json.dumps(m)
            for m in out["messages"]
        )
        assert found_in_user, "Late system should be rewritten as user envelope"

        # Top-level system text should be preserved (may have cache_control)
        system_parts = out["system"]
        assert len(system_parts) == 1
        assert system_parts[0]["type"] == "text"
        assert system_parts[0]["text"] == "You are a careful assistant."

    def test_late_system_preserved_without_top_level_system(self):
        """Late system message stays in-place, not hoisted to top-level."""
        req = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "tools": self.TOOLS,
            "messages": self._base_messages(),
        }
        pipe = ConversionPipeline("anthropic", "anthropic")
        out = pipe.convert_request(req)

        out_json = json.dumps(out)
        assert "[harness]" in out_json, "Late system message text was dropped"

        # Should NOT be hoisted to top-level system
        system_json = json.dumps(out.get("system") or "")
        assert "[harness]" not in system_json, (
            "Late system was hoisted to top-level system instead of staying in-place"
        )

    def test_leading_system_becomes_top_level(self):
        """A leading system message (before any user/assistant) goes to top-level."""
        req = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "messages": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "hello"},
            ],
        }
        pipe = ConversionPipeline("anthropic", "anthropic")
        out = pipe.convert_request(req)

        system_parts = out.get("system")
        assert system_parts is not None and len(system_parts) == 1
        assert system_parts[0]["type"] == "text"
        assert system_parts[0]["text"] == "You are helpful."


# ---------------------------------------------------------------------------
# Bug 2: redacted_thinking dropped
# ---------------------------------------------------------------------------


class TestRedactedThinking:
    """redacted_thinking blocks must round-trip through IR."""

    def test_redacted_thinking_preserved(self):
        req = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "tools": [
                {
                    "name": "get_weather",
                    "description": "weather",
                    "input_schema": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
            "messages": [
                {"role": "user", "content": "weather in Paris?"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "redacted_thinking",
                            "data": "EqQBCkYIARgCKkD...",
                        },
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {"city": "Paris"},
                        },
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "sunny",
                        }
                    ],
                },
            ],
        }
        pipe = ConversionPipeline("anthropic", "anthropic")
        out = pipe.convert_request(req)

        asst = [m for m in out["messages"] if m["role"] == "assistant"][0]
        types = [p.get("type") for p in asst["content"]]

        assert "redacted_thinking" in types, (
            f"redacted_thinking dropped; got types: {types}"
        )

        redacted = next(
            p for p in asst["content"] if p.get("type") == "redacted_thinking"
        )
        assert redacted["data"] == "EqQBCkYIARgCKkD..."

    def test_thinking_still_works(self):
        """Normal thinking blocks must still round-trip correctly."""
        req = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [
                {"role": "user", "content": "think about this"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "Let me think...",
                            "signature": "sig123",
                        },
                        {"type": "text", "text": "Here's my answer."},
                    ],
                },
            ],
        }
        pipe = ConversionPipeline("anthropic", "anthropic")
        out = pipe.convert_request(req)

        asst = [m for m in out["messages"] if m["role"] == "assistant"][0]
        thinking = next(p for p in asst["content"] if p.get("type") == "thinking")
        assert thinking["thinking"] == "Let me think..."
        assert thinking["signature"] == "sig123"

    def test_mixed_thinking_and_redacted(self):
        """Both thinking and redacted_thinking in the same turn."""
        req = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "thinking": {"type": "enabled", "budget_tokens": 1024},
            "messages": [
                {"role": "user", "content": "think"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "thinking",
                            "thinking": "visible thought",
                            "signature": "sig1",
                        },
                        {
                            "type": "redacted_thinking",
                            "data": "opaque-data-here",
                        },
                        {"type": "text", "text": "answer"},
                    ],
                },
            ],
        }
        pipe = ConversionPipeline("anthropic", "anthropic")
        out = pipe.convert_request(req)

        asst = [m for m in out["messages"] if m["role"] == "assistant"][0]
        types = [p.get("type") for p in asst["content"]]
        assert types == ["thinking", "redacted_thinking", "text"]


# ---------------------------------------------------------------------------
# Bug 3: Extra fields (caller, is_error) added
# ---------------------------------------------------------------------------


class TestNoFieldInflation:
    """Round-trip must not add fields that were not in the input."""

    def test_no_caller_added(self):
        """tool_use without caller must not get caller: {type: direct}."""
        req = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {"city": "Paris"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "sunny",
                        }
                    ],
                },
            ],
        }
        pipe = ConversionPipeline("anthropic", "anthropic")
        out = pipe.convert_request(req)

        asst = [m for m in out["messages"] if m["role"] == "assistant"][0]
        tu = next(p for p in asst["content"] if p.get("type") == "tool_use")
        assert "caller" not in tu, f"caller inflated: {tu}"

    def test_non_default_caller_preserved(self):
        """tool_use with explicit non-default caller must preserve it."""
        req = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "get_weather",
                            "input": {"city": "Paris"},
                            "caller": {"type": "orchestrator", "id": "orch_1"},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "sunny",
                        }
                    ],
                },
            ],
        }
        pipe = ConversionPipeline("anthropic", "anthropic")
        out = pipe.convert_request(req)

        asst = [m for m in out["messages"] if m["role"] == "assistant"][0]
        tu = next(p for p in asst["content"] if p.get("type") == "tool_use")
        assert tu.get("caller") == {"type": "orchestrator", "id": "orch_1"}

    def test_no_is_error_added(self):
        """tool_result without is_error must not get is_error: false."""
        req = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "fn",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "ok",
                        }
                    ],
                },
            ],
        }
        pipe = ConversionPipeline("anthropic", "anthropic")
        out = pipe.convert_request(req)

        for m in out["messages"]:
            if m["role"] == "user" and isinstance(m.get("content"), list):
                for p in m["content"]:
                    if isinstance(p, dict) and p.get("type") == "tool_result":
                        assert "is_error" not in p, f"is_error inflated: {p}"

    def test_is_error_true_preserved(self):
        """tool_result with is_error: true must preserve it."""
        req = {
            "model": "claude-3-5-sonnet-20241022",
            "max_tokens": 100,
            "messages": [
                {"role": "user", "content": "hello"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_1",
                            "name": "fn",
                            "input": {},
                        }
                    ],
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "content": "error occurred",
                            "is_error": True,
                        }
                    ],
                },
            ],
        }
        pipe = ConversionPipeline("anthropic", "anthropic")
        out = pipe.convert_request(req)

        for m in out["messages"]:
            if m["role"] == "user" and isinstance(m.get("content"), list):
                for p in m["content"]:
                    if isinstance(p, dict) and p.get("type") == "tool_result":
                        assert p.get("is_error") is True
