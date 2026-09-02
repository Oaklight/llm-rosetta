"""Tests for ``additional_tools`` input items and nested ``namespace`` tools.

Codex transports its tool definitions inside the Responses ``input`` array
as an ``additional_tools`` item wrapping ``type: "namespace"`` containers,
rather than in the top-level ``tools`` field.  These tests pin the ingest
behaviour: namespaces are flattened into IR tools, the spent item is
dropped, and the resulting request survives conversion to Chat/Anthropic.
"""

from __future__ import annotations

import json
from typing import cast

from llm_rosetta.converters.openai_responses import OpenAIResponsesConverter
from llm_rosetta.converters.openai_responses.tool_ops import (
    NAMESPACE_MARKER_KEY,
    flatten_additional_tools,
    harvest_additional_tools,
    provider_tool_names,
    strip_additional_tools_items,
)
from llm_rosetta.types.ir import Message


def _additional_tools_item() -> dict:
    """Build an ``additional_tools`` item shaped like Codex's."""
    return {
        "type": "additional_tools",
        "id": "at_7c433da1",
        "role": "developer",
        "tools": [
            {
                "type": "namespace",
                "name": "functions",
                "tools": [
                    {
                        "type": "custom",
                        "name": "exec",
                        "description": "Run a shell command.",
                        "format": {"type": "grammar", "syntax": "lark"},
                    },
                    {
                        "type": "function",
                        "name": "wait",
                        "description": "Wait for a while.",
                        "parameters": {
                            "type": "object",
                            "properties": {"ms": {"type": "integer"}},
                            "required": ["ms"],
                        },
                    },
                ],
            },
            {
                "type": "namespace",
                "name": "collaboration",
                "description": "Tools for spawning and managing sub-agents.",
                "tools": [
                    {
                        "type": "function",
                        "name": "spawn_agent",
                        "description": "Spawn a sub-agent.",
                        "parameters": {"type": "object", "properties": {}},
                    }
                ],
            },
        ],
    }


def _codex_request() -> dict:
    """A Responses request carrying tools only via ``additional_tools``."""
    return {
        "model": "gpt-5.4",
        "input": [
            _additional_tools_item(),
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "list the files"}],
            },
        ],
        "tool_choice": "auto",
        "parallel_tool_calls": False,
    }


class TestFlattenAdditionalTools:
    def test_flattens_namespaces_into_bare_named_tools(self):
        tools, warnings = flatten_additional_tools([_additional_tools_item()])
        assert [t["name"] for t in tools] == ["exec", "wait", "spawn_agent"]
        assert warnings == []

    def test_records_originating_namespace(self):
        tools, _ = flatten_additional_tools([_additional_tools_item()])
        by_name = {t["name"]: t for t in tools}
        assert by_name["exec"]["_namespace"] == "functions"
        assert by_name["spawn_agent"]["_namespace"] == "collaboration"

    def test_ignores_non_additional_tools_items(self):
        tools, warnings = flatten_additional_tools(
            [{"type": "message", "role": "user", "content": []}]
        )
        assert tools == []
        assert warnings == []

    def test_bare_tool_entry_without_namespace_is_kept(self):
        item = {
            "type": "additional_tools",
            "tools": [{"type": "function", "name": "ping", "parameters": {}}],
        }
        tools, warnings = flatten_additional_tools([item])
        assert [t["name"] for t in tools] == ["ping"]
        assert "_namespace" not in tools[0]
        assert warnings == []

    def test_duplicate_name_across_namespaces_is_qualified(self):
        item = {
            "type": "additional_tools",
            "tools": [
                {
                    "type": "namespace",
                    "name": "a",
                    "tools": [{"type": "function", "name": "run", "parameters": {}}],
                },
                {
                    "type": "namespace",
                    "name": "b",
                    "tools": [{"type": "function", "name": "run", "parameters": {}}],
                },
            ],
        }
        tools, warnings = flatten_additional_tools([item])
        # Underscore, not "::" — Chat upstreams reject names outside
        # ^[a-zA-Z0-9_-]{1,64}$ and there is no tool-name sanitizer.
        assert [t["name"] for t in tools] == ["run", "b_run"]
        assert any("renamed to 'b_run'" in w for w in warnings)

    def test_unnamed_and_malformed_children_are_skipped_with_warnings(self):
        item = {
            "type": "additional_tools",
            "tools": [
                {
                    "type": "namespace",
                    "name": "ns",
                    "tools": [{"type": "function"}, "not-a-dict"],
                },
                {"type": "namespace", "name": "empty", "tools": []},
            ],
        }
        tools, warnings = flatten_additional_tools([item])
        assert tools == []
        assert len(warnings) == 3
        assert any("Unnamed tool entry" in w for w in warnings)
        assert any("Non-dict tool entry" in w for w in warnings)
        assert any("declared no tools" in w for w in warnings)


class TestStripAdditionalToolsItems:
    def test_removes_only_additional_tools_items(self):
        items = _codex_request()["input"]
        remaining = strip_additional_tools_items(items)
        assert len(remaining) == 1
        assert remaining[0]["type"] == "message"


class TestRequestFromProviderAdditionalTools:
    def setup_method(self):
        self.converter = OpenAIResponsesConverter()

    def test_nested_tools_reach_ir(self):
        result = self.converter.request_from_provider(_codex_request())
        tools = list(result["tools"])
        assert [t["name"] for t in tools] == ["exec", "wait", "spawn_agent"]
        # "custom" is a first-class IR type and survives ingest.
        assert tools[0]["type"] == "custom"
        assert tools[0]["metadata"]["namespace"] == "functions"
        assert tools[0]["metadata"]["format"]["type"] == "grammar"
        assert tools[1]["type"] == "function"

    def test_tool_choice_survives(self):
        """Regression: tools were absent, so tool_choice was stripped as orphaned."""
        result = self.converter.request_from_provider(_codex_request())
        assert result.get("tool_choice") is not None
        assert result.get("tool_config") is not None

    def test_additional_tools_item_does_not_become_a_message(self):
        """It must not strand a content-less assistant message."""
        result = self.converter.request_from_provider(_codex_request())
        messages = cast(list[Message], result["messages"])
        assert len(messages) == 1
        assert messages[0]["role"] == "user"

    def test_top_level_tools_are_merged_with_nested_ones(self):
        request = _codex_request()
        request["tools"] = [
            {"type": "function", "name": "shell", "parameters": {"type": "object"}}
        ]
        result = self.converter.request_from_provider(request)
        assert [t["name"] for t in result["tools"]] == [
            "shell",
            "exec",
            "wait",
            "spawn_agent",
        ]

    def test_malformed_additional_tools_item_is_left_alone(self):
        """Nothing harvested → existing passthrough behaviour is unchanged."""
        request = _codex_request()
        request["input"][0] = {"type": "additional_tools", "tools": []}
        result = self.converter.request_from_provider(request)
        assert not result.get("tools")


class TestAdditionalToolsCrossFormat:
    def test_roundtrip_to_openai_chat(self):
        from llm_rosetta.pipeline import ConversionPipeline

        out = ConversionPipeline("openai_responses", "openai_chat").convert_request(
            _codex_request()
        )
        names = [t["function"]["name"] for t in out["tools"]]
        assert names == ["exec", "wait", "spawn_agent"]
        assert out["tool_choice"] == "auto"
        assert not [
            m
            for m in out["messages"]
            if m.get("role") == "assistant"
            and not m.get("content")
            and not m.get("tool_calls")
        ]

    def test_roundtrip_to_anthropic(self):
        from llm_rosetta.pipeline import ConversionPipeline

        out = ConversionPipeline("openai_responses", "anthropic").convert_request(
            _codex_request()
        )
        assert [t["name"] for t in out["tools"]] == ["exec", "wait", "spawn_agent"]


class TestAdditionalToolsDegenerate:
    """An ``additional_tools`` item that yields no usable tools."""

    def test_all_entries_malformed_still_strips_item(self):
        """The spent item must go even when nothing was harvested.

        Left in ``input`` it reaches the passthrough path and strands a
        content-less assistant message.
        """
        items = [
            {
                "type": "additional_tools",
                "tools": [{"type": "namespace", "name": "functions", "tools": [{}]}],
            },
            {"type": "message", "role": "user", "content": "hi"},
        ]
        warnings: list[str] = []
        tools, remaining = harvest_additional_tools(items, warnings)
        assert tools == []
        assert not [
            i
            for i in remaining
            if isinstance(i, dict) and i.get("type") == "additional_tools"
        ]

    def test_all_entries_malformed_still_warns(self):
        items = [
            {
                "type": "additional_tools",
                "tools": [{"type": "namespace", "name": "functions", "tools": [{}]}],
            }
        ]
        warnings: list[str] = []
        harvest_additional_tools(items, warnings)
        assert warnings, "a wholly unusable item must not be discarded silently"

    def test_no_additional_tools_leaves_input_identical(self):
        items = [{"type": "message", "role": "user", "content": "hi"}]
        warnings: list[str] = []
        tools, remaining = harvest_additional_tools(items, warnings)
        assert tools == []
        assert remaining is items
        assert warnings == []

    def test_degenerate_item_produces_no_empty_assistant(self):
        from llm_rosetta.pipeline import ConversionPipeline

        out = ConversionPipeline("openai_responses", "openai_chat").convert_request(
            {
                "model": "gpt-5.6-sol",
                "input": [
                    {
                        "type": "additional_tools",
                        "tools": [
                            {"type": "namespace", "name": "functions", "tools": [{}]}
                        ],
                    },
                    {"type": "message", "role": "user", "content": "hi"},
                ],
            }
        )
        assert not [
            m
            for m in out["messages"]
            if m.get("role") == "assistant"
            and not m.get("content")
            and not m.get("tool_calls")
        ]


class TestTopLevelCollision:
    """Nested tools must not silently duplicate a top-level tool name."""

    def test_namespaced_child_colliding_with_top_level_is_qualified(self):
        items = [
            {
                "type": "additional_tools",
                "tools": [
                    {
                        "type": "namespace",
                        "name": "functions",
                        "tools": [{"type": "function", "name": "exec"}],
                    }
                ],
            }
        ]
        warnings: list[str] = []
        tools, _ = harvest_additional_tools(
            items, warnings, [{"type": "function", "name": "exec"}]
        )
        assert [t["name"] for t in tools] == ["functions_exec"]
        assert any("Duplicate tool name 'exec'" in w for w in warnings)

    def test_flat_entry_colliding_with_top_level_is_skipped(self):
        items = [
            {
                "type": "additional_tools",
                "tools": [{"type": "function", "name": "exec"}],
            }
        ]
        warnings: list[str] = []
        tools, _ = harvest_additional_tools(
            items, warnings, [{"type": "function", "name": "exec"}]
        )
        assert tools == []
        assert any("Duplicate tool name 'exec'" in w for w in warnings)

    def test_no_duplicate_names_reach_upstream(self):
        from llm_rosetta.pipeline import ConversionPipeline

        out = ConversionPipeline("openai_responses", "openai_chat").convert_request(
            {
                "model": "gpt-5.6-sol",
                "tools": [
                    {"type": "function", "name": "exec", "description": "top"},
                ],
                "input": [
                    {
                        "type": "additional_tools",
                        "tools": [
                            {
                                "type": "namespace",
                                "name": "functions",
                                "tools": [
                                    {
                                        "type": "function",
                                        "name": "exec",
                                        "description": "nested",
                                    }
                                ],
                            }
                        ],
                    },
                ],
            }
        )
        names = [t["function"]["name"] for t in out["tools"]]
        assert len(names) == len(set(names)), f"duplicate tool names: {names}"

    def test_chat_shaped_top_level_names_are_detected(self):
        assert provider_tool_names(
            [{"type": "function", "function": {"name": "exec"}}]
        ) == {"exec"}


class TestNamespaceMarkerNeverLeaks:
    """``_namespace`` is an internal transport detail, not wire content."""

    def test_marker_absent_from_converted_output(self):
        from llm_rosetta.pipeline import ConversionPipeline

        out = ConversionPipeline("openai_responses", "openai_chat").convert_request(
            _codex_request()
        )
        assert NAMESPACE_MARKER_KEY not in json.dumps(out)

    def test_marker_absent_from_responses_roundtrip(self):
        """The passthrough path copies the provider dict — it must filter."""
        from llm_rosetta.pipeline import ConversionPipeline

        out = ConversionPipeline(
            "openai_responses", "openai_responses"
        ).convert_request(_codex_request())
        assert NAMESPACE_MARKER_KEY not in json.dumps(out)
