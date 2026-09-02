"""Tests for ``additional_tools`` input items and nested ``namespace`` tools.

Codex transports its tool definitions inside the Responses ``input`` array
as an ``additional_tools`` item wrapping ``type: "namespace"`` containers,
rather than in the top-level ``tools`` field.  These tests pin the ingest
behaviour: tools are extracted from the items, namespaces are flattened
by the converter pipeline, the spent item is dropped, and the resulting
request survives conversion to Chat/Anthropic.
"""

from __future__ import annotations

from typing import cast

from llm_rosetta.converters.openai_responses import OpenAIResponsesConverter
from llm_rosetta.converters.openai_responses.tool_ops import (
    harvest_additional_tools,
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


class TestHarvestAdditionalTools:
    """Tests for the harvest/strip pipeline."""

    def test_extracts_raw_tool_dicts(self):
        items = _codex_request()["input"]
        warnings: list[str] = []
        tools, remaining = harvest_additional_tools(items, warnings)
        # Should extract the two namespace containers as raw dicts
        assert len(tools) == 2
        assert tools[0]["type"] == "namespace"
        assert tools[1]["type"] == "namespace"

    def test_strips_additional_tools_items(self):
        items = _codex_request()["input"]
        remaining = strip_additional_tools_items(items)
        assert len(remaining) == 1
        assert remaining[0]["type"] == "message"

    def test_ignores_non_additional_tools_items(self):
        items = [{"type": "message", "role": "user", "content": []}]
        warnings: list[str] = []
        tools, remaining = harvest_additional_tools(items, warnings)
        assert tools == []
        assert remaining is items
        assert warnings == []

    def test_non_list_input_returns_empty(self):
        warnings: list[str] = []
        tools, remaining = harvest_additional_tools("just a string", warnings)
        assert tools == []
        assert remaining == "just a string"

    def test_non_dict_entries_skipped_with_warning(self):
        items = [
            {
                "type": "additional_tools",
                "tools": ["not-a-dict", 42, {"type": "function", "name": "ok"}],
            }
        ]
        warnings: list[str] = []
        tools, _ = harvest_additional_tools(items, warnings)
        assert len(tools) == 1
        assert tools[0]["name"] == "ok"
        assert len(warnings) == 2

    def test_bare_tool_entry_without_namespace(self):
        items = [
            {
                "type": "additional_tools",
                "tools": [{"type": "function", "name": "ping", "parameters": {}}],
            }
        ]
        warnings: list[str] = []
        tools, _ = harvest_additional_tools(items, warnings)
        assert len(tools) == 1
        assert tools[0]["name"] == "ping"
        assert tools[0]["type"] == "function"


class TestRequestFromProviderAdditionalTools:
    """Integration tests: additional_tools through the full converter."""

    def setup_method(self):
        self.converter = OpenAIResponsesConverter()

    def test_nested_tools_reach_ir(self):
        result = self.converter.request_from_provider(_codex_request())
        tools = list(result["tools"])
        assert [t["name"] for t in tools] == ["exec", "wait", "spawn_agent"]
        assert tools[0]["type"] == "custom"
        assert tools[0]["metadata"]["namespace"] == "functions"
        assert tools[0]["metadata"]["format"]["type"] == "grammar"
        assert tools[1]["type"] == "function"

    def test_tool_choice_survives(self):
        result = self.converter.request_from_provider(_codex_request())
        assert result.get("tool_choice") is not None
        assert result.get("tool_config") is not None

    def test_additional_tools_item_does_not_become_a_message(self):
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

    def test_empty_additional_tools_item_produces_no_tools(self):
        request = _codex_request()
        request["input"][0] = {"type": "additional_tools", "tools": []}
        result = self.converter.request_from_provider(request)
        assert not result.get("tools")


class TestAdditionalToolsCrossFormat:
    """Cross-format conversion tests."""

    def test_roundtrip_to_openai_chat(self):
        from llm_rosetta.pipeline import ConversionPipeline

        out = ConversionPipeline("openai_responses", "openai_chat").convert_request(
            _codex_request()
        )
        names = [t["function"]["name"] for t in out["tools"]]
        assert names == ["exec", "wait", "spawn_agent"]
        assert out["tool_choice"] == "auto"
        # No stranded empty assistant messages
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


class TestAdditionalToolsCollision:
    """Name collisions between top-level and additional_tools."""

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


class TestAdditionalToolsDegenerate:
    """Edge cases with malformed additional_tools items."""

    def test_all_entries_malformed_still_strips_item(self):
        items = [
            {
                "type": "additional_tools",
                "tools": [{"type": "namespace", "name": "functions", "tools": [{}]}],
            },
            {"type": "message", "role": "user", "content": "hi"},
        ]
        warnings: list[str] = []
        tools, remaining = harvest_additional_tools(items, warnings)
        # Namespace with unnamed child is still extracted as raw dict
        assert len(tools) == 1
        assert not [
            i
            for i in remaining
            if isinstance(i, dict) and i.get("type") == "additional_tools"
        ]

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


class TestNamespaceMetadataPreserved:
    """Namespace metadata survives the full pipeline."""

    def test_namespace_in_metadata_not_in_output(self):
        from llm_rosetta.pipeline import ConversionPipeline

        out = ConversionPipeline("openai_responses", "openai_chat").convert_request(
            _codex_request()
        )
        # _namespace marker key should never appear in output
        import json

        assert "_namespace" not in json.dumps(out)
