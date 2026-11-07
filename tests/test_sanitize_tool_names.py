"""Tests for `sanitize_tool_names`, the target's tool-name rule.

An MCP server exports its tools as ``srv/fs__read``.  That is legal in a URL
and in most JSON, but not in the ``^[a-zA-Z0-9_-]+$`` every OpenAI- and
Anthropic-shaped endpoint publishes, which rejects the whole request over it
(#1426).  The name is rewritten against the target, and the client's spelling
is restored from the map on the way back.
"""

import re
from typing import Any

import pytest

from llm_rosetta.capabilities import (
    STRICT_TOOL_NAME_PATTERN,
    ToolNameMap,
    build_tool_name_map,
    sanitize_tool_names,
)
from llm_rosetta.pipeline import ConversionPipeline

ILLEGAL = "mcp_toolregistry-server-local_fs/file_search-glob"
LEGAL = "mcp_toolregistry-server-local_fs_file_search-glob"


def _ir_tool(name: str, **extra: Any) -> dict[str, Any]:
    """A minimal IR tool definition."""
    return {
        "type": "function",
        "name": name,
        "description": "",
        "parameters": {},
        **extra,
    }


def _sanitize(
    tools: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], ToolNameMap, list[str]]:
    """Run the pass over a bare tool list, as the pipeline does."""
    request: dict[str, Any] = {"tools": tools}
    name_map = build_tool_name_map(request)
    warnings: list[str] = []
    sanitize_tool_names(
        request,
        pattern=STRICT_TOOL_NAME_PATTERN,
        name_map=name_map,
        warnings=warnings,
    )
    return request["tools"], name_map, warnings


class TestRename:
    """A name the target rejects is rewritten, not dropped."""

    def test_a_slash_becomes_an_underscore(self):
        tools, _, _ = _sanitize([_ir_tool(ILLEGAL)])
        assert tools[0]["name"] == LEGAL
        assert STRICT_TOOL_NAME_PATTERN.fullmatch(tools[0]["name"])

    def test_the_client_spelling_is_restored_from_the_map(self):
        _, name_map, _ = _sanitize([_ir_tool(ILLEGAL)])
        assert name_map.to_client(LEGAL) == (ILLEGAL, None)

    def test_the_client_spelling_resolves_to_the_upstream_one(self):
        """The history rewrite looks the name up in the same direction."""
        _, name_map, _ = _sanitize([_ir_tool(ILLEGAL)])
        assert name_map.to_upstream(ILLEGAL) == LEGAL

    def test_the_client_spelling_is_kept_on_the_tool_for_later_readers(self):
        """A name pass further down names the tool the way the client wrote it."""
        tools, _, _ = _sanitize([_ir_tool(ILLEGAL)])
        assert tools[0]["metadata"]["_original_name"] == ILLEGAL

    def test_a_rename_is_reported(self):
        _, _, warnings = _sanitize([_ir_tool(ILLEGAL)])
        assert len(warnings) == 1
        assert ILLEGAL in warnings[0]
        assert LEGAL in warnings[0]

    def test_a_legal_name_is_never_rewritten(self):
        tools, name_map, warnings = _sanitize([_ir_tool("read_file")])
        assert tools[0]["name"] == "read_file"
        assert warnings == []
        assert name_map.to_client("read_file") == ("read_file", None)

    def test_a_name_at_the_length_limit_is_legal(self):
        name = "a" * 64
        tools, _, warnings = _sanitize([_ir_tool(name)])
        assert tools[0]["name"] == name
        assert warnings == []

    def test_a_name_over_the_limit_is_shortened(self):
        tools, name_map, _ = _sanitize([_ir_tool("a" * 80)])
        renamed = tools[0]["name"]
        assert len(renamed) <= 64
        assert STRICT_TOOL_NAME_PATTERN.fullmatch(renamed)
        assert name_map.to_client(renamed) == ("a" * 80, None)

    def test_other_shapes_of_illegal_character_are_replaced(self):
        tools, _, _ = _sanitize([_ir_tool("tool.with:dots and spaces")])
        assert STRICT_TOOL_NAME_PATTERN.fullmatch(tools[0]["name"])
        assert "." not in tools[0]["name"]


class TestCollisions:
    """Two tools must not end up sharing one upstream name."""

    def test_a_collision_with_a_legal_name_is_disambiguated(self):
        tools, name_map, _ = _sanitize([_ir_tool("srv/read"), _ir_tool("srv_read")])
        names = [tool["name"] for tool in tools]
        assert names[1] == "srv_read"
        assert names[0] != "srv_read"
        assert len(set(names)) == 2
        # Each still restores its own client name.
        assert name_map.to_client(names[0]) == ("srv/read", None)
        assert name_map.to_client(names[1]) == ("srv_read", None)

    def test_two_illegal_names_that_sanitize_alike_stay_distinct(self):
        tools, name_map, _ = _sanitize([_ir_tool("a/b"), _ir_tool("a.b")])
        names = [tool["name"] for tool in tools]
        assert len(set(names)) == 2
        assert {name_map.to_client(name)[0] for name in names} == {"a/b", "a.b"}

    def test_the_rename_is_stable_across_identical_requests(self):
        first, _, _ = _sanitize([_ir_tool("srv/read"), _ir_tool("srv_read")])
        second, _, _ = _sanitize([_ir_tool("srv/read"), _ir_tool("srv_read")])
        assert [tool["name"] for tool in first] == [tool["name"] for tool in second]


class TestNoTools:
    """The pass is a no-op where there is nothing to rewrite."""

    def test_no_tools(self):
        request: dict[str, Any] = {}
        name_map = ToolNameMap()
        assert (
            sanitize_tool_names(
                request,
                pattern=STRICT_TOOL_NAME_PATTERN,
                name_map=name_map,
                warnings=[],
            )
            is False
        )

    def test_a_custom_pattern_is_honoured(self):
        """The target declares what it accepts; the pass asks no questions."""
        request: dict[str, Any] = {"tools": [_ir_tool("a.b")]}
        name_map = ToolNameMap()
        sanitize_tool_names(
            request,
            pattern=re.compile(r"[a-z.]+"),
            name_map=name_map,
            warnings=[],
        )
        assert request["tools"][0]["name"] == "a.b"


class TestPipeline:
    """End to end: Google GenAI in, Chat Completions out, the client's answer."""

    @staticmethod
    def _genai_body() -> dict[str, Any]:
        return {
            "contents": [{"role": "user", "parts": [{"text": "search"}]}],
            "tools": [
                {
                    "function_declarations": [
                        {
                            "name": ILLEGAL,
                            "parameters": {
                                "type": "OBJECT",
                                "properties": {
                                    "pattern": {"type": "STRING", "minLength": "1"},
                                    "head_limit": {
                                        "type": "INTEGER",
                                        "minimum": "-1",
                                    },
                                },
                                "required": ["pattern"],
                            },
                        }
                    ]
                }
            ],
        }

    def test_the_outbound_name_is_legal_and_the_numbers_are_numbers(self):
        pipeline = ConversionPipeline("google_generate", "chat_completions")
        out = pipeline.convert_request(self._genai_body())
        function = out["tools"][0]["function"]
        assert function["name"] == LEGAL
        assert function["parameters"]["properties"]["pattern"]["minLength"] == 1
        assert function["parameters"]["properties"]["head_limit"]["minimum"] == -1

    def test_the_request_reports_what_it_renamed_and_could_not_read(self):
        pipeline = ConversionPipeline("google_generate", "chat_completions")
        pipeline.convert_request(self._genai_body())
        # The stringified values were both readable, so only the rename warns.
        assert len(pipeline.warnings) == 1
        assert ILLEGAL in pipeline.warnings[0]

    def test_an_unreadable_constraint_warns_with_the_tool_and_the_path(self):
        body = self._genai_body()
        declaration = body["tools"][0]["function_declarations"][0]
        declaration["parameters"]["properties"]["pattern"]["minLength"] = "abc"
        pipeline = ConversionPipeline("google_generate", "chat_completions")
        out = pipeline.convert_request(body)
        # Forwarded as it came — the source's intent is not known.
        parameters = out["tools"][0]["function"]["parameters"]
        assert parameters["properties"]["pattern"]["minLength"] == "abc"
        messages = [str(w) for w in pipeline.warnings]
        assert any(
            "tools[0].parameters.properties.pattern.minLength" in m for m in messages
        )
        assert any(ILLEGAL in m for m in messages)

    def test_the_response_restores_the_client_name(self):
        pipeline = ConversionPipeline("google_generate", "chat_completions")
        pipeline.convert_request(self._genai_body())
        upstream = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "created": 1,
            "model": "gpt-5-nano",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "function": {
                                    "name": LEGAL,
                                    "arguments": '{"pattern": "*.py"}',
                                },
                            }
                        ],
                    },
                    "finish_reason": "tool_calls",
                }
            ],
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
            },
        }
        out = pipeline.convert_response(upstream)
        call = out["candidates"][0]["content"]["parts"][0]["functionCall"]
        assert call["name"] == ILLEGAL

    def test_a_history_call_is_rewritten_to_the_upstream_name(self):
        body = self._genai_body()
        body["contents"].extend(
            [
                {
                    "role": "model",
                    "parts": [{"functionCall": {"name": ILLEGAL, "args": {}}}],
                },
                {
                    "role": "user",
                    "parts": [
                        {
                            "functionResponse": {
                                "name": ILLEGAL,
                                "response": {"ok": True},
                            }
                        }
                    ],
                },
            ]
        )
        pipeline = ConversionPipeline("google_generate", "chat_completions")
        out = pipeline.convert_request(body)
        assistant = [m for m in out["messages"] if m["role"] == "assistant"][0]
        assert assistant["tool_calls"][0]["function"]["name"] == LEGAL

    def test_a_target_without_the_rule_keeps_the_name(self):
        """Google's own format accepts a slash; nothing is rewritten for it."""
        pipeline = ConversionPipeline("chat_completions", "google_generate")
        source = {
            "model": "gpt-5-nano",
            "messages": [{"role": "user", "content": "search"}],
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": ILLEGAL,
                        "parameters": {"type": "object", "properties": {}},
                    },
                }
            ],
        }
        out = pipeline.convert_request(source)
        declaration = out["tools"][0]["function_declarations"][0]
        assert declaration["name"] == ILLEGAL

    @pytest.mark.parametrize(
        "target", ["chat_completions", "messages", "open_responses"]
    )
    def test_every_target_publishing_the_rule_gets_it(self, target):
        pipeline = ConversionPipeline("google_generate", target)
        out = pipeline.convert_request(self._genai_body())
        text = str(out)
        assert ILLEGAL not in text
        assert LEGAL in text
