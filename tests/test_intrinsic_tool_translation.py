"""Tests for translate_intrinsic_tools and intrinsic-to-function helpers."""

import json

import pytest

from llm_rosetta.capabilities import translate_intrinsic_tools
from llm_rosetta.converters.base.tools.intrinsic import (
    intrinsic_call_to_function,
    intrinsic_result_to_function,
)


# ── Helper unit tests ──────────────────────────────────────────────────


class TestIntrinsicCallToFunction:
    def test_basic_translation(self):
        part = {
            "type": "tool_call",
            "tool_call_id": "c1",
            "tool_name": "ws",
            "tool_input": {"query": "AI"},
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "web_search"},
        }
        result = intrinsic_call_to_function(part)
        assert result["type"] == "tool_call"
        assert result["tool_type"] == "function"
        assert result["tool_name"] == "_intrinsic_web_search"
        assert result["tool_call_id"] == "c1"
        assert result["tool_input"] == {"query": "AI"}

    def test_metadata(self):
        part = {
            "type": "tool_call",
            "tool_call_id": "c1",
            "tool_name": "ws",
            "tool_input": {},
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "code_execution"},
        }
        result = intrinsic_call_to_function(part)
        pm = result["provider_metadata"]
        assert pm["_translated_from"] == "intrinsic"
        assert pm["_original_intrinsic_kind"] == "code_execution"
        assert "intrinsic_kind" not in pm

    def test_preserves_cache_hint(self):
        part = {
            "type": "tool_call",
            "tool_call_id": "c1",
            "tool_name": "ws",
            "tool_input": {},
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "web_search"},
            "cache_hint": "ephemeral",
        }
        result = intrinsic_call_to_function(part)
        assert result["cache_hint"] == "ephemeral"

    def test_no_cache_hint(self):
        part = {
            "type": "tool_call",
            "tool_call_id": "c1",
            "tool_name": "ws",
            "tool_input": {},
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "web_search"},
        }
        result = intrinsic_call_to_function(part)
        assert "cache_hint" not in result

    def test_fallback_kind_from_tool_name(self):
        part = {
            "type": "tool_call",
            "tool_call_id": "c1",
            "tool_name": "my_search",
            "tool_input": {},
            "tool_type": "intrinsic",
            "provider_metadata": {},
        }
        result = intrinsic_call_to_function(part)
        assert result["tool_name"] == "_intrinsic_my_search"

    def test_does_not_mutate_original(self):
        part = {
            "type": "tool_call",
            "tool_call_id": "c1",
            "tool_name": "ws",
            "tool_input": {"query": "AI"},
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "web_search"},
        }
        original_pm = dict(part["provider_metadata"])
        intrinsic_call_to_function(part)
        assert part["tool_type"] == "intrinsic"
        assert part["provider_metadata"] == original_pm


class TestIntrinsicResultToFunction:
    def test_string_result(self):
        part = {
            "type": "tool_result",
            "tool_call_id": "c1",
            "result": "search results here",
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "web_search"},
        }
        result = intrinsic_result_to_function(part)
        assert result["type"] == "tool_result"
        assert result["tool_type"] == "function"
        assert result["tool_call_id"] == "c1"
        assert result["result"] == "search results here"

    def test_list_result_text_blocks(self):
        part = {
            "type": "tool_result",
            "tool_call_id": "c1",
            "result": [
                {"type": "text", "text": "Result 1"},
                {"type": "text", "text": "Result 2"},
            ],
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "web_search"},
        }
        result = intrinsic_result_to_function(part)
        assert result["result"] == "Result 1\nResult 2"

    def test_list_result_mixed_blocks(self):
        part = {
            "type": "tool_result",
            "tool_call_id": "c1",
            "result": [
                {"type": "text", "text": "Found results"},
                {"type": "image", "source": "data:..."},
            ],
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "web_search"},
        }
        result = intrinsic_result_to_function(part)
        lines = result["result"].split("\n")
        assert lines[0] == "Found results"
        assert json.loads(lines[1]) == {"type": "image", "source": "data:..."}

    def test_dict_result(self):
        data = {"output": "42", "outcome": "SUCCESS"}
        part = {
            "type": "tool_result",
            "tool_call_id": "c1",
            "result": data,
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "code_execution"},
        }
        result = intrinsic_result_to_function(part)
        assert json.loads(result["result"]) == data

    def test_preserves_is_error(self):
        part = {
            "type": "tool_result",
            "tool_call_id": "c1",
            "result": "error msg",
            "tool_type": "intrinsic",
            "is_error": True,
            "provider_metadata": {"intrinsic_kind": "code_execution"},
        }
        result = intrinsic_result_to_function(part)
        assert result["is_error"] is True

    def test_no_is_error_when_false(self):
        part = {
            "type": "tool_result",
            "tool_call_id": "c1",
            "result": "ok",
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "web_search"},
        }
        result = intrinsic_result_to_function(part)
        assert "is_error" not in result

    def test_preserves_cache_hint(self):
        part = {
            "type": "tool_result",
            "tool_call_id": "c1",
            "result": "ok",
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "web_search"},
            "cache_hint": "ephemeral",
        }
        result = intrinsic_result_to_function(part)
        assert result["cache_hint"] == "ephemeral"

    def test_metadata(self):
        part = {
            "type": "tool_result",
            "tool_call_id": "c1",
            "result": "ok",
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "file_search"},
        }
        result = intrinsic_result_to_function(part)
        pm = result["provider_metadata"]
        assert pm["_translated_from"] == "intrinsic"
        assert pm["_original_intrinsic_kind"] == "file_search"


# ── translate_intrinsic_tools tests ────────────────────────────────────


class TestTranslateIntrinsicTools:
    def test_same_format_noop(self):
        ir_request = {
            "tools": [
                {"type": "intrinsic", "name": "ws", "description": "", "parameters": {}}
            ],
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "c1",
                            "tool_name": "ws",
                            "tool_input": {},
                            "tool_type": "intrinsic",
                        }
                    ],
                }
            ],
        }
        result = translate_intrinsic_tools(ir_request, same_format=True)
        assert result is ir_request

    def test_no_intrinsic_noop(self):
        ir_request = {
            "tools": [
                {"type": "function", "name": "f", "description": "", "parameters": {}}
            ],
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "c1",
                            "tool_name": "f",
                            "tool_input": {},
                            "tool_type": "function",
                        }
                    ],
                }
            ],
        }
        result = translate_intrinsic_tools(ir_request, same_format=False)
        assert result is ir_request

    def test_strip_intrinsic_definitions(self):
        ir_request = {
            "tools": [
                {"type": "function", "name": "f", "description": "", "parameters": {}},
                {
                    "type": "intrinsic",
                    "name": "ws",
                    "description": "",
                    "parameters": {},
                },
            ],
            "messages": [],
        }
        result = translate_intrinsic_tools(ir_request, same_format=False)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "f"

    def test_translate_tool_call(self):
        ir_request = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Searching..."},
                        {
                            "type": "tool_call",
                            "tool_call_id": "c1",
                            "tool_name": "ws",
                            "tool_input": {"query": "AI"},
                            "tool_type": "intrinsic",
                            "provider_metadata": {"intrinsic_kind": "web_search"},
                        },
                    ],
                }
            ],
        }
        result = translate_intrinsic_tools(ir_request, same_format=False)
        msg = result["messages"][0]
        assert len(msg["content"]) == 2
        assert msg["content"][0]["type"] == "text"
        call = msg["content"][1]
        assert call["tool_type"] == "function"
        assert call["tool_name"] == "_intrinsic_web_search"
        assert call["tool_call_id"] == "c1"
        assert call["tool_input"] == {"query": "AI"}

    def test_translate_tool_result(self):
        ir_request = {
            "messages": [
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": "c1",
                            "result": "search results",
                            "tool_type": "intrinsic",
                            "provider_metadata": {"intrinsic_kind": "web_search"},
                        }
                    ],
                }
            ],
        }
        result = translate_intrinsic_tools(ir_request, same_format=False)
        assert len(result["messages"]) == 1
        tr = result["messages"][0]["content"][0]
        assert tr["tool_type"] == "function"
        assert tr["result"] == "search results"
        assert tr["tool_call_id"] == "c1"

    def test_tool_role_message_preserved(self):
        """Tool-role messages are kept (not dropped) after translation."""
        ir_request = {
            "messages": [
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": "c1",
                            "result": "data",
                            "tool_type": "intrinsic",
                            "provider_metadata": {"intrinsic_kind": "web_search"},
                        }
                    ],
                }
            ],
        }
        result = translate_intrinsic_tools(ir_request, same_format=False)
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "tool"

    def test_mixed_intrinsic_and_function_parts(self):
        ir_request = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "c1",
                            "tool_name": "f",
                            "tool_input": {},
                            "tool_type": "function",
                        },
                        {
                            "type": "tool_call",
                            "tool_call_id": "c2",
                            "tool_name": "ws",
                            "tool_input": {},
                            "tool_type": "intrinsic",
                            "provider_metadata": {"intrinsic_kind": "web_search"},
                        },
                    ],
                }
            ],
        }
        result = translate_intrinsic_tools(ir_request, same_format=False)
        content = result["messages"][0]["content"]
        assert len(content) == 2
        assert content[0]["tool_type"] == "function"
        assert content[0]["tool_call_id"] == "c1"
        assert content[1]["tool_type"] == "function"
        assert content[1]["tool_name"] == "_intrinsic_web_search"

    def test_call_result_id_pairing(self):
        ir_request = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "c42",
                            "tool_name": "ws",
                            "tool_input": {"query": "test"},
                            "tool_type": "intrinsic",
                            "provider_metadata": {"intrinsic_kind": "web_search"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": "c42",
                            "result": "found stuff",
                            "tool_type": "intrinsic",
                            "provider_metadata": {"intrinsic_kind": "web_search"},
                        }
                    ],
                },
            ],
        }
        result = translate_intrinsic_tools(ir_request, same_format=False)
        call_id = result["messages"][0]["content"][0]["tool_call_id"]
        result_id = result["messages"][1]["content"][0]["tool_call_id"]
        assert call_id == result_id == "c42"

    def test_does_not_mutate_original(self):
        ir_request = {
            "tools": [
                {"type": "intrinsic", "name": "ws", "description": "", "parameters": {}}
            ],
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "c1",
                            "tool_name": "ws",
                            "tool_input": {},
                            "tool_type": "intrinsic",
                            "provider_metadata": {"intrinsic_kind": "web_search"},
                        }
                    ],
                }
            ],
        }
        translate_intrinsic_tools(ir_request, same_format=False)
        assert len(ir_request["tools"]) == 1
        assert ir_request["messages"][0]["content"][0]["tool_type"] == "intrinsic"

    def test_strip_defs_and_translate_parts(self):
        ir_request = {
            "tools": [
                {"type": "function", "name": "f", "description": "", "parameters": {}},
                {
                    "type": "intrinsic",
                    "name": "ws",
                    "description": "",
                    "parameters": {},
                },
            ],
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "c1",
                            "tool_name": "ws",
                            "tool_input": {},
                            "tool_type": "intrinsic",
                            "provider_metadata": {"intrinsic_kind": "web_search"},
                        }
                    ],
                },
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": "c1",
                            "result": "data",
                            "tool_type": "intrinsic",
                            "provider_metadata": {"intrinsic_kind": "web_search"},
                        }
                    ],
                },
            ],
        }
        result = translate_intrinsic_tools(ir_request, same_format=False)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "f"
        assert len(result["messages"]) == 2
        assert result["messages"][0]["content"][0]["tool_type"] == "function"
        assert result["messages"][1]["content"][0]["tool_type"] == "function"

    @pytest.mark.parametrize(
        "kind",
        [
            "web_search",
            "code_execution",
            "code_interpreter",
            "file_search",
            "google_search",
            "shell",
            "computer_use",
            "url_context",
            "google_maps",
            "retrieval",
        ],
    )
    def test_various_intrinsic_kinds(self, kind):
        ir_request = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "c1",
                            "tool_name": kind,
                            "tool_input": {},
                            "tool_type": "intrinsic",
                            "provider_metadata": {"intrinsic_kind": kind},
                        }
                    ],
                }
            ],
        }
        result = translate_intrinsic_tools(ir_request, same_format=False)
        call = result["messages"][0]["content"][0]
        assert call["tool_name"] == f"_intrinsic_{kind}"
        assert call["tool_type"] == "function"

    def test_non_list_content_passthrough(self):
        ir_request = {
            "messages": [
                {"role": "user", "content": "plain string content"},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "c1",
                            "tool_name": "ws",
                            "tool_input": {},
                            "tool_type": "intrinsic",
                            "provider_metadata": {"intrinsic_kind": "web_search"},
                        }
                    ],
                },
            ],
        }
        result = translate_intrinsic_tools(ir_request, same_format=False)
        assert result["messages"][0]["content"] == "plain string content"
        assert result["messages"][1]["content"][0]["tool_type"] == "function"
