"""Tests for Google Interactions tool_ops."""

import pytest

from llm_rosetta.converters.google_interactions.tool_ops import (
    GoogleInteractionsToolOps,
)
from llm_rosetta.types.ir.parts import ToolCallPart, ToolResultPart


class TestToolDefinitionConversion:
    def test_ir_function_to_provider(self):
        ir = {
            "type": "function",
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
        result = GoogleInteractionsToolOps.ir_tool_to_p(ir)  # ty: ignore
        assert result["type"] == "function"
        assert result["name"] == "get_weather"
        assert result["description"] == "Get current weather"
        assert result["parameters"]["properties"]["location"]["type"] == "string"

    def test_provider_function_to_ir(self):
        p = {
            "type": "function",
            "name": "get_weather",
            "description": "Get current weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
            },
        }
        result = GoogleInteractionsToolOps.p_tool_to_ir(p)
        assert result["type"] == "function"
        assert result["name"] == "get_weather"
        assert result["description"] == "Get current weather"

    def test_provider_mcp_server_to_ir(self):
        p = {
            "type": "mcp_server",
            "name": "weather_service",
            "url": "https://example.com/mcp",
        }
        result = GoogleInteractionsToolOps.p_tool_to_ir(p)
        assert result["type"] == "mcp"
        assert result["name"] == "weather_service"
        assert result["metadata"]["url"] == "https://example.com/mcp"

    def test_roundtrip_function_tool(self):
        ir = {
            "type": "function",
            "name": "calc",
            "description": "Calculate",
            "parameters": {"type": "object", "properties": {}},
        }
        p = GoogleInteractionsToolOps.ir_tool_to_p(ir)  # ty: ignore
        back = GoogleInteractionsToolOps.p_tool_to_ir(p)
        assert back["name"] == ir["name"]
        assert back["description"] == ir["description"]


class TestFunctionCallConversion:
    def test_provider_function_call_to_ir(self):
        step = {
            "type": "function_call",
            "id": "call_123",
            "name": "get_weather",
            "arguments": {"location": "San Francisco"},
        }
        result = GoogleInteractionsToolOps.p_function_call_to_ir(step)
        assert result["type"] == "tool_call"
        assert result["tool_call_id"] == "call_123"
        assert result["tool_name"] == "get_weather"
        assert result["tool_input"] == {"location": "San Francisco"}

    def test_provider_function_call_to_ir_with_signature(self):
        step = {
            "type": "function_call",
            "id": "call_sig",
            "name": "get_weather",
            "arguments": {"location": "SF"},
            "signature": "eyJhbGciOiAiUlMyNTYifQ.test-sig",
        }
        result = GoogleInteractionsToolOps.p_function_call_to_ir(step)
        assert result["type"] == "tool_call"
        assert result["tool_call_id"] == "call_sig"
        assert (
            result["provider_metadata"]["google"]["thought_signature"]
            == "eyJhbGciOiAiUlMyNTYifQ.test-sig"
        )

    def test_provider_function_call_to_ir_no_signature(self):
        step = {
            "type": "function_call",
            "id": "call_no_sig",
            "name": "fn",
            "arguments": {},
        }
        result = GoogleInteractionsToolOps.p_function_call_to_ir(step)
        assert "provider_metadata" not in result

    def test_ir_tool_call_to_provider(self):
        ir = {
            "type": "tool_call",
            "tool_call_id": "call_456",
            "tool_name": "search",
            "tool_input": {"query": "hello"},
        }
        result = GoogleInteractionsToolOps.ir_tool_call_to_p(ir)  # ty: ignore
        assert result["type"] == "function_call"
        assert result["id"] == "call_456"
        assert result["name"] == "search"
        assert result["arguments"] == {"query": "hello"}

    def test_ir_tool_call_to_provider_with_signature(self):
        ir = {
            "type": "tool_call",
            "tool_call_id": "call_sig",
            "tool_name": "fn",
            "tool_input": {},
            "provider_metadata": {"google": {"thought_signature": "sig_from_ir"}},
        }
        result = GoogleInteractionsToolOps.ir_tool_call_to_p(ir)  # ty: ignore
        assert result["type"] == "function_call"
        assert result["signature"] == "sig_from_ir"

    def test_ir_tool_call_to_provider_no_signature(self):
        ir = {
            "type": "tool_call",
            "tool_call_id": "call_no",
            "tool_name": "fn",
            "tool_input": {},
        }
        result = GoogleInteractionsToolOps.ir_tool_call_to_p(ir)  # ty: ignore
        assert "signature" not in result

    def test_ir_tool_call_to_provider_none_provider_metadata(self):
        """An explicit ``provider_metadata: None`` must not crash."""
        ir = {
            "type": "tool_call",
            "tool_call_id": "call_none",
            "tool_name": "fn",
            "tool_input": {},
            "provider_metadata": None,
        }
        result = GoogleInteractionsToolOps.ir_tool_call_to_p(ir)  # ty: ignore
        assert result["type"] == "function_call"
        assert "signature" not in result

    def test_ir_tool_call_to_provider_preserve_metadata_false(self):
        """`preserve_metadata=False` suppresses the signature (parity with GG)."""
        ir = ToolCallPart(
            type="tool_call",
            tool_call_id="call_sig",
            tool_name="fn",
            tool_input={},
            provider_metadata={"google": {"thought_signature": "sig_keep"}},
        )
        result = GoogleInteractionsToolOps.ir_tool_call_to_p(
            ir, preserve_metadata=False
        )
        assert result["type"] == "function_call"
        assert "signature" not in result

    def test_p_function_call_to_ir_preserve_metadata_false(self):
        """`preserve_metadata=False` does not capture the signature."""
        step = {
            "type": "function_call",
            "id": "call_sig",
            "name": "fn",
            "arguments": {},
            "signature": "sig_drop",
        }
        result = GoogleInteractionsToolOps.p_function_call_to_ir(
            step, preserve_metadata=False
        )
        assert "provider_metadata" not in result

    def test_intrinsic_call_restores_signature_and_drops_name(self):
        """A server call carries the signature, never a ``name`` (#992).

        The API rejects ``name`` on a ``google_search_call`` with ``400 Unknown
        parameter 'name'`` and needs the thought signature back on replay.
        """
        ir = {
            "type": "tool_call",
            "tool_call_id": "call_intr",
            "tool_name": "google_search",
            "tool_input": {},
            "tool_type": "intrinsic",
            "provider_metadata": {
                "intrinsic_kind": "google_search",
                "google": {"thought_signature": "sig_kept"},
            },
        }
        result = GoogleInteractionsToolOps.ir_tool_call_to_p(ir)  # ty: ignore
        assert result["type"] == "google_search_call"
        assert "name" not in result
        assert result["signature"] == "sig_kept"

    def test_intrinsic_call_without_signature_emits_none(self):
        """No stored signature — the step simply carries no ``signature``."""
        ir = {
            "type": "tool_call",
            "tool_call_id": "call_intr",
            "tool_name": "google_search",
            "tool_input": {},
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "google_search"},
        }
        result = GoogleInteractionsToolOps.ir_tool_call_to_p(ir)  # ty: ignore
        assert result["type"] == "google_search_call"
        assert "signature" not in result

    def test_roundtrip_function_call(self):
        step = {
            "type": "function_call",
            "id": "call_rt",
            "name": "fn",
            "arguments": {"x": 1},
        }
        ir = GoogleInteractionsToolOps.p_function_call_to_ir(step)
        back = GoogleInteractionsToolOps.ir_tool_call_to_p(ir)
        assert back["id"] == step["id"]
        assert back["name"] == step["name"]
        assert back["arguments"] == step["arguments"]

    def test_roundtrip_function_call_with_signature(self):
        step = {
            "type": "function_call",
            "id": "call_rt_sig",
            "name": "fn",
            "arguments": {"x": 1},
            "signature": "sig_roundtrip",
        }
        ir = GoogleInteractionsToolOps.p_function_call_to_ir(step)
        back = GoogleInteractionsToolOps.ir_tool_call_to_p(ir)
        assert back["id"] == step["id"]
        assert back["name"] == step["name"]
        assert back["arguments"] == step["arguments"]
        assert back["signature"] == "sig_roundtrip"


class TestFunctionResultConversion:
    def test_provider_function_result_to_ir(self):
        step = {
            "type": "function_result",
            "call_id": "call_123",
            "result": [{"type": "text", "text": '{"weather":"sunny"}'}],
        }
        result = GoogleInteractionsToolOps.p_function_result_to_ir(step)
        assert result["type"] == "tool_result"
        assert result["tool_call_id"] == "call_123"
        assert "sunny" in result["result"]

    def test_provider_function_result_string(self):
        step = {
            "type": "function_result",
            "call_id": "call_456",
            "result": "success",
        }
        result = GoogleInteractionsToolOps.p_function_result_to_ir(step)
        assert result["result"] == "success"

    def test_provider_function_result_dict(self):
        step = {
            "type": "function_result",
            "call_id": "call_789",
            "result": {"status": "ok"},
        }
        result = GoogleInteractionsToolOps.p_function_result_to_ir(step)
        assert '"status"' in result["result"]

    def test_provider_function_result_error(self):
        step = {
            "type": "function_result",
            "call_id": "call_err",
            "result": "error occurred",
            "is_error": True,
        }
        result = GoogleInteractionsToolOps.p_function_result_to_ir(step)
        assert result["is_error"] is True

    def test_ir_tool_result_to_provider(self):
        ir = {
            "type": "tool_result",
            "tool_call_id": "call_123",
            "result": "42",
        }
        result = GoogleInteractionsToolOps.ir_tool_result_to_p(ir)  # ty: ignore
        assert result["type"] == "function_result"
        assert result["call_id"] == "call_123"
        assert result["result"] == "42"

    def test_ir_tool_result_to_provider_resolves_name(self):
        """`name` is required by GI and resolved from tool_call_index (#934)."""
        ir = ToolResultPart(type="tool_result", tool_call_id="call_123", result="42")
        result = GoogleInteractionsToolOps.ir_tool_result_to_p(
            ir, tool_call_index={"call_123": "add"}
        )
        assert result["type"] == "function_result"
        assert result["name"] == "add"

    def test_ir_tool_result_to_provider_unresolved_falls_back_to_call_id(self):
        """No matching tool_call → name falls back to call_id (non-empty)."""
        ir = ToolResultPart(type="tool_result", tool_call_id="call_orphan", result="42")
        result = GoogleInteractionsToolOps.ir_tool_result_to_p(
            ir, tool_call_index={"other": "fn"}
        )
        assert result["name"] == "call_orphan"

    def test_empty_call_id_and_no_match_raises(self):
        """An unresolvable name (empty call_id) fails loudly, not silently."""
        ir = ToolResultPart(type="tool_result", tool_call_id="", result="3")
        with pytest.raises(ValueError, match="no resolvable function name"):
            GoogleInteractionsToolOps.ir_tool_result_to_p(
                ir, tool_call_index={"other": "fn"}
            )

    def test_p_function_result_to_ir_keeps_name(self):
        """The inbound function_result name is preserved on the IR part."""
        step = {
            "type": "function_result",
            "call_id": "c1",
            "name": "add",
            "result": "3",
        }
        ir = GoogleInteractionsToolOps.p_function_result_to_ir(step)
        assert ir["provider_metadata"]["google"]["tool_name"] == "add"

    def test_stored_name_used_when_no_index_match(self):
        """A result-only replay reuses the stored name, not the call_id."""
        ir = GoogleInteractionsToolOps.p_function_result_to_ir(
            {"type": "function_result", "call_id": "c1", "name": "add", "result": "3"}
        )
        result = GoogleInteractionsToolOps.ir_tool_result_to_p(  # ty: ignore
            ir, tool_call_index={}
        )
        assert result["name"] == "add"

    def test_stored_name_used_when_call_id_empty(self):
        """A known name avoids the ValueError even with an empty call_id."""
        ir = GoogleInteractionsToolOps.p_function_result_to_ir(
            {"type": "function_result", "call_id": "", "name": "add", "result": "3"}
        )
        result = GoogleInteractionsToolOps.ir_tool_result_to_p(  # ty: ignore
            ir, tool_call_index={}
        )
        assert result["name"] == "add"

    def test_index_wins_over_stored_name(self):
        """An explicit call in the batch is authoritative over a stale hint."""
        ir = ToolResultPart(
            type="tool_result",
            tool_call_id="c1",
            result="3",
            provider_metadata={"google": {"tool_name": "stale"}},
        )
        result = GoogleInteractionsToolOps.ir_tool_result_to_p(  # ty: ignore
            ir, tool_call_index={"c1": "add"}
        )
        assert result["name"] == "add"

    def test_intrinsic_result_has_no_name(self):
        """Intrinsic server results don't take a function name."""
        ir = ToolResultPart(
            type="tool_result",
            tool_call_id="c1",
            result="ok",
            tool_type="intrinsic",
            provider_metadata={"intrinsic_kind": "google_search"},
        )
        result = GoogleInteractionsToolOps.ir_tool_result_to_p(
            ir, tool_call_index={"c1": "add"}
        )
        assert result["type"] == "google_search_result"
        assert "name" not in result

    def test_roundtrip_function_result(self):
        ir = {
            "type": "tool_result",
            "tool_call_id": "call_rt",
            "result": "done",
        }
        p = GoogleInteractionsToolOps.ir_tool_result_to_p(ir)  # ty: ignore
        back = GoogleInteractionsToolOps.p_function_result_to_ir(p)
        assert back["tool_call_id"] == ir["tool_call_id"]
        assert back["result"] == ir["result"]


class TestToolChoiceConversion:
    def test_string_modes(self):
        for mode in ("auto", "any", "none"):
            ir = {"mode": mode}
            p = GoogleInteractionsToolOps.ir_tool_choice_to_p(ir)  # ty: ignore
            assert p == mode
            back = GoogleInteractionsToolOps.p_tool_choice_to_ir(p)
            assert back["mode"] == mode

    def test_specific_tool(self):
        ir = {"mode": "tool", "tool_name": "get_weather"}
        p = GoogleInteractionsToolOps.ir_tool_choice_to_p(ir)  # ty: ignore
        assert isinstance(p, dict)
        assert p["allowed_tools"]["tools"] == ["get_weather"]

    def test_provider_specific_tool_to_ir(self):
        p = {"allowed_tools": {"mode": "any", "tools": ["search"]}}
        result = GoogleInteractionsToolOps.p_tool_choice_to_ir(p)
        assert result["mode"] == "any"
        assert result["allowed_tools"] == [{"name": "search"}]

    def test_provider_specific_tool_to_ir_keeps_every_name(self):
        """The restriction lists every named tool, not just the first."""
        p = {"allowed_tools": {"mode": "any", "tools": ["search", "fetch"]}}
        result = GoogleInteractionsToolOps.p_tool_choice_to_ir(p)
        assert result["mode"] == "any"
        assert result["allowed_tools"] == [{"name": "search"}, {"name": "fetch"}]
