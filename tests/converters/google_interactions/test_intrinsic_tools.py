"""Tests for Google Interactions intrinsic server step type handling."""

from typing import Any, cast

from llm_rosetta.converters.google_interactions.tool_ops import (
    GoogleInteractionsToolOps,
)
from llm_rosetta.converters.google_interactions.message_ops import (
    GoogleInteractionsMessageOps,
)
from llm_rosetta.types.ir import ToolCallPart, ToolResultPart


class TestServerCallToIR:
    """Test server call step types → IR intrinsic."""

    def test_google_search_call(self):
        step = {
            "type": "google_search_call",
            "id": "gs_1",
            "name": "google_search",
            "arguments": {"query": "latest news"},
        }
        result = GoogleInteractionsToolOps.p_server_call_to_ir(step)
        assert result["type"] == "tool_call"
        assert result["tool_type"] == "intrinsic"
        assert result["provider_metadata"]["intrinsic_kind"] == "google_search"
        assert result["tool_call_id"] == "gs_1"

    def test_code_execution_call(self):
        step = {
            "type": "code_execution_call",
            "id": "ce_1",
            "name": "code_execution",
            "arguments": {"code": "print(1)"},
        }
        result = GoogleInteractionsToolOps.p_server_call_to_ir(step)
        assert result["tool_type"] == "intrinsic"
        assert result["provider_metadata"]["intrinsic_kind"] == "code_execution"

    def test_url_context_call(self):
        step = {
            "type": "url_context_call",
            "id": "uc_1",
            "name": "url_context",
            "arguments": {"url": "https://example.com"},
        }
        result = GoogleInteractionsToolOps.p_server_call_to_ir(step)
        assert result["tool_type"] == "intrinsic"
        assert result["provider_metadata"]["intrinsic_kind"] == "url_context"


class TestServerResultToIR:
    """Test server result step types → IR intrinsic."""

    def test_google_search_result(self):
        step = {
            "type": "google_search_result",
            "call_id": "gs_1",
            "result": "Search results here",
        }
        result = GoogleInteractionsToolOps.p_server_result_to_ir(step)
        assert result["type"] == "tool_result"
        assert result["tool_type"] == "intrinsic"
        assert result["provider_metadata"]["intrinsic_kind"] == "google_search"
        assert result["result"] == "Search results here"

    def test_code_execution_result_with_error(self):
        step = {
            "type": "code_execution_result",
            "call_id": "ce_1",
            "result": "NameError",
            "is_error": True,
        }
        result = GoogleInteractionsToolOps.p_server_result_to_ir(step)
        assert result["tool_type"] == "intrinsic"
        assert result["provider_metadata"]["intrinsic_kind"] == "code_execution"
        assert result["is_error"] is True


class TestIRIntrinsicToProvider:
    """Test IR intrinsic → server step types."""

    def test_ir_intrinsic_call_to_google_search_call(self):
        ir_part = cast(
            ToolCallPart,
            {
                "type": "tool_call",
                "tool_call_id": "gs_1",
                "tool_name": "google_search",
                "tool_input": {"query": "test"},
                "tool_type": "intrinsic",
                "provider_metadata": {"intrinsic_kind": "google_search"},
            },
        )
        result = GoogleInteractionsToolOps.ir_function_call_to_p(ir_part)
        assert result["type"] == "google_search_call"
        assert result["id"] == "gs_1"

    def test_ir_intrinsic_result_to_google_search_result(self):
        ir_part = cast(
            ToolResultPart,
            {
                "type": "tool_result",
                "tool_call_id": "gs_1",
                "result": "results",
                "tool_type": "intrinsic",
                "provider_metadata": {"intrinsic_kind": "google_search"},
            },
        )
        result = GoogleInteractionsToolOps.ir_function_result_to_p(ir_part)
        assert result["type"] == "google_search_result"
        assert result["call_id"] == "gs_1"

    def test_function_call_unchanged(self):
        """Non-intrinsic calls still produce function_call."""
        ir_part = cast(
            ToolCallPart,
            {
                "type": "tool_call",
                "tool_call_id": "fc_1",
                "tool_name": "get_weather",
                "tool_input": {"city": "Tokyo"},
            },
        )
        result = GoogleInteractionsToolOps.ir_function_call_to_p(ir_part)
        assert result["type"] == "function_call"


class TestServerStepRoundTrip:
    """Test server step types round-trip (Provider → IR → Provider)."""

    def test_google_search_round_trip(self):
        original = {
            "type": "google_search_call",
            "id": "gs_rt",
            "name": "google_search",
            "arguments": {"query": "test"},
        }
        ir = GoogleInteractionsToolOps.p_server_call_to_ir(original)
        restored = GoogleInteractionsToolOps.ir_function_call_to_p(ir)
        assert restored["type"] == "google_search_call"
        assert restored["id"] == original["id"]

    def test_code_execution_result_round_trip(self):
        original = {
            "type": "code_execution_result",
            "call_id": "ce_rt",
            "result": "42",
        }
        ir = GoogleInteractionsToolOps.p_server_result_to_ir(original)
        restored = GoogleInteractionsToolOps.ir_function_result_to_p(ir)
        assert restored["type"] == "code_execution_result"
        assert restored["call_id"] == original["call_id"]


class TestMessageOpsServerStepDispatch:
    """Test that message_ops dispatches server step types correctly."""

    def test_server_call_in_steps(self):
        ops = GoogleInteractionsMessageOps()
        steps = [
            {
                "type": "google_search_call",
                "id": "gs_msg",
                "name": "google_search",
                "arguments": {"query": "news"},
            },
        ]
        messages = ops.p_steps_to_ir_messages(steps)
        assert len(messages) == 1
        assert messages[0]["role"] == "assistant"
        tc = cast(dict[str, Any], messages[0]["content"][0])
        assert tc["tool_type"] == "intrinsic"
        assert tc["provider_metadata"]["intrinsic_kind"] == "google_search"

    def test_server_result_in_steps(self):
        ops = GoogleInteractionsMessageOps()
        steps = [
            {
                "type": "google_search_result",
                "call_id": "gs_msg",
                "result": "search results",
            },
        ]
        messages = ops.p_steps_to_ir_messages(steps)
        assert len(messages) == 1
        assert messages[0]["role"] == "tool"
        tr = cast(dict[str, Any], messages[0]["content"][0])
        assert tr["tool_type"] == "intrinsic"
        assert tr["provider_metadata"]["intrinsic_kind"] == "google_search"
