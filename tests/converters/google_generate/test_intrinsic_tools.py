"""Tests for Google Generate intrinsic tool handling.

Covers:
- executableCode / codeExecutionResult ↔ IR intrinsic tool call/result
- google_search / code_execution tool definition roundtrip
- A→IR→A roundtrip through tool_ops methods
"""

from typing import Any, cast

from llm_rosetta.converters.google_generate.content_ops import (
    GoogleGenerateContentOps,
)
from llm_rosetta.converters.google_generate.message_ops import (
    GoogleGenerateMessageOps,
)
from llm_rosetta.converters.google_generate.tool_ops import GoogleGenerateToolOps
from llm_rosetta.types.ir import ToolCallPart, ToolDefinition, ToolResultPart


def _make_message_ops() -> GoogleGenerateMessageOps:
    return GoogleGenerateMessageOps(
        content_ops=GoogleGenerateContentOps(),
        tool_ops=GoogleGenerateToolOps(),
    )


# ── Provider → IR (via message_ops) ──────────────────────────────


class TestGoogleExecutableCodeToIR:
    """Test Google executableCode → IR intrinsic tool call."""

    def test_executable_code_to_ir(self):
        ops = _make_message_ops()
        provider_msg = {
            "role": "model",
            "parts": [
                {
                    "executableCode": {
                        "code": "print('hello')",
                        "language": "PYTHON",
                    }
                }
            ],
        }
        result = cast(dict[str, Any], ops._p_message_to_ir(provider_msg))
        assert result["role"] == "assistant"
        tc = result["content"][0]
        assert tc["type"] == "tool_call"
        assert tc["tool_type"] == "intrinsic"
        assert tc["provider_metadata"]["intrinsic_kind"] == "code_execution"
        assert tc["tool_input"]["code"] == "print('hello')"
        assert tc["tool_input"]["language"] == "PYTHON"
        assert tc["tool_call_id"].startswith("google_exec_")

    def test_executable_code_snake_case(self):
        ops = _make_message_ops()
        provider_msg = {
            "role": "model",
            "parts": [
                {
                    "executable_code": {
                        "code": "x = 1",
                        "language": "PYTHON",
                    }
                }
            ],
        }
        result = cast(dict[str, Any], ops._p_message_to_ir(provider_msg))
        tc = result["content"][0]
        assert tc["tool_type"] == "intrinsic"
        assert tc["tool_input"]["code"] == "x = 1"


class TestGoogleCodeExecutionResultToIR:
    """Test Google codeExecutionResult → IR intrinsic tool result."""

    def test_code_execution_result_ok(self):
        ops = _make_message_ops()
        provider_msg = {
            "role": "model",
            "parts": [
                {
                    "codeExecutionResult": {
                        "output": "hello\n",
                        "outcome": "OUTCOME_OK",
                    }
                }
            ],
        }
        result = cast(dict[str, Any], ops._p_message_to_ir(provider_msg))
        assert result["role"] == "tool"
        tr = result["content"][0]
        assert tr["type"] == "tool_result"
        assert tr["tool_type"] == "intrinsic"
        assert tr["provider_metadata"]["intrinsic_kind"] == "code_execution"
        assert tr["result"] == "hello\n"
        assert not tr.get("is_error")

    def test_code_execution_result_failed(self):
        ops = _make_message_ops()
        provider_msg = {
            "role": "model",
            "parts": [
                {
                    "codeExecutionResult": {
                        "output": "NameError: name 'x' is not defined",
                        "outcome": "OUTCOME_FAILED",
                    }
                }
            ],
        }
        result = cast(dict[str, Any], ops._p_message_to_ir(provider_msg))
        tr = result["content"][0]
        assert tr["is_error"] is True


# ── IR → Provider (via tool_ops) ─────────────────────────────────


class TestIRIntrinsicToGoogle:
    """Test IR intrinsic → Google parts via tool_ops."""

    def test_ir_tool_call_to_executable_code(self):
        ir_part: dict[str, Any] = {
            "type": "tool_call",
            "tool_call_id": "c1",
            "tool_name": "code_execution",
            "tool_input": {"code": "print(42)", "language": "PYTHON"},
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "code_execution"},
        }
        result = GoogleGenerateToolOps.ir_intrinsic_call_to_p(
            cast(ToolCallPart, ir_part)
        )
        assert result is not None
        assert result["executableCode"]["code"] == "print(42)"
        assert result["executableCode"]["language"] == "PYTHON"

    def test_ir_tool_result_to_code_execution_result_ok(self):
        ir_part: dict[str, Any] = {
            "type": "tool_result",
            "tool_call_id": "c1",
            "result": "42\n",
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "code_execution"},
            "is_error": False,
        }
        result = GoogleGenerateToolOps.ir_intrinsic_result_to_p(
            cast(ToolResultPart, ir_part)
        )
        assert result is not None
        assert result["codeExecutionResult"]["output"] == "42\n"
        assert result["codeExecutionResult"]["outcome"] == "OUTCOME_OK"

    def test_ir_tool_result_to_code_execution_result_failed(self):
        ir_part: dict[str, Any] = {
            "type": "tool_result",
            "tool_call_id": "c1",
            "result": "Error",
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "code_execution"},
            "is_error": True,
        }
        result = GoogleGenerateToolOps.ir_intrinsic_result_to_p(
            cast(ToolResultPart, ir_part)
        )
        assert result is not None
        assert result["codeExecutionResult"]["outcome"] == "OUTCOME_FAILED"

    def test_non_code_execution_call_returns_none(self):
        ir_part: dict[str, Any] = {
            "type": "tool_call",
            "tool_call_id": "c1",
            "tool_name": "web_search",
            "tool_input": {},
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "web_search"},
        }
        assert (
            GoogleGenerateToolOps.ir_intrinsic_call_to_p(cast(ToolCallPart, ir_part))
            is None
        )

    def test_non_code_execution_result_returns_none(self):
        ir_part: dict[str, Any] = {
            "type": "tool_result",
            "tool_call_id": "c1",
            "result": "search results",
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "web_search"},
        }
        assert (
            GoogleGenerateToolOps.ir_intrinsic_result_to_p(
                cast(ToolResultPart, ir_part)
            )
            is None
        )


# ── Provider → IR (via tool_ops directly) ─────────────────────────


class TestProviderIntrinsicPartToIR:
    """Test Google intrinsic parts → IR via tool_ops.p_intrinsic_*_to_ir."""

    def test_executable_code_part_to_ir(self):
        part = {"executableCode": {"code": "1+1", "language": "PYTHON"}}
        result = GoogleGenerateToolOps.p_intrinsic_call_to_ir(part)
        assert result is not None
        assert result["tool_type"] == "intrinsic"
        assert result["provider_metadata"]["intrinsic_kind"] == "code_execution"
        assert result["tool_input"]["code"] == "1+1"

    def test_code_execution_result_part_to_ir(self):
        part = {"codeExecutionResult": {"output": "2", "outcome": "OUTCOME_OK"}}
        result = GoogleGenerateToolOps.p_intrinsic_result_to_ir(part)
        assert result is not None
        assert result["tool_type"] == "intrinsic"
        assert result["result"] == "2"
        assert not result.get("is_error")

    def test_unrecognized_part_returns_none(self):
        assert GoogleGenerateToolOps.p_intrinsic_call_to_ir({"text": "hi"}) is None
        assert GoogleGenerateToolOps.p_intrinsic_result_to_ir({"text": "hi"}) is None


# ── Roundtrip: A → IR → A ────────────────────────────────────────


class TestCodeExecutionRoundTrip:
    """executableCode / codeExecutionResult A→IR→A roundtrip via tool_ops."""

    def test_executable_code_round_trip(self):
        original = {
            "executableCode": {
                "code": "for i in range(3): print(i)",
                "language": "PYTHON",
            }
        }
        ir = GoogleGenerateToolOps.p_intrinsic_call_to_ir(original)
        assert ir is not None
        restored = GoogleGenerateToolOps.ir_intrinsic_call_to_p(ir)
        assert restored is not None
        assert restored["executableCode"]["code"] == original["executableCode"]["code"]
        assert (
            restored["executableCode"]["language"]
            == original["executableCode"]["language"]
        )

    def test_code_execution_result_round_trip(self):
        original = {
            "codeExecutionResult": {"output": "0\n1\n2\n", "outcome": "OUTCOME_OK"}
        }
        ir = GoogleGenerateToolOps.p_intrinsic_result_to_ir(original)
        assert ir is not None
        restored = GoogleGenerateToolOps.ir_intrinsic_result_to_p(ir)
        assert restored is not None
        assert (
            restored["codeExecutionResult"]["output"]
            == original["codeExecutionResult"]["output"]
        )
        assert (
            restored["codeExecutionResult"]["outcome"]
            == original["codeExecutionResult"]["outcome"]
        )

    def test_failed_result_round_trip(self):
        original = {
            "codeExecutionResult": {
                "output": "ZeroDivisionError",
                "outcome": "OUTCOME_FAILED",
            }
        }
        ir = GoogleGenerateToolOps.p_intrinsic_result_to_ir(original)
        assert ir is not None
        assert ir["is_error"] is True
        restored = GoogleGenerateToolOps.ir_intrinsic_result_to_p(ir)
        assert restored is not None
        assert restored["codeExecutionResult"]["outcome"] == "OUTCOME_FAILED"

    def test_message_level_round_trip(self):
        """Full message-level roundtrip via message_ops delegates to tool_ops."""
        ops = _make_message_ops()
        original_part = {
            "executableCode": {
                "code": "for i in range(3): print(i)",
                "language": "PYTHON",
            }
        }
        provider_msg = {"role": "model", "parts": [original_part]}
        ir_msg = cast(dict[str, Any], ops._p_message_to_ir(provider_msg))
        tc = ir_msg["content"][0]
        restored = GoogleGenerateToolOps.ir_intrinsic_call_to_p(tc)
        assert restored is not None
        assert (
            restored["executableCode"]["code"]
            == original_part["executableCode"]["code"]
        )


# ── Tool Definition Roundtrip ────────────────────────────────────


class TestIntrinsicToolDefinitionRoundTrip:
    """google_search / code_execution tool definition A→IR→A roundtrip."""

    def test_google_search_definition_to_ir(self):
        provider_tool = {"google_search": {}}
        ir = cast(
            ToolDefinition, GoogleGenerateToolOps.p_tool_definition_to_ir(provider_tool)
        )
        assert ir is not None
        assert ir["type"] == "intrinsic"
        assert ir["name"] == "google_search"
        assert ir["metadata"]["intrinsic_kind"] == "google_search"

    def test_google_search_definition_round_trip(self):
        original = {"google_search": {}}
        ir = GoogleGenerateToolOps.p_tool_definition_to_ir(original)
        assert ir is not None
        restored = GoogleGenerateToolOps.ir_tool_definition_to_p(
            cast(ToolDefinition, ir)
        )
        assert restored == original

    def test_google_search_camel_case(self):
        provider_tool = {"googleSearch": {}}
        ir = cast(
            ToolDefinition, GoogleGenerateToolOps.p_tool_definition_to_ir(provider_tool)
        )
        assert ir is not None
        assert ir["metadata"]["intrinsic_kind"] == "google_search"

    def test_code_execution_definition_to_ir(self):
        provider_tool = {"code_execution": {}}
        ir = cast(
            ToolDefinition, GoogleGenerateToolOps.p_tool_definition_to_ir(provider_tool)
        )
        assert ir is not None
        assert ir["type"] == "intrinsic"
        assert ir["name"] == "code_execution"
        assert ir["metadata"]["intrinsic_kind"] == "code_execution"

    def test_code_execution_definition_round_trip(self):
        original = {"code_execution": {}}
        ir = GoogleGenerateToolOps.p_tool_definition_to_ir(original)
        assert ir is not None
        restored = GoogleGenerateToolOps.ir_tool_definition_to_p(
            cast(ToolDefinition, ir)
        )
        assert restored == original

    def test_code_execution_camel_case(self):
        provider_tool = {"codeExecution": {}}
        ir = cast(
            ToolDefinition, GoogleGenerateToolOps.p_tool_definition_to_ir(provider_tool)
        )
        assert ir is not None
        assert ir["metadata"]["intrinsic_kind"] == "code_execution"

    def test_ir_intrinsic_definition_to_provider(self):
        ir_tool: dict[str, Any] = {
            "type": "intrinsic",
            "name": "google_search",
            "description": "",
            "parameters": {},
            "metadata": {"intrinsic_kind": "google_search"},
        }
        result = GoogleGenerateToolOps.ir_tool_definition_to_p(
            cast(ToolDefinition, ir_tool)
        )
        assert result == {"google_search": {}}

    def test_function_definition_unchanged(self):
        """Regular function definitions are not affected by intrinsic handling."""
        provider_tool = {
            "function_declarations": [
                {
                    "name": "get_weather",
                    "description": "Get weather",
                    "parameters": {"type": "object", "properties": {}},
                }
            ]
        }
        ir = cast(
            ToolDefinition, GoogleGenerateToolOps.p_tool_definition_to_ir(provider_tool)
        )
        assert ir is not None
        assert ir["type"] == "function"
        assert ir["name"] == "get_weather"
