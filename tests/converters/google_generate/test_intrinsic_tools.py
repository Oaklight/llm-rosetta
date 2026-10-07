"""Tests for Google Generate intrinsic tool handling (executableCode / codeExecutionResult)."""

from typing import Any, cast

from llm_rosetta.converters.google_generate.content_ops import (
    GoogleGenerateContentOps,
)
from llm_rosetta.converters.google_generate.message_ops import (
    GoogleGenerateMessageOps,
    _ir_intrinsic_tool_call_to_google,
    _ir_intrinsic_tool_result_to_google,
)
from llm_rosetta.converters.google_generate.tool_ops import GoogleGenerateToolOps


def _make_message_ops() -> GoogleGenerateMessageOps:
    return GoogleGenerateMessageOps(
        content_ops=GoogleGenerateContentOps(),
        tool_ops=GoogleGenerateToolOps(),
    )


class TestGoogleExecutableCodeToIR:
    """Test Google executableCode → IR intrinsic tool call."""

    def test_executable_code_to_ir(self):
        """executableCode part becomes IR ToolCallPart with intrinsic type."""
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
        """Snake-case key also works."""
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
        """Successful codeExecutionResult maps to IR ToolResultPart."""
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
        # Result goes to tool-role message
        assert result["role"] == "tool"
        tr = result["content"][0]
        assert tr["type"] == "tool_result"
        assert tr["tool_type"] == "intrinsic"
        assert tr["provider_metadata"]["intrinsic_kind"] == "code_execution"
        assert tr["result"] == "hello\n"
        assert tr["is_error"] is False

    def test_code_execution_result_failed(self):
        """Failed codeExecutionResult sets is_error=True."""
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


class TestIRIntrinsicToGoogle:
    """Test IR intrinsic → Google executableCode / codeExecutionResult."""

    def test_ir_tool_call_to_executable_code(self):
        """IR intrinsic code_execution → Google executableCode part."""
        ir_part: dict[str, Any] = {
            "type": "tool_call",
            "tool_call_id": "c1",
            "tool_name": "code_execution",
            "tool_input": {"code": "print(42)", "language": "PYTHON"},
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "code_execution"},
        }
        result = _ir_intrinsic_tool_call_to_google(ir_part)
        assert result is not None
        assert "executableCode" in result
        assert result["executableCode"]["code"] == "print(42)"
        assert result["executableCode"]["language"] == "PYTHON"

    def test_ir_tool_result_to_code_execution_result_ok(self):
        """IR intrinsic code_execution result → Google codeExecutionResult."""
        ir_part: dict[str, Any] = {
            "type": "tool_result",
            "tool_call_id": "c1",
            "result": "42\n",
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "code_execution"},
            "is_error": False,
        }
        result = _ir_intrinsic_tool_result_to_google(ir_part)
        assert result is not None
        assert result["codeExecutionResult"]["output"] == "42\n"
        assert result["codeExecutionResult"]["outcome"] == "OUTCOME_OK"

    def test_ir_tool_result_to_code_execution_result_failed(self):
        """IR intrinsic error result → Google codeExecutionResult with OUTCOME_FAILED."""
        ir_part: dict[str, Any] = {
            "type": "tool_result",
            "tool_call_id": "c1",
            "result": "Error",
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "code_execution"},
            "is_error": True,
        }
        result = _ir_intrinsic_tool_result_to_google(ir_part)
        assert result is not None
        assert result["codeExecutionResult"]["outcome"] == "OUTCOME_FAILED"

    def test_non_code_execution_returns_none(self):
        """Non-code_execution intrinsic kinds return None (unsupported by Google Generate)."""
        ir_part: dict[str, Any] = {
            "type": "tool_call",
            "tool_call_id": "c1",
            "tool_name": "web_search",
            "tool_input": {},
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": "web_search"},
        }
        assert _ir_intrinsic_tool_call_to_google(ir_part) is None

    def test_round_trip_executable_code(self):
        """executableCode → IR → executableCode preserves code and language."""
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

        restored = _ir_intrinsic_tool_call_to_google(tc)
        assert restored is not None
        assert (
            restored["executableCode"]["code"]
            == original_part["executableCode"]["code"]
        )
        assert (
            restored["executableCode"]["language"]
            == original_part["executableCode"]["language"]
        )
