"""Tests for IR validation utilities."""

import pytest

from llm_rosetta._vendor.validate import ValidationError
from llm_rosetta.types.ir.validation import (
    validate_ir_request,
    validate_ir_response,
    validate_messages,
)


# ============================================================================
# Fixtures: minimal valid IR structures
# ============================================================================


def _text_part(text: str = "hello") -> dict:
    return {"type": "text", "text": text}


def _user_message(text: str = "hello") -> dict:
    return {"role": "user", "content": [_text_part(text)]}


def _assistant_message(text: str = "hi") -> dict:
    return {"role": "assistant", "content": [_text_part(text)]}


def _finish_reason(reason: str = "stop") -> dict:
    return {"reason": reason}


def _choice(index: int = 0, text: str = "hi") -> dict:
    return {
        "index": index,
        "message": _assistant_message(text),
        "finish_reason": _finish_reason(),
    }


def _minimal_request(**overrides) -> dict:
    base = {
        "model": "gpt-4",
        "messages": [_user_message()],
    }
    base.update(overrides)
    return base


def _minimal_response(**overrides) -> dict:
    base = {
        "id": "resp-123",
        "object": "response",
        "created": 1700000000,
        "model": "gpt-4",
        "choices": [_choice()],
    }
    base.update(overrides)
    return base


# ============================================================================
# validate_ir_request
# ============================================================================


class TestValidateIRRequest:
    def test_valid_minimal(self):
        result = validate_ir_request(_minimal_request())
        assert result["model"] == "gpt-4"
        assert len(result["messages"]) == 1

    def test_valid_with_system_instruction(self):
        result = validate_ir_request(_minimal_request(system_instruction="Be helpful"))
        assert result["system_instruction"] == "Be helpful"

    def test_valid_with_generation_config(self):
        result = validate_ir_request(
            _minimal_request(generation={"temperature": 0.7, "max_tokens": 100})
        )
        assert result["generation"]["temperature"] == 0.7

    def test_missing_model(self):
        data = _minimal_request()
        del data["model"]
        with pytest.raises(ValidationError):
            validate_ir_request(data)

    def test_missing_messages(self):
        data = _minimal_request()
        del data["messages"]
        with pytest.raises(ValidationError):
            validate_ir_request(data)

    def test_invalid_message_role(self):
        data = _minimal_request(
            messages=[{"role": "invalid_role", "content": [_text_part()]}]
        )
        with pytest.raises(ValidationError):
            validate_ir_request(data)

    def test_message_missing_content(self):
        data = _minimal_request(messages=[{"role": "user"}])
        with pytest.raises(ValidationError):
            validate_ir_request(data)

    def test_multiple_messages(self):
        result = validate_ir_request(
            _minimal_request(
                messages=[
                    _user_message("hello"),
                    _assistant_message("hi"),
                    _user_message("how are you"),
                ]
            )
        )
        assert len(result["messages"]) == 3

    def test_tool_message(self):
        result = validate_ir_request(
            _minimal_request(
                messages=[
                    _user_message(),
                    {
                        "role": "tool",
                        "content": [
                            {
                                "type": "tool_result",
                                "tool_call_id": "call-1",
                                "result": "42",
                            }
                        ],
                    },
                ]
            )
        )
        assert result["messages"][1]["role"] == "tool"


# ============================================================================
# validate_ir_response
# ============================================================================


class TestValidateIRResponse:
    def test_valid_minimal(self):
        result = validate_ir_response(_minimal_response())
        assert result["id"] == "resp-123"
        assert result["object"] == "response"
        assert len(result["choices"]) == 1

    def test_valid_with_usage(self):
        result = validate_ir_response(
            _minimal_response(
                usage={
                    "prompt_tokens": 10,
                    "completion_tokens": 20,
                    "total_tokens": 30,
                }
            )
        )
        assert result["usage"]["total_tokens"] == 30

    def test_missing_id(self):
        data = _minimal_response()
        del data["id"]
        with pytest.raises(ValidationError):
            validate_ir_response(data)

    def test_missing_choices(self):
        data = _minimal_response()
        del data["choices"]
        with pytest.raises(ValidationError):
            validate_ir_response(data)

    def test_invalid_object_literal(self):
        data = _minimal_response(object="not_response")
        with pytest.raises(ValidationError):
            validate_ir_response(data)

    def test_invalid_finish_reason(self):
        data = _minimal_response(
            choices=[
                {
                    "index": 0,
                    "message": _assistant_message(),
                    "finish_reason": {"reason": "unknown_reason"},
                }
            ]
        )
        with pytest.raises(ValidationError):
            validate_ir_response(data)

    def test_valid_finish_reasons(self):
        for reason in [
            "stop",
            "length",
            "tool_calls",
            "content_filter",
            "refusal",
            "error",
            "cancelled",
        ]:
            data = _minimal_response(
                choices=[
                    {
                        "index": 0,
                        "message": _assistant_message(),
                        "finish_reason": {"reason": reason},
                    }
                ]
            )
            result = validate_ir_response(data)
            assert result["choices"][0]["finish_reason"]["reason"] == reason


# ============================================================================
# validate_messages
# ============================================================================


class TestValidateMessages:
    def test_valid_message_list(self):
        result = validate_messages([_user_message(), _assistant_message()])
        assert len(result) == 2

    def test_valid_system_message(self):
        result = validate_messages(
            [{"role": "system", "content": [_text_part("system prompt")]}]
        )
        msg = result[0]
        assert isinstance(msg, dict) and msg.get("role") == "system"

    def test_valid_extension_item(self):
        result = validate_messages(
            [
                {
                    "type": "system_event",
                    "event_type": "session_start",
                    "timestamp": "2024-01-01T00:00:00Z",
                }
            ]
        )
        ext = result[0]
        assert isinstance(ext, dict) and ext.get("type") == "system_event"

    def test_mixed_messages_and_extensions(self):
        result = validate_messages(
            [
                _user_message(),
                {
                    "type": "batch_marker",
                    "batch_id": "b1",
                    "batch_type": "start",
                },
                _assistant_message(),
            ]
        )
        assert len(result) == 3

    def test_invalid_role(self):
        with pytest.raises(ValidationError):
            validate_messages([{"role": "invalid", "content": [_text_part()]}])

    def test_not_a_dict(self):
        with pytest.raises(ValidationError):
            validate_messages(["not a dict"])

    def test_empty_list(self):
        result = validate_messages([])
        assert result == []

    def test_validation_error_has_details(self):
        with pytest.raises(ValidationError) as exc_info:
            validate_ir_request({"model": "gpt-4"})  # missing messages
        assert len(exc_info.value.errors) > 0
        assert exc_info.value.errors[0].message
