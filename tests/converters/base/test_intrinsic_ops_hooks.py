"""Tests for the intrinsic hooks on ``BaseToolOps``.

Phase 3 makes an intrinsic tool call a *derived* case of an ordinary tool
call: the base's ``ir_tool_call_to_p`` / ``ir_tool_result_to_p`` templates ask
the converter's native hooks first and otherwise apply the converter's
non-native outcome.  These tests pin that dispatch with test doubles, so a
converter's own behaviour is pinned separately by the matrix.
"""

from __future__ import annotations

from typing import Any, cast

from llm_rosetta.converters.base import BaseToolOps
from llm_rosetta.converters.base.tools.intrinsic import (
    intrinsic_kind_of,
    make_intrinsic_tool_call,
    make_intrinsic_tool_result,
)
from llm_rosetta.types.ir import ToolCallPart, ToolResultPart

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _PlainToolOps(BaseToolOps):
    """A dialect with no intrinsic vocabulary (``chat_completions``' shape)."""

    @staticmethod
    def ir_tool_definition_to_p(ir_tool: Any, **kwargs: Any) -> Any:
        return {}

    @staticmethod
    def p_tool_definition_to_ir(provider_tool: Any, **kwargs: Any) -> Any:
        return {}

    @staticmethod
    def ir_tool_choice_to_p(ir_tool_choice: Any, **kwargs: Any) -> Any:
        return {}

    @staticmethod
    def p_tool_choice_to_ir(provider_tool_choice: Any, **kwargs: Any) -> Any:
        return {}

    @staticmethod
    def p_tool_call_to_ir(provider_tool_call: Any, **kwargs: Any) -> Any:
        return {}

    @staticmethod
    def p_tool_result_to_ir(provider_tool_result: Any, **kwargs: Any) -> Any:
        return {}

    @staticmethod
    def ir_tool_config_to_p(ir_tool_config: Any, **kwargs: Any) -> Any:
        return {}

    @staticmethod
    def p_tool_config_to_ir(provider_tool_config: Any, **kwargs: Any) -> Any:
        return {}

    @classmethod
    def ir_client_call_to_p(cls, ir_tool_call: ToolCallPart, **kwargs: Any) -> Any:
        return {"shape": "client_call", "name": ir_tool_call["tool_name"]}

    @classmethod
    def ir_client_result_to_p(
        cls, ir_tool_result: ToolResultPart, **kwargs: Any
    ) -> Any:
        return {"shape": "client_result", "id": ir_tool_result["tool_call_id"]}


class _HookedToolOps(_PlainToolOps):
    """A dialect that renders one kind natively and drops the rest."""

    @classmethod
    def intrinsic_leg_from_flags(cls, kwargs: Any, *, default: Any) -> Any:
        return "response" if kwargs.get("for_response") else default

    @classmethod
    def intrinsic_renderable(cls, kind: str, fields: Any, *, leg: str) -> bool:
        return kind == "native_kind"

    @classmethod
    def ir_intrinsic_call_to_p(
        cls, ir_tool_call: ToolCallPart, *, leg: str = "history", **kwargs: Any
    ) -> Any | None:
        fields = ir_tool_call.get("tool_input")
        if not cls.intrinsic_renderable(
            intrinsic_kind_of(ir_tool_call),
            fields if isinstance(fields, dict) else {},
            leg=cast(Any, leg),
        ):
            return None
        return {"shape": "native_call", "leg": leg}

    @classmethod
    def ir_intrinsic_result_to_p(
        cls, ir_tool_result: ToolResultPart, *, leg: str = "history", **kwargs: Any
    ) -> Any | None:
        if not cls.intrinsic_renderable(
            intrinsic_kind_of(ir_tool_result), {}, leg=cast(Any, leg)
        ):
            return None
        return {"shape": "native_result", "leg": leg}

    @classmethod
    def non_native_intrinsic_call_to_p(
        cls, ir_tool_call: ToolCallPart, *, leg: str, **kwargs: Any
    ) -> Any:
        return {}

    @classmethod
    def non_native_intrinsic_result_to_p(
        cls, ir_tool_result: ToolResultPart, *, leg: str, **kwargs: Any
    ) -> Any:
        return {}


def _call(kind: str) -> ToolCallPart:
    return cast(
        ToolCallPart,
        make_intrinsic_tool_call(
            tool_call_id="c1", tool_name=kind, tool_input={}, intrinsic_kind=kind
        ),
    )


def _result(kind: str) -> ToolResultPart:
    return cast(
        ToolResultPart,
        make_intrinsic_tool_result(tool_call_id="c1", result="R", intrinsic_kind=kind),
    )


# ---------------------------------------------------------------------------
# The default: no intrinsic vocabulary, so the parent path
# ---------------------------------------------------------------------------


def test_plain_dialect_writes_an_intrinsic_call_as_a_client_call():
    out = _PlainToolOps.ir_tool_call_to_p(_call("web_search"), leg="history")
    assert out == {"shape": "client_call", "name": "web_search"}


def test_plain_dialect_writes_an_intrinsic_result_as_a_client_result():
    out = _PlainToolOps.ir_tool_result_to_p(_result("web_search"))
    assert out == {"shape": "client_result", "id": "c1"}


def test_plain_client_parts_never_reach_the_hooks():
    part = cast(
        ToolCallPart,
        {"type": "tool_call", "tool_call_id": "c1", "tool_name": "f", "tool_input": {}},
    )
    assert _PlainToolOps.ir_tool_call_to_p(part)["shape"] == "client_call"


# ---------------------------------------------------------------------------
# A hooked dialect: the hook wins, else the converter's non-native outcome
# ---------------------------------------------------------------------------


def test_a_native_kind_is_rendered_by_the_hook():
    assert (
        _HookedToolOps.ir_tool_call_to_p(_call("native_kind"))["shape"] == "native_call"
    )
    assert (
        _HookedToolOps.ir_tool_result_to_p(_result("native_kind"))["shape"]
        == "native_result"
    )


def test_a_foreign_kind_takes_the_non_native_outcome():
    """The hook declines, and the converter drops rather than degrading."""
    assert _HookedToolOps.ir_tool_call_to_p(_call("web_search")) == {}
    assert _HookedToolOps.ir_tool_result_to_p(_result("web_search")) == {}


def test_a_hook_that_declines_falls_back_to_the_client_path():
    class _DegradingToolOps(_HookedToolOps):
        @classmethod
        def non_native_intrinsic_call_to_p(
            cls, ir_tool_call: ToolCallPart, *, leg: str, **kwargs: Any
        ) -> Any:
            return cls.ir_client_call_to_p(ir_tool_call, **kwargs)

        @classmethod
        def non_native_intrinsic_result_to_p(
            cls, ir_tool_result: ToolResultPart, *, leg: str, **kwargs: Any
        ) -> Any:
            return cls.ir_client_result_to_p(ir_tool_result, **kwargs)

    assert _DegradingToolOps.ir_tool_call_to_p(_call("web_search"), leg="history") == {
        "shape": "client_call",
        "name": "web_search",
    }


# ---------------------------------------------------------------------------
# The leg
# ---------------------------------------------------------------------------


def test_the_leg_comes_from_the_converter_flag():
    assert _HookedToolOps.ir_tool_call_to_p(
        _call("native_kind"), for_response=True
    ) == {"shape": "native_call", "leg": "response"}


def test_a_caller_may_state_the_leg_outright():
    """``leg=`` wins over the flag, and is not forwarded to the hook twice."""
    assert _HookedToolOps.ir_tool_result_to_p(
        _result("native_kind"), for_response=True, leg="history"
    ) == {"shape": "native_result", "leg": "history"}


def test_each_template_passes_its_own_default_leg():
    """A dialect may answer per writer; each template keeps that answer.

    open_responses answered "response" for a call that says nothing and
    "history" for a result that says nothing, so the two templates pass those
    defaults rather than one for both.
    """
    assert _HookedToolOps.ir_tool_call_to_p(_call("native_kind"))["leg"] == "response"
    assert (
        _HookedToolOps.ir_tool_result_to_p(_result("native_kind"))["leg"] == "history"
    )


def test_the_leg_resolution_takes_the_flag_then_the_default():
    assert _HookedToolOps.intrinsic_leg({}, "history") == "history"
    assert _HookedToolOps.intrinsic_leg({}, "response") == "response"
    assert _HookedToolOps.intrinsic_leg({"for_response": True}, "history") == "response"
    assert (
        _HookedToolOps.intrinsic_leg(
            {"for_response": True, "leg": "history"}, "history"
        )
        == "history"
    )


def test_a_library_caller_still_sees_the_drop(caplog: Any):
    """No warnings sink must not mean no report.

    The pipeline hands the converter a ``warnings`` list; a library caller does
    not.  The per-converter code the base replaced logged unconditionally, so a
    response-leg drop has to keep reaching that caller through the log.
    """
    with caplog.at_level("WARNING"):
        out = _PlainToolOps.ir_tool_call_to_p(_call("web_search"), leg="response")

    assert out == {}
    assert any(
        "web_search" in record.message and "dropped" in record.message
        for record in caplog.records
    )
