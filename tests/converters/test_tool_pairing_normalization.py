"""The request-level pairing repair, applied once for every chat target.

A history that no provider accepts — an unanswered call, a result whose
declaration is missing, two declarations sharing one id — is repaired in the
IR before any dialect writes it.  These tests pin the three properties that
make it one repair instead of five:

* every target sees the same repaired history;
* exactly one warning describes each repair;
* the writer adds none of its own.
"""

from __future__ import annotations

import copy
import json
from typing import Any, cast

import pytest

from llm_rosetta.auto_detect import get_converter_for_provider
from llm_rosetta.conversion_warnings import WarningSink
from llm_rosetta.converters.llm.chat_completions.message_ops import (
    ChatCompletionsMessageOps,
)
from llm_rosetta.converters.llm.chat_completions.content_ops import (
    ChatCompletionsContentOps,
)
from llm_rosetta.converters.llm.chat_completions.tool_ops import ChatCompletionsToolOps
from llm_rosetta.ir.tools.orphan_fix import (
    UNANSWERED_TOOL_CALL,
    UnrepairableToolHistory,
    normalize_tool_pairing,
)
from llm_rosetta.pipeline import ConversionError

CHAT_FORMATS = (
    "chat_completions",
    "messages",
    "open_responses",
    "google_generate",
    "google_interactions",
)


def text(value: str) -> dict[str, Any]:
    """An IR text part."""
    return {"type": "text", "text": value}


def call(call_id: str, name: str = "f") -> dict[str, Any]:
    """An IR client tool-call part."""
    return {
        "type": "tool_call",
        "tool_call_id": call_id,
        "tool_name": name,
        "tool_input": {},
    }


def server_call(call_id: str, kind: str = "web_search") -> dict[str, Any]:
    """An IR intrinsic (provider-hosted) call part."""
    return {
        "type": "tool_call",
        "tool_call_id": call_id,
        "tool_name": kind,
        "tool_input": {},
        "tool_type": "intrinsic",
        "intrinsic_kind": kind,
    }


def result(call_id: str, value: str = "ok") -> dict[str, Any]:
    """An IR tool-result part."""
    return {"type": "tool_result", "tool_call_id": call_id, "result": value}


def message(role: str, *parts: dict[str, Any]) -> dict[str, Any]:
    """An IR message."""
    return {"role": role, "content": list(parts)}


def request(*messages: dict[str, Any]) -> dict[str, Any]:
    """An IR request around *messages*."""
    return {"model": "m", "messages": list(messages)}


def write(fmt: str, ir: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    """Run one request through a target converter, as production does."""
    converter = get_converter_for_provider(fmt)
    body, warnings = converter.request_to_provider(copy.deepcopy(cast(Any, ir)))
    return body, [str(warning) for warning in warnings]


# ==================== One repaired history, every target ====================


@pytest.mark.parametrize("fmt", CHAT_FORMATS)
def test_duplicate_declarations_are_renamed_for_every_target(fmt):
    ir = request(
        message("user", text("go")),
        message("assistant", call("q1")),
        message("tool", result("q1", "one")),
        message("assistant", call("q1")),
        message("tool", result("q1", "two")),
    )

    body, warnings = write(fmt, ir)
    wire = json.dumps(body)

    assert "q1_duplicate_2" in wire
    assert sum("Renamed duplicate tool_call_id" in w for w in warnings) == 1


@pytest.mark.parametrize("fmt", CHAT_FORMATS)
def test_an_orphan_result_never_reaches_a_wire(fmt):
    ir = request(
        message("user", text("go")),
        message("assistant", call("a")),
        message("tool", result("a", "real")),
        message("tool", result("ghost", "bad")),
    )

    body, warnings = write(fmt, ir)

    assert "ghost" not in json.dumps(body)
    assert "bad" not in json.dumps(body)
    assert sum("Dropped tool result id='ghost'" in w for w in warnings) == 1


@pytest.mark.parametrize("fmt", CHAT_FORMATS)
def test_a_forward_reference_is_dropped_and_named(fmt):
    """A result before its declaration cannot be fixed by reordering."""
    ir = request(
        message("user", result("a", "early")),
        message("assistant", call("a")),
    )

    body, warnings = write(fmt, ir)

    assert "early" not in json.dumps(body)
    assert any("arrives before its declaration" in w for w in warnings)
    # The declaration is still unanswered, so the IR fills it.
    assert UNANSWERED_TOOL_CALL in json.dumps(body)


@pytest.mark.parametrize("fmt", CHAT_FORMATS)
def test_an_unanswered_declaration_is_filled_for_every_target(fmt):
    ir = request(
        message("user", text("go")),
        message("assistant", call("a")),
    )

    body, warnings = write(fmt, ir)

    assert UNANSWERED_TOOL_CALL in json.dumps(body)
    assert (
        sum(
            "Inserted placeholder for unanswered tool_call_id='a'" in w
            for w in warnings
        )
        == 1
    )


def test_a_server_call_is_not_given_a_synthetic_result():
    """The provider ran it; a client result would fabricate the turn."""
    ir = request(
        message("user", text("search")),
        message("assistant", server_call("s1")),
    )

    body, warnings = write("chat_completions", ir)

    assert UNANSWERED_TOOL_CALL not in json.dumps(body)
    assert not any("placeholder" in w for w in warnings)


# ==================== One repair, one warning, one producer ====================


def test_the_pass_is_idempotent():
    ir = request(
        message("user", text("go")),
        message("assistant", call("q1")),
        message("assistant", call("q1")),
        message("tool", result("ghost", "bad")),
    )

    first_sink = WarningSink()
    once = normalize_tool_pairing(ir["messages"], warnings=first_sink)
    second_sink = WarningSink()
    twice = normalize_tool_pairing(once, warnings=second_sink)

    assert twice == once
    assert first_sink
    assert list(second_sink) == []


def test_the_pipeline_records_each_repair_once():
    """Converter and pipeline must not each run the repair."""
    from llm_rosetta import ConversionPipeline

    body = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "go"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "q1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "q1", "content": "one"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "q1",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "q1", "content": "two"},
        ],
    }

    pipeline = ConversionPipeline("chat_completions", "chat_completions")
    target = pipeline.convert_request(body)
    warnings = [str(warning) for warning in pipeline.warnings]

    assert "q1_duplicate_2" in json.dumps(target)
    assert sum("Renamed duplicate tool_call_id" in warning for warning in warnings) == 1


def test_the_chat_writer_no_longer_repairs():
    """One producer: the writer reorders, it does not rename or fill."""
    ops = ChatCompletionsMessageOps(
        ChatCompletionsContentOps(), ChatCompletionsToolOps()
    )
    ir = [
        message("assistant", call("q1")),
        message("assistant", call("q1")),
        message("tool", result("q1", "one")),
    ]

    out, warnings = ops.ir_messages_to_p(cast(Any, ir))
    ids = [part["id"] for item in out for part in item.get("tool_calls") or []]

    assert ids == ["q1", "q1"]
    assert not any("Renamed duplicate" in w for w in warnings)
    assert not any("placeholder" in w for w in warnings)


# ==================== An empty history is a 400, not a silent drop ====================


def test_an_orphan_only_history_fails_the_request():
    ir = request(message("user", result("ghost", "bad")))

    with pytest.raises(UnrepairableToolHistory) as raised:
        normalize_tool_pairing(ir["messages"])

    assert "ghost" in str(raised.value)


def test_the_converter_boundary_raises_the_conversion_error():
    ir = request(message("user", result("ghost", "bad")))
    converter = get_converter_for_provider("chat_completions")

    with pytest.raises(ConversionError) as raised:
        converter.request_to_provider(copy.deepcopy(cast(Any, ir)))

    assert "ghost" in str(raised.value)
    assert raised.value.phase == "ir_to_target"


def test_a_chat_orphan_message_fails_through_the_pipeline():
    """The wire shape a client actually sends, through the real entry point."""
    from llm_rosetta import ConversionPipeline

    body = {
        "model": "m",
        "messages": [{"role": "tool", "tool_call_id": "ghost", "content": "ok"}],
    }

    with pytest.raises(ConversionError) as raised:
        ConversionPipeline("chat_completions", "chat_completions").convert_request(body)

    assert "ghost" in str(raised.value)
