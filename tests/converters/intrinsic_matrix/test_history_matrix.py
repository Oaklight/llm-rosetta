"""Intrinsic handling matrix — definition, history and non-streaming paths.

Pins the CURRENT per-converter × kind behaviour of the paths that are not
streaming:

* **definition write** — an IR ``type="intrinsic"`` tool definition emitted
  into each dialect's tool list.
* **history call write** — an intrinsic ``tool_call`` in an assistant turn of
  the IR request.
* **history result write** — the matching intrinsic ``tool_result`` in a
  ``role:"tool"`` message.
* **non-streaming response write** — an intrinsic call and result in an IR
  response, written back into a provider response body.

The tables below are the matrix; each cell is one parametrized case, and the
tests next to them pin the behaviour a cell's classification cannot show (a
warning, a restored native payload, a dropped half-write).
"""

from __future__ import annotations

import json
import logging
from typing import Any, cast

import pytest

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.chat_completions import ChatCompletionsConverter
from llm_rosetta.converters.google_generate import GoogleGenerateConverter
from llm_rosetta.converters.google_interactions import GoogleInteractionsConverter
from llm_rosetta.converters.messages import MessagesConverter
from llm_rosetta.converters.open_responses import OpenAIResponsesConverter
from llm_rosetta.types.ir import IRRequest

from .conftest import (
    CONVERTERS,
    UNKNOWN_KIND,
    definition_outcome,
    history_call_outcome,
    history_result_outcome,
    intrinsic_definition,
    ir_history,
    ir_response,
    matrix_cases,
    response_call_outcome,
    response_result_outcome,
)


# google_generate warns while dropping an unrenderable kind; the tests
# that pin the warning use pytest.warns, which sets its own filter.
pytestmark = pytest.mark.filterwarnings("ignore:Unsupported intrinsic kind")


# ---------------------------------------------------------------------------
# The matrix — definition write
# ---------------------------------------------------------------------------

DEFINITION_MATRIX: dict[str, dict[str, str]] = {
    "chat_completions": {
        "web_search": "dropped",
        "google_search": "dropped",
        UNKNOWN_KIND: "dropped",
    },
    "messages": {
        "web_search": "native",
        "code_execution": "native",
        "google_search": "dropped",
        UNKNOWN_KIND: "dropped",
    },
    "open_responses": {
        "web_search": "native",
        "code_interpreter": "native",
        "google_search": "dropped",
        UNKNOWN_KIND: "dropped",
    },
    "google_generate": {
        "code_execution": "native",
        "google_search": "native",
        "web_search": "dropped",
        UNKNOWN_KIND: "dropped",
    },
    "google_interactions": {
        "google_search": "native",
        "code_execution": "native",
        "web_search": "dropped",
        UNKNOWN_KIND: "dropped",
    },
}


@pytest.mark.parametrize("converter,kind,expected", matrix_cases(DEFINITION_MATRIX))
def test_definition_write(converter: str, kind: str, expected: str):
    assert definition_outcome(converter, kind) == expected


def test_unknown_definition_is_logged_then_dropped(caplog):
    """messages / open_responses / google_generate / google_interactions log."""
    for name in (
        "messages",
        "open_responses",
        "google_generate",
        "google_interactions",
    ):
        with caplog.at_level(logging.WARNING):
            caplog.clear()
            out = CONVERTERS[name]().tool_ops.ir_tool_definition_to_p(
                intrinsic_definition(UNKNOWN_KIND)
            )
        assert out == {}
        assert any(UNKNOWN_KIND in rec.getMessage() for rec in caplog.records)


def test_unknown_definition_chat_is_silent(caplog):
    """chat_completions drops an intrinsic definition with no log line."""
    with caplog.at_level(logging.WARNING):
        caplog.clear()
        out = ChatCompletionsConverter().tool_ops.ir_tool_definition_to_p(
            intrinsic_definition(UNKNOWN_KIND)
        )
    assert out == {}
    assert caplog.records == []


# ---------------------------------------------------------------------------
# The matrix — history call write
# ---------------------------------------------------------------------------

HISTORY_CALL_MATRIX: dict[str, dict[str, str]] = {
    # No native server tools: every intrinsic call degrades to a function call
    # named the tool_name (the canonical kind for a server call).
    "chat_completions": {
        "web_search": "degraded",
        "code_execution": "degraded",
        UNKNOWN_KIND: "degraded",
    },
    # Native for a kind Anthropic has a server tool for; a foreign or unknown
    # kind falls back to a client ``tool_use``, never to a fabricated
    # ``server_tool_use`` name.
    "messages": {
        "web_search": "native",
        "code_execution": "native",
        "google_search": "degraded",
        UNKNOWN_KIND: "degraded",
    },
    # Native only for a renderable server item; everything else degrades.
    "open_responses": {
        "web_search": "native",
        "code_interpreter": "native",
        "code_execution": "degraded",
        "google_search": "degraded",
        UNKNOWN_KIND: "degraded",
    },
    # code_execution is Google's only inline server call; the rest drop.
    "google_generate": {
        "code_execution": "native",
        "google_search": "dropped",
        "web_search": "dropped",
        UNKNOWN_KIND: "dropped",
    },
    "google_interactions": {
        "google_search": "native",
        "code_execution": "native",
        "web_search": "dropped",
        UNKNOWN_KIND: "dropped",
    },
}


@pytest.mark.parametrize("converter,kind,expected", matrix_cases(HISTORY_CALL_MATRIX))
def test_history_call_write(converter: str, kind: str, expected: str):
    assert history_call_outcome(converter, kind) == expected


def test_open_responses_degrades_unrenderable_code_interpreter():
    """A code_interpreter call with no container_id cannot be replayed."""
    kind = {"code": "x"}  # no container_id
    assert history_call_outcome("open_responses", "code_interpreter") == "native"
    out = OpenAIResponsesConverter().request_to_provider(
        cast(IRRequest, ir_history("code_interpreter", tool_input=kind))
    )[0]
    assert '"function_call"' in json.dumps(out)
    assert "code_interpreter_call" not in json.dumps(out)


def test_google_generate_drops_foreign_call_with_warning():
    """The drop is reported on the conversion's own warnings channel."""
    _, warnings = GoogleGenerateConverter().request_to_provider(
        cast(IRRequest, ir_history("web_search"))
    )
    assert any("Unsupported intrinsic kind" in w for w in warnings)


def test_google_interactions_drops_foreign_call_silently(recwarn):
    body, _ = GoogleInteractionsConverter().request_to_provider(
        cast(IRRequest, ir_history("web_search"))
    )
    assert json.dumps(body, default=str) == json.dumps(
        {"model": "m", "input": "hello"}, default=str
    )
    assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]


def test_history_call_metadata_name_matches_the_kind():
    """Every native history write keeps the canonical kind as its identity."""
    body, _ = MessagesConverter().request_to_provider(
        cast(IRRequest, ir_history("web_search"))
    )
    block = body["messages"][1]["content"][0]
    assert block["type"] == "server_tool_use"
    assert block["name"] == "web_search"


# ---------------------------------------------------------------------------
# The matrix — history result write
# ---------------------------------------------------------------------------

HISTORY_RESULT_MATRIX: dict[str, dict[str, str]] = {
    # chat and messages write the result as an ordinary client result, so the
    # intrinsic kind does not survive.
    "chat_completions": {
        "web_search": "degraded",
        "code_execution": "degraded",
        UNKNOWN_KIND: "degraded",
    },
    # A server result Anthropic has a server tool for keeps it in the block
    # type (``web_search_tool_result``); a kind it has none for stays an
    # ordinary ``tool_result``.
    "messages": {
        "web_search": "native",
        "code_execution": "native",
        UNKNOWN_KIND: "degraded",
    },
    # A native server call takes no output item on the history leg, so its
    # result is dropped; a degraded call pairs with function_call_output.
    "open_responses": {
        "web_search": "dropped",
        "code_interpreter": "dropped",
        "code_execution": "degraded",
        "google_search": "degraded",
        UNKNOWN_KIND: "degraded",
    },
    "google_generate": {
        "code_execution": "native",
        "web_search": "dropped",
        "google_search": "dropped",
        UNKNOWN_KIND: "dropped",
    },
    "google_interactions": {
        "google_search": "native",
        "code_execution": "native",
        "web_search": "dropped",
        UNKNOWN_KIND: "dropped",
    },
}


@pytest.mark.parametrize("converter,kind,expected", matrix_cases(HISTORY_RESULT_MATRIX))
def test_history_result_write(converter: str, kind: str, expected: str):
    assert history_result_outcome(converter, kind) == expected


def test_messages_history_result_keeps_the_kind():
    """A role:"tool" result Anthropic has a server tool for keeps its kind.

    Phase 4 put the kind back: the writer routes a server result through the
    intrinsic writer on the history leg too, instead of flattening it to a
    kind-less ``tool_result``.  The IR carries it in a ``role:"tool"`` message;
    the writer moves it back into the assistant turn that holds its call,
    which is where Anthropic requires a server result (and where a same-dialect
    *read* keeps it).
    """
    body, _ = MessagesConverter().request_to_provider(
        cast(IRRequest, ir_history("web_search"))
    )
    assert len(body["messages"]) == 2
    result_block = body["messages"][1]["content"][1]
    assert result_block["type"] == "web_search_tool_result"
    assert result_block["tool_use_id"] == "srvtoolu_s1"


def test_messages_history_result_of_a_foreign_kind_is_a_plain_tool_result():
    """A kind Anthropic has no server tool for stays a client result."""
    body, _ = MessagesConverter().request_to_provider(
        cast(IRRequest, ir_history(UNKNOWN_KIND))
    )
    result_block = body["messages"][2]["content"][0]
    assert result_block["type"] == "tool_result"
    assert UNKNOWN_KIND not in json.dumps(result_block)


# ---------------------------------------------------------------------------
# The matrix — non-streaming response write
# ---------------------------------------------------------------------------

RESPONSE_CALL_MATRIX: dict[str, dict[str, str]] = {
    "chat_completions": {
        "web_search": "degraded",
        "code_execution": "degraded",
        UNKNOWN_KIND: "degraded",
    },
    # A server call is never handed to the client as a client call on the
    # response leg, so a kind Anthropic has no server tool for is dropped.
    "messages": {
        "web_search": "native",
        "code_execution": "native",
        "google_search": "dropped",
        UNKNOWN_KIND: "dropped",
    },
    "open_responses": {
        "web_search": "native",
        "code_interpreter": "native",
        "code_execution": "degraded",
        "google_search": "degraded",
        UNKNOWN_KIND: "degraded",
    },
    "google_generate": {
        "code_execution": "native",
        "google_search": "dropped",
        "web_search": "dropped",
        UNKNOWN_KIND: "dropped",
    },
    "google_interactions": {
        "google_search": "native",
        "code_execution": "native",
        "web_search": "dropped",
        UNKNOWN_KIND: "dropped",
    },
}

RESPONSE_RESULT_MATRIX: dict[str, dict[str, str]] = {
    # chat's response writer has no tool-result slot at all.
    "chat_completions": {
        "web_search": "dropped",
        "code_execution": "dropped",
        UNKNOWN_KIND: "dropped",
    },
    # messages writes {kind}_tool_result for a kind it has a server tool for.
    "messages": {
        "web_search": "native",
        "code_execution": "native",
        UNKNOWN_KIND: "dropped",
    },
    # The response writer emits the native output item for a kind that has
    # one, and drops the rest with a warning.
    "open_responses": {
        "web_search": "dropped",
        "code_interpreter": "native",
        "code_execution": "dropped",
        UNKNOWN_KIND: "dropped",
    },
    "google_generate": {
        "code_execution": "native",
        "google_search": "dropped",
        "web_search": "dropped",
        UNKNOWN_KIND: "dropped",
    },
    "google_interactions": {
        "google_search": "native",
        "code_execution": "native",
        "web_search": "dropped",
        UNKNOWN_KIND: "dropped",
    },
}


@pytest.mark.parametrize("converter,kind,expected", matrix_cases(RESPONSE_CALL_MATRIX))
def test_response_call_write(converter: str, kind: str, expected: str):
    assert response_call_outcome(converter, kind) == expected


@pytest.mark.parametrize(
    "converter,kind,expected", matrix_cases(RESPONSE_RESULT_MATRIX)
)
def test_response_result_write(converter: str, kind: str, expected: str):
    assert response_result_outcome(converter, kind) == expected


def test_open_responses_response_writes_the_result_part():
    """The response writer pairs a server call with its native output item."""
    out = OpenAIResponsesConverter().response_to_provider(
        cast(Any, ir_response("code_interpreter"))
    )
    types = [item["type"] for item in out["output"]]
    assert "code_interpreter_call" in types
    assert any(t.endswith("_call_output") for t in types)
    assert '"RESULT"' in json.dumps(out, default=str)


def test_messages_response_result_is_kind_named():
    out = MessagesConverter().response_to_provider(cast(Any, ir_response("web_search")))
    block = out["content"][1]
    assert block["type"] == "web_search_tool_result"
    assert block["tool_use_id"] == "srvtoolu_c1"


def test_google_generate_response_drops_foreign_kind_with_warning():
    """The drop is reported on the source converter's own warnings list."""
    ctx = ConversionContext()
    out = GoogleGenerateConverter().response_to_provider(
        cast(Any, ir_response("web_search")), context=ctx
    )
    assert "_RESULT_" not in json.dumps(out)
    assert any("Unsupported intrinsic kind" in w for w in ctx.warnings)
