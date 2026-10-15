"""Intrinsic handling matrix — streaming read and write paths.

Pins the CURRENT per-converter × kind behaviour of the streaming paths:

* **read** — a provider stream chunk carrying a native server call/result is
  turned into IR ``tool_call_start`` / ``tool_call_delta`` / ``tool_result``
  events, and the canonical kind is carried on them.
* **write** — an IR intrinsic event is emitted back into a provider chunk.

The write tables classify each cell as ``native``, ``degraded``, ``dropped`` or
``deferred`` (google_generate flushes an intrinsic call at the finish event, not
at its start).  Cells the refactor will change carry ``xfail(strict=True)``
companions.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from llm_rosetta.converters.base.tools.intrinsic import get_intrinsic_kind
from llm_rosetta.converters.google_generate import GoogleGenerateConverter
from llm_rosetta.converters.messages import MessagesConverter
from llm_rosetta.converters.open_responses import OpenResponsesConverter
from llm_rosetta.types.ir.stream import ToolCallDeltaEvent, ToolCallStartEvent
from llm_rosetta.types.ir.stream import ToolResultEvent

from .conftest import CONVERTERS, UNKNOWN_KIND, matrix_cases

# ---------------------------------------------------------------------------
# Native stream chunks, one entry per (converter, kind)
# ---------------------------------------------------------------------------

_CALL_CHUNKS: dict[tuple[str, str], list[dict[str, Any]]] = {
    ("messages", "web_search"): [
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "server_tool_use",
                "id": "c1",
                "name": "web_search",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"queries": ["q"]}'},
        },
    ],
    ("messages", "code_execution"): [
        {
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "server_tool_use",
                "id": "c1",
                "name": "code_execution",
                "input": {},
            },
        },
        {
            "type": "content_block_delta",
            "index": 1,
            "delta": {"type": "input_json_delta", "partial_json": '{"code": "x"}'},
        },
    ],
    ("open_responses", "web_search"): [
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {"type": "web_search_call", "id": "c1", "status": "in_progress"},
        },
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {
                "type": "web_search_call",
                "id": "c1",
                "status": "completed",
                "action": {"type": "search", "query": "q"},
            },
        },
    ],
    ("open_responses", "code_interpreter"): [
        {
            "type": "response.output_item.added",
            "output_index": 1,
            "item": {
                "type": "code_interpreter_call",
                "id": "c1",
                "status": "in_progress",
            },
        },
        {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {
                "type": "code_interpreter_call",
                "id": "c1",
                "status": "completed",
                "code": "print(1)",
                "container_id": "cntr_1",
                "outputs": [{"type": "logs", "logs": "1"}],
            },
        },
    ],
    ("google_generate", "code_execution"): [
        {
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "executableCode": {
                                    "code": "print(1)",
                                    "language": "PYTHON",
                                }
                            }
                        ],
                    },
                }
            ]
        }
    ],
    ("google_interactions", "google_search"): [
        {
            "event_type": "step.start",
            "index": 0,
            "step": {"id": "c1", "signature": "", "type": "google_search_call"},
        },
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {
                "signature": "S",
                "type": "google_search_call",
                "arguments": {"queries": ["q"]},
            },
        },
    ],
    ("google_interactions", "code_execution"): [
        {
            "event_type": "step.start",
            "index": 0,
            "step": {"id": "c1", "signature": "", "type": "code_execution_call"},
        },
        {
            "event_type": "step.delta",
            "index": 0,
            "delta": {
                "signature": "S",
                "type": "code_execution_call",
                "arguments": {"code": "x"},
            },
        },
    ],
}

_RESULT_CHUNKS: dict[tuple[str, str], list[dict[str, Any]]] = {
    ("messages", "web_search"): [
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": "c1",
                "content": [{"type": "web_search_result", "url": "u"}],
            },
        }
    ],
    ("messages", "code_execution"): [
        {
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "code_execution_tool_result",
                "tool_use_id": "c1",
                "content": {
                    "type": "code_execution_result",
                    "stdout": "1\n",
                    "stderr": "",
                    "return_code": 0,
                },
            },
        }
    ],
    ("open_responses", "code_interpreter"): [
        {
            "type": "response.output_item.added",
            "output_index": 2,
            "item": {
                "type": "code_interpreter_call_output",
                "call_id": "c1",
                "output": "1",
            },
        }
    ],
    ("google_generate", "code_execution"): [
        {
            "candidates": [
                {
                    "index": 0,
                    "content": {
                        "role": "model",
                        "parts": [
                            {
                                "codeExecutionResult": {
                                    "output": "1",
                                    "outcome": "OUTCOME_OK",
                                }
                            }
                        ],
                    },
                }
            ]
        }
    ],
    ("google_interactions", "google_search"): [
        {
            "event_type": "step.start",
            "index": 1,
            "step": {"call_id": "c1", "signature": "", "type": "google_search_result"},
        },
        {
            "event_type": "step.delta",
            "index": 1,
            "delta": {
                "signature": "S2",
                "type": "google_search_result",
                "result": [{"rendered_content": "r"}],
            },
        },
    ],
    ("google_interactions", "code_execution"): [
        {
            "event_type": "step.start",
            "index": 1,
            "step": {
                "call_id": "c1",
                "signature": "",
                "type": "code_execution_result",
            },
        },
        {
            "event_type": "step.delta",
            "index": 1,
            "delta": {
                "signature": "S2",
                "type": "code_execution_result",
                "result": "1\n",
            },
        },
    ],
}

READ_CALL_MATRIX: dict[str, dict[str, str]] = {
    "messages": {"web_search": "native", "code_execution": "native"},
    "open_responses": {"web_search": "native", "code_interpreter": "native"},
    "google_generate": {"code_execution": "native"},
    "google_interactions": {"google_search": "native", "code_execution": "native"},
}

READ_RESULT_MATRIX: dict[str, dict[str, str]] = {
    "messages": {"web_search": "native", "code_execution": "native"},
    "open_responses": {"code_interpreter": "native"},
    "google_generate": {"code_execution": "native"},
    "google_interactions": {"google_search": "native", "code_execution": "native"},
}


def _read(converter_name: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converter = CONVERTERS[converter_name]()
    ctx = converter.create_stream_context()
    events: list[dict[str, Any]] = []
    for chunk in chunks:
        events.extend(converter.stream_response_from_provider(chunk, ctx) or [])
    return events


def read_call_kind(converter_name: str, kind: str) -> str | None:
    for event in _read(converter_name, _CALL_CHUNKS[(converter_name, kind)]):
        if event.get("type") == "tool_call_start":
            return get_intrinsic_kind(event) or event.get("tool_name")
    return None


def read_result_kind(converter_name: str, kind: str) -> str | None:
    for event in _read(converter_name, _RESULT_CHUNKS[(converter_name, kind)]):
        if event.get("type") == "tool_result":
            return get_intrinsic_kind(event)
    return None


# google_generate warns while dropping an unrenderable kind; the tests
# that pin the warning use pytest.warns, which sets its own filter.
pytestmark = pytest.mark.filterwarnings("ignore:Unsupported intrinsic kind")


# ---------------------------------------------------------------------------
# Stream read
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("converter,kind,expected", matrix_cases(READ_CALL_MATRIX))
def test_stream_read_call(converter: str, kind: str, expected: str):
    assert read_call_kind(converter, kind) == kind


@pytest.mark.parametrize("converter,kind,expected", matrix_cases(READ_RESULT_MATRIX))
def test_stream_read_result(converter: str, kind: str, expected: str):
    assert read_result_kind(converter, kind) == kind


def test_read_call_event_is_marked_intrinsic():
    (start,) = [
        e
        for e in _read("messages", _CALL_CHUNKS[("messages", "web_search")])
        if e["type"] == "tool_call_start"
    ]
    assert start["tool_type"] == "intrinsic"
    assert start["tool_name"] == "web_search"


def test_open_responses_reader_reads_no_foreign_dialect_chunk():
    """A chunk from another dialect yields no intrinsic events here."""
    events = _read("open_responses", _CALL_CHUNKS[("messages", "web_search")])
    assert not [e for e in events if e.get("type") == "tool_call_start"]


def test_google_generate_reader_ignores_foreign_chunk():
    events = _read(
        "open_responses", _CALL_CHUNKS[("google_generate", "code_execution")]
    )
    assert not [e for e in events if e.get("type") == "tool_call_start"]


def test_google_generate_pairs_streamed_result_positionally():
    """Google's codeExecutionResult carries no id; it pairs by position."""
    events = _read(
        "google_generate",
        _CALL_CHUNKS[("google_generate", "code_execution")]
        + _RESULT_CHUNKS[("google_generate", "code_execution")],
    )
    calls = [e for e in events if e["type"] == "tool_call_start"]
    results = [e for e in events if e["type"] == "tool_result"]
    assert len(calls) == 1 and len(results) == 1
    assert results[0]["tool_call_id"] == calls[0]["tool_call_id"]


def test_read_delta_carries_no_metadata_where_the_block_carried_none():
    """CURRENT: a caller-less server block leaves its delta nothing to carry.

    A ``server_tool_use`` short of a non-``direct`` caller registers no
    metadata, so its arguments delta has exactly what the start had: nothing.
    """
    delta = next(
        e
        for e in _read("messages", _CALL_CHUNKS[("messages", "web_search")])
        if e["type"] == "tool_call_delta"
    )
    assert "provider_metadata" not in delta


def test_read_delta_carries_the_metadata_the_block_registered():
    """A server call's delta repeats the native part its start carried."""
    delta = next(
        e
        for e in _read(
            "google_generate", _CALL_CHUNKS[("google_generate", "code_execution")]
        )
        if e["type"] == "tool_call_delta"
    )
    assert delta["provider_metadata"]["_native"]["base"] == "google_generate"


def test_read_delta_carries_the_intrinsic_kind():
    """The delta carries the kind so a foreign writer can render it.

    It is read back from what the call registered at its start: a fragment of
    arguments has no identity of its own.
    """
    delta = next(
        e
        for e in _read("messages", _CALL_CHUNKS[("messages", "web_search")])
        if e["type"] == "tool_call_delta"
    )
    assert get_intrinsic_kind(delta) == "web_search"


# ---------------------------------------------------------------------------
# Stream write
# ---------------------------------------------------------------------------

WRITE_CALL_MATRIX: dict[str, dict[str, str]] = {
    "chat_completions": {
        "web_search": "degraded",
        "code_execution": "degraded",
        UNKNOWN_KIND: "degraded",
    },
    "messages": {
        "web_search": "native",
        "code_execution": "native",
        "google_search": "native",
        UNKNOWN_KIND: "native",
    },
    "open_responses": {
        "web_search": "native",
        "code_interpreter": "native",
        "code_execution": "dropped",
        "google_search": "dropped",
        UNKNOWN_KIND: "dropped",
    },
    # The start event is always empty here; the call is written at finish.
    "google_generate": {
        "code_execution": "deferred",
        "web_search": "deferred",
        UNKNOWN_KIND: "deferred",
    },
    "google_interactions": {
        "google_search": "native",
        "code_execution": "native",
        "web_search": "dropped",
        UNKNOWN_KIND: "dropped",
    },
}

WRITE_RESULT_MATRIX: dict[str, dict[str, str]] = {
    "chat_completions": {
        "web_search": "dropped",
        "code_execution": "dropped",
        UNKNOWN_KIND: "dropped",
    },
    "messages": {
        "web_search": "native",
        "code_execution": "native",
        UNKNOWN_KIND: "native",
    },
    "open_responses": {
        "web_search": "dropped",
        "code_interpreter": "native",
        "code_execution": "dropped",
        UNKNOWN_KIND: "dropped",
    },
    "google_generate": {
        "code_execution": "native",
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

_NATIVE_CALL_MARKERS: dict[str, tuple[str, ...]] = {
    "messages": ('"server_tool_use"',),
    "open_responses": ('"web_search_call"', '"code_interpreter_call"'),
    "google_interactions": ('"google_search_call"', '"code_execution_call"'),
}


def _start_event(kind: str) -> ToolCallStartEvent:
    return cast(
        ToolCallStartEvent,
        {
            "type": "tool_call_start",
            "tool_call_id": "c1",
            "tool_name": kind,
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": kind},
        },
    )


def write_call_outcome(converter_name: str, kind: str) -> str:
    converter = CONVERTERS[converter_name]()
    out = converter.stream_response_to_provider(
        _start_event(kind), converter.create_stream_context()
    )
    text = json.dumps(out, default=str)
    if converter_name == "google_generate":
        return "deferred" if not text.strip("{}[] ,") else "native"
    if converter_name == "chat_completions":
        return "degraded" if '"function"' in text else "dropped"
    if any(m in text for m in _NATIVE_CALL_MARKERS[converter_name]):
        return "native"
    return "dropped"


def write_result_outcome(converter_name: str, kind: str) -> str:
    converter = CONVERTERS[converter_name]()
    event = cast(
        ToolResultEvent,
        {
            "type": "tool_result",
            "tool_call_id": "c1",
            "result": "R",
            "tool_type": "intrinsic",
            "provider_metadata": {"intrinsic_kind": kind},
        },
    )
    out = converter.stream_response_to_provider(
        event, converter.create_stream_context()
    )
    text = json.dumps(out, default=str)
    if converter_name == "chat_completions":
        return "degraded" if '"choices"' in text else "dropped"
    if converter_name == "messages":
        return "native" if "_tool_result" in text else "dropped"
    if converter_name == "open_responses":
        return "native" if text not in ("{}", "[]") else "dropped"
    if converter_name == "google_generate":
        return "native" if "codeExecutionResult" in text else "dropped"
    if converter_name == "google_interactions":
        return "native" if "_result" in text else "dropped"
    return "dropped"


@pytest.mark.parametrize("converter,kind,expected", matrix_cases(WRITE_CALL_MATRIX))
def test_stream_write_call_start(converter: str, kind: str, expected: str):
    assert write_call_outcome(converter, kind) == expected


@pytest.mark.parametrize("converter,kind,expected", matrix_cases(WRITE_RESULT_MATRIX))
def test_stream_write_result(converter: str, kind: str, expected: str):
    assert write_result_outcome(converter, kind) == expected


WRITE_DELTA_MATRIX: dict[str, dict[str, str]] = {
    "chat_completions": {
        "web_search": "degraded",
        "code_execution": "degraded",
        UNKNOWN_KIND: "degraded",
    },
    "messages": {
        "web_search": "native",
        "code_execution": "native",
        UNKNOWN_KIND: "native",
    },
    # A native server item holds its payload for the done event, and a call
    # whose start was dropped takes its deltas with it: nothing is written for
    # either, so both are "held" rather than a leaked half-write.
    "open_responses": {
        "web_search": "held",
        "code_interpreter": "held",
        "code_execution": "held",
        UNKNOWN_KIND: "held",
    },
    "google_generate": {
        "code_execution": "deferred",
        UNKNOWN_KIND: "deferred",
    },
    "google_interactions": {
        "google_search": "native",
        "code_execution": "native",
        "web_search": "held",
        UNKNOWN_KIND: "held",
    },
}


def write_delta_outcome(converter_name: str, kind: str) -> str:
    """Classify the arguments-delta write, after the matching call start."""
    converter = CONVERTERS[converter_name]()
    ctx = converter.create_stream_context()
    converter.stream_response_to_provider(_start_event(kind), ctx)
    delta = converter.stream_response_to_provider(
        cast(
            ToolCallDeltaEvent,
            {
                "type": "tool_call_delta",
                "tool_call_id": "c1",
                "arguments_delta": json.dumps({"query": "x"}),
            },
        ),
        ctx,
    )
    text = json.dumps(delta, default=str)
    if converter_name == "google_generate":
        return "deferred" if text == "{}" else "native"
    if converter_name == "chat_completions":
        return "degraded" if '"function"' in text else "dropped"
    if converter_name == "messages":
        return "native" if "input_json_delta" in text else "dropped"
    if converter_name == "open_responses":
        return "held" if text == "{}" else "leaked"
    if converter_name == "google_interactions":
        # A native delta carries the call's own step type; an empty one is a
        # delta whose call was dropped with its start (nothing to attach to).
        # Anything else is a bare ``arguments`` step the converter should not
        # have produced.
        if text in ("{}", "[]"):
            return "held"
        return "native" if "_call" in text else "leaked"
    raise AssertionError(converter_name)


@pytest.mark.parametrize("converter,kind,expected", matrix_cases(WRITE_DELTA_MATRIX))
def test_stream_write_call_delta(converter: str, kind: str, expected: str):
    assert write_delta_outcome(converter, kind) == expected


def test_google_generate_writes_the_call_at_finish():
    """The deferred write lands natively for code_execution only."""
    converter = GoogleGenerateConverter()
    ctx = converter.create_stream_context()
    converter.stream_response_to_provider(_start_event("code_execution"), ctx)
    converter.stream_response_to_provider(
        cast(
            ToolCallDeltaEvent,
            {
                "type": "tool_call_delta",
                "tool_call_id": "c1",
                "arguments_delta": json.dumps({"code": "x", "language": "PYTHON"}),
            },
        ),
        ctx,
    )
    out = converter.stream_response_to_provider(
        {"type": "finish", "finish_reason": {"reason": "tool_calls"}}, ctx
    )
    assert "executableCode" in json.dumps(out, default=str)


def test_google_generate_drops_foreign_streamed_call_at_finish():
    converter = GoogleGenerateConverter()
    ctx = converter.create_stream_context()
    converter.stream_response_to_provider(_start_event("web_search"), ctx)
    out = converter.stream_response_to_provider(
        {"type": "finish", "finish_reason": {"reason": "tool_calls"}}, ctx
    )
    assert "functionCall" not in json.dumps(out, default=str)
    assert "executableCode" not in json.dumps(out, default=str)


def test_google_generate_stream_result_drop_is_silent(recwarn):
    """CURRENT: a non-code_execution intrinsic result is dropped with no warning."""
    converter = GoogleGenerateConverter()
    converter.stream_response_to_provider(
        cast(
            ToolResultEvent,
            {
                "type": "tool_result",
                "tool_call_id": "c1",
                "result": "R",
                "tool_type": "intrinsic",
                "provider_metadata": {"intrinsic_kind": "web_search"},
            },
        ),
        converter.create_stream_context(),
    )
    assert not [w for w in recwarn.list if issubclass(w.category, UserWarning)]


@pytest.mark.xfail(
    strict=True, reason="intrinsic refactor phase 4: google_generate silent stream drop"
)
def test_google_generate_stream_result_drop_should_warn():
    with pytest.warns(UserWarning):
        GoogleGenerateConverter().stream_response_to_provider(
            cast(
                ToolResultEvent,
                {
                    "type": "tool_result",
                    "tool_call_id": "c1",
                    "result": "R",
                    "tool_type": "intrinsic",
                    "provider_metadata": {"intrinsic_kind": "web_search"},
                },
            ),
            GoogleGenerateConverter().create_stream_context(),
        )


def test_open_responses_drops_foreign_call_start_with_its_delta():
    """A dropped start takes its arguments delta with it.

    A kind this dialect cannot render as a server item opens no item, so the
    fragments that follow belong to nothing: writing them would emit a
    ``function_call_arguments.delta`` for a call no item was announced for.
    """
    converter = OpenResponsesConverter()
    ctx = converter.create_stream_context()
    start = converter.stream_response_to_provider(_start_event("code_execution"), ctx)
    delta = converter.stream_response_to_provider(
        cast(
            ToolCallDeltaEvent,
            {
                "type": "tool_call_delta",
                "tool_call_id": "c1",
                "arguments_delta": json.dumps({"query": "x"}),
            },
        ),
        ctx,
    )
    assert start == {}
    assert delta == {}


def test_open_responses_foreign_delta_should_not_leak():
    converter = OpenResponsesConverter()
    ctx = converter.create_stream_context()
    converter.stream_response_to_provider(_start_event("code_execution"), ctx)
    delta = converter.stream_response_to_provider(
        cast(
            ToolCallDeltaEvent,
            {
                "type": "tool_call_delta",
                "tool_call_id": "c1",
                "arguments_delta": json.dumps({"query": "x"}),
            },
        ),
        ctx,
    )
    assert delta == {}


def test_open_responses_finish_writes_no_item_for_a_dropped_call():
    """A dropped intrinsic start leaves nothing for the flush to rebuild.

    The arguments were never accumulated, so the finish flush has no call to
    write — rather than the nameless ``function_call`` a leaked delta used to
    leave it with.
    """
    converter = OpenResponsesConverter()
    ctx = converter.create_stream_context()
    converter.stream_response_to_provider(_start_event("code_execution"), ctx)
    converter.stream_response_to_provider(
        cast(
            ToolCallDeltaEvent,
            {
                "type": "tool_call_delta",
                "tool_call_id": "c1",
                "arguments_delta": json.dumps({"query": "x"}),
            },
        ),
        ctx,
    )
    out = converter.stream_response_to_provider(
        {"type": "finish", "finish_reason": {"reason": "tool_calls"}}, ctx
    )
    items = [
        e["item"]
        for e in out
        if isinstance(e, dict) and e.get("type") == "response.output_item.done"
    ]
    assert not [item for item in items if item.get("type") == "function_call"]


def test_messages_stream_writes_server_tool_use_for_any_kind():
    converter = MessagesConverter()
    out = cast(
        dict[str, Any],
        converter.stream_response_to_provider(
            _start_event(UNKNOWN_KIND), converter.create_stream_context()
        ),
    )
    assert out["content_block"]["type"] == "server_tool_use"
    assert out["content_block"]["name"] == UNKNOWN_KIND


# ---------------------------------------------------------------------------
# Kind read-back: provider_metadata vs the tool name
# ---------------------------------------------------------------------------


def _kindless_start(kind_name: str) -> ToolCallStartEvent:
    """An intrinsic start event carrying NO provider_metadata."""
    return cast(
        ToolCallStartEvent,
        {
            "type": "tool_call_start",
            "tool_call_id": "c1",
            "tool_name": kind_name,
            "tool_type": "intrinsic",
        },
    )


def test_messages_stream_kind_missing_renders_an_empty_name():
    """Phase 2: a kindless start is a server call with no name, not the name."""
    converter = MessagesConverter()
    out = cast(
        dict[str, Any],
        converter.stream_response_to_provider(
            _kindless_start("web_search"), converter.create_stream_context()
        ),
    )
    assert out["content_block"]["type"] == "server_tool_use"
    assert out["content_block"]["name"] == ""


def test_open_responses_stream_kind_should_not_fall_back_to_the_tool_name():
    """An intrinsic kind comes only from ``intrinsic_kind``, never the name."""
    converter = OpenResponsesConverter()
    out = converter.stream_response_to_provider(
        _kindless_start("web_search"), converter.create_stream_context()
    )
    assert "web_search_call" not in json.dumps(out, default=str)
