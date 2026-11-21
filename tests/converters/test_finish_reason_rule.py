"""One rule for a turn that finished normally while owing the client a result.

Anthropic and the Interactions API report a tool call in their own stop reason;
Responses and Google generateContent do not, so the IR decides from the parts.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from llm_rosetta.converters.base import StreamContext
from llm_rosetta.converters.llm.chat_completions import ChatCompletionsConverter
from llm_rosetta.converters.llm.google_generate import GoogleGenerateConverter
from llm_rosetta.converters.llm.google_interactions import GoogleInteractionsConverter
from llm_rosetta.converters.llm.messages import MessagesConverter
from llm_rosetta.converters.llm.open_responses import OpenResponsesConverter
from llm_rosetta.ir.primitives.finish_reason import (
    client_tool_calls_in,
    reason_for_completed_turn,
)
from llm_rosetta.pipeline import StreamProcessor

CLIENT_CALL = {
    "type": "tool_call",
    "tool_call_id": "c1",
    "tool_name": "f",
    "tool_input": {},
}
SERVER_CALL = {**CLIENT_CALL, "tool_type": "intrinsic"}


@pytest.mark.parametrize(
    ("reason", "parts", "expected"),
    [
        ("stop", [CLIENT_CALL], "tool_calls"),
        ("stop", [SERVER_CALL], "stop"),
        ("stop", [{"type": "text", "text": "hi"}], "stop"),
        ("length", [CLIENT_CALL], "length"),
        ("content_filter", [CLIENT_CALL], "content_filter"),
        ("error", [CLIENT_CALL], "error"),
    ],
)
def test_the_rule_only_touches_a_normal_stop(reason, parts, expected):
    assert reason_for_completed_turn(reason, client_tool_calls_in(parts)) == expected


def _google_response(*parts: dict[str, Any]) -> dict[str, Any]:
    return {
        "candidates": [
            {
                "content": {"role": "model", "parts": list(parts)},
                "finishReason": "STOP",
            }
        ],
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": 2,
            "totalTokenCount": 3,
        },
    }


def _google_response_with_reason(reason: str) -> dict[str, Any]:
    response = _google_response({"functionCall": {"name": "f", "args": {}}})
    response["candidates"][0]["finishReason"] = reason
    return response


def test_a_google_function_call_is_a_tool_call_turn():
    ir = GoogleGenerateConverter().response_from_provider(
        cast(Any, _google_response({"functionCall": {"name": "f", "args": {}}}))
    )

    assert ir["choices"][0]["finish_reason"]["reason"] == "tool_calls"
    chat = ChatCompletionsConverter().response_to_provider(ir)
    assert chat["choices"][0]["finish_reason"] == "tool_calls"


def test_a_google_text_turn_still_finishes_normally():
    ir = GoogleGenerateConverter().response_from_provider(
        cast(Any, _google_response({"text": "done"}))
    )

    assert ir["choices"][0]["finish_reason"]["reason"] == "stop"


def test_a_responses_function_call_is_a_tool_call_turn():
    response = {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "m",
        "status": "completed",
        "output": [
            {
                "type": "function_call",
                "status": "completed",
                "call_id": "c1",
                "name": "f",
                "arguments": "{}",
            }
        ],
    }

    ir = OpenResponsesConverter().response_from_provider(cast(Any, response))

    assert ir["choices"][0]["finish_reason"]["reason"] == "tool_calls"


def _responses_call_added(call_id: str = "call_a") -> dict[str, Any]:
    """The frame a Responses stream opens a client call with."""
    return {
        "type": "response.output_item.added",
        "output_index": 0,
        "item": {
            "type": "function_call",
            "id": "fc_1",
            "call_id": call_id,
            "name": "f",
            "arguments": "",
        },
    }


def _responses_completed(output: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """The frame a Responses stream ends with; ``output`` is optional."""
    response: dict[str, Any] = {
        "id": "resp_1",
        "model": "m",
        "created_at": 1,
        "status": "completed",
    }
    if output is not None:
        response["output"] = output
    return {"type": "response.completed", "response": response}


def _run_responses_stream(*frames: dict[str, Any]) -> list[Any]:
    converter = OpenResponsesConverter()
    context = StreamContext()
    events: list[Any] = []
    for frame in frames:
        events.extend(converter.stream_response_from_provider(frame, context))
    return events


def _messages_view(ir_events: list[Any]) -> tuple[list[dict[str, Any]], list[str]]:
    """Write IR events as Messages; report the blocks and stop reasons sent."""
    converter = MessagesConverter()
    context = StreamContext()
    blocks: list[dict[str, Any]] = []
    stop_reasons: list[str] = []
    for event in ir_events:
        produced = converter.stream_response_to_provider(event, context)
        for chunk in produced if isinstance(produced, list) else [produced]:
            if not chunk:
                continue
            if chunk.get("type") == "content_block_start":
                blocks.append(chunk["content_block"])
            elif chunk.get("type") == "message_delta":
                stop_reasons.append(chunk["delta"]["stop_reason"])
    return blocks, stop_reasons


def _finish_reason(ir_events: list[Any]) -> str | None:
    finishes = [event for event in ir_events if event["type"] == "finish"]
    return finishes[-1]["finish_reason"]["reason"] if finishes else None


def test_a_responses_stream_without_an_output_list_is_a_tool_call_turn():
    """A completed frame need not repeat the call the stream already opened.

    ``response.completed`` is the only frame this dialect's reader decides the
    reason from, and it may carry no ``output`` list at all; the call is then
    known only from the earlier ``response.output_item.added`` (#1601).
    """
    ir_events = _run_responses_stream(_responses_call_added(), _responses_completed())

    assert _finish_reason(ir_events) == "tool_calls"

    # A Messages client must be asked to run the tool, not told the reply is
    # complete — the symptom the issue measured.
    blocks, stop_reasons = _messages_view(ir_events)
    assert [(block["type"], block["id"]) for block in blocks] == [
        ("tool_use", "call_a")
    ]
    assert stop_reasons and stop_reasons[-1] == "tool_use"


def test_a_responses_frame_that_contradicts_the_opened_call_still_runs_it():
    """The opened call outranks a frame that says nothing about it.

    A Messages stream cannot retract the ``tool_use`` block it already sent,
    so a frame whose ``output`` list carries only a message must not tell the
    client ``end_turn``.  All four chat readers behave this way.
    """
    ir_events = _run_responses_stream(
        _responses_call_added(),
        _responses_completed(
            [{"type": "message", "content": [{"type": "output_text", "text": "hi"}]}]
        ),
    )

    assert _finish_reason(ir_events) == "tool_calls"
    blocks, stop_reasons = _messages_view(ir_events)
    assert [(block["type"], block["id"]) for block in blocks] == [
        ("tool_use", "call_a")
    ]
    assert stop_reasons and stop_reasons[-1] == "tool_use"


def test_a_responses_hosted_call_only_turn_still_finishes_normally():
    """A provider-executed call is not something the client has to run.

    Widening the reader's call set to the hosted item types would flip this
    turn to ``tool_calls`` and tell a Messages client to run a tool it never
    declared.
    """
    ir_events = _run_responses_stream(
        _responses_completed(
            [{"type": "web_search_call", "id": "ws_1", "status": "completed"}]
        )
    )

    assert _finish_reason(ir_events) == "stop"
    blocks, stop_reasons = _messages_view(ir_events)
    assert [block["type"] for block in blocks] == []
    assert stop_reasons and stop_reasons[-1] == "end_turn"


def test_a_responses_tool_turn_survives_the_stream_processor():
    """The serving path, not just converter to converter.

    ``StreamProcessor`` wraps both readers and adds the deferred flush, the
    transforms and the ``on_ir_event`` callback; a regression that appears
    only there would not be caught by the converter-level case.
    """
    observed: list[Any] = []
    processor = StreamProcessor(
        target_converter=OpenResponsesConverter(),
        source_converter=MessagesConverter(),
        from_ctx=OpenResponsesConverter.create_stream_context(),
        to_ctx=MessagesConverter.create_stream_context(),
        on_ir_event=observed.append,
    )

    produced: list[dict[str, Any]] = []
    produced.extend(processor.process_chunk(_responses_call_added()))
    produced.extend(processor.process_chunk(_responses_completed()))
    produced.extend(processor.process_chunk(None))

    blocks = [
        chunk["content_block"]
        for chunk in produced
        if chunk.get("type") == "content_block_start"
    ]
    stop_reasons = [
        chunk["delta"]["stop_reason"]
        for chunk in produced
        if chunk.get("type") == "message_delta"
    ]
    assert [(block["type"], block["id"]) for block in blocks] == [
        ("tool_use", "call_a")
    ]
    assert stop_reasons and stop_reasons[-1] == "tool_use"

    # An observer sees the upgraded finish, not the pre-upgrade one.
    assert [event for event in observed if event["type"] == "finish"][-1][
        "finish_reason"
    ]["reason"] == "tool_calls"


def test_the_iterator_reports_the_same_reason_without_a_context():
    """The documented entry point must not answer differently from the route.

    ``iter_stream_from_provider`` builds its own context when the caller
    passes none, so a caller following the docstring gets the tool turn the
    pipeline reports (#1601).
    """
    events = list(
        OpenResponsesConverter().iter_stream_from_provider(
            [_responses_call_added(), _responses_completed()]
        )
    )

    assert _finish_reason(events) == "tool_calls"


@pytest.mark.parametrize(
    ("finish_reason", "expected"),
    [
        ("MAX_TOKENS", "length"),
        ("SAFETY", "content_filter"),
        ("MALFORMED_FUNCTION_CALL", "error"),
        ("LANGUAGE", "error"),
    ],
)
def test_a_google_call_does_not_overwrite_a_non_stop_reason(finish_reason, expected):
    """A malformed call is still an error: the client must not be asked to run it."""
    ir = GoogleGenerateConverter().response_from_provider(
        cast(Any, _google_response_with_reason(finish_reason))
    )

    assert ir["choices"][0]["finish_reason"]["reason"] == expected


def _run_google_stream(*chunks: dict[str, Any]) -> list[Any]:
    converter = GoogleGenerateConverter()
    context = StreamContext()
    events: list[Any] = []
    for chunk in chunks:
        events.extend(converter.stream_response_from_provider(chunk, context))
    return events


def test_a_google_stream_finish_carries_the_same_reason():
    """Google finishes in a chunk with no content, after the call's own chunk."""
    events = _run_google_stream(
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": [{"text": "checking"}]},
                    "index": 0,
                }
            ]
        },
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"functionCall": {"name": "f", "args": {}}}],
                    },
                    "index": 0,
                }
            ]
        },
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": []},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ]
        },
    )

    finishes = [event for event in events if event["type"] == "finish"]
    assert finishes and finishes[-1]["finish_reason"]["reason"] == "tool_calls"


def test_a_google_stream_of_text_only_still_finishes_normally():
    events = _run_google_stream(
        {
            "candidates": [
                {"content": {"role": "model", "parts": [{"text": "hi"}]}, "index": 0}
            ]
        },
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": []},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ]
        },
    )

    finishes = [event for event in events if event["type"] == "finish"]
    assert finishes and finishes[-1]["finish_reason"]["reason"] == "stop"


def test_an_interactions_function_call_is_a_tool_call_turn():
    """The API normally says ``requires_action``; when it says ``completed``
    the steps still make it a tool-call turn."""
    response = {
        "id": "int_1",
        "object": "interaction",
        "created": "2026-10-26T00:00:00Z",
        "model": "m",
        "status": "completed",
        "steps": [
            {"type": "function_call", "id": "c1", "name": "f", "arguments": {}},
        ],
    }

    ir = GoogleInteractionsConverter().response_from_provider(cast(Any, response))

    assert ir["choices"][0]["finish_reason"]["reason"] == "tool_calls"


def test_an_interactions_requires_action_is_unchanged():
    response = {
        "id": "int_2",
        "object": "interaction",
        "created": "2026-10-26T00:00:00Z",
        "model": "m",
        "status": "requires_action",
        "steps": [
            {"type": "function_call", "id": "c1", "name": "f", "arguments": {}},
        ],
    }

    ir = GoogleInteractionsConverter().response_from_provider(cast(Any, response))

    assert ir["choices"][0]["finish_reason"]["reason"] == "tool_calls"


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        # A hosted call the provider ran itself completes the turn; the same
        # exchange without request context falls back to the name table.
        ("googleSearch", "stop"),
        ("f", "tool_calls"),
    ],
    ids=["hosted-call", "client-call"],
)
def test_a_google_exchange_finishes_the_same_way_in_json_and_stream(name, expected):
    """The same logical exchange, read from either Google dialect.

    generateContent answers in one JSON body and as a chunk stream; the two
    readers must not disagree about a turn, which is the defect family here
    (#1227).  No request leg is recorded, so both take the same name-table
    fallback.
    """
    ir = GoogleGenerateConverter().response_from_provider(
        cast(Any, _google_response({"functionCall": {"name": name, "args": {}}}))
    )
    assert ir["choices"][0]["finish_reason"]["reason"] == expected

    events = _run_google_stream(
        {
            "candidates": [
                {
                    "content": {
                        "role": "model",
                        "parts": [{"functionCall": {"name": name, "args": {}}}],
                    },
                    "index": 0,
                }
            ]
        },
        {
            "candidates": [
                {
                    "content": {"role": "model", "parts": []},
                    "finishReason": "STOP",
                    "index": 0,
                }
            ]
        },
    )

    finishes = [event for event in events if event["type"] == "finish"]
    assert finishes and finishes[-1]["finish_reason"]["reason"] == expected


def test_an_interactions_status_follows_the_ir_reason():
    """A ``stop`` turn keeps ``completed`` even with a function-call step.

    The writer used to force ``requires_action`` from the steps, which would
    overturn a correct reason — including a provider-hosted call, which is
    ``stop`` by design and must not ask the client to run it.
    """
    from llm_rosetta.converters.llm.google_interactions import (
        GoogleInteractionsConverter,
    )

    ir = {
        "id": "int_3",
        "object": "response",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "c1",
                            "tool_name": "googleSearch",
                            "tool_input": {},
                        }
                    ],
                },
                "finish_reason": {"reason": "stop"},
            }
        ],
    }

    out = GoogleInteractionsConverter().response_to_provider(cast(Any, ir))

    assert out["status"] == "completed"


# ==================== Chat and Messages: the same rule, both legs ====================


def test_a_chat_stop_with_client_calls_is_a_tool_call_turn():
    """A compatible Chat upstream may answer ``stop`` with tool calls."""
    response = {
        "id": "c1",
        "object": "chat.completion",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {"name": "f", "arguments": "{}"},
                        }
                    ],
                },
            }
        ],
    }

    ir = ChatCompletionsConverter().response_from_provider(cast(Any, response))

    assert ir["choices"][0]["finish_reason"]["reason"] == "tool_calls"


def test_a_chat_stream_stop_after_a_client_call_is_a_tool_call_turn():
    converter = ChatCompletionsConverter()
    context = StreamContext()
    converter.stream_response_from_provider(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call_1",
                                "type": "function",
                                "index": 0,
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ]
                    },
                }
            ]
        },
        context,
    )

    events = converter.stream_response_from_provider(
        {"choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        context,
    )

    finishes = [event for event in events if event["type"] == "finish"]
    assert finishes and finishes[-1]["finish_reason"]["reason"] == "tool_calls"


def test_a_messages_end_turn_with_tool_use_is_a_tool_call_turn():
    """A compatible Anthropic upstream may answer ``end_turn`` with tool_use."""
    response = {
        "id": "m1",
        "type": "message",
        "role": "assistant",
        "model": "m",
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "f", "input": {}}],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }

    ir = MessagesConverter().response_from_provider(cast(Any, response))

    assert ir["choices"][0]["finish_reason"]["reason"] == "tool_calls"


def test_a_messages_server_tool_use_keeps_stop():
    """A provider-hosted call is finished, not waiting on the client."""
    response = {
        "id": "m2",
        "type": "message",
        "role": "assistant",
        "model": "m",
        "content": [
            {
                "type": "server_tool_use",
                "id": "srvtoolu_1",
                "name": "web_search",
                "input": {},
            }
        ],
        "stop_reason": "end_turn",
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }

    ir = MessagesConverter().response_from_provider(cast(Any, response))

    assert ir["choices"][0]["finish_reason"]["reason"] == "stop"


def test_a_messages_stream_end_turn_after_tool_use_is_a_tool_call_turn():
    converter = MessagesConverter()
    context = StreamContext()
    converter.stream_response_from_provider(
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_1",
                "name": "f",
                "input": {},
            },
        },
        context,
    )

    events = converter.stream_response_from_provider(
        {"type": "message_delta", "delta": {"stop_reason": "end_turn"}},
        context,
    )

    finishes = [event for event in events if event["type"] == "finish"]
    assert finishes and finishes[-1]["finish_reason"]["reason"] == "tool_calls"


def test_a_chat_stream_decides_the_reason_per_choice():
    """One choice's call must not make another choice a tool turn."""
    converter = ChatCompletionsConverter()
    context = StreamContext()
    converter.stream_response_from_provider(
        {
            "choices": [
                {
                    "index": 0,
                    "delta": {
                        "tool_calls": [
                            {
                                "id": "call_0",
                                "type": "function",
                                "index": 0,
                                "function": {"name": "f", "arguments": "{}"},
                            }
                        ]
                    },
                },
                {"index": 1, "delta": {"content": "hi"}},
            ]
        },
        context,
    )

    events = converter.stream_response_from_provider(
        {
            "choices": [
                {"index": 0, "delta": {}, "finish_reason": "stop"},
                {"index": 1, "delta": {}, "finish_reason": "stop"},
            ]
        },
        context,
    )

    finishes = {
        (event["choice_index"], event["finish_reason"]["reason"])
        for event in events
        if event["type"] == "finish"
    }
    assert finishes == {(0, "tool_calls"), (1, "stop")}


def test_an_interactions_stream_status_follows_the_ir_reason():
    """The stream twin of the JSON status mapping, pinned on its own."""
    converter = GoogleInteractionsConverter()
    context = StreamContext()
    converter._handle_ir_stream_start_to_p(
        {"type": "stream_start", "response_id": "i1", "model": "m", "created": 0},
        context,
    )
    converter._handle_ir_finish_to_p(
        {"type": "finish", "finish_reason": {"reason": "tool_calls"}}, context
    )

    event = converter._handle_ir_stream_end_to_p({"type": "stream_end"}, context)

    assert event["interaction"]["status"] == "requires_action"


# ---------------------------------------------------------------------------
# An incomplete Responses turn with no reason stopped short (#1612)
# ---------------------------------------------------------------------------

_FUNCTION_CALL_ITEM = {
    "type": "function_call",
    "status": "completed",
    "call_id": "call_a",
    "name": "f",
    "arguments": "{}",
}
_TEXT_ITEM = {"type": "message", "content": [{"type": "output_text", "text": "hi"}]}

# Every way a frame can say nothing about why it is incomplete.
_NO_REASON = [
    pytest.param({}, id="details_absent"),
    pytest.param({"incomplete_details": None}, id="details_null"),
    pytest.param({"incomplete_details": {}}, id="details_empty"),
    pytest.param({"incomplete_details": {"reason": None}}, id="reason_null"),
]


def _incomplete_response(
    output: list[dict[str, Any]], details: dict[str, Any]
) -> dict[str, Any]:
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "m",
        "status": "incomplete",
        "output": output,
        **details,
    }


def _responses_incomplete(details: dict[str, Any]) -> dict[str, Any]:
    """A ``response.incomplete`` terminal frame; ``details`` may omit the reason."""
    response = {"id": "resp_1", "model": "m", "created_at": 1, "status": "incomplete"}
    return {"type": "response.incomplete", "response": {**response, **details}}


@pytest.mark.parametrize("details", _NO_REASON)
@pytest.mark.parametrize(
    "output", [[_FUNCTION_CALL_ITEM], [_TEXT_ITEM]], ids=["call", "text"]
)
def test_an_incomplete_response_with_no_reason_is_a_length_stop(output, details):
    """The JSON leg: the call in the output must not make it a tool turn."""
    ir = OpenResponsesConverter().response_from_provider(
        cast(Any, _incomplete_response(output, details))
    )

    assert ir["choices"][0]["finish_reason"]["reason"] == "length"
    chat = ChatCompletionsConverter().response_to_provider(ir)
    assert chat["choices"][0]["finish_reason"] == "length"


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("max_output_tokens", "length"),
        ("content_filter", "content_filter"),
        # A reason this reader does not know still means the turn stopped short.
        ("some_future_reason", "length"),
    ],
)
def test_an_incomplete_response_keeps_a_named_reason(reason, expected):
    ir = OpenResponsesConverter().response_from_provider(
        cast(
            Any,
            _incomplete_response(
                [_FUNCTION_CALL_ITEM], {"incomplete_details": {"reason": reason}}
            ),
        )
    )

    assert ir["choices"][0]["finish_reason"]["reason"] == expected


@pytest.mark.parametrize("details", _NO_REASON)
def test_an_incomplete_stream_with_no_reason_is_a_length_stop(details):
    """The stream leg: the call the stream opened must not upgrade it."""
    ir_events = _run_responses_stream(
        _responses_call_added(), _responses_incomplete(details)
    )

    assert _finish_reason(ir_events) == "length"
    _, stop_reasons = _messages_view(ir_events)
    assert stop_reasons and stop_reasons[-1] == "max_tokens"


def test_an_incomplete_stream_with_no_reason_survives_the_stream_processor():
    """The serving path: the processor's finalizer must not upgrade it either."""
    observed: list[Any] = []
    processor = StreamProcessor(
        target_converter=OpenResponsesConverter(),
        source_converter=MessagesConverter(),
        from_ctx=OpenResponsesConverter.create_stream_context(),
        to_ctx=MessagesConverter.create_stream_context(),
        on_ir_event=observed.append,
    )

    produced: list[dict[str, Any]] = []
    produced.extend(processor.process_chunk(_responses_call_added()))
    produced.extend(processor.process_chunk(_responses_incomplete({})))
    produced.extend(processor.process_chunk(None))

    stop_reasons = [
        chunk["delta"]["stop_reason"]
        for chunk in produced
        if chunk.get("type") == "message_delta"
    ]
    assert stop_reasons and stop_reasons[-1] == "max_tokens"
    assert [event for event in observed if event["type"] == "finish"][-1][
        "finish_reason"
    ]["reason"] == "length"


# A reason that is not a string (a broken upstream) is an unknown reason.
_UNHASHABLE_REASONS = [
    pytest.param(["max_output_tokens"], id="list"),
    pytest.param({"code": "x"}, id="object"),
    pytest.param(5, id="number"),
]


@pytest.mark.parametrize("reason", _UNHASHABLE_REASONS)
def test_a_non_string_reason_reads_as_unknown_in_the_reader(reason):
    ir = OpenResponsesConverter().response_from_provider(
        cast(
            Any,
            _incomplete_response(
                [_FUNCTION_CALL_ITEM], {"incomplete_details": {"reason": reason}}
            ),
        )
    )

    assert ir["choices"][0]["finish_reason"]["reason"] == "length"


@pytest.mark.parametrize("reason", _UNHASHABLE_REASONS)
def test_a_non_string_reason_reads_as_unknown_in_the_stream(reason):
    ir_events = _run_responses_stream(
        _responses_call_added(),
        _responses_incomplete({"incomplete_details": {"reason": reason}}),
    )

    assert _finish_reason(ir_events) == "length"


# ---------------------------------------------------------------------------
# Incomplete without a ``status`` (#1612 review)
# ---------------------------------------------------------------------------


def _statusless(output: list[dict[str, Any]] | None, **extra: Any) -> dict[str, Any]:
    """A response body that carries no ``status`` field."""
    body: dict[str, Any] = {"id": "resp_1", "model": "m", "created_at": 1, **extra}
    if output is not None:
        body["output"] = output
    return body


@pytest.mark.parametrize(
    "details",
    [{}, {"reason": None}, {"reason": "max_output_tokens"}],
    ids=["empty", "reason_null", "max_output_tokens"],
)
def test_a_statusless_body_with_incomplete_details_is_incomplete(details):
    """``incomplete_details`` says the turn stopped short when ``status`` is absent."""
    ir = OpenResponsesConverter().response_from_provider(
        cast(Any, _statusless([_FUNCTION_CALL_ITEM], incomplete_details=details))
    )

    assert ir["choices"][0]["finish_reason"]["reason"] == "length"


def test_a_statusless_body_keeps_a_named_reason():
    ir = OpenResponsesConverter().response_from_provider(
        cast(
            Any,
            _statusless(
                [_FUNCTION_CALL_ITEM],
                incomplete_details={"reason": "content_filter"},
            ),
        )
    )

    assert ir["choices"][0]["finish_reason"]["reason"] == "content_filter"


def test_a_statusless_body_without_incomplete_details_is_still_a_tool_turn():
    """The guard: no status and no details is a completed turn, as before."""
    ir = OpenResponsesConverter().response_from_provider(
        cast(Any, _statusless([_FUNCTION_CALL_ITEM], incomplete_details=None))
    )

    assert ir["choices"][0]["finish_reason"]["reason"] == "tool_calls"


# The two stream shapes: an incomplete event whose body has no status, and a
# completed event whose statusless body carries incomplete_details.
_STATUSLESS_STREAM_TERMINALS = [
    pytest.param(
        {"type": "response.incomplete", "response": _statusless(None)},
        id="incomplete_event_no_status",
    ),
    pytest.param(
        {
            "type": "response.incomplete",
            "response": _statusless(None, status="completed"),
        },
        id="incomplete_event_status_completed",
    ),
    pytest.param(
        {
            "type": "response.completed",
            "response": _statusless(None, incomplete_details={"reason": None}),
        },
        id="completed_event_with_incomplete_details",
    ),
]


@pytest.mark.parametrize("terminal", _STATUSLESS_STREAM_TERMINALS)
def test_a_statusless_incomplete_stream_is_a_length_stop(terminal):
    ir_events = _run_responses_stream(_responses_call_added(), terminal)

    assert _finish_reason(ir_events) == "length"
    _, stop_reasons = _messages_view(ir_events)
    assert stop_reasons and stop_reasons[-1] == "max_tokens"


@pytest.mark.parametrize("terminal", _STATUSLESS_STREAM_TERMINALS)
def test_a_statusless_incomplete_stream_survives_the_stream_processor(terminal):
    observed: list[Any] = []
    processor = StreamProcessor(
        target_converter=OpenResponsesConverter(),
        source_converter=MessagesConverter(),
        from_ctx=OpenResponsesConverter.create_stream_context(),
        to_ctx=MessagesConverter.create_stream_context(),
        on_ir_event=observed.append,
    )

    produced: list[dict[str, Any]] = []
    produced.extend(processor.process_chunk(_responses_call_added()))
    produced.extend(processor.process_chunk(terminal))
    produced.extend(processor.process_chunk(None))

    stop_reasons = [
        chunk["delta"]["stop_reason"]
        for chunk in produced
        if chunk.get("type") == "message_delta"
    ]
    assert stop_reasons and stop_reasons[-1] == "max_tokens"
    assert [event for event in observed if event["type"] == "finish"][-1][
        "finish_reason"
    ]["reason"] == "length"


def test_a_statusless_completed_stream_is_still_a_tool_turn():
    """The guard: a completed event with no status and no details runs the call."""
    ir_events = _run_responses_stream(
        _responses_call_added(),
        {"type": "response.completed", "response": _statusless(None)},
    )

    assert _finish_reason(ir_events) == "tool_calls"
