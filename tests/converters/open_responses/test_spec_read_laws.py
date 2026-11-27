"""Open Responses read-path rules, stated over generated bodies and streams.

Source: the Open Responses specification (https://www.openresponses.org/specification)
and its OpenAPI document (``ResponseResource``, ``IncompleteDetails``, the item
status enums ``in_progress`` / ``completed`` / ``incomplete``).  Each test names the
rule it states.  Bodies are written by hand to the spec's shape, not by the
converter, so the reader is checked against the spec and not against itself.

``ResponseResource`` requires ``status``, ``output`` (an array, possibly empty),
``incomplete_details`` (nullable) and ``error`` (nullable).

Tests marked ``xfail`` (non-strict) record a rule the reader does not meet today;
each names the issue or PR that owns the gap.  They pass either way, so the fix
does not have to touch this file.  The one ``pin`` test records behaviour that is
outside the spec and asserts no rule.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from llm_rosetta.converters.llm.open_responses import (  # noqa: E402
    OpenAIResponsesConverter,
    OpenResponsesConverter,
    _wire_helpers,
    converter as converter_module,
    stream_from_ops,
)
from llm_rosetta.converters.registry import _CONVERTER_CLASSES  # noqa: E402

CONVERTERS = [OpenResponsesConverter, OpenAIResponsesConverter]
TERMINAL_EVENT = {
    "completed": "response.completed",
    "incomplete": "response.incomplete",
    "failed": "response.failed",
}
LAW = settings(max_examples=60, deadline=None, derandomize=True)

_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    min_size=1,
    max_size=20,
)


def _message(index: int, text: str, status: str = "completed") -> dict[str, Any]:
    return {
        "type": "message",
        "id": f"msg_{index}",
        "role": "assistant",
        "status": status,
        "content": [{"type": "output_text", "text": text, "annotations": []}],
    }


def _call(index: int, status: str = "completed") -> dict[str, Any]:
    return {
        "type": "function_call",
        "id": f"fc_{index}",
        "call_id": f"call_{index}",
        "name": "lookup",
        "arguments": f'{{"q": {index}}}',
        "status": status,
    }


@st.composite
def _output(draw, *, min_size: int = 0, max_size: int = 4) -> list[dict[str, Any]]:
    """Output items of the two kinds every reader must carry, all completed."""
    items = []
    for index in range(draw(st.integers(min_size, max_size))):
        if draw(st.booleans()):
            items.append(_message(index, draw(_TEXT)))
        else:
            items.append(_call(index))
    return items


def _body(
    status: str | None,
    output: list[dict[str, Any]],
    *,
    details: dict[str, Any] | None = None,
    error: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "id": "resp_1",
        "object": "response",
        "created_at": 1,
        "model": "m",
        "output": output,
        "incomplete_details": details,
        "error": error,
    }
    if status is not None:
        body["status"] = status
    return body


def _read_body(converter_cls: type, body: dict[str, Any]) -> dict[str, Any]:
    return converter_cls().response_from_provider(body)


def _reasons(ir_response: dict[str, Any]) -> list[str]:
    return [c["finish_reason"]["reason"] for c in ir_response["choices"]]


def _stream_of(body: dict[str, Any]) -> list[dict[str, Any]]:
    """The events a conforming server sends for the final response *body*."""
    final = body["status"]
    events: list[dict[str, Any]] = [
        {
            "type": "response.created",
            "response": {**body, "status": "in_progress", "output": []},
        }
    ]
    for i, item in enumerate(body["output"]):
        events.append(
            {
                "type": "response.output_item.added",
                "output_index": i,
                "item": {**item, "status": "in_progress"},
            }
        )
        if item["type"] == "message":
            text = item["content"][0]["text"]
            ref = {"output_index": i, "item_id": item["id"], "content_index": 0}
            events += [
                {
                    **ref,
                    "type": "response.content_part.added",
                    "part": {"type": "output_text", "text": "", "annotations": []},
                },
                {**ref, "type": "response.output_text.delta", "delta": text},
                {**ref, "type": "response.output_text.done", "text": text},
                {
                    **ref,
                    "type": "response.content_part.done",
                    "part": item["content"][0],
                },
            ]
        elif item["type"] == "function_call":
            ref = {"output_index": i, "item_id": item["id"]}
            events += [
                {
                    **ref,
                    "type": "response.function_call_arguments.delta",
                    "delta": item["arguments"],
                },
                {
                    **ref,
                    "type": "response.function_call_arguments.done",
                    "arguments": item["arguments"],
                },
            ]
        events.append(
            {"type": "response.output_item.done", "output_index": i, "item": item}
        )
    events.append({"type": TERMINAL_EVENT[final], "response": body})
    return events


def _read_stream(converter_cls: type, body: dict[str, Any]) -> list[dict[str, Any]]:
    converter = converter_cls()
    return list(
        converter.iter_stream_from_provider(
            _stream_of(body), context=converter.create_stream_context()
        )
    )


def _stream_reasons(events: list[dict[str, Any]]) -> list[str]:
    return [e["finish_reason"]["reason"] for e in events if e["type"] == "finish"]


def _expected_completed(output: list[dict[str, Any]]) -> str:
    return "tool_calls" if any(i["type"] == "function_call" for i in output) else "stop"


# --- the target is the product's own -----------------------------------------


def test_the_readers_are_the_ones_the_product_runs() -> None:
    assert _CONVERTER_CLASSES["llm"]["open_responses"] is OpenResponsesConverter
    assert issubclass(OpenAIResponsesConverter, OpenResponsesConverter)
    # Both legs decide the terminal state through these two functions.
    assert (
        converter_module._response_status_reason
        is _wire_helpers._response_status_reason
    )
    assert stream_from_ops._stream_finish_reason is _wire_helpers._stream_finish_reason


# --- rule 1: ResponseResource status -> choices and finish reason -----------


@pytest.mark.parametrize("converter_cls", CONVERTERS)
@LAW
@given(output=_output(min_size=1))
def test_a_completed_body_reads_as_one_choice(converter_cls, output) -> None:
    """status completed, output non-empty: one choice, stop (or tool_calls)."""
    ir = _read_body(converter_cls, _body("completed", output))
    assert _reasons(ir) == [_expected_completed(output)]
    assert [c["index"] for c in ir["choices"]] == [0]


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_a_completed_body_with_empty_output_reads_as_one_stop_choice(
    converter_cls,
) -> None:
    ir = _read_body(converter_cls, _body("completed", []))
    assert _reasons(ir) == ["stop"]


@pytest.mark.parametrize("converter_cls", CONVERTERS)
@LAW
@given(
    output=_output(),
    code=st.sampled_from(["server_error", "rate_limit_exceeded"]),
    message=_TEXT,
)
def test_a_failed_body_reads_as_error_and_keeps_the_error(
    converter_cls, output, code, message
) -> None:
    """status failed: finish reason error, with or without output."""
    error = {"code": code, "message": message}
    ir = _read_body(converter_cls, _body("failed", output, error=error))
    assert _reasons(ir) == ["error"]
    metadata = ir["choices"][0]["message"].get("provider_metadata", {})
    assert metadata.get("responses_error") == error


INCOMPLETE_REASON = {
    "max_output_tokens": "length",
    "content_filter": "content_filter",
}


@pytest.mark.parametrize("converter_cls", CONVERTERS)
@LAW
@given(
    output=_output(),
    reason=st.one_of(
        st.sampled_from(sorted(INCOMPLETE_REASON)),
        st.text(max_size=12),
        st.none(),
    ),
)
def test_an_incomplete_body_reads_as_the_reason_it_names(
    converter_cls, output, reason
) -> None:
    """status incomplete: the reason incomplete_details names, else length."""
    details = None if reason is None else {"reason": reason}
    ir = _read_body(converter_cls, _body("incomplete", output, details=details))
    assert _reasons(ir) == [INCOMPLETE_REASON.get(reason, "length")]


@pytest.mark.parametrize("converter_cls", CONVERTERS)
@pytest.mark.parametrize("status", ["in_progress", "queued"])
def test_an_unfinished_body_with_no_output_reads_as_no_choice(
    converter_cls, status
) -> None:
    assert _read_body(converter_cls, _body(status, []))["choices"] == []


@pytest.mark.xfail(
    strict=False,
    reason="a non-terminal body with output reads as a stop/tool_calls choice; "
    "owner not decided (reported to llm-rosetta-03)",
)
@pytest.mark.parametrize("converter_cls", CONVERTERS)
@pytest.mark.parametrize("status", ["in_progress", "queued"])
def test_an_unfinished_body_with_output_reads_as_no_choice(
    converter_cls, status
) -> None:
    ir = _read_body(converter_cls, _body(status, [_message(0, "partial")]))
    assert ir["choices"] == []


# --- rule 2: an incomplete item is last, and the response is incomplete -----


@pytest.mark.parametrize("converter_cls", CONVERTERS)
@LAW
@given(
    before=_output(),
    reason=st.sampled_from(["max_output_tokens", None]),
    cut=st.sampled_from(["function_call", "message"]),
)
def test_an_incomplete_last_item_reads_as_length_and_stays_last(
    converter_cls, before, reason, cut
) -> None:
    """The unfinished item ends the output; the turn is length, never a call."""
    last = (
        _call(len(before), "incomplete")
        if cut == "function_call"
        else _message(len(before), "cut off", "incomplete")
    )
    output = [*before, last]
    details = None if reason is None else {"reason": reason}
    ir = _read_body(converter_cls, _body("incomplete", output, details=details))
    assert _reasons(ir) == ["length"]
    content = ir["choices"][0]["message"]["content"]
    assert [p["type"] for p in content] == [
        "tool_call" if i["type"] == "function_call" else "text" for i in output
    ]
    assert content[-1]["type"] == ("tool_call" if cut == "function_call" else "text")


@pytest.mark.parametrize("converter_cls", CONVERTERS)
@LAW
@given(before=_output())
def test_an_incomplete_call_is_never_a_tool_calls_stop_on_either_leg(
    converter_cls, before
) -> None:
    output = [*before, _call(len(before), "incomplete")]
    body = _body("incomplete", output, details={"reason": "max_output_tokens"})
    assert _reasons(_read_body(converter_cls, body)) == ["length"]
    assert _stream_reasons(_read_stream(converter_cls, body)) == ["length"]


# --- rule 3: the stream ends with a terminal event; the legs agree ----------


def _terminal_body(draw) -> dict[str, Any]:
    status = draw(st.sampled_from(sorted(TERMINAL_EVENT)))
    output = draw(_output(min_size=1))
    if status == "incomplete":
        reason = draw(st.sampled_from(sorted(INCOMPLETE_REASON)))
        return _body(status, output, details={"reason": reason})
    if status == "failed":
        return _body(status, output, error={"code": "server_error", "message": "x"})
    return _body(status, output)


@pytest.mark.parametrize("converter_cls", CONVERTERS)
@LAW
@given(data=st.data())
def test_the_stream_leg_and_the_body_leg_give_one_finish_reason(
    converter_cls, data
) -> None:
    """The same final response reads to the same reason, on both legs."""
    body = _terminal_body(data.draw)
    body_reasons = _reasons(_read_body(converter_cls, body))
    events = _read_stream(converter_cls, body)
    assert len(body_reasons) == 1
    assert _stream_reasons(events) == body_reasons
    assert [e["type"] for e in events[-2:]] == ["finish", "stream_end"]


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_the_legs_agree_on_a_completed_response_with_no_output(converter_cls) -> None:
    body = _body("completed", [])
    assert _reasons(_read_body(converter_cls, body)) == _stream_reasons(
        _read_stream(converter_cls, body)
    )


@pytest.mark.parametrize("converter_cls", CONVERTERS)
@LAW
@given(data=st.data())
def test_a_terminal_frame_alone_gives_the_body_legs_reason(converter_cls, data) -> None:
    """A stream that opened no items still ends as its final response reads."""
    body = _terminal_body(data.draw)
    converter = converter_cls()
    events = list(
        converter.iter_stream_from_provider(
            [{"type": TERMINAL_EVENT[body["status"]], "response": body}],
            context=converter.create_stream_context(),
        )
    )
    assert _stream_reasons(events) == _reasons(_read_body(converter_cls, body))


WRITER_TERMINAL = {
    "stop": "response.completed",
    "tool_calls": "response.completed",
    "length": "response.incomplete",
    "content_filter": "response.incomplete",
    "error": "response.failed",
}


@pytest.mark.parametrize("converter_cls", CONVERTERS)
@pytest.mark.parametrize("reason", sorted(WRITER_TERMINAL))
def test_a_written_stream_ends_with_its_terminal_event(converter_cls, reason) -> None:
    converter = converter_cls()
    context = converter.create_stream_context()
    chunks: list[dict[str, Any]] = []
    for event in [
        {"type": "stream_start", "response_id": "r", "model": "m"},
        {"type": "content_block_start", "block_index": 0, "block_type": "text"},
        {"type": "text_delta", "text": "x"},
        {"type": "content_block_end", "block_index": 0},
        {"type": "finish", "finish_reason": {"reason": reason}},
        {"type": "stream_end"},
    ]:
        out = converter.stream_response_to_provider(event, context=context)
        chunks.extend(out if isinstance(out, list) else [out])
    terminal = [c["type"] for c in chunks if c["type"] in TERMINAL_EVENT.values()]
    assert terminal == [WRITER_TERMINAL[reason]]
    assert chunks[-1]["type"] == WRITER_TERMINAL[reason]


def _write_stream(
    converter_cls: type, reason: str, *, with_call: bool = False
) -> list[dict[str, Any]]:
    converter = converter_cls()
    context = converter.create_stream_context()
    chunks: list[dict[str, Any]] = []
    body: list[dict[str, Any]] = [
        {"type": "content_block_start", "block_index": 0, "block_type": "text"},
        {"type": "text_delta", "text": "x"},
        {"type": "content_block_end", "block_index": 0},
    ]
    if with_call:
        body += [
            {
                "type": "tool_call_start",
                "tool_call_id": "c1",
                "tool_name": "lookup",
                "tool_call_index": 0,
            },
            {"type": "tool_call_delta", "tool_call_id": "c1", "arguments_delta": "{}"},
        ]
    for event in [
        {"type": "stream_start", "response_id": "r", "model": "m"},
        *body,
        {"type": "finish", "finish_reason": {"reason": reason}},
        {"type": "stream_end"},
    ]:
        out = converter.stream_response_to_provider(event, context=context)
        chunks.extend(out if isinstance(out, list) else [out])
    return chunks


@pytest.mark.parametrize("converter_cls", CONVERTERS)
@pytest.mark.parametrize("with_call", [False, True], ids=["text", "after-a-call"])
def test_pin_a_cancelled_turn_closes_with_a_completed_frame_that_says_cancelled(
    converter_cls, with_call
) -> None:
    """Pin (#1741): outside the spec, and not a defect.

    The spec's status enum is ``in_progress`` / ``completed`` / ``incomplete``
    and it has three terminal events, so it has no cancelled turn and no event
    for one.  OpenAI's own Responses API has a ``cancelled`` status on the body
    it returns for the cancel call and for a later ``GET``.  A long generation
    cancelled midway on its stream ends with none of the three terminal events
    (measured live, 2026-11-26, ``gpt-5-mini``, reproduced by a second reader).
    No OpenAI stream was seen to write ``cancelled`` into a frame, so the writer
    does something OpenAI's stream does not: it keeps the fact in
    ``response.status`` and closes with the one terminal event that is neither
    an error nor a stop short, so a client that reads the stream to its end is
    not left waiting.  (A short run, cancelled before the stream replayed, ended
    with ``response.completed`` whose status was ``completed`` in 8 runs of one
    reader and got HTTP 400 for the other; it is a race and nothing here
    depends on it.)  The reader maps the frame back to ``cancelled``, also when a
    tool call is open, where a ``response.completed`` could be taken for a
    finished call turn.

    The cost: ``cancelled`` is not in the spec's status enum, so a client that
    validates the frame against the spec's schema rejects it, and a client that
    reads only the event type sees a completed turn.  This pin records what the
    writer does and is not an endorsement by the spec.
    """
    chunks = _write_stream(converter_cls, "cancelled", with_call=with_call)
    last = chunks[-1]
    assert last["type"] == "response.completed"
    assert last["response"]["status"] == "cancelled"
    assert last["response"].get("error") is None
    assert ("function_call" in [i["type"] for i in last["response"]["output"]]) == (
        with_call
    )
    converter = converter_cls()
    events = converter.iter_stream_from_provider(
        chunks, context=converter.create_stream_context()
    )
    assert _stream_reasons(list(events)) == ["cancelled"]


# --- rule 4: unknown provider-slug item types are kept ----------------------


@st.composite
def _slugged(draw) -> list[tuple[int, dict[str, Any]]]:
    count = draw(st.integers(1, 3))
    slug = st.text(alphabet="abcdefgh", min_size=1, max_size=6)
    kind = st.text(alphabet="abcdefgh_", min_size=1, max_size=10)
    return [
        (
            draw(st.integers(0, 3)),
            {
                "type": f"{draw(slug)}:{draw(kind)}",
                "id": f"x_{i}",
                "data": {"n": draw(st.integers(0, 9))},
            },
        )
        for i in range(count)
    ]


@pytest.mark.parametrize("converter_cls", CONVERTERS)
@LAW
@given(known=_output(min_size=1, max_size=3), extra=_slugged())
def test_a_provider_slug_item_is_read_and_kept_as_passthrough(
    converter_cls, known, extra
) -> None:
    output = list(known)
    for at, item in extra:
        output.insert(min(at, len(output)), item)
    ir = _read_body(converter_cls, _body("completed", output))
    kept = ir.get("provider_passthrough_items", [])
    slugged = [(p, i) for p, i in enumerate(output) if ":" in i["type"]]
    assert [(k["position"], k["payload"]) for k in kept] == slugged
    # The known items around them still read.
    assert _reasons(ir) == [_expected_completed(known)]


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_a_provider_slug_item_does_not_break_the_stream(converter_cls) -> None:
    body = _body(
        "completed",
        [
            {"type": "acme:search_result", "id": "x_1", "data": {"q": 1}},
            _message(1, "hi"),
        ],
    )
    events = _read_stream(converter_cls, body)
    assert _stream_reasons(events) == ["stop"]
    assert [e["text"] for e in events if e["type"] == "text_delta"] == ["hi"]


@pytest.mark.parametrize("converter_cls", CONVERTERS)
@LAW
@given(known=_output(min_size=1, max_size=3), extra=_slugged())
def test_a_provider_slug_item_is_kept_on_the_stream_leg(
    converter_cls, known, extra
) -> None:
    """The stream leg keeps every item the body leg keeps (#1740).

    An item of a type the reader does not know is an opaque record.  The stream
    carries it as passthrough events whose payload holds that item, once for the
    item being added and once for it being done, and the items around it still
    read.
    """
    output = list(known)
    for at, item in extra:
        output.insert(min(at, len(output)), item)
    events = _read_stream(converter_cls, _body("completed", output))
    slugged = [item for item in output if ":" in item["type"]]
    carried = [e["payload"] for e in events if e["type"] == "provider_passthrough"]
    for kind in ("added", "done"):
        frames = [p for p in carried if p["type"] == f"response.output_item.{kind}"]
        assert [p["item"]["id"] for p in frames] == [i["id"] for i in slugged]
        if kind == "done":
            assert [p["item"] for p in frames] == slugged
    assert _stream_reasons(events) == [_expected_completed(known)]


OPAQUE_ITEMS = [
    {"type": "acme:search_result", "id": "x_1", "data": {"q": 1}},
    {"type": "compaction", "id": "cmp_1", "encrypted_content": "abc"},
    {"type": "image_generation_call", "id": "ig_1", "status": "completed"},
    {"type": "tool_search_call", "id": "ts_1", "execution": "server"},
    {"type": "local_shell_call", "id": "ls_1", "status": "completed"},
    {"type": "mcp_list_tools", "id": "mcp_1", "tools": []},
    {"type": "apply_patch_call", "id": "ap_1", "status": "completed"},
    {"type": "item_reference", "id": "ref_1"},
]


def _folded_output(converter_cls: type, body: dict[str, Any]) -> list[dict[str, Any]]:
    """The output list of the body a stream folds to, as the gateway builds it."""
    from llm_rosetta.pipeline import ConversionPipeline

    converter = converter_cls()
    events = list(
        converter.iter_stream_from_provider(
            _stream_of(body), context=converter.create_stream_context()
        )
    )
    pipeline = ConversionPipeline("open_responses", "open_responses")
    pipeline.convert_request({"model": "m", "input": "hi"})
    return pipeline.convert_stream_events(events)["output"]


@pytest.mark.parametrize("converter_cls", [OpenResponsesConverter])
@pytest.mark.parametrize("item", OPAQUE_ITEMS, ids=lambda i: i["type"])
def test_a_folded_stream_holds_the_opaque_item_the_body_leg_keeps(
    converter_cls, item
) -> None:
    """The fold keeps one record holding the item, as the body leg does (#1740).

    Only the base class: the ``open_responses`` pipeline writes its own dialect's
    opaque items, and the OpenAI profile's are another dialect's.
    """
    body = _body("completed", [_message(0, "hi"), item])
    kept = _folded_output(converter_cls, body)
    assert [i["type"] for i in kept] == ["message", item["type"]]
    assert kept[1] == item
    via_body = converter_cls().response_to_provider(_read_body(converter_cls, body))
    assert [i["type"] for i in via_body["output"]] == [i["type"] for i in kept]


@pytest.mark.parametrize("reader", CONVERTERS)
@pytest.mark.parametrize("writer", CONVERTERS)
@pytest.mark.parametrize("item", OPAQUE_ITEMS[:3], ids=lambda i: i["type"])
def test_the_fold_keeps_an_opaque_item_exactly_where_the_body_leg_does(
    reader, writer, item
) -> None:
    """Pin (#1740): an opaque item belongs to the dialect that read it.

    A passthrough record is restored only by the converter whose tag it carries
    (``restore_provider_passthrough_item``): the base class tags ``open_responses``
    and the OpenAI profile ``openai_responses``.  Folding a stream into a writer
    of the other class drops the item with a ``dropped`` warning, exactly as the
    body leg does between the same two classes, so the legs agree in all four
    pairs.  The product builds every ``open_responses`` and
    ``openai--open_responses`` pipeline on the base class
    (``test_a_shim_target_keeps_the_item``), so the pairs that mix the two
    classes are only reachable from a direct library call.
    """
    from llm_rosetta.ir.primitives import fold_response_events

    body = _body("completed", [_message(0, "hi"), item])
    source = reader()
    events = list(
        source.iter_stream_from_provider(
            _stream_of(body), context=source.create_stream_context()
        )
    )
    folded = writer().response_to_provider(fold_response_events(events))
    via_body = writer().response_to_provider(source.response_from_provider(body))
    kinds = [i["type"] for i in folded["output"]]
    assert kinds == [i["type"] for i in via_body["output"]]
    kept = reader is writer
    assert kinds == (["message", item["type"]] if kept else ["message"])


def test_a_shim_target_keeps_the_item() -> None:
    """The real stream-only path: ``openai--open_responses`` is the base class."""
    from llm_rosetta.pipeline import ConversionPipeline

    item = OPAQUE_ITEMS[0]
    source = OpenResponsesConverter()
    events: list[dict[str, Any]] = [
        dict(e)
        for e in source.iter_stream_from_provider(
            _stream_of(_body("completed", [_message(0, "hi"), item])),
            context=source.create_stream_context(),
        )
    ]
    for client in ("open_responses", "openai--open_responses"):
        pipeline = ConversionPipeline(client, "openai--open_responses")
        pipeline.convert_request({"model": "m", "input": "hi"})
        body = pipeline.convert_stream_events(events)
        assert [i["type"] for i in body["output"]] == ["message", item["type"]]


@pytest.mark.parametrize("converter_cls", CONVERTERS)
@pytest.mark.parametrize(
    "item_type",
    ["computer_call_output", "shell_call_output", "code_interpreter_call_output"],
)
def test_a_server_result_item_is_a_tool_result_and_not_also_passthrough(
    converter_cls, item_type
) -> None:
    """The stream turns these into IR events itself, so it keeps no second copy."""
    item = {"type": item_type, "id": "r_1", "call_id": "call_1", "output": []}
    events = _read_stream(converter_cls, _body("completed", [item]))
    assert not [e for e in events if e["type"] == "provider_passthrough"]


@pytest.mark.parametrize("converter_cls", CONVERTERS)
@pytest.mark.parametrize("item_type", ["", None, 5])
def test_pin_the_legs_differ_on_an_item_with_no_usable_type(
    converter_cls, item_type
) -> None:
    """Pin (#1740): the one place the two legs still disagree.

    The body leg keeps any dict it does not rebuild, so an item whose ``type`` is
    empty, missing or not a string reads to a passthrough record.  The stream leg
    drops it: a frame with no type name is malformed and not an unknown type,
    and there is nothing to name the opaque record by.
    """
    item: dict[str, Any] = {"id": "x_1", "data": 1}
    if item_type is not None:
        item["type"] = item_type
    ir = _read_body(converter_cls, _body("completed", [item]))
    assert [k["payload"] for k in ir.get("provider_passthrough_items", [])] == [item]
    frames = [
        {"type": "response.output_item.added", "output_index": 0, "item": item},
        {"type": "response.output_item.done", "output_index": 0, "item": item},
    ]
    converter = converter_cls()
    context = converter.create_stream_context()
    events = [
        e
        for frame in frames
        for e in converter.stream_response_from_provider(frame, context)
    ]
    assert not [e for e in events if e["type"] == "provider_passthrough"]


@pytest.mark.parametrize("converter_cls", CONVERTERS)
def test_pin_the_writer_closes_an_item_after_the_next_one_opens(converter_cls) -> None:
    """Pin (#1740): item frames replay in the order the writer closes items.

    The spec gives each item the lifecycle ``added``, deltas, ``done`` and says
    nothing of the order between two items.  The writer closes a message item
    when the response finishes or the next message opens, so after a message the
    message's ``done`` comes after the next item's ``added``.  A tool call after a
    message does the same, so an opaque item is no different.  Each item still
    has exactly one ``added`` before its ``done``.
    """
    body = _body("completed", [_message(0, "hi"), OPAQUE_ITEMS[0]])
    converter = converter_cls()
    events = list(
        converter.iter_stream_from_provider(
            _stream_of(body), context=converter.create_stream_context()
        )
    )
    context = converter.create_stream_context()
    written: list[dict[str, Any]] = []
    for event in events:
        out = converter.stream_response_to_provider(event, context=context)
        written.extend(out if isinstance(out, list) else [out])
    lifecycle = [
        (c["type"].rsplit(".", 1)[1], c["output_index"], c["item"]["type"])
        for c in written
        if c["type"] in ("response.output_item.added", "response.output_item.done")
    ]
    assert sorted(lifecycle) == sorted(
        [
            ("added", 0, "message"),
            ("done", 0, "message"),
            ("added", 1, "acme:search_result"),
            ("done", 1, "acme:search_result"),
        ]
    )
    for kind_index in (0, 1):
        pair = [e for e in lifecycle if e[1] == kind_index]
        assert [e[0] for e in pair] == ["added", "done"]


# --- rule 5: outside the spec -----------------------------------------------


@pytest.mark.parametrize("converter_cls", CONVERTERS)
@pytest.mark.parametrize("status", [None, "cancelled", "totally_new_status"])
def test_pin_a_missing_or_unknown_status_reads_as_the_parts_say(
    converter_cls, status
) -> None:
    """PIN, not a rule: ``status`` is required, so the spec says nothing here.

    Records what the reader does today so a change is noticed: ``stop`` for a
    body with only a message, ``tool_calls`` once a call is present.  (A status
    the reader maps itself, such as ``cancelled``, is excluded.)
    """
    if status == "cancelled":
        pytest.skip("mapped by the reader: cancelled -> cancelled")
    assert _reasons(_read_body(converter_cls, _body(status, [_message(0, "hi")]))) == [
        "stop"
    ]
    assert _reasons(
        _read_body(converter_cls, _body(status, [_message(0, "hi"), _call(1)]))
    ) == ["tool_calls"]
