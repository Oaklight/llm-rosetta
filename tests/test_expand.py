"""A whole response, rendered as the event sequence a stream of it would emit.

The property under test is correspondence, not shape alone: what a body keeps
on a message, the events must carry on the event that ends the turn; what a
body keeps on a part, the events must carry on that part's block.  The
cross-format check lives in ``tests/converters/test_leg_agreement.py``, which
drives all five stream writers with this primitive and compares the result to
the hand-written fixture.
"""

from __future__ import annotations

import json
from typing import Any, cast, get_args

from llm_rosetta.ir.primitives import expand_response
from llm_rosetta.types.ir.parts import ContentPart


def _expand(response: Any) -> list[dict[str, Any]]:
    """Expand *response* and view the events as the plain dicts they are."""
    return cast(list[dict[str, Any]], expand_response(response))


def _response(
    content: list[dict[str, Any]],
    *,
    reason: str = "stop",
    usage: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    choices: int = 1,
) -> Any:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if metadata is not None:
        message["provider_metadata"] = metadata
    response: dict[str, Any] = {
        "id": "r1",
        "object": "response",
        "created": 7,
        "model": "m",
        "choices": [
            {
                "index": index,
                "message": dict(message),
                "finish_reason": {"reason": reason},
            }
            for index in range(choices)
        ],
    }
    if usage is not None:
        response["usage"] = usage
    return response


def _types(events: list[dict[str, Any]]) -> list[str]:
    return [event["type"] for event in events]


def test_it_starts_and_ends_a_stream():
    events = _expand(_response([]))

    assert _types(events) == ["stream_start", "finish", "stream_end"]
    assert events[0] == {
        "type": "stream_start",
        "response_id": "r1",
        "model": "m",
        "created": 7,
    }


def test_text_is_one_block_with_one_delta():
    events = _expand(_response([{"type": "text", "text": "hello"}]))

    assert _types(events) == [
        "stream_start",
        "content_block_start",
        "text_delta",
        "content_block_end",
        "finish",
        "stream_end",
    ]
    assert events[1]["block_type"] == "text"
    assert events[1]["block_index"] == 0
    assert events[2]["text"] == "hello"
    assert events[2]["block_index"] == 0
    assert events[3]["block_index"] == 0


def test_block_indices_run_across_parts():
    events = _expand(
        _response(
            [
                {"type": "reasoning", "reasoning": "think"},
                {"type": "text", "text": "hello"},
            ]
        )
    )

    starts = [e for e in events if e["type"] == "content_block_start"]
    assert [(s["block_index"], s["block_type"]) for s in starts] == [
        (0, "thinking"),
        (1, "text"),
    ]


def test_reasoning_carries_its_signature_and_opaque_blob():
    events = _expand(
        _response(
            [
                {
                    "type": "reasoning",
                    "reasoning": "think",
                    "signature": "sig",
                    "redacted_data": "blob",
                }
            ]
        )
    )

    delta = next(e for e in events if e["type"] == "reasoning_delta")
    assert delta["reasoning"] == "think"
    assert delta["signature"] == "sig"
    assert delta["encrypted_content"] == "blob"


def test_a_refusal_is_its_own_block():
    events = _expand(_response([{"type": "refusal", "refusal": "no"}]))

    start = next(e for e in events if e["type"] == "content_block_start")
    delta = next(e for e in events if e["type"] == "refusal_delta")
    assert start["block_type"] == "refusal"
    assert delta["refusal"] == "no"


def test_a_part_metadata_rides_its_block():
    """Item identity lives on the part in a body and on the block in a stream."""
    events = _expand(
        _response(
            [
                {
                    "type": "text",
                    "text": "hello",
                    "provider_metadata": {"responses_item_id": "msg_1"},
                }
            ]
        )
    )

    start = next(e for e in events if e["type"] == "content_block_start")
    assert start["provider_metadata"] == {"responses_item_id": "msg_1"}


def test_a_tool_call_is_a_start_and_a_delta():
    events = _expand(
        _response(
            [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_1",
                    "tool_name": "lookup",
                    "tool_input": {"q": "x"},
                    "intrinsic_kind": "web_search",
                }
            ]
        )
    )

    start = next(e for e in events if e["type"] == "tool_call_start")
    delta = next(e for e in events if e["type"] == "tool_call_delta")
    assert start["tool_call_id"] == "call_1"
    assert start["tool_name"] == "lookup"
    assert start["intrinsic_kind"] == "web_search"
    assert delta["intrinsic_kind"] == "web_search"
    assert json.loads(delta["arguments_delta"]) == {"q": "x"}


def test_a_tool_result_is_one_event():
    events = _expand(
        _response(
            [
                {
                    "type": "tool_result",
                    "tool_call_id": "call_1",
                    "result": {"hits": 3},
                    "tool_type": "intrinsic",
                    "is_error": False,
                }
            ]
        )
    )

    result = next(e for e in events if e["type"] == "tool_result")
    assert result["result"] == {"hits": 3}
    assert result["tool_type"] == "intrinsic"
    assert result["is_error"] is False


def test_a_citation_rides_the_block_of_the_text_it_follows():
    events = _expand(
        _response(
            [
                {"type": "text", "text": "hello"},
                {"type": "citation", "citation_type": "url", "url": "https://x"},
            ]
        )
    )

    citation = next(e for e in events if e["type"] == "citation_delta")
    assert citation["block_index"] == 0
    assert citation["citation"]["url"] == "https://x"


def test_a_citation_after_a_tool_call_still_names_the_text_block():
    """Google appends its citations after the answer's last part.

    A turn that reads text → call → citation annotates the text, so the
    citation must name block 0 even though a tool call sits between them.
    """
    events = _expand(
        _response(
            [
                {"type": "text", "text": "hello"},
                {
                    "type": "tool_call",
                    "tool_call_id": "call_1",
                    "tool_name": "lookup",
                    "tool_input": {},
                },
                {"type": "citation", "citation_type": "url", "url": "https://x"},
            ]
        )
    )

    citation = next(e for e in events if e["type"] == "citation_delta")
    assert citation["block_index"] == 0


def test_a_citation_after_reasoning_still_names_the_text_block():
    """The same rule across a reasoning block: it is not a text block either.

    A future reader might restore the reset in the reasoning branch believing
    it equivalent to the tool-call one; this pins that it is not.
    """
    events = _expand(
        _response(
            [
                {"type": "text", "text": "hello"},
                {"type": "reasoning", "reasoning": "think"},
                {"type": "citation", "citation_type": "url", "url": "https://x"},
            ]
        )
    )

    citation = next(e for e in events if e["type"] == "citation_delta")
    assert citation["block_index"] == 0


def test_a_citation_with_no_text_before_it_names_no_block():
    events = _expand(
        _response([{"type": "citation", "citation_type": "url", "url": "https://x"}])
    )

    citation = next(e for e in events if e["type"] == "citation_delta")
    assert "block_index" not in citation


def test_the_message_metadata_rides_the_finish_event():
    """What a body keeps on the message, a stream keeps on the finish."""
    record = {"origin": "google_generate", "finish_reason": "OTHER"}
    events = _expand(
        _response(
            [{"type": "text", "text": "partial"}],
            reason="error",
            metadata={"_native_response": record},
        )
    )

    finish = next(e for e in events if e["type"] == "finish")
    assert finish["finish_reason"] == {"reason": "error"}
    assert finish["provider_metadata"] == {"_native_response": record}


def test_usage_comes_before_the_finish():
    usage = {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    events = _expand(_response([], usage=usage))

    assert _types(events) == ["stream_start", "usage", "finish", "stream_end"]
    assert events[1]["usage"] == usage


def test_a_second_choice_is_tagged():
    response = _response([{"type": "text", "text": "hello"}], choices=2)
    response["choices"][1]["message"]["content"] = [{"type": "text", "text": "world"}]

    events = _expand(response)

    deltas = [e for e in events if e["type"] == "text_delta"]
    assert [d["choice_index"] for d in deltas] == [0, 1]
    assert [d["text"] for d in deltas] == ["hello", "world"]
    finishes = [e for e in events if e["type"] == "finish"]
    assert [f["choice_index"] for f in finishes] == [0, 1]


def test_block_boundaries_are_not_tagged():
    """No reader produces a choice index on a block, so none is written.

    A field nothing reads is a field with no meaning; a fold that needs to
    attribute a block to a choice can read the delta inside it.
    """
    response = _response([{"type": "text", "text": "hello"}], choices=2)
    response["choices"][1]["message"]["content"] = [{"type": "text", "text": "world"}]

    starts = [e for e in _expand(response) if e["type"] == "content_block_start"]

    assert [s["block_index"] for s in starts] == [0, 1]
    assert all("choice_index" not in s for s in starts)


def test_the_events_do_not_alias_the_response():
    """A caller can mutate an event without changing the body it came from."""
    usage = {"prompt_tokens": 3, "completion_tokens": 4, "total_tokens": 7}
    response = _response([{"type": "text", "text": "hello"}], usage=usage)
    events = _expand(response)

    next(e for e in events if e["type"] == "usage")["usage"]["prompt_tokens"] = 999
    next(e for e in events if e["type"] == "text_delta")["text"] = "changed"

    assert response["usage"]["prompt_tokens"] == 3
    assert response["choices"][0]["message"]["content"][0]["text"] == "hello"


def test_parallel_tool_calls_carry_their_index():
    """The IR has no per-call index; position is the only identity there is.

    The Chat writer indexes deltas by ``tool_call_index``, so two parallel
    calls announced at the same index would have their arguments folded into
    one call by a client that groups on it.
    """
    events = _expand(
        _response(
            [
                {
                    "type": "tool_call",
                    "tool_call_id": "c1",
                    "tool_name": "a",
                    "tool_input": {"x": 1},
                },
                {
                    "type": "tool_call",
                    "tool_call_id": "c2",
                    "tool_name": "b",
                    "tool_input": {"y": 2},
                },
            ]
        )
    )

    starts = [e for e in events if e["type"] == "tool_call_start"]
    deltas = [e for e in events if e["type"] == "tool_call_delta"]
    assert [(s["tool_call_id"], s["tool_call_index"]) for s in starts] == [
        ("c1", 0),
        ("c2", 1),
    ]
    assert [(d["tool_call_id"], d["tool_call_index"]) for d in deltas] == [
        ("c1", 0),
        ("c2", 1),
    ]


def test_the_choice_index_is_the_choices_own():
    """A choice that declares its index keeps it; a fold cannot renumber it."""
    response = _response([{"type": "text", "text": "hello"}], choices=2)
    response["choices"][1]["message"]["content"] = [{"type": "text", "text": "world"}]
    response["choices"][0]["index"] = 5
    response["choices"][1]["index"] = 9

    deltas = [e for e in _expand(response) if e["type"] == "text_delta"]

    assert [d["choice_index"] for d in deltas] == [5, 9]


def test_one_choice_is_not_tagged():
    """An absent index means the only choice, which is the IR's convention."""
    events = _expand(_response([{"type": "text", "text": "hello"}]))

    assert all("choice_index" not in event for event in events)


def test_a_stop_sequence_rides_the_finish_reason():
    response = _response([], reason="stop")
    response["choices"][0]["finish_reason"]["stop_sequence"] = "END"

    finish = next(e for e in _expand(response) if e["type"] == "finish")

    assert finish["finish_reason"] == {"reason": "stop", "stop_sequence": "END"}


#: The IR part types no event carries.  The primitive never claimed to carry
#: them, so it loses nothing; what the split decides is which body writers may
#: fold.  A writer that can emit one of these (Google's formats write image,
#: file and audio parts) would lose it in a fold, which is why those writers
#: are not folds.
UNREPRESENTABLE_PART_TYPES = frozenset({"image", "file", "audio"})

#: The part types :func:`expand_response` renders as events.
REPRESENTABLE_PART_TYPES = frozenset(
    {"text", "reasoning", "refusal", "citation", "tool_call", "tool_result"}
)

#: The events that frame a turn rather than carry a part of it.
_ENVELOPE_EVENTS = frozenset({"stream_start", "stream_end", "usage", "finish"})


def _declared_part_types() -> set[str]:
    """Every ``type`` value the IR's ``ContentPart`` union declares."""
    declared: set[str] = set()
    for part in get_args(ContentPart):
        # ``Required[Literal["text"]]`` on every part today, but a bare
        # ``Literal`` is legal TypedDict too, so unwrap until nothing is left.
        annotation: Any = part.__annotations__["type"]
        while args := get_args(annotation):
            annotation = args[0]
        declared.add(annotation)
    return declared


def test_every_declared_part_type_is_classified():
    """A new IR part type must be a decision: given an event, or not.

    The classified set is what a body writer may fold over.  A part type
    outside it is one a fold would drop, so adding a part type to the IR fails
    here until someone decides which side it is on.  That the primitive really
    emits nothing for the unrepresentable ones is
    ``test_an_unrepresentable_part_adds_no_block``; that a fold would then lose
    them is why the four body writers that write them are not folds.
    """
    declared = _declared_part_types()

    assert declared == REPRESENTABLE_PART_TYPES | UNREPRESENTABLE_PART_TYPES


def test_an_unrepresentable_part_adds_no_block():
    """An image, file or audio part contributes no content event.

    The parts use the fields the IR declares (``image_url``, ``file_url``,
    ``url``), not a flat spelling: a body writer reads those fields, so the
    witness has to be a part a writer would really accept.  Only the content
    events are asserted — a change to the usage or finish event is not this
    test's subject, and failing on one would misname the regression.
    """
    for part in (
        {"type": "image", "image_url": "https://example.test/a.png"},
        {
            "type": "file",
            "file_url": "https://example.test/a.pdf",
            "file_type": "application/pdf",
        },
        {"type": "audio", "url": "https://example.test/a.wav"},
    ):
        content = [
            event
            for event in _expand(_response([part]))
            if event["type"] not in _ENVELOPE_EVENTS
        ]
        assert content == [], part
