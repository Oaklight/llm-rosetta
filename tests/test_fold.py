"""A stream of IR events, folded back into the response it renders.

The property under test is the inverse of ``tests/test_expand.py``: whatever
``expand_response()`` renders, ``fold_response_events()`` must read back.  The
case table belongs to the cross-leg harness, so the same turns the writers are
measured with are the ones folded here — a fold that only handles one
hand-written stream proves nothing.

The caller that needs this is the gateway's stream-only bridge: a shim declares
``connection.stream_required``, the gateway streams upstream, and a client that
asked for one body gets the folded response.  What the fold cannot recover is
what no event carries, and
``test_a_response_level_fact_no_event_carries_is_not_invented`` pins that
boundary rather than pretending the body legs agree exactly; the fields a
dialect *does* repeat on its chunks are handed in, and
``test_an_echo_field_the_stream_states_is_put_back`` pins that path.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from llm_rosetta.ir.primitives import expand_response, fold_response_events
from llm_rosetta.pipeline import ConversionPipeline
from llm_rosetta.ir.model import IRResponse
from tests.converters.test_leg_agreement import (
    CASE_TABLE,
    Turn,
    _ir_response,
    _view_of_body,
)

_TURNS = {case_id: turn for case_id, turn, _reason in CASE_TABLE}


def _render(turn: Turn) -> list[dict[str, Any]]:
    """The events ``expand_response`` renders for *turn*."""
    body = cast(IRResponse, _ir_response(turn))
    return cast(list[dict[str, Any]], expand_response(body))


def _fold(turn: Turn) -> dict[str, Any]:
    """Render *turn* as events, then fold them back."""
    return cast(dict[str, Any], fold_response_events(_render(turn)))


@pytest.mark.parametrize("case_id", list(_TURNS))
def test_a_response_survives_a_render_and_a_fold(case_id: str) -> None:
    """``fold(expand(turn))`` is the turn, compared as the harness compares it.

    The turn view is the comparison the stream and body legs already use, so a
    part either leg can carry is compared by value rather than by dict shape.
    """
    turn = _TURNS[case_id]

    assert _view_of_body(_fold(turn)) == _view_of_body(_ir_response(turn))


def test_a_terminal_event_with_no_output_still_yields_the_turn() -> None:
    """The case #1061 described: the route reports an empty ``output``.

    The items arrive on ``output_item.*`` events, so a fold that read the
    terminal payload alone would answer an empty reply.
    """
    events = _render(Turn(text="hello"))
    terminal = events[-2]
    assert terminal["type"] == "finish"

    folded = fold_response_events(events)

    parts = folded["choices"][0]["message"]["content"]
    assert parts == [{"type": "text", "text": "hello"}]


def test_a_call_whose_arguments_never_parsed_is_named_as_a_loss(caplog) -> None:
    """A truncated stream leaves JSON that cannot be a call's input.

    Keeping the fragment under a key the writers do not confuse with input is
    what the Chat reader does with the same shape; the point of the assertion
    is that the loss is reported rather than passed off as ``{}``.
    """
    import logging

    events = [
        {"type": "stream_start", "response_id": "r1", "model": "m"},
        {
            "type": "tool_call_start",
            "tool_call_id": "call_1",
            "tool_name": "lookup",
            "tool_type": "function",
        },
        {
            "type": "tool_call_delta",
            "tool_call_id": "call_1",
            "arguments_delta": '{"q": ',
        },
        {"type": "finish", "finish_reason": {"reason": "tool_calls"}},
    ]

    with caplog.at_level(logging.WARNING):
        folded = fold_response_events(events)

    part = folded["choices"][0]["message"]["content"][0]
    assert part["type"] == "tool_call"
    call = cast(Any, part)
    assert call["tool_input"] == {"raw_arguments": '{"q": '}
    assert "did not parse as JSON" in caplog.text


def test_a_bare_delta_joins_the_open_block_of_its_kind() -> None:
    """Chat Completions sends no ``block_index``; the block still has to be found.

    The writers close a block before switching kinds, so an open block whose
    kind disagrees with the delta is not the delta's target.
    """
    events = [
        {"type": "stream_start", "response_id": "r1", "model": "m"},
        {"type": "content_block_start", "block_index": 0, "block_type": "thinking"},
        {"type": "reasoning_delta", "reasoning": "hmm"},
        {"type": "text_delta", "text": "answer"},
        {"type": "finish", "finish_reason": {"reason": "stop"}},
    ]

    parts = fold_response_events(events)["choices"][0]["message"]["content"]

    assert parts == [
        {"type": "reasoning", "reasoning": "hmm"},
        {"type": "text", "text": "answer"},
    ]


def test_the_envelope_facts_the_caller_keeps_are_used() -> None:
    """A stream whose ``stream_start`` never reached the fold keeps its identity."""
    folded = fold_response_events(
        [{"type": "finish", "finish_reason": {"reason": "stop"}}],
        response={"id": "r1", "model": "m", "created": 7},
    )

    assert folded["id"] == "r1"
    assert folded["model"] == "m"
    assert folded["created"] == 7


def test_an_echo_field_the_stream_states_is_put_back() -> None:
    """No event carries ``system_fingerprint``; the caller that saw it supplies it.

    A Chat-compatible stream repeats the field on every chunk and the old
    chunk-merger copied it into the body, so a fold that dropped it would hand
    a client less than it got before this reassembly existed.
    """
    folded = fold_response_events(
        [{"type": "finish", "finish_reason": {"reason": "stop"}}],
        response={"system_fingerprint": "fp_abc", "service_tier": "default"},
    )

    assert folded["system_fingerprint"] == "fp_abc"
    assert folded["service_tier"] == "default"


def test_a_response_level_fact_no_event_carries_is_not_invented() -> None:
    """The boundary: what neither an event nor the caller states stays absent.

    ``service_tier`` and ``system_fingerprint`` ride the seed; nothing else
    does, so a caller with an empty seed gets a body without them rather than a
    guessed value.  This is the trade-off ``expand_response`` documents, and a
    fold that filled the gap would make it invisible.
    """
    folded = fold_response_events(
        [{"type": "finish", "finish_reason": {"reason": "stop"}}]
    )

    assert "system_fingerprint" not in folded
    assert "service_tier" not in folded


def test_argument_fragments_accumulate_rather_than_replace() -> None:
    """Every vendor splits a call's arguments across chunks.

    A fold that assigned instead of appending keeps only the last fragment —
    here ``' "n": 2}'``, which is not JSON — so a whole call the client could
    have run is answered as a damaged one, and a fragment that happened to
    parse would drop the rest of the input silently.  The assertion is on the
    value the fragments spell together.
    """
    events = [
        {"type": "stream_start", "response_id": "r1", "model": "m"},
        {
            "type": "tool_call_start",
            "tool_call_id": "call_1",
            "tool_name": "lookup",
            "tool_type": "function",
        },
        {
            "type": "tool_call_delta",
            "tool_call_id": "call_1",
            "arguments_delta": '{"q"',
        },
        {
            "type": "tool_call_delta",
            "tool_call_id": "call_1",
            "arguments_delta": ': "x",',
        },
        {
            "type": "tool_call_delta",
            "tool_call_id": "call_1",
            "arguments_delta": ' "n": 2}',
        },
        {"type": "finish", "finish_reason": {"reason": "tool_calls"}},
    ]

    folded = fold_response_events(events)

    part = folded["choices"][0]["message"]["content"][0]
    assert part["type"] == "tool_call"
    call = cast(Any, part)
    assert call["tool_input"] == {"q": "x", "n": 2}


def test_text_fragments_in_one_block_accumulate() -> None:
    """A dialect may send one block as several deltas, and the fold joins them.

    The deliberate twin of the tool-argument case: assigning instead of
    appending kept only the last fragment, so three ``response.output_text.
    delta`` events in one block answered ``"!"`` instead of the sentence.  The
    writers never open a block twice for one run of text, so nothing else in
    this file can see it.
    """
    events = [
        {"type": "stream_start", "response_id": "r1", "model": "m"},
        {"type": "content_block_start", "block_index": 0, "block_type": "text"},
        {"type": "text_delta", "text": "Hello", "block_index": 0},
        {"type": "text_delta", "text": ", ", "block_index": 0},
        {"type": "text_delta", "text": "world!", "block_index": 0},
        {"type": "content_block_end", "block_index": 0},
        {"type": "finish", "finish_reason": {"reason": "stop"}},
    ]

    parts = fold_response_events(events)["choices"][0]["message"]["content"]

    assert parts == [{"type": "text", "text": "Hello, world!"}]


def test_reasoning_fragments_in_one_block_accumulate() -> None:
    """The same join for reasoning, which is the other field a delta carries."""
    events = [
        {"type": "stream_start", "response_id": "r1", "model": "m"},
        {"type": "content_block_start", "block_index": 0, "block_type": "thinking"},
        {"type": "reasoning_delta", "reasoning": "weigh", "block_index": 0},
        {"type": "reasoning_delta", "reasoning": "ing it", "block_index": 0},
        {"type": "content_block_end", "block_index": 0},
        {"type": "finish", "finish_reason": {"reason": "stop"}},
    ]

    parts = fold_response_events(events)["choices"][0]["message"]["content"]

    assert parts == [{"type": "reasoning", "reasoning": "weighing it"}]


def test_a_usage_report_refines_the_one_before_it() -> None:
    """A dialect may report usage more than once, each report a different subset.

    The later report wins where it states something; a counter it leaves at
    zero is *not stated*, so the earlier value survives.  Both halves are
    asserted: replacing instead of merging loses ``prompt_tokens``, and letting
    a zero through zeroes ``completion_tokens``.
    """
    events = [
        {"type": "stream_start", "response_id": "r1", "model": "m"},
        {
            "type": "usage",
            "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        },
        {
            "type": "usage",
            "usage": {"prompt_tokens": 0, "completion_tokens": 9, "total_tokens": 0},
        },
        {"type": "finish", "finish_reason": {"reason": "stop"}},
    ]

    folded = fold_response_events(events)

    assert folded["usage"] == {
        "prompt_tokens": 5,
        "completion_tokens": 9,
        "total_tokens": 7,
    }


def test_a_second_choices_block_lands_in_its_own_message() -> None:
    """``block_index`` is a stream-wide counter, not a per-choice one.

    The fold opens a part when the block starts, before any delta says which
    choice it belongs to, so a boundary that names no choice put a second
    choice's text in the first choice's message.  Every case in the cross-leg
    table is single-choice, which is why the round-trip property could not see
    it.
    """
    response = cast(IRResponse, _ir_response(Turn(text="first")))
    response["choices"].append(
        {
            "index": 1,
            "message": {
                "role": "assistant",
                "content": [{"type": "text", "text": "second"}],
            },
            "finish_reason": {"reason": "stop"},
        }
    )

    folded = fold_response_events(cast(list[dict], expand_response(response)))

    texts = [
        [cast(Any, part).get("text") for part in choice["message"]["content"]]
        for choice in folded["choices"]
    ]
    assert texts == [["first"], ["second"]]


def test_a_delta_with_no_id_or_index_joins_the_call_just_started() -> None:
    """A dialect may name neither: the open call is the only one it can mean."""
    events = [
        {"type": "stream_start", "response_id": "r1", "model": "m"},
        {"type": "tool_call_start", "tool_call_id": "call_1", "tool_name": "a"},
        {"type": "tool_call_start", "tool_call_id": "call_2", "tool_name": "b"},
        {"type": "tool_call_delta", "tool_call_id": "", "arguments_delta": '{"q": 1}'},
        {"type": "finish", "finish_reason": {"reason": "tool_calls"}},
    ]

    folded = fold_response_events(events)

    calls = folded["choices"][0]["message"]["content"]
    first, second = cast(Any, calls[0]), cast(Any, calls[1])
    assert first["tool_input"] == {}
    assert second["tool_input"] == {"q": 1}


def test_a_citation_is_a_part_of_the_message() -> None:
    """Four readers emit ``citation_delta``; the fold has to place one."""
    citation = {
        "type": "citation",
        "cited_text": "source",
        "url": "https://example.invalid/x",
    }
    events = [
        {"type": "stream_start", "response_id": "r1", "model": "m"},
        {"type": "content_block_start", "block_index": 0, "block_type": "text"},
        {"type": "text_delta", "text": "answer", "block_index": 0},
        {"type": "citation_delta", "citation": dict(citation), "block_index": 0},
        {"type": "finish", "finish_reason": {"reason": "stop"}},
    ]

    folded = fold_response_events(events)

    assert folded["choices"][0]["message"]["content"] == [
        {"type": "text", "text": "answer"},
        citation,
    ]


class TestThePipelineMethod:
    """``ConversionPipeline.convert_stream_events`` — the fold, then the writer.

    The gateway's stream-only bridge calls this for every source format, so
    what it produces has to be the body that dialect would have got from a
    JSON upstream: same content, same terminal reason, same usage.
    """

    @staticmethod
    def _chat_stream() -> tuple[ConversionPipeline, list[dict]]:
        """A Chat pipeline whose request is converted, and the chunks it reads."""
        from llm_rosetta.converters.llm.chat_completions import (
            ChatCompletionsConverter,
        )

        pipeline = ConversionPipeline("chat_completions", "chat_completions")
        pipeline.convert_request(
            {
                "model": "m",
                "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
                "tools": [
                    {
                        "type": "function",
                        "function": {
                            "name": "lookup",
                            "description": "d",
                            "parameters": {
                                "type": "object",
                                "properties": {"q": {"type": "string"}},
                            },
                        },
                    }
                ],
            }
        )
        chunks = [
            {
                "id": "c1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "m",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "content": "hi"}}
                ],
            },
            {
                "id": "c1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "delta": {
                            "tool_calls": [
                                {
                                    "index": 0,
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "lookup",
                                        "arguments": '{"q":"x"}',
                                    },
                                }
                            ]
                        },
                    }
                ],
            },
            {
                "id": "c1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "m",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
            },
        ]
        assert ChatCompletionsConverter  # the pipeline picked this dialect
        return pipeline, chunks

    def _events(self, pipeline, chunks) -> list[dict]:
        """Drive the chunks through the processor, keeping the IR stream."""
        events: list[dict] = []
        processor = pipeline.create_stream_processor(on_ir_event=events.append)
        for chunk in chunks:
            processor.process_chunk(chunk)
        processor.process_chunk(None)
        return events

    def test_a_chat_stream_becomes_a_chat_body(self) -> None:
        """The dialect this reassembly used to special-case is now the same fold."""
        pipeline, chunks = self._chat_stream()

        body = pipeline.convert_stream_events(self._events(pipeline, chunks))

        assert body["object"] == "chat.completion"
        assert body["id"] == "c1"
        choice = body["choices"][0]
        assert choice["message"]["content"] == "hi"
        assert choice["finish_reason"] == "tool_calls"
        call = choice["message"]["tool_calls"][0]
        assert call["id"] == "call_1"
        assert call["function"]["name"] == "lookup"
        # The writer re-serialises the parsed input, so spacing may differ
        # from the fragment; the value is what the client acts on.
        assert json.loads(call["function"]["arguments"]) == {"q": "x"}
        assert body["usage"]["completion_tokens"] == 2

    def test_a_refusal_the_stream_stated_only_on_its_finish(self) -> None:
        """The Chat reader puts a refusal's words on the finish, not in a block.

        ``delta.refusal`` is accumulated on the reader's context and only the
        finish event carries it, but a body spells a refusal as a content
        part.  A fold that copied the metadata and no part answered
        ``refusal: null`` — the field the Chat chunk-merger this replaces kept.
        """
        pipeline = ConversionPipeline("chat_completions", "chat_completions")
        pipeline.convert_request(
            {
                "model": "m",
                "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
            }
        )
        chunks = [
            {
                "id": "c1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "m",
                "choices": [
                    {"index": 0, "delta": {"role": "assistant", "refusal": "no"}}
                ],
            },
            {
                "id": "c1",
                "object": "chat.completion.chunk",
                "created": 1,
                "model": "m",
                "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
            },
        ]

        body = pipeline.convert_stream_events(self._events(pipeline, chunks))

        assert body["choices"][0]["message"]["refusal"] == "no"

    def test_the_fold_needs_a_converted_request(self) -> None:
        """The response leg reads request-leg state; without it there is none."""
        pipeline = ConversionPipeline("chat_completions", "chat_completions")

        with pytest.raises(RuntimeError):
            pipeline.convert_stream_events([])

    def test_a_baseline_pipeline_has_no_ir_leg_to_fold(self) -> None:
        """A same-format passthrough forwards chunks; there is nothing to fold."""
        pipeline = ConversionPipeline(
            "chat_completions", "chat_completions", baseline=True
        )
        pipeline.convert_request(
            {
                "model": "m",
                "stream": False,
                "messages": [{"role": "user", "content": "hi"}],
            }
        )

        with pytest.raises(RuntimeError, match="baseline"):
            pipeline.convert_stream_events([])
