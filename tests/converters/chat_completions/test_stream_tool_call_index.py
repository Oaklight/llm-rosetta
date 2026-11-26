"""The stream writer keys a tool call's ``index`` by the call, not by 0.

``tool_call_index`` is optional on the IR tool-call events while
``tool_call_id`` is required (#1719).  An event without the index must land on
its own call's slot, so two calls never merge into ``tool_calls[0]``.
"""

from typing import Any

import pytest

from llm_rosetta.converters.llm.chat_completions import ChatCompletionsConverter
from llm_rosetta.converters.llm.google_generate import GoogleGenerateConverter
from llm_rosetta.converters.llm.google_interactions import GoogleInteractionsConverter
from llm_rosetta.converters.llm.messages import MessagesConverter
from llm_rosetta.converters.llm.open_responses import OpenResponsesConverter
from llm_rosetta.ir.context import StreamContext


def _entries(converter: ChatCompletionsConverter, events: list[Any]) -> list[dict]:
    """Write ``events`` and return every ``tool_calls`` entry in order."""
    context = StreamContext()
    entries: list[dict] = []
    for event in events:
        chunk = converter.stream_response_to_provider(event, context)
        assert isinstance(chunk, dict)
        for choice in chunk.get("choices", []):
            entries.extend(choice["delta"].get("tool_calls", []))
    return entries


class TestToolCallIndexFromContext:
    def setup_method(self):
        self.converter = ChatCompletionsConverter()

    def test_events_without_index_use_the_call_position(self):
        entries = _entries(
            self.converter,
            [
                {"type": "stream_start", "response_id": "r", "model": "m"},
                {"type": "tool_call_start", "tool_call_id": "a", "tool_name": "f"},
                {"type": "tool_call_start", "tool_call_id": "b", "tool_name": "g"},
                {
                    "type": "tool_call_delta",
                    "tool_call_id": "a",
                    "arguments_delta": "1",
                },
                {
                    "type": "tool_call_delta",
                    "tool_call_id": "b",
                    "arguments_delta": "2",
                },
            ],
        )
        assert [e["index"] for e in entries] == [0, 1, 0, 1]
        assert entries[2]["function"] == {"arguments": "1"}
        assert entries[3]["function"] == {"arguments": "2"}

    def test_a_source_position_does_not_leak_into_the_slot(self):
        """Chat counts from 0 without gaps, whatever index the source gave."""
        entries = _entries(
            self.converter,
            [
                {"type": "stream_start", "response_id": "r", "model": "m"},
                {
                    "type": "tool_call_start",
                    "tool_call_id": "a",
                    "tool_name": "f",
                    "tool_call_index": 3,
                },
                {
                    "type": "tool_call_delta",
                    "tool_call_id": "a",
                    "arguments_delta": "1",
                    "tool_call_index": 3,
                },
            ],
        )
        assert [e["index"] for e in entries] == [0, 0]

    def test_dropped_intrinsic_call_does_not_take_a_slot(self):
        entries = _entries(
            self.converter,
            [
                {"type": "stream_start", "response_id": "r", "model": "m"},
                {
                    "type": "tool_call_start",
                    "tool_call_id": "srv",
                    "tool_name": "web_search",
                    "tool_type": "intrinsic",
                },
                {"type": "tool_call_start", "tool_call_id": "a", "tool_name": "f"},
                {
                    "type": "tool_call_delta",
                    "tool_call_id": "a",
                    "arguments_delta": "1",
                },
            ],
        )
        assert [e["index"] for e in entries] == [0, 0]

    def test_index_counts_per_choice(self):
        """``tool_calls[].index`` restarts at 0 in every choice."""
        events: list[Any] = [{"type": "stream_start", "response_id": "r", "model": "m"}]
        for choice in (0, 1):
            for call in ("x", "y"):
                events.append(
                    {
                        "type": "tool_call_start",
                        "tool_call_id": f"{call}{choice}",
                        "tool_name": "f",
                        "choice_index": choice,
                    }
                )
        for choice in (0, 1):
            for call in ("x", "y"):
                events.append(
                    {
                        "type": "tool_call_delta",
                        "tool_call_id": f"{call}{choice}",
                        "arguments_delta": "{}",
                        "choice_index": choice,
                    }
                )
        entries = _entries(self.converter, events)
        assert [e["index"] for e in entries] == [0, 1, 0, 1, 0, 1, 0, 1]

    def test_the_same_call_id_in_two_choices_has_a_slot_in_each(self):
        events: list[Any] = [{"type": "stream_start", "response_id": "r", "model": "m"}]
        for choice, ids in ((0, ("x", "y")), (1, ("y",))):
            for call in ids:
                events.append(
                    {
                        "type": "tool_call_start",
                        "tool_call_id": call,
                        "tool_name": "f",
                        "choice_index": choice,
                    }
                )
        for choice in (0, 1):
            events.append(
                {
                    "type": "tool_call_delta",
                    "tool_call_id": "y",
                    "arguments_delta": "{}",
                    "choice_index": choice,
                }
            )
        entries = _entries(self.converter, events)
        assert [e["index"] for e in entries] == [0, 1, 0, 1, 0]

    def test_a_delta_for_a_call_that_never_opened_gets_its_own_slot(self):
        entries = _entries(
            self.converter,
            [
                {"type": "stream_start", "response_id": "r", "model": "m"},
                {"type": "tool_call_start", "tool_call_id": "a", "tool_name": "f"},
                {
                    "type": "tool_call_delta",
                    "tool_call_id": "a",
                    "arguments_delta": "A",
                },
                {
                    "type": "tool_call_delta",
                    "tool_call_id": "z",
                    "arguments_delta": "Z",
                },
            ],
        )
        assert [(e["index"], e["function"].get("arguments")) for e in entries[1:]] == [
            (0, "A"),
            (1, "Z"),
        ]

    def test_an_id_reused_for_a_second_call_opens_a_second_slot(self):
        entries = _entries(
            self.converter,
            [
                {"type": "stream_start", "response_id": "r", "model": "m"},
                {"type": "tool_call_start", "tool_call_id": "a", "tool_name": "f"},
                {
                    "type": "tool_call_delta",
                    "tool_call_id": "a",
                    "arguments_delta": "1",
                },
                {"type": "tool_call_start", "tool_call_id": "a", "tool_name": "f"},
                {
                    "type": "tool_call_delta",
                    "tool_call_id": "a",
                    "arguments_delta": "2",
                },
            ],
        )
        assert [e["index"] for e in entries] == [0, 0, 1, 1]

    def test_without_a_context_the_index_is_zero(self):
        """With no context there is nothing to resolve from: index 0."""
        start = self.converter.stream_response_to_provider(
            {"type": "tool_call_start", "tool_call_id": "a", "tool_name": "f"}, None
        )
        delta = self.converter.stream_response_to_provider(
            {"type": "tool_call_delta", "tool_call_id": "a", "arguments_delta": "1"},
            None,
        )
        assert isinstance(start, dict) and isinstance(delta, dict)
        assert start["choices"][0]["delta"]["tool_calls"][0]["index"] == 0
        assert delta["choices"][0]["delta"]["tool_calls"][0]["index"] == 0


def _ir_stream_with_text_before_calls() -> list[Any]:
    """A turn with text, then two calls whose deltas carry no index."""
    return [
        {"type": "stream_start", "response_id": "r", "model": "m"},
        {"type": "text_delta", "text": "hi"},
        {"type": "tool_call_start", "tool_call_id": "a", "tool_name": "f"},
        {"type": "tool_call_start", "tool_call_id": "b", "tool_name": "g"},
        {"type": "tool_call_delta", "tool_call_id": "a", "arguments_delta": '{"a":1}'},
        {"type": "tool_call_delta", "tool_call_id": "b", "arguments_delta": '{"b":2}'},
        {"type": "finish", "finish_reason": {"reason": "tool_calls"}},
    ]


@pytest.mark.parametrize(
    "source",
    [MessagesConverter, OpenResponsesConverter, GoogleInteractionsConverter],
    ids=["messages", "open_responses", "google_interactions"],
)
def test_a_foreign_stream_written_as_chat_keeps_each_call_on_its_own_slot(source):
    """The source's block / output positions do not leak into Chat's index.

    A text block (Messages) or message item (Open Responses) comes before the
    calls, so the source's own index of the first call is not 0.  Chat counts
    tool calls from 0 and the second call's deltas must reach the second slot.
    """
    src = source()
    src_context = src.create_stream_context()
    chunks: list[dict] = []
    for event in _ir_stream_with_text_before_calls():
        out = src.stream_response_to_provider(event, context=src_context)
        chunks.extend(out if isinstance(out, list) else [out])
    read_context = src.create_stream_context()
    ir_events = list(
        src.iter_stream_from_provider([c for c in chunks if c], context=read_context)
    )

    chat = ChatCompletionsConverter()
    chat_context = chat.create_stream_context()
    entries: list[dict] = []
    for event in ir_events:
        out = chat.stream_response_to_provider(event, context=chat_context)
        for chunk in out if isinstance(out, list) else [out]:
            for choice in chunk.get("choices", []):
                entries.extend(choice["delta"].get("tool_calls", []))

    opens = [e for e in entries if "id" in e]
    assert [e["index"] for e in opens] == [0, 1]
    args: dict[int, str] = {}
    for entry in entries:
        if "function" in entry and "arguments" in entry["function"]:
            args[entry["index"]] = (
                args.get(entry["index"], "") + (entry["function"]["arguments"])
            )
    assert args == {0: '{"a":1}', 1: '{"b":2}'}


def test_google_generate_candidates_each_count_their_calls_from_zero():
    """Google numbers calls across candidates (2, 3 in the second); Chat per choice."""
    google = GoogleGenerateConverter()

    def candidate(index: int, names: tuple[str, str]) -> dict:
        return {
            "index": index,
            "content": {
                "role": "model",
                "parts": [{"functionCall": {"name": n, "args": {}}} for n in names],
            },
            "finishReason": "STOP",
        }

    chunk = {"candidates": [candidate(0, ("f", "g")), candidate(1, ("f", "g"))]}
    ir_events = list(
        google.iter_stream_from_provider(
            [chunk], context=google.create_stream_context()
        )
    )
    chat = ChatCompletionsConverter()
    context = chat.create_stream_context()
    opens: list[tuple[int, int]] = []
    for event in ir_events:
        out = chat.stream_response_to_provider(event, context=context)
        for chunk_out in out if isinstance(out, list) else [out]:
            for choice in chunk_out.get("choices", []):
                for entry in choice["delta"].get("tool_calls", []):
                    if "id" in entry:
                        opens.append((choice["index"], entry["index"]))
    assert opens == [(0, 0), (0, 1), (1, 0), (1, 1)]


def test_pin_a_chat_source_that_reuses_an_id_stays_one_call():
    """PIN, not a rule: the Chat reader keys a call by id, so a second tool call
    that reuses the first one's id (it logs "Conflicting Chat Completions tool
    IDs") yields no second start; both argument strings land on slot 0."""

    def chunk(index: int, name: str, arguments: str) -> dict:
        call = {
            "index": index,
            "id": "p",
            "type": "function",
            "function": {"name": name, "arguments": arguments},
        }
        return {
            "id": "x",
            "object": "chat.completion.chunk",
            "model": "m",
            "choices": [
                {"index": 0, "delta": {"tool_calls": [call]}, "finish_reason": None}
            ],
        }

    chat = ChatCompletionsConverter()
    ir_events = list(
        chat.iter_stream_from_provider(
            [chunk(0, "f", '{"p":1}'), chunk(1, "g", '{"q":2}')],
            context=chat.create_stream_context(),
        )
    )
    assert [e["type"] for e in ir_events].count("tool_call_start") == 1

    writer = ChatCompletionsConverter()
    context = writer.create_stream_context()
    entries: list[dict] = []
    for event in ir_events:
        out = writer.stream_response_to_provider(event, context=context)
        for chunk_out in out if isinstance(out, list) else [out]:
            for choice in chunk_out.get("choices", []):
                entries.extend(choice["delta"].get("tool_calls", []))
    assert [e["index"] for e in entries] == [0, 0, 0]
    assert (
        "".join(e["function"]["arguments"] for e in entries if "function" in e)
        == '{"p":1}{"q":2}'
    )
