"""A custom tool call's input text through the Open Responses writer and reader (#1720).

The IR holds a custom call's text as ``{"input": text}``; a JSON object in the
text stays structured.  The body reader keeps every other text, empty and JSON
scalars included, as the string the wire carried.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from llm_rosetta.converters.llm.chat_completions import ChatCompletionsConverter
from llm_rosetta.converters.llm.google_generate import GoogleGenerateConverter
from llm_rosetta.converters.llm.messages import MessagesConverter
from llm_rosetta.converters.llm.open_responses import OpenResponsesConverter
from llm_rosetta.ir.primitives import expand_response, fold_response_events
from llm_rosetta.ir.tools.custom_input import custom_input_from_text, custom_input_text


def _ir_with_custom_call(tool_input: dict[str, Any]) -> Any:
    return {
        "id": "r",
        "object": "response",
        "created": 1,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "tool_name": "n",
                            "tool_input": tool_input,
                            "tool_type": "custom",
                        }
                    ],
                },
                "finish_reason": {"reason": "tool_calls"},
            }
        ],
    }


def _custom_body(text: str) -> dict[str, Any]:
    return {
        "id": "r",
        "status": "completed",
        "model": "m",
        "output": [
            {
                "type": "custom_tool_call",
                "call_id": "call_1",
                "name": "n",
                "input": text,
            }
        ],
    }


def _find(node: Any, key: str) -> Any:
    """The first value stored under *key* anywhere in a nested body."""
    if isinstance(node, dict):
        if key in node:
            return node[key]
        for value in node.values():
            found = _find(value, key)
            if found is not None:
                return found
    elif isinstance(node, list):
        for value in node:
            found = _find(value, key)
            if found is not None:
                return found
    return None


def _call_input(ir: Any) -> Any:
    return ir["choices"][0]["message"]["content"][0]["tool_input"]


class TestWriterDirection:
    @pytest.mark.parametrize("text", ["", "0", "true", "[1, 2]"])
    def test_the_body_writer_puts_the_text_on_the_wire(self, text):
        body = OpenResponsesConverter().response_to_provider(
            _ir_with_custom_call({"input": text})
        )
        (item,) = body["output"]
        assert item["type"] == "custom_tool_call"
        assert item["input"] == text

    @pytest.mark.parametrize("text", ["", "0", "true", "[1, 2]"])
    def test_a_written_body_reads_back_as_the_same_ir_input(self, text):
        converter = OpenResponsesConverter()
        ir = _ir_with_custom_call({"input": text})
        back = converter.response_from_provider(converter.response_to_provider(ir))
        assert _call_input(back) == {"input": text}


class TestCrossFormat:
    @pytest.mark.parametrize("text", ["0", "true", ""])
    def test_scalar_text_survives_a_write_as_chat(self, text):
        """Chat has a native custom call: the text is its ``input`` as it was."""
        ir = OpenResponsesConverter().response_from_provider(_custom_body(text))
        out = ChatCompletionsConverter().response_to_provider(ir)
        assert _find(out, "custom")["input"] == text

    @pytest.mark.parametrize(
        ("writer", "key"),
        [(MessagesConverter, "input"), (GoogleGenerateConverter, "args")],
        ids=["messages", "google"],
    )
    @pytest.mark.parametrize("text", ["0", "true", ""])
    def test_scalar_text_survives_a_downgrade_to_a_function_call(
        self, writer, key, text
    ):
        """Without a custom call the text is a function call's one ``input``
        argument: still the string, not a decoded value."""
        ir = OpenResponsesConverter().response_from_provider(_custom_body(text))
        out = writer().response_to_provider(ir)
        assert _find(out, key) == {"input": text}


class TestPinnedAmbiguity:
    def test_pin_text_that_is_the_wrapper_object_reads_as_the_dict(self):
        """PIN, not a rule: input text that is itself ``{"input": "x"}`` reads as
        the dict, so the IR cannot tell it from the text ``x``.  The wire text
        is lost; the writer then sends ``x``."""
        converter = OpenResponsesConverter()
        ir = converter.response_from_provider(_custom_body('{"input": "x"}'))
        assert _call_input(ir) == {"input": "x"}
        body = converter.response_to_provider(ir)
        assert body["output"][0]["input"] == "x"


class TestStreamWriter:
    """The stream writer sends the text on the wire, as the body writer does (#1736)."""

    @staticmethod
    def _chunks(
        delta_events: list[str], *, wrapped: bool = False
    ) -> list[dict[str, Any]]:
        flags = {"input_wrapped": True} if wrapped else {}
        converter = OpenResponsesConverter()
        context = converter.create_stream_context()
        events: list[Any] = [
            {"type": "stream_start", "response_id": "r", "model": "m"},
            {
                "type": "tool_call_start",
                "tool_call_id": "call_1",
                "tool_name": "n",
                "tool_type": "custom",
            },
        ]
        events += [
            {
                "type": "tool_call_delta",
                "tool_call_id": "call_1",
                "arguments_delta": d,
                **flags,
            }
            for d in delta_events
        ]
        events.append({"type": "finish", "finish_reason": {"reason": "tool_calls"}})
        out: list[dict[str, Any]] = []
        for event in events:
            got = converter.stream_response_to_provider(event, context=context)
            out.extend(got if isinstance(got, list) else [got])
        return out

    @staticmethod
    def _input(chunks: list[dict[str, Any]]) -> str:
        (done,) = [
            c["item"] for c in chunks if c.get("type") == "response.output_item.done"
        ]
        return done["input"]

    @pytest.mark.parametrize("text", ["0", "plain text", "true"])
    def test_a_wrapper_delta_is_written_as_its_text(self, text):
        chunks = self._chunks([json.dumps({"input": text})], wrapped=True)
        assert self._input(chunks) == text
        deltas = [c["delta"] for c in chunks if "input.delta" in c.get("type", "")]
        assert deltas == [text]

    def test_an_empty_wrapper_delta_is_written_as_the_empty_string(self):
        assert self._input(self._chunks(['{"input": ""}'], wrapped=True)) == ""

    @pytest.mark.parametrize(
        "deltas",
        [['{"input":"x"}'], ['{"input":', '"x"}'], ['{"input": 5}']],
        ids=["one-delta", "split", "number"],
    )
    def test_raw_text_that_looks_like_the_wrapper_is_never_unwrapped(self, deltas):
        """Only a delta flagged ``input_wrapped`` is unwrapped, so raw text reaches
        the wire as sent, whatever it looks like and however it is chunked."""
        assert self._input(self._chunks(deltas)) == "".join(deltas)

    @pytest.mark.parametrize("text", ["0", "plain text", ""])
    def test_an_expanded_body_is_written_as_its_text(self, text):
        """The events ``expand_response`` builds from a body carry the flag."""
        ir: Any = _ir_with_custom_call({"input": text})
        converter = OpenResponsesConverter()
        context = converter.create_stream_context()
        chunks: list[dict[str, Any]] = []
        for event in expand_response(ir):
            got = converter.stream_response_to_provider(event, context=context)
            chunks.extend(got if isinstance(got, list) else [got])
        assert self._input(chunks) == text

    def test_raw_text_fragments_pass_through(self):
        assert self._input(self._chunks(["plain ", "text"])) == "plain text"


class TestFold:
    """Folding a custom call's streamed text gives the same input the body reader does."""

    @staticmethod
    def _folded(deltas: list[str]) -> Any:
        events: list[Any] = [
            {"type": "stream_start", "response_id": "r", "model": "m"},
            {
                "type": "tool_call_start",
                "tool_call_id": "c",
                "tool_name": "n",
                "tool_type": "custom",
                "tool_call_index": 0,
            },
        ]
        events += [
            {"type": "tool_call_delta", "tool_call_id": "c", "arguments_delta": d}
            for d in deltas
        ]
        events.append({"type": "finish", "finish_reason": {"reason": "tool_calls"}})
        folded: Any = fold_response_events(events)
        call = folded["choices"][0]["message"]["content"][0]
        return call["tool_input"]

    @pytest.mark.parametrize("text", ["0", "true", "plain text", "[1]"])
    def test_raw_text_deltas_fold_to_the_wrapped_text(self, text):
        assert self._folded([text]) == {"input": text}
        assert self._folded([text[:1], text[1:]]) == {"input": text}

    def test_no_delta_folds_to_an_empty_input(self):
        assert self._folded([]) == {"input": ""}

    def test_a_json_object_text_folds_to_the_object(self):
        assert self._folded(['{"a": 1}']) == {"a": 1}


class TestHelpers:
    def test_a_value_json_cannot_encode_falls_back_to_str_without_raising(self):
        assert custom_input_text({"input": b"x"}) == str(b"x")
        assert custom_input_text({"input": {1}}) == str({1})

    @pytest.mark.parametrize(
        ("value", "wire"),
        [(True, "true"), (False, "false"), (None, "null"), ([1], "[1]"), (5, "5")],
    )
    def test_a_non_string_input_is_written_as_json_not_a_python_repr(self, value, wire):
        assert custom_input_text({"input": value}) == wire
        body = OpenResponsesConverter().response_to_provider(
            _ir_with_custom_call({"input": value})
        )
        assert body["output"][0]["input"] == wire

    @pytest.mark.parametrize("text", ["", "0", "plain", " "])
    def test_text_and_input_round_trip(self, text):
        assert custom_input_text(custom_input_from_text(text)) == text
