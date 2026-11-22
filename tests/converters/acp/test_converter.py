"""The ACP lane as a converter: turn bodies in, IR responses out.

The lane is a target only.  These tests pin the three things that make it
usable in that direction — the turn body it writes, the response it assembles
from collected updates, and the refusals that keep it from looking like a
client source format.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.llm.acp import AcpConverter, target_finish_reason
from llm_rosetta.conversion_warnings import ConversionWarning, WarningSink
from llm_rosetta.types.ir import IRRequest, IRResponse


def _converter() -> AcpConverter:
    return AcpConverter()


def _context() -> ConversionContext:
    ctx = ConversionContext()
    ctx.warnings = WarningSink(leg="request")
    return ctx


class TestTheTurnBody:
    """What the request leg writes: model, system string, labelled blocks."""

    def test_a_conversation_becomes_labelled_prompt_blocks(self):
        ir: IRRequest = {
            "model": "gemini-3.1-pro",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
                {"role": "user", "content": [{"type": "text", "text": "and then?"}]},
            ],
        }
        body, _warnings = _converter().request_to_provider(ir)

        assert body["model"] == "gemini-3.1-pro"
        assert body["prompt"] == [
            {"type": "text", "text": "[User]\nhi"},
            {"type": "text", "text": "[Assistant]\nhello"},
            {"type": "text", "text": "[User]\nand then?"},
        ]

    def test_the_system_instruction_is_one_flat_string(self):
        ir: IRRequest = {
            "model": "m",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "system_instruction": [
                {"type": "text", "text": "Be brief."},
                {"type": "text", "text": "Be exact."},
            ],
        }
        body, _ = _converter().request_to_provider(ir)

        assert body["system"] == "Be brief.\nBe exact."

    def test_an_image_travels_beside_its_message_text(self):
        ir: IRRequest = {
            "model": "m",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this?"},
                        {
                            "type": "image",
                            "image_data": {"data": "QUJD", "media_type": "image/jpeg"},
                        },
                    ],
                }
            ],
        }
        body, _ = _converter().request_to_provider(ir)

        assert body["prompt"][0] == {"type": "text", "text": "[User]\nwhat is this?"}
        assert body["prompt"][1] == {
            "type": "image",
            "data": "QUJD",
            "mimeType": "image/jpeg",
        }

    def test_tool_declarations_are_written_in_mcp_shape(self):
        """The transport serves these from its loopback MCP endpoint."""
        ir: IRRequest = {
            "model": "m",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "description": "read a file",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        }
        ctx = _context()
        body, warnings = _converter().request_to_provider(ir, context=ctx)

        assert body["tools"] == [
            {
                "name": "read_file",
                "description": "read a file",
                "parameters": {"type": "object", "properties": {}},
            }
        ]
        assert not any("tool" in str(w).lower() for w in warnings)

    def test_generation_settings_are_reported_one_by_one(self):
        """A turn carries a prompt and a model; sampling parameters are lost."""
        ir: IRRequest = {
            "model": "m",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            "generation": {"temperature": 0.2, "max_tokens": 64},
        }
        ctx = _context()
        body, warnings = _converter().request_to_provider(ir, context=ctx)

        assert "temperature" not in body
        losses = [w for w in warnings if isinstance(w, ConversionWarning)]
        assert {w.subject for w in losses} == {
            "generation.temperature",
            "generation.max_tokens",
        }
        assert {w.code for w in losses} == {"dropped"}

    def test_an_unset_setting_is_not_reported_again(self):
        """``None`` is reported once, by the pass that drops it, not by ACP."""
        ir: IRRequest = {
            "model": "m",
            "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
            # ``None`` is how a source writes "named no value"; the shared
            # pass drops it before the converter sees the request.
            "generation": cast(Any, {"temperature": 0.2, "top_p": None}),
        }
        ctx = _context()
        _body, warnings = _converter().request_to_provider(ir, context=ctx)

        losses = [w for w in warnings if isinstance(w, ConversionWarning)]
        reported = [w for w in losses if "sampling parameters" in str(w)]
        assert [w.subject for w in reported] == ["generation.temperature"]


class TestTheResponse:
    """What the response leg assembles from the transport's chunk list."""

    def test_updates_become_parts_with_a_mapped_finish_reason(self):
        provider_response = {
            "updates": [
                {"type": "session_start", "session_id": "sess-1", "model": "gemini-x"},
                {"type": "thought", "text": "hmm"},
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": " there"},
                {"type": "usage", "used": 42, "size": 1000},
                {"type": "stop", "stop_reason": "max_tokens"},
            ],
            "stop_reason": "max_tokens",
            "session_id": "sess-1",
            "model": "gemini-x",
            "created": 1700000000,
        }
        ir = _converter().response_from_provider(provider_response)

        assert ir["id"] == "sess-1"
        assert ir["model"] == "gemini-x"
        choice = ir["choices"][0]
        assert choice["message"]["content"] == [
            {"type": "reasoning", "reasoning": "hmm"},
            {"type": "text", "text": "Hello there"},
        ]
        assert choice["finish_reason"] == {"reason": "length"}

    def test_an_unknown_stop_reason_is_a_plain_stop(self):
        """The agent ended the turn; what arrived is the answer."""
        assert target_finish_reason("something_new")["reason"] == "stop"

    def test_an_empty_turn_still_has_one_text_part(self):
        ir = _converter().response_from_provider({"updates": []})

        assert ir["choices"][0]["message"]["content"] == [{"type": "text", "text": ""}]

    def test_a_tool_call_makes_a_tool_use_part_and_a_tool_finish(self):
        provider_response = {
            "updates": [
                {
                    "type": "tool_call",
                    "tool_call_id": "call_1",
                    "tool_name": "echo",
                    "tool_input": {"text": "hi"},
                },
                {"type": "stop", "stop_reason": "tool_calls"},
            ],
            "stop_reason": "tool_calls",
            "session_id": "sess-1",
            "model": "m",
        }
        ir = _converter().response_from_provider(provider_response)

        choice = ir["choices"][0]
        assert choice["message"]["content"] == [
            {
                "type": "tool_call",
                "tool_call_id": "call_1",
                "tool_name": "echo",
                "tool_input": {"text": "hi"},
                "tool_type": "function",
            }
        ]
        assert choice["finish_reason"] == {"reason": "tool_calls"}


class TestTheStream:
    """A transport chunk becomes the stream events a chat client reads."""

    def test_a_tool_call_chunk_becomes_tool_use_events(self):
        events = _converter().stream_response_from_provider(
            {
                "type": "tool_call",
                "tool_call_id": "call_1",
                "tool_name": "echo",
                "tool_input": {"text": "hi"},
            }
        )

        assert [event["type"] for event in events] == [
            "content_block_start",
            "tool_call_start",
            "tool_call_delta",
        ]
        tool = [cast(dict[str, Any], event) for event in events]
        assert tool[0]["block_type"] == "tool_use"
        assert tool[1]["tool_call_id"] == "call_1"
        assert tool[1]["tool_name"] == "echo"
        assert json.loads(tool[2]["arguments_delta"]) == {"text": "hi"}

    def test_a_stop_chunk_carries_the_tool_finish_reason(self):
        events = _converter().stream_response_from_provider(
            {"type": "stop", "stop_reason": "tool_calls"}
        )

        assert events == [{"type": "finish", "finish_reason": {"reason": "tool_calls"}}]


class TestTargetOnly:
    """The lane never writes the client's side."""

    def test_a_response_is_refused(self):
        ir = IRResponse(
            id="r",
            object="response",
            created=0,
            model="m",
            choices=[
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": []},
                    "finish_reason": {"reason": "stop"},
                }
            ],
        )
        with pytest.raises(NotImplementedError, match="upstream lane"):
            _converter().response_to_provider(ir)


class TestReadingTheBodyBack:
    """A body this lane wrote can be read back, which is what round-trips use."""

    def test_the_labels_survive_a_round_trip(self):
        ir: IRRequest = {
            "model": "m",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hello"}]},
            ],
        }
        body, _ = _converter().request_to_provider(ir)
        back = _converter().request_from_provider(body)

        messages = cast(list[dict[str, Any]], back["messages"])
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[1]["content"] == [{"type": "text", "text": "hello"}]
