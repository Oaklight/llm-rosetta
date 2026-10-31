"""The open_responses body is a fold of the stream it would have been.

The body writer no longer walks the IR parts on its own: it drives the stream
writer over :func:`llm_rosetta.ir.primitives.expand_response` and reads the
response off the terminal event.  What a whole body carries and no event
carries — the message's ``status``, a text part's ``annotations`` and
``logprobs``, ``service_tier`` — is applied afterwards.

These tests pin both halves: the shapes the fold adds, and the two places the
stream leg itself used to lose content it had already announced (a refusal and
a server tool result).
"""

from __future__ import annotations

from typing import Any

from llm_rosetta.converters.llm.open_responses import OpenResponsesConverter
from llm_rosetta.ir.primitives import expand_response


def _ir(
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
        "id": "resp_1",
        "object": "response",
        "created": 100,
        "model": "gpt-5",
        "choices": [
            {
                "index": index,
                "message": {**message, "content": list(content)},
                "finish_reason": {"reason": reason},
            }
            for index in range(choices)
        ],
    }
    if usage is not None:
        response["usage"] = usage
    return response


def _body(ir: Any) -> dict[str, Any]:
    return OpenResponsesConverter().response_to_provider(ir)


def _stream_terminal(ir: Any) -> dict[str, Any]:
    """Drive the stream writer with the expansion and return the last event."""
    converter = OpenResponsesConverter()
    context = converter.create_stream_context()
    terminal: dict[str, Any] = {}
    for event in expand_response(ir):
        result = converter.stream_response_to_provider(event, context=context)
        for chunk in result if isinstance(result, list) else [result]:
            if (
                isinstance(chunk, dict)
                and chunk.get("type", "").startswith("response.")
                and chunk["type"]
                in {
                    "response.completed",
                    "response.failed",
                    "response.incomplete",
                }
            ):
                terminal = chunk
    return terminal


class TestTheStreamCompletedPayload:
    """The completed payload must carry what the same stream just announced."""

    def test_a_refusal_reaches_the_completed_payload(self):
        terminal = _stream_terminal(
            _ir([{"type": "refusal", "refusal": "no"}], reason="refusal")
        )

        output = terminal["response"]["output"]
        assert output[0]["content"] == [{"type": "refusal", "refusal": "no"}]

    def test_a_server_result_reaches_the_completed_payload(self):
        terminal = _stream_terminal(
            _ir(
                [
                    {
                        "type": "tool_call",
                        "tool_call_id": "c1",
                        "tool_name": "code_interpreter",
                        "tool_input": {"code": "1+1"},
                        "tool_type": "intrinsic",
                        "intrinsic_kind": "code_interpreter",
                    },
                    {
                        "type": "tool_result",
                        "tool_call_id": "c1",
                        "result": "2",
                        "tool_type": "intrinsic",
                        "intrinsic_kind": "code_interpreter",
                    },
                ]
            )
        )

        types = [item["type"] for item in terminal["response"]["output"]]
        assert "code_interpreter_call_output" in types


class TestTheBodyShape:
    """What a whole body carries and an event does not."""

    def test_a_message_item_is_completed_and_its_text_part_annotated(self):
        body = _body(_ir([{"type": "text", "text": "hello"}]))

        item = body["output"][0]
        assert item["status"] == "completed"
        assert item["content"][0]["annotations"] == []
        assert item["content"][0]["logprobs"] == []

    def test_a_citation_lands_on_the_text_part(self):
        body = _body(
            _ir(
                [
                    {"type": "text", "text": "hi"},
                    {
                        "type": "citation",
                        "citation_type": "url_citation",
                        "url": "https://a",
                        "title": "A",
                    },
                ]
            )
        )

        annotations = body["output"][0]["content"][0]["annotations"]
        assert annotations == [
            {"type": "url_citation", "url": "https://a", "title": "A"}
        ]

    def test_service_tier_rides_the_body(self):
        ir = _ir([{"type": "text", "text": "hi"}])
        ir["service_tier"] = "default"

        assert _body(ir)["service_tier"] == "default"

    def test_a_function_call_is_named_with_its_fc_prefix(self):
        body = _body(
            _ir(
                [
                    {
                        "type": "tool_call",
                        "tool_call_id": "c1",
                        "tool_name": "f",
                        "tool_input": {"a": 1},
                    }
                ],
                reason="tool_calls",
            )
        )

        assert body["output"][0]["id"] == "fc_c1"

    def test_a_custom_tool_call_keeps_its_plain_text_input(self):
        body = _body(
            _ir(
                [
                    {
                        "type": "tool_call",
                        "tool_call_id": "c1",
                        "tool_name": "patch",
                        "tool_input": {"input": "*** Begin Patch"},
                        "tool_type": "custom",
                    }
                ],
                reason="tool_calls",
            )
        )

        item = body["output"][0]
        assert item["type"] == "custom_tool_call"
        assert item["input"] == "*** Begin Patch"

    def test_a_reasoning_item_keeps_its_raw_content(self):
        body = _body(
            _ir(
                [
                    {
                        "type": "reasoning",
                        "reasoning": "think",
                        "provider_metadata": {
                            "responses_reasoning_content": [
                                {"type": "reasoning_text", "text": "raw"}
                            ]
                        },
                    }
                ]
            )
        )

        item = body["output"][0]
        assert item["type"] == "reasoning"
        assert item["content"] == [{"type": "reasoning_text", "text": "raw"}]

    def test_each_choice_gets_its_own_message_item(self):
        ir = _ir([{"type": "text", "text": "SAME"}], choices=2)

        ids = [item["id"] for item in _body(ir)["output"]]

        assert len(ids) == 2
        assert len(set(ids)) == 2

    def test_a_failed_turn_carries_the_error_envelope(self):
        body = _body(_ir([{"type": "text", "text": ""}], reason="error"))

        assert body["status"] == "failed"
        assert body["error"]["code"] == "server_error"

    def test_a_client_result_is_written_after_its_call(self):
        """A client result is never a stream item — a body still writes it.

        The stream leg writes a client result on the *request* leg, so the
        fold would drop it from the response entirely; a body has always put
        it right after the call it answers.
        """
        body = _body(
            _ir(
                [
                    {
                        "type": "tool_call",
                        "tool_call_id": "c9",
                        "tool_name": "f",
                        "tool_input": {"a": 1},
                    },
                    {"type": "tool_result", "tool_call_id": "c9", "result": "R"},
                ],
                reason="tool_calls",
            )
        )

        types = [item["type"] for item in body["output"]]
        assert types == ["function_call", "function_call_output"]
        assert body["output"][1]["output"] == "R"

    def test_each_client_result_follows_its_own_call(self):
        """Two calls and two results: each result sits after the call it answers.

        Appending every result at the end would also put the last one in
        place, so only a turn with two calls can tell insertion from
        appending.
        """
        body = _body(
            _ir(
                [
                    {
                        "type": "tool_call",
                        "tool_call_id": "c1",
                        "tool_name": "f",
                        "tool_input": {"a": 1},
                    },
                    {
                        "type": "tool_call",
                        "tool_call_id": "c2",
                        "tool_name": "g",
                        "tool_input": {"b": 2},
                    },
                    {"type": "tool_result", "tool_call_id": "c1", "result": "R1"},
                    {"type": "tool_result", "tool_call_id": "c2", "result": "R2"},
                ],
                reason="tool_calls",
            )
        )

        assert [(item["type"], item["call_id"]) for item in body["output"]] == [
            ("function_call", "c1"),
            ("function_call_output", "c1"),
            ("function_call", "c2"),
            ("function_call_output", "c2"),
        ]
        assert [item["output"] for item in body["output"][1::2]] == ["R1", "R2"]

    def test_an_unrenderable_server_result_reports_the_loss(self):
        """A server result with no item is dropped, and the caller hears it."""
        from llm_rosetta.conversion_warnings import WarningSink, recording_to

        caller = WarningSink(leg="caller")
        with recording_to(caller):
            body = _body(
                _ir(
                    [
                        {
                            "type": "tool_call",
                            "tool_call_id": "c1",
                            "tool_name": "image_generation",
                            "tool_input": {},
                            "tool_type": "intrinsic",
                            "intrinsic_kind": "image_generation",
                        },
                        {
                            "type": "tool_result",
                            "tool_call_id": "c1",
                            "result": "x",
                            "tool_type": "intrinsic",
                            "intrinsic_kind": "image_generation",
                        },
                    ]
                )
            )

        assert [item["type"] for item in body["output"]] == []
        assert [(w.subject, w.code) for w in caller] == [
            ("image_generation", "dropped")
        ]
