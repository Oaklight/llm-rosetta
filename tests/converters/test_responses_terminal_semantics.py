"""Responses terminal semantics at the public Chat conversion boundary."""

import json
from typing import Any, cast

import pytest

from llm_rosetta.converters.llm.open_responses import OpenResponsesConverter
from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.pipeline import ConversionError, ConversionPipeline
from llm_rosetta.types.ir import FinishEvent


FINISH_CASES = [
    pytest.param("completed", "text", None, "stop", id="completed-text"),
    pytest.param("completed", "function", None, "tool_calls", id="completed-function"),
    pytest.param("completed", "custom", None, "tool_calls", id="completed-custom"),
    pytest.param("incomplete", "text", "max_output_tokens", "length", id="length-text"),
    pytest.param(
        "incomplete", "function", "max_output_tokens", "length", id="length-function"
    ),
    pytest.param(
        "incomplete", "custom", "max_output_tokens", "length", id="length-custom"
    ),
    pytest.param(
        "incomplete", "text", "content_filter", "content_filter", id="filter-text"
    ),
    pytest.param(
        "incomplete",
        "function",
        "content_filter",
        "content_filter",
        id="filter-function",
    ),
    pytest.param(
        "incomplete", "custom", "content_filter", "content_filter", id="filter-custom"
    ),
]


def _response(
    status: str,
    kind: str = "text",
    incomplete_reason: str | None = None,
    *,
    has_output: bool = True,
) -> dict[str, Any]:
    item_status = "incomplete" if status == "incomplete" else "completed"
    item: dict[str, Any]
    if kind == "function":
        item = {
            "id": "fc_example",
            "type": "function_call",
            "status": item_status,
            "call_id": "call_example",
            "name": "lookup",
            "arguments": '{"key":"example"}',
        }
    elif kind == "custom":
        item = {
            "id": "ct_example",
            "type": "custom_tool_call",
            "status": item_status,
            "call_id": "call_example",
            "name": "lookup",
            "input": "example input",
        }
    else:
        item = {
            "id": "msg_example",
            "type": "message",
            "role": "assistant",
            "status": item_status,
            "content": [
                {"type": "output_text", "text": "Example output", "annotations": []}
            ],
        }
    response: dict[str, Any] = {
        "id": "resp_example",
        "object": "response",
        "created_at": 1,
        "model": "example-model",
        "status": status,
        "output": [item] if has_output else [],
        "usage": {"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
    }
    if incomplete_reason is not None:
        response["incomplete_details"] = {"reason": incomplete_reason}
    if status == "failed":
        response["error"] = {
            "type": "example_error",
            "code": "server_error",
            "message": "Example upstream failure after partial output.",
            "details": {"attempt": 1},
        }
    return response


def _pipeline() -> ConversionPipeline:
    pipeline = ConversionPipeline(
        "chat_completions", "open_responses", supports_custom_tools=True
    )
    pipeline.convert_request(
        {
            "model": "example-model",
            "messages": [{"role": "user", "content": "Look up a value."}],
        }
    )
    return pipeline


def _stream(response: dict[str, Any]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [
        {
            "type": "response.created",
            "response": {
                **response,
                "status": "in_progress",
                "output": [],
                "error": None,
                "incomplete_details": None,
            },
        }
    ]
    for item in response["output"]:
        initial = {**item, "status": "in_progress"}
        if item["type"] == "function_call":
            initial["arguments"] = ""
            delta_type = "response.function_call_arguments.delta"
            delta = item["arguments"]
        elif item["type"] == "custom_tool_call":
            initial["input"] = ""
            delta_type = "response.custom_tool_call_input.delta"
            delta = item["input"]
        else:
            initial["content"] = []
            delta_type = "response.output_text.delta"
            delta = item["content"][0]["text"]
        events.extend(
            [
                {
                    "type": "response.output_item.added",
                    "output_index": 0,
                    "item": initial,
                },
                {
                    "type": delta_type,
                    "item_id": item["id"],
                    "output_index": 0,
                    "content_index": 0,
                    "delta": delta,
                },
                {"type": "response.output_item.done", "output_index": 0, "item": item},
            ]
        )
    # Cancellation is a response status, not a separate standard SSE event.
    terminal = "completed" if response["status"] == "cancelled" else response["status"]
    events.append({"type": f"response.{terminal}", "response": response})
    for sequence_number, event in enumerate(events):
        event["sequence_number"] = sequence_number
    return events


@pytest.mark.parametrize("status,kind,incomplete_reason,expected", FINISH_CASES)
def test_json_finish_reason_preserves_terminal_status(
    status, kind, incomplete_reason, expected
):
    converted = _pipeline().convert_response(_response(status, kind, incomplete_reason))
    choice = converted["choices"][0]
    assert choice["finish_reason"] == expected
    if kind == "function":
        call = choice["message"]["tool_calls"][0]
        assert call["id"] == "call_example"
        assert call["function"]["name"] == "lookup"
        assert json.loads(call["function"]["arguments"]) == {"key": "example"}
    elif kind == "custom":
        call = choice["message"]["tool_calls"][0]
        assert call["type"] == "custom"
        assert call["custom"] == {"name": "lookup", "input": "example input"}
    else:
        assert choice["message"]["content"] == "Example output"


@pytest.mark.parametrize(
    "incomplete_reason,expected",
    [("max_output_tokens", "length"), ("content_filter", "content_filter")],
)
def test_empty_json_incomplete_preserves_terminal_reason(incomplete_reason, expected):
    response = _response(
        "incomplete", incomplete_reason=incomplete_reason, has_output=False
    )
    choice = _pipeline().convert_response(response)["choices"][0]
    assert choice["finish_reason"] == expected
    assert choice["message"]["content"] == ""


@pytest.mark.parametrize("status,kind,incomplete_reason,expected", FINISH_CASES)
def test_sse_finish_reason_preserves_terminal_status(
    status, kind, incomplete_reason, expected
):
    processor = _pipeline().create_stream_processor()
    chunks = [
        converted
        for chunk in _stream(_response(status, kind, incomplete_reason))
        for converted in processor.process_chunk(chunk)
    ]
    choices = [choice for chunk in chunks for choice in chunk.get("choices", [])]
    assert [
        choice["finish_reason"] for choice in choices if choice.get("finish_reason")
    ] == [expected]
    assert [
        chunk["usage"]["total_tokens"] for chunk in chunks if chunk.get("usage")
    ] == [5]
    if kind in {"function", "custom"}:
        calls = [
            call for choice in choices for call in choice["delta"].get("tool_calls", [])
        ]
        assert any(call.get("id") == "call_example" for call in calls)
    else:
        assert "".join(choice["delta"].get("content", "") for choice in choices) == (
            "Example output"
        )


@pytest.mark.parametrize("status", ["failed", "cancelled"])
@pytest.mark.parametrize("has_output", [True, False], ids=["partial", "empty"])
def test_json_failure_raises_with_original_response(status, has_output):
    response = _response(status, has_output=has_output)
    response["instructions"] = "Example internal instructions"
    with pytest.raises(ConversionError) as error:
        _pipeline().convert_response(response)
    assert response["id"] in str(error.value)
    assert status in str(error.value)
    assert response["instructions"] not in str(error.value)
    assert "Example output" not in str(error.value)
    if status == "failed":
        for key in ("type", "code", "message"):
            assert response["error"][key] in str(error.value)
        assert '"attempt": 1' in str(error.value)


@pytest.mark.parametrize("status", ["failed", "cancelled"])
@pytest.mark.parametrize("has_output", [True, False], ids=["partial", "empty"])
def test_sse_failure_raises_without_successful_finish(status, has_output):
    response = _response(status, has_output=has_output)
    response["instructions"] = "Example internal instructions"
    processor = _pipeline().create_stream_processor()
    chunks = []
    with pytest.raises(ValueError) as error:
        for event in _stream(response):
            chunks.extend(processor.process_chunk(event))
    assert response["id"] in str(error.value)
    assert status in str(error.value)
    assert response["instructions"] not in str(error.value)
    assert "Example output" not in str(error.value)
    assert not any(
        choice.get("finish_reason")
        for chunk in chunks
        for choice in chunk.get("choices", [])
    )
    if status == "failed":
        assert response["error"]["message"] in str(error.value)
        assert '"attempt": 1' in str(error.value)


@pytest.mark.parametrize("status", ["failed", "cancelled"])
@pytest.mark.parametrize("has_output", [True, False], ids=["partial", "empty"])
def test_native_json_failure_round_trip(status, has_output):
    converter = OpenResponsesConverter()
    response = _response(status, has_output=has_output)
    converted = converter.response_to_provider(
        converter.response_from_provider(response)
    )
    assert converted["status"] == status
    assert converted.get("error") == response.get("error")
    assert bool(converted["output"]) is has_output


@pytest.mark.parametrize("metadata_mode", ["strip", "preserve"])
def test_failure_response_stays_in_context(metadata_mode):
    converter = OpenResponsesConverter()
    response = _response("failed")
    response["instructions"] = "Example internal instructions"
    context = ConversionContext(options={"metadata_mode": metadata_mode})
    ir_response = converter.response_from_provider(response, context=context)
    metadata = ir_response["choices"][0]["message"]["provider_metadata"]
    assert metadata["responses_error"] == response["error"]
    assert "responses_response" not in metadata
    assert context.metadata["responses_response"] == response


def test_native_sse_failure_keeps_its_error_envelope():
    converter = OpenResponsesConverter()
    response = _response("failed", has_output=False)
    context = converter.create_stream_context()
    events = cast(
        list[dict[str, Any]],
        converter.stream_response_from_provider(
            {"type": "response.failed", "response": response}, context=context
        ),
    )
    finish = next(event for event in events if event["type"] == "finish")
    assert finish["provider_metadata"] == {"responses_error": response["error"]}
    assert context.metadata["responses_response"] == response
    chunks = cast(
        list[dict[str, Any]],
        converter.stream_response_to_provider(cast(FinishEvent, finish)),
    )
    failed = next(chunk for chunk in chunks if chunk["type"] == "response.failed")
    assert failed["response"]["error"] == response["error"]
