"""Preserve opaque Responses tool output strings through public pipelines."""

import copy

import pytest

from llm_rosetta.pipeline import ConversionPipeline


def _request(output):
    return {
        "model": "example-model",
        "input": [
            {"role": "user", "content": "Inspect the result."},
            {
                "type": "function_call",
                "call_id": "call_lookup",
                "name": "lookup",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_lookup",
                "output": output,
            },
        ],
    }


@pytest.mark.parametrize(
    "output",
    [
        '[{"path":"example.txt","lineNumber":1}]',
        "[1, 2, 3]",
        "[]",
        '{ "ok": true }',
        '"quoted text"',
        "true",
        "false",
        "null",
        "1e3",
        '  [\n  {"value": "unchanged"}\n]  ',
        '[{"type":"input_text","text":"JSON data, not a content block"}]',
        "ordinary text",
    ],
)
@pytest.mark.parametrize("target", ["openai_chat", "anthropic", "google"])
def test_string_output_keeps_its_type_and_exact_contents(output, target):
    request = _request(output)
    original = copy.deepcopy(request)
    converted = ConversionPipeline("openai_responses", target).convert_request(request)
    assert request == original
    if target == "openai_chat":
        tool = next(m for m in converted["messages"] if m["role"] == "tool")
        assert isinstance(tool["content"], str)
        assert tool["content"] == output
    restored = ConversionPipeline(target, "openai_responses").convert_request(converted)
    tool_output = next(
        item for item in restored["input"] if item["type"] == "function_call_output"
    )
    assert tool_output["output"] == output


def test_actual_multimodal_array_still_preserves_image_blocks():
    image = "data:image/png;base64,aGVsbG8="
    request = _request(
        [
            {"type": "input_text", "text": "A tool image"},
            {"type": "input_image", "image_url": image},
        ]
    )
    original = copy.deepcopy(request)
    converted = ConversionPipeline("openai_responses", "openai_chat").convert_request(
        request
    )
    assert request == original
    # Chat's existing multimodal fallback hoists the image into a tagged user
    # message. Preserve that behavior; do not serialize actual blocks as text.
    images = [
        part
        for message in converted["messages"]
        if isinstance(message.get("content"), list)
        for part in message["content"]
        if part.get("type") == "image_url"
    ]
    assert len(images) == 1
    assert images[0]["image_url"]["url"] == image


def test_responses_same_format_ir_roundtrip_keeps_json_text():
    output = '[{"type":"input_image","image_url":"ordinary tool data"}]'
    request = _request(output)
    converted = ConversionPipeline(
        "openai_responses", "openai_responses", baseline=False
    ).convert_request(request)
    tool_output = next(
        item for item in converted["input"] if item["type"] == "function_call_output"
    )
    assert tool_output["output"] == output
