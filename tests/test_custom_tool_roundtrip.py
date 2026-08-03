"""Tests for custom (freeform) tool round-trip across a Chat Completions upstream.

A client such as Codex declares ``apply_patch`` as an OpenAI Responses
``custom`` tool.  Chat Completions has no equivalent, so the definition is
downgraded to a JSON function taking a single ``input`` string.  The upstream
then answers with an ordinary tool call, and without the restore step the
response is emitted as ``function_call`` and the client rejects it.
"""

import json

import pytest

from llm_rosetta.pipeline import (
    ConversionPipeline,
    _collect_custom_tool_names,
    _restore_custom_tool_calls,
    _unwrap_custom_tool_input,
)

PATCH_TEXT = "*** Begin Patch\n*** Add File: a.txt\n+hi\n*** End Patch"

CUSTOM_TOOL = {
    "type": "custom",
    "name": "apply_patch",
    "description": "Apply a V4A patch.",
    "format": {"type": "grammar", "syntax": "lark", "definition": "start: /.+/s"},
}


def _request():
    return {
        "model": "gpt-5.6-sol",
        "tools": [CUSTOM_TOOL],
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "create a.txt"}],
            }
        ],
    }


def _pipeline():
    pipe = ConversionPipeline("openai_responses", "openai_chat")
    pipe.convert_request(_request())
    return pipe


def _chat_completion():
    return {
        "id": "x",
        "object": "chat.completion",
        "created": 1,
        "model": "gpt-5.6-sol",
        "choices": [
            {
                "index": 0,
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "type": "function",
                            "function": {
                                "name": "apply_patch",
                                "arguments": json.dumps({"input": PATCH_TEXT}),
                            },
                        }
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
    }


def _chat_chunks():
    def chunk(delta, finish=None):
        return {
            "id": "x",
            "object": "chat.completion.chunk",
            "created": 1,
            "model": "gpt-5.6-sol",
            "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
        }

    args = json.dumps({"input": PATCH_TEXT})
    half = len(args) // 2
    return [
        chunk(
            {
                "role": "assistant",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "apply_patch", "arguments": ""},
                    }
                ],
            }
        ),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": args[:half]}}]}),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": args[half:]}}]}),
        chunk({}, finish="tool_calls"),
    ]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_collect_custom_tool_names_reads_provider_type():
    pipe = _pipeline()
    assert pipe._custom_tool_names == frozenset({"apply_patch"})


def test_collect_custom_tool_names_ignores_plain_functions():
    ir = {
        "tools": [
            {"name": "grep", "metadata": {"provider_type": "function"}},
            {"name": "ls"},
        ]
    }
    assert _collect_custom_tool_names(ir) == frozenset()


@pytest.mark.parametrize(
    "raw,expected",
    [
        (json.dumps({"input": PATCH_TEXT}), PATCH_TEXT),
        ("not json at all", "not json at all"),
        (
            json.dumps({"input": "a", "other": "b"}),
            json.dumps({"input": "a", "other": "b"}),
        ),
    ],
)
def test_unwrap_custom_tool_input(raw, expected):
    assert _unwrap_custom_tool_input(raw) == expected


def test_restore_is_a_noop_without_custom_names():
    ir = {
        "choices": [
            {
                "message": {
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_name": "apply_patch",
                            "tool_type": "function",
                        }
                    ]
                }
            }
        ]
    }
    _restore_custom_tool_calls(ir, frozenset())
    assert ir["choices"][0]["message"]["content"][0]["tool_type"] == "function"


# ---------------------------------------------------------------------------
# Non-streaming
# ---------------------------------------------------------------------------


def test_non_streaming_emits_custom_tool_call():
    out = _pipeline().convert_response(_chat_completion())
    items = out["output"]
    assert [i["type"] for i in items] == ["custom_tool_call"]
    assert items[0]["input"] == PATCH_TEXT
    assert "arguments" not in items[0]


def test_non_streaming_plain_function_is_untouched():
    pipe = ConversionPipeline("openai_responses", "openai_chat")
    req = _request()
    req["tools"] = [
        {
            "type": "function",
            "name": "apply_patch",
            "parameters": {"type": "object", "properties": {}},
        }
    ]
    pipe.convert_request(req)
    out = pipe.convert_response(_chat_completion())
    assert [i["type"] for i in out["output"]] == ["function_call"]


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


def test_streaming_emits_custom_tool_call_with_unwrapped_input():
    proc = _pipeline().create_stream_processor()
    events = []
    for chunk in _chat_chunks():
        events.extend(proc.process_chunk(chunk))

    done = [e["item"] for e in events if e.get("type", "").endswith("output_item.done")]
    assert [d["type"] for d in done] == ["custom_tool_call"]
    # The upstream sent JSON across two fragments; the client needs raw text.
    assert done[0]["input"] == PATCH_TEXT

    kinds = {e.get("type") for e in events}
    assert "response.custom_tool_call_input.done" in kinds
    assert "response.function_call_arguments.done" not in kinds
