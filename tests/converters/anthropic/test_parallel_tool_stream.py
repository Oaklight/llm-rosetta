"""Keep parallel tool arguments on distinct Anthropic stream blocks."""

import json
from typing import Any, cast

import pytest

from llm_rosetta.converters.anthropic import AnthropicConverter
from llm_rosetta.converters.base.context import StreamContext
from llm_rosetta.pipeline import ConversionPipeline
from llm_rosetta.types.ir.stream import IRStreamEvent


def chat_chunk(delta, finish=None):
    return {
        "id": "chatcmpl-parallel",
        "object": "chat.completion.chunk",
        "created": 1700000000,
        "model": "demo",
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }


def collect_tools(events):
    """Validate each block lifecycle and reconstruct tool arguments by ID."""
    blocks: dict[int, dict[str, Any]] = {}
    for event in events:
        kind = event["type"]
        index = event.get("index")
        if kind == "content_block_start":
            assert index not in blocks, f"duplicate block index {index}"
            blocks[index] = {
                "block": event["content_block"],
                "json": "",
                "closed": False,
            }
        elif kind == "content_block_delta":
            assert index in blocks and not blocks[index]["closed"]
            if event["delta"]["type"] == "input_json_delta":
                blocks[index]["json"] += event["delta"]["partial_json"]
        elif kind == "content_block_stop":
            assert index in blocks and not blocks[index]["closed"]
            blocks[index]["closed"] = True
    assert all(block["closed"] for block in blocks.values())
    assert sum(event["type"] == "message_stop" for event in events) == 1
    return {
        block["block"]["id"]: json.loads(block["json"])
        for block in blocks.values()
        if block["block"]["type"] == "tool_use"
    }


@pytest.mark.parametrize("count", [1, 2, 3])
@pytest.mark.parametrize("prefix", [None, "content", "reasoning_content"])
@pytest.mark.parametrize("suffix", [False, True])
def test_chat_parallel_tool_stream_keeps_ids_arguments_and_block_lifecycles(
    prefix, suffix, count
):
    pipeline = ConversionPipeline("anthropic", "openai_chat")
    pipeline.convert_request(
        {
            "model": "demo",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "Read both files"}],
        }
    )
    processor = pipeline.create_stream_processor()
    chunks = [chat_chunk({"role": "assistant"})]
    if prefix:
        chunks.append(chat_chunk({prefix: "Checking files."}))
    chunks.extend(
        [
            chat_chunk(
                {
                    "tool_calls": [
                        {
                            "index": i,
                            "id": f"call_{i}",
                            "type": "function",
                            "function": {"name": "read", "arguments": '{"path":'},
                        }
                        for i in range(count)
                    ]
                }
            ),
            chat_chunk(
                {
                    "tool_calls": [
                        {"index": i, "function": {"arguments": f'"{i}.txt"}}'}}
                        for i in reversed(range(count))
                    ]
                }
            ),
        ]
    )
    if suffix:
        chunks.append(chat_chunk({"content": "Done."}))
    chunks.extend([chat_chunk({}, "tool_calls"), {"choices": []}])
    events = [event for chunk in chunks for event in processor.process_chunk(chunk)]
    assert collect_tools(events) == {
        f"call_{i}": {"path": f"{i}.txt"} for i in range(count)
    }


def test_native_parallel_tool_stream_preserves_explicit_block_indices():
    pipeline = ConversionPipeline("anthropic", "anthropic")
    pipeline.convert_request(
        {
            "model": "demo",
            "max_tokens": 64,
            "stream": True,
            "messages": [{"role": "user", "content": "Read both files"}],
        }
    )
    processor = pipeline.create_stream_processor()
    chunks = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_native",
                "model": "demo",
                "role": "assistant",
                "content": [],
            },
        }
    ]
    chunks.extend(
        {
            "type": "content_block_start",
            "index": i,
            "content_block": {
                "type": "tool_use",
                "id": f"call_{i}",
                "name": "read",
                "input": {},
            },
        }
        for i in range(2)
    )
    chunks.extend(
        {
            "type": "content_block_delta",
            "index": i,
            "delta": {
                "type": "input_json_delta",
                "partial_json": json.dumps({"path": f"{i}.txt"}),
            },
        }
        for i in [1, 0]
    )
    chunks.extend({"type": "content_block_stop", "index": i} for i in [0, 1])
    chunks.extend(
        [
            {"type": "message_delta", "delta": {"stop_reason": "tool_use"}},
            {"type": "message_stop"},
        ]
    )
    events = [event for chunk in chunks for event in processor.process_chunk(chunk)]
    assert collect_tools(events) == {
        "call_0": {"path": "0.txt"},
        "call_1": {"path": "1.txt"},
    }


def test_synthetic_tool_block_does_not_reuse_a_completed_explicit_index():
    converter = AnthropicConverter()
    context = StreamContext()
    events = [
        {"type": "content_block_start", "block_index": 0, "block_type": "text"},
        {"type": "content_block_end", "block_index": 0},
        {"type": "tool_call_start", "tool_call_id": "call_next", "tool_name": "read"},
    ]
    output = []
    for event in events:
        converted = converter.stream_response_to_provider(
            cast(IRStreamEvent, event), context=context
        )
        output.extend(converted if isinstance(converted, list) else [converted])
    starts = [event for event in output if event.get("type") == "content_block_start"]
    assert [event["index"] for event in starts] == [0, 1]
