"""Tests for interleaved tool call streaming (issue #627).

When multiple tool_use content blocks are open simultaneously and deltas
arrive interleaved, each delta must be bound to the correct tool call
via the chunk's ``index`` field, not the last-registered tool call ID.
"""

from llm_rosetta import AnthropicConverter
from llm_rosetta.converters.base.context import StreamContext


def _make_events(interleaved: bool = True) -> list[dict]:
    """Build a two-tool-call stream.

    When interleaved=True, both content_block_start events arrive before
    any delta, and deltas reference their blocks by index.
    When interleaved=False, block 0 completes fully before block 1 starts.
    """
    events = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": "test",
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "tool_use",
                "id": "toolu_A",
                "name": "get_weather",
                "input": {},
            },
        },
    ]
    if interleaved:
        events.append(
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_B",
                    "name": "get_time",
                    "input": {},
                },
            }
        )
        events.extend(
            [
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": '{"city":'},
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": '{"tz":'},
                },
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {"type": "input_json_delta", "partial_json": ' "Paris"}'},
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {"type": "input_json_delta", "partial_json": ' "UTC"}'},
                },
                {"type": "content_block_stop", "index": 0},
                {"type": "content_block_stop", "index": 1},
            ]
        )
    else:
        events.extend(
            [
                {
                    "type": "content_block_delta",
                    "index": 0,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '{"city": "Paris"}',
                    },
                },
                {"type": "content_block_stop", "index": 0},
                {
                    "type": "content_block_start",
                    "index": 1,
                    "content_block": {
                        "type": "tool_use",
                        "id": "toolu_B",
                        "name": "get_time",
                        "input": {},
                    },
                },
                {
                    "type": "content_block_delta",
                    "index": 1,
                    "delta": {
                        "type": "input_json_delta",
                        "partial_json": '{"tz": "UTC"}',
                    },
                },
                {"type": "content_block_stop", "index": 1},
            ]
        )
    events.extend(
        [
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
    )
    return events


def _collect_tool_args(events_list):
    converter = AnthropicConverter()
    context = StreamContext()
    ir = []
    for chunk in events_list:
        ir.extend(
            dict(e)
            for e in converter.stream_response_from_provider(chunk, context=context)
        )
    calls = {}
    for e in ir:
        if e.get("type") == "tool_call_start":
            calls.setdefault(e["tool_call_id"], "")
        elif e.get("type") == "tool_call_delta":
            calls[e["tool_call_id"]] = calls.get(e["tool_call_id"], "") + str(
                e.get("arguments_delta", "")
            )
    return calls


class TestInterleavedToolCalls:
    def test_interleaved_deltas_bind_correctly(self):
        """Each delta goes to the right tool call even when interleaved."""
        calls = _collect_tool_args(_make_events(interleaved=True))
        assert calls["toolu_A"] == '{"city": "Paris"}'
        assert calls["toolu_B"] == '{"tz": "UTC"}'

    def test_sequential_deltas_still_work(self):
        """Existing sequential behavior is preserved."""
        calls = _collect_tool_args(_make_events(interleaved=False))
        assert calls["toolu_A"] == '{"city": "Paris"}'
        assert calls["toolu_B"] == '{"tz": "UTC"}'

    def test_single_tool_call(self):
        """Single tool call still works."""
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "test",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_A",
                    "name": "get_weather",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"city": "Paris"}',
                },
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
        calls = _collect_tool_args(events)
        assert calls["toolu_A"] == '{"city": "Paris"}'

    def test_text_block_before_tool_blocks(self):
        """Text block at index 0, tool blocks at index 1 and 2."""
        events = [
            {
                "type": "message_start",
                "message": {
                    "id": "msg_1",
                    "type": "message",
                    "role": "assistant",
                    "content": [],
                    "model": "test",
                    "stop_reason": None,
                    "stop_sequence": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                },
            },
            {
                "type": "content_block_start",
                "index": 0,
                "content_block": {"type": "text", "text": ""},
            },
            {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": "Let me help."},
            },
            {"type": "content_block_stop", "index": 0},
            {
                "type": "content_block_start",
                "index": 1,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_A",
                    "name": "get_weather",
                    "input": {},
                },
            },
            {
                "type": "content_block_start",
                "index": 2,
                "content_block": {
                    "type": "tool_use",
                    "id": "toolu_B",
                    "name": "get_time",
                    "input": {},
                },
            },
            {
                "type": "content_block_delta",
                "index": 1,
                "delta": {
                    "type": "input_json_delta",
                    "partial_json": '{"city": "Paris"}',
                },
            },
            {
                "type": "content_block_delta",
                "index": 2,
                "delta": {"type": "input_json_delta", "partial_json": '{"tz": "UTC"}'},
            },
            {"type": "content_block_stop", "index": 1},
            {"type": "content_block_stop", "index": 2},
            {
                "type": "message_delta",
                "delta": {"stop_reason": "tool_use", "stop_sequence": None},
                "usage": {"output_tokens": 1},
            },
            {"type": "message_stop"},
        ]
        calls = _collect_tool_args(events)
        assert calls["toolu_A"] == '{"city": "Paris"}'
        assert calls["toolu_B"] == '{"tz": "UTC"}'
