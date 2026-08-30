"""Tests for tool_search_call/tool_search_output passthrough."""

from __future__ import annotations

from llm_rosetta.converters.openai_responses import OpenAIResponsesConverter


class TestToolSearchStreamingPassthrough:
    def test_output_item_added_tool_search_call(self):
        c = OpenAIResponsesConverter()
        ctx = c.create_stream_context()
        chunk = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "tool_search_call",
                "id": "tsc_1",
                "execution": "server",
                "call_id": None,
                "status": "completed",
                "arguments": {"paths": ["crm"]},
            },
        }
        events = c.stream_response_from_provider(chunk, ctx)
        passthrough = [e for e in events if e["type"] == "provider_passthrough"]
        assert len(passthrough) == 1
        assert passthrough[0]["payload"]["item"]["type"] == "tool_search_call"

    def test_output_item_done_tool_search_output(self):
        c = OpenAIResponsesConverter()
        ctx = c.create_stream_context()
        chunk = {
            "type": "response.output_item.done",
            "output_index": 1,
            "item": {
                "type": "tool_search_output",
                "id": "tso_1",
                "execution": "server",
                "call_id": None,
                "status": "completed",
                "tools": [{"type": "function", "name": "list_orders"}],
            },
        }
        events = c.stream_response_from_provider(chunk, ctx)
        passthrough = [e for e in events if e["type"] == "provider_passthrough"]
        assert len(passthrough) == 1
        assert passthrough[0]["payload"]["item"]["type"] == "tool_search_output"

    def test_streaming_round_trip(self):
        c = OpenAIResponsesConverter()
        ctx = c.create_stream_context()
        chunk = {
            "type": "response.output_item.added",
            "output_index": 0,
            "item": {
                "type": "tool_search_call",
                "id": "tsc_1",
                "execution": "server",
            },
        }
        ir_events = c.stream_response_from_provider(chunk, ctx)
        passthrough = [e for e in ir_events if e["type"] == "provider_passthrough"]
        assert passthrough
        result = c.stream_response_to_provider(passthrough[0], ctx)
        p_chunk = result if isinstance(result, dict) else result[0]
        assert p_chunk["type"] == "response.output_item.added"
        assert p_chunk["item"]["type"] == "tool_search_call"


class TestToolSearchNonStreamingPassthrough:
    def test_response_round_trip(self):
        c = OpenAIResponsesConverter()
        response = {
            "id": "resp_test",
            "object": "response",
            "created_at": 0,
            "model": "gpt-5.4",
            "status": "completed",
            "output": [
                {
                    "type": "tool_search_call",
                    "execution": "server",
                    "call_id": None,
                    "status": "completed",
                    "arguments": {"paths": ["crm"]},
                },
                {
                    "type": "tool_search_output",
                    "execution": "server",
                    "call_id": None,
                    "status": "completed",
                    "tools": [{"type": "function", "name": "list_orders"}],
                },
                {
                    "type": "function_call",
                    "name": "list_orders",
                    "call_id": "call_123",
                    "arguments": '{"customer_id": "C1"}',
                },
            ],
        }
        ir = c.response_from_provider(response)
        passthrough = ir.get("provider_passthrough_items", [])
        assert len(passthrough) == 2
        assert passthrough[0]["payload"]["type"] == "tool_search_call"
        assert passthrough[1]["payload"]["type"] == "tool_search_output"

        p = c.response_to_provider(ir)
        output_types = [item.get("type") for item in p.get("output", [])]
        assert "tool_search_call" in output_types
        assert "tool_search_output" in output_types
        assert "function_call" in output_types


class TestToolSearchInputPassthrough:
    def test_input_items_preserved(self):
        c = OpenAIResponsesConverter()
        items = [
            {
                "type": "tool_search_call",
                "execution": "server",
                "call_id": None,
                "status": "completed",
                "arguments": {"paths": ["crm"]},
            },
            {
                "type": "tool_search_output",
                "execution": "server",
                "call_id": None,
                "status": "completed",
                "tools": [],
            },
        ]
        ir_messages = c.message_ops.p_messages_to_ir(items)
        assert len(ir_messages) > 0
        has_passthrough = any(
            "_passthrough_items"
            in (m.get("metadata", {}).get("custom", {}) or m.get("custom", {}))
            for m in ir_messages
        )
        assert has_passthrough
