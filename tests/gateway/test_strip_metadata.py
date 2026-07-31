"""Tests for _strip_internal_metadata in proxy.py."""

from llm_rosetta.gateway.proxy import _strip_internal_metadata


class TestStripInternalMetadata:
    def test_strips_top_level(self):
        body = {"model": "gpt-4", "_provider_metadata": {"id": "x"}}
        _strip_internal_metadata(body)
        assert "_provider_metadata" not in body
        assert body["model"] == "gpt-4"

    def test_strips_nested_in_tool_calls(self):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call_1",
                            "function": {"name": "fn", "arguments": "{}"},
                            "_provider_metadata": {"responses_item_id": "fc_1"},
                        },
                        {
                            "id": "call_2",
                            "function": {"name": "fn2", "arguments": "{}"},
                            "_provider_metadata": {"responses_item_id": "fc_2"},
                        },
                    ],
                }
            ]
        }
        _strip_internal_metadata(body)
        for tc in body["messages"][0]["tool_calls"]:
            assert "_provider_metadata" not in tc
            assert "id" in tc

    def test_strips_in_content_blocks(self):
        body = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "text",
                            "text": "hi",
                            "_provider_metadata": {"k": "v"},
                        },
                    ],
                }
            ]
        }
        _strip_internal_metadata(body)
        assert "_provider_metadata" not in body["messages"][0]["content"][0]

    def test_no_op_without_metadata(self):
        body = {
            "model": "gpt-4",
            "messages": [{"role": "user", "content": "hi"}],
        }
        original = body.copy()
        _strip_internal_metadata(body)
        assert body == original

    def test_deeply_nested(self):
        body = {"a": {"b": [{"c": {"_provider_metadata": {"deep": True}, "keep": 1}}]}}
        _strip_internal_metadata(body)
        assert body["a"]["b"][0]["c"] == {"keep": 1}

    def test_preserves_other_underscore_fields(self):
        body = {"_other_field": "keep", "_provider_metadata": "strip"}
        _strip_internal_metadata(body)
        assert "_other_field" in body
        assert "_provider_metadata" not in body

    def test_empty_body(self):
        body = {}
        _strip_internal_metadata(body)
        assert body == {}
