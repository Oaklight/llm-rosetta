"""Tests for strip_intrinsic_tools capability enforcement."""

from llm_rosetta.capabilities import strip_intrinsic_tools


class TestStripIntrinsicTools:
    """Tests for cross-format intrinsic tool stripping."""

    def test_same_format_noop(self):
        """Same-format requests pass through unchanged."""
        ir_request = {
            "tools": [
                {
                    "type": "intrinsic",
                    "name": "web_search",
                    "description": "",
                    "parameters": {},
                }
            ],
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "c1",
                            "tool_name": "ws",
                            "tool_input": {},
                            "tool_type": "intrinsic",
                        },
                    ],
                },
            ],
        }
        result = strip_intrinsic_tools(ir_request, same_format=True)
        assert result is ir_request

    def test_no_intrinsic_noop(self):
        """Requests without intrinsic tools pass through unchanged."""
        ir_request = {
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "",
                    "parameters": {},
                }
            ],
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "c1",
                            "tool_name": "get_weather",
                            "tool_input": {},
                            "tool_type": "function",
                        },
                    ],
                },
            ],
        }
        result = strip_intrinsic_tools(ir_request, same_format=False)
        assert result is ir_request

    def test_strip_intrinsic_tool_definitions(self):
        """Intrinsic tool definitions are removed cross-format."""
        ir_request = {
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "",
                    "parameters": {},
                },
                {
                    "type": "intrinsic",
                    "name": "web_search",
                    "description": "",
                    "parameters": {},
                },
            ],
            "messages": [],
        }
        result = strip_intrinsic_tools(ir_request, same_format=False)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "get_weather"

    def test_strip_intrinsic_tool_call_parts(self):
        """Intrinsic tool_call parts are removed from message content."""
        ir_request = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": "Let me search for that."},
                        {
                            "type": "tool_call",
                            "tool_call_id": "c1",
                            "tool_name": "ws",
                            "tool_input": {},
                            "tool_type": "intrinsic",
                        },
                    ],
                },
            ],
        }
        result = strip_intrinsic_tools(ir_request, same_format=False)
        assert len(result["messages"]) == 1
        assert len(result["messages"][0]["content"]) == 1
        assert result["messages"][0]["content"][0]["type"] == "text"

    def test_strip_intrinsic_tool_result_parts(self):
        """Intrinsic tool_result parts are removed from message content."""
        ir_request = {
            "messages": [
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": "c1",
                            "result": "search results",
                            "tool_type": "intrinsic",
                        },
                    ],
                },
            ],
        }
        result = strip_intrinsic_tools(ir_request, same_format=False)
        # Empty tool-role message should be dropped entirely
        assert len(result["messages"]) == 0

    def test_keep_non_tool_role_empty_messages(self):
        """Non-tool messages that become empty after stripping are kept."""
        ir_request = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "c1",
                            "tool_name": "ws",
                            "tool_input": {},
                            "tool_type": "intrinsic",
                        },
                    ],
                },
            ],
        }
        result = strip_intrinsic_tools(ir_request, same_format=False)
        # Assistant message kept (empty content) to preserve alternation
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "assistant"
        assert result["messages"][0]["content"] == []

    def test_mixed_intrinsic_and_function_parts(self):
        """Messages with both intrinsic and function parts keep only function parts."""
        ir_request = {
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "c1",
                            "tool_name": "get_weather",
                            "tool_input": {},
                            "tool_type": "function",
                        },
                        {
                            "type": "tool_call",
                            "tool_call_id": "c2",
                            "tool_name": "ws",
                            "tool_input": {},
                            "tool_type": "intrinsic",
                        },
                    ],
                },
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": "c1",
                            "result": "sunny",
                            "tool_type": "function",
                        },
                        {
                            "type": "tool_result",
                            "tool_call_id": "c2",
                            "result": "results",
                            "tool_type": "intrinsic",
                        },
                    ],
                },
            ],
        }
        result = strip_intrinsic_tools(ir_request, same_format=False)
        assert len(result["messages"]) == 2
        assert len(result["messages"][0]["content"]) == 1
        assert result["messages"][0]["content"][0]["tool_call_id"] == "c1"
        assert len(result["messages"][1]["content"]) == 1
        assert result["messages"][1]["content"][0]["tool_call_id"] == "c1"

    def test_strip_both_defs_and_parts(self):
        """Both tool definitions and message parts are stripped together."""
        ir_request = {
            "tools": [
                {
                    "type": "function",
                    "name": "func",
                    "description": "",
                    "parameters": {},
                },
                {
                    "type": "intrinsic",
                    "name": "ws",
                    "description": "",
                    "parameters": {},
                },
            ],
            "messages": [
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "c1",
                            "tool_name": "ws",
                            "tool_input": {},
                            "tool_type": "intrinsic",
                        },
                    ],
                },
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": "c1",
                            "result": "data",
                            "tool_type": "intrinsic",
                        },
                    ],
                },
            ],
        }
        result = strip_intrinsic_tools(ir_request, same_format=False)
        assert len(result["tools"]) == 1
        assert result["tools"][0]["name"] == "func"
        # tool-role message dropped, assistant kept (empty)
        assert len(result["messages"]) == 1
        assert result["messages"][0]["role"] == "assistant"

    def test_does_not_mutate_original(self):
        """Original ir_request is not mutated."""
        ir_request = {
            "tools": [
                {
                    "type": "intrinsic",
                    "name": "ws",
                    "description": "",
                    "parameters": {},
                },
            ],
            "messages": [],
        }
        result = strip_intrinsic_tools(ir_request, same_format=False)
        assert result is not ir_request
        assert len(ir_request["tools"]) == 1
        assert len(result["tools"]) == 0
