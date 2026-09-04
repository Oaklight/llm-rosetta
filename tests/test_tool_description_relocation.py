"""Tests for oversized tool description relocation."""

from llm_rosetta.capabilities import (
    relocate_oversized_tool_descriptions,
)


class TestRelocateOversizedToolDescriptions:
    def _make_ir(self, tools, messages=None):
        return {
            "tools": tools,
            "messages": messages
            or [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
        }

    def _make_tool(self, name="t", desc="short"):
        return {"name": name, "description": desc, "type": "function", "parameters": {}}

    def test_none_threshold_is_noop(self):
        ir = self._make_ir([self._make_tool(desc="x" * 5000)])
        result = relocate_oversized_tool_descriptions(ir, max_description_length=None)
        assert result is ir

    def test_no_tools_is_noop(self):
        ir = {"messages": [{"role": "user", "content": "hi"}]}
        result = relocate_oversized_tool_descriptions(ir, max_description_length=1024)
        assert result is ir

    def test_short_description_unchanged(self):
        ir = self._make_ir([self._make_tool(desc="short")])
        result = relocate_oversized_tool_descriptions(ir, max_description_length=1024)
        assert result["tools"][0]["description"] == "short"
        assert len(result["messages"]) == 1

    def test_long_description_relocated(self):
        long_desc = "A" * 2000
        ir = self._make_ir([self._make_tool(name="my_tool", desc=long_desc)])
        result = relocate_oversized_tool_descriptions(ir, max_description_length=1024)

        assert (
            result["tools"][0]["description"]
            == "[Full documentation for 'my_tool' provided separately.]"
        )
        assert result["tools"][0]["metadata"]["_description_relocated"] is True
        assert len(result["messages"]) == 2
        sys_msg = result["messages"][-1]
        assert sys_msg["role"] == "system"
        text = sys_msg["content"][0]["text"]
        assert "## Tool: my_tool" in text
        assert long_desc in text

    def test_multiple_oversized_consolidated(self):
        ir = self._make_ir(
            [
                self._make_tool(name="tool_a", desc="A" * 2000),
                self._make_tool(name="tool_b", desc="short"),
                self._make_tool(name="tool_c", desc="C" * 3000),
            ]
        )
        result = relocate_oversized_tool_descriptions(ir, max_description_length=1024)

        assert (
            result["tools"][0]["description"]
            == "[Full documentation for 'tool_a' provided separately.]"
        )
        assert result["tools"][1]["description"] == "short"
        assert (
            result["tools"][2]["description"]
            == "[Full documentation for 'tool_c' provided separately.]"
        )
        assert len(result["messages"]) == 2
        sys_text = result["messages"][-1]["content"][0]["text"]
        assert "## Tool: tool_a" in sys_text
        assert "## Tool: tool_c" in sys_text
        assert "---" in sys_text

    def test_original_not_mutated(self):
        orig_desc = "X" * 2000
        tool = self._make_tool(desc=orig_desc)
        ir = self._make_ir([tool])
        _ = relocate_oversized_tool_descriptions(ir, max_description_length=1024)
        assert tool["description"] == orig_desc
        assert "metadata" not in tool

    def test_exactly_at_threshold_not_relocated(self):
        ir = self._make_ir([self._make_tool(desc="A" * 1024)])
        result = relocate_oversized_tool_descriptions(ir, max_description_length=1024)
        assert result["tools"][0]["description"] == "A" * 1024
        assert len(result["messages"]) == 1

    def test_custom_threshold(self):
        ir = self._make_ir([self._make_tool(desc="A" * 100)])
        result = relocate_oversized_tool_descriptions(ir, max_description_length=50)
        assert (
            result["tools"][0]["description"]
            == "[Full documentation for 't' provided separately.]"
        )


class TestRelocationWithPipeline:
    def test_relocation_before_hoist(self):
        from llm_rosetta.converters.base.helpers.system_message_hoist import (
            hoist_late_system_messages_ir,
        )

        long_desc = "D" * 2000
        ir = {
            "tools": [
                {
                    "name": "big_tool",
                    "description": long_desc,
                    "type": "function",
                    "parameters": {},
                }
            ],
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hello"}]},
                {"role": "assistant", "content": [{"type": "text", "text": "hi"}]},
                {"role": "user", "content": [{"type": "text", "text": "use tool"}]},
            ],
        }

        relocated = relocate_oversized_tool_descriptions(
            ir, max_description_length=1024
        )
        assert relocated["messages"][-1]["role"] == "system"

        hoisted = hoist_late_system_messages_ir(relocated)
        last_msg = hoisted["messages"][-1]
        assert last_msg["role"] == "user"
        assert "<system>" in last_msg["content"][0]["text"]
        assert long_desc in last_msg["content"][0]["text"]
