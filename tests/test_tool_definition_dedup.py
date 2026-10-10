"""Tests for duplicate tool definition collapsing."""

from llm_rosetta.capabilities import dedupe_tool_definitions


def _tool(name, desc="d"):
    return {"name": name, "description": desc, "type": "function", "parameters": {}}


class TestDedupeToolDefinitions:
    def test_no_tools_is_noop(self):
        ir = {"messages": []}
        assert dedupe_tool_definitions(ir) is ir

    def test_single_tool_is_noop(self):
        ir = {"tools": [_tool("a")]}
        assert dedupe_tool_definitions(ir) is ir

    def test_distinct_tools_unchanged(self):
        ir = {"tools": [_tool("a"), _tool("b")]}
        assert dedupe_tool_definitions(ir) is ir

    def test_exact_duplicate_collapsed(self):
        ir = {"tools": [_tool("a"), _tool("b"), _tool("a")]}
        out = dedupe_tool_definitions(ir)
        assert [t["name"] for t in out["tools"]] == ["a", "b"]

    def test_exact_duplicate_order_preserved(self):
        ir = {"tools": [_tool("b"), _tool("a"), _tool("b")]}
        out = dedupe_tool_definitions(ir)
        assert [t["name"] for t in out["tools"]] == ["b", "a"]

    def test_exact_duplicate_keeps_first_occurrence(self):
        first = _tool("a", desc="first")
        ir = {"tools": [first, _tool("a", desc="first")]}
        out = dedupe_tool_definitions(ir)
        assert out["tools"][0] is first

    def test_exact_duplicate_warns(self):
        warnings = []
        ir = {"tools": [_tool("a"), _tool("a")]}
        dedupe_tool_definitions(ir, warnings=warnings)
        assert len(warnings) == 1
        assert "'a'" in warnings[0]

    def test_conflicting_definitions_kept_and_warned(self):
        warnings = []
        ir = {"tools": [_tool("a", desc="one"), _tool("a", desc="two")]}
        out = dedupe_tool_definitions(ir, warnings=warnings)
        assert out is ir
        assert len(out["tools"]) == 2
        assert len(warnings) == 1
        assert "'a'" in warnings[0]

    def test_conflicting_plus_exact_duplicate(self):
        warnings = []
        ir = {
            "tools": [
                _tool("a", desc="one"),
                _tool("a", desc="one"),
                _tool("a", desc="two"),
            ]
        }
        out = dedupe_tool_definitions(ir, warnings=warnings)
        assert [t["name"] for t in out["tools"]] == ["a", "a"]
        assert len(warnings) == 2

    def test_no_warnings_list_is_safe(self):
        ir = {"tools": [_tool("a"), _tool("a")]}
        out = dedupe_tool_definitions(ir)
        assert len(out["tools"]) == 1

    def test_does_not_mutate_input(self):
        ir = {"tools": [_tool("a"), _tool("a")]}
        dedupe_tool_definitions(ir)
        assert len(ir["tools"]) == 2

    def test_other_request_keys_preserved(self):
        ir = {"tools": [_tool("a"), _tool("a")], "model": "m", "messages": [{"a": 1}]}
        out = dedupe_tool_definitions(ir)
        assert out["model"] == "m"
        assert out["messages"] == [{"a": 1}]

    def test_non_dict_and_unnamed_tools_passed_through(self):
        ir = {"tools": [{"name": "a"}, "junk", {"description": "no name"}]}
        out = dedupe_tool_definitions(ir)
        assert out is ir

    def test_namespaced_tools_never_collapsed(self):
        """A namespace is part of a tool's identity, even when identical."""
        tool = {
            "name": "wait",
            "description": "d",
            "parameters": {},
            "metadata": {"namespace": ""},
        }
        ir = {"tools": [dict(tool), dict(tool)]}
        assert dedupe_tool_definitions(ir) is ir

    def test_top_level_dup_collapsed_beside_namespaced_same_name(self):
        top1 = {"name": "wait", "description": "d", "parameters": {}}
        top2 = {"name": "wait", "description": "d", "parameters": {}}
        ns = {
            "name": "wait",
            "description": "d",
            "parameters": {},
            "metadata": {"namespace": ""},
        }
        out = dedupe_tool_definitions({"tools": [top1, ns, top2]})
        assert out["tools"] == [top1, ns]

    def test_non_dict_metadata_is_safe(self):
        """A malformed metadata value must not raise."""
        tool = {"name": "a", "description": "d", "metadata": "not-a-dict"}
        out = dedupe_tool_definitions({"tools": [dict(tool), dict(tool)]})
        assert len(out["tools"]) == 1


class TestDedupeInPipeline:
    """End-to-end: the pass runs on the anthropic → openai_chat path."""

    def _anthropic_req(self, tools):
        return {
            "model": "m",
            "max_tokens": 50,
            "tools": tools,
            "messages": [{"role": "user", "content": "hi"}],
        }

    @staticmethod
    def _tool(name, desc):
        return {
            "name": name,
            "description": desc,
            "input_schema": {"type": "object", "properties": {}},
        }

    def test_exact_duplicates_collapsed(self):
        from llm_rosetta.pipeline import ConversionPipeline

        dup = self._tool("dup", "x")
        other = self._tool("other", "y")
        pipe = ConversionPipeline("anthropic", "openai_chat")
        body = pipe.convert_request(self._anthropic_req([dup, dict(dup), other]))
        assert [t["function"]["name"] for t in body["tools"]] == ["dup", "other"]

    def test_conflicting_definitions_kept(self):
        from llm_rosetta.pipeline import ConversionPipeline

        pipe = ConversionPipeline("anthropic", "openai_chat")
        body = pipe.convert_request(
            self._anthropic_req([self._tool("dup", "one"), self._tool("dup", "two")])
        )
        assert [t["function"]["name"] for t in body["tools"]] == ["dup", "dup"]
