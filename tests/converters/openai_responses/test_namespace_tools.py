"""Tests for namespace tool flattening in OpenAI Responses converter."""

from typing import Any

from llm_rosetta.converters.openai_responses.converter import (
    OpenAIResponsesConverter,
)
from llm_rosetta.converters.openai_responses.tool_ops import (
    OpenAIResponsesToolOps,
    _flatten_namespace_tool,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ns_tool(
    name: str = "functions",
    description: str = "",
    children: list[Any] | None = None,
) -> dict[str, Any]:
    tool: dict[str, Any] = {"type": "namespace", "name": name}
    if description:
        tool["description"] = description
    if children is not None:
        tool["tools"] = children
    else:
        tool["tools"] = []
    return tool


def _func_tool(
    name: str, description: str = "", params: dict | None = None, **extra: Any
) -> dict[str, Any]:
    tool: dict[str, Any] = {
        "type": "function",
        "name": name,
        "description": description,
        "parameters": params or {},
    }
    tool.update(extra)
    return tool


def _custom_tool(name: str, description: str = "") -> dict[str, Any]:
    return {"type": "custom", "name": name, "description": description}


# ===========================================================================
# _flatten_namespace_tool unit tests
# ===========================================================================


class TestFlattenNamespaceTool:
    """Unit tests for the _flatten_namespace_tool helper."""

    def test_basic_flatten_two_functions(self):
        ns = _ns_tool(
            "functions",
            children=[
                _func_tool(
                    "exec", "Run a command", {"type": "object", "properties": {}}
                ),
                _func_tool("wait", "Wait for a process"),
            ],
        )
        result = _flatten_namespace_tool(ns)
        assert len(result) == 2
        assert result[0]["name"] == "exec"
        assert result[1]["name"] == "wait"
        assert result[0]["metadata"]["namespace"] == "functions"
        assert result[1]["metadata"]["namespace"] == "functions"

    def test_namespace_description_preserved(self):
        ns = _ns_tool(
            "crm",
            description="CRM tools for customer lookup",
            children=[
                _func_tool("get_customer"),
            ],
        )
        result = _flatten_namespace_tool(ns)
        assert len(result) == 1
        assert result[0]["metadata"]["namespace"] == "crm"
        assert (
            result[0]["metadata"]["namespace_description"]
            == "CRM tools for customer lookup"
        )

    def test_no_namespace_description_omitted(self):
        ns = _ns_tool("tools", children=[_func_tool("ping")])
        result = _flatten_namespace_tool(ns)
        assert "namespace_description" not in result[0]["metadata"]

    def test_mixed_function_and_custom(self):
        ns = _ns_tool(
            "tools",
            children=[
                _func_tool("exec", "Run cmd"),
                _custom_tool("apply_patch", "Apply a patch"),
            ],
        )
        result = _flatten_namespace_tool(ns)
        assert len(result) == 2
        names = {r["name"] for r in result}
        assert names == {"exec", "apply_patch"}

    def test_empty_namespace(self):
        ns = _ns_tool("empty", children=[])
        result = _flatten_namespace_tool(ns)
        assert result == []

    def test_no_tools_key(self):
        ns = {"type": "namespace", "name": "bare"}
        result = _flatten_namespace_tool(ns)
        assert result == []

    def test_nested_namespace_skipped(self):
        ns = _ns_tool(
            "outer",
            children=[
                _func_tool("ok_tool"),
                _ns_tool("inner", children=[_func_tool("hidden")]),
            ],
        )
        result = _flatten_namespace_tool(ns)
        assert len(result) == 1
        assert result[0]["name"] == "ok_tool"

    def test_non_dict_child_skipped(self):
        ns = _ns_tool(
            "ns",
            children=[
                _func_tool("real"),
                "not a dict",
                42,
            ],
        )
        result = _flatten_namespace_tool(ns)
        assert len(result) == 1
        assert result[0]["name"] == "real"

    def test_unnamed_child_skipped(self):
        ns = _ns_tool(
            "ns",
            children=[
                {"type": "function", "description": "no name"},
                _func_tool("named"),
            ],
        )
        result = _flatten_namespace_tool(ns)
        assert len(result) == 1
        assert result[0]["name"] == "named"

    def test_defer_loading_preserved(self):
        ns = _ns_tool(
            "crm",
            children=[
                _func_tool("get_orders", defer_loading=True),
                _func_tool("get_profile"),
            ],
        )
        result = _flatten_namespace_tool(ns)
        assert result[0]["metadata"].get("defer_loading") is True
        assert "defer_loading" not in result[1]["metadata"]

    def test_passthrough_child_type(self):
        """Unknown child type goes through _synthesize_passthrough_tool."""
        ns = _ns_tool(
            "ns",
            children=[
                {"type": "web_search", "name": "search"},
            ],
        )
        result = _flatten_namespace_tool(ns)
        assert len(result) == 1
        assert result[0]["metadata"]["namespace"] == "ns"
        assert result[0]["metadata"].get("provider_type") == "web_search"

    def test_nested_format_child(self):
        """Child in nested format: {"type": "function", "function": {...}}."""
        ns = _ns_tool(
            "ns",
            children=[
                {
                    "type": "function",
                    "function": {
                        "name": "nested_func",
                        "description": "A nested function",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
        )
        result = _flatten_namespace_tool(ns)
        assert len(result) == 1
        assert result[0]["name"] == "nested_func"
        assert result[0]["metadata"]["namespace"] == "ns"

    def test_metadata_does_not_mutate_original(self):
        child = _func_tool("tool1")
        ns = _ns_tool("ns", children=[child])
        _flatten_namespace_tool(ns)
        # Original child should not have namespace metadata
        ir_direct = OpenAIResponsesToolOps.p_tool_definition_to_ir(child)
        assert isinstance(ir_direct, dict)
        assert "namespace" not in (ir_direct.get("metadata") or {})


# ===========================================================================
# p_tool_definition_to_ir dispatch tests
# ===========================================================================


class TestNamespaceDispatch:
    """Test that p_tool_definition_to_ir dispatches namespace correctly."""

    def test_namespace_returns_list(self):
        ns = _ns_tool(
            "funcs",
            children=[
                _func_tool("a"),
                _func_tool("b"),
            ],
        )
        result = OpenAIResponsesToolOps.p_tool_definition_to_ir(ns)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_regular_function_returns_single(self):
        result = OpenAIResponsesToolOps.p_tool_definition_to_ir(_func_tool("regular"))
        assert isinstance(result, dict)
        assert result["name"] == "regular"

    def test_empty_namespace_returns_empty_list(self):
        result = OpenAIResponsesToolOps.p_tool_definition_to_ir(
            _ns_tool("empty", children=[])
        )
        assert result == []


# ===========================================================================
# _dedup_ir_tool_names tests
# ===========================================================================


class TestDedupIrToolNames:
    """Tests for the converter-level name dedup method."""

    @staticmethod
    def _ir_tool(name: str, ns: str = "") -> dict[str, Any]:
        meta: dict[str, Any] = {}
        if ns:
            meta["namespace"] = ns
        return {
            "type": "function",
            "name": name,
            "description": "",
            "parameters": {},
            "metadata": meta,
        }

    def test_no_collision_unchanged(self):
        tools = [self._ir_tool("a"), self._ir_tool("b", "ns")]
        result = OpenAIResponsesConverter._dedup_ir_tool_names(tools)
        assert [t["name"] for t in result] == ["a", "b"]

    def test_namespaced_vs_toplevel_collision(self):
        tools = [
            self._ir_tool("exec"),  # top-level, no namespace
            self._ir_tool("exec", "functions"),  # from namespace
        ]
        result = OpenAIResponsesConverter._dedup_ir_tool_names(tools)
        assert result[0]["name"] == "exec"
        assert result[1]["name"] == "functions_exec"

    def test_cross_namespace_collision(self):
        tools = [
            self._ir_tool("exec", "ns_a"),
            self._ir_tool("exec", "ns_b"),
        ]
        result = OpenAIResponsesConverter._dedup_ir_tool_names(tools)
        names = {t["name"] for t in result}
        assert "ns_a_exec" in names
        assert "ns_b_exec" in names

    def test_no_mutation_of_originals(self):
        original = self._ir_tool("exec", "ns")
        tools = [self._ir_tool("exec"), original]
        result = OpenAIResponsesConverter._dedup_ir_tool_names(tools)
        # Original dict should be unchanged
        assert original["name"] == "exec"
        # Result entry is a copy
        assert result[1]["name"] == "ns_exec"
        assert result[1] is not original

    def test_long_qualified_name_truncated(self):
        long_ns = "a" * 60
        tools = [
            self._ir_tool("tool"),
            self._ir_tool("tool", long_ns),
        ]
        result = OpenAIResponsesConverter._dedup_ir_tool_names(tools)
        assert len(result[1]["name"]) <= 64

    def test_single_tool_no_dedup(self):
        tools = [self._ir_tool("only_one", "ns")]
        result = OpenAIResponsesConverter._dedup_ir_tool_names(tools)
        assert result[0]["name"] == "only_one"


# ===========================================================================
# Full request conversion integration tests
# ===========================================================================


class TestNamespaceRequestConversion:
    """Integration tests for namespace tools through the full converter."""

    def setup_method(self):
        self.converter = OpenAIResponsesConverter()

    def test_namespace_in_tools_array(self):
        request = {
            "model": "gpt-5",
            "tools": [
                _func_tool("top_level", "A top-level function"),
                _ns_tool(
                    "functions",
                    children=[
                        _func_tool(
                            "exec",
                            "Run a command",
                            {
                                "type": "object",
                                "properties": {"cmd": {"type": "string"}},
                                "required": ["cmd"],
                            },
                        ),
                        _func_tool("wait", "Wait for process"),
                    ],
                ),
            ],
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "list files"}],
                },
            ],
        }
        ir = self.converter.request_from_provider(request)
        tool_names = {t["name"] for t in ir["tools"]}
        assert "top_level" in tool_names
        assert "exec" in tool_names
        assert "wait" in tool_names
        assert len(ir["tools"]) == 3

    def test_namespace_collision_with_toplevel(self):
        request = {
            "model": "gpt-5",
            "tools": [
                _func_tool("exec", "Top-level exec"),
                _ns_tool(
                    "functions",
                    children=[
                        _func_tool("exec", "Namespace exec"),
                    ],
                ),
            ],
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
            ],
        }
        ir = self.converter.request_from_provider(request)
        tool_names = [t["name"] for t in ir["tools"]]
        assert "exec" in tool_names
        assert "functions_exec" in tool_names
        assert len(ir["tools"]) == 2

    def test_namespace_tools_params_preserved(self):
        params = {
            "type": "object",
            "properties": {"cmd": {"type": "string"}},
            "required": ["cmd"],
        }
        request = {
            "model": "gpt-5",
            "tools": [
                _ns_tool(
                    "ns",
                    children=[
                        _func_tool("exec", "Run", params),
                    ],
                ),
            ],
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "go"}],
                },
            ],
        }
        ir = self.converter.request_from_provider(request)
        tool = ir["tools"][0]
        assert tool["parameters"] == params
        assert tool["required_parameters"] == ["cmd"]

    def test_cross_format_responses_to_chat(self):
        """Namespace tools survive Responses → Chat Completions conversion."""
        from llm_rosetta.pipeline import ConversionPipeline

        pipeline = ConversionPipeline("openai_responses", "openai_chat")
        request = {
            "model": "gpt-5",
            "tools": [
                _ns_tool(
                    "functions",
                    children=[
                        _func_tool(
                            "exec",
                            "Run a command",
                            {
                                "type": "object",
                                "properties": {"cmd": {"type": "string"}},
                            },
                        ),
                    ],
                ),
            ],
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "list files"}],
                },
            ],
        }
        result = pipeline.convert_request(request)
        tool_names = [
            t.get("function", {}).get("name") or t.get("name")
            for t in result.get("tools", [])
        ]
        assert "exec" in tool_names

    def test_cross_format_responses_to_anthropic(self):
        """Namespace tools survive Responses → Anthropic conversion."""
        from llm_rosetta.pipeline import ConversionPipeline

        pipeline = ConversionPipeline("openai_responses", "anthropic")
        request = {
            "model": "gpt-5",
            "tools": [
                _ns_tool(
                    "functions",
                    children=[
                        _func_tool(
                            "exec",
                            "Run a command",
                            {
                                "type": "object",
                                "properties": {"cmd": {"type": "string"}},
                            },
                        ),
                    ],
                ),
            ],
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "list files"}],
                },
            ],
        }
        result = pipeline.convert_request(request)
        tool_names = [t["name"] for t in result.get("tools", [])]
        assert "exec" in tool_names
