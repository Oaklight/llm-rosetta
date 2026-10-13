"""Tests for namespace tool flattening in OpenAI Responses converter."""

from typing import Any

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.open_responses.converter import (
    OpenAIResponsesConverter,
)
from llm_rosetta.converters.open_responses.tool_ops import (
    OpenAIResponsesToolOps,
    _flatten_namespace_tool,
)
from llm_rosetta.pipeline import ConversionPipeline


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
        assert result[0]["tool_group"] == "functions"
        assert result[1]["tool_group"] == "functions"

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
        assert result[0]["tool_group"] == "crm"
        assert result[0]["tool_group_description"] == "CRM tools for customer lookup"

    def test_no_group_description_omitted(self):
        ns = _ns_tool("tools", children=[_func_tool("ping")])
        result = _flatten_namespace_tool(ns)
        assert "tool_group_description" not in result[0]

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

    def test_child_keeps_its_own_metadata_beside_defer_loading(self):
        """The flag joins what the child carried rather than replacing it.

        A custom child keeps its ``format`` in the child-owned metadata, and a
        container may also ask for ``defer_loading`` on that same child.
        Writing the flag over the metadata instead of into it would drop the
        format, which changes how the tool is invoked.
        """
        ns = _ns_tool(
            "crm",
            children=[
                {
                    "type": "custom",
                    "name": "apply_patch",
                    "format": {"type": "text"},
                    "defer_loading": True,
                },
            ],
        )
        result = _flatten_namespace_tool(ns)

        assert result[0]["metadata"].get("defer_loading") is True
        assert result[0]["metadata"].get("format") == {"type": "text"}

    def test_intrinsic_child_type(self):
        """A server-tool child is recognized as an intrinsic definition."""
        ns = _ns_tool(
            "ns",
            children=[
                {"type": "web_search", "name": "search"},
            ],
        )
        result = _flatten_namespace_tool(ns)
        assert len(result) == 1
        assert result[0]["tool_group"] == "ns"
        assert result[0]["type"] == "intrinsic"
        assert result[0]["metadata"].get("intrinsic_kind") == "web_search"

    def test_passthrough_child_type(self):
        """An unknown child type goes through _synthesize_passthrough_tool."""
        ns = _ns_tool(
            "ns",
            children=[
                {"type": "totally_unknown_type", "name": "x"},
            ],
        )
        result = _flatten_namespace_tool(ns)
        assert len(result) == 1
        assert result[0]["tool_group"] == "ns"
        assert result[0]["metadata"].get("provider_type") == "totally_unknown_type"

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
        assert result[0]["tool_group"] == "ns"

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
# Full request conversion integration tests
# ===========================================================================


class TestNamespaceRequestConversion:
    """Integration tests for namespace tools through the full converter."""

    def test_passthrough_tool_carries_its_rename_upstream(self):
        """A hosted tool keeps the name dedup gave it, not the one it came with.

        Hosted types are re-emitted from the untouched provider dict stored
        in ``_passthrough``.  Left alone that dict still holds the original
        name, so a tool renamed to break a collision would reintroduce it on
        the wire and disagree with every other part of the request.
        """
        request = {
            "model": "test-model",
            "tools": [
                {"type": "function", "name": "wait", "parameters": {}},
                {
                    "type": "namespace",
                    "name": "agents",
                    "tools": [{"type": "local_shell", "name": "wait"}],
                },
            ],
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "go"}],
                }
            ],
        }
        pipe = ConversionPipeline("openai_responses", "openai_responses")
        upstream = pipe.convert_request(request)

        names = [t.get("name") for t in upstream["tools"]]
        assert names == ["wait", "agents_wait"], names

    def test_passthrough_tool_without_a_name_does_not_gain_one(self):
        """Nameless hosted tools must not acquire the synthesized IR name."""
        request = {
            "model": "test-model",
            "tools": [{"type": "web_search_preview"}],
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "go"}],
                }
            ],
        }
        pipe = ConversionPipeline("openai_responses", "openai_responses")
        upstream = pipe.convert_request(request)

        assert "name" not in upstream["tools"][0], upstream["tools"][0]

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

    def test_long_grouped_name_is_left_to_the_target_leg(self):
        """Naming is the target's call, so this leg leaves the collision.

        Whether a group name has to be folded into the tool's own name at all
        depends on what the target can express, which the source side is not
        what knows.  Both tools keep their name here and keep their group; the
        pipeline's `qualify_tool_names` is what makes them unique, and it is
        where the generated-name warning comes from.
        """
        long_name = "x" * 64
        context = ConversionContext()
        request = {
            "model": "gpt-5",
            "tools": [
                _func_tool(long_name, "Top-level"),
                _ns_tool("functions", children=[_func_tool(long_name, "Namespaced")]),
            ],
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [{"type": "input_text", "text": "hi"}],
                },
            ],
        }
        ir = self.converter.request_from_provider(request, context=context)

        names = [t["name"] for t in ir["tools"]]
        assert names == [long_name, long_name]
        assert [t.get("tool_group") for t in ir["tools"]] == [None, "functions"]

    def test_namespace_collision_with_toplevel_is_left_to_the_target_leg(self):
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
        assert tool_names == ["exec", "exec"], (
            "the source leg flattens and records the group; naming is the "
            "target leg's, so the collision survives to here"
        )
        assert [t.get("tool_group") for t in ir["tools"]] == [None, "functions"]
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

        pipeline = ConversionPipeline("openai_responses", "messages")
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
