"""Tests for namespace tool flattening in OpenAI Responses converter."""

from typing import Any

import pytest

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.llm.open_responses.converter import (
    OpenAIResponsesConverter,
)
from llm_rosetta.converters.llm.open_responses.tool_ops import (
    OpenResponsesToolOps,
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
        assert result[0].get("intrinsic_kind") == "web_search"

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
        ir_direct = OpenResponsesToolOps.p_tool_definition_to_ir(child)
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
        result = OpenResponsesToolOps.p_tool_definition_to_ir(ns)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_regular_function_returns_single(self):
        result = OpenResponsesToolOps.p_tool_definition_to_ir(_func_tool("regular"))
        assert isinstance(result, dict)
        assert result["name"] == "regular"

    def test_empty_namespace_returns_empty_list(self):
        result = OpenResponsesToolOps.p_tool_definition_to_ir(
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

        The rename only exists on a flat target: one that carries the group
        keeps the child's own name and hands the container back instead — the
        companion test below.  The neutral base is the flat side here.
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
        pipe = ConversionPipeline("open_responses", "open_responses")
        upstream = pipe.convert_request(request)

        names = [t.get("name") for t in upstream["tools"]]
        assert names == ["wait", "agents_wait"], names

    def test_a_carrying_target_keeps_the_passthrough_tool_inside_its_container(self):
        """The other side of the same request: no rename, the container instead.

        A target that puts the group on the wire separates the two ``wait``
        tools by position, so the hosted tool goes back out under the name the
        client gave it, inside the client's container.
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
        pipe = ConversionPipeline("open_responses", "openai--open_responses")
        upstream = pipe.convert_request(request)

        assert upstream["tools"][0]["name"] == "wait"
        container = upstream["tools"][1]
        assert container["type"] == "namespace"
        assert container["name"] == "agents"
        assert [child["name"] for child in container["tools"]] == ["wait"]

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
        pipe = ConversionPipeline("openai--open_responses", "openai--open_responses")
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

        pipeline = ConversionPipeline("openai--open_responses", "chat_completions")
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

        pipeline = ConversionPipeline("openai--open_responses", "messages")
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


# ===========================================================================
# Round trips
# ===========================================================================


def _grouping(ir_tools: list[Any]) -> list[tuple[Any, Any, Any]]:
    """The part of a tool list a container has to reproduce exactly."""
    return [
        (t.get("name"), t.get("tool_group"), t.get("tool_group_description"))
        for t in ir_tools
    ]


def _round_trip_request(**overrides: Any) -> dict[str, Any]:
    request: dict[str, Any] = {
        "model": "gpt-5",
        "tool_choice": "auto",
        "tools": [
            _func_tool("plain", "A tool declared on its own"),
            _ns_tool(
                "tasks",
                description="Task management tools.",
                children=[
                    _func_tool("exec", "Run a command"),
                    _func_tool("wait", "Wait for a process"),
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
    request.update(overrides)
    return request


class TestContainerRoundTrip:
    """A→IR→A and A→IR→B for a container and for the ``tool_choice`` beside it.

    The IR keeps every tool flat, with ``tool_group`` recording where it came
    from; a writer that can carry the group rebuilds the container on the way
    out.  With the grouping in the IR the correctness question is no longer
    "is the name right" but "did the IR survive the round trip unchanged", so
    these tests compare names, groups, group descriptions and order rather
    than the wire shape alone.
    """

    def _ctx(self, *, carries_groups: bool) -> ConversionContext:
        ctx = ConversionContext()
        ctx.options["tool_groups"] = carries_groups
        return ctx

    def test_a_carrying_target_round_trips_the_grouping_through_the_wire(self):
        converter = OpenAIResponsesConverter()
        ir = converter.request_from_provider(_round_trip_request())
        assert _grouping(ir["tools"]) == [
            ("plain", None, None),
            ("exec", "tasks", "Task management tools."),
            ("wait", "tasks", "Task management tools."),
        ]

        wire, _ = converter.request_to_provider(
            ir, context=self._ctx(carries_groups=True)
        )
        assert [t["type"] for t in wire["tools"]] == ["function", "namespace"]
        assert wire["tools"][1]["name"] == "tasks"

        back = converter.request_from_provider(wire)
        assert _grouping(back["tools"]) == _grouping(ir["tools"]), (
            "the container is the client's own, so reading it back has to "
            "produce the same names, groups and order the client's grouping "
            "produced the first time"
        )

    def test_a_token_only_target_round_trips_its_own_flat_names(self):
        """The other half of the matrix: a flat wire invents no grouping.

        This is the behaviour the flat targets keep.  Its cost is visible
        here — the group is gone from the second read — which is why the
        carry path exists rather than being a formatting preference.
        """
        converter = OpenAIResponsesConverter()
        ir = converter.request_from_provider(_round_trip_request())

        wire, _ = converter.request_to_provider(
            ir, context=self._ctx(carries_groups=False)
        )
        assert [t["type"] for t in wire["tools"]] == ["function"] * 3

        back = converter.request_from_provider(wire)
        assert [t.get("tool_group") for t in back["tools"]] == [None, None, None]

    def test_every_ordinary_tool_choice_survives_the_round_trip(self):
        """A selector that names no container is not made lossy by any of this."""
        converter = OpenAIResponsesConverter()
        for wire_choice, ir_mode in [
            ("auto", "auto"),
            ("required", "any"),
            ("none", "none"),
            ({"type": "function", "name": "plain"}, "tool"),
        ]:
            ir = converter.request_from_provider(
                _round_trip_request(tool_choice=wire_choice)
            )
            assert ir["tool_choice"]["mode"] == ir_mode, wire_choice
            wire, _ = converter.request_to_provider(
                ir, context=self._ctx(carries_groups=True)
            )
            assert wire["tool_choice"] == wire_choice, ir_mode

    def test_a_container_addressed_tool_choice_is_the_one_documented_loss(self):
        """The IR has no group slot for a selector, so this shape degrades.

        Pinned rather than fixed: the IR ``ToolChoice`` states a tool by name,
        and a call inside a container cannot be addressed by a top-level name
        on any upstream measured so far.  The read leg reports the drop and
        falls back to ``auto``, which is what the writer can then reproduce.
        """
        converter = OpenAIResponsesConverter()
        ctx = ConversionContext()
        ir = converter.request_from_provider(
            _round_trip_request(
                tool_choice={
                    "type": "namespace",
                    "name": "tasks",
                    "tools": [{"type": "function", "name": "exec"}],
                }
            ),
            context=ctx,
        )
        assert ir["tool_choice"]["mode"] == "auto"
        assert any("namespace tool_choice" in w for w in ctx.warnings)

        wire, _ = converter.request_to_provider(
            ir, context=self._ctx(carries_groups=True)
        )
        assert wire["tool_choice"] == "auto"

    def test_the_wire_a_carrying_target_writes_is_a_fixed_point(self):
        """A→IR→B→IR→B: the container the writer emits reads back unchanged.

        Idempotence is the property that makes the pass-through safe to sit in
        a request loop — a second turn carrying the previous body must not
        shift a name or reshape a container.
        """
        import json

        from llm_rosetta.pipeline import ConversionPipeline

        def convert(body: dict[str, Any]) -> dict[str, Any]:
            pipe = ConversionPipeline("open_responses", "openai--open_responses")
            return pipe.convert_request(body)

        first = convert(dict(_round_trip_request()))
        second = convert(json.loads(json.dumps(first)))
        assert second["tools"] == first["tools"]

    def test_the_response_leg_returns_the_call_to_its_container(self):
        """A→IR→B→IR: the upstream's ``(name, namespace)`` reaches the client."""
        from llm_rosetta.pipeline import ConversionPipeline

        pipe = ConversionPipeline("open_responses", "openai--open_responses")
        pipe.convert_request(dict(_round_trip_request()))

        out = pipe.convert_response(
            {
                "id": "resp_1",
                "object": "response",
                "created_at": 1,
                "status": "completed",
                "model": "gpt-5",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "exec",
                        "namespace": "tasks",
                        "arguments": "{}",
                    }
                ],
                "usage": {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            }
        )
        calls = [i for i in out["output"] if i.get("type") == "function_call"]
        assert len(calls) == 1
        assert calls[0]["name"] == "exec"
        assert calls[0]["namespace"] == "tasks"


class TestIllegalContainerName:
    """A container name the target's rule rejects is reported, not repaired.

    The container name is the client's own identity for the tools inside it.
    Repairing it would need a wire-group → client-group translation so the
    response leg could hand the client back the namespace it wrote, and no
    such translation exists; a repaired container would therefore return under
    a namespace the client never wrote.  So the writer leaves the name alone
    and reports what that costs: the upstream refuses the whole request, and
    the report says which container and which rule, instead of leaving the
    client with nothing but the upstream's own 400.

    Pinned from both sides — an illegal name reports, a legal one stays quiet —
    so returning to "unrepaired *and* unreported" reds a test.
    """

    @staticmethod
    def _convert(name: str, target: str = "openai--open_responses"):
        pipe = ConversionPipeline("open_responses", target)
        body = pipe.convert_request(
            {
                "model": "test-model",
                "tool_choice": "auto",
                "tools": [
                    _func_tool("plain", "A tool declared on its own"),
                    _ns_tool(name, children=[_func_tool("wait", "Wait for it")]),
                ],
                "input": [
                    {
                        "type": "message",
                        "role": "user",
                        "content": [{"type": "input_text", "text": "hi"}],
                    }
                ],
            }
        )
        return body, [w for w in pipe.warnings if "not a legal group name" in w]

    @pytest.mark.parametrize("name", ["my.tools", "ta sks"])
    def test_an_illegal_container_name_is_reported(self, name: str):
        body, reports = self._convert(name)
        container = next(t for t in body["tools"] if t.get("type") == "namespace")
        assert container["name"] == name, (
            "the name is written as the client gave it; the report, not a "
            "rename, is what this writer owes the caller"
        )
        assert len(reports) == 1, reports
        assert name in reports[0]
        assert "reject" in reports[0]

    def test_a_legal_container_name_stays_quiet(self):
        _, reports = self._convert("tasks")
        assert reports == []

    @pytest.mark.parametrize("target", ["open_responses", "chat_completions"])
    def test_a_flattening_target_has_no_container_to_report(self, target: str):
        body, reports = self._convert("my.tools", target)
        assert "namespace" not in [t.get("type") for t in body["tools"]]
        assert reports == [], (
            "the flat path legalises the folded child name, so there is no "
            "container left for the upstream to refuse"
        )
