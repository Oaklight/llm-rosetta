"""Tests for `qualify_tool_names`, the target-aware naming pass.

Colliding tool names are made unique where the target is known, not where the
container was flattened: whether a group name has to be folded into the tool's
own name at all depends on what the target can express.
"""

import re
from typing import Any

import pytest

from llm_rosetta.capabilities import qualify_tool_names
from llm_rosetta.converters.base.helpers.tool_group import GENERATED_GROUP_MARKER


def _qualify(tools: list[Any], warnings: list[str]) -> list[Any]:
    """Run the pass over a bare tool list, as the pipeline would."""
    return qualify_tool_names({"tools": tools}, warnings=warnings)["tools"]


class TestQualifyToolNames:
    """Tests for the target-aware name qualification pass."""

    @pytest.fixture
    def warnings(self) -> list[str]:
        """Sink for unresolvable-collision warnings; asserted empty by default."""
        return []

    @staticmethod
    def _ir_tool(
        name: str, group: str | None = None, *, generated: bool = False
    ) -> dict[str, Any]:
        """An IR tool.  ``group=None`` is top-level.

        ``generated=True`` marks the group as one this library invented for a
        container that had no name — the spelling the flatten writes — which
        is what decides between collapsing and qualifying.
        """
        tool: dict[str, Any] = {
            "type": "function",
            "name": name,
            "description": "",
            "parameters": {},
        }
        if group is not None:
            tool["tool_group"] = group
        if generated:
            tool["metadata"] = {GENERATED_GROUP_MARKER: True}
        return tool

    def test_named_group_keeps_its_identical_children_apart(self):
        """A group the client named is theirs, so both declarations stand.

        Only a *generated* group folds its byte-identical children; a named
        one is what #955 settled, and the upstreams agree — DeepSeek answers
        "Tool names within a namespace must be unique" and Volcengine
        "duplicate downstream function name" when a container carries the same
        child name twice, so the pair has to be separable.
        """
        tools = [
            self._ir_tool("wait", "tasks"),
            self._ir_tool("wait", "tasks"),
        ]

        result = _qualify(tools, warnings=[])

        names = [t["name"] for t in result]
        assert len(result) == 2, "a named group's children are not folded"
        assert len(set(names)) == 2, names
        assert names[0] == "tasks_wait"

    def test_no_collision_unchanged(self, warnings):
        tools = [self._ir_tool("a"), self._ir_tool("b", "ns")]
        result = _qualify(tools, warnings)
        assert [t["name"] for t in result] == ["a", "b"]

    def test_namespaced_vs_toplevel_collision(self, warnings):
        tools = [
            self._ir_tool("exec"),  # top-level, no namespace
            self._ir_tool("exec", "functions"),  # from namespace
        ]
        result = _qualify(tools, warnings)
        assert result[0]["name"] == "exec"
        assert result[1]["name"] == "functions_exec"
        assert warnings == []

    def test_cross_namespace_collision(self, warnings):
        tools = [
            self._ir_tool("exec", "grp_a"),
            self._ir_tool("exec", "grp_b"),
        ]
        result = _qualify(tools, warnings)
        names = {t["name"] for t in result}
        assert "grp_a_exec" in names
        assert "grp_b_exec" in names
        assert warnings == []

    def test_no_mutation_of_originals(self, warnings):
        original = self._ir_tool("exec", "ns")
        tools = [self._ir_tool("exec"), original]
        result = _qualify(tools, warnings)
        # Original dict should be unchanged
        assert original["name"] == "exec"
        # Result entry is a copy
        assert result[1]["name"] == "ns_exec"
        assert result[1] is not original

    def test_long_qualified_name_truncated(self, warnings):
        long_ns = "a" * 60
        tools = [
            self._ir_tool("tool"),
            self._ir_tool("tool", long_ns),
        ]
        result = _qualify(tools, warnings)
        assert len(result[1]["name"]) <= 64
        assert result[0]["name"] != result[1]["name"]
        # Truncating the namespace still produced a distinct name — no warning
        assert warnings == []

    def test_very_long_name_still_gets_a_distinct_name(self, warnings):
        """A name filling the budget leaves no room to prefix — generate one.

        Letting the tool keep the bare name would put two different tools
        upstream spelled the same way, and every call naming it would be a
        coin flip.  A generated name is a worse hint for the model and a
        correct address for the provider; the warning covers the former.
        """
        long_name = "x" * 64
        tools = [
            self._ir_tool(long_name),
            self._ir_tool(long_name, "ns"),
        ]
        result = _qualify(tools, warnings)

        assert result[0]["name"] == long_name, "the top-level tool is never renamed"
        generated = result[1]["name"]
        assert generated != long_name
        assert len(generated) <= 64
        assert re.fullmatch(r"[a-zA-Z0-9_-]+", generated), generated
        assert result[1]["metadata"]["_original_name"] == long_name
        assert len(warnings) == 1
        assert "generated name" in warnings[0]

    def test_qualified_name_collides_falls_back_to_a_generated_name(self, warnings):
        """The qualified spelling can itself be taken by a top-level tool."""
        tools = [
            self._ir_tool("exec"),
            self._ir_tool("exec", "ns"),
            self._ir_tool("ns_exec"),  # squats on the qualified spelling
        ]
        result = _qualify(tools, warnings)

        names = [t["name"] for t in result]
        assert names[0] == "exec" and names[2] == "ns_exec", "neither is renamed"
        assert len(set(names)) == 3, f"every tool needs its own name; got {names}"
        assert names[1].startswith("ns_exec_")
        assert len(names[1]) <= 64
        assert result[1]["metadata"]["_original_name"] == "exec"
        assert len(warnings) == 1
        assert "generated name" in warnings[0]

    def test_generated_names_are_deterministic(self, warnings):
        """Same tools in, same names out — or every request is a cache miss."""

        def build() -> list[str]:
            sink: list[str] = []
            tools = [
                self._ir_tool("exec"),
                self._ir_tool("exec", "ns"),
                self._ir_tool("ns_exec"),
            ]
            return [t["name"] for t in _qualify(tools, sink)]

        assert build() == build()

    def test_two_unqualifiable_tools_get_different_names(self, warnings):
        """The digest must separate namespaces, not just mark a failure."""
        long_name = "x" * 64
        tools = [
            self._ir_tool(long_name, "grp_a"),
            self._ir_tool(long_name, "grp_b"),
        ]
        result = _qualify(tools, warnings)

        first, second = (t["name"] for t in result)
        assert first != second, "two tools sharing one upstream name is the bug"
        assert len(first) <= 64 and len(second) <= 64

    def test_generated_name_can_itself_be_squatted(self, warnings):
        """The digest is a guess at a free name, so it has to be checked too.

        Take the name the qualifier would generate, hand it to an unrelated
        top-level tool, and ask again: the answer has to move, not collide.
        """
        # "ns_exec" is taken, so this run is the one that reaches the digest.
        squatted = [
            self._ir_tool("exec"),
            self._ir_tool("exec", "ns"),
            self._ir_tool("ns_exec"),
        ]
        generated = _qualify(squatted, [])[1]["name"]
        assert generated.startswith("ns_exec_"), generated

        # Now squat that too, leaving the first digest no longer available.
        tools = squatted + [self._ir_tool(generated)]
        result = _qualify(tools, warnings)

        names = [t["name"] for t in result]
        assert names[2] == "ns_exec" and names[3] == generated, "squatters stay put"
        assert names[1] != generated, f"collided with the squatter: {names}"
        assert len(set(names)) == 4, names
        assert len(names[1]) <= 64

    def test_single_tool_no_dedup(self, warnings):
        tools = [self._ir_tool("only_one", "ns")]
        result = _qualify(tools, warnings)
        assert result[0]["name"] == "only_one"
        assert warnings == []
