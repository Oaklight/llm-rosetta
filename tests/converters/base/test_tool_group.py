"""Tests for the tool-group helpers: the generated id and the grouping."""

from typing import Any

from llm_rosetta.converters.base.helpers.tool_group import (
    GENERATED_GROUP_ID_LENGTH,
    ToolGroup,
    generated_group_id,
    group_tools,
)


def _tool(
    name: str, group: str | None = None, description: str | None = None
) -> dict[str, Any]:
    tool: dict[str, Any] = {
        "type": "function",
        "name": name,
        "description": "",
        "parameters": {},
    }
    if group is not None:
        tool["tool_group"] = group
    if description is not None:
        tool["tool_group_description"] = description
    return tool


def _shape(entries: list[Any]) -> list[Any]:
    """Each entry as ``name``, or ``(group, [children])``, for easy asserting."""

    def label(entry: Any) -> Any:
        if isinstance(entry, ToolGroup):
            return (entry.name, [t["name"] for t in entry.tools])
        return entry["name"] if isinstance(entry, dict) else entry

    return [label(e) for e in entries]


class TestGeneratedGroupId:
    def test_stable_for_identical_content(self):
        assert generated_group_id("g", ["a", "b"]) == generated_group_id(
            "g", ["a", "b"]
        )

    def test_content_changes_the_id(self):
        assert generated_group_id("g", ["a"]) != generated_group_id("h", ["a"])
        assert generated_group_id("g", ["a"]) != generated_group_id("g", ["b"])
        assert generated_group_id("g", ["a", "b"]) != generated_group_id(
            "g", ["b", "a"]
        )

    def test_name_safe(self):
        ident = generated_group_id("", [])
        assert len(ident) == GENERATED_GROUP_ID_LENGTH
        assert ident.islower()
        # No ``_`` or ``-``: either would be mistaken for the separator the
        # naming ladder joins a group to a tool name with.
        assert all(c.isalnum() for c in ident)


class TestGroupTools:
    def test_flag_off_returns_the_tools_untouched(self):
        tools = [_tool("wait", "tasks"), _tool("plain")]
        out = group_tools(tools, carries_groups=False)
        assert out == tools
        assert out is not tools

    def test_no_groups_returns_the_tools_untouched(self):
        tools = [_tool("a"), _tool("b")]
        assert group_tools(tools, carries_groups=True) == tools

    def test_group_sits_where_its_first_member_stood(self):
        tools = [
            _tool("plain"),
            _tool("wait", "tasks"),
            _tool("edit", "tasks"),
            _tool("tail"),
        ]
        assert _shape(group_tools(tools, carries_groups=True)) == [
            "plain",
            ("tasks", ["wait", "edit"]),
            "tail",
        ]

    def test_two_groups_keep_their_own_positions(self):
        tools = [
            _tool("a", "first"),
            _tool("b"),
            _tool("c", "second"),
            _tool("d", "first"),
        ]
        assert _shape(group_tools(tools, carries_groups=True)) == [
            ("first", ["a", "d"]),
            "b",
            ("second", ["c"]),
        ]

    def test_description_is_carried_once(self):
        tools = [_tool("a", "tasks", "Tools for tasks."), _tool("b", "tasks")]
        (group,) = group_tools(tools, carries_groups=True)
        assert group.description == "Tools for tasks."

    def test_the_first_description_wins(self):
        """The description is the container's, but it is read off its members.

        Every member carries the same one in practice; when they disagree the
        group keeps the first, because that is the container the client wrote
        first and nothing downstream can tell which of them it came from.
        """
        tools = [_tool("a", "tasks", "First."), _tool("b", "tasks", "Second.")]

        (group,) = group_tools(tools, carries_groups=True)

        assert group.description == "First."

    def test_group_without_a_description_reports_empty(self):
        (group,) = group_tools([_tool("a", "tasks")], carries_groups=True)
        assert group.description == ""

    def test_members_are_the_same_dicts(self):
        tool = _tool("a", "tasks")
        (group,) = group_tools([tool], carries_groups=True)
        assert group.tools == (tool,)

    def test_non_dict_entries_pass_through(self):
        tools = ["junk", _tool("a", "tasks")]
        assert _shape(group_tools(tools, carries_groups=True)) == [
            "junk",
            ("tasks", ["a"]),
        ]

    def test_empty_group_name_is_not_a_group(self):
        """`""` is what a nameless container used to flatten to; it groups nothing."""
        tools = [_tool("a", ""), _tool("a", "")]
        assert _shape(group_tools(tools, carries_groups=True)) == ["a", "a"]
