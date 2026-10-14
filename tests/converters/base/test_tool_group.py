"""Tests for the tool-group helpers: the generated id and the grouping."""

from typing import Any

import pytest

from llm_rosetta.converters.base.helpers.tool_group import (
    GENERATED_GROUP_ID_LENGTH,
    ToolGroup,
    generated_group_id,
    group_tools,
    tool_call_group,
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


class TestToolCallGroup:
    """Reading the group off a call, first-class field first."""

    def test_reads_the_first_class_field(self):
        assert tool_call_group({"tool_name": "wait", "tool_group": "tasks"}) == "tasks"

    def test_reads_a_tool_call_start_event(self):
        """The same reader serves the streaming event, which is a dict too."""
        event = {"type": "tool_call_start", "tool_name": "wait", "tool_group": "tasks"}
        assert tool_call_group(event) == "tasks"

    def test_falls_back_to_the_legacy_metadata_spelling(self):
        """IR built before ``tool_group`` existed keeps working."""
        part = {"tool_name": "wait", "provider_metadata": {"namespace": "tasks"}}
        assert tool_call_group(part) == "tasks"

    def test_the_first_class_field_wins_over_the_legacy_key(self):
        """When both are present they must not be able to disagree."""
        part = {
            "tool_name": "wait",
            "tool_group": "tasks",
            "provider_metadata": {"namespace": "agents"},
        }
        assert tool_call_group(part) == "tasks"

    @pytest.mark.parametrize(
        "part",
        [
            pytest.param({"tool_name": "wait"}, id="neither-field"),
            pytest.param({"tool_name": "wait", "tool_group": ""}, id="empty-field"),
            pytest.param({"tool_name": "wait", "tool_group": 7}, id="non-string-field"),
            pytest.param(
                {"tool_name": "wait", "provider_metadata": {"namespace": ""}},
                id="empty-legacy",
            ),
            pytest.param(
                {"tool_name": "wait", "provider_metadata": {"namespace": 7}},
                id="non-string-legacy",
            ),
            pytest.param(
                {"tool_name": "wait", "provider_metadata": "not-a-dict"},
                id="non-dict-metadata",
            ),
        ],
    )
    def test_absent_reads_as_none(self, part):
        """A missing, empty, or non-string group is no group at all."""
        assert tool_call_group(part) is None
