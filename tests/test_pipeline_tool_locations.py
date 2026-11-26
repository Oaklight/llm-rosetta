"""Tests for reading where a request body declares its tools.

The reader is what lets an upstream rejection be attributed to the client's
own request: an upstream names ``tools[16]`` or its sanitized spelling, and the
client holds a body in another dialect (#1427).  Each format's shape is
covered, because the path it reports is quoted back to the client verbatim.
"""

from __future__ import annotations

from llm_rosetta.pipeline.tool_locations import (
    ToolDeclaration,
    read_tool_declarations,
)


def _paths(declarations: list[ToolDeclaration]) -> list[str]:
    return [declaration.path for declaration in declarations]


def _names(declarations: list[ToolDeclaration]) -> list[str | None]:
    return [declaration.name for declaration in declarations]


class TestChatCompletions:
    def test_a_function_tool_names_its_function_field(self):
        body = {
            "tools": [
                {"type": "function", "function": {"name": "exec", "parameters": {}}}
            ]
        }
        (declaration,) = read_tool_declarations("chat_completions", body)
        assert declaration.name == "exec"
        assert declaration.path == "tools[0]"
        assert declaration.wrapper == "function"
        assert declaration.schema_field == "parameters"
        assert declaration.field_path("name") == "tools[0].function.name"

    def test_a_custom_tool_carries_its_name_under_custom(self):
        body = {"tools": [{"type": "custom", "custom": {"name": "apply_patch"}}]}
        (declaration,) = read_tool_declarations("chat_completions", body)
        assert (declaration.name, declaration.wrapper) == ("apply_patch", "custom")
        assert declaration.schema_field is None

    def test_a_server_tool_is_positional_but_unnamed(self):
        """A built-in entry can be pointed at by index, and names no tool."""
        body = {
            "tools": [
                {"type": "web_search"},
                {"type": "function", "function": {"name": "exec"}},
            ]
        }
        first, second = read_tool_declarations("chat_completions", body)
        assert first.name is None
        assert first.number == 0
        assert second.number == 1

    def test_legacy_format_names_read_the_same_body(self):
        body = {"tools": [{"type": "function", "function": {"name": "exec"}}]}
        assert _names(read_tool_declarations("chat_completions", body)) == ["exec"]


class TestMessages:
    def test_a_declaration_carries_its_schema_under_input_schema(self):
        body = {"tools": [{"name": "search", "input_schema": {"type": "object"}}]}
        (declaration,) = read_tool_declarations("messages", body)
        assert declaration.name == "search"
        assert declaration.path == "tools[0]"
        assert declaration.wrapper == ""
        assert declaration.schema_field == "input_schema"

    def test_a_server_tool_is_read_like_any_other_entry(self):
        body = {"tools": [{"type": "web_search_20250305", "name": "web_search"}]}
        (declaration,) = read_tool_declarations("messages", body)
        assert declaration.name == "web_search"
        assert declaration.schema_field == "input_schema"


class TestOpenResponses:
    def test_a_flat_function_entry(self):
        body = {"tools": [{"type": "function", "name": "exec", "parameters": {}}]}
        (declaration,) = read_tool_declarations("open_responses", body)
        assert declaration.name == "exec"
        assert declaration.path == "tools[0]"
        assert declaration.schema_field == "parameters"

    def test_a_namespace_container_reports_the_child_path(self):
        """The child is what the client declared; the container is its parent."""
        body = {
            "tools": [
                {
                    "type": "namespace",
                    "name": "fs",
                    "tools": [
                        {"type": "function", "name": "read"},
                        {"type": "function", "name": "write"},
                    ],
                }
            ]
        }
        read, write = read_tool_declarations("open_responses", body)
        assert read.path == "tools[0].tools[0]"
        assert write.path == "tools[0].tools[1]"
        assert (read.number, write.number) == (1, 2)

    def test_a_custom_tool_carries_its_schema_under_schema(self):
        body = {"tools": [{"type": "custom", "name": "apply_patch", "schema": {}}]}
        (declaration,) = read_tool_declarations("open_responses", body)
        assert declaration.schema_field == "schema"


class TestGoogleGenerate:
    def test_camel_case_declarations(self):
        body = {"tools": [{"functionDeclarations": [{"name": "one"}, {"name": "two"}]}]}
        declarations = read_tool_declarations("google_generate", body)
        assert _paths(declarations) == [
            "tools[0].functionDeclarations[0]",
            "tools[0].functionDeclarations[1]",
        ]
        assert _names(declarations) == ["one", "two"]

    def test_snake_case_is_reported_as_the_body_spells_it(self):
        body = {"tools": [{"function_declarations": [{"name": "one"}]}]}
        (declaration,) = read_tool_declarations("google_generate", body)
        assert declaration.path == "tools[0].function_declarations[0]"

    def test_a_bare_declaration_without_the_nested_list(self):
        body = {"tools": [{"name": "one", "parameters": {}}]}
        (declaration,) = read_tool_declarations("google_generate", body)
        assert (declaration.name, declaration.path) == ("one", "tools[0]")

    def test_a_hosted_tool_names_nothing(self):
        body = {"tools": [{"googleSearch": {}}]}
        (declaration,) = read_tool_declarations("google_generate", body)
        assert declaration.name is None

    def test_the_legacy_google_spelling_reads_the_same_body(self):
        body = {"tools": [{"functionDeclarations": [{"name": "one"}]}]}
        assert _names(read_tool_declarations("google_generate", body)) == ["one"]


class TestGoogleInteractions:
    def test_a_flat_function_entry(self):
        body = {"tools": [{"type": "function", "name": "exec", "parameters": {}}]}
        (declaration,) = read_tool_declarations("google_interactions", body)
        assert (declaration.name, declaration.schema_field) == ("exec", "parameters")


class TestUnreadableBodies:
    def test_no_tools(self):
        assert read_tool_declarations("chat_completions", {"model": "x"}) == []

    def test_tools_not_a_list(self):
        assert read_tool_declarations("chat_completions", {"tools": {}}) == []

    def test_a_body_that_is_not_a_mapping(self):
        assert read_tool_declarations("chat_completions", None) == []

    def test_an_unknown_format(self):
        assert read_tool_declarations("not_a_format", {"tools": [{}]}) == []

    def test_a_non_mapping_entry_is_skipped(self):
        body = {"tools": ["nonsense", {"type": "function", "function": {"name": "x"}}]}
        assert _names(read_tool_declarations("chat_completions", body)) == ["x"]
