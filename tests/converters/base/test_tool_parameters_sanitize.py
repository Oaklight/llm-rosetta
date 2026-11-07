"""Tests for reading stringified numeric constraints back into numbers.

The Antigravity CLI serializes every JSON Schema keyword as a string, so a
tool schema arrives with ``minLength: "1"`` where every strict target expects
the number (#1426).  The intent is unambiguous only where it really is: the
keyword decides what can be read, and anything else is forwarded untouched.
"""

import pytest

from llm_rosetta.converters.base.tools.schema import (
    normalize_numeric_constraints,
    sanitize_schema,
    sanitize_tool_parameters,
)

INTEGER_KEYWORDS = [
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "minProperties",
    "maxProperties",
]

NUMBER_KEYWORDS = ["minimum", "maximum", "multipleOf"]


class TestIntegerKeywords:
    """A count keyword takes a non-negative integer and nothing else."""

    @pytest.mark.parametrize("keyword", INTEGER_KEYWORDS)
    def test_a_plain_count_string_becomes_the_number(self, keyword):
        assert normalize_numeric_constraints({keyword: "1"}) == {keyword: 1}

    @pytest.mark.parametrize("value", ["0", "10", "007"])
    def test_leading_zeroes_are_read_as_the_count_they_spell(self, value):
        """``"007"`` is not valid JSON, but it is unambiguously a count."""
        assert normalize_numeric_constraints({"minItems": value}) == {
            "minItems": int(value)
        }

    @pytest.mark.parametrize(
        "value", ["1.0", "-1", "1e2", "true", "NaN", "abc", "", " 1", "1.5"]
    )
    def test_anything_but_a_plain_count_is_left_alone(self, value):
        """Each of these is a number at best, and this keyword takes a count."""
        assert normalize_numeric_constraints({"minLength": value}) == {
            "minLength": value
        }


class TestNumberKeywords:
    """A number keyword takes any JSON number, negative or fractional."""

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("-1", -1),
            ("1.0", 1.0),
            ("1e-3", 0.001),
            ("2", 2),
            ("-0.5", -0.5),
        ],
    )
    def test_a_number_string_becomes_the_number(self, value, expected):
        assert normalize_numeric_constraints({"minimum": value}) == {
            "minimum": expected
        }

    @pytest.mark.parametrize("keyword", NUMBER_KEYWORDS)
    def test_every_number_keyword_is_read(self, keyword):
        assert normalize_numeric_constraints({keyword: "-2"}) == {keyword: -2}

    @pytest.mark.parametrize(
        "value", ["01", "true", "false", "NaN", "Infinity", "-Infinity", "abc", ""]
    )
    def test_a_non_number_is_left_alone(self, value):
        assert normalize_numeric_constraints({"minimum": value}) == {"minimum": value}


class TestNestedSchemas:
    """The walk reaches every schema node a tool definition can carry."""

    def test_properties_and_items_are_walked(self):
        schema = {
            "type": "object",
            "properties": {"tags": {"type": "array", "items": {"minItems": "2"}}},
        }
        result = normalize_numeric_constraints(schema)
        assert result["properties"]["tags"]["items"]["minItems"] == 2

    def test_combination_branches_are_walked(self):
        schema = {"anyOf": [{"type": "string", "minLength": "3"}, {"type": "null"}]}
        result = normalize_numeric_constraints(schema)
        assert result["anyOf"][0]["minLength"] == 3

    def test_definitions_are_walked(self):
        """A ``$ref`` is inlined later, so the definition is where it lives."""
        schema = {
            "$defs": {"Name": {"type": "string", "minLength": "1"}},
            "properties": {"name": {"$ref": "#/$defs/Name"}},
        }
        assert normalize_numeric_constraints(schema)["$defs"]["Name"]["minLength"] == 1

    def test_a_non_numeric_value_under_a_constraint_keyword_is_untouched(self):
        """A property *named* after a keyword is a name, not a constraint."""
        schema = {"properties": {"minimum": {"type": "number"}}}
        assert normalize_numeric_constraints(schema) == schema


class TestNoMutation:
    """The IR keeps what the source sent; only the outbound copy changes."""

    def test_the_input_is_not_mutated(self):
        schema = {"properties": {"a": {"minLength": "1"}}}
        normalize_numeric_constraints(schema)
        assert schema == {"properties": {"a": {"minLength": "1"}}}

    def test_a_schema_with_nothing_to_read_comes_back_by_identity(self):
        """The sanitizer caches by identity-assumed immutability; keep it cheap."""
        schema = {"properties": {"a": {"type": "string", "minLength": 1}}}
        assert normalize_numeric_constraints(schema) is schema


class TestIssues:
    """What cannot be read is reported with its path, never guessed at."""

    def test_an_unreadable_count_is_reported_with_its_path(self):
        issues: list[str] = []
        normalize_numeric_constraints(
            {"properties": {"age": {"minLength": "abc"}}},
            path="parameters",
            issues=issues,
        )
        assert len(issues) == 1
        assert "parameters.properties.age.minLength" in issues[0]
        assert "'abc'" in issues[0]
        assert "integer" in issues[0]

    def test_an_unreadable_number_says_number_not_integer(self):
        issues: list[str] = []
        normalize_numeric_constraints({"maximum": "abc"}, issues=issues)
        assert len(issues) == 1
        assert "number" in issues[0]

    def test_a_value_that_was_read_is_not_reported(self):
        issues: list[str] = []
        normalize_numeric_constraints({"minLength": "1"}, issues=issues)
        assert issues == []

    def test_a_list_index_is_spelled_in_the_path(self):
        issues: list[str] = []
        normalize_numeric_constraints(
            {"anyOf": [{"minLength": "x"}]}, path="parameters", issues=issues
        )
        assert "parameters.anyOf[0].minLength" in issues[0]


class TestSanitizeSchema:
    """The public sanitizer applies the normalization on the way out."""

    def test_a_stringified_count_reaches_the_sanitized_schema_as_a_number(self):
        result = sanitize_schema(
            {"type": "object", "properties": {"name": {"minLength": "1"}}}
        )
        assert result["properties"]["name"]["minLength"] == 1

    def test_a_stringified_number_survives_sanitization_as_a_number(self):
        result = sanitize_schema({"type": "number", "multipleOf": "1e-3"})
        assert result["multipleOf"] == 0.001

    def test_the_schema_is_reached_through_a_ref(self):
        result = sanitize_schema(
            {
                "$defs": {"Name": {"type": "string", "minLength": "1"}},
                "type": "object",
                "properties": {"name": {"$ref": "#/$defs/Name"}},
            }
        )
        assert result["properties"]["name"]["minLength"] == 1

    def test_an_unreadable_value_is_forwarded_unchanged(self):
        result = sanitize_schema({"type": "string", "maxLength": "abc"})
        assert result["maxLength"] == "abc"


class TestSanitizeToolParameters:
    """The tool-definition entry point names the tool and the path."""

    def test_an_unreadable_constraint_is_recorded_against_the_tool(self):
        warnings: list[str] = []
        sanitize_tool_parameters(
            {"properties": {"name": {"minLength": "abc"}}},
            tool_name="send_message",
            tool_index=2,
            warnings=warnings,
        )
        assert len(warnings) == 1
        assert "send_message" in warnings[0]
        assert "tools[2].parameters.properties.name.minLength" in warnings[0]

    def test_a_renamed_tool_is_named_the_way_the_client_wrote_it(self):
        """The name pass runs first, so the tool's own name is the upstream one.

        The warning has to lead with the spelling the operator can find in the
        request the client wrote, and keep the upstream spelling beside it.
        """
        warnings: list[str] = []
        sanitize_tool_parameters(
            {"minLength": "abc"},
            tool={"name": "srv_fs_read", "metadata": {"_original_name": "srv/fs/read"}},
            tool_index=0,
            warnings=warnings,
        )
        assert warnings[0].startswith("Tool 'srv/fs/read' (upstream 'srv_fs_read')")

    def test_a_tool_whose_name_was_not_rewritten_is_named_once(self):
        warnings: list[str] = []
        sanitize_tool_parameters(
            {"minLength": "abc"},
            tool={"name": "read_file"},
            warnings=warnings,
        )
        assert warnings[0].startswith("Tool 'read_file' carries")

    def test_the_index_is_optional(self):
        warnings: list[str] = []
        sanitize_tool_parameters({"minLength": "abc"}, tool_name="t", warnings=warnings)
        assert "parameters.minLength" in warnings[0]

    def test_every_unreadable_value_earns_its_own_warning(self):
        warnings: list[str] = []
        sanitize_tool_parameters(
            {
                "properties": {
                    "a": {"minLength": "x"},
                    "b": {"minimum": "y"},
                }
            },
            tool_name="t",
            tool_index=0,
            warnings=warnings,
        )
        assert len(warnings) == 2

    def test_a_clean_schema_warns_about_nothing(self):
        warnings: list[str] = []
        sanitize_tool_parameters(
            {"properties": {"a": {"minLength": 1}}},
            tool_name="t",
            tool_index=0,
            warnings=warnings,
        )
        assert warnings == []

    def test_the_issue_is_read_on_a_cache_hit_too(self):
        """Issues are collected before the sanitize cache is consulted.

        The tool-definition cache is what usually makes repeats cheap, but this
        entry point is also reachable directly; the warning must not depend on
        which conversion happened to fill the cache first.
        """
        schema = {"properties": {"a": {"minLength": "still-abc"}}}
        first: list[str] = []
        second: list[str] = []
        sanitize_tool_parameters(schema, tool_name="t", warnings=first)
        sanitize_tool_parameters(schema, tool_name="t", warnings=second)
        assert first == second
        assert len(second) == 1

    def test_extra_strip_keys_still_apply(self):
        result = sanitize_tool_parameters(
            {"type": "object", "additionalProperties": False, "title": "x"},
            tool_name="t",
            extra_strip_keys={"additionalProperties", "title"},
        )
        assert result == {"type": "object"}
