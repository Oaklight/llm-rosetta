"""Tests for convert_nullable_to_type_array in converters/base/schema.py."""

from llm_rosetta.converters.base.helpers.schema import convert_nullable_to_type_array


class TestConvertNullableToTypeArray:
    """Tests for nullable → type array conversion."""

    def test_simple_nullable_string(self):
        schema = {"type": "string", "nullable": True}
        result = convert_nullable_to_type_array(schema)
        assert result["type"] == ["string", "null"]
        assert "nullable" not in result

    def test_simple_nullable_integer(self):
        schema = {"type": "integer", "nullable": True}
        result = convert_nullable_to_type_array(schema)
        assert result["type"] == ["integer", "null"]
        assert "nullable" not in result

    def test_non_nullable_unchanged(self):
        schema = {"type": "string", "description": "a field"}
        result = convert_nullable_to_type_array(schema)
        assert result["type"] == "string"
        assert result["description"] == "a field"

    def test_nullable_false_stripped(self):
        schema = {"type": "string", "nullable": False}
        result = convert_nullable_to_type_array(schema)
        assert result["type"] == "string"
        assert "nullable" not in result

    def test_type_already_list(self):
        schema = {"type": ["string", "integer"], "nullable": True}
        result = convert_nullable_to_type_array(schema)
        assert result["type"] == ["string", "integer", "null"]
        assert "nullable" not in result

    def test_type_list_already_has_null(self):
        schema = {"type": ["string", "null"], "nullable": True}
        result = convert_nullable_to_type_array(schema)
        assert result["type"] == ["string", "null"]
        assert "nullable" not in result

    def test_nullable_without_type_strips_key(self):
        schema = {"nullable": True, "description": "no type"}
        result = convert_nullable_to_type_array(schema)
        assert "nullable" not in result
        assert "type" not in result
        assert result["description"] == "no type"

    def test_nested_in_properties(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "cwd": {"type": "string", "nullable": True, "default": None},
            },
        }
        result = convert_nullable_to_type_array(schema)
        assert result["properties"]["name"]["type"] == "string"
        assert result["properties"]["cwd"]["type"] == ["string", "null"]
        assert "nullable" not in result["properties"]["cwd"]

    def test_nested_in_items(self):
        schema = {
            "type": "array",
            "items": {"type": "string", "nullable": True},
        }
        result = convert_nullable_to_type_array(schema)
        assert result["items"]["type"] == ["string", "null"]
        assert "nullable" not in result["items"]

    def test_deeply_nested(self):
        schema = {
            "type": "object",
            "properties": {
                "outer": {
                    "type": "object",
                    "properties": {
                        "inner": {"type": "integer", "nullable": True},
                    },
                }
            },
        }
        result = convert_nullable_to_type_array(schema)
        inner = result["properties"]["outer"]["properties"]["inner"]
        assert inner["type"] == ["integer", "null"]
        assert "nullable" not in inner

    def test_preserves_other_fields(self):
        schema = {
            "type": "string",
            "nullable": True,
            "description": "Optional field",
            "default": None,
            "enum": ["a", "b"],
        }
        result = convert_nullable_to_type_array(schema)
        assert result["type"] == ["string", "null"]
        assert result["description"] == "Optional field"
        assert result["default"] is None
        assert result["enum"] == ["a", "b"]
        assert "nullable" not in result

    def test_pydantic_style_schema(self):
        """Reproduce the exact schema from issue #372."""
        schema = {
            "type": "object",
            "properties": {
                "cwd": {
                    "default": None,
                    "title": "Cwd",
                    "type": "string",
                    "nullable": True,
                }
            },
        }
        result = convert_nullable_to_type_array(schema)
        cwd = result["properties"]["cwd"]
        assert cwd["type"] == ["string", "null"]
        assert "nullable" not in cwd
        assert cwd["title"] == "Cwd"  # title not stripped by this function

    def test_list_of_schemas(self):
        schema = {
            "anyOf": [
                {"type": "string", "nullable": True},
                {"type": "integer"},
            ]
        }
        result = convert_nullable_to_type_array(schema)
        assert result["anyOf"][0]["type"] == ["string", "null"]
        assert result["anyOf"][1]["type"] == "integer"

    def test_nullable_with_anyof_no_type(self):
        """nullable + anyOf without type → inject null variant into anyOf."""
        schema = {
            "anyOf": [{"type": "string"}, {"type": "integer"}],
            "nullable": True,
        }
        result = convert_nullable_to_type_array(schema)
        assert "nullable" not in result
        assert {"type": "null"} in result["anyOf"]
        assert len(result["anyOf"]) == 3

    def test_nullable_with_oneof_no_type(self):
        """nullable + oneOf without type → inject null variant into oneOf."""
        schema = {
            "oneOf": [{"type": "string"}, {"type": "integer"}],
            "nullable": True,
        }
        result = convert_nullable_to_type_array(schema)
        assert "nullable" not in result
        assert {"type": "null"} in result["oneOf"]
        assert len(result["oneOf"]) == 3

    def test_nullable_with_anyof_already_has_null(self):
        """nullable + anyOf that already has null → no duplicate."""
        schema = {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "nullable": True,
        }
        result = convert_nullable_to_type_array(schema)
        assert "nullable" not in result
        null_count = sum(1 for v in result["anyOf"] if v == {"type": "null"})
        assert null_count == 1
