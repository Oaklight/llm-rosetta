"""Tests for sanitize_schema and _flatten_combination in converters/base/schema.py."""

from llm_rosetta.converters.base.helpers.schema import (
    sanitize_schema,
    _flatten_combination,
)


class TestFlattenCombination:
    """Tests for _flatten_combination deep-merge behavior."""

    def test_allof_preserves_parent_properties(self):
        """allOf flattening should merge properties, not overwrite."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name", "age"],
            "allOf": [{"type": "object", "properties": {"age": {"type": "integer"}}}],
        }
        result = _flatten_combination(schema)
        assert "name" in result["properties"]
        assert "age" in result["properties"]

    def test_anyof_preserves_parent_properties(self):
        """anyOf nullable flattening should merge properties."""
        schema = {
            "type": "object",
            "properties": {"base_field": {"type": "string"}},
            "required": ["base_field", "extra"],
            "anyOf": [
                {"type": "object", "properties": {"extra": {"type": "integer"}}},
                {"type": "null"},
            ],
        }
        result = _flatten_combination(schema)
        assert "base_field" in result["properties"]
        assert "extra" in result["properties"]

    def test_oneof_preserves_parent_properties(self):
        """oneOf flattening should merge properties."""
        schema = {
            "type": "object",
            "properties": {"parent": {"type": "string"}},
            "oneOf": [
                {"type": "object", "properties": {"child": {"type": "integer"}}},
                {"type": "null"},
            ],
        }
        result = _flatten_combination(schema)
        assert "parent" in result["properties"]
        assert "child" in result["properties"]

    def test_overlay_wins_on_conflict(self):
        """When both have the same property, overlay should win."""
        schema = {
            "type": "object",
            "properties": {"field": {"type": "string", "description": "old"}},
            "allOf": [
                {"properties": {"field": {"type": "string", "description": "new"}}},
            ],
        }
        result = _flatten_combination(schema)
        assert result["properties"]["field"]["description"] == "new"

    def test_no_combination_keywords(self):
        """Schema without anyOf/oneOf/allOf should be returned as-is."""
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        result = _flatten_combination(schema)
        assert result is schema  # Same object reference


class TestSanitizeSchemaRequiredValidation:
    """Tests for required vs properties validation in sanitize_schema."""

    def test_orphaned_required_stripped(self):
        """Required entries not in properties should be removed."""
        schema = {
            "type": "object",
            "properties": {"location": {"type": "string"}},
            "required": ["location", "nonexistent"],
        }
        result = sanitize_schema(schema)
        assert result["required"] == ["location"]

    def test_all_required_orphaned(self):
        """If ALL required entries are orphaned, remove the key entirely."""
        schema = {
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["a", "b"],
        }
        result = sanitize_schema(schema)
        assert "required" not in result

    def test_valid_required_preserved(self):
        """Required entries that exist in properties should be kept."""
        schema = {
            "type": "object",
            "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
            "required": ["a", "b"],
        }
        result = sanitize_schema(schema)
        assert result["required"] == ["a", "b"]

    def test_ref_resolution_preserves_properties(self):
        """$ref resolution should deep-merge, not overwrite properties."""
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "required": ["name", "id"],
            "allOf": [{"$ref": "#/$defs/Base"}],
            "$defs": {
                "Base": {
                    "type": "object",
                    "properties": {"id": {"type": "string"}},
                }
            },
        }
        result = sanitize_schema(schema)
        assert "name" in result["properties"]
        assert "id" in result["properties"]
        assert set(result["required"]) == {"name", "id"}

    def test_nested_required_validated(self):
        """Nested schemas should also have required validated."""
        schema = {
            "type": "object",
            "properties": {
                "nested": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                    "required": ["x", "ghost"],
                }
            },
        }
        result = sanitize_schema(schema)
        nested = result["properties"]["nested"]
        assert nested["required"] == ["x"]

    def test_required_without_properties_unchanged(self):
        """Required without properties key should be left as-is."""
        schema = {
            "type": "object",
            "required": ["a", "b"],
        }
        result = sanitize_schema(schema)
        assert result["required"] == ["a", "b"]
