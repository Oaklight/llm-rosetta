"""Tests for the shared Argo model_list_transform hook."""

from __future__ import annotations


from llm_rosetta.shims.providers.argo.model_utils import model_list_transform


class TestModelListTransform:
    """Unit tests for model_list_transform slug generation and upstream mapping."""

    def test_normal_input(self):
        """Human-readable id is slugified; internal_id is mapped."""
        raw = [{"id": "Claude Opus 5", "internal_id": "claudeopus5"}]
        ids, upstream = model_list_transform(raw)
        assert ids == ["claude-opus-5"]
        assert upstream == {"claude-opus-5": "claudeopus5"}

    def test_multiple_models(self):
        """Multiple entries are all processed."""
        raw = [
            {"id": "Claude Opus 5", "internal_id": "claudeopus5"},
            {"id": "GPT 4o Mini", "internal_id": "gpt4omini"},
        ]
        ids, upstream = model_list_transform(raw)
        assert ids == ["claude-opus-5", "gpt-4o-mini"]
        assert upstream == {
            "claude-opus-5": "claudeopus5",
            "gpt-4o-mini": "gpt4omini",
        }

    def test_empty_id_skipped(self):
        """Entries with empty id are excluded from the result."""
        raw = [
            {"id": "", "internal_id": "ghost"},
            {"id": "Claude Opus 5", "internal_id": "claudeopus5"},
        ]
        ids, upstream = model_list_transform(raw)
        assert ids == ["claude-opus-5"]
        assert upstream == {"claude-opus-5": "claudeopus5"}

    def test_missing_internal_id_falls_back_to_id(self):
        """When internal_id is absent, the raw id is used as upstream value."""
        raw = [{"id": "Some Model"}]
        ids, upstream = model_list_transform(raw)
        assert ids == ["some-model"]
        # Slug "some-model" differs from raw id "Some Model" → mapped.
        assert upstream == {"some-model": "Some Model"}

    def test_display_equals_upstream_no_map_entry(self):
        """When slug matches the upstream value, no mapping is needed."""
        raw = [{"id": "gpt-4o", "internal_id": "gpt-4o"}]
        ids, upstream = model_list_transform(raw)
        assert ids == ["gpt-4o"]
        assert upstream == {}

    def test_double_spaces(self):
        """Double spaces produce a single hyphen, not double hyphens."""
        raw = [{"id": "Claude  Opus  5", "internal_id": "claudeopus5"}]
        ids, upstream = model_list_transform(raw)
        assert ids == ["claude-opus-5"]

    def test_special_characters(self):
        """Parentheses, dots, slashes, and other special chars are collapsed."""
        raw = [{"id": "Model (v2.1/beta)", "internal_id": "modelv21beta"}]
        ids, upstream = model_list_transform(raw)
        assert ids == ["model-v2-1-beta"]
        assert upstream == {"model-v2-1-beta": "modelv21beta"}

    def test_leading_trailing_whitespace(self):
        """Leading/trailing whitespace does not produce leading/trailing hyphens."""
        raw = [{"id": "  Claude Opus 5  ", "internal_id": "claudeopus5"}]
        ids, upstream = model_list_transform(raw)
        assert ids == ["claude-opus-5"]
        assert "claude-opus-5" not in ["-claude-opus-5-", "-claude-opus-5"]

    def test_empty_list(self):
        """Empty input returns empty results."""
        ids, upstream = model_list_transform([])
        assert ids == []
        assert upstream == {}

    def test_missing_id_key(self):
        """Entry without id key at all is skipped."""
        raw = [{"internal_id": "orphan"}]
        ids, upstream = model_list_transform(raw)
        assert ids == []
        assert upstream == {}
