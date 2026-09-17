"""Tests for _append_provider_to_model, _merge_provider_into_model, and _add_new_models helpers."""

from __future__ import annotations

from llm_rosetta.gateway.admin.routes.config import (
    _add_new_models,
    _append_provider_to_model,
    _merge_provider_into_model,
)


class TestAppendProviderToModel:
    """_append_provider_to_model conversion paths."""

    def test_string_entry_to_multi(self):
        models = {"gpt-4o": "provider_a"}
        result = _append_provider_to_model(
            models, "gpt-4o", "provider_b", "gpt-4o", "", {}, ["text"]
        )
        assert result == "appended"
        entry = models["gpt-4o"]
        assert "providers" in entry
        assert len(entry["providers"]) == 2
        assert entry["providers"][0] == {"name": "provider_a", "weight": 1}
        assert entry["providers"][1] == {"name": "provider_b", "weight": 1}
        assert entry["capabilities"] == ["text"]

    def test_single_provider_dict_to_multi(self):
        models = {
            "gpt-4o": {
                "provider": "provider_a",
                "capabilities": ["text", "vision"],
                "type": "llm",
            }
        }
        result = _append_provider_to_model(
            models, "gpt-4o", "provider_b", "gpt-4o", "", {}, ["text"]
        )
        assert result == "appended"
        entry = models["gpt-4o"]
        assert "providers" in entry
        assert "provider" not in entry
        assert entry["providers"][0] == {"name": "provider_a", "weight": 1}
        assert entry["providers"][1] == {"name": "provider_b", "weight": 1}
        # Original capabilities and type preserved
        assert entry["capabilities"] == ["text", "vision"]
        assert entry["type"] == "llm"

    def test_single_provider_dict_preserves_upstream_model(self):
        models = {
            "gpt-4o": {
                "provider": "provider_a",
                "upstream_model": "gpt-4o-2024",
                "capabilities": ["text"],
            }
        }
        result = _append_provider_to_model(
            models, "gpt-4o", "provider_b", "gpt-4o", "", {}, ["text"]
        )
        assert result == "appended"
        entry = models["gpt-4o"]
        # Old upstream_model moved to per-provider
        old_p = entry["providers"][0]
        assert isinstance(old_p, dict)
        assert old_p["upstream_model"] == "gpt-4o-2024"
        assert "upstream_model" not in entry

    def test_multi_provider_append(self):
        models = {
            "gpt-4o": {
                "providers": [
                    {"name": "provider_a", "weight": 1},
                    {"name": "provider_b", "weight": 2},
                ],
                "capabilities": ["text"],
            }
        }
        result = _append_provider_to_model(
            models, "gpt-4o", "provider_c", "gpt-4o", "", {}, ["text"]
        )
        assert result == "appended"
        assert len(models["gpt-4o"]["providers"]) == 3

    def test_multi_provider_skip_duplicate(self):
        models = {
            "gpt-4o": {
                "providers": [
                    {"name": "provider_a", "weight": 1},
                ],
                "capabilities": ["text"],
            }
        }
        result = _append_provider_to_model(
            models, "gpt-4o", "provider_a", "gpt-4o", "", {}, ["text"]
        )
        assert result == "skipped"
        assert len(models["gpt-4o"]["providers"]) == 1

    def test_multi_provider_skip_duplicate_string(self):
        models = {
            "gpt-4o": {
                "providers": ["provider_a", "provider_b"],
                "capabilities": ["text"],
            }
        }
        result = _append_provider_to_model(
            models, "gpt-4o", "provider_a", "gpt-4o", "", {}, ["text"]
        )
        assert result == "skipped"

    def test_prefix_adds_upstream(self):
        models = {"gpt-4o": {"provider": "provider_a", "capabilities": ["text"]}}
        result = _append_provider_to_model(
            models, "gpt-4o", "provider_b", "gpt-4o-base", "pfx/", {}, ["text"]
        )
        assert result == "appended"
        new_p = models["gpt-4o"]["providers"][-1]
        assert isinstance(new_p, dict)
        assert new_p["upstream_model"] == "gpt-4o-base"

    def test_upstream_map(self):
        models = {"gpt-4o": {"provider": "provider_a", "capabilities": ["text"]}}
        result = _append_provider_to_model(
            models,
            "gpt-4o",
            "provider_b",
            "gpt-4o",
            "",
            {"gpt-4o": "internal-gpt-4o"},
            ["text"],
        )
        assert result == "appended"
        new_p = models["gpt-4o"]["providers"][-1]
        assert isinstance(new_p, dict)
        assert new_p["upstream_model"] == "internal-gpt-4o"

    def test_non_dict_ignored(self):
        models = {"gpt-4o": 42}  # type: ignore[dict-item]
        result = _append_provider_to_model(
            models, "gpt-4o", "provider_b", "gpt-4o", "", {}, ["text"]
        )
        assert result == "ignored"


class TestMergeProviderIntoModel:
    """_merge_provider_into_model for rename-merge flow."""

    def test_merge_into_string_entry(self):
        models = {"gpt-4o": "provider_a"}
        new_p = {"name": "provider_b", "weight": 1, "upstream_model": "openai/gpt-4o"}
        _merge_provider_into_model(models, "gpt-4o", new_p, ["text", "tools"])
        entry = models["gpt-4o"]
        assert "providers" in entry
        assert len(entry["providers"]) == 2
        assert entry["providers"][0] == {"name": "provider_a", "weight": 1}
        assert entry["providers"][1] == new_p
        assert entry["capabilities"] == ["text", "tools"]

    def test_merge_into_single_provider_dict(self):
        models = {
            "gpt-4o": {
                "provider": "provider_a",
                "capabilities": ["text", "vision"],
            }
        }
        new_p = {"name": "provider_b", "weight": 1, "upstream_model": "openai/gpt-4o"}
        _merge_provider_into_model(models, "gpt-4o", new_p, ["text"])
        entry = models["gpt-4o"]
        assert "providers" in entry
        assert "provider" not in entry
        assert entry["providers"][0] == {"name": "provider_a", "weight": 1}
        assert entry["providers"][1] == new_p
        assert entry["capabilities"] == ["text", "vision"]

    def test_merge_preserves_existing_upstream_model(self):
        models = {
            "gpt-4o": {
                "provider": "provider_a",
                "upstream_model": "gpt-4o-2024",
                "capabilities": ["text"],
            }
        }
        new_p = {"name": "provider_b", "weight": 1}
        _merge_provider_into_model(models, "gpt-4o", new_p, ["text"])
        entry = models["gpt-4o"]
        old_p = entry["providers"][0]
        assert isinstance(old_p, dict)
        assert old_p["upstream_model"] == "gpt-4o-2024"
        assert "upstream_model" not in entry

    def test_merge_into_multi_provider(self):
        models = {
            "gpt-4o": {
                "providers": [
                    {"name": "provider_a", "weight": 1},
                    {"name": "provider_b", "weight": 2},
                ],
                "capabilities": ["text"],
            }
        }
        new_p = {"name": "provider_c", "weight": 1}
        _merge_provider_into_model(models, "gpt-4o", new_p, ["text"])
        assert len(models["gpt-4o"]["providers"]) == 3
        assert models["gpt-4o"]["providers"][-1] == new_p

    def test_merge_skips_duplicate_provider(self):
        models = {
            "gpt-4o": {
                "providers": [
                    {"name": "provider_a", "weight": 1},
                ],
                "capabilities": ["text"],
            }
        }
        new_p = {"name": "provider_a", "weight": 1, "upstream_model": "x"}
        _merge_provider_into_model(models, "gpt-4o", new_p, ["text"])
        assert len(models["gpt-4o"]["providers"]) == 1

    def test_merge_non_dict_is_noop(self):
        models = {"gpt-4o": 42}  # type: ignore[dict-item]
        new_p = {"name": "provider_b", "weight": 1}
        _merge_provider_into_model(models, "gpt-4o", new_p, ["text"])
        assert models["gpt-4o"] == 42


class TestAddNewModels:
    """_add_new_models adds model entries with correct shapes."""

    def test_add_new(self):
        models: dict = {}
        added, skipped = _add_new_models(
            models,
            ["model-a", "model-b"],
            "prov",
            "",
            {},
            {"type": "llm", "capabilities": ["text"]},
        )
        assert added == ["model-a", "model-b"]
        assert skipped == []
        assert models["model-a"]["provider"] == "prov"

    def test_skip_existing(self):
        models = {"model-a": {"provider": "old"}}
        added, skipped = _add_new_models(
            models,
            ["model-a"],
            "prov",
            "",
            {},
            {"type": "llm", "capabilities": ["text"]},
        )
        assert added == []
        assert skipped == ["model-a"]
        assert models["model-a"]["provider"] == "old"

    def test_prefix_and_upstream(self):
        models: dict = {}
        added, _ = _add_new_models(
            models,
            ["gpt-4o"],
            "prov",
            "pfx/",
            {},
            {"type": "llm", "capabilities": ["text"]},
        )
        assert added == ["pfx/gpt-4o"]
        assert models["pfx/gpt-4o"]["upstream_model"] == "gpt-4o"
