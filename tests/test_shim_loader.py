"""Tests for the declarative YAML provider shim loader."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from llm_rosetta.shims.provider_shim import (
    _reset_registry,
    get_shim,
)
from llm_rosetta.shims.providers import load_providers, _load_transforms


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the shim registry before and after each test."""
    _reset_registry()
    yield
    _reset_registry()


class TestLoadTransforms:
    """Unit tests for _load_transforms helper."""

    def test_no_transforms_file(self, tmp_path: Path):
        """Returns empty tuples when transforms.py does not exist."""
        from_t, to_t = _load_transforms(tmp_path)
        assert from_t == ()
        assert to_t == ()

    def test_transforms_with_to_only(self, tmp_path: Path):
        """Loads to_transforms from transforms.py."""
        tf = tmp_path / "transforms.py"
        tf.write_text(
            textwrap.dedent("""\
            from llm_rosetta.shims.transforms import strip_fields
            to_transforms = (strip_fields("foo"),)
        """)
        )
        from_t, to_t = _load_transforms(tmp_path)
        assert from_t == ()
        assert len(to_t) == 1
        # Verify the transform works
        body = {"foo": 1, "bar": 2}
        result = to_t[0](body)
        assert "foo" not in result
        assert result["bar"] == 2

    def test_transforms_with_both(self, tmp_path: Path):
        """Loads both from_transforms and to_transforms."""
        tf = tmp_path / "transforms.py"
        tf.write_text(
            textwrap.dedent("""\
            from llm_rosetta.shims.transforms import strip_fields, rename_field
            to_transforms = (strip_fields("x"),)
            from_transforms = (rename_field("a", "b"),)
        """)
        )
        from_t, to_t = _load_transforms(tmp_path)
        assert len(from_t) == 1
        assert len(to_t) == 1


class TestLoadProviders:
    """Integration tests for load_providers directory scanner."""

    def _make_provider_dir(
        self,
        parent: Path,
        name: str,
        yaml_content: str,
        transforms_content: str | None = None,
    ) -> Path:
        """Create a provider directory with provider.yaml and optional transforms.py."""
        d = parent / name
        d.mkdir()
        (d / "provider.yaml").write_text(yaml_content)
        if transforms_content:
            (d / "transforms.py").write_text(transforms_content)
        return d

    def test_loads_from_builtin_directory(self):
        """Verify the real providers/ directory loads all 11 built-in shims."""
        shims = load_providers()
        names = {s.name for s in shims}
        assert names == {
            "openai",
            "openai_responses",
            "anthropic",
            "google",
            "deepseek",
            "minimax",
            "moonshot",
            "qwen",
            "volcengine",
            "xai",
            "zhipu",
        }

    def test_all_registered_after_load(self):
        """After load_providers, all shims are queryable via get_shim."""
        load_providers()
        for name in (
            "openai",
            "anthropic",
            "google",
            "deepseek",
            "volcengine",
            "xai",
            "qwen",
            "moonshot",
            "minimax",
            "zhipu",
        ):
            shim = get_shim(name)
            assert shim is not None
            assert shim.name == name

    def test_volcengine_has_transforms(self):
        """Volcengine shim should have strip_fields transforms loaded."""
        load_providers()
        v = get_shim("volcengine")
        assert v is not None
        assert len(v.to_transforms) == 1
        assert len(v.from_transforms) == 0
        # Verify it strips the right fields
        body = {"logprobs": True, "top_logprobs": 5, "messages": []}
        result = v.to_transforms[0](body)
        assert "logprobs" not in result
        assert "messages" in result

    def test_base_types_correct(self):
        """Each shim should have the expected base converter type."""
        load_providers()
        expected = {
            "openai": "openai_chat",
            "openai_responses": "openai_responses",
            "anthropic": "anthropic",
            "google": "google",
            "deepseek": "openai_chat",
            "minimax": "openai_chat",
            "moonshot": "openai_chat",
            "qwen": "openai_chat",
            "volcengine": "openai_chat",
            "xai": "openai_chat",
            "zhipu": "openai_chat",
        }
        for name, base in expected.items():
            shim = get_shim(name)
            assert shim is not None, f"Shim {name!r} not found"
            assert shim.base == base, (
                f"{name}: expected base={base!r}, got {shim.base!r}"
            )

    def test_all_shims_have_logos(self):
        """Every built-in shim should have a logo URL set."""
        shims = load_providers()
        for shim in shims:
            assert shim.logo is not None, f"Shim {shim.name!r} missing logo"
            assert shim.logo.startswith("https://"), (
                f"Shim {shim.name!r} logo should be an HTTPS URL"
            )

    def test_skips_non_directory(self, tmp_path: Path, monkeypatch):
        """Files in the providers directory are ignored."""
        (tmp_path / "not_a_dir.txt").write_text("hello")
        self._make_provider_dir(tmp_path, "valid", "name: valid\nbase: openai_chat\n")
        import llm_rosetta.shims.providers as mod

        monkeypatch.setattr(mod, "_PROVIDERS_DIR", tmp_path)
        shims = load_providers()
        assert len(shims) == 1
        assert shims[0].name == "valid"

    def test_skips_dir_without_yaml(self, tmp_path: Path, monkeypatch):
        """Directories without provider.yaml are skipped."""
        (tmp_path / "empty_dir").mkdir()
        self._make_provider_dir(tmp_path, "valid", "name: valid\nbase: openai_chat\n")
        import llm_rosetta.shims.providers as mod

        monkeypatch.setattr(mod, "_PROVIDERS_DIR", tmp_path)
        shims = load_providers()
        assert len(shims) == 1

    def test_skips_yaml_without_required_fields(self, tmp_path: Path, monkeypatch):
        """YAML without 'name' or 'base' is skipped with warning."""
        self._make_provider_dir(tmp_path, "bad", "description: no name or base\n")
        self._make_provider_dir(tmp_path, "good", "name: good\nbase: openai_chat\n")
        import llm_rosetta.shims.providers as mod

        monkeypatch.setattr(mod, "_PROVIDERS_DIR", tmp_path)
        shims = load_providers()
        assert len(shims) == 1
        assert shims[0].name == "good"
