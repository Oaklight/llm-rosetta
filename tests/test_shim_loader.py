"""Tests for the declarative YAML provider shim loader."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from llm_rosetta.shims.provider_shim import (
    _reset_registry,
    get_shim,
)
from llm_rosetta.shims.providers import (
    _load_plugin_shims,
    _load_transforms,
    load_providers,
    load_providers_from_dir,
)


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
        """Verify the real providers/ directory loads all 16 built-in shims."""
        shims = load_providers()
        names = {s.name for s in shims}
        assert names == {
            "argo--anthropic",
            "argo--openai_chat",
            "openai",
            "openai_responses",
            "openrouter--openai_chat",
            "openrouter--anthropic",
            "anthropic",
            "google",
            "deepseek",
            "minimax--openai_chat",
            "minimax--anthropic",
            "moonshot",
            "qwen",
            "volcengine--openai_chat",
            "volcengine--openai_responses",
            "xai",
            "zhipu",
        }

    def test_all_registered_after_load(self):
        """After load_providers, all shims are queryable via get_shim."""
        load_providers()
        for name in (
            "openai",
            "openrouter--openai_chat",
            "openrouter--anthropic",
            "anthropic",
            "google",
            "deepseek",
            "volcengine--openai_chat",
            "volcengine--openai_responses",
            "xai",
            "qwen",
            "moonshot",
            "minimax--openai_chat",
            "minimax--anthropic",
            "zhipu",
        ):
            shim = get_shim(name)
            assert shim is not None
            assert shim.name == name

    def test_volcengine_has_transforms(self):
        """Volcengine shim should have strip_fields transforms loaded."""
        load_providers()
        v = get_shim("volcengine--openai_chat")
        assert v is not None
        assert len(v.to_transforms) == 1
        assert len(v.from_transforms) == 0
        # Verify it strips the right fields
        body = {"logprobs": True, "top_logprobs": 5, "messages": []}
        result = v.to_transforms[0](body)
        assert "logprobs" not in result
        assert "messages" in result

    def test_deepseek_has_transforms(self):
        """DeepSeek shim should strip n, logit_bias, seed."""
        load_providers()
        s = get_shim("deepseek")
        assert s is not None
        assert len(s.to_transforms) == 1
        assert len(s.from_transforms) == 0
        body = {"n": 2, "logit_bias": {}, "seed": 42, "messages": []}
        result = s.to_transforms[0](body)
        assert "n" not in result
        assert "logit_bias" not in result
        assert "seed" not in result
        assert "messages" in result

    def test_xai_has_transforms(self):
        """xAI shim should strip logit_bias."""
        load_providers()
        s = get_shim("xai")
        assert s is not None
        assert len(s.to_transforms) == 1
        assert len(s.from_transforms) == 0
        body = {"logit_bias": {"50256": -100}, "messages": []}
        result = s.to_transforms[0](body)
        assert "logit_bias" not in result
        assert "messages" in result

    def test_moonshot_has_transforms(self):
        """Moonshot shim should strip logprobs, top_logprobs, logit_bias, seed."""
        load_providers()
        s = get_shim("moonshot")
        assert s is not None
        assert len(s.to_transforms) == 1
        assert len(s.from_transforms) == 0
        body = {
            "logprobs": True,
            "top_logprobs": 5,
            "logit_bias": {},
            "seed": 123,
            "messages": [],
        }
        result = s.to_transforms[0](body)
        assert "logprobs" not in result
        assert "top_logprobs" not in result
        assert "logit_bias" not in result
        assert "seed" not in result
        assert "messages" in result

    def test_qwen_has_transforms(self):
        """Qwen shim should strip frequency_penalty, logit_bias."""
        load_providers()
        s = get_shim("qwen")
        assert s is not None
        assert len(s.to_transforms) == 1
        assert len(s.from_transforms) == 0
        body = {"frequency_penalty": 0.5, "logit_bias": {}, "messages": []}
        result = s.to_transforms[0](body)
        assert "frequency_penalty" not in result
        assert "logit_bias" not in result
        assert "messages" in result

    def test_minimax_has_transforms(self):
        """MiniMax shim should strip fields + inject reasoning_split."""
        load_providers()
        s = get_shim("minimax--openai_chat")
        assert s is not None
        assert len(s.to_transforms) == 2  # strip_fields + inject_reasoning_split
        assert len(s.from_transforms) == 1  # parse_think_tags
        body = {
            "logprobs": True,
            "top_logprobs": 5,
            "seed": 42,
            "stop": ["\n"],
            "messages": [],
        }
        result = s.to_transforms[0](body)
        assert "logprobs" not in result
        assert "top_logprobs" not in result
        assert "seed" not in result
        assert "stop" not in result
        assert "messages" in result

    def test_zhipu_has_transforms(self):
        """Zhipu shim should strip n, penalties, logprobs, logit_bias, seed."""
        load_providers()
        s = get_shim("zhipu")
        assert s is not None
        assert len(s.to_transforms) == 1
        assert len(s.from_transforms) == 0
        body = {
            "n": 2,
            "presence_penalty": 0.5,
            "frequency_penalty": 0.5,
            "logprobs": True,
            "top_logprobs": 5,
            "logit_bias": {},
            "seed": 42,
            "messages": [],
        }
        result = s.to_transforms[0](body)
        assert "n" not in result
        assert "presence_penalty" not in result
        assert "frequency_penalty" not in result
        assert "logprobs" not in result
        assert "top_logprobs" not in result
        assert "logit_bias" not in result
        assert "seed" not in result
        assert "messages" in result

    def test_base_types_correct(self):
        """Each shim should have the expected base converter type."""
        load_providers()
        expected = {
            "openai": "openai_chat",
            "openai_responses": "openai_responses",
            "openrouter--openai_chat": "openai_chat",
            "openrouter--anthropic": "anthropic",
            "anthropic": "anthropic",
            "google": "google",
            "deepseek": "openai_chat",
            "minimax--openai_chat": "openai_chat",
            "minimax--anthropic": "anthropic",
            "moonshot": "openai_chat",
            "qwen": "openai_chat",
            "volcengine--openai_chat": "openai_chat",
            "volcengine--openai_responses": "openai_responses",
            "xai": "openai_chat",
            "zhipu": "openai_chat",
        }
        for name, base in expected.items():
            shim = get_shim(name)
            assert shim is not None, f"Shim {name!r} not found"
            assert shim.base == base, (
                f"{name}: expected base={base!r}, got {shim.base!r}"
            )

    # Shims that intentionally have no public logo
    _LOGO_EXEMPT = {"argo--anthropic", "argo--openai_chat"}

    def test_all_shims_have_logos(self):
        """Every built-in shim (except exempted ones) should have a logo URL."""
        shims = load_providers()
        for shim in shims:
            if shim.name in self._LOGO_EXEMPT:
                continue
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


class TestLoadProvidersFromDir:
    """Tests for the public load_providers_from_dir API."""

    def test_loads_from_arbitrary_path(self, tmp_path: Path):
        """load_providers_from_dir loads from any directory."""
        d = tmp_path / "mything"
        d.mkdir()
        (d / "provider.yaml").write_text("name: mything\nbase: openai_chat\n")
        shims = load_providers_from_dir(tmp_path)
        assert any(s.name == "mything" for s in shims)

    def test_plugin_transforms_loaded(self, tmp_path: Path):
        """Plugin transforms are loaded from arbitrary directories."""
        d = tmp_path / "myplugin"
        d.mkdir()
        (d / "provider.yaml").write_text("name: myplugin\nbase: openai_chat\n")
        (d / "transforms.py").write_text(
            "from llm_rosetta.shims.transforms import strip_fields\n"
            'to_transforms = (strip_fields("foo"),)\n'
        )
        shims = load_providers_from_dir(tmp_path)
        s = [s for s in shims if s.name == "myplugin"][0]
        assert len(s.to_transforms) == 1
        # Verify the transform works
        body = {"foo": 1, "bar": 2}
        result = s.to_transforms[0](body)
        assert "foo" not in result
        assert result["bar"] == 2


class TestPluginEntryPoints:
    """Tests for the entry-point plugin loader."""

    def test_invokes_entry_points(self, monkeypatch):
        """_load_plugin_shims discovers and calls entry points."""
        calls: list[str] = []

        class FakeEP:
            name = "fake"

            def load(self):
                def register():
                    calls.append("called")

                return register

        class FakeEPs:
            def select(self, *, group: str):
                assert group == "llm_rosetta.shim_providers"
                return [FakeEP()]

        monkeypatch.setattr(
            "llm_rosetta.shims.providers.entry_points", lambda: FakeEPs()
        )
        _load_plugin_shims()
        assert calls == ["called"]

    def test_collects_returned_shims(self, monkeypatch):
        """Entry points that return list[ProviderShim] are collected."""
        from llm_rosetta.shims.provider_shim import ProviderShim

        test_shim = ProviderShim(name="ep-test", base="openai_chat")

        class FakeEP:
            name = "returner"

            def load(self):
                def register():
                    return [test_shim]

                return register

        class FakeEPs:
            def select(self, *, group: str):
                return [FakeEP()]

        monkeypatch.setattr(
            "llm_rosetta.shims.providers.entry_points", lambda: FakeEPs()
        )
        result = _load_plugin_shims()
        assert test_shim in result

    def test_handles_plugin_errors_gracefully(self, monkeypatch):
        """A failing plugin does not crash the loader."""

        class BadEP:
            name = "bad"

            def load(self):
                def register():
                    raise RuntimeError("plugin broken")

                return register

        class FakeEPs:
            def select(self, *, group: str):
                return [BadEP()]

        monkeypatch.setattr(
            "llm_rosetta.shims.providers.entry_points", lambda: FakeEPs()
        )
        result = _load_plugin_shims()
        assert result == []
