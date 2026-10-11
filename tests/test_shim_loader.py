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
        from_t, to_t, ir_t, resp_t, mod = _load_transforms(tmp_path)
        assert from_t == ()
        assert to_t == ()
        assert mod is None

    def test_transforms_with_to_only(self, tmp_path: Path):
        """Loads post_ir_transforms from transforms.py."""
        tf = tmp_path / "transforms.py"
        tf.write_text(
            textwrap.dedent("""\
            from llm_rosetta.transforms import strip_fields
            post_ir_transforms = (strip_fields("foo"),)
        """)
        )
        from_t, to_t, ir_t, resp_t, mod = _load_transforms(tmp_path)
        assert from_t == ()
        assert len(to_t) == 1
        assert mod is not None
        # Verify the transform works
        body = {"foo": 1, "bar": 2}
        result = to_t[0](body)
        assert "foo" not in result
        assert result["bar"] == 2

    def test_transforms_with_both(self, tmp_path: Path):
        """Loads both pre_ir_transforms and post_ir_transforms."""
        tf = tmp_path / "transforms.py"
        tf.write_text(
            textwrap.dedent("""\
            from llm_rosetta.transforms import strip_fields, rename_field
            post_ir_transforms = (strip_fields("x"),)
            pre_ir_transforms = (rename_field("a", "b"),)
        """)
        )
        from_t, to_t, ir_t, resp_t, mod = _load_transforms(tmp_path)
        assert len(from_t) == 1
        assert len(to_t) == 1
        assert mod is not None


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
        """Verify the real providers/ directory loads all built-in shims."""
        shims = load_providers()
        names = {s.name for s in shims}
        assert names == {
            "argo--anthropic",
            "argo--openai_chat",
            "argo--openai_responses",
            "asksage--openai_chat",
            "asksage--openai_responses",
            "asksage--anthropic",
            "asksage--google_generate",
            "openai",
            "openai_decisions",
            "openai_responses",
            "open_responses",
            "openrouter--openai_chat",
            "openrouter--anthropic",
            "kilo--openai_chat",
            "anthropic",
            "google_generate",
            "deepseek--openai_chat",
            "deepseek--openai_responses",
            "minimax--openai_chat",
            "minimax--anthropic",
            "moonshot",
            "qwen",
            "volcengine--openai_chat",
            "volcengine--openai_responses",
            "xai",
            "zhipu",
            "google_interactions",
            "alcf--sophia",
            "alcf--metis",
            "alcf--minerva",
            "typesafe",
        }, (
            f"Unexpected shim diff: {names.symmetric_difference({'argo--anthropic', 'argo--openai_chat', 'argo--openai_responses', 'asksage--openai_chat', 'asksage--openai_responses', 'asksage--anthropic', 'asksage--google_generate', 'openai', 'openai_decisions', 'openai_responses', 'open_responses', 'openrouter--openai_chat', 'openrouter--anthropic', 'kilo--openai_chat', 'anthropic', 'google_generate', 'deepseek--openai_chat', 'deepseek--openai_responses', 'minimax--openai_chat', 'minimax--anthropic', 'moonshot', 'qwen', 'volcengine--openai_chat', 'volcengine--openai_responses', 'xai', 'zhipu', 'google_interactions', 'alcf--sophia', 'alcf--metis', 'alcf--minerva', 'typesafe'})}"
        )

    def test_all_registered_after_load(self):
        """After load_providers, all shims are queryable via get_shim."""
        load_providers()
        for name in (
            "openai",
            "openrouter--openai_chat",
            "openrouter--anthropic",
            "kilo--openai_chat",
            "anthropic",
            "google_generate",
            "deepseek--openai_chat",
            "deepseek--openai_responses",
            "volcengine--openai_chat",
            "volcengine--openai_responses",
            "xai",
            "qwen",
            "moonshot",
            "minimax--openai_chat",
            "minimax--anthropic",
            "zhipu",
            "google_interactions",
            "alcf--sophia",
            "alcf--metis",
            "alcf--minerva",
            "typesafe",
            "asksage--openai_chat",
            "asksage--openai_responses",
            "asksage--anthropic",
            "asksage--google_generate",
        ):
            shim = get_shim(name)
            assert shim is not None
            assert shim.name == name

    def test_volcengine_has_transforms(self):
        """Volcengine shim should have strip_fields transforms loaded."""
        load_providers()
        v = get_shim("volcengine--openai_chat")
        assert v is not None
        assert len(v.post_ir_transforms) == 1
        assert len(v.pre_ir_transforms) == 0
        # Verify it strips the right fields
        body = {"logprobs": True, "top_logprobs": 5, "messages": []}
        result = v.post_ir_transforms[0](body)
        assert "logprobs" not in result
        assert "messages" in result

    def test_deepseek_has_transforms(self):
        """DeepSeek chat shim should strip n, logit_bias, seed."""
        load_providers()
        s = get_shim("deepseek--openai_chat")
        assert s is not None
        assert len(s.post_ir_transforms) == 1
        assert len(s.pre_ir_transforms) == 0
        body = {"n": 2, "logit_bias": {}, "seed": 42, "messages": []}
        result = s.post_ir_transforms[0](body)
        assert "n" not in result
        assert "logit_bias" not in result
        assert "seed" not in result
        assert "messages" in result

    def test_xai_has_transforms(self):
        """xAI shim should strip logit_bias."""
        load_providers()
        s = get_shim("xai")
        assert s is not None
        assert len(s.post_ir_transforms) == 1
        assert len(s.pre_ir_transforms) == 0
        body = {"logit_bias": {"50256": -100}, "messages": []}
        result = s.post_ir_transforms[0](body)
        assert "logit_bias" not in result
        assert "messages" in result

    def test_moonshot_has_transforms(self):
        """Moonshot shim should strip logprobs, top_logprobs, logit_bias, seed."""
        load_providers()
        s = get_shim("moonshot")
        assert s is not None
        assert len(s.post_ir_transforms) == 1
        assert len(s.pre_ir_transforms) == 0
        body = {
            "logprobs": True,
            "top_logprobs": 5,
            "logit_bias": {},
            "seed": 123,
            "messages": [],
        }
        result = s.post_ir_transforms[0](body)
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
        assert len(s.post_ir_transforms) == 1
        assert len(s.pre_ir_transforms) == 0
        body = {"frequency_penalty": 0.5, "logit_bias": {}, "messages": []}
        result = s.post_ir_transforms[0](body)
        assert "frequency_penalty" not in result
        assert "logit_bias" not in result
        assert "messages" in result

    def test_minimax_has_transforms(self):
        """MiniMax shim should strip fields + inject reasoning_split."""
        load_providers()
        s = get_shim("minimax--openai_chat")
        assert s is not None
        assert len(s.post_ir_transforms) == 2  # strip_fields + inject_reasoning_split
        assert len(s.pre_ir_transforms) == 1  # parse_think_tags
        body = {
            "logprobs": True,
            "top_logprobs": 5,
            "seed": 42,
            "stop": ["\n"],
            "messages": [],
        }
        result = s.post_ir_transforms[0](body)
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
        assert len(s.post_ir_transforms) == 1
        assert len(s.pre_ir_transforms) == 0
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
        result = s.post_ir_transforms[0](body)
        assert "n" not in result
        assert "presence_penalty" not in result
        assert "frequency_penalty" not in result
        assert "logprobs" not in result
        assert "top_logprobs" not in result
        assert "logit_bias" not in result
        assert "seed" not in result
        assert "messages" in result

    def test_asksage_openai_chat_has_transforms(self):
        """AskSage OpenAI Chat shim should rename max_tokens to max_completion_tokens."""
        load_providers()
        s = get_shim("asksage--openai_chat")
        assert s is not None
        assert len(s.post_ir_transforms) == 1
        assert len(s.pre_ir_transforms) == 0
        body = {"max_tokens": 100, "messages": []}
        result = s.post_ir_transforms[0](body)
        assert "max_tokens" not in result
        assert result["max_completion_tokens"] == 100
        assert result["messages"] == []

    def test_asksage_openai_responses_has_transforms(self):
        """AskSage OpenAI Responses shim should rename max_tokens."""
        load_providers()
        s = get_shim("asksage--openai_responses")
        assert s is not None
        assert len(s.post_ir_transforms) == 1
        body = {"max_tokens": 50, "input": "hello"}
        result = s.post_ir_transforms[0](body)
        assert "max_tokens" not in result
        assert result["max_completion_tokens"] == 50

    def test_asksage_anthropic_no_transforms(self):
        """AskSage Anthropic shim should have no transforms."""
        load_providers()
        s = get_shim("asksage--anthropic")
        assert s is not None
        assert len(s.post_ir_transforms) == 0
        assert len(s.pre_ir_transforms) == 0

    def test_asksage_google_generate_auth_header(self):
        """AskSage Google Generate shim should use x-access-tokens auth header."""
        load_providers()
        s = get_shim("asksage--google_generate")
        assert s is not None
        assert s.connection.auth_header == "x-access-tokens"

    def test_base_types_correct(self):
        """Each shim should have the expected base converter type."""
        load_providers()
        expected = {
            "openai": "openai_chat",
            "openai_responses": "openai_responses",
            "openrouter--openai_chat": "openai_chat",
            "openrouter--anthropic": "anthropic",
            "kilo--openai_chat": "openai_chat",
            "anthropic": "anthropic",
            "google_generate": "google_generate",
            "google_interactions": "google_interactions",
            "deepseek--openai_chat": "openai_chat",
            "deepseek--openai_responses": "openai_responses",
            "argo--openai_responses": "openai_responses",
            "minimax--openai_chat": "openai_chat",
            "minimax--anthropic": "anthropic",
            "moonshot": "openai_chat",
            "qwen": "openai_chat",
            "volcengine--openai_chat": "openai_chat",
            "volcengine--openai_responses": "openai_responses",
            "xai": "openai_chat",
            "zhipu": "openai_chat",
            "alcf--sophia": "openai_chat",
            "alcf--metis": "openai_chat",
            "alcf--minerva": "openai_chat",
            "typesafe": "decision",
            "asksage--openai_chat": "openai_chat",
            "asksage--openai_responses": "openai_responses",
            "asksage--anthropic": "anthropic",
            "asksage--google_generate": "google_generate",
        }
        for name, base in expected.items():
            shim = get_shim(name)
            assert shim is not None, f"Shim {name!r} not found"
            assert shim.base == base, (
                f"{name}: expected base={base!r}, got {shim.base!r}"
            )

    # Shims that intentionally have no public logo
    _LOGO_EXEMPT = {
        "argo--anthropic",
        "argo--openai_chat",
        "argo--openai_responses",
        "asksage--openai_chat",
        "asksage--openai_responses",
        "asksage--anthropic",
        "asksage--google_generate",
        "typesafe",
        "openai_decisions",
    }

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

    def test_multimodal_tool_result_from_yaml(self, tmp_path: Path):
        """multimodal_tool_result is parsed from provider YAML."""
        d = tmp_path / "multimodal_provider"
        d.mkdir()
        (d / "provider.yaml").write_text(
            "name: multimodal_provider\nbase: openai_chat\nmultimodal_tool_result: true\n"
        )
        shims = load_providers_from_dir(tmp_path)
        assert len(shims) == 1
        assert shims[0].multimodal_tool_result is True

    def test_multimodal_tool_result_false_from_yaml(self, tmp_path: Path):
        """multimodal_tool_result: false is parsed correctly."""
        d = tmp_path / "nomm"
        d.mkdir()
        (d / "provider.yaml").write_text(
            "name: nomm\nbase: anthropic\nmultimodal_tool_result: false\n"
        )
        shims = load_providers_from_dir(tmp_path)
        assert len(shims) == 1
        assert shims[0].multimodal_tool_result is False

    def test_multimodal_tool_result_absent_from_yaml(self, tmp_path: Path):
        """Missing multimodal_tool_result defaults to None."""
        d = tmp_path / "plain"
        d.mkdir()
        (d / "provider.yaml").write_text("name: plain\nbase: openai_chat\n")
        shims = load_providers_from_dir(tmp_path)
        assert len(shims) == 1
        assert shims[0].multimodal_tool_result is None


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
            "from llm_rosetta.transforms import strip_fields\n"
            'post_ir_transforms = (strip_fields("foo"),)\n'
        )
        shims = load_providers_from_dir(tmp_path)
        s = [s for s in shims if s.name == "myplugin"][0]
        assert len(s.post_ir_transforms) == 1
        # Verify the transform works
        body = {"foo": 1, "bar": 2}
        result = s.post_ir_transforms[0](body)
        assert "foo" not in result
        assert result["bar"] == 2

    def test_plugin_transforms_relative_import(self, tmp_path: Path):
        """A plugin transforms.py can import a sibling helper via `from .x`."""
        d = tmp_path / "relimp"
        d.mkdir()
        (d / "provider.yaml").write_text("name: relimp\nbase: openai_chat\n")
        (d / "helpers.py").write_text("def mark():\n    return 'sentinel'\n")
        (d / "transforms.py").write_text(
            "from .helpers import mark\n"
            "from llm_rosetta.transforms import strip_fields\n"
            "post_ir_transforms = (strip_fields(mark()),)\n"
        )
        shims = load_providers_from_dir(tmp_path)
        s = [s for s in shims if s.name == "relimp"][0]
        assert len(s.post_ir_transforms) == 1
        assert s.post_ir_transforms[0]({"sentinel": 1, "keep": 2}) == {"keep": 2}

    def test_plugin_grouped_relative_import(self, tmp_path: Path):
        """Grouped plugin layout also supports sibling imports."""
        leaf = tmp_path / "grp" / "leaf"
        leaf.mkdir(parents=True)
        (leaf / "provider.yaml").write_text("name: grp--leaf\nbase: openai_chat\n")
        (leaf / "helper.py").write_text("NAME = 'gone'\n")
        (leaf / "transforms.py").write_text(
            "from .helper import NAME\n"
            "from llm_rosetta.transforms import strip_fields\n"
            "post_ir_transforms = (strip_fields(NAME),)\n"
        )
        shims = load_providers_from_dir(tmp_path)
        s = [s for s in shims if s.name == "grp--leaf"][0]
        assert s.post_ir_transforms[0]({"gone": 1, "keep": 2}) == {"keep": 2}

    def test_distinct_roots_do_not_collide(self, tmp_path: Path):
        """Two roots with the same group/leaf names get independent modules."""
        import sys

        for root, marker in ((tmp_path / "a", "a"), (tmp_path / "b", "b")):
            leaf = root / "grp" / "leaf"
            leaf.mkdir(parents=True)
            (leaf / "provider.yaml").write_text("name: grp--leaf\nbase: openai_chat\n")
            (leaf / "tag.py").write_text(f"MARK = {marker!r}\n")
            (leaf / "transforms.py").write_text(
                "from .tag import MARK\nLOADED = MARK\n"
            )
            load_providers_from_dir(root)

        leaves = [
            m
            for name, m in sys.modules.items()
            if name.startswith("_llm_rosetta_plugin_shims")
            and name.endswith("grp.leaf.transforms")
        ]
        # Two distinct module objects, each resolving its own root's MARK — not
        # one shared module reached under two namespace prefixes.
        assert len(leaves) == 2
        assert len({id(m) for m in leaves}) == 2
        assert {m.LOADED for m in leaves} == {"a", "b"}

    def test_failed_plugin_transform_rolls_back(self, tmp_path: Path):
        """A transforms.py that raises leaves nothing half-registered —
        including a sibling module it imported before raising."""
        import sys

        d = tmp_path / "bad"
        d.mkdir()
        (d / "provider.yaml").write_text("name: bad\nbase: openai_chat\n")
        (d / "leaky.py").write_text("VALUE = 1\n")
        (d / "transforms.py").write_text(
            "from . import leaky\nraise RuntimeError('boom')\n"
        )
        before = {m for m in sys.modules if m.startswith("_llm_rosetta_plugin_shims")}

        with pytest.raises(RuntimeError, match="boom"):
            load_providers_from_dir(tmp_path)

        after = {m for m in sys.modules if m.startswith("_llm_rosetta_plugin_shims")}
        assert after == before

    def test_failed_load_does_not_poison_later_loads(self, tmp_path: Path):
        """A failed plugin load must not make the next load raise KeyError —
        and must not evict an already-loaded shim's modules."""
        import sys

        good = tmp_path / "aaa_good"
        good.mkdir()
        (good / "provider.yaml").write_text("name: aaa_good\nbase: openai_chat\n")
        (good / "transforms.py").write_text("post_ir_transforms = ()\n")
        load_providers_from_dir(tmp_path)  # namespace + aaa_good.transforms now exist

        bad = tmp_path / "zzz_bad"
        bad.mkdir()
        (bad / "provider.yaml").write_text("name: zzz_bad\nbase: openai_chat\n")
        (bad / "transforms.py").write_text("raise RuntimeError('boom')\n")
        with pytest.raises(RuntimeError, match="boom"):
            load_providers_from_dir(tmp_path)

        # The earlier good shim's modules survive the failed load.
        assert any(n.endswith("aaa_good.transforms") for n in sys.modules)

        # The shim's own error again, not a KeyError from stale bookkeeping.
        with pytest.raises(RuntimeError, match="boom"):
            load_providers_from_dir(tmp_path)

        (bad / "transforms.py").write_text("post_ir_transforms = ()\n")
        shims = load_providers_from_dir(tmp_path)
        assert {s.name for s in shims} == {"aaa_good", "zzz_bad"}

    def test_plugin_namespaces_cleared_on_reset(self, tmp_path: Path):
        """_reset_registry drops the synthetic plugin packages from sys.modules."""
        import sys

        from llm_rosetta.shims.provider_shim import _reset_registry
        from llm_rosetta.shims.providers import load_providers

        d = tmp_path / "resettest"
        d.mkdir()
        (d / "provider.yaml").write_text("name: resettest\nbase: openai_chat\n")
        (d / "helpers.py").write_text("VALUE = 1\n")
        (d / "transforms.py").write_text("from .helpers import VALUE\n")
        load_providers_from_dir(tmp_path)
        assert any(m.startswith("_llm_rosetta_plugin_shims") for m in sys.modules)

        _reset_registry()
        assert not any(m.startswith("_llm_rosetta_plugin_shims") for m in sys.modules)
        load_providers()  # restore built-ins for the remaining tests


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
