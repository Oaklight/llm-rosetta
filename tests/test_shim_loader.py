"""Tests for the declarative YAML provider shim loader."""

from __future__ import annotations

import json
import os
import textwrap
from pathlib import Path

import pytest

from llm_rosetta.shims.provider_shim import (
    _reset_registry,
    get_shim,
)
from llm_rosetta.shims.providers import (
    _TYPE_DIRS,
    _load_plugin_shims,
    _load_transforms,
    load_providers,
    load_providers_from_dir,
)

#: The registry recorded before the model-type layout landed.  It is the old
#: tree's answer, kept as the expectation the new tree has to reproduce.
SNAPSHOT_PATH = Path(__file__).with_name("shim_registry_snapshot.json")


@pytest.fixture(autouse=True)
def _clean_registry():
    """Give each test in this file a registry of its own.

    The tests here register synthetic shims and assert on the whole registry,
    so they need to start from nothing and leave nothing behind.  What they must
    not leave behind is an *empty* registry — see the fixture below.
    """
    _reset_registry()
    yield
    _reset_registry()


@pytest.fixture(scope="module", autouse=True)
def _restore_builtins_when_done():
    """Put the shipped shims back once this file is finished.

    Every other test file resolves shims through ``get_shim``, so a file
    collected after this one would otherwise see ``None`` for every name and
    fail for a reason that has nothing to do with what it tests — collection
    order is not a property a test should depend on.  Module scope so the
    rebuild happens once, after the last test here rather than after each.
    """
    yield
    load_providers()


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
        expected = {
            # Grouped vendor shims use the canonical ``<provider>--<base>`` form.
            "anthropic--messages",
            # Embedding and rerank shims.
            "cohere--embedding",
            "cohere--rerank",
            "google--embedding",
            "jina--embedding",
            "jina--rerank",
            "openai--embedding",
            "voyage--embedding",
            "voyage--rerank",
            "deepseek--chat_completions",
            "deepseek--open_responses",
            "google--google_generate",
            "google--google_interactions",
            "kilo--chat_completions",
            "minimax--messages",
            "minimax--chat_completions",
            "moonshot",
            "openai--chat_completions",
            "openai--openai_decisions",
            "openai--open_responses",
            "openai_chatgpt--open_responses",
            "openrouter--messages",
            "openrouter--chat_completions",
            "openrouter--open_responses",
            "opencode--chat_completions",
            "opencode--messages",
            "opencode--open_responses",
            "opencode_zen_free--chat_completions",
            "qwen",
            "typesafe--systemone",
            "volcengine--chat_completions",
            "volcengine--open_responses",
            "xai",
            "zhipu",
        }
        assert names == expected, (
            f"Unexpected shim diff: {names.symmetric_difference(expected)}"
        )

    def test_all_registered_after_load(self):
        """After load_providers, all shims are queryable via get_shim."""
        load_providers()
        for name in (
            "openai--chat_completions",
            "openrouter--chat_completions",
            "openrouter--messages",
            "openrouter--open_responses",
            "opencode--chat_completions",
            "opencode--messages",
            "opencode--open_responses",
            "kilo--chat_completions",
            "anthropic--messages",
            "google--google_generate",
            "deepseek--chat_completions",
            "deepseek--open_responses",
            "volcengine--chat_completions",
            "volcengine--open_responses",
            "xai",
            "qwen",
            "moonshot",
            "minimax--chat_completions",
            "minimax--messages",
            "zhipu",
            "google--google_interactions",
            "typesafe--systemone",
            "openai--openai_decisions",
        ):
            shim = get_shim(name)
            assert shim is not None
            assert shim.name == name

        # Legacy spellings are deprecated aliases of the same shim (#917).
        for legacy, canonical in (
            ("openai", "openai--chat_completions"),
            ("anthropic", "anthropic--messages"),
            ("deepseek--chat_completions", "deepseek--chat_completions"),
            ("volcengine--open_responses", "volcengine--open_responses"),
        ):
            assert get_shim(legacy) is get_shim(canonical)

    def test_volcengine_has_transforms(self):
        """Volcengine shim should have strip_fields transforms loaded."""
        load_providers()
        v = get_shim("volcengine--chat_completions")
        assert v is not None
        assert len(v.post_ir_transforms) == 1
        assert len(v.pre_ir_transforms) == 0
        # Verify it strips the right fields
        body = {"logprobs": True, "top_logprobs": 5, "messages": []}
        result = v.post_ir_transforms[0](body)
        assert "logprobs" not in result
        assert "messages" in result

    def test_deepseek_has_transforms(self):
        """DeepSeek chat shim strips n, logit_bias, seed and declares replay."""
        load_providers()
        s = get_shim("deepseek--chat_completions")
        assert s is not None
        assert len(s.post_ir_transforms) == 1
        assert len(s.pre_ir_transforms) == 0
        body = {"n": 2, "logit_bias": {}, "seed": 42, "messages": []}
        result = s.post_ir_transforms[0](body)
        assert "n" not in result
        assert "logit_bias" not in result
        assert "seed" not in result
        assert "messages" in result
        # The reasoning-replay requirement is a capability, not a transform.
        assert s.reasoning is not None
        assert s.reasoning.replay_missing == "fill_empty"

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
        s = get_shim("minimax--chat_completions")
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

    def test_base_types_correct(self):
        """Each shim should have the expected base converter type."""
        load_providers()
        expected = {
            "openai--chat_completions": "chat_completions",
            "openai--open_responses": "open_responses",
            "openai_chatgpt--open_responses": "open_responses",
            "openrouter--chat_completions": "chat_completions",
            "openrouter--messages": "messages",
            "openrouter--open_responses": "open_responses",
            "opencode--chat_completions": "chat_completions",
            "opencode--messages": "messages",
            "opencode--open_responses": "open_responses",
            "opencode_zen_free--chat_completions": "chat_completions",
            "kilo--chat_completions": "chat_completions",
            "anthropic--messages": "messages",
            "google--google_generate": "google_generate",
            "google--google_interactions": "google_interactions",
            "deepseek--chat_completions": "chat_completions",
            "deepseek--open_responses": "open_responses",
            "minimax--chat_completions": "chat_completions",
            "minimax--messages": "messages",
            "moonshot": "chat_completions",
            "qwen": "chat_completions",
            "volcengine--chat_completions": "chat_completions",
            "volcengine--open_responses": "open_responses",
            "xai": "chat_completions",
            "zhipu": "chat_completions",
            "typesafe--systemone": "systemone",
            "openai--openai_decisions": "openai_decisions",
        }
        for name, base in expected.items():
            shim = get_shim(name)
            assert shim is not None, f"Shim {name!r} not found"
            assert shim.base == base, (
                f"{name}: expected base={base!r}, got {shim.base!r}"
            )

    # Shims that intentionally have no public logo
    _LOGO_EXEMPT = {
        "typesafe--systemone",
        "openai--openai_decisions",
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
            "name: nomm\nbase: messages\nmultimodal_tool_result: false\n"
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


class TestDeclaredGrouping:
    """How a shim declares its place in the admin picker (epic #927).

    The picker builds its top level from ``group``, titles it with
    ``group_label`` and labels each shim's own row with ``display_name``.  Each
    falls back to the historical ``vendor--base`` name split, so a shim that
    declares none of them renders exactly as it did before those fields
    existed.

    Grouping is exercised with synthetic shims: the provider that used to
    declare it (ALCF) now ships as a standalone package.
    """

    @staticmethod
    def _write(root: Path, name: str, extra: str = "") -> None:
        d = root / name.split("--")[0]
        d.mkdir()
        (d / "provider.yaml").write_text(
            f"name: {name}\nmodel_type: llm\nformat: chat_completions\n{extra}"
        )

    def test_one_declared_group_spans_several_name_prefixes(self, tmp_path: Path):
        """The cluster is the shim's label; the vendor is the group."""
        root = tmp_path / "providers"
        root.mkdir()
        for cluster in ("metis", "minerva", "sophia"):
            self._write(
                root,
                f"acme_{cluster}--chat_completions",
                f"group: acme\ndisplay_name: {cluster}\n",
            )

        shims = load_providers_from_dir(root)
        assert {s.group for s in shims} == {"acme"}
        assert {s.display_name for s in shims} == {"metis", "minerva", "sophia"}
        # No declared title: the group renders as its key, "acme".
        assert all(s.group_label is None for s in shims)
        # Every cluster speaks the same format, so the picker reads
        # "metis · chat_completions" inside the one group.
        assert {s.base for s in shims} == {"chat_completions"}

    def test_a_title_can_be_declared_without_a_group_key(self, tmp_path: Path):
        root = tmp_path / "providers"
        root.mkdir()
        self._write(root, "acme--chat_completions", "group_label: Acme\n")

        (shim,) = load_providers_from_dir(root)
        assert shim.group is None
        assert shim.group_label == "Acme"
        assert shim.display_name is None

    def test_no_builtin_declares_a_group(self):
        shims = load_providers()
        assert {s.name for s in shims if s.group} == set()

    def test_a_vendor_may_title_its_group(self):
        """A title can be declared without a group key, and it is what renders.

        Kilo titles its group because its key is lower-case; the ChatGPT plan
        profile titles its own because the fallback would be the mangled key
        (``Openai_chatgpt``).  Neither declares ``group`` — the name split
        already puts each shim where it belongs.

        Deliberately not a whitelist: the set of titled shims is not pinned,
        so a new vendor adding one does not have to edit this test.  What is
        pinned is that a title never comes with a ``group`` key it does not
        need, and that these two render exactly as intended.
        """
        shims = load_providers()
        titled = {s.name: s for s in shims if s.group_label}

        assert "openai_chatgpt--open_responses" in titled
        assert all(s.group is None for s in titled.values())
        assert titled["kilo--chat_completions"].group_label == "Kilo"
        assert (
            titled["openai_chatgpt--open_responses"].group_label
            == "OpenAI (ChatGPT plan)"
        )
        # The shim keeps no label of its own, so its row reads as the base.
        assert titled["kilo--chat_completions"].display_name is None
        assert titled["kilo--chat_completions"].group is None

    def test_undeclared_shims_leave_both_fields_none(self):
        """A shim that declares neither renders from its name alone."""
        titled = {"kilo--chat_completions", "openai_chatgpt--open_responses"}
        shims = load_providers()
        assert all(
            s.group is None and s.group_label is None
            for s in shims
            if s.name not in titled
        )


class TestShimPair:
    """A shim declares ``(model_type, format)``; the pair is validated."""

    def _load(self, tmp_path: Path, content: str):
        leaf = tmp_path / "leaf"
        leaf.mkdir()
        (leaf / "provider.yaml").write_text(content)
        return load_providers_from_dir(tmp_path, on_error="raise")

    def test_format_is_canonicalised(self, tmp_path: Path):
        shims = self._load(tmp_path, "name: x\nmodel_type: llm\nformat: openai_chat\n")
        assert shims[0].format == "chat_completions"
        assert shims[0].model_type == "llm"

    def test_format_requires_model_type(self, tmp_path: Path):
        """A new-style shim cannot omit the category.

        Without this, ``format:`` would silently default to ``llm`` and an
        embedding shim that forgot ``model_type`` would load as a chat shim.
        """
        with pytest.raises(ValueError, match="model_type is required"):
            self._load(tmp_path, "name: x\nformat: chat_completions\n")

    def test_legacy_base_is_accepted(self, tmp_path: Path):
        shims = self._load(tmp_path, "name: x\nbase: openai_chat\n")
        assert shims[0].format == "chat_completions"

    def test_non_llm_pair(self, tmp_path: Path):
        shims = self._load(
            tmp_path, "name: x\nmodel_type: embeddings\nformat: openai\n"
        )
        assert shims[0].model_type == "embeddings"
        assert shims[0].format == "openai"

    def test_the_legacy_type_spelling_is_accepted(self, tmp_path: Path):
        """``embedding`` is rewritten to the canonical ``embeddings``."""
        with pytest.warns(DeprecationWarning, match="model_type 'embedding'"):
            shims = self._load(
                tmp_path, "name: x\nmodel_type: embedding\nformat: openai\n"
            )
        assert shims[0].model_type == "embeddings"

    def test_mismatched_pair_is_refused(self, tmp_path: Path):
        with pytest.raises(ValueError, match="not a embeddings format"):
            self._load(
                tmp_path,
                "name: x\nmodel_type: embeddings\nformat: chat_completions\n",
            )

    def test_unknown_model_type_is_refused(self, tmp_path: Path):
        with pytest.raises(ValueError, match="unknown model_type"):
            self._load(
                tmp_path,
                "name: x\nmodel_type: speech\nformat: chat_completions\n",
            )

    def test_legacy_paradigm_base_needs_a_format(self, tmp_path: Path):
        with pytest.raises(ValueError, match="add a 'format'"):
            self._load(tmp_path, "name: x\nbase: decision\n")

    def test_legacy_paradigm_base_with_format_needs_model_type(self, tmp_path: Path):
        """``format`` makes the category mandatory even beside a legacy ``base``.

        A shim mid-migration that keeps ``base`` for one release and forgets
        ``model_type`` must not silently load as ``llm``; only a declaration
        with ``base`` and no ``format`` may omit the category.
        """
        with pytest.raises(ValueError, match="model_type is required"):
            self._load(tmp_path, "name: x\nbase: decision\nformat: systemone\n")

    def test_legacy_paradigm_base_with_format_and_model_type(self, tmp_path: Path):
        shims = self._load(
            tmp_path,
            "name: x\nmodel_type: decision\nbase: decision\nformat: systemone\n",
        )
        assert shims[0].model_type == "decision"
        assert shims[0].format == "systemone"

    def test_every_builtin_shim_declares_a_valid_pair(self):
        """The totality check behind "every shim speaks the pair".

        A per-file test cannot catch the shim that was left out of the sweep;
        iterating the registry can.  The registry's import-time check already
        pins the converter maps to the vocabulary, so membership in
        ``FORMATS_BY_TYPE[model_type]`` is what a load has to satisfy.
        """
        from llm_rosetta.model_vocabulary import FORMATS_BY_TYPE
        from llm_rosetta.shims import list_shims

        load_providers()
        shims = list_shims()
        assert shims, "the built-in shims did not register"
        for shim in shims:
            assert shim.format, shim.name
            assert shim.model_type in FORMATS_BY_TYPE, shim.name
            assert shim.format in FORMATS_BY_TYPE[shim.model_type], shim.name


class TestConnectionTokenCommand:
    """A shim can declare how it turns a stored credential into the upstream one.

    The exchange is a vendor fact (AskSage mints an access token from the
    account key, ALCF refreshes a Globus bearer token), so a shim that ships it
    is usable without the operator wiring ``token_command`` by hand.
    """

    def _load(self, tmp_path: Path, connection: str, extra: dict | None = None):
        d = tmp_path / "vendor"
        d.mkdir(exist_ok=True)
        for filename, content in (extra or {}).items():
            (d / filename).write_text(content)
        (d / "provider.yaml").write_text(
            f"name: vendor\nbase: openai_chat\nconnection:\n{connection}"
        )
        shims = load_providers_from_dir(tmp_path)
        assert len(shims) == 1
        return shims[0]

    def test_a_shipped_script_resolves_against_the_shim_directory(self, tmp_path):
        shim = self._load(
            tmp_path,
            "  base_url: https://vendor.example/v1\n"
            "  token_command: [python3, token.py]\n",
            extra={"token.py": "print('tok')\n"},
        )

        assert shim.connection.token_command == (
            "python3",
            str((tmp_path / "vendor" / "token.py").resolve()),
        )

    def test_a_bare_name_that_is_not_shipped_stays_a_path_lookup(self, tmp_path):
        shim = self._load(
            tmp_path,
            "  base_url: https://vendor.example/v1\n"
            "  token_command: [token-helper, --flag]\n",
        )

        assert shim.connection.token_command == ("token-helper", "--flag")

    def test_an_absolute_first_argument_is_left_alone(self, tmp_path):
        shim = self._load(
            tmp_path,
            "  base_url: https://vendor.example/v1\n"
            "  token_command: [/opt/bin/token, --flag]\n",
        )

        assert shim.connection.token_command == ("/opt/bin/token", "--flag")

    @pytest.mark.parametrize("declaration", ["not-a-list", "[]", "[1, 2]", "['']"])
    def test_a_malformed_declaration_is_ignored(self, tmp_path, declaration):
        """The shim's other facts stay usable; a broken command is not fatal."""
        shim = self._load(
            tmp_path,
            f"  base_url: https://vendor.example/v1\n  token_command: {declaration}\n",
        )

        assert shim.connection.token_command is None

    def test_the_refresh_interval_is_parsed(self, tmp_path):
        shim = self._load(
            tmp_path,
            "  base_url: https://vendor.example/v1\n"
            "  token_command: [token-helper]\n"
            "  token_refresh_interval: 7200\n",
        )

        assert shim.connection.token_refresh_interval == 7200

    def test_a_shim_without_a_command_declares_nothing(self, tmp_path):
        shim = self._load(tmp_path, "  base_url: https://vendor.example/v1\n")

        assert shim.connection.token_command is None
        assert shim.connection.token_refresh_interval is None


class TestAuthExchange:
    """A shim can ship its credential exchange as code, not as an argv.

    An OAuth profile does not fit in ``connection.token_command``: it has an
    interactive first run, a rotating refresh token and a stored record.  The
    module form is the same declaration, so a shim ships one or the other.
    """

    def _write(self, tmp_path: Path, yaml_body: str, auth: str | None = None):
        d = tmp_path / "vendor"
        d.mkdir(exist_ok=True)
        if auth is not None:
            (d / "auth.py").write_text(textwrap.dedent(auth))
        (d / "provider.yaml").write_text(yaml_body)

    def test_a_shipped_exchange_is_attached(self, tmp_path):
        self._write(
            tmp_path,
            "name: vendor\nbase: openai_chat\n"
            "connection:\n  base_url: https://vendor.example/v1\n"
            "  token_refresh_interval: 1800\n",
            """
            from llm_rosetta.shims.auth_exchange import BaseAuthExchange

            class Exchange(BaseAuthExchange):
                kind = "oauth"

                async def credential(self, ctx):
                    return "access-token"

            EXCHANGE = Exchange()
            """,
        )

        shims = load_providers_from_dir(tmp_path)

        assert len(shims) == 1
        assert shims[0].auth is not None
        assert shims[0].auth.kind == "oauth"
        assert shims[0].connection.token_refresh_interval == 1800

    def test_a_shim_without_a_module_attaches_nothing(self, tmp_path):
        self._write(
            tmp_path,
            "name: vendor\nbase: openai_chat\n"
            "connection:\n  base_url: https://vendor.example/v1\n",
        )

        assert load_providers_from_dir(tmp_path)[0].auth is None

    def test_a_module_without_an_exchange_attaches_nothing(self, tmp_path):
        self._write(
            tmp_path,
            "name: vendor\nbase: openai_chat\n"
            "connection:\n  base_url: https://vendor.example/v1\n",
            "HELPERS = ()\n",
        )

        assert load_providers_from_dir(tmp_path)[0].auth is None

    def test_an_exchange_without_credential_is_rejected(self, tmp_path):
        self._write(
            tmp_path,
            "name: vendor\nbase: openai_chat\n"
            "connection:\n  base_url: https://vendor.example/v1\n",
            "EXCHANGE = object()\n",
        )

        with pytest.raises(ValueError, match="must implement credential"):
            load_providers_from_dir(tmp_path)

    def test_both_forms_at_once_is_rejected(self, tmp_path):
        """One slot, one declaration: the loader refuses to pick a winner."""
        self._write(
            tmp_path,
            "name: vendor\nbase: openai_chat\n"
            "connection:\n  base_url: https://vendor.example/v1\n"
            "  token_command: [token-helper]\n",
            """
            class Exchange:
                kind = "oauth"

                async def credential(self, ctx):
                    return "access-token"

            EXCHANGE = Exchange()
            """,
        )

        with pytest.raises(
            ValueError, match="both auth.py and connection.token_command"
        ):
            load_providers_from_dir(tmp_path)

    def test_a_broken_module_is_not_fatal_for_a_tolerant_scan(self, tmp_path):
        """An external source degrades to a warning, like a broken transforms.py."""
        self._write(
            tmp_path,
            "name: vendor\nbase: openai_chat\n"
            "connection:\n  base_url: https://vendor.example/v1\n",
            "raise RuntimeError('boom')\n",
        )

        assert load_providers_from_dir(tmp_path, on_error="warn") == []


def _snapshot_rows() -> list[dict]:
    """The registry facts a directory move must not change."""
    from llm_rosetta.shims import list_shims

    load_providers()
    return [
        {
            "name": s.name,
            "model_type": s.model_type,
            "format": s.format,
            "base_url": s.connection.base_url,
            "api_key_env": s.connection.api_key_env,
            "default_path": s.connection.default_path,
            "group": s.group,
            "group_label": s.group_label,
            "display_name": s.display_name,
            "free_source": s.free_source,
        }
        for s in sorted(list_shims(), key=lambda s: s.name)
    ]


class TestBuiltinRegistrySnapshot:
    """Where a built-in shim sits must not change what the registry says.

    The snapshot was recorded from the tree before the model-type layout
    landed, so the expectation is the *old* tree's registry and this tree has
    to reproduce it.  Regenerate it deliberately, never to make a red test
    green:

        SHIM_SNAPSHOT_REGEN=1 pytest \
            tests/test_shim_loader.py::TestBuiltinRegistrySnapshot
    """

    def test_the_registry_matches_the_committed_snapshot(self):
        actual = _snapshot_rows()
        if os.environ.get("SHIM_SNAPSHOT_REGEN"):
            SNAPSHOT_PATH.write_text(
                json.dumps(actual, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            pytest.skip("snapshot regenerated")
        expected = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))
        assert actual == expected


class TestModelTypeDirectories:
    """A vendor groups its shims by model type, and the path is checked.

    The type directory is named after the endpoint the type serves
    (``embeddings``, ``rerank``, ``decisions``), not after the registry key,
    so the loader keeps an explicit mapping.  Chat keeps the format level
    because a vendor serves several chat dialects; a non-chat vendor normally
    serves one, so the type directory is the leaf.
    """

    @staticmethod
    def _write(root: Path, rel: str, name: str, model_type: str, fmt: str) -> Path:
        d = root / rel
        d.mkdir(parents=True)
        (d / "provider.yaml").write_text(
            f"name: {name}\nmodel_type: {model_type}\nformat: {fmt}\n",
            encoding="utf-8",
        )
        return d

    def test_the_type_directory_can_hold_the_leaf(self, tmp_path: Path):
        self._write(
            tmp_path, "cohere/embeddings", "cohere--embedding", "embeddings", "cohere"
        )
        (shim,) = load_providers_from_dir(tmp_path)
        assert shim.model_type == "embeddings"
        assert shim.format == "cohere"

    def test_the_type_directory_can_group_format_leaves(self, tmp_path: Path):
        self._write(
            tmp_path,
            "openai/llm/chat_completions",
            "openai--chat_completions",
            "llm",
            "chat_completions",
        )
        self._write(
            tmp_path,
            "openai/llm/open_responses",
            "openai--open_responses",
            "llm",
            "open_responses",
        )
        shims = load_providers_from_dir(tmp_path)
        assert {s.name for s in shims} == {
            "openai--chat_completions",
            "openai--open_responses",
        }

    def test_a_mismatch_names_the_directory_and_the_declaration(self, tmp_path: Path):
        self._write(
            tmp_path, "vendor/embeddings", "vendor--x", "llm", "chat_completions"
        )
        with pytest.raises(ValueError, match=r"'embeddings'.*model_type='llm'"):
            load_providers_from_dir(tmp_path)

    def test_a_legacy_format_directory_still_loads(self, tmp_path: Path):
        """``embedding/`` is a format directory, not the ``embeddings`` level."""
        self._write(
            tmp_path, "vendor/embedding", "vendor--embedding", "embeddings", "openai"
        )
        (shim,) = load_providers_from_dir(tmp_path)
        assert shim.name == "vendor--embedding"

    def test_a_legacy_group_still_loads(self, tmp_path: Path):
        self._write(
            tmp_path,
            "vendor/chat_completions",
            "vendor--chat_completions",
            "llm",
            "chat_completions",
        )
        (shim,) = load_providers_from_dir(tmp_path)
        assert shim.format == "chat_completions"

    def test_every_builtin_sits_under_its_model_type(self):
        import llm_rosetta.shims.providers as providers_mod
        from llm_rosetta._vendor.yaml import load as yaml_load

        root = Path(providers_mod.__file__).parent
        paths = sorted(root.glob("*/*/provider.yaml")) + sorted(
            root.glob("*/*/*/provider.yaml")
        )
        assert paths
        for path in paths:
            rel = path.relative_to(root).parts
            assert rel[1] in _TYPE_DIRS, path
            cfg = yaml_load(path.read_text(encoding="utf-8"))
            assert _TYPE_DIRS[rel[1]] == cfg["model_type"], path
