"""Tests for declarative external shim sources (local directories)."""

from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

from llm_rosetta.shims import get_shim, list_shims, register_shim
from llm_rosetta_gateway.shims import ShimSourceError, rebuild_shims
from llm_rosetta_gateway.shims import sources as sources_mod
from llm_rosetta.contracts.shim import ProviderShim
from llm_rosetta.shims.provider_shim import _reset_registry

PROVIDER_YAML = (
    "name: {name}\n"
    "format: chat_completions\nmodel_type: llm\n"
    "connection:\n"
    "  base_url: https://{slug}.example/v1\n"
)


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the shim registry before and after each test."""
    _reset_registry()
    sources_mod._source_shims.clear()
    yield
    _reset_registry()
    sources_mod._source_shims.clear()


def _names() -> set[str]:
    return {s.name for s in list_shims()}


def _write_provider(root: Path, *parts: str, name: str | None = None) -> None:
    """Write ``root/<parts>/provider.yaml`` with a valid shim."""
    directory = root.joinpath(*parts)
    directory.mkdir(parents=True, exist_ok=True)
    slug = (name or parts[-1]).split("--")[0]
    (directory / "provider.yaml").write_text(
        PROVIDER_YAML.format(name=name or parts[-1], slug=slug)
    )


class TestLocalDirs:
    def test_flat_and_grouped_layouts(self, tmp_path: Path):
        root = tmp_path / "shims"
        _write_provider(root, "solo", name="solo--chat_completions")
        _write_provider(root, "grp", "chat_completions", name="grp--chat_completions")
        _write_provider(root, "grp", "messages", name="grp--messages")

        report = rebuild_shims(extra_dirs=[root])

        assert report.failures() == []
        assert report.sources[0].shim_count == 3
        assert {
            "solo--chat_completions",
            "grp--chat_completions",
            "grp--messages",
        } <= _names()

    def test_model_typed_layout(self, tmp_path: Path):
        """A declared source may group by model type, like the built-ins."""
        root = tmp_path / "shims"
        _write_provider(
            root, "acme", "llm", "chat_completions", name="acme--chat_completions"
        )
        embedding = root / "acme" / "embeddings"
        embedding.mkdir(parents=True)
        (embedding / "provider.yaml").write_text(
            "name: acme--embedding\n"
            "model_type: embeddings\n"
            "format: openai\n"
            "connection:\n"
            "  base_url: https://acme.example/v1\n"
        )

        report = rebuild_shims(extra_dirs=[root])

        assert report.failures() == []
        assert report.sources[0].shim_count == 2
        assert {"acme--chat_completions", "acme--embedding"} <= _names()

    def test_missing_dir_is_reported_not_raised(self, tmp_path: Path):
        report = rebuild_shims(extra_dirs=[tmp_path / "nope"])
        assert report.total_shims > 0  # built-ins still load
        failures = report.failures()
        assert len(failures) == 1
        assert failures[0].ok is False
        assert "not a directory" in (failures[0].error or "")

    def test_bad_yaml_skipped_others_still_load(self, tmp_path: Path):
        root = tmp_path / "shims"
        _write_provider(root, "good", name="good--chat_completions")
        broken = root / "broken"
        broken.mkdir(parents=True)
        (broken / "provider.yaml").write_text(
            "name: [oops\nformat: chat_completions\nmodel_type: llm\n"
        )

        report = rebuild_shims(extra_dirs=[root])

        assert report.sources[0].ok is True
        assert report.sources[0].shim_names == ("good--chat_completions",)
        assert "good--chat_completions" in _names()

    def test_strict_raises(self, tmp_path: Path):
        with pytest.raises(ShimSourceError):
            rebuild_shims(extra_dirs=[tmp_path / "nope"], strict=True)


class TestRebuild:
    def test_idempotent(self, tmp_path: Path):
        root = tmp_path / "shims"
        _write_provider(root, "acme", name="acme--chat_completions")
        rebuild_shims(extra_dirs=[root])
        first = _names()
        rebuild_shims(extra_dirs=[root])
        assert _names() == first

    def test_removing_source_drops_shims(self, tmp_path: Path):
        root = tmp_path / "shims"
        _write_provider(root, "acme", name="acme--chat_completions")
        rebuild_shims(extra_dirs=[root])
        assert "acme--chat_completions" in _names()

        report = rebuild_shims()
        assert "acme--chat_completions" not in _names()
        assert report.total_shims > 0  # built-ins intact
        assert "chat_completions" in {s.format for s in list_shims()}

    def test_programmatic_without_config(self, tmp_path: Path):
        """rebuild_shims is usable with no config file at all."""
        root = tmp_path / "shims"
        _write_provider(root, "acme", name="acme--chat_completions")
        report = rebuild_shims(extra_dirs=[root])
        assert report.sources[0].ok is True
        assert "acme--chat_completions" in _names()

    def test_report_is_json_safe(self, tmp_path: Path):
        report = rebuild_shims(extra_dirs=[tmp_path / "nope"])
        json.dumps(report.to_dict())  # must not raise


class TestOutOfBandShims:
    """In-process ``register_shim`` calls must survive a rebuild."""

    @staticmethod
    def _register(name: str) -> None:
        register_shim(ProviderShim(name=name, format="chat_completions"))

    def test_survives_a_no_op_rebuild(self):
        """The CLI runs apply_shim_sources() at startup even with no `shims`."""
        self._register("embedder--x")
        report = rebuild_shims()
        assert report.sources == []
        assert get_shim("embedder--x") is not None

    def test_survives_a_rebuild_with_sources(self, tmp_path: Path):
        root = tmp_path / "shims"
        _write_provider(root, "acme", name="acme--chat_completions")
        self._register("embedder--x")

        rebuild_shims(extra_dirs=[root])

        assert get_shim("embedder--x") is not None
        assert get_shim("acme--chat_completions") is not None

    def test_source_shims_are_still_dropped(self, tmp_path: Path):
        root = tmp_path / "shims"
        _write_provider(root, "acme", name="acme--chat_completions")
        self._register("embedder--x")
        rebuild_shims(extra_dirs=[root])

        rebuild_shims()  # source removed

        assert get_shim("acme--chat_completions") is None
        assert get_shim("embedder--x") is not None


def _write_extension(root: Path, ext_id: str, shim: str, *, base_url: str) -> None:
    """An installed extension: ``llm_rosetta_ext/<id>`` under *root*."""
    folder = root / "llm_rosetta_ext" / ext_id
    (folder / "llm").mkdir(parents=True)
    (folder / "plugin.yaml").write_text(
        f"id: {ext_id}\nname: {ext_id}\nversion: 0.1.0\nmaturity: experimental\n"
        "requires: {surface: 1, ir: 1}\n"
        f"provides: {{shims: [{shim}]}}\n"
    )
    (folder / "llm" / "provider.yaml").write_text(
        f"name: {shim}\nformat: chat_completions\nmodel_type: llm\n"
        f"connection:\n  base_url: {base_url}\n"
    )


@pytest.fixture
def _extension_namespace():
    """Forget the cached ``llm_rosetta_ext`` namespace after a test."""
    yield
    for name in [n for n in sys.modules if n.split(".")[0] == "llm_rosetta_ext"]:
        del sys.modules[name]
    importlib.invalidate_caches()


class TestExtensionTier:
    """Installed extensions survive alongside dirs and manifest installs (precedence tier)."""

    def test_extension_shim_and_dir_shim_coexist(
        self, tmp_path: Path, monkeypatch, _extension_namespace
    ):
        site = tmp_path / "site"
        _write_extension(
            site, "plugin", "plugin--x", base_url="https://plugin.example/v1"
        )
        monkeypatch.syspath_prepend(str(site))
        importlib.invalidate_caches()

        root = tmp_path / "shims"
        _write_provider(root, "acme", name="acme--chat_completions")
        rebuild_shims(extra_dirs=[root])

        assert get_shim("plugin--x") is not None
        assert get_shim("acme--chat_completions") is not None

    def test_source_can_override_an_extension_shim(
        self, tmp_path: Path, monkeypatch, _extension_namespace
    ):
        site = tmp_path / "site"
        _write_extension(site, "dup", "dup--x", base_url="https://plugin.example/v1")
        monkeypatch.syspath_prepend(str(site))
        importlib.invalidate_caches()

        root = tmp_path / "shims"
        directory = root / "dup"
        directory.mkdir(parents=True)
        (directory / "provider.yaml").write_text(
            "name: dup--x\n"
            "format: chat_completions\nmodel_type: llm\n"
            "connection:\n"
            "  base_url: https://source.example/v1\n"
        )
        rebuild_shims(extra_dirs=[root])

        shim = get_shim("dup--x")
        assert shim is not None
        assert shim.connection.base_url == "https://source.example/v1"
