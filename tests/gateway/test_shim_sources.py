"""Tests for the gateway <-> shim-source bridge (config resolution + wiring)."""

from __future__ import annotations

import os
from types import SimpleNamespace
from pathlib import Path

import pytest

from llm_rosetta.gateway.admin.routes.config import (
    _shim_status_payload,
    _validate_shim_sources_body,
)
from llm_rosetta.gateway.config import GatewayConfig
from llm_rosetta.gateway.shim_sources import (
    apply_shim_section,
    apply_shim_sources,
    resolve_source_config,
    shims_signature,
)
from llm_rosetta.shims.provider_shim import _reset_registry


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the shim registry before and after each test."""
    _reset_registry()
    yield
    _reset_registry()


def _write_provider(root: Path, slug: str, name: str) -> None:
    directory = root / slug
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "provider.yaml").write_text(
        f"name: {name}\n"
        "base: chat_completions\n"
        "connection:\n"
        f"  base_url: https://{slug}.example/v1\n"
        "  api_key_env: ACME_API_KEY\n"
    )


class TestShimsSignature:
    def test_empty(self):
        assert shims_signature({}) == ""
        assert shims_signature({"shims": {}}) == ""

    def test_key_order_insensitive(self):
        a = shims_signature({"shims": {"paths": ["x"], "strict": True}})
        b = shims_signature({"shims": {"strict": True, "paths": ["x"]}})
        assert a == b

    def test_changes_with_section(self):
        base = shims_signature({"shims": {"paths": ["x"]}})
        assert base != shims_signature({"shims": {"paths": ["y"]}})
        assert base != shims_signature({})


class TestResolveSourceConfig:
    def test_relative_paths_anchor_to_config_dir(self, tmp_path: Path):
        cfg = str(tmp_path / "config.jsonc")
        dirs, remotes, cache = resolve_source_config(
            {"paths": ["./shims", "../shared"]}, cfg
        )
        assert dirs == [
            os.path.join(str(tmp_path), "./shims"),
            os.path.join(str(tmp_path), "../shared"),
        ]
        assert remotes == []
        assert cache == os.path.join(str(tmp_path), "shims-cache")

    def test_absolute_path_preserved(self, tmp_path: Path):
        cfg = str(tmp_path / "config.jsonc")
        dirs, _, _ = resolve_source_config({"paths": ["/abs/shims"]}, cfg)
        assert dirs == ["/abs/shims"]

    def test_cache_dir_override_anchored(self, tmp_path: Path):
        cfg = str(tmp_path / "config.jsonc")
        _, _, cache = resolve_source_config({"cache_dir": "./cc"}, cfg)
        assert cache == os.path.join(str(tmp_path), "./cc")

    def test_no_config_path_uses_home_cache(self):
        _, _, cache = resolve_source_config({"paths": ["x"]}, None)
        assert cache == str(Path("~/.cache/llm-rosetta/shims").expanduser())

    def test_remotes_copied(self, tmp_path: Path):
        spec = {"url": "https://github.com/a/b"}
        _, remotes, _ = resolve_source_config(
            {"remotes": [spec]}, str(tmp_path / "c.jsonc")
        )
        assert remotes == [spec]
        assert remotes[0] is not spec  # defensive copy

    @pytest.mark.parametrize(
        "section",
        [
            {"paths": "not-a-list"},
            {"paths": [123]},
            {"remotes": "nope"},
            {"remotes": ["not-a-dict"]},
            {"cache_dir": 5},
        ],
    )
    def test_bad_shapes_raise(self, section, tmp_path: Path):
        with pytest.raises(ValueError):
            resolve_source_config(section, str(tmp_path / "c.jsonc"))


class TestApplyShimSources:
    def test_empty_section_only_builtins(self):
        report = apply_shim_sources({"server": {}}, None)
        assert report.sources == []
        assert report.total_shims > 0

    def test_provider_resolves_through_declared_source(self, tmp_path: Path):
        shims_dir = tmp_path / "shims"
        _write_provider(shims_dir, "acme", "acme--chat_completions")
        cfg_path = str(tmp_path / "config.jsonc")
        raw = {
            "providers": {"acme": {"shim": "acme--chat_completions", "api_key": "k"}},
            "models": {"m": "acme"},
            "server": {},
            "shims": {"paths": ["./shims"]},
        }

        report = apply_shim_sources(raw, cfg_path)
        assert report.sources[0].ok is True

        config = GatewayConfig(raw)
        assert config.provider_types["acme"] == "chat_completions"
        assert config.providers["acme"].base_url == "https://acme.example/v1"
        assert config.shims == {"paths": ["./shims"]}

    def test_dropping_the_section_drops_the_shims(self, tmp_path: Path):
        shims_dir = tmp_path / "shims"
        _write_provider(shims_dir, "acme", "acme--chat_completions")
        cfg_path = str(tmp_path / "config.jsonc")
        apply_shim_sources({"shims": {"paths": ["./shims"]}}, cfg_path)

        report = apply_shim_sources({"shims": {}}, cfg_path)
        assert report.sources == []
        from llm_rosetta.shims import get_shim

        assert get_shim("acme--chat_completions") is None

    def test_force_refresh_marks_remotes(self, tmp_path: Path, monkeypatch):
        seen: list[dict] = []

        def fake_rebuild(dirs, remotes, **kwargs):
            seen.extend(remotes)
            return SimpleNamespace(
                sources=[], total_shims=0, failures=lambda: [], to_dict=lambda: {}
            )

        import llm_rosetta.gateway.shim_sources as mod

        monkeypatch.setattr(mod, "rebuild_shims", fake_rebuild)
        apply_shim_section(
            {"remotes": [{"url": "https://github.com/a/b", "refresh": "never"}]},
            str(tmp_path / "c.jsonc"),
            force_refresh=True,
        )

        assert seen and seen[0]["refresh"] == "always"


class TestAdminHelpers:
    def test_status_payload_without_report(self):
        request = SimpleNamespace(app=SimpleNamespace(shim_source_report=None))
        payload = _shim_status_payload(request)
        assert payload == {
            "sources": [],
            "total_shims": 0,
            "rebuilt_at": 0.0,
            "failure_count": 0,
        }

    def test_status_payload_with_report(self):
        report = SimpleNamespace(to_dict=lambda: {"sources": [{"ok": True}]})
        request = SimpleNamespace(app=SimpleNamespace(shim_source_report=report))
        assert _shim_status_payload(request) == {"sources": [{"ok": True}]}

    def test_validate_accepts_good_body(self, tmp_path: Path):
        cfg = str(tmp_path / "config.jsonc")
        assert _validate_shim_sources_body({"paths": ["a"], "remotes": []}, cfg) is None

    @pytest.mark.parametrize(
        "body",
        [
            {"paths": "x"},
            {"paths": [123]},  # element type — resolver rejects, not the route
            {"remotes": "x"},
            {"remotes": [{"ref": "v1"}]},
            {"remotes": ["nope"]},
            {"cache_dir": 123},  # type — resolver rejects
            {"remote_ttl": "soon"},  # non-numeric TTL — resolver rejects
        ],
    )
    def test_validate_rejects_bad_body(self, body, tmp_path: Path):
        cfg = str(tmp_path / "config.jsonc")
        assert _validate_shim_sources_body(body, cfg) is not None
