"""The ``shims`` section is parsed and validated once (#1330).

A document the config loader accepts must be a document the apply path accepts
too: they used to keep and validate the section differently, so a loader said
"fine" to a shape the apply refused.
"""

from __future__ import annotations

import os

import pytest

from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.shims.shim_sources import (
    ShimSources,
    apply_shim_section,
    apply_shim_sources,
    parse_shims_section,
    shims_signature,
)


class TestParseShimsSection:
    def test_an_absent_section_is_falsy(self):
        sources = parse_shims_section(None)
        assert isinstance(sources, ShimSources)
        assert not sources

    def test_a_declared_section_is_truthy(self):
        assert parse_shims_section({"paths": []})

    def test_paths_and_cache_dir_anchor_to_the_config_directory(self, tmp_path):
        config_path = str(tmp_path / "config.jsonc")
        sources = parse_shims_section(
            {"paths": ["./shims"], "cache_dir": ".cache"}, config_path
        )
        assert os.path.normpath(sources.paths[0]) == str(tmp_path / "shims")
        assert sources.cache_root is not None
        assert os.path.normpath(sources.cache_root) == str(tmp_path / ".cache")

    @pytest.mark.parametrize(
        "key,value",
        [
            ("presets", ["alcf"]),
            ("include", ["argo--chat_completions"]),
            ("exclude", ["argo--chat_completions"]),
            ("remotes", [{"url": "https://github.com/Oaklight/x"}]),
        ],
    )
    def test_a_retired_key_is_refused_by_name(self, key, value):
        with pytest.raises(ValueError, match=rf"'shims\.{key}'.*0\.15\.0"):
            parse_shims_section({key: value})

    def test_a_retired_key_is_refused_even_when_not_strict(self):
        with pytest.raises(ValueError, match="0.15.0"):
            parse_shims_section({"presets": ["alcf"]}, on_unknown="skip")

    def test_a_declared_install_selector_reaches_the_manifest_plan(self):
        sources = parse_shims_section({"install": ["argonne:argo"]})
        (selector,) = sources.manifests.selectors
        assert selector.raw == "argonne:argo"
        # With no ``shims.sources``, the default sources are read.
        assert [s.key for s in sources.manifests.sources] == ["contrib", "argonne"]

    def test_an_unknown_source_is_skipped_unless_strict(self):
        assert not parse_shims_section({"sources": ["ghost"]}).manifests
        with pytest.raises(ValueError):
            parse_shims_section({"sources": ["ghost"], "strict": True})
        with pytest.raises(ValueError):
            parse_shims_section({"sources": ["ghost"]}, on_unknown="raise")
        # An explicit mode wins over the document, in both directions: a
        # strict document can still be read tolerantly.
        tolerant = parse_shims_section(
            {"sources": ["ghost"], "strict": True}, on_unknown="skip"
        )
        assert not tolerant.manifests

    def test_a_non_array_install_value_is_rejected(self):
        with pytest.raises(ValueError, match="must be an array"):
            parse_shims_section({"install": "argonne:argo"})

    def test_disabled_names_are_kept_in_declaration_order(self):
        sources = parse_shims_section({"disabled": ["b--chat_completions", "a"]})
        assert sources.disabled == ("b--chat_completions", "a")


class TestStrictOverride:
    def test_an_explicit_strict_refuses_an_unknown_source(self):
        with pytest.raises(ValueError):
            apply_shim_section({"sources": ["ghost"]}, None, strict=True)

    def test_an_explicit_strict_on_the_document_also_refuses_it(self):
        with pytest.raises(ValueError):
            apply_shim_sources({"shims": {"sources": ["ghost"]}}, None, strict=True)

    def test_an_explicit_non_strict_skips_it(self):
        apply_shim_section({"sources": ["ghost"]}, None, strict=False)

    def test_an_explicit_non_strict_accepts_a_parsed_section(self):
        # Nothing is left to skip: the parse already dropped the entry, and the
        # override must not turn that into an error.
        sources = parse_shims_section({"sources": ["ghost"]})
        report = apply_shim_section(sources, None, strict=False)
        assert report.failures() == []


class TestLoaderAndApplyAgree:
    def test_the_loader_rejects_the_shape_the_apply_rejects(self):
        document = {"shims": {"install": "argonne:argo"}}
        with pytest.raises(ValueError) as applied:
            apply_shim_sources(document, None)
        with pytest.raises(ValueError, match="install.*must be an array") as loaded:
            GatewayConfig(document)
        assert str(loaded.value) == str(applied.value)

    def test_an_unknown_source_does_not_stop_the_loader(self):
        config = GatewayConfig({"shims": {"sources": ["ghost"]}})
        assert not config.shims.manifests

    def test_a_retired_key_stops_the_loader_and_the_apply_alike(self):
        document = {"shims": {"presets": ["alcf"]}}
        with pytest.raises(ValueError, match="0.15.0"):
            apply_shim_sources(document, None)
        with pytest.raises(ValueError, match="0.15.0"):
            GatewayConfig(document)

    def test_the_config_keeps_the_parsed_section(self):
        config = GatewayConfig({"shims": {"paths": ["shims"]}})
        assert isinstance(config.shims, ShimSources)
        assert config.shims.paths == ("shims",)

    def test_the_config_records_the_raw_section_signature(self):
        document = {"shims": {"paths": ["shims"], "strict": True}}
        config = GatewayConfig(document)
        assert config.shims_signature == shims_signature(document)
