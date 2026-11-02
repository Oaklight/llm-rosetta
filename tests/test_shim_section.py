"""The ``shims`` section is parsed and validated once (#1330).

A document the config loader accepts must be a document the apply path accepts
too: they used to keep and validate the section differently, so a loader said
"fine" to a shape the apply refused.
"""

from __future__ import annotations

import os

import pytest

from llm_rosetta_gateway.config import GatewayConfig
from llm_rosetta_gateway.shim_sources import (
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

    def test_a_preset_expands_to_its_catalogue_remote(self):
        sources = parse_shims_section({"presets": ["argo"]})
        assert sources.presets == ("argo",)
        assert sources.remotes
        assert sources.remotes[0]["url"]

    def test_an_unknown_preset_is_skipped_unless_strict(self):
        assert parse_shims_section({"presets": ["ghost"]}).remotes == ()
        with pytest.raises(ValueError):
            parse_shims_section({"presets": ["ghost"], "strict": True})
        with pytest.raises(ValueError):
            parse_shims_section({"presets": ["ghost"]}, on_unknown="raise")
        # An explicit mode wins over the document, in both directions: a
        # strict document can still be read tolerantly.
        tolerant = parse_shims_section(
            {"presets": ["ghost"], "strict": True}, on_unknown="skip"
        )
        assert tolerant.remotes == ()

    def test_a_non_array_presets_value_is_rejected(self):
        with pytest.raises(ValueError, match="must be an array"):
            parse_shims_section({"presets": "argo"})

    def test_disabled_names_are_kept_in_declaration_order(self):
        sources = parse_shims_section({"disabled": ["b--chat_completions", "a"]})
        assert sources.disabled == ("b--chat_completions", "a")


class TestStrictOverride:
    def test_an_explicit_strict_refuses_an_unknown_preset(self):
        with pytest.raises(ValueError, match="unknown shim preset"):
            apply_shim_section({"presets": ["ghost"]}, None, strict=True)

    def test_an_explicit_strict_on_the_document_also_refuses_it(self):
        with pytest.raises(ValueError, match="unknown shim preset"):
            apply_shim_sources({"shims": {"presets": ["ghost"]}}, None, strict=True)

    def test_an_explicit_non_strict_skips_it(self):
        apply_shim_section({"presets": ["ghost"]}, None, strict=False)

    def test_a_parsed_section_cannot_contradict_an_explicit_strict(self):
        # The loader read the document tolerantly and dropped the key; asking
        # for strict here must not report success for a rule never applied.
        sources = parse_shims_section({"presets": ["ghost"]})
        with pytest.raises(ValueError, match="unknown shim preset"):
            apply_shim_section(sources, None, strict=True)

    def test_an_explicit_non_strict_accepts_a_parsed_section(self):
        # Nothing is left to skip: the parse already dropped the key, and the
        # override must not turn that into an error.
        sources = parse_shims_section({"presets": ["ghost"]})
        report = apply_shim_section(sources, None, strict=False)
        assert report.failures() == []


class TestLoaderAndApplyAgree:
    def test_the_loader_rejects_the_shape_the_apply_rejects(self):
        document = {"shims": {"presets": "argo"}}
        with pytest.raises(ValueError) as applied:
            apply_shim_sources(document, None)
        with pytest.raises(ValueError, match="presets.*must be an array") as loaded:
            GatewayConfig(document)
        assert str(loaded.value) == str(applied.value)

    def test_an_unknown_preset_does_not_stop_the_loader(self):
        config = GatewayConfig({"shims": {"presets": ["ghost"]}})
        assert config.shims.remotes == ()

    def test_the_config_keeps_the_parsed_section(self):
        config = GatewayConfig({"shims": {"paths": ["shims"]}})
        assert isinstance(config.shims, ShimSources)
        assert config.shims.paths == ("shims",)

    def test_the_config_records_the_raw_section_signature(self):
        document = {"shims": {"paths": ["shims"], "strict": True}}
        config = GatewayConfig(document)
        assert config.shims_signature == shims_signature(document)
