"""The functions the host-facing surface gained in #1630 PR 6."""

from __future__ import annotations

import pytest

import llm_rosetta.api as api
from llm_rosetta.shims import builtin as providers
from llm_rosetta.shims import ProviderShim, ReasoningCapability
from llm_rosetta.shims.provider_shim import get_shim, register_shim
from llm_rosetta.shims._loader import _CONNECTION_KEYS, _IDENTITY_KEYS
from llm_rosetta.shims._loader import _model_list_transforms


# ---------------------------------------------------------------- spec_for
def test_spec_for_returns_the_catalog_record():
    spec = api.spec_for("llm", "messages")
    assert spec is not None
    assert (spec.model_type, spec.format) == ("llm", "messages")
    assert (
        spec is api.FORMAT_SPECS[[s.format for s in api.FORMAT_SPECS].index("messages")]
    )


def test_spec_for_unknown_pair_is_none():
    assert api.spec_for("llm", "no_such_format") is None
    assert api.spec_for("embeddings", "messages") is None


def test_spec_for_does_not_translate_a_retired_model_type():
    assert api.spec_for("embedding", "openai") is None
    assert api.spec_for("embeddings", "openai") is not None


# ------------------------------------------------------ typed_response_usage
_EMBED_BODY = {
    "object": "list",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
    "model": "m",
    "usage": {"prompt_tokens": 7, "total_tokens": 7},
}


def test_typed_usage_embeddings_matches_family_reader():
    from llm_rosetta.converters.embeddings.pipeline import response_usage

    got = api.typed_response_usage("embeddings", "openai", _EMBED_BODY)
    assert got == response_usage("openai", _EMBED_BODY)
    assert got  # the body carries usage, so the dict is not empty


def test_typed_usage_dispatches_per_family():
    from llm_rosetta.converters.decisions.pipeline import response_usage as dec
    from llm_rosetta.converters.rerank.pipeline import response_usage as rer

    rerank_body = {
        "model": "m",
        "usage": {"total_tokens": 5},
        "results": [{"index": 0, "relevance_score": 0.5}],
    }
    got = api.typed_response_usage("rerank", "jina", rerank_body)
    assert got == rer("jina", rerank_body)
    assert got  # non-empty, so a wrong family reader would not match
    decision_body = {
        "model": "m",
        "answers": [],
        "usage": {"input_tokens": 3, "output_tokens": 1},
    }
    got = api.typed_response_usage("decisions", "openai_decisions", decision_body)
    assert got == dec("openai_decisions", decision_body)
    assert got


def test_typed_usage_rejects_a_retired_spelling():
    with pytest.raises(ValueError, match="'embeddings'"):
        api.typed_response_usage("embedding", "openai", _EMBED_BODY)


def test_typed_usage_empty_or_unreadable_body_is_empty():
    assert api.typed_response_usage("embeddings", "openai", None) == {}
    assert api.typed_response_usage("embeddings", "no_such", _EMBED_BODY) == {}


def test_typed_usage_rejects_non_typed_model_type():
    with pytest.raises(ValueError):
        api.typed_response_usage("llm", "messages", {"usage": {}})


# ------------------------------------------------------ reasoning_capability
def _shim(**kw) -> ProviderShim:
    return ProviderShim(name="t-reasoning", format="chat_completions", **kw)


_BASE = ReasoningCapability(effort_field="reasoning_effort", budget_ratio=0.5)
_MODEL = ReasoningCapability(effort_field="model_field")


def test_reasoning_capability_provider_default():
    assert api.reasoning_capability(_shim(reasoning=_BASE), "any") is _BASE


def test_reasoning_capability_model_entry_beats_provider():
    shim = _shim(reasoning=_BASE, model_reasoning={"special": _MODEL})
    assert api.reasoning_capability(shim, "special") is _MODEL
    assert api.reasoning_capability(shim, "other") is _BASE


def test_reasoning_capability_override_merges_field_by_field():
    cap = api.reasoning_capability(_shim(reasoning=_BASE), "m", {"effort_field": "x.y"})
    assert cap is not None
    assert cap.effort_field == "x.y"
    assert cap.budget_ratio == 0.5  # inherited


def test_reasoning_capability_override_applies_on_model_entry():
    shim = _shim(reasoning=_BASE, model_reasoning={"special": _MODEL})
    cap = api.reasoning_capability(shim, "special", {"budget_ratio": 0.9})
    assert cap is not None
    assert (cap.effort_field, cap.budget_ratio) == ("model_field", 0.9)


def test_reasoning_capability_none_cases():
    assert api.reasoning_capability(None) is None
    assert api.reasoning_capability("no-such-shim-name") is None
    # a shim without reasoning has no base, so an override has nothing to merge onto
    assert api.reasoning_capability(_shim(), "m", {"effort_field": "x"}) is None


def test_reasoning_capability_resolves_registered_name():
    register_shim(_shim(reasoning=_BASE))
    assert api.reasoning_capability("t-reasoning", "m") is _BASE


# ---------------------------------------------------- describe_vendor_manifest
def _vendor(tmp_path):
    v = tmp_path / "acme"
    (v / "llm" / "chat_completions").mkdir(parents=True)
    (v / "embeddings").mkdir()
    (v / "provider.yaml").write_text(
        "connection:\n  base_url: https://a.example/v1\n  api_key_env: ACME_KEY\n"
        "  headers:\n    x-a: '1'\n"
    )
    (v / "llm" / "chat_completions" / "provider.yaml").write_text(
        "name: acme--chat_completions\nformat: chat_completions\n"
        "connection:\n  api_key_env: OTHER_KEY\n  headers:\n    x-b: '2'\n"
    )
    (v / "embeddings" / "provider.yaml").write_text(
        "name: acme--embeddings\nmodel_type: embeddings\nformat: openai\n"
    )
    return v


def test_describe_manifest_effective_connection(tmp_path):
    d = api.describe_vendor_manifest(_vendor(tmp_path))
    assert d.vendor == "acme"
    assert d.connection["base_url"] == "https://a.example/v1"
    by_name = {leaf.name: leaf for leaf in d.leaves}
    assert set(by_name) == {"acme--chat_completions", "acme--embeddings"}
    chat = by_name["acme--chat_completions"]
    assert chat.own == {"api_key_env": "OTHER_KEY", "headers": {"x-b": "2"}}
    assert chat.effective["base_url"] == "https://a.example/v1"
    assert chat.effective["api_key_env"] == "OTHER_KEY"  # leaf wins
    assert chat.effective["headers"] == {"x-a": "1", "x-b": "2"}  # per-key merge
    emb = by_name["acme--embeddings"]
    assert emb.own == {}
    assert emb.effective["api_key_env"] == "ACME_KEY"


def test_describe_manifest_accepts_file_path_and_preview(tmp_path):
    v = _vendor(tmp_path)
    d = api.describe_vendor_manifest(
        v / "provider.yaml", connection={"base_url": "https://b.example"}
    )
    assert d.connection == {"base_url": "https://b.example"}
    emb = next(leaf for leaf in d.leaves if leaf.name == "acme--embeddings")
    assert emb.effective == {"base_url": "https://b.example"}


def test_describe_manifest_exposes_keys_and_schema(tmp_path):
    d = api.describe_vendor_manifest(_vendor(tmp_path))
    assert "base_url" in d.connection_keys
    assert {"name", "format"} <= set(d.identity_keys)
    assert "connection" in d.schema["manifest"].get("properties", {})
    assert "base_url" in d.schema["connection"].get("properties", {})


def test_describe_manifest_skips_nested_vendor_manifest(tmp_path):
    v = _vendor(tmp_path)
    nested = v / "lane"
    (nested / "llm" / "messages").mkdir(parents=True)
    (nested / "provider.yaml").write_text("connection:\n  base_url: https://n\n")
    (nested / "llm" / "messages" / "provider.yaml").write_text(
        "name: acme-lane--messages\nformat: messages\n"
    )
    names = {leaf.name for leaf in api.describe_vendor_manifest(v).leaves}
    assert "acme-lane--messages" not in names
    assert "acme--embeddings" in names


def test_describe_manifest_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        api.describe_vendor_manifest(tmp_path)


# --------------------------------------------- manifest text, no file on disk
def test_manifest_vocabulary_matches_the_loader_and_a_described_manifest(tmp_path):

    vocab = api.manifest_vocabulary()
    assert vocab.connection_keys == _CONNECTION_KEYS
    assert vocab.identity_keys == _IDENTITY_KEYS
    assert set(vocab.schema) == {"manifest", "connection"}
    assert set(vocab.schema["connection"]["properties"]) == set(_CONNECTION_KEYS)
    d = api.describe_vendor_manifest(_vendor(tmp_path))
    assert d.schema == vocab.schema
    assert d.connection_keys == vocab.connection_keys
    assert d.identity_keys == vocab.identity_keys


def test_leaf_connection_reads_the_connection_block_only():
    assert api.leaf_connection({"connection": {"base_url": "u"}}) == {"base_url": "u"}
    # The flat spellings were removed in 0.14.0 and are not read.
    assert api.leaf_connection({"name": "x", "base_url": "u"}) == {}
    assert api.leaf_connection({"base_url": "f", "connection": {}}) == {}


@pytest.mark.parametrize("cfg", [None, [], "text", 3])
def test_leaf_connection_of_a_non_mapping_is_empty(cfg):
    assert api.leaf_connection(cfg) == {}


def test_merge_connection_leaf_wins_and_headers_merge_per_key():
    manifest = {"base_url": "m", "api_key_env": "MK", "headers": {"a": "1", "b": "2"}}
    leaf = {"api_key_env": "LK", "headers": {"b": "9"}}
    merged = api.merge_connection(manifest, leaf)
    assert merged == {
        "base_url": "m",
        "api_key_env": "LK",
        "headers": {"a": "1", "b": "9"},
    }
    # Neither argument is modified.
    assert manifest["headers"] == {"a": "1", "b": "2"}
    assert leaf == {"api_key_env": "LK", "headers": {"b": "9"}}


def test_merge_connection_agrees_with_describe_vendor_manifest(tmp_path):
    d = api.describe_vendor_manifest(_vendor(tmp_path))
    for leaf in d.leaves:
        assert api.merge_connection(d.connection, leaf.own) == leaf.effective


# ----------------------------------------------------------- staging_registry
def test_staging_registry_drops_derived_caches_on_entry():
    from llm_rosetta.auto_detect import _converter_cache

    _converter_cache["sentinel"] = object()  # type: ignore[assignment]
    _model_list_transforms["sentinel"] = lambda *a: a  # type: ignore[assignment]
    with api.staging_registry():
        assert "sentinel" not in _converter_cache
        assert "sentinel" not in _model_list_transforms


def test_staging_registry_failure_keeps_registry_and_transforms():

    register_shim(_shim(reasoning=_BASE))

    def marker(models):  # type: ignore[no-untyped-def]
        return [], {}

    _model_list_transforms["t-reasoning"] = marker  # type: ignore[assignment]
    with pytest.raises(RuntimeError):
        with api.staging_registry():
            raise RuntimeError("boom")
    assert get_shim("t-reasoning") is not None
    assert _model_list_transforms["t-reasoning"] is marker


def test_staging_registry_success_swaps_registry():
    with api.staging_registry():
        register_shim(_shim(reasoning=_BASE))
    assert get_shim("t-reasoning") is not None


# ------------------------------------------ the last non-surface imports (#1630)
def test_builtin_providers_dir_is_the_root_load_providers_scans():
    root = api.builtin_providers_dir()
    assert root.is_dir()
    assert (root / "openai").is_dir()
    assert root == providers._PROVIDERS_DIR


def test_plugin_source_is_none_for_a_directory_shim():
    assert api.plugin_source("openai--open_responses") is None
    assert api.plugin_source("no-such-shim") is None


def test_the_scaffold_entry_points_are_the_scaffold_module_s():
    import argparse

    from llm_rosetta.shims import scaffold

    assert api.run_scaffold is scaffold.run
    assert api.run_scaffold_check is scaffold.run_check
    assert api.run_scaffold_migrate is scaffold.run_migrate
    parser = argparse.ArgumentParser()
    api.add_scaffold_arguments(parser, default_root="providers")
    assert parser.parse_args(["acme"]).__dict__
