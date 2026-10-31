"""The 0.13.0 ``converters.embedding`` path still resolves, and warns.

The package was renamed to the plural spelling in 0.14.0; the old path stays as
a deprecated re-export until 0.16.0 (#919).  ``converters.decision`` needs no
alias: 0.13.0 never shipped it.
"""

import importlib
import warnings

import pytest


def test_the_old_package_path_warns_and_reexports() -> None:
    import llm_rosetta.converters.embedding as old
    import llm_rosetta.converters.embeddings as new

    assert old.OpenAIEmbeddingConverter is new.OpenAIEmbeddingConverter
    assert old.EmbeddingConversionPipeline is new.EmbeddingConversionPipeline
    assert old.__all__ == new.__all__


def test_the_old_submodule_paths_resolve_to_the_same_module() -> None:
    canonical = importlib.import_module("llm_rosetta.converters.embeddings.pipeline")
    legacy = importlib.import_module("llm_rosetta.converters.embedding.pipeline")
    assert legacy is canonical

    canonical_minimax = importlib.import_module(
        "llm_rosetta.converters.embeddings.minimax"
    )
    legacy_minimax = importlib.import_module("llm_rosetta.converters.embedding.minimax")
    assert legacy_minimax is canonical_minimax


def test_the_deprecation_warning_points_at_the_caller() -> None:
    import sys

    saved = {
        name: module
        for name, module in list(sys.modules.items())
        if name.startswith("llm_rosetta.converters.embedding")
    }
    for name in saved:
        del sys.modules[name]

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        importlib.import_module("llm_rosetta.converters.embedding")

    deprecations = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert len(deprecations) == 1
    assert "llm_rosetta.converters.embedding is deprecated" in str(
        deprecations[0].message
    )

    # Restore the modules so later imports in this session stay quiet.
    for name, module in saved.items():
        sys.modules[name] = module


def test_decision_has_no_legacy_package_path() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("llm_rosetta.converters.decision")
