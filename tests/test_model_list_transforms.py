"""Tests for built-in model_list_transform hooks (Jina, Cohere)."""

from __future__ import annotations

import pytest

from llm_rosetta.shims.providers import (
    _jina_model_list_transform,
    _cohere_model_list_transform,
    get_model_list_transform,
    load_providers,
)


@pytest.fixture(autouse=True)
def _ensure_transforms_loaded():
    """Repopulate built-in transforms in case another test reset the registry."""
    load_providers()
    yield


class TestJinaModelListTransform:
    """Unit tests for the Jina model_list_transform."""

    def test_strips_prefix(self):
        raw = [{"id": "jina-ai/jina-embeddings-v3"}]
        ids, upstream = _jina_model_list_transform(raw)
        assert ids == ["jina-embeddings-v3"]
        assert upstream == {"jina-embeddings-v3": "jina-ai/jina-embeddings-v3"}

    def test_multiple_models(self):
        raw = [
            {"id": "jina-ai/jina-reranker-v3"},
            {"id": "jina-ai/jina-embeddings-v3"},
            {"id": "jina-ai/jina-clip-v2"},
        ]
        ids, upstream = _jina_model_list_transform(raw)
        assert ids == ["jina-reranker-v3", "jina-embeddings-v3", "jina-clip-v2"]
        assert all(v.startswith("jina-ai/") for v in upstream.values())

    def test_no_prefix(self):
        raw = [{"id": "some-model-without-prefix"}]
        ids, upstream = _jina_model_list_transform(raw)
        assert ids == ["some-model-without-prefix"]
        assert upstream == {}

    def test_empty_id_skipped(self):
        raw = [{"id": ""}, {"id": "jina-ai/valid"}]
        ids, upstream = _jina_model_list_transform(raw)
        assert ids == ["valid"]

    def test_empty_input(self):
        ids, upstream = _jina_model_list_transform([])
        assert ids == []
        assert upstream == {}

    def test_registered(self):
        t = get_model_list_transform("jina")
        assert t is not None
        ids, _ = t([{"id": "jina-ai/test"}])
        assert ids == ["test"]


class TestCohereModelListTransform:
    """Unit tests for the Cohere model_list_transform."""

    def test_parses_name_field(self):
        raw = [{"name": "rerank-v3.5", "endpoints": ["rerank"]}]
        ids, upstream = _cohere_model_list_transform(raw)
        assert ids == ["rerank-v3.5"]
        assert upstream == {}

    def test_multiple_models(self):
        raw = [
            {"name": "command-a-03-2025", "endpoints": ["chat"]},
            {"name": "embed-english-v3.0", "endpoints": ["embed"]},
            {"name": "rerank-v3.5", "endpoints": ["rerank"]},
        ]
        ids, upstream = _cohere_model_list_transform(raw)
        assert ids == ["command-a-03-2025", "embed-english-v3.0", "rerank-v3.5"]

    def test_empty_name_skipped(self):
        raw = [{"name": ""}, {"name": "valid-model"}]
        ids, upstream = _cohere_model_list_transform(raw)
        assert ids == ["valid-model"]

    def test_empty_input(self):
        ids, upstream = _cohere_model_list_transform([])
        assert ids == []
        assert upstream == {}

    def test_registered(self):
        t = get_model_list_transform("cohere")
        assert t is not None
        ids, _ = t([{"name": "test-model"}])
        assert ids == ["test-model"]
