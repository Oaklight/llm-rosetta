"""Tests for built-in model_list_transform hooks (Jina, Cohere)."""

from __future__ import annotations

import pytest

from llm_rosetta.shims.providers import (
    _jina_model_list_transform,
    _cohere_model_list_transform,
    get_model_list_transform,
)
from _moved_vendors import load_providers_with_moved


@pytest.fixture(autouse=True)
def _ensure_transforms_loaded():
    """Repopulate built-in transforms in case another test reset the registry."""
    load_providers_with_moved()
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


class TestSiliconFlowModelListTransform:
    """The SiliconFlow catalogue is typed, so each shim keeps its own slice.

    One playground response carries every endpoint (chat, embedding, image,
    video, speech, rerank) and marks entries ``status: normal`` or
    ``status: disable``.  A picker must see only the active entries of its own
    subtype, and the id is ``modelName`` — the string the endpoint accepts.
    """

    ENTRIES = [
        {"modelName": "Qwen/Qwen3.6-27B", "status": "normal", "subType": "chat"},
        {
            "modelName": "deepseek-ai/DeepSeek-V4.1-Flash",
            "status": "normal",
            "subType": "chat",
        },
        {"modelName": "old/Retired-Chat", "status": "disable", "subType": "chat"},
        {"modelName": "BAAI/bge-m3", "status": "normal", "subType": "embedding"},
        {
            "modelName": "BAAI/bge-reranker-v2-m3",
            "status": "normal",
            "subType": "reranker",
        },
        {
            "modelName": "Wan-AI/Wan2.2-T2V-A14B",
            "status": "normal",
            "subType": "text-to-video",
        },
    ]

    def _ids(self, shim_name: str) -> list[str]:
        transform = get_model_list_transform(shim_name)
        assert transform is not None
        ids, upstream = transform(self.ENTRIES)
        assert upstream == {}
        return ids

    def test_chat_keeps_active_chat_models_only(self):
        assert self._ids("siliconflow--chat_completions") == [
            "Qwen/Qwen3.6-27B",
            "deepseek-ai/DeepSeek-V4.1-Flash",
        ]

    def test_embedding_keeps_active_embedding_models_only(self):
        assert self._ids("siliconflow--embedding") == ["BAAI/bge-m3"]

    def test_rerank_keeps_active_reranker_models_only(self):
        assert self._ids("siliconflow--rerank") == ["BAAI/bge-reranker-v2-m3"]

    def test_an_entry_without_a_name_is_dropped(self):
        transform = get_model_list_transform("siliconflow--chat_completions")
        assert transform is not None
        ids, _ = transform([{"status": "normal", "subType": "chat"}, self.ENTRIES[0]])
        assert ids == ["Qwen/Qwen3.6-27B"]
