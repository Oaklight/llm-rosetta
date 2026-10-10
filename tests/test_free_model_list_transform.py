"""Tests for the free-source shim's model_list_transform."""

from __future__ import annotations

import pytest

from llm_rosetta.shims.providers import get_model_list_transform, load_providers


@pytest.fixture(autouse=True)
def _ensure_transforms_loaded():
    load_providers()
    yield


def _transform():
    t = get_model_list_transform("free--openai_chat")
    assert t is not None
    return t


class TestFreeModelListTransform:
    def test_keeps_only_free_models(self):
        raw = [
            {"id": "nvidia/nemotron-3-ultra-550b-a55b:free", "isFree": True},
            {"id": "gpt-6", "isFree": False},
            {"id": "stepfun/step-5-preview-free", "isFree": True},
        ]
        ids, upstream = _transform()(raw)

        assert ids == [
            "nvidia/nemotron-3-ultra-550b-a55b:free",
            "stepfun/step-5-preview-free",
        ]
        assert upstream == {}

    def test_drops_branded_router_ids(self):
        raw = [
            {"id": "kilo-auto/free", "isFree": True},
            {"id": "cohere/north-mini-code:free", "isFree": True},
        ]
        ids, _ = _transform()(raw)

        assert ids == ["cohere/north-mini-code:free"]

    def test_missing_isFree_is_dropped(self):
        raw = [{"id": "some/model"}]
        ids, _ = _transform()(raw)

        assert ids == []

    def test_empty_id_is_dropped(self):
        raw = [{"id": "", "isFree": True}]
        ids, _ = _transform()(raw)

        assert ids == []
