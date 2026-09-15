"""Tests for ALCF model_list_transform and absolute models_path support."""

from __future__ import annotations

import pytest

from llm_rosetta.shims.providers import get_model_list_transform, load_providers
from llm_rosetta.shims.providers.alcf.model_utils import (
    make_alcf_model_list_transform,
)


@pytest.fixture(autouse=True)
def _ensure_transforms_loaded():
    load_providers()
    yield


# ── Factory tests ────────────────────────────────────────────────────


class TestMakeAlcfModelListTransform:
    def test_filters_by_framework(self):
        raw = [
            {"id": "openai/gpt-oss-120b", "framework": "vllm"},
            {"id": "dinov3", "framework": "dinoserver"},
            {"id": "sam3", "framework": "sam3service"},
        ]
        transform = make_alcf_model_list_transform("vllm")
        ids, upstream = transform(raw)
        assert ids == ["openai/gpt-oss-120b"]
        assert upstream == {}

    def test_api_framework(self):
        raw = [
            {"id": "gpt-oss-120b", "framework": "api"},
            {"id": "nemotron-3-ultra", "framework": "api"},
        ]
        transform = make_alcf_model_list_transform("api")
        ids, _ = transform(raw)
        assert ids == ["gpt-oss-120b", "nemotron-3-ultra"]

    def test_empty_input(self):
        transform = make_alcf_model_list_transform("vllm")
        ids, upstream = transform([])
        assert ids == []
        assert upstream == {}

    def test_no_matching_framework(self):
        raw = [{"id": "dinov3", "framework": "dinoserver"}]
        transform = make_alcf_model_list_transform("vllm")
        ids, _ = transform(raw)
        assert ids == []

    def test_missing_framework_field(self):
        raw = [{"id": "some-model"}]
        transform = make_alcf_model_list_transform("vllm")
        ids, _ = transform(raw)
        assert ids == []

    def test_empty_id_skipped(self):
        raw = [{"id": "", "framework": "vllm"}, {"id": "valid", "framework": "vllm"}]
        transform = make_alcf_model_list_transform("vllm")
        ids, _ = transform(raw)
        assert ids == ["valid"]


# ── Registration tests ───────────────────────────────────────────────


class TestAlcfTransformRegistration:
    @pytest.mark.parametrize(
        "shim_name",
        ["alcf--sophia", "alcf--metis", "alcf--minerva"],
    )
    def test_registered(self, shim_name: str):
        t = get_model_list_transform(shim_name)
        assert t is not None

    def test_sophia_filters_vllm(self):
        t = get_model_list_transform("alcf--sophia")
        assert t is not None
        raw = [
            {"id": "openai/gpt-oss-120b", "framework": "vllm"},
            {"id": "dinov3", "framework": "dinoserver"},
        ]
        ids, _ = t(raw)
        assert "openai/gpt-oss-120b" in ids
        assert "dinov3" not in ids

    def test_metis_filters_api(self):
        t = get_model_list_transform("alcf--metis")
        assert t is not None
        raw = [
            {"id": "gpt-oss-120b", "framework": "api"},
            {"id": "other", "framework": "vllm"},
        ]
        ids, _ = t(raw)
        assert ids == ["gpt-oss-120b"]


# ── models_path on shim ──────────────────────────────────────────────


class TestAlcfModelsPath:
    def test_sophia_has_absolute_models_path(self):
        from llm_rosetta.shims import get_shim

        shim = get_shim("alcf--sophia")
        assert shim is not None
        assert shim.models_path is not None
        assert shim.models_path.startswith("https://")
        assert "sophia/models" in shim.models_path

    def test_metis_has_absolute_models_path(self):
        from llm_rosetta.shims import get_shim

        shim = get_shim("alcf--metis")
        assert shim is not None
        assert shim.models_path is not None
        assert shim.models_path.startswith("https://")

    def test_minerva_has_absolute_models_path(self):
        from llm_rosetta.shims import get_shim

        shim = get_shim("alcf--minerva")
        assert shim is not None
        assert shim.models_path is not None
        assert shim.models_path.startswith("https://")


# ── _extract_model_ids bare-array handling ───────────────────────────


class TestExtractModelIdsBareArray:
    def test_bare_array_with_transform(self):
        from llm_rosetta.gateway.admin.routes.config import _extract_model_ids

        body = [
            {"id": "openai/gpt-oss-120b", "framework": "vllm"},
            {"id": "dinov3", "framework": "dinoserver"},
        ]
        ids, upstream = _extract_model_ids(body, "openai_chat", "alcf--sophia", None)
        assert "openai/gpt-oss-120b" in ids
        assert "dinov3" not in ids
