"""Unit tests for fetch_upstream_models helpers."""

from __future__ import annotations

from llm_rosetta.gateway.admin.routes.config import (
    _build_models_url,
    _extract_google_model_ids,
    _extract_model_ids,
    _extract_openai_model_ids,
)


class TestBuildModelsUrl:
    def test_google(self):
        assert _build_models_url("google", "https://x") == "https://x/v1beta/models"

    def test_anthropic(self):
        assert _build_models_url("anthropic", "https://x") == "https://x/v1/models"

    def test_openai_default(self):
        assert _build_models_url("openai_chat", "https://x") == "https://x/models"
        assert _build_models_url("unknown", "https://x") == "https://x/models"


class TestExtractGoogleModelIds:
    def test_strips_models_prefix(self):
        body = {"models": [{"name": "models/gemini-2.0"}, {"name": "gemini-1.5"}]}
        assert _extract_google_model_ids(body, None) == [
            "gemini-2.0",
            "gemini-1.5",
        ]

    def test_uses_id_field_when_set(self):
        body = {"models": [{"name": "models/gemini-2.0", "internal_id": "g20"}]}
        assert _extract_google_model_ids(body, "internal_id") == ["g20"]

    def test_missing_models(self):
        assert _extract_google_model_ids({}, None) == []


class TestExtractOpenAIModelIds:
    def test_default_id(self):
        body = {"data": [{"id": "gpt-4"}, {"id": "gpt-5"}]}
        assert _extract_openai_model_ids(body, None) == ["gpt-4", "gpt-5"]

    def test_id_field_override(self):
        body = {"data": [{"id": "gpt-4", "internal_id": "argo:gpt-4"}]}
        assert _extract_openai_model_ids(body, "internal_id") == ["argo:gpt-4"]

    def test_missing_data(self):
        assert _extract_openai_model_ids({}, None) == []


class TestExtractModelIds:
    def test_dispatch_google(self):
        body = {"models": [{"name": "models/z"}, {"name": "models/a"}]}
        # Filtered and sorted
        assert _extract_model_ids(body, "google", None) == ["a", "z"]

    def test_dispatch_openai(self):
        body = {"data": [{"id": "b"}, {"id": ""}, {"id": "a"}]}
        # Empty ids dropped, sorted
        assert _extract_model_ids(body, "openai_chat", None) == ["a", "b"]

    def test_anthropic_uses_data_format(self):
        body = {"data": [{"id": "claude-3"}]}
        assert _extract_model_ids(body, "anthropic", None) == ["claude-3"]
