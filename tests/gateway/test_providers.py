"""Tests for gateway provider metadata and auth behavior."""

from __future__ import annotations

from llm_rosetta.gateway.providers import build_provider_info
from llm_rosetta.shims.providers import load_providers


class TestBuildProviderInfo:
    def test_argo_openai_chat_uses_bearer_auth(self, monkeypatch):
        load_providers()
        monkeypatch.setenv("ARGO_API_KEY", "pding")

        info = build_provider_info("argo--openai_chat", {})

        assert info.auth_headers() == {"Authorization": "Bearer pding"}
        assert (
            info.upstream_url("gpt5")
            == "https://apps.inside.anl.gov/argoapi/v1/chat/completions"
        )

    def test_argo_anthropic_uses_x_api_key_auth(self, monkeypatch):
        load_providers()
        monkeypatch.setenv("ARGO_API_KEY", "pding")

        info = build_provider_info("argo--anthropic", {})

        assert info.auth_headers() == {
            "x-api-key": "pding",
            "anthropic-version": "2023-06-01",
        }
        assert (
            info.upstream_url("claudeopus47")
            == "https://apps.inside.anl.gov/argoapi/v1/messages"
        )


class TestOpenResponsesDefaults:
    """The spec has no canonical host, so open_responses defaults to OpenRouter."""

    def test_resolves_to_openrouter_without_config(self):
        info = build_provider_info("open_responses", {})
        assert info.upstream_url("m") == "https://openrouter.ai/api/v1/responses"

    def test_registry_defaults(self):
        from llm_rosetta.gateway.providers import (
            get_default_api_key_env,
            get_default_base_url,
        )

        assert get_default_base_url("open_responses") == "https://openrouter.ai/api/v1"
        assert get_default_api_key_env("open_responses") == "OPENROUTER_API_KEY"
