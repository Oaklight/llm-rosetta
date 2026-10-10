"""Tests for gateway provider metadata and auth behavior."""

from __future__ import annotations

import pytest

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


class TestOpenResponsesVendorNeutral:
    """open_responses has no canonical host but keeps the default key env."""

    def test_missing_base_url_raises_clear_error(self):
        with pytest.raises(ValueError, match="no base_url configured"):
            build_provider_info("open_responses", {})

    def test_default_key_env_is_preserved(self):
        from llm_rosetta.gateway.providers import (
            get_default_api_key_env,
            get_default_base_url,
        )

        assert get_default_base_url("open_responses") == ""
        assert get_default_api_key_env("open_responses") == "OPENAI_API_KEY"
