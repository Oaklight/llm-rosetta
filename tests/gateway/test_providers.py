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


class TestKeylessProvider:
    """A free-source shim sends no auth header, and accepts a key for paid use."""

    def test_free_shim_sends_no_auth(self):
        load_providers()
        info = build_provider_info("openai_chat", {}, shim_name="kilo--openai_chat")

        assert info.auth_headers() == {}
        assert info.ready is True
        assert info.keyless is True
        assert info.base_url == "https://api.kilo.ai/api/gateway"

    def test_free_shim_with_key_sends_bearer(self):
        load_providers()
        info = build_provider_info(
            "openai_chat", {"api_key": "sk-x"}, shim_name="kilo--openai_chat"
        )

        assert info.auth_headers() == {"Authorization": "Bearer sk-x"}
        assert info.keyless is False

    def test_non_keyless_provider_without_key_still_raises(self):
        load_providers()
        info = build_provider_info("openai_chat", {"api_key": ""})

        with pytest.raises(ValueError, match="No API keys configured"):
            info.auth_headers()
