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

    def test_token_command_disables_keyless(self):
        """A keyless shim driven by token_command is not treated as keyless —
        the command (and its dynamic credential) must not be dropped."""
        load_providers()
        info = build_provider_info(
            "openai_chat",
            {"token_command": ["echo", "tok"]},
            shim_name="kilo--openai_chat",
        )

        assert info.keyless is False
        assert info.token_command == ["echo", "tok"]

    def test_empty_base_url_falls_back_to_shim_default(self):
        """An empty base_url string must not shadow the shim's default."""
        load_providers()
        info = build_provider_info(
            "openai_chat",
            {"api_key": "sk-x", "base_url": ""},
            shim_name="kilo--openai_chat",
        )

        assert info.base_url == "https://api.kilo.ai/api/gateway"

    def test_base_type_env_key_disables_keyless(self, monkeypatch):
        """A key resolved from the base-type env var turns keyless off — it is
        computed after the env fallback, not before."""
        load_providers()
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

        info = build_provider_info("openai_chat", {}, shim_name="kilo--openai_chat")

        assert info.keyless is False
        assert info.auth_headers() == {"Authorization": "Bearer sk-env"}
