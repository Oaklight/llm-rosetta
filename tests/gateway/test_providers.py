"""Tests for gateway provider metadata and auth behavior."""

from __future__ import annotations

import pytest

from llm_rosetta.gateway.providers import build_provider_info
from llm_rosetta.shims.providers import load_providers


class TestBuildProviderInfo:
    def test_argo_openai_chat_uses_bearer_auth(self, monkeypatch):
        load_providers()
        monkeypatch.setenv("ARGO_API_KEY", "pding")

        info = build_provider_info("argo--chat_completions", {})

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


class TestVendorNeutralBaseFormats:
    """A bare base format is vendor-neutral: no implied host or credential env.

    The vendor URL is only a UI placeholder hint (``get_base_type_hints``);
    runtime resolution requires an explicit ``base_url`` or a vendor shim.
    """

    def test_bare_open_responses_requires_explicit_base_url(self):
        with pytest.raises(ValueError, match="no base_url configured"):
            build_provider_info("open_responses", {})

    def test_explicit_base_url_is_used(self):
        info = build_provider_info("open_responses", {"base_url": "https://x/v1"})
        assert info.upstream_url("m") == "https://x/v1/responses"

    def test_vendor_shim_supplies_base_url(self):
        load_providers()
        info = build_provider_info(
            "open_responses", {"api_key": "k"}, shim_name="openrouter--open_responses"
        )
        assert info.upstream_url("m") == "https://openrouter.ai/api/v1/responses"

    def test_registry_defaults_are_empty(self):
        from llm_rosetta.gateway.providers import (
            get_default_api_key_env,
            get_default_base_url,
        )

        assert get_default_base_url("open_responses") == ""
        assert get_default_api_key_env("open_responses") == ""

    def test_hints_are_ui_placeholders(self):
        from llm_rosetta.gateway.providers import get_base_type_hints

        assert get_base_type_hints("open_responses") == {
            "base_url": "https://openrouter.ai/api/v1",
            "api_key_env": "OPENROUTER_API_KEY",
        }


class TestKeylessProvider:
    """A free-source shim sends no auth header, and accepts a key for paid use."""

    def test_free_shim_sends_no_auth(self):
        load_providers()
        info = build_provider_info(
            "openai_chat", {}, shim_name="kilo--chat_completions"
        )

        assert info.auth_headers() == {}
        assert info.ready is True
        assert info.keyless is True
        assert info.base_url == "https://api.kilo.ai/api/gateway"

    def test_free_shim_with_key_sends_bearer(self):
        load_providers()
        info = build_provider_info(
            "openai_chat", {"api_key": "sk-x"}, shim_name="kilo--chat_completions"
        )

        assert info.auth_headers() == {"Authorization": "Bearer sk-x"}
        assert info.keyless is False

    def test_non_keyless_provider_without_key_still_raises(self):
        load_providers()
        info = build_provider_info(
            "openai_chat", {"api_key": "", "base_url": "https://x/v1"}
        )

        with pytest.raises(ValueError, match="No API keys configured"):
            info.auth_headers()

    def test_token_command_disables_keyless(self):
        """A keyless shim driven by token_command is not treated as keyless —
        the command (and its dynamic credential) must not be dropped."""
        load_providers()
        info = build_provider_info(
            "openai_chat",
            {"token_command": ["echo", "tok"]},
            shim_name="kilo--chat_completions",
        )

        assert info.keyless is False
        assert info.token_command == ["echo", "tok"]

    def test_empty_base_url_falls_back_to_shim_default(self):
        """An empty base_url string must not shadow the shim's default."""
        load_providers()
        info = build_provider_info(
            "openai_chat",
            {"api_key": "sk-x", "base_url": ""},
            shim_name="kilo--chat_completions",
        )

        assert info.base_url == "https://api.kilo.ai/api/gateway"

    def test_base_type_env_key_does_not_disable_keyless(self, monkeypatch):
        """An unrelated base-type env var must not turn a keyless shim into an
        authed provider — it is not this provider's credential."""
        load_providers()
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

        info = build_provider_info(
            "openai_chat", {}, shim_name="kilo--chat_completions"
        )

        assert info.keyless is True
        assert info.auth_headers() == {}

    def test_explicit_empty_api_key_is_not_refilled_from_env(self, monkeypatch):
        """A key the operator cleared ("") stays cleared, even with the env set."""
        load_providers()
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

        info = build_provider_info(
            "openai_chat", {"api_key": "", "base_url": "https://x/v1"}
        )

        assert info.keyless is False
        with pytest.raises(ValueError, match="No API keys configured"):
            info.auth_headers()

    def test_base_type_env_key_is_not_consulted(self, monkeypatch):
        """A bare base format does not inherit a vendor credential env var.

        ``chat_completions`` is vendor-neutral: an ``OPENAI_API_KEY`` in the
        environment is not this provider's credential.
        """
        load_providers()
        monkeypatch.setenv("OPENAI_API_KEY", "sk-env")

        info = build_provider_info("openai_chat", {"base_url": "https://x/v1"})

        with pytest.raises(ValueError, match="No API keys configured"):
            info.auth_headers()
