"""Tests for gateway provider metadata and auth behavior."""

from __future__ import annotations

import pytest

from llm_rosetta_gateway.providers import build_provider_info
from llm_rosetta_gateway.transport.provider_info import TOKEN_PENDING_SENTINEL
from llm_rosetta.shims.providers import load_providers


class TestBuildProviderInfo:
    def test_chat_shim_uses_bearer_auth(self, monkeypatch):
        load_providers()
        monkeypatch.setenv("OPENAI_API_KEY", "pding")

        info = build_provider_info("openai--chat_completions", {})

        assert info.auth_headers() == {"Authorization": "Bearer pding"}
        assert info.upstream_url("gpt5") == "https://api.openai.com/v1/chat/completions"

    def test_messages_shim_uses_x_api_key_auth(self, monkeypatch):
        load_providers()
        monkeypatch.setenv("ANTHROPIC_API_KEY", "pding")

        info = build_provider_info("anthropic--messages", {})

        assert info.auth_headers() == {
            "x-api-key": "pding",
            "anthropic-version": "2023-06-01",
        }
        assert (
            info.upstream_url("claude-opus-4-6")
            == "https://api.anthropic.com/v1/messages"
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
        from llm_rosetta_gateway.providers import (
            get_default_api_key_env,
            get_default_base_url,
        )

        assert get_default_base_url("open_responses") == ""
        assert get_default_api_key_env("open_responses") == ""

    def test_hints_are_ui_placeholders(self):
        from llm_rosetta_gateway.providers import get_base_type_hints

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

    def test_token_command_is_not_shadowed_by_the_shims_env_var(self, monkeypatch):
        """Exporting the shim's key variable must not break a command provider.

        The shim's ``api_key_env`` is a *default* credential and the command is
        a different source for the same slot.  The env value used to be
        injected first, and the pair is then rejected as mutually exclusive —
        so the documented setup (``export ALCF_API_KEY=...`` next to
        ``token_command``) failed to start.
        """
        load_providers()
        monkeypatch.setenv("XAI_API_KEY", "sk-from-the-environment")

        info = build_provider_info(
            "chat_completions",
            {"token_command": ["echo", "tok"]},
            shim_name="xai",
        )

        assert info.token_command == ["echo", "tok"]
        assert info.key_ring.contains_only(TOKEN_PENDING_SENTINEL)

    def test_an_empty_stored_key_is_not_a_second_source(self):
        """``"api_key": ""`` beside a command is the admin's "no key", not a conflict.

        The provider editor always sends the key field; a command provider's is
        empty, and the save therefore stores ``""``.  Rejecting that made every
        saved command provider come back auto-disabled.
        """
        load_providers()

        info = build_provider_info(
            "chat_completions",
            {"api_key": "", "token_command": ["echo", "tok"]},
            shim_name="xai",
        )

        assert info.token_command == ["echo", "tok"]
        assert info.key_ring.contains_only(TOKEN_PENDING_SENTINEL)

    def test_mutual_exclusion_names_the_shim_the_operator_wrote(self):
        """The error must name a key the config actually contains."""
        load_providers()

        with pytest.raises(ValueError, match=r"Provider 'xai'"):
            build_provider_info(
                "chat_completions",
                {"api_key": "sk-x", "token_command": ["echo", "tok"]},
                shim_name="xai",
            )

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


class TestSessionHeader:
    """Shim-declared per-conversation session headers."""

    def test_opencode_shim_declares_session_header(self, monkeypatch):
        load_providers()
        monkeypatch.setenv("OPENCODE_API_KEY", "oc_sk_test")

        info = build_provider_info("opencode--chat_completions", {})

        assert info.session_header == "x-opencode-session"
        assert info.upstream_url("glm-5.3") == (
            "https://opencode.ai/zen/go/v1/chat/completions"
        )
        # No id derived yet -> no header emitted.
        assert info.session_headers() == {}
        assert info.with_session("abc").session_headers() == {
            "x-opencode-session": "abc"
        }

    def test_opencode_messages_variant_uses_x_api_key(self, monkeypatch):
        load_providers()
        monkeypatch.setenv("OPENCODE_API_KEY", "oc_sk_test")

        info = build_provider_info("opencode--messages", {})

        assert info.auth_headers() == {
            "x-api-key": "oc_sk_test",
            "anthropic-version": "2023-06-01",
        }
        assert info.upstream_url("m") == "https://opencode.ai/zen/go/v1/messages"

    def test_config_overrides_session_header(self, monkeypatch):
        load_providers()
        monkeypatch.setenv("OPENCODE_API_KEY", "oc_sk_test")

        info = build_provider_info(
            "opencode--chat_completions", {"session_header": "x-other-session"}
        )

        assert info.session_header == "x-other-session"

    def test_provider_without_session_header(self):
        load_providers()
        info = build_provider_info("xai", {"api_key": "k"})

        assert info.session_header is None
        assert info.session_headers() == {}
        assert info.with_session("abc") is info


class TestShimDeclaredTokenCommand:
    """A shim ships its credential exchange; the provider's own command wins."""

    @pytest.fixture()
    def vendor_shim(self):
        from llm_rosetta.shims.provider_shim import (
            ConnectionConfig,
            ProviderShim,
            register_shim,
            unregister_shim,
        )

        register_shim(
            ProviderShim(
                name="__test_credential_command__",
                format="chat_completions",
                connection=ConnectionConfig(
                    base_url="https://vendor.example/v1",
                    api_key_env="VENDOR_API_KEY",
                    token_command=("python3", "-c", "print('minted')"),
                    token_refresh_interval=7200,
                ),
            )
        )
        yield "__test_credential_command__"
        unregister_shim("__test_credential_command__")

    def test_the_declared_command_is_the_default(self, vendor_shim):
        """A provider needs no token_command of its own to use the shim's."""
        info = build_provider_info("openai_chat", {}, shim_name=vendor_shim)

        assert info.token_command == ["python3", "-c", "print('minted')"]
        assert info.token_refresh_interval == 7200
        assert info.key_ring.contains_only(TOKEN_PENDING_SENTINEL)

    def test_the_providers_own_command_wins(self, vendor_shim):
        info = build_provider_info(
            "openai_chat",
            {"token_command": ["acme-token", "--json"]},
            shim_name=vendor_shim,
        )

        assert info.token_command == ["acme-token", "--json"]
        # The interval is the vendor's token lifetime, so it still applies.
        assert info.token_refresh_interval == 7200

    def test_the_interval_can_be_overridden(self, vendor_shim):
        info = build_provider_info(
            "openai_chat", {"token_refresh_interval": 120}, shim_name=vendor_shim
        )

        assert info.token_refresh_interval == 120

    def test_the_shims_key_variable_is_not_injected(self, vendor_shim, monkeypatch):
        """The command owns the credential, so the env default must stay out."""
        monkeypatch.setenv("VENDOR_API_KEY", "sk-from-the-environment")

        info = build_provider_info("openai_chat", {}, shim_name=vendor_shim)

        assert info.token_command == ["python3", "-c", "print('minted')"]
        assert info.key_ring.contains_only(TOKEN_PENDING_SENTINEL)
