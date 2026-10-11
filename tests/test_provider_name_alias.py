"""Tests for the shared legacy provider-name alias + auto-migrate mechanism.

Covers the four boundaries named in #908: the alias helper itself, converter
lookup / request auto-detection, library conversion arguments, and gateway
JSONC config resolution.
"""

from __future__ import annotations

from typing import get_args

import pytest

from llm_rosetta import (
    ConversionPipeline,
    ProviderType,
    convert,
    convert_response,
    get_converter_for_provider,
    normalize_provider_name,
)
from llm_rosetta.provider_names import LEGACY_PROVIDER_ALIASES
from llm_rosetta.shims import resolve_base

OPENAI_CHAT_BODY = {
    "model": "gpt-4o",
    "messages": [{"role": "user", "content": "hello"}],
}


class TestNormalizeProviderName:
    """The single source-of-truth helper."""

    @pytest.mark.parametrize(
        "legacy,canonical",
        [
            ("google", "google_generate"),
            ("google-genai", "google_generate"),
            ("openai_chat", "chat_completions"),
            ("openai-chat", "chat_completions"),
            ("openai-responses", "openai_responses"),
            ("open-responses", "open_responses"),
            ("google-interactions", "google_interactions"),
        ],
    )
    def test_legacy_names_map_to_canonical(self, legacy: str, canonical: str):
        with pytest.warns(DeprecationWarning, match=canonical):
            assert normalize_provider_name(legacy) == canonical

    @pytest.mark.parametrize(
        "name",
        [
            "openai_responses",
            "open_responses",
            "anthropic",
            "google_generate",
            "google_interactions",
            "decision",
        ],
    )
    def test_canonical_names_pass_through_without_warning(self, name: str):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert normalize_provider_name(name) == name

    def test_unknown_names_pass_through(self):
        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert normalize_provider_name("deepseek") == "deepseek"

    def test_warning_names_replacement(self):
        with pytest.warns(DeprecationWarning) as record:
            normalize_provider_name("google")
        assert "google" in str(record.list[0].message)
        assert "google_generate" in str(record.list[0].message)

    def test_every_alias_target_is_a_valid_provider_type(self):
        """The map may only map onto ProviderType members."""
        canonical = set(get_args(ProviderType))
        assert set(LEGACY_PROVIDER_ALIASES.values()) <= canonical

    def test_no_alias_target_is_a_legacy_key(self):
        """Canonical targets must not themselves be aliases (no chains)."""
        assert not (
            set(LEGACY_PROVIDER_ALIASES.values()) & set(LEGACY_PROVIDER_ALIASES)
        )


class TestConverterLookupBoundary:
    """``get_converter_for_provider`` migrates legacy names."""

    @pytest.mark.parametrize(
        "legacy,canonical",
        [
            ("google", "google_generate"),
            ("google-genai", "google_generate"),
            ("openai-chat", "chat_completions"),
        ],
    )
    def test_legacy_resolves_to_same_converter(self, legacy: str, canonical: str):
        with pytest.warns(DeprecationWarning):
            legacy_converter = get_converter_for_provider(legacy)
        assert legacy_converter is get_converter_for_provider(canonical)

    def test_toolless_unknown_still_raises(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            get_converter_for_provider("not-a-provider")


class TestResolveBaseBoundary:
    """``resolve_base`` migrates legacy names before shim lookup."""

    def test_google_legacy_migrates(self):
        with pytest.warns(DeprecationWarning):
            assert resolve_base("google") == "google_generate"

    def test_google_genai_migrates(self):
        with pytest.warns(DeprecationWarning):
            assert resolve_base("google-genai") == "google_generate"


class TestLibraryArgumentBoundary:
    """``convert`` / ``ConversionPipeline`` migrate legacy names."""

    def test_convert_legacy_target_equals_canonical(self):
        with pytest.warns(DeprecationWarning):
            legacy = convert(OPENAI_CHAT_BODY, "google-genai")
        canonical = convert(OPENAI_CHAT_BODY, "google_generate")
        assert legacy == canonical

    def test_convert_legacy_source_equals_canonical(self):
        body = {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "model": "gemini-2.0-flash",
        }
        with pytest.warns(DeprecationWarning):
            legacy = convert(body, "openai_chat", source_provider="google")
        canonical = convert(body, "openai_chat", source_provider="google_generate")
        assert legacy == canonical

    def test_pipeline_legacy_names_equal_canonical(self):
        with pytest.warns(DeprecationWarning):
            legacy = ConversionPipeline("openai-chat", "google-genai")
        canonical = ConversionPipeline("openai_chat", "google_generate")
        assert legacy._target_provider == canonical._target_provider
        assert legacy._source_provider == canonical._source_provider

    def test_convert_response_legacy_names_equal_canonical(self):
        response = {
            "id": "chatcmpl-1",
            "object": "chat.completion",
            "model": "gpt-4o",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hi"},
                    "finish_reason": "stop",
                }
            ],
        }
        with pytest.warns(DeprecationWarning):
            legacy = convert_response(
                response,
                OPENAI_CHAT_BODY,
                source_provider="openai-chat",
                target_provider="openai_chat",
            )
        canonical = convert_response(
            response,
            OPENAI_CHAT_BODY,
            source_provider="openai_chat",
            target_provider="openai_chat",
        )
        assert legacy == canonical


class TestGatewayConfigBoundary:
    """Gateway JSONC ``type:`` / ``shim:`` resolution migrates legacy names."""

    @staticmethod
    def _raw(provider_cfg: dict) -> dict:
        return {
            "providers": {
                "g": {"api_key": "k", "base_url": "https://x", **provider_cfg}
            },
            "models": {"gemini": "g"},
        }

    def test_type_field_migrates(self):
        from llm_rosetta.gateway.config import GatewayConfig

        with pytest.warns(DeprecationWarning):
            cfg = GatewayConfig(self._raw({"type": "google"}))
        assert cfg.provider_types["g"] == "google_generate"

    def test_hyphen_type_field_migrates(self):
        from llm_rosetta.gateway.config import GatewayConfig

        with pytest.warns(DeprecationWarning):
            cfg = GatewayConfig(self._raw({"type": "google-genai"}))
        assert cfg.provider_types["g"] == "google_generate"

    def test_shim_field_migrates(self):
        from llm_rosetta.gateway.config import GatewayConfig

        with pytest.warns(DeprecationWarning):
            cfg = GatewayConfig(self._raw({"shim": "google-genai"}))
        assert cfg.provider_types["g"] == "google_generate"

    def test_provider_name_fallback_migrates(self):
        from llm_rosetta.gateway.config import GatewayConfig

        raw = {
            "providers": {
                "google": {"api_key": "k", "base_url": "https://x"},
            },
            "models": {"gemini": "google"},
        }
        with pytest.warns(DeprecationWarning):
            cfg = GatewayConfig(raw)
        assert cfg.provider_types["google"] == "google_generate"


class TestCanonicalNameResolvesToShim:
    """A shim registered under its canonical name must stay reachable.

    Regression: normalising a name before a raw shim lookup silently dropped
    the shim when it was registered only under the other spelling.
    """

    def test_canonical_and_legacy_resolve(self):
        from llm_rosetta.shims import resolve_shim

        assert resolve_shim("google_generate") is not None
        with pytest.warns(DeprecationWarning):
            assert resolve_shim("google") is not None


class TestRegisterShimNormalises:
    """``register_shim`` keys on the normalised name, matching lookups."""

    def test_legacy_name_is_normalised_on_register(self):
        import llm_rosetta.shims.provider_shim as ps
        from llm_rosetta.shims import ProviderShim, register_shim

        saved = dict(ps._SHIM_REGISTRY)
        try:
            with pytest.warns(DeprecationWarning):
                register_shim(ProviderShim(name="google", base="google_generate"))
            assert "google_generate" in ps._SHIM_REGISTRY
            assert "google" not in ps._SHIM_REGISTRY
            assert ps.get_shim("google_generate") is ps.get_shim("google")
        finally:
            ps._SHIM_REGISTRY.clear()
            ps._SHIM_REGISTRY.update(saved)
