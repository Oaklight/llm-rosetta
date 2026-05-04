"""Tests for the provider shim layer."""

from __future__ import annotations

import pytest

from llm_rosetta.shims.provider_shim import (
    ModelShim,
    ProviderShim,
    _reset_registry,
    get_shim,
    list_shims,
    register_shim,
    resolve_base,
    unregister_shim,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clean_registry():
    """Reset the shim registry before and after each test."""
    _reset_registry()
    yield
    _reset_registry()


# ---------------------------------------------------------------------------
# ModelShim
# ---------------------------------------------------------------------------


class TestModelShim:
    def test_creation_defaults(self):
        m = ModelShim("gpt-*")
        assert m.pattern == "gpt-*"
        assert m.capabilities == frozenset()

    def test_creation_with_capabilities(self):
        m = ModelShim("o3-*", frozenset({"reasoning", "tools"}))
        assert m.capabilities == {"reasoning", "tools"}

    def test_frozen(self):
        m = ModelShim("gpt-*")
        with pytest.raises(AttributeError):
            m.pattern = "other"  # type: ignore


# ---------------------------------------------------------------------------
# ProviderShim
# ---------------------------------------------------------------------------


class TestProviderShim:
    def test_creation_minimal(self):
        s = ProviderShim(name="test", base="openai_chat")
        assert s.name == "test"
        assert s.base == "openai_chat"
        assert s.default_base_url is None
        assert s.default_api_key_env is None
        assert s.models == ()

    def test_creation_full(self):
        models = (
            ModelShim("o3-*", frozenset({"reasoning"})),
            ModelShim("gpt-*", frozenset({"tools"})),
        )
        s = ProviderShim(
            name="openai",
            base="openai_chat",
            default_base_url="https://api.openai.com/v1",
            default_api_key_env="OPENAI_API_KEY",
            models=models,
        )
        assert s.default_base_url == "https://api.openai.com/v1"
        assert len(s.models) == 2

    def test_frozen(self):
        s = ProviderShim(name="test", base="openai_chat")
        with pytest.raises(AttributeError):
            s.name = "other"  # type: ignore

    def test_get_model_shim_match(self):
        s = ProviderShim(
            name="openai",
            base="openai_chat",
            models=(
                ModelShim("o3-*", frozenset({"reasoning", "tools"})),
                ModelShim("gpt-*", frozenset({"tools"})),
            ),
        )
        m = s.get_model_shim("o3-mini")
        assert m is not None
        assert "reasoning" in m.capabilities

        m2 = s.get_model_shim("gpt-4o")
        assert m2 is not None
        assert m2.capabilities == {"tools"}

    def test_get_model_shim_no_match(self):
        s = ProviderShim(
            name="openai",
            base="openai_chat",
            models=(ModelShim("gpt-*", frozenset({"tools"})),),
        )
        assert s.get_model_shim("claude-3") is None

    def test_get_model_shim_first_match_wins(self):
        s = ProviderShim(
            name="test",
            base="openai_chat",
            models=(
                ModelShim("o3-*", frozenset({"reasoning"})),
                ModelShim("o3-mini*", frozenset({"tools"})),
            ),
        )
        m = s.get_model_shim("o3-mini")
        assert m is not None
        assert m.capabilities == {"reasoning"}  # first match

    def test_get_model_shim_empty_models(self):
        s = ProviderShim(name="test", base="openai_chat")
        assert s.get_model_shim("anything") is None


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class TestRegistry:
    def test_register_and_get(self):
        s = ProviderShim(name="test-provider", base="openai_chat")
        register_shim(s)
        assert get_shim("test-provider") is s

    def test_get_nonexistent(self):
        assert get_shim("nonexistent") is None

    def test_register_replaces(self):
        s1 = ProviderShim(name="test", base="openai_chat")
        s2 = ProviderShim(name="test", base="anthropic")
        register_shim(s1)
        register_shim(s2)
        result = get_shim("test")
        assert result is not None
        assert result.base == "anthropic"

    def test_unregister(self):
        s = ProviderShim(name="test", base="openai_chat")
        register_shim(s)
        removed = unregister_shim("test")
        assert removed is s
        assert get_shim("test") is None

    def test_unregister_nonexistent(self):
        assert unregister_shim("nonexistent") is None

    def test_list_shims(self):
        s1 = ProviderShim(name="a", base="openai_chat")
        s2 = ProviderShim(name="b", base="anthropic")
        register_shim(s1)
        register_shim(s2)
        shims = list_shims()
        assert len(shims) == 2
        names = {s.name for s in shims}
        assert names == {"a", "b"}

    def test_list_shims_empty(self):
        assert list_shims() == []


# ---------------------------------------------------------------------------
# resolve_base
# ---------------------------------------------------------------------------


class TestResolveBase:
    def test_base_type_passthrough(self):
        assert resolve_base("openai_chat") == "openai_chat"
        assert resolve_base("anthropic") == "anthropic"
        assert resolve_base("google") == "google"
        assert resolve_base("openai_responses") == "openai_responses"
        assert resolve_base("open_responses") == "open_responses"

    def test_shim_name_resolves(self):
        register_shim(ProviderShim(name="deepseek", base="openai_chat"))
        assert resolve_base("deepseek") == "openai_chat"

    def test_unknown_name_passthrough(self):
        assert resolve_base("unknown") == "unknown"


# ---------------------------------------------------------------------------
# Built-in shims
# ---------------------------------------------------------------------------


class TestBuiltinShims:
    @pytest.fixture(autouse=True)
    def _load_builtins(self):
        """Re-register builtins for this test class."""
        from llm_rosetta.shims.builtins import _register_builtins

        _register_builtins()

    def test_official_providers_registered(self):
        for name in ("openai", "openai_responses", "anthropic", "google"):
            shim = get_shim(name)
            assert shim is not None, f"Built-in shim '{name}' not registered"

    def test_third_party_providers_registered(self):
        for name in ("deepseek", "volcengine"):
            shim = get_shim(name)
            assert shim is not None, f"Built-in shim '{name}' not registered"

    def test_openai_base_type(self):
        shim = get_shim("openai")
        assert shim is not None
        assert shim.base == "openai_chat"

    def test_deepseek_base_type(self):
        shim = get_shim("deepseek")
        assert shim is not None
        assert shim.base == "openai_chat"

    def test_anthropic_base_type(self):
        shim = get_shim("anthropic")
        assert shim is not None
        assert shim.base == "anthropic"

    def test_google_base_type(self):
        shim = get_shim("google")
        assert shim is not None
        assert shim.base == "google"

    def test_openai_has_nested_models(self):
        shim = get_shim("openai")
        assert shim is not None
        assert len(shim.models) > 0
        # o3 should have reasoning capability
        m = shim.get_model_shim("o3-mini")
        assert m is not None
        assert "reasoning" in m.capabilities

    def test_deepseek_has_nested_models(self):
        shim = get_shim("deepseek")
        assert shim is not None
        m = shim.get_model_shim("deepseek-v4-flash")
        assert m is not None
        assert "reasoning" in m.capabilities
        assert "tools" in m.capabilities

    def test_google_model_capabilities(self):
        shim = get_shim("google")
        assert shim is not None
        # gemini-2.5-pro should have reasoning
        m = shim.get_model_shim("gemini-2.5-pro")
        assert m is not None
        assert "reasoning" in m.capabilities
        # gemini-2.0-flash should NOT have reasoning
        m2 = shim.get_model_shim("gemini-2.0-flash")
        assert m2 is not None
        assert "reasoning" not in m2.capabilities


# ---------------------------------------------------------------------------
# Integration: shim → converter
# ---------------------------------------------------------------------------


class TestShimConverterIntegration:
    @pytest.fixture(autouse=True)
    def _load_builtins(self):
        from llm_rosetta.shims.builtins import _register_builtins

        _register_builtins()

    def test_deepseek_resolves_to_openai_chat_converter(self):
        from llm_rosetta.auto_detect import get_converter_for_provider
        from llm_rosetta.converters import OpenAIChatConverter

        converter = get_converter_for_provider("deepseek")
        assert isinstance(converter, OpenAIChatConverter)

    def test_volcengine_resolves_to_openai_chat_converter(self):
        from llm_rosetta.auto_detect import get_converter_for_provider
        from llm_rosetta.converters import OpenAIChatConverter

        converter = get_converter_for_provider("volcengine")
        assert isinstance(converter, OpenAIChatConverter)

    def test_base_types_still_work(self):
        from llm_rosetta.auto_detect import get_converter_for_provider
        from llm_rosetta.converters import (
            AnthropicConverter,
            GoogleConverter,
            OpenAIChatConverter,
        )

        assert isinstance(
            get_converter_for_provider("openai_chat"), OpenAIChatConverter
        )
        assert isinstance(get_converter_for_provider("anthropic"), AnthropicConverter)
        assert isinstance(get_converter_for_provider("google"), GoogleConverter)

    def test_unknown_provider_raises(self):
        from llm_rosetta.auto_detect import get_converter_for_provider

        with pytest.raises(ValueError, match="Unsupported provider"):
            get_converter_for_provider("totally_unknown")
