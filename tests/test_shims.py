"""Tests for the provider shim layer."""

from __future__ import annotations

import pytest

from llm_rosetta.shims.provider_shim import (
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
# ProviderShim
# ---------------------------------------------------------------------------


class TestProviderShim:
    def test_creation_minimal(self):
        s = ProviderShim(name="test", base="openai_chat")
        assert s.name == "test"
        assert s.base == "openai_chat"
        assert s.default_base_url is None
        assert s.default_api_key_env is None
        assert s.logo is None

    def test_creation_full(self):
        s = ProviderShim(
            name="openai",
            base="openai_chat",
            default_base_url="https://api.openai.com/v1",
            default_api_key_env="OPENAI_API_KEY",
            logo="https://example.com/openai.svg",
        )
        assert s.default_base_url == "https://api.openai.com/v1"
        assert s.logo == "https://example.com/openai.svg"

    def test_frozen(self):
        s = ProviderShim(name="test", base="openai_chat")
        with pytest.raises(AttributeError):
            s.name = "other"  # type: ignore


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
# Built-in shims (loaded from providers/ directory)
# ---------------------------------------------------------------------------


class TestBuiltinShims:
    @pytest.fixture(autouse=True)
    def _load_builtins(self):
        """Load provider shims from the YAML directory."""
        from llm_rosetta.shims.providers import load_providers

        load_providers()

    def test_official_providers_registered(self):
        for name in ("openai", "openai_responses", "anthropic", "google"):
            shim = get_shim(name)
            assert shim is not None, f"Built-in shim '{name}' not registered"

    def test_third_party_providers_registered(self):
        for name in ("deepseek", "volcengine", "openrouter"):
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


# ---------------------------------------------------------------------------
# Integration: shim → converter
# ---------------------------------------------------------------------------


class TestShimConverterIntegration:
    @pytest.fixture(autouse=True)
    def _load_builtins(self):
        from llm_rosetta.shims.providers import load_providers

        load_providers()

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


# ---------------------------------------------------------------------------
# Grouped provider directories (e.g. argo/anthropic/, argo/openai_chat/)
# ---------------------------------------------------------------------------


class TestGroupedProviders:
    @pytest.fixture(autouse=True)
    def _load_builtins(self):
        """Load provider shims from the YAML directory."""
        from llm_rosetta.shims.providers import load_providers

        load_providers()

    def test_grouped_providers_registered(self):
        """Shims under a group folder register with their YAML name."""
        for name in ("argo_anthropic", "argo_openai_chat"):
            shim = get_shim(name)
            assert shim is not None, f"Grouped shim '{name}' not registered"

    def test_grouped_provider_base_types(self):
        """Grouped shims resolve to the correct base converter."""
        anth = get_shim("argo_anthropic")
        oai = get_shim("argo_openai_chat")
        assert anth is not None and anth.base == "anthropic"
        assert oai is not None and oai.base == "openai_chat"

    def test_grouped_provider_transforms_loaded(self):
        """Grouped shims have their transforms.py imported."""
        anth = get_shim("argo_anthropic")
        assert anth is not None
        # argo_anthropic has to_transforms and from_transforms
        assert len(anth.to_transforms) > 0
        assert len(anth.from_transforms) > 0

    def test_grouped_provider_reasoning_configs_loaded(self):
        """Grouped shims load reasoning capability configs."""
        anth = get_shim("argo_anthropic")
        oai = get_shim("argo_openai_chat")
        assert anth is not None and anth.reasoning is not None
        assert oai is not None and oai.reasoning is not None
        assert anth.reasoning.effort_field == "output_config.effort"
        assert anth.reasoning.effort_map["xhigh"] == "xhigh"
        assert oai.reasoning.max_effort == "high"

    def test_mixed_flat_and_grouped(self):
        """Flat shims and grouped shims coexist in the registry."""
        flat_names = ("openai", "anthropic", "deepseek", "google")
        grouped_names = ("argo_anthropic", "argo_openai_chat")
        for name in (*flat_names, *grouped_names):
            assert get_shim(name) is not None, f"Shim '{name}' not found"
