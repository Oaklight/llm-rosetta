"""Tests for the shared legacy provider-name alias + auto-migrate mechanism.

Covers the four boundaries named in #908: the alias helper itself, converter
lookup / request auto-detection, library conversion arguments, and gateway
JSONC config resolution.
"""

from __future__ import annotations

import warnings
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
            ("openai_responses", "openai--open_responses"),
            ("openai-responses", "openai--open_responses"),
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
            "open_responses",
            "openai--open_responses",
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

    def test_every_alias_target_is_a_valid_provider_name(self):
        """The map may only target a base ProviderType or a registered shim.

        Since #909 the Responses aliases target the ``openai--open_responses``
        shim (the OpenAI profile), not the neutral ``open_responses`` base, so
        shim names are legitimate alias targets.
        """
        from llm_rosetta.shims import list_shims

        valid = set(get_args(ProviderType)) | {s.name for s in list_shims()}
        assert set(LEGACY_PROVIDER_ALIASES.values()) <= valid

    def test_no_alias_target_is_a_legacy_key(self):
        """Canonical targets must not themselves be aliases (no chains)."""
        assert not (
            set(LEGACY_PROVIDER_ALIASES.values()) & set(LEGACY_PROVIDER_ALIASES)
        )


class TestConverterLookupBoundary:
    """``get_converter_for_provider`` is an internal lookup.

    It migrates legacy names silently: the public boundary
    (``convert`` / ``convert_response`` / ``ConversionPipeline``) owns the
    deprecation warning, so resolving a converter does not warn again.
    """

    @pytest.mark.parametrize(
        "legacy,canonical",
        [
            ("google", "google_generate"),
            ("google-genai", "google_generate"),
            ("openai-chat", "chat_completions"),
        ],
    )
    def test_legacy_resolves_to_same_converter_without_warning(
        self, legacy: str, canonical: str
    ):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            legacy_converter = get_converter_for_provider(legacy)
        assert legacy_converter is get_converter_for_provider(canonical)

    def test_toolless_unknown_still_raises(self):
        with pytest.raises(ValueError, match="Unsupported provider"):
            get_converter_for_provider("not-a-provider")


class TestResolveBaseBoundary:
    """``resolve_base`` is an internal lookup: legacy names migrate silently."""

    def test_google_legacy_migrates_without_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert resolve_base("google") == "google_generate"

    def test_google_genai_migrates_without_warning(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
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
    """Gateway JSONC ``type:`` / ``shim:`` resolution migrates legacy names.

    Config resolution migrates a legacy spelling and emits one
    ``DeprecationWarning`` per legacy provider, so an operator who writes a
    legacy ``type:`` / ``shim:`` in the JSONC file is nudged to migrate
    before 0.16.0 (#919).
    """

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
    the shim when it was registered only under the other spelling.  Shim
    lookup is internal and must not warn.
    """

    def test_canonical_and_legacy_resolve(self):
        from llm_rosetta.shims import resolve_shim

        assert resolve_shim("google_generate") is not None
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert resolve_shim("google") is not None


class TestBoundaryWarningAttribution:
    """Public boundaries warn exactly once, at the user's frame (#921).

    A legacy name must not produce duplicated warnings, and each warning must
    point at the caller of the public entry point rather than at a library
    internal frame.
    """

    def test_normalize_warns_at_caller_frame(self):
        with pytest.warns(DeprecationWarning) as record:
            normalize_provider_name("google")
        assert record.list[0].filename == __file__

    def test_convert_same_legacy_name_warns_once(self):
        """One distinct legacy name used for both roles warns a single time."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            convert(OPENAI_CHAT_BODY, "openai_chat", source_provider="openai_chat")
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep) == 1
        assert dep[0].filename == __file__

    def test_convert_distinct_legacy_names_warn_once_each(self):
        """Each distinct user-supplied legacy name warns exactly once."""
        body = {
            "contents": [{"role": "user", "parts": [{"text": "hi"}]}],
            "model": "gemini-2.0-flash",
        }
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            convert(body, "openai_chat", source_provider="google")
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep) == 2
        assert {w.filename for w in dep} == {__file__}

    def test_pipeline_warns_once_at_caller_frame(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ConversionPipeline("openai_chat", "anthropic")
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep) == 1
        assert dep[0].filename == __file__

    def test_convert_response_warns_once_at_caller_frame(self):
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
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            convert_response(
                response,
                OPENAI_CHAT_BODY,
                source_provider="openai_chat",
                target_provider="openai_chat",
            )
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep) == 1
        assert dep[0].filename == __file__

    def test_internal_lookups_never_warn(self):
        from llm_rosetta.shims import get_shim

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            get_converter_for_provider("openai_chat")
            resolve_base("google")
            get_shim("google")

    def test_canonical_names_never_warn(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            convert(OPENAI_CHAT_BODY, "chat_completions")
            ConversionPipeline("chat_completions", "anthropic")


class TestRegisterShimKeys:
    """``register_shim`` keys on the exact name, with a free canonical alias."""

    def test_exact_name_is_primary_and_canonical_alias_added_when_free(self):
        import llm_rosetta.shims.provider_shim as ps
        from llm_rosetta.shims import ProviderShim, register_shim

        saved = dict(ps._SHIM_REGISTRY)
        try:
            ps._SHIM_REGISTRY.clear()
            register_shim(ProviderShim(name="google", base="google_generate"))
            assert "google" in ps._SHIM_REGISTRY
            assert "google_generate" in ps._SHIM_REGISTRY
            assert ps.get_shim("google_generate") is ps.get_shim("google")
        finally:
            ps._SHIM_REGISTRY.clear()
            ps._SHIM_REGISTRY.update(saved)

    def test_canonical_shim_not_clobbered_by_legacy_named_shim(self):
        """A shim named like a legacy alias keeps its own identity (#909).

        ``openai_responses`` is an alias key that normalises to the
        ``openai--open_responses`` shim; if a (plugin) shim is registered under
        that exact legacy spelling, it must stay reachable under its exact name
        rather than being clobbered by the canonical alias resolution.
        """
        import llm_rosetta.shims.provider_shim as ps
        from llm_rosetta.shims import ProviderShim, register_shim

        saved = dict(ps._SHIM_REGISTRY)
        try:
            ps._SHIM_REGISTRY.clear()
            neutral = ProviderShim(name="open_responses", base="open_responses")
            profile = ProviderShim(name="openai_responses", base="open_responses")
            register_shim(neutral)
            register_shim(profile)
            assert ps.get_shim("open_responses") is neutral
            assert ps.get_shim("openai_responses") is profile
        finally:
            ps._SHIM_REGISTRY.clear()
            ps._SHIM_REGISTRY.update(saved)


class TestWarnLegacyShimNames:
    """The shim-name helper mirrors :func:`warn_legacy_provider_names` (#923).

    Legacy *shim* names live in a table separate from provider aliases, so the
    warning is emitted by a distinct helper; only names present in
    :data:`LEGACY_SHIM_ALIASES` warn, and each distinct name warns once.
    """

    def test_warns_with_canonical_replacement(self):
        from llm_rosetta.provider_names import warn_legacy_shim_names

        with pytest.warns(DeprecationWarning, match="deepseek--chat_completions"):
            warn_legacy_shim_names("deepseek--openai_chat")

    def test_dedupes_repeated_names_and_ignores_none(self):
        from llm_rosetta.provider_names import warn_legacy_shim_names

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_legacy_shim_names("openai", None, "openai")
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep) == 1

    @pytest.mark.parametrize(
        "name",
        [
            "deepseek--chat_completions",  # canonical shim
            "openai--open_responses",  # canonical shim
            "chat_completions",  # base format
            "deepseek",  # bare provider, not a shim alias
            "nonexistent",
            None,
        ],
    )
    def test_non_legacy_names_stay_silent(self, name):
        from llm_rosetta.provider_names import warn_legacy_shim_names

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            warn_legacy_shim_names(name)


class TestShimBoundaryWarnings:
    """User boundaries warn once per legacy shim name; internals stay silent."""

    @staticmethod
    def _gw(provider_cfg: dict) -> dict:
        return {
            "providers": {
                "g": {"api_key": "k", "base_url": "https://x", **provider_cfg}
            },
            "models": {"m": "g"},
        }

    def test_pipeline_warns_once_at_caller_frame(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ConversionPipeline(
                "chat_completions",
                "chat_completions",
                target_shim="deepseek--openai_chat",
            )
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep) == 1
        assert dep[0].filename == __file__
        assert "deepseek--chat_completions" in str(dep[0].message)

    def test_pipeline_same_legacy_shim_in_both_roles_warns_once(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ConversionPipeline(
                "chat_completions",
                "chat_completions",
                source_shim="deepseek--openai_chat",
                target_shim="deepseek--openai_chat",
            )
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep) == 1
        assert dep[0].filename == __file__

    def test_pipeline_distinct_legacy_shims_warn_once_each(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            ConversionPipeline(
                "chat_completions",
                "chat_completions",
                source_shim="deepseek--openai_chat",
                target_shim="argo--openai_chat",
            )
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep) == 2
        assert {w.filename for w in dep} == {__file__}

    def test_pipeline_canonical_shim_is_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            ConversionPipeline(
                "chat_completions",
                "chat_completions",
                target_shim="deepseek--chat_completions",
            )

    def test_pipeline_shim_object_is_silent(self):
        from llm_rosetta.shims import get_shim

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            ConversionPipeline(
                "chat_completions",
                "chat_completions",
                target_shim=get_shim("deepseek--chat_completions"),
            )

    def test_pipeline_bare_legacy_shim_string_warns(self):
        """A bare legacy shim name (``google_generate``) is legacy in a shim arg."""
        with pytest.warns(DeprecationWarning, match="Shim name"):
            ConversionPipeline(
                "chat_completions",
                "chat_completions",
                target_shim="google_generate",
            )

    def test_gateway_shim_field_warns(self):
        from llm_rosetta.gateway.config import GatewayConfig

        with pytest.warns(DeprecationWarning, match="Shim name"):
            GatewayConfig(self._gw({"shim": "deepseek--openai_chat"}))

    def test_gateway_type_field_warns_for_legacy_shim(self):
        from llm_rosetta.gateway.config import GatewayConfig

        with pytest.warns(DeprecationWarning, match="Shim name"):
            GatewayConfig(self._gw({"type": "deepseek--openai_chat"}))

    def test_gateway_name_fallback_warns_for_legacy_shim(self):
        from llm_rosetta.gateway.config import GatewayConfig

        raw = {
            "providers": {
                "deepseek--openai_chat": {
                    "api_key": "k",
                    "base_url": "https://x",
                }
            },
            "models": {"m": "deepseek--openai_chat"},
        }
        with pytest.warns(DeprecationWarning, match="Shim name"):
            GatewayConfig(raw)

    def test_gateway_bare_shim_field_warns(self):
        from llm_rosetta.gateway.config import GatewayConfig

        with pytest.warns(DeprecationWarning, match="Shim name"):
            GatewayConfig(self._gw({"shim": "google_generate"}))

    def test_gateway_base_format_type_does_not_warn_shim(self):
        """``type: google_generate`` names the base format, not a shim.

        ``google_generate`` doubles as a legacy bare shim name, but in the
        ``type:`` field it is the canonical base format, so a shim warning
        there would be a false positive.
        """
        from llm_rosetta.gateway.config import GatewayConfig

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            GatewayConfig(self._gw({"type": "google_generate"}))
        shim_warnings = [w for w in caught if "Shim name" in str(w.message)]
        assert not shim_warnings

    def test_internal_shim_lookups_never_warn(self):
        from llm_rosetta.shims import get_shim, resolve_shim

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            get_shim("deepseek--openai_chat")
            get_shim("openai")
            resolve_base("deepseek--openai_chat")
            resolve_shim("deepseek--openai_chat")


def _chat_response() -> dict:
    return {
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


class TestConvertBoundaryShimWarnings:
    """``convert`` / ``convert_response`` warn for legacy *shim* names (#923).

    These boundaries resolve a shim name to a :class:`ProviderShim` object
    before the pipeline's string check, so the shim warning must be emitted at
    the boundary itself; base formats and provider aliases must not trip it.
    """

    def test_convert_legacy_shim_target_warns_once(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            convert(OPENAI_CHAT_BODY, "deepseek--openai_chat")
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep) == 1
        assert dep[0].filename == __file__
        assert "deepseek--chat_completions" in str(dep[0].message)

    def test_convert_legacy_shim_source_warns_once(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            convert(
                OPENAI_CHAT_BODY,
                "chat_completions",
                source_provider="deepseek--openai_chat",
            )
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep) == 1
        assert dep[0].filename == __file__

    def test_convert_same_legacy_shim_in_both_roles_warns_once(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            convert(
                OPENAI_CHAT_BODY,
                "deepseek--openai_chat",
                source_provider="deepseek--openai_chat",
            )
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep) == 1
        assert dep[0].filename == __file__

    def test_convert_bare_legacy_shim_warns(self):
        with pytest.warns(DeprecationWarning, match="openai--chat_completions"):
            convert(OPENAI_CHAT_BODY, "openai")

    def test_convert_base_format_does_not_warn_shim(self):
        """``google_generate`` is a base format, not the legacy shim spelling."""
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            convert(OPENAI_CHAT_BODY, "google_generate")

    def test_convert_provider_alias_warns_provider_not_shim(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            convert(OPENAI_CHAT_BODY, "openai_chat")
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep) == 1
        assert "Provider name" in str(dep[0].message)

    def test_convert_canonical_names_are_silent(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            convert(OPENAI_CHAT_BODY, "deepseek--chat_completions")

    def test_convert_response_legacy_shim_warns_once(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            convert_response(
                _chat_response(),
                OPENAI_CHAT_BODY,
                source_provider="deepseek--openai_chat",
                target_provider="deepseek--openai_chat",
            )
        dep = [w for w in caught if issubclass(w.category, DeprecationWarning)]
        assert len(dep) == 1
        assert dep[0].filename == __file__


class TestLegacyShimAliasStillResolves:
    """Normalisation must keep legacy shim spellings reachable and silent."""

    @pytest.mark.parametrize(
        "legacy,canonical",
        [
            ("deepseek--openai_chat", "deepseek--chat_completions"),
            ("argo--openai_chat", "argo--chat_completions"),
            ("openai", "openai--chat_completions"),
            ("google_generate", "google--google_generate"),
        ],
    )
    def test_get_shim_legacy_is_canonical(self, legacy: str, canonical: str):
        from llm_rosetta.shims import get_shim

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            legacy_shim = get_shim(legacy)
            canonical_shim = get_shim(canonical)
        assert legacy_shim is not None
        assert legacy_shim is canonical_shim
