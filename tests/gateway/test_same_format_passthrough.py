"""Tests for same-format provider affinity and passthrough opt-in.

Covers ``server.prefer_same_format``: when a model is served by providers
in more than one API standard, prefer the one that already speaks the
client's format and skip the IR round-trip for it.
"""

from __future__ import annotations

import pytest

from llm_rosetta.gateway.config import GatewayConfig

PROVIDERS = {
    "chat_a": {
        "type": "openai_chat",
        "api_key": "sk-a",
        "base_url": "https://a.example.com",
    },
    "chat_b": {
        "type": "openai_chat",
        "api_key": "sk-b",
        "base_url": "https://b.example.com",
    },
    "resp_a": {
        "type": "openai_responses",
        "api_key": "sk-c",
        "base_url": "https://c.example.com",
    },
    "resp_b": {
        "type": "openai_responses",
        "api_key": "sk-d",
        "base_url": "https://d.example.com",
    },
    "claude": {
        "type": "anthropic",
        "api_key": "sk-e",
        "base_url": "https://e.example.com",
    },
}


def _make_config(
    models: dict,
    *,
    prefer_same_format: bool | None = None,
    providers: dict | None = None,
) -> GatewayConfig:
    """Build a GatewayConfig, optionally enabling same-format affinity."""
    server: dict = {}
    if prefer_same_format is not None:
        server["prefer_same_format"] = prefer_same_format
    return GatewayConfig(
        {
            "providers": providers if providers is not None else PROVIDERS,
            "models": models,
            "server": server,
        }
    )


class TestDefaultsUnchanged:
    """With the flag absent or off, nothing about routing changes."""

    def test_flag_defaults_to_off(self):
        config = _make_config({"m": "chat_a"})
        assert config.prefer_same_format is False

    def test_same_format_still_converts_when_off(self):
        """The historical behaviour: same format, full IR round-trip."""
        config = _make_config({"m": "chat_a"})
        route, _ = config.resolve("openai_chat", "m")
        assert route.target_provider == "openai_chat"
        assert route.force_conversion is True

    def test_mixed_route_ignores_format_when_off(self):
        """Without the flag, a responses client still lands on WRR's pick."""
        config = _make_config({"m": {"providers": ["chat_a", "resp_a"]}})
        picked = {
            config.resolve("openai_responses", "m")[0].provider_name for _ in range(10)
        }
        assert picked == {"chat_a", "resp_a"}

    def test_explicit_false_is_off(self):
        config = _make_config({"m": "chat_a"}, prefer_same_format=False)
        assert config.prefer_same_format is False
        route, _ = config.resolve("openai_chat", "m")
        assert route.force_conversion is True


class TestPassthroughOptIn:
    """``force_conversion`` is the pipeline's passthrough switch."""

    def test_same_format_passes_through(self):
        config = _make_config({"m": "chat_a"}, prefer_same_format=True)
        route, _ = config.resolve("openai_chat", "m")
        assert route.source_provider == route.target_provider == "openai_chat"
        assert route.force_conversion is False

    def test_cross_format_still_converts(self):
        config = _make_config({"m": "claude"}, prefer_same_format=True)
        route, _ = config.resolve("openai_chat", "m")
        assert route.target_provider == "anthropic"
        assert route.force_conversion is True

    def test_responses_client_to_responses_provider(self):
        config = _make_config({"m": "resp_a"}, prefer_same_format=True)
        route, _ = config.resolve("openai_responses", "m")
        assert route.force_conversion is False

    def test_chat_client_to_responses_provider_converts(self):
        """Same vendor, different API standard — still a conversion."""
        config = _make_config({"m": "resp_a"}, prefer_same_format=True)
        route, _ = config.resolve("openai_chat", "m")
        assert route.force_conversion is True


class TestFormatAffinity:
    """Provider selection prefers the client's own format."""

    def test_picks_matching_format_over_wrr(self):
        config = _make_config(
            {"m": {"providers": ["chat_a", "resp_a"]}}, prefer_same_format=True
        )
        for _ in range(10):
            route, _ = config.resolve("openai_responses", "m")
            assert route.provider_name == "resp_a"
            assert route.force_conversion is False

    def test_affinity_is_per_source_format(self):
        """The same model routes differently for two client formats."""
        config = _make_config(
            {"m": {"providers": ["chat_a", "resp_a"]}}, prefer_same_format=True
        )
        assert config.resolve("openai_responses", "m")[0].provider_name == "resp_a"
        assert config.resolve("openai_chat", "m")[0].provider_name == "chat_a"

    def test_no_match_falls_back_to_full_route(self):
        """An Anthropic client has no affine provider here."""
        config = _make_config(
            {"m": {"providers": ["chat_a", "resp_a"]}}, prefer_same_format=True
        )
        picked = {config.resolve("anthropic", "m")[0].provider_name for _ in range(10)}
        assert picked == {"chat_a", "resp_a"}

    def test_homogeneous_route_needs_no_subroute(self):
        """All providers already match — reuse the route, keep its WRR state."""
        config = _make_config(
            {"m": {"providers": ["chat_a", "chat_b"]}}, prefer_same_format=True
        )
        picked = [
            config.resolve("openai_chat", "m")[0].provider_name for _ in range(10)
        ]
        assert set(picked) == {"chat_a", "chat_b"}
        assert config._same_format_routes[("m", "openai_chat")] is None
        # Passthrough still applies — affinity was unnecessary, not absent.
        assert config.resolve("openai_chat", "m")[0].force_conversion is False

    def test_load_balances_within_matched_subset(self):
        """Two same-format providers still share load, excluding the third."""
        config = _make_config(
            {"m": {"providers": ["chat_a", "resp_a", "resp_b"]}},
            prefer_same_format=True,
        )
        picked = [
            config.resolve("openai_responses", "m")[0].provider_name for _ in range(20)
        ]
        assert set(picked) == {"resp_a", "resp_b"}
        # Weighted round-robin alternates rather than pinning one provider;
        # a reset sub-route strategy would return resp_a every time.
        assert picked.count("resp_a") == picked.count("resp_b") == 10

    def test_weights_respected_within_subset(self):
        config = _make_config(
            {
                "m": {
                    "providers": [
                        "chat_a",
                        {"name": "resp_a", "weight": 3},
                        {"name": "resp_b", "weight": 1},
                    ]
                }
            },
            prefer_same_format=True,
        )
        picked = [
            config.resolve("openai_responses", "m")[0].provider_name for _ in range(20)
        ]
        assert picked.count("resp_a") == 15
        assert picked.count("resp_b") == 5

    def test_subroute_is_cached(self):
        config = _make_config(
            {"m": {"providers": ["chat_a", "resp_a"]}}, prefer_same_format=True
        )
        config.resolve("openai_responses", "m")
        first = config._same_format_routes[("m", "openai_responses")]
        config.resolve("openai_responses", "m")
        assert config._same_format_routes[("m", "openai_responses")] is first

    def test_upstream_model_comes_from_matched_entry(self):
        """Per-entry upstream_model follows the affine provider, not WRR."""
        config = _make_config(
            {
                "m": {
                    "providers": [
                        {"name": "chat_a", "upstream_model": "chat-id"},
                        {"name": "resp_a", "upstream_model": "resp-id"},
                    ]
                }
            },
            prefer_same_format=True,
        )
        route, _ = config.resolve("openai_responses", "m")
        assert route.upstream_model == "resp-id"


class TestReadinessInteraction:
    """Affinity must not strand a request on an initializing provider."""

    def _unready(self, config: GatewayConfig, *names: str) -> None:
        """Swap providers for ones whose deferred token fetch is pending."""
        from llm_rosetta.gateway.transport.provider_info import (
            TOKEN_PENDING_SENTINEL,
            ProviderInfo,
            openai_auth,
        )

        for name in names:
            config.providers[name] = ProviderInfo(
                name=name,
                api_key=TOKEN_PENDING_SENTINEL,
                base_url=config.providers[name].base_url,
                auth_header_fn=openai_auth,
                url_template="{base_url}/chat/completions",
                token_command=["echo", "x"],
            )
            assert config.providers[name].ready is False

    def test_unready_affine_falls_back_to_ready_other_format(self):
        config = _make_config(
            {"m": {"providers": ["chat_a", "resp_a"]}}, prefer_same_format=True
        )
        self._unready(config, "resp_a")
        route, _ = config.resolve("openai_responses", "m")
        assert route.provider_name == "chat_a"
        # Fell back across formats, so the conversion must run again.
        assert route.force_conversion is True

    def test_unready_affine_prefers_ready_affine(self):
        config = _make_config(
            {"m": {"providers": ["chat_a", "resp_a", "resp_b"]}},
            prefer_same_format=True,
        )
        self._unready(config, "resp_a")
        for _ in range(5):
            route, _ = config.resolve("openai_responses", "m")
            assert route.provider_name == "resp_b"
            assert route.force_conversion is False

    def test_all_unready_still_raises(self):
        from llm_rosetta.gateway.deferred_startup import ProviderNotReady

        config = _make_config(
            {"m": {"providers": ["chat_a", "resp_a"]}}, prefer_same_format=True
        )
        self._unready(config, "chat_a", "resp_a")
        with pytest.raises(ProviderNotReady):
            config.resolve("openai_responses", "m")


class TestPipelineReachesPassthrough:
    """The route field is only useful if the pipeline actually gets it.

    These mirror how ``gateway/proxy.py`` builds its pipeline, so the
    wiring is covered rather than just the flag.
    """

    @staticmethod
    def _pipeline_for(route):
        from llm_rosetta.pipeline import ConversionPipeline

        return ConversionPipeline(
            route.source_provider,
            route.target_provider,
            target_shim=route.shim_name,
            upstream_model="m",
            model_capabilities=route.model_capabilities,
            reasoning_config_override=route.reasoning_override,
            supports_custom_tools=route.supports_custom_tools,
            max_tool_description_length=route.max_tool_description_length,
            hoist_system_messages=route.hoist_system_messages,
            force_conversion=route.force_conversion,
        )

    def test_opted_in_route_yields_passthrough_pipeline(self):
        config = _make_config({"m": "resp_a"}, prefer_same_format=True)
        route, _ = config.resolve("openai_responses", "m")
        assert self._pipeline_for(route)._passthrough is True

    def test_default_route_yields_converting_pipeline(self):
        config = _make_config({"m": "resp_a"})
        route, _ = config.resolve("openai_responses", "m")
        assert self._pipeline_for(route)._passthrough is False

    def test_passthrough_leaves_body_untouched(self):
        """The whole point: an unsupported field survives to the upstream."""
        config = _make_config({"m": "resp_a"}, prefer_same_format=True)
        route, _ = config.resolve("openai_responses", "m")
        body = {
            "model": "m",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "some_future_field": {"nested": True},
        }
        assert self._pipeline_for(route).convert_request(body) == body

    def test_conversion_drops_unknown_field(self):
        """Contrast: the IR round-trip is what loses it."""
        config = _make_config({"m": "resp_a"})
        route, _ = config.resolve("openai_responses", "m")
        body = {
            "model": "m",
            "input": [{"type": "message", "role": "user", "content": "hi"}],
            "some_future_field": {"nested": True},
        }
        assert "some_future_field" not in self._pipeline_for(route).convert_request(
            body
        )

    def test_passthrough_stream_processor_is_selected(self):
        from llm_rosetta.pipeline import PassthroughStreamProcessor

        config = _make_config({"m": "resp_a"}, prefer_same_format=True)
        route, _ = config.resolve("openai_responses", "m")
        pipeline = self._pipeline_for(route)
        pipeline.convert_request(
            {"model": "m", "input": [{"type": "message", "role": "user"}]}
        )
        assert isinstance(
            pipeline.create_stream_processor(), PassthroughStreamProcessor
        )


class TestRouteContract:
    """``ResolvedRoute.force_conversion`` defaults to the safe value."""

    def test_dataclass_default_is_true(self):
        from llm_rosetta.routing import ResolvedRoute

        route = ResolvedRoute(
            source_provider="openai_chat",
            target_provider="openai_chat",
            provider_name="p",
        )
        assert route.force_conversion is True
