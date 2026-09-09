"""Tests for multi-provider model config parsing."""

from __future__ import annotations

import pytest

from llm_rosetta.gateway.config import GatewayConfig
from llm_rosetta.gateway.routing_strategy import ModelRoute


def _make_config(models: dict, providers: dict | None = None) -> GatewayConfig:
    """Build a GatewayConfig from a minimal raw dict."""
    if providers is None:
        providers = {
            "openai_a": {
                "type": "openai_chat",
                "api_key": "sk-a",
                "base_url": "https://a.example.com",
            },
            "openai_b": {
                "type": "openai_chat",
                "api_key": "sk-b",
                "base_url": "https://b.example.com",
            },
            "anthropic_a": {
                "type": "anthropic",
                "api_key": "sk-c",
                "base_url": "https://c.example.com",
            },
        }
    return GatewayConfig({"providers": providers, "models": models})


class TestSingleProviderBackwardCompat:
    """Existing single-provider config forms still work."""

    def test_string_form(self):
        config = _make_config({"gpt-4o": "openai_a"})
        route = config.models["gpt-4o"]
        assert isinstance(route, ModelRoute)
        assert not route.is_multi
        assert route.provider_names == ["openai_a"]
        assert route.select() == "openai_a"

    def test_dict_form(self):
        config = _make_config({"gpt-4o": {"provider": "openai_a"}})
        route = config.models["gpt-4o"]
        assert not route.is_multi
        assert route.select() == "openai_a"

    def test_dict_with_capabilities(self):
        config = _make_config(
            {"gpt-4o": {"provider": "openai_a", "capabilities": ["text", "vision"]}}
        )
        assert config.model_capabilities["gpt-4o"] == ["text", "vision"]

    def test_dict_with_upstream_model(self):
        config = _make_config(
            {"gpt-4o": {"provider": "openai_a", "upstream_model": "gpt-4o-2024-08-06"}}
        )
        assert config.model_upstream_names["gpt-4o"] == "gpt-4o-2024-08-06"


class TestMultiProviderParsing:
    """New multi-provider config forms."""

    def test_providers_string_list(self):
        config = _make_config({"gpt-4o": {"providers": ["openai_a", "openai_b"]}})
        route = config.models["gpt-4o"]
        assert route.is_multi
        assert set(route.provider_names) == {"openai_a", "openai_b"}
        # Equal weights
        for p in route.providers:
            assert p.weight == 1

    def test_providers_dict_list_with_weights(self):
        config = _make_config(
            {
                "gpt-4o": {
                    "providers": [
                        {"name": "openai_a", "weight": 3},
                        {"name": "openai_b", "weight": 1},
                    ],
                }
            }
        )
        route = config.models["gpt-4o"]
        assert route.is_multi
        assert route.providers[0].name == "openai_a"
        assert route.providers[0].weight == 3
        assert route.providers[1].name == "openai_b"
        assert route.providers[1].weight == 1

    def test_providers_mixed_list(self):
        config = _make_config(
            {
                "gpt-4o": {
                    "providers": [
                        "openai_a",
                        {"name": "openai_b", "weight": 2},
                    ],
                }
            }
        )
        route = config.models["gpt-4o"]
        assert route.providers[0].name == "openai_a"
        assert route.providers[0].weight == 1
        assert route.providers[1].name == "openai_b"
        assert route.providers[1].weight == 2

    def test_providers_with_strategy(self):
        config = _make_config(
            {
                "gpt-4o": {
                    "providers": ["openai_a", "openai_b"],
                    "strategy": "weighted_round_robin",
                }
            }
        )
        route = config.models["gpt-4o"]
        assert route.is_multi

    def test_providers_with_capabilities(self):
        config = _make_config(
            {
                "gpt-4o": {
                    "providers": ["openai_a", "openai_b"],
                    "capabilities": ["text", "vision"],
                }
            }
        )
        assert config.model_capabilities["gpt-4o"] == ["text", "vision"]

    def test_provider_skipped_if_not_in_providers(self):
        config = _make_config(
            {
                "gpt-4o": {
                    "providers": ["openai_a", "nonexistent"],
                }
            }
        )
        route = config.models["gpt-4o"]
        assert route.provider_names == ["openai_a"]

    def test_model_skipped_if_all_providers_missing(self):
        config = _make_config(
            {"gpt-4o": {"providers": ["nonexistent_a", "nonexistent_b"]}}
        )
        assert "gpt-4o" not in config.models


class TestMultiProviderValidation:
    """Config validation for multi-provider."""

    def test_provider_and_providers_mutually_exclusive(self):
        with pytest.raises(ValueError, match="both 'provider' and 'providers'"):
            _make_config(
                {"gpt-4o": {"provider": "openai_a", "providers": ["openai_b"]}}
            )

    def test_empty_providers_list_raises(self):
        with pytest.raises(ValueError, match="empty or invalid"):
            _make_config({"gpt-4o": {"providers": []}})

    def test_provider_entry_missing_name_raises(self):
        with pytest.raises(ValueError, match="without a 'name' field"):
            _make_config({"gpt-4o": {"providers": [{"weight": 3}]}})

    def test_invalid_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown routing strategy"):
            _make_config(
                {
                    "gpt-4o": {
                        "providers": ["openai_a"],
                        "strategy": "does_not_exist",
                    }
                }
            )


class TestMultiProviderResolve:
    """resolve() works with multi-provider models."""

    def test_resolve_returns_route(self):
        config = _make_config(
            {
                "gpt-4o": {
                    "providers": [
                        {"name": "openai_a", "weight": 3},
                        {"name": "openai_b", "weight": 1},
                    ],
                }
            }
        )
        route, pinfo = config.resolve("openai_chat", "gpt-4o")
        assert route.provider_name in ("openai_a", "openai_b")
        assert pinfo.name in ("openai_chat", "openai_chat")

    def test_resolve_rotates_providers(self):
        config = _make_config({"gpt-4o": {"providers": ["openai_a", "openai_b"]}})
        providers_seen = set()
        for _ in range(10):
            route, _ = config.resolve("openai_chat", "gpt-4o")
            providers_seen.add(route.provider_name)
        assert providers_seen == {"openai_a", "openai_b"}

    def test_resolve_weighted(self):
        config = _make_config(
            {
                "gpt-4o": {
                    "providers": [
                        {"name": "openai_a", "weight": 3},
                        {"name": "openai_b", "weight": 1},
                    ],
                }
            }
        )
        results = []
        for _ in range(4):
            route, _ = config.resolve("openai_chat", "gpt-4o")
            results.append(route.provider_name)
        assert results.count("openai_a") == 3
        assert results.count("openai_b") == 1

    def test_zero_weight_raises(self):
        with pytest.raises(ValueError, match="invalid weight 0"):
            _make_config({"gpt-4o": {"providers": [{"name": "openai_a", "weight": 0}]}})

    def test_negative_weight_raises(self):
        with pytest.raises(ValueError, match="invalid weight -1"):
            _make_config(
                {"gpt-4o": {"providers": [{"name": "openai_a", "weight": -1}]}}
            )
