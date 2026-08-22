"""Tests for unified provider config with embedding/rerank capabilities."""

from __future__ import annotations


from llm_rosetta.gateway.config import GatewayConfig


def _base_config(**overrides):
    raw = {
        "providers": {
            "openai": {
                "api_key": "sk-test",
                "base_url": "https://api.openai.com",
                "type": "openai",
            },
        },
        "models": {"gpt-4o": "openai"},
    }
    raw.update(overrides)
    return raw


class TestUnifiedProviderEmbeddingRerank:
    """Providers can declare embedding/rerank capabilities via unified fields."""

    def test_embedding_format_creates_embedding_provider(self):
        raw = _base_config()
        raw["providers"]["openai"]["embedding_format"] = "openai"
        raw["providers"]["openai"]["embedding_path"] = "/v1/embeddings"
        cfg = GatewayConfig(raw)
        assert "openai" in cfg.embedding_providers
        assert cfg.embedding_providers["openai"]["format"] == "openai"
        assert cfg.embedding_providers["openai"]["embedding_path"] == "/v1/embeddings"

    def test_rerank_format_creates_rerank_provider(self):
        raw = _base_config()
        raw["providers"]["jina"] = {
            "api_key": "jina-test",
            "base_url": "https://api.jina.ai",
            "type": "openai",
            "rerank_format": "jina",
            "rerank_path": "/v1/rerank",
        }
        cfg = GatewayConfig(raw)
        assert "jina" in cfg.rerank_providers
        assert cfg.rerank_providers["jina"]["format"] == "jina"

    def test_default_paths(self):
        raw = _base_config()
        raw["providers"]["openai"]["embedding_format"] = "openai"
        raw["providers"]["openai"]["rerank_format"] = "jina"
        cfg = GatewayConfig(raw)
        assert cfg.embedding_providers["openai"]["embedding_path"] == "/v1/embeddings"
        assert cfg.rerank_providers["openai"]["rerank_path"] == "/v1/rerank"

    def test_provider_info_constructed(self):
        raw = _base_config()
        raw["providers"]["openai"]["embedding_format"] = "openai"
        cfg = GatewayConfig(raw)
        pinfo = cfg.embedding_provider_infos["openai"]
        assert pinfo.name == "embedding:openai"
        assert "api.openai.com" in pinfo.base_url

    def test_rerank_provider_info_constructed(self):
        raw = _base_config()
        raw["providers"]["jina"] = {
            "api_key": "jina-test",
            "base_url": "https://api.jina.ai",
            "type": "openai",
            "rerank_format": "jina",
        }
        cfg = GatewayConfig(raw)
        pinfo = cfg.rerank_provider_infos["jina"]
        assert pinfo.name == "rerank:jina"
        assert "api.jina.ai" in pinfo.base_url


class TestLegacyBackwardCompat:
    """Old separate rerank_providers/embedding_providers keys still work."""

    def test_legacy_rerank_providers(self):
        raw = _base_config(
            rerank_providers={
                "jina": {
                    "api_key": "jina-test",
                    "base_url": "https://api.jina.ai",
                    "format": "jina",
                }
            },
            rerank_models={"jina-reranker": "jina"},
        )
        cfg = GatewayConfig(raw)
        assert "jina" in cfg.rerank_providers
        assert "jina-reranker" in cfg.rerank_models

    def test_legacy_embedding_providers(self):
        raw = _base_config(
            embedding_providers={
                "embed-prov": {
                    "api_key": "e-test",
                    "base_url": "https://embed.example.com",
                    "format": "openai",
                }
            },
            embedding_models={"text-embed": "embed-prov"},
        )
        cfg = GatewayConfig(raw)
        assert "embed-prov" in cfg.embedding_providers
        assert "text-embed" in cfg.embedding_models

    def test_unified_takes_precedence_over_legacy(self):
        raw = _base_config(
            rerank_providers={
                "cohere": {
                    "api_key": "old-key",
                    "base_url": "https://old.cohere.com",
                    "format": "voyage",
                }
            },
        )
        raw["providers"]["cohere"] = {
            "api_key": "new-key",
            "base_url": "https://api.cohere.com",
            "type": "openai",
            "rerank_format": "cohere",
            "rerank_path": "/v2/rerank",
        }
        cfg = GatewayConfig(raw)
        assert cfg.rerank_providers["cohere"]["format"] == "cohere"

    def test_legacy_fills_gaps(self):
        raw = _base_config(
            rerank_providers={
                "voyage": {
                    "api_key": "v-test",
                    "base_url": "https://api.voyageai.com",
                    "format": "voyage",
                }
            },
            rerank_models={"rerank-lite": "voyage"},
        )
        cfg = GatewayConfig(raw)
        assert "voyage" in cfg.rerank_providers
        assert "rerank-lite" in cfg.rerank_models


class TestModelTypeField:
    """Models with type=embedding/rerank are routed to specialized pools."""

    def test_embedding_model_moved_to_pool(self):
        raw = _base_config()
        raw["providers"]["openai"]["embedding_format"] = "openai"
        raw["models"]["text-embed-3"] = {
            "provider": "openai",
            "type": "embedding",
        }
        cfg = GatewayConfig(raw)
        assert "text-embed-3" in cfg.embedding_models
        assert "text-embed-3" not in cfg.models

    def test_rerank_model_moved_to_pool(self):
        raw = _base_config()
        raw["providers"]["jina"] = {
            "api_key": "jina-test",
            "base_url": "https://api.jina.ai",
            "type": "openai",
            "rerank_format": "jina",
        }
        raw["models"]["jina-reranker"] = {
            "provider": "jina",
            "type": "rerank",
        }
        cfg = GatewayConfig(raw)
        assert "jina-reranker" in cfg.rerank_models
        assert "jina-reranker" not in cfg.models

    def test_llm_model_stays_in_main_pool(self):
        raw = _base_config()
        raw["models"]["gpt-4o"] = {
            "provider": "openai",
            "type": "llm",
            "capabilities": ["text", "vision"],
        }
        cfg = GatewayConfig(raw)
        assert "gpt-4o" in cfg.models
        assert "gpt-4o" not in cfg.embedding_models
        assert "gpt-4o" not in cfg.rerank_models

    def test_default_type_is_llm(self):
        raw = _base_config()
        cfg = GatewayConfig(raw)
        assert "gpt-4o" in cfg.models

    def test_resolve_embedding_via_type(self):
        raw = _base_config()
        raw["providers"]["openai"]["embedding_format"] = "openai"
        raw["providers"]["openai"]["embedding_path"] = "/v1/embeddings"
        raw["models"]["text-embed-3"] = {
            "provider": "openai",
            "type": "embedding",
        }
        cfg = GatewayConfig(raw)
        route = cfg.resolve_embedding("text-embed-3")
        assert route.provider_name == "openai"
        assert route.format == "openai"
        assert route.embedding_path == "/v1/embeddings"

    def test_resolve_rerank_via_type(self):
        raw = _base_config()
        raw["providers"]["jina"] = {
            "api_key": "jina-test",
            "base_url": "https://api.jina.ai",
            "type": "openai",
            "rerank_format": "jina",
            "rerank_path": "/v1/rerank",
        }
        raw["models"]["jina-reranker"] = {
            "provider": "jina",
            "type": "rerank",
        }
        cfg = GatewayConfig(raw)
        route = cfg.resolve_rerank("jina-reranker")
        assert route.provider_name == "jina"
        assert route.format == "jina"
        assert route.rerank_path == "/v1/rerank"

    def test_disabled_model_not_distributed(self):
        raw = _base_config()
        raw["providers"]["openai"]["embedding_format"] = "openai"
        raw["models"]["text-embed-disabled"] = {
            "provider": "openai",
            "type": "embedding",
            "enabled": False,
        }
        cfg = GatewayConfig(raw)
        assert "text-embed-disabled" not in cfg.embedding_models
        assert "text-embed-disabled" not in cfg.models


class TestMultiCapProvider:
    """A single provider can support LLM + embedding + rerank."""

    def test_all_three_capabilities(self):
        raw = _base_config()
        raw["providers"]["cohere"] = {
            "api_key": "co-test",
            "base_url": "https://api.cohere.com",
            "type": "openai",
            "embedding_format": "cohere",
            "embedding_path": "/v2/embed",
            "rerank_format": "cohere",
            "rerank_path": "/v2/rerank",
        }
        raw["models"].update(
            {
                "command-r": {"provider": "cohere", "capabilities": ["text", "tools"]},
                "embed-v4": {"provider": "cohere", "type": "embedding"},
                "rerank-v3": {"provider": "cohere", "type": "rerank"},
            }
        )
        cfg = GatewayConfig(raw)
        assert "command-r" in cfg.models
        assert "embed-v4" in cfg.embedding_models
        assert "rerank-v3" in cfg.rerank_models
        assert "cohere" in cfg.providers
        assert "cohere" in cfg.embedding_providers
        assert "cohere" in cfg.rerank_providers
