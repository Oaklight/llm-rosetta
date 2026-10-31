"""Tests for unified provider config with embedding/rerank capabilities."""

from __future__ import annotations


from llm_rosetta_gateway.config import GatewayConfig


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

    def test_embeddings_format_creates_embedding_provider(self):
        raw = _base_config()
        raw["providers"]["openai"]["embeddings_format"] = "openai"
        raw["providers"]["openai"]["embeddings_path"] = "/v1/embeddings"
        cfg = GatewayConfig(raw)
        assert "openai" in cfg.embeddings_providers
        assert cfg.embeddings_providers["openai"]["format"] == "openai"
        assert cfg.embeddings_providers["openai"]["embeddings_path"] == "/v1/embeddings"

    def test_one_entry_serves_chat_and_embeddings_from_the_shims(self):
        """A chat shim plus an embedding shim route both endpoints, one base_url."""
        cfg = GatewayConfig(
            {
                "providers": {
                    "ark": {
                        "api_key": "k",
                        "type": "volcengine--chat_completions",
                        "embeddings_shim": "volcengine--embedding",
                    }
                },
                "models": {
                    "doubao-pro": {"provider": "ark"},
                    "doubao-embed": {"providers": ["ark"], "type": "embeddings"},
                },
            }
        )

        assert cfg.provider_identities["ark"].format == "chat_completions"
        assert cfg.embeddings_providers["ark"]["format"] == "openai"
        assert cfg.embeddings_providers["ark"]["embeddings_path"] == "/embeddings"
        info = cfg.embeddings_provider_infos["ark"]
        assert info.base_url == "https://ark.cn-beijing.volces.com/api/v3"
        assert cfg.resolve_embedding("doubao-embed").provider_name == "ark"

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
        raw["providers"]["openai"]["embeddings_format"] = "openai"
        raw["providers"]["openai"]["rerank_format"] = "jina"
        cfg = GatewayConfig(raw)
        assert cfg.embeddings_providers["openai"]["embeddings_path"] == "/v1/embeddings"
        assert cfg.rerank_providers["openai"]["rerank_path"] == "/v1/rerank"

    def test_provider_info_constructed(self):
        raw = _base_config()
        raw["providers"]["openai"]["embeddings_format"] = "openai"
        cfg = GatewayConfig(raw)
        pinfo = cfg.embeddings_provider_infos["openai"]
        assert pinfo.name == "embeddings:openai"
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

    def test_decisions_format_creates_decision_provider(self):
        raw = _base_config()
        raw["providers"]["typesafe"] = {
            "api_key": "ts-test",
            "base_url": "https://api.typesafe.ai",
            "decisions_format": "systemone",
        }
        raw["models"]["sys-one"] = {"provider": "typesafe", "type": "decisions"}
        cfg = GatewayConfig(raw)
        assert cfg.decisions_providers["typesafe"]["format"] == "systemone"
        assert cfg.decisions_models["sys-one"] == "typesafe"

    def test_decision_default_path_follows_the_format(self):
        """``openai_decisions`` is served at /v1/decisions, not the default."""
        raw = _base_config()
        raw["providers"]["openai"]["decisions_format"] = "openai_decisions"
        cfg = GatewayConfig(raw)
        assert cfg.decisions_providers["openai"]["decisions_path"] == "/v1/decisions"
        pinfo = cfg.decisions_provider_infos["openai"]
        assert pinfo.upstream_url("m") == "https://api.openai.com/v1/decisions"

    def test_decision_explicit_path_wins(self):
        raw = _base_config()
        raw["providers"]["openai"]["decisions_format"] = "openai_decisions"
        raw["providers"]["openai"]["decisions_path"] = "/custom/decisions"
        cfg = GatewayConfig(raw)
        assert (
            cfg.decisions_providers["openai"]["decisions_path"] == "/custom/decisions"
        )


class TestLegacyBackwardCompat:
    """Old separate rerank/embedding/decision provider keys still work."""

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

    def test_legacy_embeddings_providers(self):
        raw = _base_config(
            embeddings_providers={
                "embed-prov": {
                    "api_key": "e-test",
                    "base_url": "https://embed.example.com",
                    "format": "openai",
                }
            },
            embeddings_models={"text-embed": "embed-prov"},
        )
        cfg = GatewayConfig(raw)
        assert "embed-prov" in cfg.embeddings_providers
        assert "text-embed" in cfg.embeddings_models

    def test_legacy_decisions_providers(self):
        raw = _base_config(
            decisions_providers={
                "ts": {
                    "api_key": "ts-test",
                    "base_url": "https://api.typesafe.ai",
                    "format": "systemone",
                }
            },
            decisions_models={"sys-one": "ts"},
        )
        cfg = GatewayConfig(raw)
        assert cfg.decisions_providers["ts"]["format"] == "systemone"
        assert cfg.decisions_models["sys-one"] == "ts"
        assert cfg.decisions_provider_infos["ts"].name == "decisions:ts"

    def test_legacy_decisions_path_follows_the_format(self):
        raw = _base_config(
            decisions_providers={
                "oa": {
                    "api_key": "k",
                    "base_url": "https://api.openai.com",
                    "type": "openai_decisions",
                }
            },
        )
        cfg = GatewayConfig(raw)
        assert cfg.decisions_providers["oa"]["format"] == "openai_decisions"
        assert cfg.decisions_providers["oa"]["decisions_path"] == "/v1/decisions"

    def test_legacy_decisions_shim_supplies_connection_defaults(self):
        """A legacy entry may be just ``decisions_shim``."""
        raw = _base_config(
            decisions_providers={"jev": {"decisions_shim": "typesafe--systemone"}},
        )
        cfg = GatewayConfig(raw)
        assert cfg.decisions_providers["jev"]["format"] == "systemone"
        assert cfg.decisions_providers["jev"]["decisions_path"] == "/v1/systemone"
        assert cfg.decisions_provider_infos["jev"].upstream_url("m") == (
            "https://api.typesafe.ai/v1/systemone"
        )

    def test_legacy_decisions_models_accepts_the_list_form(self):
        """A failover ``providers`` list still routes a typed model."""
        raw = _base_config(
            decisions_providers={
                "ts": {
                    "api_key": "ts-test",
                    "base_url": "https://api.typesafe.ai",
                    "format": "systemone",
                }
            },
            decisions_models={
                "sys-one": {"providers": ["missing", {"name": "ts"}]},
            },
        )
        cfg = GatewayConfig(raw)
        assert cfg.decisions_models["sys-one"] == "ts"

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
        raw["providers"]["openai"]["embeddings_format"] = "openai"
        raw["models"]["text-embed-3"] = {
            "provider": "openai",
            "type": "embeddings",
        }
        cfg = GatewayConfig(raw)
        assert "text-embed-3" in cfg.embeddings_models
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
        assert "gpt-4o" not in cfg.embeddings_models
        assert "gpt-4o" not in cfg.rerank_models

    def test_default_type_is_llm(self):
        raw = _base_config()
        cfg = GatewayConfig(raw)
        assert "gpt-4o" in cfg.models

    def test_resolve_embedding_via_type(self):
        raw = _base_config()
        raw["providers"]["openai"]["embeddings_format"] = "openai"
        raw["providers"]["openai"]["embeddings_path"] = "/v1/embeddings"
        raw["models"]["text-embed-3"] = {
            "provider": "openai",
            "type": "embeddings",
        }
        cfg = GatewayConfig(raw)
        route = cfg.resolve_embedding("text-embed-3")
        assert route.provider_name == "openai"
        assert route.format == "openai"
        assert route.path == "/v1/embeddings"

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
        assert route.path == "/v1/rerank"

    def test_disabled_model_not_distributed(self):
        raw = _base_config()
        raw["providers"]["openai"]["embeddings_format"] = "openai"
        raw["models"]["text-embed-disabled"] = {
            "provider": "openai",
            "type": "embeddings",
            "enabled": False,
        }
        cfg = GatewayConfig(raw)
        assert "text-embed-disabled" not in cfg.embeddings_models
        assert "text-embed-disabled" not in cfg.models


class TestMultiCapProvider:
    """A single provider can support LLM + embedding + rerank."""

    def test_all_three_capabilities(self):
        raw = _base_config()
        raw["providers"]["cohere"] = {
            "api_key": "co-test",
            "base_url": "https://api.cohere.com",
            "type": "openai",
            "embeddings_format": "cohere",
            "embeddings_path": "/v2/embed",
            "rerank_format": "cohere",
            "rerank_path": "/v2/rerank",
        }
        raw["models"].update(
            {
                "command-r": {"provider": "cohere", "capabilities": ["text", "tools"]},
                "embed-v4": {"provider": "cohere", "type": "embeddings"},
                "rerank-v3": {"provider": "cohere", "type": "rerank"},
            }
        )
        cfg = GatewayConfig(raw)
        assert "command-r" in cfg.models
        assert "embed-v4" in cfg.embeddings_models
        assert "rerank-v3" in cfg.rerank_models
        assert "cohere" in cfg.providers
        assert "cohere" in cfg.embeddings_providers
        assert "cohere" in cfg.rerank_providers
