"""Tests for the config migration framework."""

from __future__ import annotations

import json


from llm_rosetta.gateway.migrations import CURRENT_VERSION, migrate


class TestMigrationFramework:
    """Core migration framework behavior."""

    def test_no_version_treated_as_v0(self):
        raw = {"providers": {}, "models": {}}
        _, changed = migrate(raw)
        assert changed
        assert raw["config_version"] == CURRENT_VERSION

    def test_current_version_is_noop(self):
        raw = {"config_version": CURRENT_VERSION, "providers": {}, "models": {}}
        _, changed = migrate(raw)
        assert not changed

    def test_idempotent(self):
        raw = {
            "providers": {
                "openai": {"api_key": "sk-test", "base_url": "https://api.openai.com"}
            },
            "models": {"gpt-4o": "openai"},
            "rerank_providers": {
                "jina": {
                    "api_key": "j-test",
                    "base_url": "https://api.jina.ai",
                    "format": "jina",
                }
            },
            "rerank_models": {"jina-reranker": "jina"},
        }
        migrate(raw)
        snapshot = json.dumps(raw, sort_keys=True)
        migrate(raw)
        assert json.dumps(raw, sort_keys=True) == snapshot


class TestMigrationV0ToV1:
    """Migration 001: merge legacy rerank/embedding into unified providers."""

    def test_rerank_provider_merged_onto_existing(self):
        raw = {
            "providers": {
                "jina": {"api_key": "j-test", "base_url": "https://api.jina.ai"}
            },
            "models": {},
            "rerank_providers": {
                "jina": {
                    "api_key": "j-old",
                    "base_url": "https://old.jina.ai",
                    "format": "jina",
                    "rerank_path": "/v1/rerank",
                }
            },
        }
        migrate(raw)
        prov = raw["providers"]["jina"]
        assert prov["rerank_format"] == "jina"
        assert prov["rerank_path"] == "/v1/rerank"
        assert prov["api_key"] == "j-test"
        assert "rerank_providers" not in raw

    def test_rerank_provider_created_if_new(self):
        raw = {
            "providers": {},
            "models": {},
            "rerank_providers": {
                "jina": {
                    "api_key": "j-test",
                    "base_url": "https://api.jina.ai",
                    "format": "jina",
                }
            },
        }
        migrate(raw)
        assert "jina" in raw["providers"]
        assert raw["providers"]["jina"]["rerank_format"] == "jina"
        assert raw["providers"]["jina"]["api_key"] == "j-test"

    def test_embedding_provider_merged(self):
        raw = {
            "providers": {
                "openai": {"api_key": "sk-test", "base_url": "https://api.openai.com"}
            },
            "models": {},
            "embedding_providers": {
                "openai": {
                    "api_key": "sk-old",
                    "base_url": "https://old.openai.com",
                    "format": "openai",
                    "embedding_path": "/v1/embeddings",
                }
            },
        }
        migrate(raw)
        prov = raw["providers"]["openai"]
        assert prov["embedding_format"] == "openai"
        assert prov["embedding_path"] == "/v1/embeddings"
        assert prov["api_key"] == "sk-test"

    def test_rerank_models_migrated(self):
        raw = {
            "providers": {
                "jina": {"api_key": "j-test", "base_url": "https://api.jina.ai"}
            },
            "models": {"gpt-4o": "openai"},
            "rerank_providers": {"jina": {"format": "jina"}},
            "rerank_models": {
                "jina-reranker": "jina",
                "reranker-dict": {"provider": "jina"},
            },
        }
        migrate(raw)
        assert raw["models"]["jina-reranker"] == {"provider": "jina", "type": "rerank"}
        assert raw["models"]["reranker-dict"] == {"provider": "jina", "type": "rerank"}
        assert "rerank_models" not in raw

    def test_embedding_models_migrated(self):
        raw = {
            "providers": {},
            "models": {},
            "embedding_providers": {"openai": {"format": "openai"}},
            "embedding_models": {"text-embed-3": "openai"},
        }
        migrate(raw)
        assert raw["models"]["text-embed-3"] == {
            "provider": "openai",
            "type": "embedding",
        }

    def test_existing_model_not_overwritten(self):
        raw = {
            "providers": {},
            "models": {
                "text-embed-3": {
                    "provider": "openai",
                    "type": "embedding",
                    "custom": True,
                }
            },
            "embedding_models": {"text-embed-3": "openai"},
        }
        migrate(raw)
        assert raw["models"]["text-embed-3"]["custom"] is True

    def test_legacy_keys_removed(self):
        raw = {
            "providers": {},
            "models": {},
            "rerank_providers": {},
            "rerank_models": {},
            "default_rerank_format": "jina",
            "embedding_providers": {},
            "embedding_models": {},
            "default_embedding_format": "openai",
        }
        migrate(raw)
        for key in (
            "rerank_providers",
            "rerank_models",
            "default_rerank_format",
            "embedding_providers",
            "embedding_models",
            "default_embedding_format",
        ):
            assert key not in raw

    def test_env_var_placeholders_preserved(self):
        raw = {
            "providers": {},
            "models": {},
            "rerank_providers": {
                "jina": {
                    "api_key": "${JINA_API_KEY}",
                    "base_url": "https://api.jina.ai",
                    "format": "jina",
                }
            },
            "rerank_models": {"jina-reranker": "jina"},
        }
        migrate(raw)
        assert raw["providers"]["jina"]["api_key"] == "${JINA_API_KEY}"

    def test_disabled_provider_preserved(self):
        raw = {
            "providers": {},
            "models": {},
            "rerank_providers": {
                "jina": {
                    "api_key": "j-test",
                    "base_url": "https://api.jina.ai",
                    "format": "jina",
                    "enabled": False,
                }
            },
        }
        migrate(raw)
        assert raw["providers"]["jina"]["enabled"] is False

    def test_full_migration_roundtrip(self):
        """Migrated config should parse correctly in GatewayConfig."""
        from llm_rosetta.gateway.config import GatewayConfig

        raw = {
            "providers": {
                "openai": {
                    "api_key": "sk-test",
                    "base_url": "https://api.openai.com",
                    "type": "openai",
                }
            },
            "models": {"gpt-4o": "openai"},
            "rerank_providers": {
                "jina": {
                    "api_key": "j-test",
                    "base_url": "https://api.jina.ai",
                    "format": "jina",
                }
            },
            "rerank_models": {"jina-reranker": "jina"},
            "embedding_providers": {
                "openai": {
                    "api_key": "sk-test",
                    "base_url": "https://api.openai.com",
                    "format": "openai",
                }
            },
            "embedding_models": {"text-embed-3": "openai"},
        }
        migrate(raw)
        cfg = GatewayConfig(raw)
        assert "jina-reranker" in cfg.rerank_models
        assert "text-embed-3" in cfg.embedding_models
        route = cfg.resolve_rerank("jina-reranker")
        assert route.provider_name == "jina"


class TestConfigIOIntegration:
    """JsoncConfigIO applies migrations on load and writes back."""

    def test_load_migrates_and_writes_back(self, tmp_path):
        config_path = str(tmp_path / "config.jsonc")
        old_config = {
            "providers": {
                "openai": {"api_key": "sk-test", "base_url": "https://api.openai.com"}
            },
            "models": {"gpt-4o": "openai"},
            "rerank_providers": {
                "jina": {
                    "api_key": "j-test",
                    "base_url": "https://api.jina.ai",
                    "format": "jina",
                }
            },
            "rerank_models": {"jina-reranker": "jina"},
        }
        with open(config_path, "w") as f:
            json.dump(old_config, f)

        from llm_rosetta.gateway.config import JsoncConfigIO

        io = JsoncConfigIO()
        raw = io.load(config_path)

        assert raw["config_version"] == CURRENT_VERSION
        assert "rerank_providers" not in raw
        assert "jina" in raw["providers"]
        assert raw["providers"]["jina"]["rerank_format"] == "jina"

        # Verify written back to disk
        with open(config_path) as f:
            on_disk = json.load(f)
        assert on_disk["config_version"] == CURRENT_VERSION
        assert "rerank_providers" not in on_disk

    def test_load_raw_migrates_preserving_env_vars(self, tmp_path):
        config_path = str(tmp_path / "config.jsonc")
        old_config = {
            "providers": {},
            "models": {},
            "embedding_providers": {
                "openai": {
                    "api_key": "${OPENAI_API_KEY}",
                    "base_url": "https://api.openai.com",
                    "format": "openai",
                }
            },
            "embedding_models": {"text-embed": "openai"},
        }
        with open(config_path, "w") as f:
            json.dump(old_config, f)

        from llm_rosetta.gateway.config import JsoncConfigIO

        io = JsoncConfigIO()
        raw = io.load_raw(config_path)

        assert raw["providers"]["openai"]["api_key"] == "${OPENAI_API_KEY}"
        assert raw["providers"]["openai"]["embedding_format"] == "openai"
        assert "embedding_providers" not in raw
