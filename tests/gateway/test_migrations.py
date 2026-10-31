"""Tests for the config migration framework."""

from __future__ import annotations

import json
import logging
import os
from typing import Any


from llm_rosetta_gateway.maintenance.migrations import CURRENT_VERSION, migrate


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
        assert prov["embeddings_format"] == "openai"
        assert prov["embeddings_path"] == "/v1/embeddings"
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
            "type": "embeddings",
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
        from llm_rosetta_gateway.config import GatewayConfig

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
        assert "text-embed-3" in cfg.embeddings_models
        route = cfg.resolve_rerank("jina-reranker")
        assert route.provider_name == "jina"


class TestMigrationV1ToV2:
    """Migration 002: rename the ``embedding`` model type to ``embeddings``."""

    def test_provider_endpoint_keys_renamed(self):
        raw = {
            "config_version": 1,
            "providers": {
                "openai": {
                    "api_key": "sk-test",
                    "base_url": "https://api.openai.com",
                    "embedding_format": "openai",
                    "embedding_path": "/v1/embeddings",
                    "embedding_shim": "openai--embedding",
                }
            },
            "models": {},
        }
        migrate(raw)
        prov = raw["providers"]["openai"]
        assert prov["embeddings_format"] == "openai"
        assert prov["embeddings_path"] == "/v1/embeddings"
        assert prov["embeddings_shim"] == "openai--embedding"
        assert "embedding_format" not in prov

    def test_model_type_renamed(self):
        raw = {
            "config_version": 1,
            "providers": {},
            "models": {"text-embed-3": {"provider": "openai", "type": "embedding"}},
        }
        migrate(raw)
        assert raw["models"]["text-embed-3"]["type"] == "embeddings"

    def test_default_format_key_renamed(self):
        raw = {
            "config_version": 1,
            "providers": {},
            "models": {},
            "default_embedding_format": "cohere",
        }
        migrate(raw)
        assert raw["default_embeddings_format"] == "cohere"
        assert "default_embedding_format" not in raw

    def test_pool_sections_renamed(self):
        """A v1 file that kept its pools says ``embedding`` in no place."""
        raw: dict[str, Any] = {
            "config_version": 1,
            "providers": {},
            "models": {},
            "embedding_providers": {
                "jina": {
                    "api_key": "j-test",
                    "base_url": "https://api.jina.ai",
                    "format": "jina",
                }
            },
            "embedding_models": {"jina-embed": "jina"},
        }
        migrate(raw)
        assert raw["embeddings_providers"]["jina"]["format"] == "jina"
        assert raw["embeddings_models"] == {"jina-embed": "jina"}
        assert "embedding_providers" not in raw
        assert "embedding_models" not in raw

    def test_pool_sections_merge_entry_by_entry(self):
        """Both spellings present: the canonical entry wins, nothing is lost."""
        raw: dict[str, Any] = {
            "config_version": 1,
            "providers": {},
            "models": {},
            "embeddings_providers": {"jina": {"format": "jina", "api_key": "new"}},
            "embedding_providers": {
                "jina": {"format": "cohere", "api_key": "old"},
                "cohere": {"format": "cohere"},
            },
            "embeddings_models": {"jina-embed": "jina"},
            "embedding_models": {"cohere-embed": "cohere"},
        }
        migrate(raw)
        assert raw["embeddings_providers"]["jina"]["api_key"] == "new"
        assert raw["embeddings_providers"]["cohere"]["format"] == "cohere"
        assert raw["embeddings_models"] == {
            "jina-embed": "jina",
            "cohere-embed": "cohere",
        }
        assert "embedding_providers" not in raw
        assert "embedding_models" not in raw

    def test_v0_document_reaches_v2(self):
        raw = {
            "providers": {},
            "models": {},
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
        assert raw["providers"]["openai"]["embeddings_format"] == "openai"
        assert raw["models"]["text-embed-3"]["type"] == "embeddings"
        assert raw["config_version"] == CURRENT_VERSION


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

        from llm_rosetta_gateway.config import JsoncConfigIO

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

        from llm_rosetta_gateway.config import JsoncConfigIO

        io = JsoncConfigIO()
        raw = io.load_raw(config_path)

        assert raw["providers"]["openai"]["api_key"] == "${OPENAI_API_KEY}"
        assert raw["providers"]["openai"]["embeddings_format"] == "openai"
        assert "embedding_providers" not in raw

    def test_v1_pool_sections_written_back_renamed(self, tmp_path):
        """A v1 file keeps its pools but stops saying ``embedding``.

        The migration reads the section on the way in and writes the
        canonical spelling out, so a migrated file does not say both.
        """
        config_path = str(tmp_path / "config.jsonc")
        v1_config = {
            "config_version": 1,
            "providers": {},
            "models": {},
            "embedding_providers": {
                "jina": {
                    "api_key": "j-test",
                    "base_url": "https://api.jina.ai",
                    "format": "jina",
                }
            },
            "embedding_models": {"jina-embed": "jina"},
        }
        with open(config_path, "w") as f:
            json.dump(v1_config, f)

        from llm_rosetta_gateway.config import JsoncConfigIO

        io = JsoncConfigIO()
        io.load(config_path)

        with open(config_path) as f:
            on_disk = json.load(f)
        assert on_disk["config_version"] == CURRENT_VERSION
        assert on_disk["embeddings_providers"]["jina"]["format"] == "jina"
        assert on_disk["embeddings_models"] == {"jina-embed": "jina"}
        assert "embedding_providers" not in on_disk
        assert "embedding_models" not in on_disk

    def test_load_does_not_write_substituted_env_vars(self, tmp_path, monkeypatch):
        """Critical: load() must not write substituted secrets back to disk."""
        monkeypatch.setenv("TEST_API_KEY", "sk-real-secret-value")
        config_path = str(tmp_path / "config.jsonc")
        old_config = {
            "providers": {},
            "models": {},
            "rerank_providers": {
                "jina": {
                    "api_key": "${TEST_API_KEY}",
                    "base_url": "https://api.jina.ai",
                    "format": "jina",
                }
            },
            "rerank_models": {"jina-reranker": "jina"},
        }
        with open(config_path, "w") as f:
            json.dump(old_config, f)

        from llm_rosetta_gateway.config import JsoncConfigIO

        io = JsoncConfigIO()
        raw = io.load(config_path)

        # Runtime should have the substituted value
        assert raw["providers"]["jina"]["api_key"] == "sk-real-secret-value"

        # Disk should still have the placeholder
        with open(config_path) as f:
            on_disk = json.load(f)
        assert on_disk["providers"]["jina"]["api_key"] == "${TEST_API_KEY}"
        assert on_disk["config_version"] == CURRENT_VERSION

    def test_backup_created_on_migration(self, tmp_path):
        config_path = str(tmp_path / "config.jsonc")
        old_config = {
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
        with open(config_path, "w") as f:
            json.dump(old_config, f)

        from llm_rosetta_gateway.config import JsoncConfigIO

        io = JsoncConfigIO()
        io.load(config_path)

        bak_path = config_path + ".pre-migration.bak"
        assert os.path.exists(bak_path)
        with open(bak_path) as f:
            bak_data = json.load(f)
        assert "rerank_providers" in bak_data


class TestTheVersionField:
    """``config_version`` is this module's bookkeeping, not the operator's input.

    The scaffold ``llm-rosetta-gateway init`` writes carries no such key, so a
    document is never *asked* for one — which means a value we cannot read is
    our problem to work around, never a reason to refuse the document.
    """

    def _catching(self, caplog):
        """The gateway logger does not propagate, so it gets caplog's handler."""
        gateway_logger = logging.getLogger("llm-rosetta-gateway")
        gateway_logger.addHandler(caplog.handler)
        return gateway_logger

    def test_a_quoted_number_is_the_version_it_names(self):
        """A template that renders ``"1"`` means v1, not "unreadable"."""
        current = str(CURRENT_VERSION)
        raw = {"config_version": current, "providers": {}, "models": {}}
        _raw, changed = migrate(raw)
        assert changed is False
        assert raw["config_version"] == current, "and it is left as written"

    def test_a_json_float_is_the_version_it_names(self):
        raw = {"config_version": float(CURRENT_VERSION), "providers": {}, "models": {}}
        assert migrate(raw)[1] is False

    def test_an_unreadable_version_means_none_recorded(self, caplog):
        """Refusing a document over our own field is nobody's business."""
        raw = {"config_version": "one", "providers": {}, "models": {}}
        gateway_logger = self._catching(caplog)
        try:
            _raw, changed = migrate(raw)
        finally:
            gateway_logger.removeHandler(caplog.handler)

        assert changed is True, "the document is brought forward"
        assert raw["config_version"] == CURRENT_VERSION
        assert "not a version number" in caplog.text

    def test_an_explicit_null_means_none_recorded(self):
        raw = {"config_version": None, "providers": {}, "models": {}}
        _raw, changed = migrate(raw)
        assert changed is True
        assert raw["config_version"] == CURRENT_VERSION

    def test_a_boolean_is_not_a_version(self, caplog):
        """``True`` is an ``int`` in Python, and it is not v1."""
        raw = {"config_version": True, "providers": {}, "models": {}}
        gateway_logger = self._catching(caplog)
        try:
            _raw, changed = migrate(raw)
        finally:
            gateway_logger.removeHandler(caplog.handler)

        assert changed is True
        assert raw["config_version"] == CURRENT_VERSION
        assert "not a version number" in caplog.text

    def test_a_newer_version_is_served_as_it_is(self, caplog):
        """A document from a release ahead of this one is not ours to take apart."""
        raw = {
            "config_version": CURRENT_VERSION + 1,
            "providers": {"x": {"api_key": "k", "base_url": "https://x"}},
            "rerank_providers": {"jina": {"api_key": "k", "base_url": "https://j"}},
        }
        gateway_logger = self._catching(caplog)
        try:
            _raw, changed = migrate(raw)
        finally:
            gateway_logger.removeHandler(caplog.handler)

        assert changed is False
        assert raw["config_version"] == CURRENT_VERSION + 1, (
            "stamping our own version on it would record work that did not happen"
        )
        assert "rerank_providers" in raw, "and its shape is left alone"
        assert "newer than this release knows" in caplog.text
