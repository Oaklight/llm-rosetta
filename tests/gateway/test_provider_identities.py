"""Tests for provider identities and capability-scoped endpoint resolution."""

from __future__ import annotations

import pytest

from llm_rosetta_gateway.admin.routes import config as admin_config
from llm_rosetta_gateway.config import GatewayConfig, ProviderIdentity
from llm_rosetta.shims.provider_shim import (
    ConnectionConfig,
    ProviderShim,
    register_shim,
    unregister_shim,
)

SHIM_NAME = "acme--embedding"


@pytest.fixture()
def embedding_shim():
    """A registered embedding shim with its own path and connection defaults."""
    shim = ProviderShim(
        name=SHIM_NAME,
        format="openai",
        model_type="embedding",
        connection=ConnectionConfig(
            base_url="https://embed.acme.test",
            api_key_env="ACME_EMBED_KEY",
            default_path="/shim/embeddings",
        ),
    )
    register_shim(shim)
    yield shim
    unregister_shim(SHIM_NAME)


class TestProviderIdentities:
    def test_chat_provider_identity(self):
        cfg = GatewayConfig(
            {
                "providers": {"openai": {"type": "openai", "api_key": "k"}},
                "models": {"m": "openai"},
            }
        )
        assert cfg.provider_identities["openai"] == ProviderIdentity(
            model_type="llm", format="chat_completions"
        )
        # provider_types keeps meaning the format for existing consumers.
        assert cfg.provider_types["openai"] == "chat_completions"

    def test_embedding_only_provider_takes_the_shim_identity(self, embedding_shim):
        cfg = GatewayConfig(
            {
                "providers": {"acme": {"embedding_shim": SHIM_NAME}},
                "models": {},
            }
        )
        assert cfg.provider_identities["acme"] == ProviderIdentity(
            model_type="embedding", format="openai"
        )

    def test_chat_identity_is_not_overridden_by_an_endpoint_shim(self, embedding_shim):
        cfg = GatewayConfig(
            {
                "providers": {"acme": {"type": "openai", "embedding_shim": SHIM_NAME}},
                "models": {"m": "acme"},
            }
        )
        assert cfg.provider_identities["acme"].model_type == "llm"
        # The endpoint still resolves through the shim.
        assert cfg.embedding_providers["acme"]["format"] == "openai"

    @pytest.mark.parametrize(
        ("fmt", "expected"),
        [
            ("cohere", "embedding"),
            ("jina", "embedding"),
            ("voyage", "embedding"),
            ("systemone", "decision"),
        ],
    )
    def test_known_non_chat_format_does_not_default_to_llm(self, fmt, expected):
        cfg = GatewayConfig(
            {
                "providers": {
                    "p": {
                        "type": fmt,
                        "api_key": "k",
                        "base_url": "https://x.test",
                    }
                },
                "models": {},
            }
        )
        identity = cfg.provider_identities["p"]
        assert identity.model_type == expected
        assert identity.format == fmt

    @pytest.mark.parametrize(
        ("endpoint_key", "format_name", "expected_type"),
        [
            ("embedding_format", "openai", "embedding"),
            ("rerank_format", "jina", "rerank"),
            ("decision_format", "openai_decisions", "decision"),
        ],
    )
    def test_format_only_provider_takes_the_endpoint_identity(
        self, endpoint_key, format_name, expected_type
    ):
        """A shim-free endpoint declaration names the provider's category.

        The format defaults to the provider *name* when nothing else declares
        one, so without reading ``<type>_format`` here the identity would stay
        ``llm`` — hiding the provider from the admin picker and from the
        LLM-model validator.
        """
        cfg = GatewayConfig(
            {
                "providers": {
                    "acme": {
                        endpoint_key: format_name,
                        "api_key": "k",
                        "base_url": "https://acme.test",
                    }
                },
                "models": {},
            }
        )
        assert cfg.provider_identities["acme"] == ProviderIdentity(
            model_type=expected_type, format=format_name
        )

    def test_format_only_decision_provider_rejects_an_llm_model(self):
        with pytest.raises(ValueError, match="serves decision"):
            GatewayConfig(
                {
                    "providers": {
                        "d": {
                            "decision_format": "openai_decisions",
                            "api_key": "k",
                            "base_url": "https://d.test",
                        }
                    },
                    "models": {"m": {"providers": ["d"]}},
                }
            )

    def test_declared_endpoint_category_wins_over_registry_order(self):
        cfg = GatewayConfig(
            {
                "providers": {
                    "p": {
                        "type": "cohere",
                        "api_key": "k",
                        "base_url": "https://x.test",
                        "rerank_format": "cohere",
                    }
                },
                "models": {},
            }
        )
        assert cfg.provider_identities["p"].model_type == "rerank"

    def test_ambiguous_format_provider_rejects_an_llm_model(self):
        with pytest.raises(ValueError, match="serves embedding"):
            GatewayConfig(
                {
                    "providers": {
                        "p": {
                            "type": "cohere",
                            "api_key": "k",
                            "base_url": "https://x.test",
                        }
                    },
                    "models": {"m": "p"},
                }
            )

    def test_provider_named_after_a_chat_shim_keeps_its_identity(self, embedding_shim):
        from llm_rosetta.shims.providers import load_providers

        load_providers()
        cfg = GatewayConfig(
            {
                "providers": {
                    "openai--chat_completions": {"embedding_shim": SHIM_NAME}
                },
                "models": {},
            }
        )
        assert cfg.provider_identities["openai--chat_completions"] == ProviderIdentity(
            model_type="llm", format="chat_completions"
        )


class TestEndpointResolution:
    def test_embedding_shim_supplies_format_path_and_connection(
        self, embedding_shim, monkeypatch
    ):
        monkeypatch.setenv("ACME_EMBED_KEY", "secret")
        cfg = GatewayConfig(
            {
                "providers": {"acme": {"embedding_shim": SHIM_NAME}},
                "models": {},
            }
        )
        assert cfg.embedding_providers["acme"]["format"] == "openai"
        assert cfg.embedding_providers["acme"]["embedding_path"] == "/shim/embeddings"
        info = cfg.embedding_provider_infos["acme"]
        assert info.base_url == "https://embed.acme.test"
        assert info.auth_headers() == {"Authorization": "Bearer secret"}
        assert info.upstream_url("m") == "https://embed.acme.test/shim/embeddings"

    def test_explicit_path_beats_the_shim_path(self, embedding_shim):
        cfg = GatewayConfig(
            {
                "providers": {
                    "acme": {
                        "embedding_shim": SHIM_NAME,
                        "embedding_path": "/custom/embeddings",
                    }
                },
                "models": {},
            }
        )
        assert cfg.embedding_providers["acme"]["embedding_path"] == "/custom/embeddings"

    def test_shim_path_is_not_used_for_a_different_format(self, embedding_shim):
        cfg = GatewayConfig(
            {
                "providers": {
                    "acme": {
                        "embedding_shim": SHIM_NAME,
                        "embedding_format": "cohere",
                    }
                },
                "models": {},
            }
        )
        assert cfg.embedding_providers["acme"]["format"] == "cohere"
        # The shim's path belongs to its own format, so the effective format's
        # own default applies instead.
        assert cfg.embedding_providers["acme"]["embedding_path"] == "/v2/embed"

    def test_explicit_format_is_canonicalised(self):
        cfg = GatewayConfig(
            {
                "providers": {
                    "acme": {
                        "base_url": "https://acme.test",
                        "api_key": "k",
                        "embedding_format": "google",
                    }
                },
                "models": {},
            }
        )
        assert cfg.embedding_providers["acme"]["format"] == "google_generate"
        # Google's path shape is per-format, not the embedding default.
        assert (
            cfg.embedding_providers["acme"]["embedding_path"]
            == "/v1beta/models/{model}:embedContent"
        )
        info = cfg.embedding_provider_infos["acme"]
        assert info.auth_headers() == {"x-goog-api-key": "k"}

    def test_legacy_google_embedding_uses_the_format_default_path(self):
        cfg = GatewayConfig(
            {
                "providers": {},
                "embedding_providers": {
                    "g": {
                        "base_url": "https://generativelanguage.googleapis.com",
                        "api_key": "k",
                        "format": "google_generate",
                    }
                },
                "embedding_models": {"m": "g"},
            }
        )
        assert (
            cfg.embedding_providers["g"]["embedding_path"]
            == "/v1beta/models/{model}:embedContent"
        )

    def test_mismatched_endpoint_shim_is_rejected(self):
        from llm_rosetta.shims.providers import load_providers

        load_providers()
        with pytest.raises(ValueError, match="serves llm, not embedding"):
            GatewayConfig(
                {
                    "providers": {
                        "acme": {"embedding_shim": "deepseek--chat_completions"}
                    },
                    "models": {},
                }
            )

    def test_unknown_endpoint_shim_is_rejected(self):
        with pytest.raises(ValueError, match="embedding_shim names unknown shim"):
            GatewayConfig(
                {
                    "providers": {"acme": {"embedding_shim": "nope"}},
                    "models": {},
                }
            )


class TestValidation:
    def test_llm_model_cannot_route_to_an_embedding_provider(self, embedding_shim):
        with pytest.raises(
            ValueError,
            match="model 'gpt-4' is an LLM model but provider 'acme' serves embedding",
        ):
            GatewayConfig(
                {
                    "providers": {"acme": {"embedding_shim": SHIM_NAME}},
                    "models": {"gpt-4": "acme"},
                }
            )

    def test_typed_embedding_model_is_accepted(self, embedding_shim):
        cfg = GatewayConfig(
            {
                "providers": {"acme": {"embedding_shim": SHIM_NAME}},
                "models": {"e": {"provider": "acme", "type": "embedding"}},
            }
        )
        assert cfg.embedding_models["e"] == "acme"


class TestBuiltinEndpointShims:
    """The shipped embedding/rerank shims resolve an endpoint-only provider.

    Each case names the shim alone (no ``*_format`` / ``*_path`` / connection
    values in the config) so the shim supplies the whole endpoint: format,
    path and base URL.  The auth header follows the format — Google takes
    ``x-goog-api-key``, the OpenAI-shaped formats take a Bearer token.
    """

    @pytest.mark.parametrize(
        ("shim_name", "type_name", "fmt", "url"),
        [
            (
                "openai--embedding",
                "embedding",
                "openai",
                "https://api.openai.com/v1/embeddings",
            ),
            (
                "cohere--embedding",
                "embedding",
                "cohere",
                "https://api.cohere.com/v2/embed",
            ),
            (
                "jina--embedding",
                "embedding",
                "jina",
                "https://api.jina.ai/v1/embeddings",
            ),
            (
                "voyage--embedding",
                "embedding",
                "voyage",
                "https://api.voyageai.com/v1/embeddings",
            ),
            (
                "google--embedding",
                "embedding",
                "google_generate",
                "https://generativelanguage.googleapis.com"
                "/v1beta/models/m:embedContent",
            ),
        ],
    )
    def test_embedding_shims(self, shim_name, type_name, fmt, url):
        from llm_rosetta.shims.providers import load_providers

        load_providers()
        cfg = GatewayConfig(
            {
                "providers": {"acme": {f"{type_name}_shim": shim_name, "api_key": "k"}},
                "models": {},
            }
        )
        assert cfg.provider_identities["acme"] == ProviderIdentity(
            model_type=type_name, format=fmt
        )
        assert getattr(cfg, f"{type_name}_providers")["acme"]["format"] == fmt
        info = getattr(cfg, f"{type_name}_provider_infos")["acme"]
        assert info.upstream_url("m") == url
        expected_auth = (
            {"x-goog-api-key": "k"}
            if fmt == "google_generate"
            else {"Authorization": "Bearer k"}
        )
        assert info.auth_headers() == expected_auth

    @pytest.mark.parametrize(
        ("shim_name", "fmt", "url"),
        [
            ("cohere--rerank", "cohere", "https://api.cohere.com/v2/rerank"),
            ("jina--rerank", "jina", "https://api.jina.ai/v1/rerank"),
            ("voyage--rerank", "voyage", "https://api.voyageai.com/v1/rerank"),
        ],
    )
    def test_rerank_shims(self, shim_name, fmt, url):
        from llm_rosetta.shims.providers import load_providers

        load_providers()
        cfg = GatewayConfig(
            {
                "providers": {"acme": {"rerank_shim": shim_name, "api_key": "k"}},
                "models": {},
            }
        )
        assert cfg.provider_identities["acme"] == ProviderIdentity(
            model_type="rerank", format=fmt
        )
        assert cfg.rerank_providers["acme"]["format"] == fmt
        info = cfg.rerank_provider_infos["acme"]
        assert info.upstream_url("m") == url
        assert info.auth_headers() == {"Authorization": "Bearer k"}


class TestAdminMetadata:
    def test_model_types_expose_the_endpoint_keys(self):
        meta = {d["name"]: d for d in admin_config._get_model_type_metadata()}
        assert meta["embedding"]["config_shim_key"] == "embedding_shim"
        assert meta["embedding"]["default_format"] == "openai"
        assert meta["embedding"]["default_paths"] == {
            "cohere": "/v2/embed",
            "google_generate": "/v1beta/models/{model}:embedContent",
        }
        assert meta["rerank"]["default_paths"] == {"cohere": "/v2/rerank"}
        assert meta["rerank"]["config_shim_key"] == "rerank_shim"
        assert meta["rerank"]["default_format"] == "jina"
        assert meta["decision"]["config_shim_key"] == "decision_shim"
        assert meta["decision"]["default_paths"] == {
            "openai_decisions": "/v1/decisions"
        }
        assert meta["llm"]["config_shim_key"] is None
