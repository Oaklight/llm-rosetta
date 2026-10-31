"""Unit tests for embedding gateway pipeline and config."""

from __future__ import annotations

import pytest

from llm_rosetta.converters.embeddings.pipeline import (
    EmbeddingConversionPipeline,
    get_embedding_converter,
)


class TestEmbeddingConversionPipeline:
    def test_same_format_goes_through_the_ir(self) -> None:
        """Same-format requests are canonicalised by the IR round-trip."""
        pipeline = EmbeddingConversionPipeline("openai", "openai")
        body = {"model": "test", "input": "hello"}
        result = pipeline.convert_request(body)
        assert result is not body
        assert result == {"model": "test", "input": ["hello"]}

    def test_same_format_response_is_rebuilt(self) -> None:
        """Same-format responses are rebuilt from the IR, not passed through."""
        pipeline = EmbeddingConversionPipeline("openai", "openai")
        upstream = {
            "object": "list",
            "data": [{"object": "embedding", "embedding": [0.1], "index": 0}],
            "model": "test",
            "provider_extra": "dropped",
        }
        result = pipeline.convert_response(upstream)
        assert result == {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
            "model": "test",
        }

    def test_openai_to_cohere_request(self) -> None:
        pipeline = EmbeddingConversionPipeline("openai", "cohere")
        body = {"model": "test", "input": ["hello world"]}
        result = pipeline.convert_request(body)
        assert result["texts"] == ["hello world"]
        assert "input" not in result
        assert result["embedding_types"] == ["float"]

    def test_cohere_to_openai_response(self) -> None:
        pipeline = EmbeddingConversionPipeline("openai", "cohere")
        cohere_resp = {
            "id": "abc",
            "embeddings": {"float": [[0.1, 0.2, 0.3]]},
            "meta": {"billed_units": {"input_tokens": 1}},
            "response_type": "embeddings_by_type",
        }
        result = pipeline.convert_response(cohere_resp)
        assert result["object"] == "list"
        assert len(result["data"]) == 1
        assert result["data"][0]["object"] == "embedding"
        assert result["data"][0]["embedding"] == [0.1, 0.2, 0.3]

    def test_openai_to_jina_request_task(self) -> None:
        pipeline = EmbeddingConversionPipeline("jina", "openai")
        jina_req = {
            "model": "jina-embeddings-v3",
            "input": ["hello"],
            "task": "retrieval.query",
        }
        openai_req = pipeline.convert_request(jina_req)
        assert openai_req["input"] == ["hello"]

    def test_jina_to_voyage_response(self) -> None:
        pipeline = EmbeddingConversionPipeline("voyage", "jina")
        jina_resp = {
            "model": "jina-embeddings-v3",
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.5]}],
            "usage": {"total_tokens": 10},
        }
        result = pipeline.convert_response(jina_resp)
        assert result["object"] == "list"
        assert len(result["data"]) == 1

    def test_warnings_accessible(self) -> None:
        pipeline = EmbeddingConversionPipeline("openai", "cohere")
        assert pipeline.warnings == []

    def test_response_conversion_needs_no_request_first(self) -> None:
        """Both legs are independent: the pipeline holds no request state."""
        pipeline = EmbeddingConversionPipeline("openai", "cohere")
        result = pipeline.convert_response(
            {
                "id": "abc",
                "embeddings": {"float": [[0.1, 0.2, 0.3]]},
                "meta": {"billed_units": {"input_tokens": 1}},
                "response_type": "embeddings_by_type",
            }
        )
        assert result["data"][0]["embedding"] == [0.1, 0.2, 0.3]


class TestGetEmbeddingConverter:
    def test_valid_formats(self) -> None:
        for fmt in ("openai", "jina", "voyage", "cohere"):
            conv = get_embedding_converter(fmt)
            assert conv is not None

    def test_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="Unknown embedding format"):
            get_embedding_converter("nonexistent")


class TestEmbeddingConfig:
    def test_parse_embedding_config(self) -> None:
        from llm_rosetta_gateway.config import GatewayConfig

        raw = {
            "providers": {
                "openai_chat": {
                    "api_key": "sk-test",
                    "base_url": "https://api.openai.com/v1",
                }
            },
            "models": {"gpt-4o": "openai_chat"},
            "embeddings_providers": {
                "openai": {
                    "api_key": "sk-test",
                    "base_url": "https://api.openai.com",
                    "format": "openai",
                    "embeddings_path": "/v1/embeddings",
                },
                "cohere": {
                    "api_key": "cohere-key",
                    "base_url": "https://api.cohere.com",
                    "format": "cohere",
                    "embeddings_path": "/v2/embed",
                },
            },
            "embeddings_models": {
                "text-embedding-3-small": "openai",
                "embed-v4.0": "cohere",
            },
            "default_embeddings_format": "openai",
        }
        config = GatewayConfig(raw)
        assert len(config.embeddings_providers) == 2
        assert config.embeddings_providers["cohere"]["format"] == "cohere"
        assert config.embeddings_providers["cohere"]["embeddings_path"] == "/v2/embed"
        assert len(config.embeddings_models) == 2
        assert config.default_embeddings_format == "openai"

    def test_resolve_embedding(self) -> None:
        from llm_rosetta_gateway.config import GatewayConfig

        raw = {
            "providers": {
                "openai_chat": {
                    "api_key": "sk-test",
                    "base_url": "https://api.openai.com/v1",
                }
            },
            "models": {"gpt-4o": "openai_chat"},
            "embeddings_providers": {
                "jina": {
                    "api_key": "jina-key",
                    "base_url": "https://api.jina.ai",
                    "format": "jina",
                    "embeddings_path": "/v1/embeddings",
                }
            },
            "embeddings_models": {"jina-embeddings-v3": "jina"},
        }
        config = GatewayConfig(raw)
        route = config.resolve_embedding("jina-embeddings-v3")
        assert route.provider_name == "jina"
        assert route.format == "jina"
        assert route.path == "/v1/embeddings"
        assert route.provider_info.base_url == "https://api.jina.ai"
        assert route.provider_info.auth_headers()["Authorization"] == "Bearer jina-key"

    def test_legacy_embeddings_models_accept_a_provider_list(self) -> None:
        from llm_rosetta_gateway.config import GatewayConfig

        raw = {
            "providers": {},
            "embeddings_providers": {
                "g": {
                    "api_key": "k",
                    "base_url": "https://generativelanguage.googleapis.com",
                    "format": "google_generate",
                }
            },
            "embeddings_models": {
                "m": {"providers": ["g"]},
            },
        }
        config = GatewayConfig(raw)
        assert config.embeddings_models["m"] == "g"

    def test_empty_embedding_config(self) -> None:
        from llm_rosetta_gateway.config import GatewayConfig

        raw = {
            "providers": {
                "openai_chat": {
                    "api_key": "sk-test",
                    "base_url": "https://api.openai.com/v1",
                }
            },
            "models": {"gpt-4o": "openai_chat"},
        }
        config = GatewayConfig(raw)
        assert config.embeddings_providers == {}
        assert config.embeddings_models == {}
        assert config.default_embeddings_format == "openai"

    def test_resolve_embedding_unknown_model(self) -> None:
        from llm_rosetta_gateway.config import GatewayConfig

        raw = {
            "providers": {
                "openai_chat": {
                    "api_key": "sk-test",
                    "base_url": "https://api.openai.com/v1",
                }
            },
            "models": {"gpt-4o": "openai_chat"},
        }
        config = GatewayConfig(raw)
        with pytest.raises(KeyError):
            config.resolve_embedding("nonexistent")
