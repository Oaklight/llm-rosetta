"""Unit tests for rerank gateway pipeline and config."""

from __future__ import annotations

import pytest

from llm_rosetta.gateway.rerank_pipeline import (
    RerankConversionPipeline,
    get_rerank_converter,
)


# ============================================================================
# Pipeline tests
# ============================================================================


class TestRerankConversionPipeline:
    def test_same_format_passthrough(self) -> None:
        pipeline = RerankConversionPipeline("jina", "jina")
        body = {
            "model": "test",
            "query": "hello",
            "documents": ["doc1"],
            "top_n": 1,
        }
        result = pipeline.convert_request(body)
        assert result is body  # same object, no copy

    def test_jina_to_cohere_request(self) -> None:
        pipeline = RerankConversionPipeline("jina", "cohere")
        body = {
            "model": "test",
            "query": "capital of France",
            "documents": ["Paris is the capital.", "Berlin is the capital."],
            "top_n": 1,
        }
        result = pipeline.convert_request(body)
        assert result["model"] == "test"
        assert result["query"] == "capital of France"
        assert result["documents"] == [
            "Paris is the capital.",
            "Berlin is the capital.",
        ]
        assert result["top_n"] == 1

    def test_jina_to_voyage_request_top_k(self) -> None:
        pipeline = RerankConversionPipeline("jina", "voyage")
        body = {
            "model": "test",
            "query": "hello",
            "documents": ["doc1"],
            "top_n": 2,
        }
        result = pipeline.convert_request(body)
        assert result["top_k"] == 2
        assert "top_n" not in result

    def test_voyage_to_jina_request_top_n(self) -> None:
        pipeline = RerankConversionPipeline("voyage", "jina")
        body = {
            "model": "test",
            "query": "hello",
            "documents": ["doc1"],
            "top_k": 3,
        }
        result = pipeline.convert_request(body)
        assert result["top_n"] == 3
        assert "top_k" not in result

    def test_cohere_to_jina_response(self) -> None:
        pipeline = RerankConversionPipeline("jina", "cohere")
        cohere_resp = {
            "id": "abc",
            "results": [
                {"index": 0, "relevance_score": 0.9},
                {"index": 1, "relevance_score": 0.3},
            ],
            "meta": {"billed_units": {"search_units": 1}},
        }
        result = pipeline.convert_response(cohere_resp)
        assert result["object"] == "list"
        assert "results" in result
        assert len(result["results"]) == 2
        assert result["results"][0]["relevance_score"] == pytest.approx(0.9)

    def test_jina_to_voyage_response(self) -> None:
        pipeline = RerankConversionPipeline("voyage", "jina")
        jina_resp = {
            "model": "jina-reranker-v2",
            "object": "list",
            "usage": {"total_tokens": 50},
            "results": [
                {"index": 0, "relevance_score": 0.8, "document": "doc text"},
            ],
        }
        result = pipeline.convert_response(jina_resp)
        assert result["object"] == "list"
        assert "data" in result
        assert result["data"][0]["document"] == "doc text"

    def test_full_roundtrip_jina_cohere(self) -> None:
        pipeline_fwd = RerankConversionPipeline("jina", "cohere")
        pipeline_bwd = RerankConversionPipeline("cohere", "jina")

        jina_req = {
            "model": "test",
            "query": "hello",
            "documents": ["a", "b"],
            "top_n": 1,
        }
        cohere_req = pipeline_fwd.convert_request(jina_req)
        jina_req_back = pipeline_bwd.convert_request(cohere_req)
        assert jina_req_back["query"] == "hello"
        assert jina_req_back["top_n"] == 1

    def test_warnings_accessible(self) -> None:
        # source=cohere, target=jina: convert_response goes
        # jina response (with docs) → IR → cohere response (drops docs → warning)
        pipeline = RerankConversionPipeline("cohere", "jina")
        jina_resp = {
            "model": "test",
            "object": "list",
            "results": [
                {"index": 0, "relevance_score": 0.5, "document": "has doc"},
            ],
        }
        pipeline.convert_response(jina_resp)
        assert any("document data" in w for w in pipeline.warnings)


class TestGetRerankConverter:
    def test_valid_formats(self) -> None:
        for fmt in ("jina", "cohere", "voyage"):
            conv = get_rerank_converter(fmt)
            assert conv is not None

    def test_invalid_format(self) -> None:
        with pytest.raises(ValueError, match="Unknown rerank format"):
            get_rerank_converter("nonexistent")


# ============================================================================
# Config tests
# ============================================================================


class TestRerankConfig:
    def test_parse_rerank_config(self) -> None:
        from llm_rosetta.gateway.config import GatewayConfig

        raw = {
            "providers": {
                "openai_chat": {
                    "api_key": "sk-test",
                    "base_url": "https://api.openai.com/v1",
                }
            },
            "models": {"gpt-4o": "openai_chat"},
            "rerank_providers": {
                "jina": {
                    "api_key": "jina-key",
                    "base_url": "https://api.jina.ai",
                    "format": "jina",
                    "rerank_path": "/v1/rerank",
                },
                "cohere": {
                    "api_key": "cohere-key",
                    "base_url": "https://api.cohere.com",
                    "format": "cohere",
                    "rerank_path": "/v2/rerank",
                },
            },
            "rerank_models": {
                "jina-reranker-v2": "jina",
                "rerank-v3.5": "cohere",
            },
            "default_rerank_format": "cohere",
        }
        config = GatewayConfig(raw)
        assert len(config.rerank_providers) == 2
        assert config.rerank_providers["jina"]["format"] == "jina"
        assert config.rerank_providers["cohere"]["rerank_path"] == "/v2/rerank"
        assert len(config.rerank_models) == 2
        assert config.default_rerank_format == "cohere"

    def test_resolve_rerank(self) -> None:
        from llm_rosetta.gateway.config import GatewayConfig

        raw = {
            "providers": {
                "openai_chat": {
                    "api_key": "sk-test",
                    "base_url": "https://api.openai.com/v1",
                }
            },
            "models": {"gpt-4o": "openai_chat"},
            "rerank_providers": {
                "jina": {
                    "api_key": "jina-key",
                    "base_url": "https://api.jina.ai",
                    "format": "jina",
                    "rerank_path": "/v1/rerank",
                }
            },
            "rerank_models": {"jina-reranker-v2": "jina"},
        }
        config = GatewayConfig(raw)
        route = config.resolve_rerank("jina-reranker-v2")
        assert route.provider_name == "jina"
        assert route.format == "jina"
        assert route.rerank_path == "/v1/rerank"
        assert route.provider_info.base_url == "https://api.jina.ai"
        assert route.provider_info.auth_headers()["Authorization"] == "Bearer jina-key"

    def test_resolve_rerank_unknown_model(self) -> None:
        from llm_rosetta.gateway.config import GatewayConfig

        raw = {
            "providers": {
                "openai_chat": {
                    "api_key": "sk-test",
                    "base_url": "https://api.openai.com/v1",
                }
            },
            "models": {"gpt-4o": "openai_chat"},
            "rerank_providers": {},
            "rerank_models": {},
        }
        config = GatewayConfig(raw)
        with pytest.raises(KeyError):
            config.resolve_rerank("nonexistent")

    def test_empty_rerank_config(self) -> None:
        from llm_rosetta.gateway.config import GatewayConfig

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
        assert config.rerank_providers == {}
        assert config.rerank_models == {}
        assert config.default_rerank_format == "jina"

    def test_disabled_rerank_provider_skipped(self) -> None:
        from llm_rosetta.gateway.config import GatewayConfig

        raw = {
            "providers": {
                "openai_chat": {
                    "api_key": "sk-test",
                    "base_url": "https://api.openai.com/v1",
                }
            },
            "models": {"gpt-4o": "openai_chat"},
            "rerank_providers": {
                "jina": {
                    "api_key": "key",
                    "base_url": "https://api.jina.ai",
                    "format": "jina",
                    "enabled": False,
                }
            },
            "rerank_models": {"jina-reranker-v2": "jina"},
        }
        config = GatewayConfig(raw)
        assert "jina" not in config.rerank_providers
        assert len(config.rerank_models) == 0
