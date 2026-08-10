"""Unit tests for rerank converters using real API response fixtures."""

from __future__ import annotations

import pytest

from llm_rosetta.converters.rerank import (
    CohereRerankConverter,
    JinaRerankConverter,
    VoyageRerankConverter,
)
from llm_rosetta.types.ir.rerank import (
    IRRerankRequest,
    RerankDocument,
)

# ============================================================================
# Fixtures — captured from real API calls
# ============================================================================

JINA_RESPONSE = {
    "model": "jina-reranker-v2-base-multilingual",
    "object": "list",
    "usage": {"total_tokens": 54},
    "results": [
        {
            "index": 0,
            "relevance_score": 0.83973396,
            "document": "Paris is the capital of France.",
        },
        {
            "index": 2,
            "relevance_score": 0.16451646,
            "document": "The Eiffel Tower is in Paris.",
        },
    ],
}

COHERE_RESPONSE = {
    "id": "c317b8b2-d572-4725-af60-cfb856aa28c8",
    "results": [
        {"index": 0, "relevance_score": 0.8923472},
        {"index": 2, "relevance_score": 0.25163436},
    ],
    "meta": {
        "api_version": {"version": "2"},
        "billed_units": {"search_units": 1},
    },
}

SILICONFLOW_RESPONSE = {
    "id": "019fea4450327dbea799b2175a8cc34c",
    "results": [
        {
            "index": 0,
            "document": None,
            "relevance_score": 0.9998469352722168,
        },
        {
            "index": 2,
            "document": None,
            "relevance_score": 0.19185182452201843,
        },
    ],
    "meta": {
        "tokens": {"input_tokens": 54, "output_tokens": 0, "image_tokens": 0},
        "billed_units": {
            "input_tokens": 54,
            "output_tokens": 0,
            "image_tokens": 0,
            "search_units": 0,
            "classifications": 0,
        },
    },
}

VOYAGE_RESPONSE = {
    "object": "list",
    "data": [
        {
            "relevance_score": 0.71875,
            "index": 0,
            "document": "Paris is the capital of France.",
        },
        {
            "relevance_score": 0.498046875,
            "index": 2,
            "document": "The Eiffel Tower is in Paris.",
        },
    ],
    "model": "rerank-2-lite",
    "usage": {"total_tokens": 32},
}

SAMPLE_REQUEST = {
    "model": "test-model",
    "query": "What is the capital of France?",
    "documents": [
        "Paris is the capital of France.",
        "Berlin is the capital of Germany.",
        "The Eiffel Tower is in Paris.",
    ],
    "top_n": 2,
}

SAMPLE_IR_REQUEST = IRRerankRequest(
    model="test-model",
    query="What is the capital of France?",
    documents=[
        RerankDocument(text="Paris is the capital of France."),
        RerankDocument(text="Berlin is the capital of Germany."),
        RerankDocument(text="The Eiffel Tower is in Paris."),
    ],
    top_n=2,
)


# ============================================================================
# Jina converter tests
# ============================================================================


class TestJinaRerankConverter:
    def setup_method(self) -> None:
        self.converter = JinaRerankConverter()

    def test_response_from_provider(self) -> None:
        ir = self.converter.response_from_provider(JINA_RESPONSE)
        assert ir["object"] == "rerank"
        assert ir["model"] == "jina-reranker-v2-base-multilingual"
        assert len(ir["results"]) == 2
        assert ir["results"][0]["index"] == 0
        assert ir["results"][0]["relevance_score"] == pytest.approx(0.8397, abs=1e-3)
        assert ir["results"][0]["document"]["text"] == "Paris is the capital of France."
        assert ir["usage"]["total_tokens"] == 54

    def test_request_to_provider(self) -> None:
        result, warnings = self.converter.request_to_provider(SAMPLE_IR_REQUEST)
        assert result["model"] == "test-model"
        assert result["query"] == "What is the capital of France?"
        assert result["documents"] == [
            "Paris is the capital of France.",
            "Berlin is the capital of Germany.",
            "The Eiffel Tower is in Paris.",
        ]
        assert result["top_n"] == 2
        assert warnings == []

    def test_request_from_provider(self) -> None:
        ir = self.converter.request_from_provider(SAMPLE_REQUEST)
        assert ir["model"] == "test-model"
        assert ir["query"] == "What is the capital of France?"
        assert len(ir["documents"]) == 3
        assert ir["documents"][0]["text"] == "Paris is the capital of France."
        assert ir["top_n"] == 2

    def test_response_to_provider(self) -> None:
        ir = self.converter.response_from_provider(JINA_RESPONSE)
        provider = self.converter.response_to_provider(ir)
        assert provider["object"] == "list"
        assert provider["model"] == "jina-reranker-v2-base-multilingual"
        assert len(provider["results"]) == 2
        assert provider["results"][0]["document"] == "Paris is the capital of France."
        assert provider["usage"]["total_tokens"] == 54

    def test_request_roundtrip(self) -> None:
        provider, _ = self.converter.request_to_provider(SAMPLE_IR_REQUEST)
        ir_back = self.converter.request_from_provider(provider)
        assert ir_back["model"] == SAMPLE_IR_REQUEST["model"]
        assert ir_back["query"] == SAMPLE_IR_REQUEST["query"]
        assert len(ir_back["documents"]) == len(SAMPLE_IR_REQUEST["documents"])
        for orig, back in zip(
            SAMPLE_IR_REQUEST["documents"], ir_back["documents"], strict=True
        ):
            assert orig["text"] == back["text"]

    def test_response_roundtrip(self) -> None:
        ir = self.converter.response_from_provider(JINA_RESPONSE)
        provider = self.converter.response_to_provider(ir)
        ir_back = self.converter.response_from_provider(provider)
        assert ir_back["results"][0]["index"] == ir["results"][0]["index"]
        assert ir_back["results"][0]["relevance_score"] == pytest.approx(
            ir["results"][0]["relevance_score"]
        )


# ============================================================================
# Cohere converter tests
# ============================================================================


class TestCohereRerankConverter:
    def setup_method(self) -> None:
        self.converter = CohereRerankConverter()

    def test_response_from_provider(self) -> None:
        ir = self.converter.response_from_provider(COHERE_RESPONSE)
        assert ir["object"] == "rerank"
        assert ir["id"] == "c317b8b2-d572-4725-af60-cfb856aa28c8"
        assert len(ir["results"]) == 2
        assert ir["results"][0]["index"] == 0
        assert ir["results"][0]["relevance_score"] == pytest.approx(0.8923, abs=1e-3)
        # Cohere v3 doesn't have meta.tokens, so no usage
        assert "usage" not in ir

    def test_response_from_provider_siliconflow(self) -> None:
        ir = self.converter.response_from_provider(SILICONFLOW_RESPONSE)
        assert ir["object"] == "rerank"
        assert ir["id"] == "019fea4450327dbea799b2175a8cc34c"
        assert len(ir["results"]) == 2
        # Siliconflow sends document: null — should not appear in IR
        assert "document" not in ir["results"][0]
        # Siliconflow has meta.tokens
        assert ir["usage"]["total_tokens"] == 54
        assert ir["usage"]["prompt_tokens"] == 54

    def test_request_to_provider(self) -> None:
        result, warnings = self.converter.request_to_provider(SAMPLE_IR_REQUEST)
        assert result["model"] == "test-model"
        assert result["documents"] == [
            "Paris is the capital of France.",
            "Berlin is the capital of Germany.",
            "The Eiffel Tower is in Paris.",
        ]
        assert result["top_n"] == 2
        assert "return_documents" not in result

    def test_request_from_provider(self) -> None:
        ir = self.converter.request_from_provider(SAMPLE_REQUEST)
        assert ir["model"] == "test-model"
        assert len(ir["documents"]) == 3

    def test_response_to_provider(self) -> None:
        ir = self.converter.response_from_provider(COHERE_RESPONSE)
        provider = self.converter.response_to_provider(ir)
        assert provider["id"] == "c317b8b2-d572-4725-af60-cfb856aa28c8"
        assert len(provider["results"]) == 2
        assert provider["results"][0]["relevance_score"] == pytest.approx(
            0.8923, abs=1e-3
        )

    def test_max_tokens_per_doc_roundtrip(self) -> None:
        ir = IRRerankRequest(
            model="rerank-v3.5",
            query="test",
            documents=[RerankDocument(text="doc")],
            max_tokens_per_doc=2048,
        )
        provider, _ = self.converter.request_to_provider(ir)
        assert provider["max_tokens_per_doc"] == 2048
        ir_back = self.converter.request_from_provider(provider)
        assert ir_back["max_tokens_per_doc"] == 2048


# ============================================================================
# Voyage converter tests
# ============================================================================


class TestVoyageRerankConverter:
    def setup_method(self) -> None:
        self.converter = VoyageRerankConverter()

    def test_response_from_provider(self) -> None:
        ir = self.converter.response_from_provider(VOYAGE_RESPONSE)
        assert ir["object"] == "rerank"
        assert ir["model"] == "rerank-2-lite"
        assert len(ir["results"]) == 2
        assert ir["results"][0]["index"] == 0
        assert ir["results"][0]["relevance_score"] == pytest.approx(0.7188, abs=1e-3)
        assert ir["results"][0]["document"]["text"] == "Paris is the capital of France."
        assert ir["usage"]["total_tokens"] == 32

    def test_request_to_provider_top_k(self) -> None:
        result, _ = self.converter.request_to_provider(SAMPLE_IR_REQUEST)
        # Voyage uses top_k, not top_n
        assert "top_k" in result
        assert result["top_k"] == 2
        assert "top_n" not in result

    def test_request_from_provider_top_k(self) -> None:
        voyage_request = {
            "model": "rerank-2-lite",
            "query": "test",
            "documents": ["doc1", "doc2"],
            "top_k": 1,
            "return_documents": True,
            "truncation": False,
        }
        ir = self.converter.request_from_provider(voyage_request)
        # top_k mapped to canonical top_n
        assert ir["top_n"] == 1
        assert ir["return_documents"] is True
        assert ir["truncation"] is False

    def test_response_to_provider_uses_data(self) -> None:
        ir = self.converter.response_from_provider(VOYAGE_RESPONSE)
        provider = self.converter.response_to_provider(ir)
        assert provider["object"] == "list"
        assert "data" in provider
        assert "results" not in provider
        assert len(provider["data"]) == 2
        assert provider["data"][0]["document"] == "Paris is the capital of France."
        assert provider["usage"]["total_tokens"] == 32

    def test_response_roundtrip(self) -> None:
        ir = self.converter.response_from_provider(VOYAGE_RESPONSE)
        provider = self.converter.response_to_provider(ir)
        ir_back = self.converter.response_from_provider(provider)
        assert ir_back["model"] == ir["model"]
        assert len(ir_back["results"]) == len(ir["results"])
        for orig, back in zip(ir["results"], ir_back["results"], strict=True):
            assert orig["index"] == back["index"]
            assert orig["relevance_score"] == pytest.approx(back["relevance_score"])

    def test_truncation_roundtrip(self) -> None:
        ir = IRRerankRequest(
            model="rerank-2-lite",
            query="test",
            documents=[RerankDocument(text="doc")],
            truncation=False,
        )
        provider, _ = self.converter.request_to_provider(ir)
        assert provider["truncation"] is False
        ir_back = self.converter.request_from_provider(provider)
        assert ir_back["truncation"] is False


# ============================================================================
# Cross-provider round-trip tests
# ============================================================================


class TestCrossProviderRoundtrip:
    """Test converting responses from one provider format to another via IR."""

    def test_jina_to_cohere(self) -> None:
        jina = JinaRerankConverter()
        cohere = CohereRerankConverter()
        ir = jina.response_from_provider(JINA_RESPONSE)
        cohere_resp = cohere.response_to_provider(ir)
        assert "results" in cohere_resp
        assert len(cohere_resp["results"]) == 2
        assert cohere_resp["results"][0]["index"] == 0

    def test_cohere_to_voyage(self) -> None:
        cohere = CohereRerankConverter()
        voyage = VoyageRerankConverter()
        ir = cohere.response_from_provider(COHERE_RESPONSE)
        voyage_resp = voyage.response_to_provider(ir)
        assert voyage_resp["object"] == "list"
        assert "data" in voyage_resp
        assert len(voyage_resp["data"]) == 2

    def test_voyage_to_jina(self) -> None:
        voyage = VoyageRerankConverter()
        jina = JinaRerankConverter()
        ir = voyage.response_from_provider(VOYAGE_RESPONSE)
        jina_resp = jina.response_to_provider(ir)
        assert jina_resp["object"] == "list"
        assert "results" in jina_resp
        assert jina_resp["results"][0]["document"] == "Paris is the capital of France."

    def test_siliconflow_to_jina(self) -> None:
        cohere = CohereRerankConverter()
        jina = JinaRerankConverter()
        ir = cohere.response_from_provider(SILICONFLOW_RESPONSE)
        jina_resp = jina.response_to_provider(ir)
        assert jina_resp["usage"]["total_tokens"] == 54

    def test_request_jina_to_voyage(self) -> None:
        jina = JinaRerankConverter()
        voyage = VoyageRerankConverter()
        ir = jina.request_from_provider(SAMPLE_REQUEST)
        voyage_req, _ = voyage.request_to_provider(ir)
        assert voyage_req["top_k"] == 2
        assert "top_n" not in voyage_req
        assert voyage_req["documents"] == SAMPLE_REQUEST["documents"]
