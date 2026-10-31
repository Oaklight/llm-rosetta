"""Unit tests for embedding converters using real API response fixtures."""

from __future__ import annotations


from llm_rosetta.converters.embedding import (
    CohereEmbeddingConverter,
    GoogleEmbeddingConverter,
    JinaEmbeddingConverter,
    MiniMaxEmbeddingConverter,
    OpenAIEmbeddingConverter,
    VoyageEmbeddingConverter,
)
from llm_rosetta.types.ir.embedding import (
    IREmbeddingRequest,
)

# ============================================================================
# Fixtures — captured from real API calls
# ============================================================================

OPENAI_RESPONSE = {
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [-0.07, -0.41, 0.35, 0.30],
            "index": 0,
        }
    ],
    "model": "text-embedding-3-small",
    "usage": {"prompt_tokens": 2, "total_tokens": 2},
}

JINA_RESPONSE = {
    "model": "jina-embeddings-v3",
    "object": "list",
    "usage": {"total_tokens": 17},
    "data": [
        {
            "object": "embedding",
            "index": 0,
            "embedding": [0.31, -0.16, 0.67, -0.11],
        }
    ],
}

COHERE_RESPONSE = {
    "id": "c91871ab-919f-47fc-b3c7-598e33333164",
    "texts": ["hello world"],
    "embeddings": {"float": [[-0.007, -0.012, 0.034]]},
    "meta": {
        "api_version": {"version": "2"},
        "billed_units": {"input_tokens": 1, "image_tokens": 0},
    },
    "response_type": "embeddings_by_type",
}

VOYAGE_RESPONSE = {
    "object": "list",
    "data": [
        {
            "object": "embedding",
            "embedding": [-0.035, -0.028, 0.060],
            "index": 0,
        }
    ],
    "model": "voyage-3-lite",
    "usage": {"total_tokens": 1},
}

GOOGLE_SINGLE_RESPONSE = {
    "embedding": {"values": [0.12, -0.34, 0.56]},
    "usageMetadata": {"promptTokenCount": 3, "totalTokenCount": 3},
}

GOOGLE_BATCH_RESPONSE = {
    "embeddings": [
        {"values": [0.12, -0.34]},
        {"values": [0.56, 0.78]},
    ],
}

SAMPLE_REQUEST = {
    "model": "test-model",
    "input": ["hello world"],
}

# Captured from a live `embo-01` call on 2026-10-31; the vectors are trimmed
# from 1536 floats to 2 to keep the fixture readable — the shape is the
# vendor's, including the missing `model` echo and the top-level token count.
MINIMAX_RESPONSE = {
    "vectors": [[-0.01, -0.02], [0.21, 0.22]],
    "total_tokens": 7,
    "base_resp": {"status_code": 0, "status_msg": "success"},
}

# An unknown model: HTTP 200 with the verdict in base_resp.
MINIMAX_ERROR_RESPONSE = {
    "vectors": None,
    "total_tokens": None,
    "base_resp": {"status_code": 2013, "status_msg": "invalid params"},
}


# ============================================================================
# OpenAI converter tests
# ============================================================================


class TestOpenAIEmbeddingConverter:
    def setup_method(self) -> None:
        self.converter = OpenAIEmbeddingConverter()

    def test_response_from_provider(self) -> None:
        ir = self.converter.response_from_provider(OPENAI_RESPONSE)
        assert ir["object"] == "list"
        assert ir["model"] == "text-embedding-3-small"
        assert len(ir["data"]) == 1
        assert ir["data"][0]["index"] == 0
        assert len(ir["data"][0]["embedding"]) == 4
        assert ir["usage"]["total_tokens"] == 2
        assert ir["usage"]["prompt_tokens"] == 2

    def test_request_to_provider(self) -> None:
        ir = IREmbeddingRequest(model="test", input=["hello"], dimensions=8)
        result, warnings = self.converter.request_to_provider(ir)
        assert result["model"] == "test"
        assert result["input"] == ["hello"]
        assert result["dimensions"] == 8
        assert warnings == []

    def test_request_from_provider_string_input(self) -> None:
        ir = self.converter.request_from_provider(
            {"model": "test", "input": "single string"}
        )
        assert ir["input"] == ["single string"]

    def test_response_roundtrip(self) -> None:
        ir = self.converter.response_from_provider(OPENAI_RESPONSE)
        provider = self.converter.response_to_provider(ir)
        ir_back = self.converter.response_from_provider(provider)
        assert ir_back["data"][0]["embedding"] == ir["data"][0]["embedding"]
        assert ir_back["usage"]["total_tokens"] == ir["usage"]["total_tokens"]


# ============================================================================
# Jina converter tests
# ============================================================================


class TestJinaEmbeddingConverter:
    def setup_method(self) -> None:
        self.converter = JinaEmbeddingConverter()

    def test_response_from_provider(self) -> None:
        ir = self.converter.response_from_provider(JINA_RESPONSE)
        assert ir["object"] == "list"
        assert ir["model"] == "jina-embeddings-v3"
        assert len(ir["data"]) == 1
        assert ir["usage"]["total_tokens"] == 17

    def test_task_mapping_to_provider(self) -> None:
        ir = IREmbeddingRequest(
            model="jina-embeddings-v3",
            input=["hello"],
            task_type="retrieval_query",
        )
        result, _ = self.converter.request_to_provider(ir)
        assert result["task"] == "retrieval.query"

    def test_task_mapping_from_provider(self) -> None:
        ir = self.converter.request_from_provider(
            {"model": "test", "input": ["hello"], "task": "retrieval.passage"}
        )
        assert ir["task_type"] == "retrieval_document"

    def test_encoding_format_mapping(self) -> None:
        ir = IREmbeddingRequest(model="test", input=["hello"], encoding_format="base64")
        result, _ = self.converter.request_to_provider(ir)
        assert result["embedding_type"] == "base64"


# ============================================================================
# Voyage converter tests
# ============================================================================


class TestVoyageEmbeddingConverter:
    def setup_method(self) -> None:
        self.converter = VoyageEmbeddingConverter()

    def test_response_from_provider(self) -> None:
        ir = self.converter.response_from_provider(VOYAGE_RESPONSE)
        assert ir["object"] == "list"
        assert ir["model"] == "voyage-3-lite"
        assert len(ir["data"]) == 1
        assert ir["usage"]["total_tokens"] == 1

    def test_input_type_to_provider(self) -> None:
        ir = IREmbeddingRequest(
            model="voyage-3-lite",
            input=["hello"],
            task_type="retrieval_query",
        )
        result, _ = self.converter.request_to_provider(ir)
        assert result["input_type"] == "query"

    def test_input_type_from_provider(self) -> None:
        ir = self.converter.request_from_provider(
            {"model": "test", "input": ["hello"], "input_type": "document"}
        )
        assert ir["task_type"] == "retrieval_document"

    def test_output_dtype_mapping(self) -> None:
        ir = IREmbeddingRequest(model="test", input=["hello"], encoding_format="int8")
        result, _ = self.converter.request_to_provider(ir)
        assert result["output_dtype"] == "int8"


# ============================================================================
# Cohere converter tests
# ============================================================================


class TestCohereEmbeddingConverter:
    def setup_method(self) -> None:
        self.converter = CohereEmbeddingConverter()

    def test_response_from_provider(self) -> None:
        ir = self.converter.response_from_provider(COHERE_RESPONSE)
        assert ir["object"] == "list"
        assert len(ir["data"]) == 1
        assert ir["data"][0]["index"] == 0
        assert ir["data"][0]["embedding"] == [-0.007, -0.012, 0.034]
        assert ir["id"] == "c91871ab-919f-47fc-b3c7-598e33333164"
        assert ir["usage"]["total_tokens"] == 1

    def test_request_to_provider(self) -> None:
        ir = IREmbeddingRequest(
            model="embed-v4.0",
            input=["hello world"],
            task_type="retrieval_query",
        )
        result, _ = self.converter.request_to_provider(ir)
        assert result["texts"] == ["hello world"]
        assert "input" not in result
        assert result["input_type"] == "search_query"
        assert result["embedding_types"] == ["float"]

    def test_request_from_provider(self) -> None:
        ir = self.converter.request_from_provider(
            {
                "model": "embed-v4.0",
                "texts": ["hello"],
                "input_type": "search_document",
                "embedding_types": ["float"],
            }
        )
        assert ir["input"] == ["hello"]
        assert ir["task_type"] == "retrieval_document"
        assert ir["encoding_format"] == "float"

    def test_response_to_provider(self) -> None:
        ir = self.converter.response_from_provider(COHERE_RESPONSE)
        provider = self.converter.response_to_provider(ir)
        assert "embeddings" in provider
        assert "float" in provider["embeddings"]
        assert provider["response_type"] == "embeddings_by_type"

    def test_truncation_mapping(self) -> None:
        ir = IREmbeddingRequest(model="test", input=["hello"], truncation=True)
        result, _ = self.converter.request_to_provider(ir)
        assert result["truncate"] == "END"

        ir2 = IREmbeddingRequest(model="test", input=["hello"], truncation=False)
        result2, _ = self.converter.request_to_provider(ir2)
        assert result2["truncate"] == "NONE"


# ============================================================================
# MiniMax converter tests
# ============================================================================


class TestMiniMaxEmbeddingConverter:
    def setup_method(self) -> None:
        self.converter = MiniMaxEmbeddingConverter()

    def test_response_from_provider(self) -> None:
        ir = self.converter.response_from_provider(MINIMAX_RESPONSE)
        assert ir["object"] == "list"
        assert len(ir["data"]) == 2
        assert ir["data"][0]["index"] == 0
        assert ir["data"][1]["embedding"] == [0.21, 0.22]
        assert ir["usage"]["total_tokens"] == 7
        assert ir["usage"]["prompt_tokens"] == 7

    def test_model_falls_back_to_the_context(self) -> None:
        # MiniMax does not echo the model; the pipeline supplies it.
        ctx = MiniMaxEmbeddingConverter.create_conversion_context()
        ctx.metadata["model"] = "embo-01"
        ir = self.converter.response_from_provider(MINIMAX_RESPONSE, context=ctx)
        assert ir["model"] == "embo-01"

    def test_request_to_provider_uses_texts_and_type(self) -> None:
        ir = IREmbeddingRequest(model="embo-01", input=["hello", "world"])
        result, _ = self.converter.request_to_provider(ir)
        assert result["texts"] == ["hello", "world"]
        assert "input" not in result
        # The vendor requires `type`; a request with no task is a document.
        assert result["type"] == "db"

    def test_task_type_mapping(self) -> None:
        query = IREmbeddingRequest(
            model="embo-01", input=["hi"], task_type="retrieval_query"
        )
        assert self.converter.request_to_provider(query)[0]["type"] == "query"

        document = IREmbeddingRequest(
            model="embo-01", input=["hi"], task_type="retrieval_document"
        )
        assert self.converter.request_to_provider(document)[0]["type"] == "db"

        # MiniMax has no equivalent for the other IR task types.
        other = IREmbeddingRequest(
            model="embo-01", input=["hi"], task_type="classification"
        )
        assert self.converter.request_to_provider(other)[0]["type"] == "db"

    def test_request_from_provider(self) -> None:
        ir = self.converter.request_from_provider(
            {"model": "embo-01", "texts": ["hello"], "type": "query"}
        )
        assert ir["input"] == ["hello"]
        assert ir["task_type"] == "retrieval_query"

    def test_response_roundtrip(self) -> None:
        ir = self.converter.response_from_provider(MINIMAX_RESPONSE)
        provider = self.converter.response_to_provider(ir)
        assert provider["base_resp"] == {"status_code": 0, "status_msg": "success"}
        ir_back = self.converter.response_from_provider(provider)
        assert ir_back["data"][0]["embedding"] == ir["data"][0]["embedding"]
        assert ir_back["usage"]["total_tokens"] == ir["usage"]["total_tokens"]

    def test_upstream_error_reads_base_resp(self) -> None:
        # A refusal is HTTP 200 with a non-zero status in the body.
        assert (
            self.converter.upstream_error(MINIMAX_ERROR_RESPONSE)
            == "MiniMax error 2013: invalid params"
        )
        assert self.converter.upstream_error(MINIMAX_RESPONSE) is None
        assert self.converter.upstream_error({"vectors": [[0.1]]}) is None
        assert self.converter.upstream_error(None) is None

    def test_cross_format_request_and_response(self) -> None:
        openai = OpenAIEmbeddingConverter()
        ir = openai.request_from_provider(
            {"model": "embo-01", "input": ["hello"], "dimensions": 8}
        )
        body, _ = self.converter.request_to_provider(ir)
        assert body == {"model": "embo-01", "texts": ["hello"], "type": "db"}

        ir_response = self.converter.response_from_provider(MINIMAX_RESPONSE)
        out = openai.response_to_provider(ir_response)
        assert out["data"][0]["embedding"] == [-0.01, -0.02]
        assert out["usage"]["total_tokens"] == 7


# ============================================================================
# Google GenAI converter tests
# ============================================================================


class TestGoogleEmbeddingConverter:
    def setup_method(self) -> None:
        self.converter = GoogleEmbeddingConverter()

    def test_response_from_provider_single(self) -> None:
        ir = self.converter.response_from_provider(GOOGLE_SINGLE_RESPONSE)
        assert ir["object"] == "list"
        assert len(ir["data"]) == 1
        assert ir["data"][0]["embedding"] == [0.12, -0.34, 0.56]
        assert ir["usage"]["prompt_tokens"] == 3
        assert ir["usage"]["total_tokens"] == 3

    def test_response_from_provider_batch(self) -> None:
        ir = self.converter.response_from_provider(GOOGLE_BATCH_RESPONSE)
        assert [item["embedding"] for item in ir["data"]] == [
            [0.12, -0.34],
            [0.56, 0.78],
        ]
        # batchEmbedContents returns no usageMetadata
        assert "usage" not in ir

    def test_request_to_provider_single(self) -> None:
        ir = IREmbeddingRequest(
            model="text-embedding-004",
            input=["hello world"],
            task_type="retrieval_query",
            dimensions=256,
            truncation=True,
            provider_extensions={"title": "doc"},
        )
        result, _ = self.converter.request_to_provider(ir)
        assert result == {
            "content": {"parts": [{"text": "hello world"}]},
            "taskType": "RETRIEVAL_QUERY",
            "outputDimensionality": 256,
            "autoTruncate": True,
            "title": "doc",
        }

    def test_batch_response_keeps_input_order(self) -> None:
        """Vector i must belong to input i — a transposition is silent."""
        ir = self.converter.response_from_provider(
            {
                "embeddings": [
                    {"values": [1.0]},
                    {"values": [2.0]},
                    {"values": [3.0]},
                ]
            }
        )
        assert [item["index"] for item in ir["data"]] == [0, 1, 2]
        assert [item["embedding"] for item in ir["data"]] == [[1.0], [2.0], [3.0]]

    def test_request_to_provider_batch(self) -> None:
        ir = IREmbeddingRequest(model="text-embedding-004", input=["a", "b", "c"])
        result, _ = self.converter.request_to_provider(ir)
        assert "content" not in result
        assert [request["model"] for request in result["requests"]] == [
            "models/text-embedding-004",
            "models/text-embedding-004",
            "models/text-embedding-004",
        ]
        assert [
            request["content"]["parts"][0]["text"] for request in result["requests"]
        ] == ["a", "b", "c"]

    def test_unsupported_task_type_warns(self) -> None:
        ir = IREmbeddingRequest(
            model="text-embedding-004",
            input=["hello"],
            task_type="code_retrieval_document",
        )
        result, warnings = self.converter.request_to_provider(ir)
        assert "taskType" not in result
        assert any("code_retrieval_document" in warning for warning in warnings)

    def test_request_from_provider(self) -> None:
        ir = self.converter.request_from_provider(
            {
                "model": "models/text-embedding-004",
                "content": {"parts": [{"text": "hello"}]},
                "taskType": "RETRIEVAL_DOCUMENT",
                "outputDimensionality": 128,
            }
        )
        assert ir["model"] == "text-embedding-004"
        assert ir["input"] == ["hello"]
        assert ir["task_type"] == "retrieval_document"
        assert ir["dimensions"] == 128

    def test_response_to_provider(self) -> None:
        ir = self.converter.response_from_provider(GOOGLE_BATCH_RESPONSE)
        provider = self.converter.response_to_provider(ir)
        assert provider["embeddings"] == [
            {"values": [0.12, -0.34]},
            {"values": [0.56, 0.78]},
        ]


# ============================================================================
# Cross-provider round-trip tests
# ============================================================================


class TestCrossProviderRoundtrip:
    def test_openai_to_cohere_response(self) -> None:
        openai = OpenAIEmbeddingConverter()
        cohere = CohereEmbeddingConverter()
        ir = openai.response_from_provider(OPENAI_RESPONSE)
        cohere_resp = cohere.response_to_provider(ir)
        assert "embeddings" in cohere_resp
        assert "float" in cohere_resp["embeddings"]
        assert len(cohere_resp["embeddings"]["float"]) == 1

    def test_cohere_to_openai_response(self) -> None:
        cohere = CohereEmbeddingConverter()
        openai = OpenAIEmbeddingConverter()
        ir = cohere.response_from_provider(COHERE_RESPONSE)
        openai_resp = openai.response_to_provider(ir)
        assert openai_resp["object"] == "list"
        assert len(openai_resp["data"]) == 1
        assert openai_resp["data"][0]["object"] == "embedding"

    def test_jina_to_voyage_response(self) -> None:
        jina = JinaEmbeddingConverter()
        voyage = VoyageEmbeddingConverter()
        ir = jina.response_from_provider(JINA_RESPONSE)
        voyage_resp = voyage.response_to_provider(ir)
        assert voyage_resp["object"] == "list"
        assert len(voyage_resp["data"]) == 1

    def test_request_openai_to_jina(self) -> None:
        openai = OpenAIEmbeddingConverter()
        jina = JinaEmbeddingConverter()
        ir = openai.request_from_provider(SAMPLE_REQUEST)
        jina_req, _ = jina.request_to_provider(ir)
        assert jina_req["input"] == ["hello world"]

    def test_request_cohere_to_voyage(self) -> None:
        cohere = CohereEmbeddingConverter()
        voyage = VoyageEmbeddingConverter()
        cohere_req = {
            "model": "embed-v4.0",
            "texts": ["hello"],
            "input_type": "search_query",
            "embedding_types": ["float"],
        }
        ir = cohere.request_from_provider(cohere_req)
        voyage_req, _ = voyage.request_to_provider(ir)
        assert voyage_req["input"] == ["hello"]
        assert voyage_req["input_type"] == "query"

    def test_empty_usage_not_set(self) -> None:
        openai = OpenAIEmbeddingConverter()
        response_with_empty_usage = {
            "object": "list",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.1]}],
            "model": "test",
            "usage": {},
        }
        ir = openai.response_from_provider(response_with_empty_usage)
        assert "usage" not in ir
