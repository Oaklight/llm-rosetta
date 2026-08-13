"""
LLM-Rosetta - Cohere Embedding Converter

Cohere v2 embed API has a fundamentally different structure:

Request:
  { model, texts: [...], input_type, embedding_types: ["float"], truncate }
  (uses "texts" instead of "input", and "input_type" instead of task)

Response:
  { id, texts, embeddings: { float: [[...], [...]] }, meta: { billed_units }, response_type }
  (embeddings keyed by type, each is a list of vectors)

Cohere input_type values:
  search_query → retrieval_query, search_document → retrieval_document,
  classification, clustering, image
"""

from __future__ import annotations

from typing import Any

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.base.embedding_converter import BaseEmbeddingConverter
from llm_rosetta.types.ir.embedding import (
    EmbeddingItem,
    EmbeddingTaskType,
    EmbeddingUsageInfo,
    IREmbeddingRequest,
    IREmbeddingResponse,
)

_COHERE_INPUT_TYPE_TO_IR: dict[str, EmbeddingTaskType] = {
    "search_query": "retrieval_query",
    "search_document": "retrieval_document",
    "classification": "classification",
    "clustering": "clustering",
}

_IR_TASK_TO_COHERE: dict[EmbeddingTaskType, str] = {
    "retrieval_query": "search_query",
    "retrieval_document": "search_document",
    "classification": "classification",
    "clustering": "clustering",
}


class CohereEmbeddingConverter(BaseEmbeddingConverter):
    _CONVERTER_TAG = "cohere_embedding"

    def _do_request_to_provider(
        self,
        ir_request: IREmbeddingRequest,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model": ir_request["model"],
            "texts": ir_request["input"],
        }
        if "task_type" in ir_request:
            cohere_type = _IR_TASK_TO_COHERE.get(ir_request["task_type"])
            if cohere_type:
                result["input_type"] = cohere_type
        enc = ir_request.get("encoding_format", "float")
        result["embedding_types"] = [enc]
        if "truncation" in ir_request:
            result["truncate"] = "START" if ir_request["truncation"] else "NONE"
        return result

    def _do_request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IREmbeddingRequest:
        texts = provider_request.get("texts", [])
        if isinstance(texts, str):
            texts = [texts]

        ir = IREmbeddingRequest(
            model=provider_request["model"],
            input=texts,
        )
        if "input_type" in provider_request:
            ir_task = _COHERE_INPUT_TYPE_TO_IR.get(provider_request["input_type"])
            if ir_task:
                ir["task_type"] = ir_task
        embed_types = provider_request.get("embedding_types", ["float"])
        if embed_types:
            ir["encoding_format"] = embed_types[0]
        if "truncate" in provider_request:
            ir["truncation"] = provider_request["truncate"] != "NONE"
        return ir

    def _do_response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IREmbeddingResponse:
        embeddings = provider_response.get("embeddings", {})

        # Cohere returns embeddings keyed by type: {float: [[...], [...]]}
        # Pick the first available type
        embed_type = next(iter(embeddings), "float")
        vectors = embeddings.get(embed_type, [])

        data = [EmbeddingItem(index=i, embedding=vec) for i, vec in enumerate(vectors)]

        ir = IREmbeddingResponse(
            object="list",
            model=provider_response.get("model", ""),
            data=data,
        )
        if embed_type != "float":
            ir["encoding_format"] = embed_type

        if "id" in provider_response:
            ir["id"] = provider_response["id"]

        meta = provider_response.get("meta", {})
        billed = meta.get("billed_units")
        if billed:
            usage = self._build_p_usage_to_ir(billed)
            if usage:
                ir["usage"] = usage

        return ir

    def _do_response_to_provider(
        self,
        ir_response: IREmbeddingResponse,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]:
        embed_type = ir_response.get("encoding_format", "float")
        vectors = [item["embedding"] for item in ir_response["data"]]

        result: dict[str, Any] = {
            "embeddings": {embed_type: vectors},
            "response_type": "embeddings_by_type",
        }
        if "id" in ir_response:
            result["id"] = ir_response["id"]
        if "usage" in ir_response:
            result["meta"] = {
                "billed_units": self._build_ir_usage_to_p(ir_response["usage"]),
            }
        return result

    @staticmethod
    def _build_p_usage_to_ir(p_usage: dict[str, Any]) -> EmbeddingUsageInfo:
        usage = EmbeddingUsageInfo()
        input_tokens = p_usage.get("input_tokens")
        if input_tokens is not None:
            usage["total_tokens"] = input_tokens
            usage["prompt_tokens"] = input_tokens
        return usage

    @staticmethod
    def _build_ir_usage_to_p(ir_usage: EmbeddingUsageInfo) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if "prompt_tokens" in ir_usage:
            result["input_tokens"] = ir_usage["prompt_tokens"]
        elif "total_tokens" in ir_usage:
            result["input_tokens"] = ir_usage["total_tokens"]
        return result
