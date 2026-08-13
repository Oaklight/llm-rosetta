"""
LLM-Rosetta - Jina Embedding Converter

Jina follows OpenAI format with additional fields:
- Request: task (maps to EmbeddingTaskType), late_chunking, normalized
- Response: identical to OpenAI (object:"list", data:[...], usage)

Jina task values:
  retrieval.query → retrieval_query, retrieval.passage → retrieval_document,
  text-matching → semantic_similarity, classification, separation, clustering
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

_JINA_TASK_TO_IR: dict[str, EmbeddingTaskType] = {
    "retrieval.query": "retrieval_query",
    "retrieval.passage": "retrieval_document",
    "text-matching": "semantic_similarity",
    "classification": "classification",
    "separation": "clustering",
    "clustering": "clustering",
}

_IR_TASK_TO_JINA: dict[EmbeddingTaskType, str] = {
    "retrieval_query": "retrieval.query",
    "retrieval_document": "retrieval.passage",
    "semantic_similarity": "text-matching",
    "classification": "classification",
    "clustering": "clustering",
}


class JinaEmbeddingConverter(BaseEmbeddingConverter):
    _CONVERTER_TAG = "jina_embedding"

    def _do_request_to_provider(
        self,
        ir_request: IREmbeddingRequest,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model": ir_request["model"],
            "input": ir_request["input"],
        }
        if "task_type" in ir_request:
            jina_task = _IR_TASK_TO_JINA.get(ir_request["task_type"])
            if jina_task:
                result["task"] = jina_task
        if "dimensions" in ir_request:
            result["dimensions"] = ir_request["dimensions"]
        if "encoding_format" in ir_request:
            result["embedding_type"] = ir_request["encoding_format"]
        if "truncation" in ir_request:
            result["truncate"] = ir_request["truncation"]
        return result

    def _do_request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IREmbeddingRequest:
        inp = provider_request["input"]
        if isinstance(inp, str):
            inp = [inp]

        ir = IREmbeddingRequest(
            model=provider_request["model"],
            input=inp,
        )
        if "task" in provider_request:
            ir_task = _JINA_TASK_TO_IR.get(provider_request["task"])
            if ir_task:
                ir["task_type"] = ir_task
        if "dimensions" in provider_request:
            ir["dimensions"] = provider_request["dimensions"]
        if "embedding_type" in provider_request:
            ir["encoding_format"] = provider_request["embedding_type"]
        if "truncate" in provider_request:
            ir["truncation"] = provider_request["truncate"]
        return ir

    def _do_response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IREmbeddingResponse:
        data = [
            EmbeddingItem(index=item["index"], embedding=item["embedding"])
            for item in provider_response.get("data", [])
        ]

        ir = IREmbeddingResponse(
            object="list",
            model=provider_response.get("model", ""),
            data=data,
        )
        if "usage" in provider_response and provider_response["usage"]:
            usage = self._build_p_usage_to_ir(provider_response["usage"])
            if usage:
                ir["usage"] = usage
        return ir

    def _do_response_to_provider(
        self,
        ir_response: IREmbeddingResponse,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]:
        data = [
            {
                "object": "embedding",
                "index": item["index"],
                "embedding": item["embedding"],
            }
            for item in ir_response["data"]
        ]

        result: dict[str, Any] = {
            "model": ir_response["model"],
            "object": "list",
            "data": data,
        }
        if "usage" in ir_response:
            result["usage"] = self._build_ir_usage_to_p(ir_response["usage"])
        return result

    @staticmethod
    def _build_p_usage_to_ir(p_usage: dict[str, Any]) -> EmbeddingUsageInfo:
        usage = EmbeddingUsageInfo()
        if "total_tokens" in p_usage:
            usage["total_tokens"] = p_usage["total_tokens"]
        if "prompt_tokens" in p_usage:
            usage["prompt_tokens"] = p_usage["prompt_tokens"]
        return usage

    @staticmethod
    def _build_ir_usage_to_p(ir_usage: EmbeddingUsageInfo) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if "total_tokens" in ir_usage:
            result["total_tokens"] = ir_usage["total_tokens"]
        return result
