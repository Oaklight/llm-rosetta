"""
LLM-Rosetta - OpenAI Embedding Converter

Near-identity converter — the IR embedding types follow OpenAI conventions.

OpenAI request:  { model, input: [...], encoding_format?, dimensions?, user? }
OpenAI response: { object:"list", data:[{object:"embedding", embedding:[...], index}], model, usage }
"""

from __future__ import annotations

from typing import Any

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.base.embedding_converter import BaseEmbeddingConverter
from llm_rosetta.types.ir.embedding import (
    EmbeddingItem,
    EmbeddingUsageInfo,
    IREmbeddingRequest,
    IREmbeddingResponse,
)


class OpenAIEmbeddingConverter(BaseEmbeddingConverter):
    _CONVERTER_TAG = "openai_embedding"

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
        if "encoding_format" in ir_request:
            result["encoding_format"] = ir_request["encoding_format"]
        if "dimensions" in ir_request:
            result["dimensions"] = ir_request["dimensions"]
        if "user" in ir_request:
            result["user"] = ir_request["user"]
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
        if "encoding_format" in provider_request:
            ir["encoding_format"] = provider_request["encoding_format"]
        if "dimensions" in provider_request:
            ir["dimensions"] = provider_request["dimensions"]
        if "user" in provider_request:
            ir["user"] = provider_request["user"]
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
            "object": "list",
            "data": data,
            "model": ir_response["model"],
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
        if "prompt_tokens" in ir_usage:
            result["prompt_tokens"] = ir_usage["prompt_tokens"]
        return result
