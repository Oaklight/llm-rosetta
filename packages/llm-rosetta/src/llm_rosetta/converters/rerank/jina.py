"""
LLM-Rosetta - Jina Rerank Converter

Jina Rerank API format converter.
Also compatible with GPUStack, vLLM (/v1/rerank), Xinference, llama-box.

Jina response format:
{
    "model": "jina-reranker-v2-base-multilingual",
    "object": "list",
    "usage": {"total_tokens": 54},
    "results": [
        {"index": 0, "relevance_score": 0.84, "document": "..."},
        ...
    ]
}
"""

from __future__ import annotations

from typing import Any

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.base.rerank_converter import BaseRerankConverter
from llm_rosetta.types.ir.rerank import (
    IRRerankRequest,
    IRRerankResponse,
    RerankDocument,
    RerankResultItem,
    RerankUsageInfo,
)


class JinaRerankConverter(BaseRerankConverter):
    _CONVERTER_TAG = "jina_rerank"

    def _do_request_to_provider(
        self,
        ir_request: IRRerankRequest,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "model": ir_request["model"],
            "query": ir_request["query"],
            "documents": [doc["text"] for doc in ir_request["documents"]],
        }
        if "top_n" in ir_request:
            result["top_n"] = ir_request["top_n"]
        if "return_documents" in ir_request:
            result["return_documents"] = ir_request["return_documents"]
        return result

    def _do_request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IRRerankRequest:
        documents = provider_request["documents"]
        ir_docs: list[RerankDocument] = []
        for doc in documents:
            if isinstance(doc, str):
                ir_docs.append(RerankDocument(text=doc))
            elif isinstance(doc, dict) and "text" in doc:
                ir_docs.append(RerankDocument(text=doc["text"]))
            else:
                context.warnings.append(
                    f"Unexpected document type {type(doc).__name__}, coercing to str"
                )
                ir_docs.append(RerankDocument(text=str(doc)))

        ir_request = IRRerankRequest(
            model=provider_request["model"],
            query=provider_request["query"],
            documents=ir_docs,
        )
        if "top_n" in provider_request:
            ir_request["top_n"] = provider_request["top_n"]
        if "return_documents" in provider_request:
            ir_request["return_documents"] = provider_request["return_documents"]
        return ir_request

    def _do_response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IRRerankResponse:
        results: list[RerankResultItem] = []
        for item in provider_response.get("results", []):
            result_item = RerankResultItem(
                index=item["index"],
                relevance_score=item["relevance_score"],
            )
            # Jina returns document as plain string, not {text: "..."}
            if "document" in item:
                doc = item["document"]
                if isinstance(doc, str):
                    result_item["document"] = RerankDocument(text=doc)
                elif isinstance(doc, dict) and "text" in doc:
                    result_item["document"] = RerankDocument(text=doc["text"])
            results.append(result_item)

        ir_response = IRRerankResponse(
            object="rerank",
            model=provider_response.get("model", ""),
            results=results,
        )
        if "usage" in provider_response and provider_response["usage"]:
            usage = self._build_p_usage_to_ir(provider_response["usage"])
            if usage:
                ir_response["usage"] = usage
        return ir_response

    def _do_response_to_provider(
        self,
        ir_response: IRRerankResponse,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]:
        results = []
        for item in ir_response["results"]:
            p_item: dict[str, Any] = {
                "index": item["index"],
                "relevance_score": item["relevance_score"],
            }
            if "document" in item:
                p_item["document"] = item["document"]["text"]
            results.append(p_item)

        result: dict[str, Any] = {
            "model": ir_response["model"],
            "object": "list",
            "results": results,
        }
        if "usage" in ir_response:
            result["usage"] = self._build_ir_usage_to_p(ir_response["usage"])
        return result

    @staticmethod
    def _build_p_usage_to_ir(p_usage: dict[str, Any]) -> RerankUsageInfo:
        usage = RerankUsageInfo()
        if "total_tokens" in p_usage:
            usage["total_tokens"] = p_usage["total_tokens"]
        if "prompt_tokens" in p_usage:
            usage["prompt_tokens"] = p_usage["prompt_tokens"]
        return usage

    @staticmethod
    def _build_ir_usage_to_p(ir_usage: RerankUsageInfo) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if "total_tokens" in ir_usage:
            result["total_tokens"] = ir_usage["total_tokens"]
        if "prompt_tokens" in ir_usage:
            result["prompt_tokens"] = ir_usage["prompt_tokens"]
        return result
