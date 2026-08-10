"""
LLM-Rosetta - Voyage Rerank Converter

Voyage AI rerank API format converter.
Uses OpenAI Embeddings-style response structure (data instead of results).

Voyage response format:
{
    "object": "list",
    "data": [
        {"relevance_score": 0.72, "index": 0, "document": "..."},
        ...
    ],
    "model": "rerank-2-lite",
    "usage": {"total_tokens": 32}
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


class VoyageRerankConverter(BaseRerankConverter):
    _CONVERTER_TAG = "voyage_rerank"

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
        # Voyage uses top_k, not top_n
        if "top_n" in ir_request:
            result["top_k"] = ir_request["top_n"]
        if "return_documents" in ir_request:
            result["return_documents"] = ir_request["return_documents"]
        if "truncation" in ir_request:
            result["truncation"] = ir_request["truncation"]
        return result

    def _do_request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IRRerankRequest:
        documents = provider_request["documents"]
        ir_docs = [
            RerankDocument(text=doc)
            if isinstance(doc, str)
            else RerankDocument(text=str(doc))
            for doc in documents
        ]

        ir_request = IRRerankRequest(
            model=provider_request["model"],
            query=provider_request["query"],
            documents=ir_docs,
        )
        # Voyage uses top_k, map to canonical top_n
        if "top_k" in provider_request:
            ir_request["top_n"] = provider_request["top_k"]
        if "return_documents" in provider_request:
            ir_request["return_documents"] = provider_request["return_documents"]
        if "truncation" in provider_request:
            ir_request["truncation"] = provider_request["truncation"]
        return ir_request

    def _do_response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IRRerankResponse:
        # Voyage uses "data" instead of "results"
        results: list[RerankResultItem] = []
        for item in provider_response.get("data", []):
            result_item = RerankResultItem(
                index=item["index"],
                relevance_score=item["relevance_score"],
            )
            # Voyage returns document as plain string
            if "document" in item and item["document"] is not None:
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
        # Voyage uses "data" instead of "results"
        data = []
        for item in ir_response["results"]:
            p_item: dict[str, Any] = {
                "index": item["index"],
                "relevance_score": item["relevance_score"],
            }
            if "document" in item:
                p_item["document"] = item["document"]["text"]
            data.append(p_item)

        result: dict[str, Any] = {
            "object": "list",
            "data": data,
            "model": ir_response["model"],
        }
        if "usage" in ir_response:
            result["usage"] = self._build_ir_usage_to_p(ir_response["usage"])
        return result

    @staticmethod
    def _build_p_usage_to_ir(p_usage: dict[str, Any]) -> RerankUsageInfo:
        usage = RerankUsageInfo()
        if "total_tokens" in p_usage:
            usage["total_tokens"] = p_usage["total_tokens"]
        return usage

    @staticmethod
    def _build_ir_usage_to_p(ir_usage: RerankUsageInfo) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if "total_tokens" in ir_usage:
            result["total_tokens"] = ir_usage["total_tokens"]
        return result
