"""
LLM-Rosetta - Cohere Rerank Converter

Cohere Rerank API (v2) format converter.
Also handles the Siliconflow variant (richer meta.tokens + meta.billed_units).

Cohere response format:
{
    "id": "...",
    "results": [{"index": 0, "relevance_score": 0.89}],
    "meta": {
        "api_version": {"version": "2"},
        "billed_units": {"search_units": 1}
    }
}

Siliconflow variant:
{
    "id": "...",
    "results": [{"index": 0, "relevance_score": 0.99, "document": null}],
    "meta": {
        "tokens": {"input_tokens": 54, "output_tokens": 0},
        "billed_units": {"input_tokens": 54, "output_tokens": 0, "search_units": 0, ...}
    }
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


class CohereRerankConverter(BaseRerankConverter):
    _CONVERTER_TAG = "cohere_rerank"

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
        if "max_tokens_per_doc" in ir_request:
            result["max_tokens_per_doc"] = ir_request["max_tokens_per_doc"]
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
        if "top_n" in provider_request:
            ir_request["top_n"] = provider_request["top_n"]
        if "max_tokens_per_doc" in provider_request:
            ir_request["max_tokens_per_doc"] = provider_request["max_tokens_per_doc"]
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
            # Siliconflow sends document: null explicitly
            doc = item.get("document")
            if doc is not None:
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
        if "id" in provider_response:
            ir_response["id"] = provider_response["id"]

        # Extract usage from meta.tokens (Cohere v4 / Siliconflow)
        meta = provider_response.get("meta", {})
        tokens = meta.get("tokens")
        if tokens:
            ir_response["usage"] = self._build_p_usage_to_ir(tokens)

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
            results.append(p_item)

        result: dict[str, Any] = {"results": results}
        if "id" in ir_response:
            result["id"] = ir_response["id"]
        if "usage" in ir_response:
            result["meta"] = {
                "tokens": self._build_ir_usage_to_p(ir_response["usage"]),
            }
        return result

    @staticmethod
    def _build_p_usage_to_ir(p_usage: dict[str, Any]) -> RerankUsageInfo:
        usage = RerankUsageInfo()
        # meta.tokens uses input_tokens, map to our canonical names
        input_tokens = p_usage.get("input_tokens")
        if input_tokens is not None:
            usage["total_tokens"] = input_tokens
            usage["prompt_tokens"] = input_tokens
        if "cached_tokens" in p_usage:
            usage["cached_tokens"] = p_usage["cached_tokens"]
        return usage

    @staticmethod
    def _build_ir_usage_to_p(ir_usage: RerankUsageInfo) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if "prompt_tokens" in ir_usage:
            result["input_tokens"] = ir_usage["prompt_tokens"]
        elif "total_tokens" in ir_usage:
            result["input_tokens"] = ir_usage["total_tokens"]
        result["output_tokens"] = 0
        return result
