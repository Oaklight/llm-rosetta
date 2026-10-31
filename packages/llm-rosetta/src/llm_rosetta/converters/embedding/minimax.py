"""
LLM-Rosetta - MiniMax Embedding Converter

MiniMax's embedding API, measured against ``api.minimaxi.com`` on 2026-10-31:

Request:
  { model, texts: [...], type: "db" | "query" }

  It uses ``texts`` (not ``input``) and requires ``type``; an OpenAI-shaped
  body answers ``base_resp.status_code 2013`` with
  ``binding: expr_path=texts, cause=missing required parameter``.
  ``dimensions`` is accepted but ignored — the vector stays 1536 wide.

Response:
  { vectors: [[...], ...], total_tokens, base_resp: {status_code, status_msg} }

An error is an HTTP **200** whose body carries ``base_resp.status_code != 0``
(2013 for invalid params, e.g. an unknown model); the HTTP status says
nothing.  :meth:`upstream_error` is what the gateway asks before converting.

Type mapping: MiniMax has two embedding types, ``db`` (document) and
``query``.  They map onto the IR's ``retrieval_document`` and
``retrieval_query``; the other IR task types have no equivalent, so a request
carrying one is sent as ``db`` — the document default the vendor documents for
stored text.
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

_MINIMAX_TYPE_TO_IR: dict[str, EmbeddingTaskType] = {
    "db": "retrieval_document",
    "query": "retrieval_query",
}

_IR_TASK_TO_MINIMAX: dict[EmbeddingTaskType, str] = {
    "retrieval_document": "db",
    "retrieval_query": "query",
}

#: The vendor's default type for text that is stored, not searched with.
_MINIMAX_DEFAULT_TYPE = "db"


class MiniMaxEmbeddingConverter(BaseEmbeddingConverter):
    """Convert between MiniMax's ``texts``/``vectors`` dialect and the IR."""

    _CONVERTER_TAG = "minimax_embedding"

    @classmethod
    def upstream_error(cls, body: dict[str, Any] | None) -> str | None:
        """Return MiniMax's error message when *body* carries one.

        The vendor answers HTTP 200 for parameter errors and puts the verdict
        in ``base_resp.status_code``; without this check the gateway would
        report an empty success.
        """
        if not isinstance(body, dict):
            return None
        base = body.get("base_resp")
        if not isinstance(base, dict):
            return None
        code = base.get("status_code")
        if code in (None, 0):
            return None
        return f"MiniMax error {code}: {base.get('status_msg') or 'unknown error'}"

    def _do_request_to_provider(
        self,
        ir_request: IREmbeddingRequest,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]:
        texts = ir_request["input"]
        if isinstance(texts, str):
            texts = [texts]
        task_type = ir_request.get("task_type")
        return {
            "model": ir_request["model"],
            "texts": list(texts),
            "type": _IR_TASK_TO_MINIMAX.get(task_type, _MINIMAX_DEFAULT_TYPE),
        }

    def _do_request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IREmbeddingRequest:
        texts = provider_request.get("texts", [])
        if isinstance(texts, str):
            texts = [texts]
        ir = IREmbeddingRequest(model=provider_request["model"], input=texts)
        task_type = _MINIMAX_TYPE_TO_IR.get(provider_request.get("type", ""))
        if task_type:
            ir["task_type"] = task_type
        return ir

    def _do_response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IREmbeddingResponse:
        vectors = provider_response.get("vectors") or []
        # The vendor does not echo the model; the pipeline supplies it.
        model = provider_response.get("model") or str(context.metadata.get("model", ""))
        ir = IREmbeddingResponse(
            object="list",
            model=model,
            data=[
                EmbeddingItem(index=i, embedding=vec) for i, vec in enumerate(vectors)
            ],
        )
        # MiniMax reports the token count at the top level, not under a
        # ``usage`` object; shape it the way the usage hook expects.
        total_tokens = provider_response.get("total_tokens")
        if total_tokens is not None:
            usage = self._build_p_usage_to_ir({"total_tokens": total_tokens})
            if usage:
                ir["usage"] = usage
        return ir

    def _do_response_to_provider(
        self,
        ir_response: IREmbeddingResponse,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]:
        result: dict[str, Any] = {
            "vectors": [item["embedding"] for item in ir_response["data"]],
            "base_resp": {"status_code": 0, "status_msg": "success"},
        }
        if "usage" in ir_response:
            usage = self._build_ir_usage_to_p(ir_response["usage"])
            if "total_tokens" in usage:
                result["total_tokens"] = usage["total_tokens"]
        return result

    @staticmethod
    def _build_p_usage_to_ir(p_usage: dict[str, Any]) -> EmbeddingUsageInfo:
        usage = EmbeddingUsageInfo()
        total = p_usage.get("total_tokens")
        if total is not None:
            usage["total_tokens"] = total
            usage["prompt_tokens"] = total
        return usage

    @staticmethod
    def _build_ir_usage_to_p(ir_usage: EmbeddingUsageInfo) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if "prompt_tokens" in ir_usage:
            result["total_tokens"] = ir_usage["prompt_tokens"]
        elif "total_tokens" in ir_usage:
            result["total_tokens"] = ir_usage["total_tokens"]
        return result
