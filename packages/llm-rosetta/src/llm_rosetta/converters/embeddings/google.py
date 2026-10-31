"""LLM-Rosetta - Google GenAI Embedding Converter

Google's embedContent API differs from the OpenAI-shaped formats in three
ways that matter to the IR:

Request:
  Single  — ``POST /v1beta/models/{model}:embedContent``
            ``{content: {parts: [{text}]}, taskType, outputDimensionality, autoTruncate, title}``
  Batch   — ``POST /v1beta/models/{model}:batchEmbedContents``
            ``{requests: [{model: "models/{model}", content, taskType}, ...]}``
  The model lives in the URL (and, for batch, in every request), never at the
  top level of a single request.

Response:
  Single  — ``{embedding: {values: [...]}, usageMetadata: {promptTokenCount, totalTokenCount}}``
  Batch   — ``{embeddings: [{values: [...]}, ...]}`` (no usage metadata)
  There is no ``object``/``data``/``index`` wrapper and no model echo; the
  converter takes the model from the request context when the pipeline
  supplies it.

Only float output is supported: Google has no ``encoding_format``.  A caller
that requests another encoding gets a warning and a float result.
"""

from __future__ import annotations

import copy
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

# IR task type → Google taskType.  Google's enum is a strict subset of the IR
# vocabulary; values the IR has but Google does not are dropped (with a
# warning) rather than mapped onto a semantically different task.
_IR_TASK_TO_GOOGLE: dict[EmbeddingTaskType, str] = {
    "retrieval_query": "RETRIEVAL_QUERY",
    "retrieval_document": "RETRIEVAL_DOCUMENT",
    "semantic_similarity": "SEMANTIC_SIMILARITY",
    "classification": "CLASSIFICATION",
    "clustering": "CLUSTERING",
    "question_answering": "QUESTION_ANSWERING",
    "fact_verification": "FACT_VERIFICATION",
    "code_retrieval_query": "CODE_RETRIEVAL_QUERY",
    # Google has no CODE_RETRIEVAL_DOCUMENT; the closest documented value is
    # the query one, but remapping would change what the caller asked for, so
    # it is omitted.
}

_GOOGLE_TASK_TO_IR: dict[str, EmbeddingTaskType] = {
    google: ir for ir, google in _IR_TASK_TO_GOOGLE.items()
}


def _content(text: str) -> dict[str, Any]:
    """The minimal Google ``Content`` carrying one text part."""
    return {"parts": [{"text": text}]}


def _with_model_prefix(model: str) -> str:
    """Google batch requests take ``models/{id}``; accept either spelling."""
    return model if model.startswith("models/") else f"models/{model}"


class GoogleEmbeddingConverter(BaseEmbeddingConverter):
    _CONVERTER_TAG = "google_embedding"

    def _do_request_to_provider(
        self,
        ir_request: IREmbeddingRequest,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]:
        model = ir_request["model"]
        inputs = ir_request["input"]

        # Fields shared by every request in both shapes.
        shared: dict[str, Any] = {}
        task_type = ir_request.get("task_type")
        if task_type is not None:
            google_task = _IR_TASK_TO_GOOGLE.get(task_type)
            if google_task is None:
                context.warnings.append(
                    f"google embedding: task type {task_type!r} has no Google "
                    "equivalent; omitting taskType"
                )
            else:
                shared["taskType"] = google_task
        if "dimensions" in ir_request:
            shared["outputDimensionality"] = ir_request["dimensions"]
        if "truncation" in ir_request:
            shared["autoTruncate"] = ir_request["truncation"]

        encoding = ir_request.get("encoding_format")
        if encoding is not None and encoding != "float":
            context.warnings.append(
                f"google embedding: encoding_format {encoding!r} is not "
                "supported; returning float vectors"
            )
        if "user" in ir_request:
            context.warnings.append(
                "google embedding: 'user' has no Google equivalent; dropping it"
            )

        extensions = ir_request.get("provider_extensions") or {}
        title = extensions.get("title")
        if title:
            shared["title"] = title

        if len(inputs) == 1:
            return {"content": _content(inputs[0]), **shared}

        requests: list[dict[str, Any]] = []
        for text in inputs:
            request: dict[str, Any] = {
                "model": _with_model_prefix(model),
                "content": _content(text),
                **copy.deepcopy(shared),
            }
            requests.append(request)
        return {"requests": requests}

    def _do_request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IREmbeddingRequest:
        if "requests" in provider_request:
            requests = provider_request.get("requests") or []
            texts = [
                part.get("text", "")
                for request in requests
                for part in (request.get("content") or {}).get("parts", [])
            ]
            first = requests[0] if requests else {}
            model = str(first.get("model", "")).removeprefix("models/")
            shared = first
        else:
            parts = (provider_request.get("content") or {}).get("parts", [])
            texts = [part.get("text", "") for part in parts]
            model = str(provider_request.get("model", "")).removeprefix("models/")
            shared = provider_request

        ir = IREmbeddingRequest(model=model, input=texts)

        google_task = shared.get("taskType")
        if google_task is not None:
            ir_task = _GOOGLE_TASK_TO_IR.get(google_task)
            if ir_task is not None:
                ir["task_type"] = ir_task
        if "outputDimensionality" in shared:
            ir["dimensions"] = shared["outputDimensionality"]
        if "autoTruncate" in shared:
            ir["truncation"] = shared["autoTruncate"]
        if "title" in shared:
            ir["provider_extensions"] = {"title": shared["title"]}
        ir["encoding_format"] = "float"
        return ir

    def _do_response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IREmbeddingResponse:
        if "embeddings" in provider_response:
            vectors = [
                item.get("values", []) for item in provider_response["embeddings"]
            ]
        else:
            single = provider_response.get("embedding") or {}
            vectors = [single.get("values", [])]

        data = [
            EmbeddingItem(index=index, embedding=vector)
            for index, vector in enumerate(vectors)
        ]

        ir = IREmbeddingResponse(
            object="list",
            model=str(context.metadata.get("model", "")),
            data=data,
            encoding_format="float",
        )

        usage_metadata = provider_response.get("usageMetadata") or {}
        usage = self._build_p_usage_to_ir(usage_metadata)
        if usage:
            ir["usage"] = usage
        return ir

    def _do_response_to_provider(
        self,
        ir_response: IREmbeddingResponse,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]:
        vectors = [item["embedding"] for item in ir_response["data"]]
        usage = ir_response.get("usage") or {}

        if len(vectors) == 1:
            result: dict[str, Any] = {"embedding": {"values": vectors[0]}}
            usage_metadata = self._build_ir_usage_to_p(usage)
            if usage_metadata:
                result["usageMetadata"] = usage_metadata
            return result
        return {"embeddings": [{"values": vector} for vector in vectors]}

    @staticmethod
    def _build_p_usage_to_ir(p_usage: dict[str, Any]) -> EmbeddingUsageInfo:
        usage = EmbeddingUsageInfo()
        # ``totalTokenCount`` is the billed total; ``promptTokenCount`` is the
        # input count the IR calls ``prompt_tokens``.
        if "totalTokenCount" in p_usage:
            usage["total_tokens"] = p_usage["totalTokenCount"]
        if "promptTokenCount" in p_usage:
            usage["prompt_tokens"] = p_usage["promptTokenCount"]
        return usage

    @staticmethod
    def _build_ir_usage_to_p(ir_usage: EmbeddingUsageInfo) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if "total_tokens" in ir_usage:
            result["totalTokenCount"] = ir_usage["total_tokens"]
        if "prompt_tokens" in ir_usage:
            result["promptTokenCount"] = ir_usage["prompt_tokens"]
        return result
