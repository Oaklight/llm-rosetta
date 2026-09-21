"""
LLM-Rosetta - Reranker Decision Converter

Decision converter backed by rerank API relevance scores.

For each question, builds (query, document) pairs where query = state +
instructions and documents = option texts.  Rerank relevance scores are
softmaxed into probability distributions, then mapped to noul/choice/score
answers.

Requires a rerank endpoint that returns per-document relevance_score.
Compatible with Jina, Cohere, Voyage, and any rerank API returning
scored results.

The converter produces one rerank query per question.  Standard rerank
APIs accept a single query per request, so the caller/gateway is
expected to fan out the ``queries`` list into individual API calls and
collect the results in order.

Reference: oaklight/jev-explore model/src/reranker_scorer.py
"""

from __future__ import annotations

from typing import Any

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.base.decision_converter import BaseDecisionConverter
from llm_rosetta.types.ir.decision import (
    DecisionUsageInfo,
    IRDecisionRequest,
    IRDecisionResponse,
)

from .score_ops import (
    build_context,
    build_ir_usage_to_p,
    build_p_usage_to_ir,
    get_option_texts,
    scores_to_answer,
)


class RerankerDecisionConverter(BaseDecisionConverter):
    """Decision converter backed by rerank API relevance scores.

    Args:
        model: Default rerank model name.
        temperature: Softmax temperature for score sharpening.
    """

    _CONVERTER_TAG = "reranker_decision"

    def __init__(self, *, model: str = "", temperature: float = 1.0) -> None:
        self._model = model
        self._temperature = temperature

    # ==================== Request conversion ====================

    def _do_request_to_provider(
        self,
        ir_request: IRDecisionRequest,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]:
        questions = ir_request["questions"]
        context.options["_decision_questions"] = questions
        model = ir_request.get("model", self._model) or self._model

        queries: list[dict[str, Any]] = []
        for _qid, q in questions.items():
            query_text = build_context(ir_request["state"], q["instructions"])
            documents = get_option_texts(q)
            queries.append(
                {
                    "query": query_text,
                    "documents": documents,
                    "model": model,
                }
            )

        return {"queries": queries, "model": model}

    def _do_request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IRDecisionRequest:
        raise NotImplementedError(
            "RerankerDecisionConverter does not support rerank → IR request conversion."
        )

    # ==================== Response conversion ====================

    def _do_response_from_provider(
        self,
        provider_response: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IRDecisionResponse:
        questions = context.options.get("_decision_questions")
        if questions is None:
            raise ValueError(
                "No _decision_questions in context — was request_to_provider "
                "called with the same ConversionContext?"
            )

        rerank_results = provider_response.get("results", [])
        answers: dict[str, Any] = {}

        for i, (qid, q) in enumerate(questions.items()):
            if i >= len(rerank_results):
                break
            result = rerank_results[i]
            scored_docs = result.get("results", result.get("data", []))
            n_options = len(get_option_texts(q))
            scores = [0.0] * n_options
            for doc in scored_docs:
                idx = doc.get("index", 0)
                if idx < n_options:
                    scores[idx] = doc.get("relevance_score", 0.0)
            answers[qid] = scores_to_answer(scores, q, temperature=self._temperature)

        result_resp: IRDecisionResponse = {
            "object": "decision",
            "model": provider_response.get("model", ""),
            "answers": answers,
        }
        p_usage = provider_response.get("usage")
        if p_usage:
            result_resp["usage"] = self._build_p_usage_to_ir(p_usage)
        return result_resp

    def _do_response_to_provider(
        self,
        ir_response: IRDecisionResponse,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "RerankerDecisionConverter does not support IR → rerank "
            "response conversion."
        )

    # ==================== Usage conversion ====================

    @staticmethod
    def _build_p_usage_to_ir(p_usage: dict[str, Any]) -> DecisionUsageInfo:
        return build_p_usage_to_ir(p_usage)

    @staticmethod
    def _build_ir_usage_to_p(ir_usage: DecisionUsageInfo) -> dict[str, Any]:
        return build_ir_usage_to_p(ir_usage)
