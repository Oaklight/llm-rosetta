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

from .score_ops import build_context, get_option_texts, scores_to_answer


class RerankerDecisionConverter(BaseDecisionConverter):
    """Decision converter backed by rerank API relevance scores.

    Converts decision requests into rerank queries (one per question),
    then maps relevance scores → softmax → typed answers.

    The converter produces a batch of rerank requests via
    ``request_to_provider`` and parses a batch of rerank responses
    via ``response_from_provider``.  The wire format follows the
    Jina/Cohere rerank convention::

        # Request (one per question, batched in a list):
        {"query": "state + instructions", "documents": ["opt1", "opt2", ...], "model": "..."}

        # Response (list of rerank results, one per question):
        [{"results": [{"index": 0, "relevance_score": 0.85}, ...]}, ...]
    """

    _CONVERTER_TAG = "reranker_decision"

    def __init__(self, *, model: str = "") -> None:
        self._model = model

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
        for qid, q in questions.items():
            query_text = build_context(ir_request["state"], q["instructions"])
            documents = get_option_texts(q)
            queries.append(
                {
                    "query": query_text,
                    "documents": documents,
                    "model": model,
                    "_question_id": qid,
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
            answers[qid] = scores_to_answer(scores, q)

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
        usage: DecisionUsageInfo = {}
        if "total_tokens" in p_usage:
            usage["input_tokens"] = p_usage["total_tokens"]
        elif "prompt_tokens" in p_usage:
            usage["input_tokens"] = p_usage["prompt_tokens"]
        return usage

    @staticmethod
    def _build_ir_usage_to_p(ir_usage: DecisionUsageInfo) -> dict[str, Any]:
        result: dict[str, Any] = {}
        if "input_tokens" in ir_usage:
            result["total_tokens"] = ir_usage["input_tokens"]
        return result
