"""
LLM-Rosetta - Embedding Decision Converter

Decision converter backed by embedding API cosine similarity.

For each question, embeds the context (state + instructions) and each
option text, computes cosine similarity, then softmaxes into probability
distributions mapped to noul/choice/score answers.

Requires an embedding endpoint that returns vector arrays.
Compatible with OpenAI, Cohere, Voyage, Jina, and any embedding API.

Reference: oaklight/jev-explore model/src/encoder_scorer.py
"""

from __future__ import annotations

import math
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


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    if len(a) != len(b):
        raise ValueError(f"Embedding dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class EmbeddingDecisionConverter(BaseDecisionConverter):
    """Decision converter backed by embedding cosine similarity.

    Converts decision requests into embedding queries, then maps
    cosine similarity scores → softmax → typed answers.

    The converter produces a single batched embedding request via
    ``request_to_provider``.  The ``input`` list is laid out as::

        [ctx1, opt1a, opt1b, ctx2, opt2a, opt2b, opt2c, ...]

    The layout (which indices belong to which question) is stored
    in ``context.options["_embedding_layout"]`` for response parsing.

    Args:
        model: Default embedding model name.
        temperature: Softmax temperature for score sharpening.
            Cosine similarities are typically in a narrow range,
            so lower temperatures (e.g. 0.1) produce more peaked
            distributions.
    """

    _CONVERTER_TAG = "embedding_decision"

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

        texts: list[str] = []
        layout: list[tuple[str, int]] = []

        for qid, q in questions.items():
            ctx_text = build_context(ir_request["state"], q["instructions"])
            options = get_option_texts(q)
            texts.append(ctx_text)
            for opt in options:
                texts.append(opt)
            layout.append((qid, len(options)))

        context.options["_embedding_layout"] = layout

        return {"input": texts, "model": model}

    def _do_request_from_provider(
        self,
        provider_request: dict[str, Any],
        *,
        context: ConversionContext,
    ) -> IRDecisionRequest:
        raise NotImplementedError(
            "EmbeddingDecisionConverter does not support embedding → IR "
            "request conversion."
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
        layout = context.options.get("_embedding_layout", [])

        data = provider_response.get("data", [])
        embeddings: list[list[float] | None] = [None] * len(data)
        for item in data:
            idx = item.get("index", 0)
            if idx < len(embeddings):
                embeddings[idx] = item.get("embedding", [])

        answers: dict[str, Any] = {}
        pos = 0
        for qid, n_options in layout:
            if pos >= len(embeddings):
                break
            ctx_emb = embeddings[pos]
            pos += 1
            if ctx_emb is None:
                pos += n_options
                continue
            scores: list[float] = []
            for _j in range(n_options):
                opt_emb = embeddings[pos] if pos < len(embeddings) else None
                pos += 1
                if opt_emb is not None:
                    scores.append(_cosine_similarity(ctx_emb, opt_emb))
                else:
                    scores.append(0.0)
            q = questions[qid]
            answers[qid] = scores_to_answer(scores, q, temperature=self._temperature)

        result: IRDecisionResponse = {
            "object": "decision",
            "model": provider_response.get("model", ""),
            "answers": answers,
        }
        p_usage = provider_response.get("usage")
        if p_usage:
            result["usage"] = self._build_p_usage_to_ir(p_usage)
        return result

    def _do_response_to_provider(
        self,
        ir_response: IRDecisionResponse,
        *,
        context: ConversionContext,
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "EmbeddingDecisionConverter does not support IR → embedding "
            "response conversion."
        )

    # ==================== Usage conversion ====================

    @staticmethod
    def _build_p_usage_to_ir(p_usage: dict[str, Any]) -> DecisionUsageInfo:
        return build_p_usage_to_ir(p_usage)

    @staticmethod
    def _build_ir_usage_to_p(ir_usage: DecisionUsageInfo) -> dict[str, Any]:
        return build_ir_usage_to_p(ir_usage)
