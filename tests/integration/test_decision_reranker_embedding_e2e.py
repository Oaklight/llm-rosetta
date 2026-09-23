"""
Decision Converter E2E — Reranker and Embedding backends

Tests RerankerDecisionConverter and EmbeddingDecisionConverter against
real rerank/embedding APIs.  These are baseline-quality backends —
signal is weaker than LLM-backed converters but the conversion pipeline
must work end-to-end.

Requires API keys in .env (see project root).

Usage:
    conda activate llm-rosetta
    pytest tests/integration/test_decision_reranker_embedding_e2e.py -v -s
"""

from __future__ import annotations

import os
from typing import Any

import dotenv
import httpx
import pytest

dotenv.load_dotenv(
    os.path.join(os.path.dirname(__file__), "..", "..", ".env"), override=True
)

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.decision.embedding import EmbeddingDecisionConverter
from llm_rosetta.converters.decision.reranker import RerankerDecisionConverter
from llm_rosetta.types.ir.decision import (
    ChoiceQuestion,
    IRDecisionRequest,
    NoulQuestion,
    ScoreQuestion,
)

DECISION_REQUEST: IRDecisionRequest = {
    "model": "",
    "state": (
        "I want my money back for order #4521. Nobody responds to my emails. "
        "This is ridiculous. I'm disputing the charge."
    ),
    "questions": {
        "wants_refund": NoulQuestion(
            type="noul",
            instructions="Is the customer requesting a refund?",
            criteria={
                "true": "Asks for money back or refund",
                "false": "No refund request",
            },
        ),
        "department": ChoiceQuestion(
            type="choice",
            instructions="Which team should handle this?",
            criteria={
                "billing": "Payments, refunds, charges",
                "support": "General customer help",
                "escalation": "Urgent complaints, legal threats",
            },
        ),
        "frustration": ScoreQuestion(
            type="score",
            instructions="How frustrated is the customer?",
            criteria=[
                "Calm and factual",
                "Annoyed but civil",
                "Very frustrated and angry",
            ],
        ),
    },
}


def _make_request(model: str) -> IRDecisionRequest:
    return {**DECISION_REQUEST, "model": model}


def _send_rerank(url: str, api_key: str, queries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    results = []
    for q in queries:
        resp = httpx.post(
            url,
            json={
                "model": q["model"],
                "query": q["query"],
                "documents": q["documents"],
            },
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            timeout=30,
        )
        resp.raise_for_status()
        results.append(resp.json())
    return results


def _send_embedding(url: str, api_key: str, wire: dict[str, Any]) -> dict[str, Any]:
    resp = httpx.post(
        url,
        json={"model": wire["model"], "input": wire["input"]},
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def _validate_structure(ir: dict[str, Any]) -> None:
    """Validate answer structure (not semantic correctness — baselines are weak)."""
    assert ir["object"] == "decision"
    a = ir["answers"]

    noul_a = a["wants_refund"]
    assert noul_a["type"] == "noul"
    assert 0.01 <= noul_a["noul"] <= 0.99

    choice_a = a["department"]
    assert choice_a["type"] == "choice"
    assert choice_a["choice"] in ("billing", "support", "escalation")
    assert abs(sum(choice_a["probabilities"].values()) - 1.0) < 0.01

    score_a = a["frustration"]
    assert score_a["type"] == "score"
    assert 0.0 <= score_a["score"] <= 2.0
    assert abs(sum(score_a["probabilities"].values()) - 1.0) < 0.01


# ============================================================================
# Reranker backends
# ============================================================================


class TestJinaReranker:
    @pytest.fixture(autouse=True)
    def _check_keys(self):
        if not os.getenv("JINA_API_KEY"):
            pytest.skip("JINA_API_KEY not set")

    def test_jina_reranker(self):
        converter = RerankerDecisionConverter(temperature=0.3)
        ctx = ConversionContext()
        wire, _ = converter.request_to_provider(
            _make_request("jina-reranker-v2-base-multilingual"), context=ctx
        )
        results = _send_rerank(
            "https://api.jina.ai/v1/rerank",
            os.environ["JINA_API_KEY"],
            wire["queries"],
        )
        ir = converter.response_from_provider(
            {"results": results, "model": wire["model"]}, context=ctx
        )
        _validate_structure(ir)
        print(
            f"\n  Jina Reranker: refund={ir['answers']['wants_refund']['noul']:.2f}, "
            f"dept={ir['answers']['department']['choice']}, "
            f"frustration={ir['answers']['frustration']['score']:.2f}"
        )


class TestCohereReranker:
    @pytest.fixture(autouse=True)
    def _check_keys(self):
        if not os.getenv("COHERE_API_KEY"):
            pytest.skip("COHERE_API_KEY not set")

    def test_cohere_reranker(self):
        converter = RerankerDecisionConverter(temperature=0.3)
        ctx = ConversionContext()
        wire, _ = converter.request_to_provider(
            _make_request("rerank-v3.5"), context=ctx
        )
        results = _send_rerank(
            "https://api.cohere.com/v2/rerank",
            os.environ["COHERE_API_KEY"],
            wire["queries"],
        )
        ir = converter.response_from_provider(
            {"results": results, "model": wire["model"]}, context=ctx
        )
        _validate_structure(ir)
        print(
            f"\n  Cohere Reranker: refund={ir['answers']['wants_refund']['noul']:.2f}, "
            f"dept={ir['answers']['department']['choice']}, "
            f"frustration={ir['answers']['frustration']['score']:.2f}"
        )


class TestVoyageReranker:
    @pytest.fixture(autouse=True)
    def _check_keys(self):
        if not os.getenv("VOYAGE_API_KEY"):
            pytest.skip("VOYAGE_API_KEY not set")

    def test_voyage_reranker(self):
        converter = RerankerDecisionConverter(temperature=0.3)
        ctx = ConversionContext()
        wire, _ = converter.request_to_provider(
            _make_request("rerank-2"), context=ctx
        )
        results = _send_rerank(
            "https://api.voyageai.com/v1/rerank",
            os.environ["VOYAGE_API_KEY"],
            wire["queries"],
        )
        ir = converter.response_from_provider(
            {"results": results, "model": wire["model"]}, context=ctx
        )
        _validate_structure(ir)
        print(
            f"\n  Voyage Reranker: refund={ir['answers']['wants_refund']['noul']:.2f}, "
            f"dept={ir['answers']['department']['choice']}, "
            f"frustration={ir['answers']['frustration']['score']:.2f}"
        )


# ============================================================================
# Embedding backends
# ============================================================================


class TestOpenAIEmbedding:
    @pytest.fixture(autouse=True)
    def _check_keys(self):
        if not os.getenv("OPENAI_API_KEY"):
            pytest.skip("OPENAI_API_KEY not set")

    def test_openai_embedding(self):
        converter = EmbeddingDecisionConverter(temperature=0.05)
        ctx = ConversionContext()
        wire, _ = converter.request_to_provider(
            _make_request("text-embedding-3-small"), context=ctx
        )
        base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        resp = _send_embedding(
            f"{base_url.rstrip('/')}/embeddings",
            os.environ["OPENAI_API_KEY"],
            wire,
        )
        ir = converter.response_from_provider(resp, context=ctx)
        _validate_structure(ir)
        print(
            f"\n  OpenAI Embedding: refund={ir['answers']['wants_refund']['noul']:.2f}, "
            f"dept={ir['answers']['department']['choice']}, "
            f"frustration={ir['answers']['frustration']['score']:.2f}"
        )


class TestJinaEmbedding:
    @pytest.fixture(autouse=True)
    def _check_keys(self):
        if not os.getenv("JINA_API_KEY"):
            pytest.skip("JINA_API_KEY not set")

    def test_jina_embedding(self):
        converter = EmbeddingDecisionConverter(temperature=0.05)
        ctx = ConversionContext()
        wire, _ = converter.request_to_provider(
            _make_request("jina-embeddings-v3"), context=ctx
        )
        resp = _send_embedding(
            "https://api.jina.ai/v1/embeddings",
            os.environ["JINA_API_KEY"],
            wire,
        )
        ir = converter.response_from_provider(resp, context=ctx)
        _validate_structure(ir)
        print(
            f"\n  Jina Embedding: refund={ir['answers']['wants_refund']['noul']:.2f}, "
            f"dept={ir['answers']['department']['choice']}, "
            f"frustration={ir['answers']['frustration']['score']:.2f}"
        )
