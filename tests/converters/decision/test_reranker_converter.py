"""Tests for the reranker decision converter."""

from typing import Any

import pytest

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.decision.reranker import RerankerDecisionConverter
from llm_rosetta.types.ir.decision import (
    ChoiceQuestion,
    IRDecisionRequest,
    AssertionQuestion,
    ScoreQuestion,
)


@pytest.fixture
def converter():
    return RerankerDecisionConverter()


IR_REQUEST: IRDecisionRequest = {
    "model": "cross-encoder/ettin-reranker-150m-v1",
    "state": "Customer wants a refund for broken item",
    "questions": {
        "wants_refund": AssertionQuestion(
            type="assertion",
            instructions="Is the customer requesting a refund?",
            criteria=[
                {"label": False, "description": "No refund request"},
                {"label": True, "description": "Wants money back"},
            ],
        ),
        "department": ChoiceQuestion(
            type="choice",
            instructions="Which team?",
            criteria=[
                {"label": "billing", "description": "Payments"},
                {"label": "support", "description": "Help"},
            ],
        ),
        "frustration": ScoreQuestion(
            type="score",
            instructions="How frustrated?",
            criteria=[
                {"label": "Calm"},
                {"label": "Annoyed"},
                {"label": "Angry"},
            ],
        ),
    },
}

MOCK_RERANK_RESPONSE: dict[str, Any] = {
    "model": "cross-encoder/ettin-reranker-150m-v1",
    "results": [
        {
            "results": [
                {"index": 0, "relevance_score": 0.1},
                {"index": 1, "relevance_score": 0.9},
            ]
        },
        {
            "results": [
                {"index": 0, "relevance_score": 0.8},
                {"index": 1, "relevance_score": 0.3},
            ]
        },
        {
            "results": [
                {"index": 0, "relevance_score": 0.1},
                {"index": 1, "relevance_score": 0.3},
                {"index": 2, "relevance_score": 0.9},
            ]
        },
    ],
}


class TestRequestToProvider:
    def test_produces_queries(self, converter: RerankerDecisionConverter):
        ctx = ConversionContext()
        wire, _ = converter.request_to_provider(IR_REQUEST, context=ctx)
        queries = wire["queries"]
        assert len(queries) == 3

    def test_query_contains_state_and_instructions(
        self, converter: RerankerDecisionConverter
    ):
        ctx = ConversionContext()
        wire, _ = converter.request_to_provider(IR_REQUEST, context=ctx)
        q = wire["queries"][0]
        assert "refund" in q["query"].lower()
        assert "requesting" in q["query"].lower()

    def test_query_documents_are_option_texts(
        self, converter: RerankerDecisionConverter
    ):
        ctx = ConversionContext()
        wire, _ = converter.request_to_provider(IR_REQUEST, context=ctx)
        assertion_q = wire["queries"][0]
        assert assertion_q["documents"] == ["No refund request", "Wants money back"]

    def test_stores_questions_in_context(self, converter: RerankerDecisionConverter):
        ctx = ConversionContext()
        converter.request_to_provider(IR_REQUEST, context=ctx)
        assert "_decision_questions" in ctx.options


class TestResponseFromProvider:
    def test_parses_assertion(self, converter: RerankerDecisionConverter):
        ctx = ConversionContext()
        converter.request_to_provider(IR_REQUEST, context=ctx)
        ir = converter.response_from_provider(MOCK_RERANK_RESPONSE, context=ctx)
        a = ir["answers"]["wants_refund"]
        assert a["type"] == "assertion"
        assert a["probability"] > 0.5

    def test_parses_choice(self, converter: RerankerDecisionConverter):
        ctx = ConversionContext()
        converter.request_to_provider(IR_REQUEST, context=ctx)
        ir = converter.response_from_provider(MOCK_RERANK_RESPONSE, context=ctx)
        a: Any = ir["answers"]["department"]
        assert a["type"] == "choice"
        assert a["choice"] == "billing"
        assert abs(sum(a["probabilities"].values()) - 1.0) < 1e-6

    def test_parses_score(self, converter: RerankerDecisionConverter):
        ctx = ConversionContext()
        converter.request_to_provider(IR_REQUEST, context=ctx)
        ir = converter.response_from_provider(MOCK_RERANK_RESPONSE, context=ctx)
        a: Any = ir["answers"]["frustration"]
        assert a["type"] == "score"
        assert a["score"] > 1.0
        assert set(a["probabilities"]) == {"Calm", "Annoyed", "Angry"}

    def test_object_field(self, converter: RerankerDecisionConverter):
        ctx = ConversionContext()
        converter.request_to_provider(IR_REQUEST, context=ctx)
        ir = converter.response_from_provider(MOCK_RERANK_RESPONSE, context=ctx)
        assert ir["object"] == "decision"

    def test_missing_context_raises(self, converter: RerankerDecisionConverter):
        ctx = ConversionContext()
        with pytest.raises(ValueError, match="_decision_questions"):
            converter.response_from_provider(MOCK_RERANK_RESPONSE, context=ctx)

    def test_not_implemented_methods(self, converter: RerankerDecisionConverter):
        with pytest.raises(NotImplementedError):
            converter.request_from_provider({})
        with pytest.raises(NotImplementedError):
            converter.response_to_provider(
                {"object": "decision", "model": "m", "answers": {}}
            )

    def test_converter_tag(self, converter: RerankerDecisionConverter):
        assert converter._CONVERTER_TAG == "reranker_decision"
