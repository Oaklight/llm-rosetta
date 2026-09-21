"""Tests for the embedding decision converter."""

from typing import Any

import pytest

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.decision.embedding import (
    EmbeddingDecisionConverter,
    _cosine_similarity,
)
from llm_rosetta.types.ir.decision import (
    ChoiceQuestion,
    IRDecisionRequest,
    NoulQuestion,
)


@pytest.fixture
def converter():
    return EmbeddingDecisionConverter()


IR_REQUEST: IRDecisionRequest = {
    "model": "text-embedding-3-small",
    "state": "Customer wants a refund",
    "questions": {
        "wants_refund": NoulQuestion(
            type="noul",
            instructions="Requesting refund?",
            criteria={"true": "Wants money back", "false": "No refund"},
        ),
        "department": ChoiceQuestion(
            type="choice",
            instructions="Which team?",
            criteria={"billing": "Payments", "support": "Help"},
        ),
    },
}


def _make_embedding_response(embeddings: list[list[float]]) -> dict[str, Any]:
    return {
        "model": "text-embedding-3-small",
        "data": [{"index": i, "embedding": emb} for i, emb in enumerate(embeddings)],
    }


class TestCosineSimilarity:
    def test_identical(self):
        assert _cosine_similarity([1, 0, 0], [1, 0, 0]) == pytest.approx(1.0)

    def test_orthogonal(self):
        assert _cosine_similarity([1, 0], [0, 1]) == pytest.approx(0.0)

    def test_opposite(self):
        assert _cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)

    def test_zero_vector(self):
        assert _cosine_similarity([0, 0], [1, 0]) == 0.0


class TestRequestToProvider:
    def test_produces_input_texts(self, converter: EmbeddingDecisionConverter):
        ctx = ConversionContext()
        wire, _ = converter.request_to_provider(IR_REQUEST, context=ctx)
        # 2 questions: noul has context+2opts=3, choice has context+2opts=3 → 6 total
        assert len(wire["input"]) == 6

    def test_stores_layout_in_context(self, converter: EmbeddingDecisionConverter):
        ctx = ConversionContext()
        converter.request_to_provider(IR_REQUEST, context=ctx)
        layout = ctx.options["_embedding_layout"]
        assert layout == [("wants_refund", 2), ("department", 2)]

    def test_model_passed_through(self, converter: EmbeddingDecisionConverter):
        ctx = ConversionContext()
        wire, _ = converter.request_to_provider(IR_REQUEST, context=ctx)
        assert wire["model"] == "text-embedding-3-small"


class TestResponseFromProvider:
    def test_parses_noul(self, converter: EmbeddingDecisionConverter):
        ctx = ConversionContext()
        converter.request_to_provider(IR_REQUEST, context=ctx)
        # ctx=[1,0], true_opt=[0.9,0.1] (sim~0.9), false_opt=[0.1,0.9] (sim~0.1)
        resp = _make_embedding_response(
            [
                [1.0, 0.0],  # context for q1
                [0.9, 0.1],  # "Wants money back" (similar to context)
                [0.1, 0.9],  # "No refund" (dissimilar)
                [1.0, 0.0],  # context for q2
                [0.8, 0.2],  # "Payments" (similar)
                [0.2, 0.8],  # "Help" (dissimilar)
            ]
        )
        ir = converter.response_from_provider(resp, context=ctx)
        assert ir["answers"]["wants_refund"]["type"] == "noul"
        assert ir["answers"]["wants_refund"]["noul"] > 0.5

    def test_parses_choice(self, converter: EmbeddingDecisionConverter):
        ctx = ConversionContext()
        converter.request_to_provider(IR_REQUEST, context=ctx)
        resp = _make_embedding_response(
            [
                [1.0, 0.0],
                [0.9, 0.1],
                [0.1, 0.9],
                [1.0, 0.0],
                [0.9, 0.1],  # billing - similar
                [0.1, 0.9],  # support - dissimilar
            ]
        )
        ir = converter.response_from_provider(resp, context=ctx)
        a: Any = ir["answers"]["department"]
        assert a["type"] == "choice"
        assert a["choice"] == "billing"

    def test_object_field(self, converter: EmbeddingDecisionConverter):
        ctx = ConversionContext()
        converter.request_to_provider(IR_REQUEST, context=ctx)
        resp = _make_embedding_response([[1, 0]] * 6)
        ir = converter.response_from_provider(resp, context=ctx)
        assert ir["object"] == "decision"

    def test_missing_context_raises(self, converter: EmbeddingDecisionConverter):
        ctx = ConversionContext()
        with pytest.raises(ValueError, match="_decision_questions"):
            converter.response_from_provider({}, context=ctx)

    def test_not_implemented_methods(self, converter: EmbeddingDecisionConverter):
        with pytest.raises(NotImplementedError):
            converter.request_from_provider({})
        with pytest.raises(NotImplementedError):
            converter.response_to_provider(
                {"object": "decision", "model": "m", "answers": {}}
            )

    def test_converter_tag(self, converter: EmbeddingDecisionConverter):
        assert converter._CONVERTER_TAG == "embedding_decision"


class TestDimensionMismatch:
    def test_mismatched_dimensions_raises(self):
        with pytest.raises(ValueError, match="dimension mismatch"):
            _cosine_similarity([1, 0], [1, 0, 0])
