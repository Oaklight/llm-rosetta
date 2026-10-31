"""Shim-declared request defaults fill the IR; a client value always wins."""

from __future__ import annotations

from llm_rosetta.converters.base.defaults import apply_ir_defaults
from llm_rosetta.converters.decision.pipeline import DecisionConversionPipeline
from llm_rosetta.converters.embedding.pipeline import EmbeddingConversionPipeline
from llm_rosetta.converters.rerank.pipeline import RerankConversionPipeline


class TestApplyIrDefaults:
    def test_fills_an_absent_field(self):
        assert apply_ir_defaults({"model": "m"}, {"top_n": 3}) == {
            "model": "m",
            "top_n": 3,
        }

    def test_fills_a_none_field(self):
        """The source converter may write the key with no value."""
        assert apply_ir_defaults({"top_n": None}, {"top_n": 3})["top_n"] == 3

    def test_a_client_value_wins(self):
        assert apply_ir_defaults({"top_n": 1}, {"top_n": 3})["top_n"] == 1

    def test_no_defaults_returns_the_same_mapping(self):
        ir = {"model": "m"}
        assert apply_ir_defaults(ir, None) is ir

    def test_the_input_is_not_mutated(self):
        ir = {"model": "m"}
        apply_ir_defaults(ir, {"top_n": 3})
        assert ir == {"model": "m"}


class TestEmbeddingDefaults:
    def test_a_default_reaches_the_target_dialect(self):
        pipeline = EmbeddingConversionPipeline("openai", "cohere")

        body = pipeline.convert_request(
            {"model": "m", "input": "hello"},
            defaults={"task_type": "retrieval_document"},
        )

        assert body["input_type"] == "search_document"

    def test_a_client_value_wins(self):
        pipeline = EmbeddingConversionPipeline("cohere", "cohere")

        body = pipeline.convert_request(
            {"model": "m", "texts": ["hello"], "input_type": "search_query"},
            defaults={"task_type": "retrieval_document"},
        )

        assert body["input_type"] == "search_query"


class TestRerankDefaults:
    def test_defaults_reach_the_target_dialect(self):
        pipeline = RerankConversionPipeline("jina", "cohere")

        body = pipeline.convert_request(
            {"model": "m", "query": "q", "documents": ["a", "b"]},
            defaults={"top_n": 1, "max_tokens_per_doc": 5},
        )

        assert body["top_n"] == 1
        assert body["max_tokens_per_doc"] == 5

    def test_a_client_value_wins(self):
        pipeline = RerankConversionPipeline("jina", "cohere")

        body = pipeline.convert_request(
            {"model": "m", "query": "q", "documents": ["a", "b"], "top_n": 2},
            defaults={"top_n": 1},
        )

        assert body["top_n"] == 2


class TestDecisionDefaults:
    def test_the_pipeline_accepts_defaults(self):
        """Decision declares no parameters, but the shared flow still passes {}."""
        pipeline = DecisionConversionPipeline("systemone", "systemone")
        request = {
            "model": "jev-latest",
            "state": "Help!",
            "questions": {
                "is_urgent": {
                    "type": "noul",
                    "instructions": "Urgent?",
                    "criteria": {"true": "Yes", "false": "No"},
                }
            },
        }

        body = pipeline.convert_request(request, defaults={})

        assert body["model"] == "jev-latest"
