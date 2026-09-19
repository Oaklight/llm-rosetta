"""Tests for decision schema generation and answer parsing."""

from typing import Any

import pytest

from llm_rosetta.converters.decision.schema_ops import (
    build_decision_schema,
    build_system_prompt,
    compute_confidence,
    parse_decision_answers,
    serialize_state,
)
from llm_rosetta.types.ir.decision import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
)


# ============================================================================
# Schema generation
# ============================================================================


class TestBuildDecisionSchema:
    def test_noul_question(self):
        questions = {
            "q": NoulQuestion(type="noul", instructions="Is this urgent?"),
        }
        schema = build_decision_schema(questions)
        prop = schema["properties"]["answers"]["properties"]["q"]
        assert prop["type"] == "number"
        assert "P(true)" in prop["description"]

    def test_choice_question(self):
        questions = {
            "q": ChoiceQuestion(
                type="choice",
                instructions="Which team?",
                criteria={"billing": "Payments", "tech": "Bugs"},
            ),
        }
        schema = build_decision_schema(questions)
        prop = schema["properties"]["answers"]["properties"]["q"]
        assert prop["type"] == "object"
        assert "billing" in prop["properties"]
        assert "tech" in prop["properties"]
        assert prop["additionalProperties"] is False
        assert set(prop["required"]) == {"billing", "tech"}

    def test_score_question(self):
        questions = {
            "q": ScoreQuestion(
                type="score",
                instructions="How frustrated?",
                criteria=["Calm", "Frustrated", "Very angry"],
            ),
        }
        schema = build_decision_schema(questions)
        prop = schema["properties"]["answers"]["properties"]["q"]
        assert prop["type"] == "object"
        assert "0" in prop["properties"]
        assert "1" in prop["properties"]
        assert "2" in prop["properties"]
        assert prop["properties"]["0"]["description"] == "Calm"
        assert prop["additionalProperties"] is False

    def test_multi_question(self):
        questions = {
            "a": NoulQuestion(type="noul", instructions="q1"),
            "b": ChoiceQuestion(
                type="choice", instructions="q2", criteria={"x": None, "y": None}
            ),
        }
        schema = build_decision_schema(questions)
        answers_props = schema["properties"]["answers"]["properties"]
        assert "a" in answers_props
        assert "b" in answers_props
        assert schema["properties"]["answers"]["required"] == ["a", "b"]

    def test_top_level_structure(self):
        questions = {"q": NoulQuestion(type="noul", instructions="test")}
        schema = build_decision_schema(questions)
        assert schema["type"] == "object"
        assert schema["required"] == ["answers"]
        assert schema["additionalProperties"] is False


# ============================================================================
# System prompt
# ============================================================================


class TestBuildSystemPrompt:
    def test_contains_question_ids(self):
        questions = {
            "is_urgent": NoulQuestion(type="noul", instructions="Is this urgent?"),
            "dept": ChoiceQuestion(
                type="choice",
                instructions="Which team?",
                criteria={"billing": "pay", "tech": "bugs"},
            ),
        }
        prompt = build_system_prompt(questions)
        assert "is_urgent" in prompt
        assert "dept" in prompt
        assert "(noul)" in prompt
        assert "(choice)" in prompt

    def test_noul_with_criteria(self):
        questions = {
            "q": NoulQuestion(
                type="noul",
                instructions="Is it true?",
                criteria={"true": "Yes", "false": "No"},
            ),
        }
        prompt = build_system_prompt(questions)
        assert "true=Yes" in prompt
        assert "false=No" in prompt

    def test_score_levels(self):
        questions = {
            "q": ScoreQuestion(
                type="score",
                instructions="Rate it",
                criteria=["Low", "Medium", "High"],
            ),
        }
        prompt = build_system_prompt(questions)
        assert "0=Low" in prompt
        assert "1=Medium" in prompt
        assert "2=High" in prompt


# ============================================================================
# State serialization
# ============================================================================


class TestSerializeState:
    def test_string(self):
        assert serialize_state("hello") == "hello"

    def test_dict(self):
        result = serialize_state({"key": "value"})
        assert '"key"' in result
        assert '"value"' in result

    def test_list(self):
        result = serialize_state([1, 2, 3])
        assert result == "[1, 2, 3]"


# ============================================================================
# Answer parsing
# ============================================================================


class TestParseDecisionAnswers:
    def test_noul_answer(self):
        questions = {"q": NoulQuestion(type="noul", instructions="test")}
        raw = {"answers": {"q": 0.85}}
        answers = parse_decision_answers(raw, questions)
        assert answers["q"]["type"] == "noul"
        assert answers["q"]["noul"] == 0.85

    def test_choice_answer(self):
        questions = {
            "q": ChoiceQuestion(
                type="choice",
                instructions="test",
                criteria={"a": "A", "b": "B"},
            )
        }
        raw = {"answers": {"q": {"a": 0.7, "b": 0.3}}}
        answers = parse_decision_answers(raw, questions)
        a = answers["q"]
        assert a["type"] == "choice"
        assert a["choice"] == "a"
        assert a["probabilities"] == {"a": 0.7, "b": 0.3}
        assert 0 <= a["confidence"] <= 1

    def test_score_answer(self):
        questions = {
            "q": ScoreQuestion(
                type="score",
                instructions="test",
                criteria=["Low", "Mid", "High"],
            )
        }
        raw = {"answers": {"q": {"0": 0.1, "1": 0.3, "2": 0.6}}}
        answers = parse_decision_answers(raw, questions)
        a = answers["q"]
        assert a["type"] == "score"
        assert a["score"] == pytest.approx(0 * 0.1 + 1 * 0.3 + 2 * 0.6)
        assert a["legend"] == {"0": "Low", "1": "Mid", "2": "High"}

    def test_missing_answer_skipped(self):
        questions = {"q": NoulQuestion(type="noul", instructions="test")}
        raw = {"answers": {}}
        answers = parse_decision_answers(raw, questions)
        assert "q" not in answers

    def test_multi_question(self):
        questions = {
            "n": NoulQuestion(type="noul", instructions="noul q"),
            "c": ChoiceQuestion(
                type="choice", instructions="choice q", criteria={"x": None, "y": None}
            ),
        }
        raw = {"answers": {"n": 0.5, "c": {"x": 0.4, "y": 0.6}}}
        answers = parse_decision_answers(raw, questions)
        assert answers["n"]["type"] == "noul"
        assert answers["c"]["type"] == "choice"


# ============================================================================
# Confidence computation
# ============================================================================


class TestComputeConfidence:
    def test_peaked_distribution(self):
        assert compute_confidence({"a": 1.0, "b": 0.0}) == 1.0

    def test_uniform_distribution(self):
        assert compute_confidence({"a": 0.5, "b": 0.5}) == pytest.approx(0.0)

    def test_three_way_uniform(self):
        c = compute_confidence({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3})
        assert c == pytest.approx(0.0, abs=1e-10)

    def test_single_option(self):
        assert compute_confidence({"a": 1.0}) == 1.0

    def test_slightly_peaked(self):
        c = compute_confidence({"a": 0.8, "b": 0.2})
        assert 0.0 < c < 1.0

    def test_empty(self):
        assert compute_confidence({}) == 1.0


class TestProbabilityNormalization:
    def test_choice_normalizes(self):
        questions = {
            "q": ChoiceQuestion(
                type="choice", instructions="test", criteria={"a": "A", "b": "B"}
            )
        }
        raw = {"answers": {"q": {"a": 0.6, "b": 0.7}}}
        answers = parse_decision_answers(raw, questions)
        a: Any = answers["q"]
        probs = a["probabilities"]
        assert abs(sum(probs.values()) - 1.0) < 1e-9

    def test_score_normalizes_expectation(self):
        questions = {
            "q": ScoreQuestion(
                type="score", instructions="test", criteria=["Low", "High"]
            )
        }
        raw = {"answers": {"q": {"0": 0.4, "1": 0.6}}}
        answers = parse_decision_answers(raw, questions)
        a1: Any = answers["q"]
        assert a1["score"] == pytest.approx(0.6)

        raw_unnorm = {"answers": {"q": {"0": 0.8, "1": 1.2}}}
        answers2 = parse_decision_answers(raw_unnorm, questions)
        a2: Any = answers2["q"]
        assert a2["score"] == pytest.approx(0.6)
