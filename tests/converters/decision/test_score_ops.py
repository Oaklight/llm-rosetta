"""Tests for decision score operations (shared scoring logic)."""

from typing import Any, cast

import pytest

from llm_rosetta.converters.decision.score_ops import (
    build_context,
    get_option_texts,
    scores_to_answer,
    softmax,
)
from llm_rosetta.types.ir.decision import (
    ChoiceQuestion,
    NoulQuestion,
    ScoreQuestion,
)


class TestSoftmax:
    def test_uniform(self):
        result = softmax([0.0, 0.0, 0.0])
        assert all(abs(p - 1 / 3) < 1e-6 for p in result)

    def test_peaked(self):
        result = softmax([10.0, 0.0, 0.0])
        assert result[0] > 0.99

    def test_negative(self):
        result = softmax([-1.0, -2.0])
        assert abs(sum(result) - 1.0) < 1e-9
        assert result[0] > result[1]

    def test_empty(self):
        assert softmax([]) == []

    def test_single(self):
        assert softmax([5.0]) == [1.0]


class TestBuildContext:
    def test_string_state(self):
        ctx = build_context("hello", "is this urgent?")
        assert "hello" in ctx
        assert "is this urgent?" in ctx

    def test_dict_state(self):
        ctx = build_context({"key": "val"}, "question")
        assert "key" in ctx
        assert "question" in ctx


class TestGetOptionTexts:
    def test_noul_with_criteria(self):
        q = NoulQuestion(
            type="noul",
            instructions="test",
            criteria={"true": "Yes it is", "false": "No it isn't"},
        )
        opts = get_option_texts(q)
        assert opts == ["Yes it is", "No it isn't"]

    def test_noul_without_criteria(self):
        q = NoulQuestion(type="noul", instructions="test")
        opts = get_option_texts(q)
        assert opts == ["yes", "no"]

    def test_choice(self):
        q = ChoiceQuestion(
            type="choice",
            instructions="test",
            criteria={"billing": "Payments", "tech": None},
        )
        opts = get_option_texts(q)
        assert opts == ["Payments", "tech"]

    def test_score(self):
        q = ScoreQuestion(
            type="score",
            instructions="test",
            criteria=["Low", "Medium", "High"],
        )
        opts = get_option_texts(q)
        assert opts == ["Low", "Medium", "High"]


class TestScoresToAnswer:
    def test_noul_high_true(self):
        q = NoulQuestion(type="noul", instructions="test")
        answer = scores_to_answer([5.0, -1.0], q)
        assert answer["type"] == "noul"
        assert cast(Any, answer)["noul"] > 0.9

    def test_noul_high_false(self):
        q = NoulQuestion(type="noul", instructions="test")
        answer = scores_to_answer([-1.0, 5.0], q)
        assert answer["type"] == "noul"
        assert cast(Any, answer)["noul"] < 0.1

    def test_noul_clamped(self):
        q = NoulQuestion(type="noul", instructions="test")
        answer = scores_to_answer([100.0, -100.0], q)
        assert cast(Any, answer)["noul"] <= 0.99
        answer2 = scores_to_answer([-100.0, 100.0], q)
        assert cast(Any, answer2)["noul"] >= 0.01

    def test_choice(self):
        q = ChoiceQuestion(
            type="choice",
            instructions="test",
            criteria={"a": "A", "b": "B", "c": "C"},
        )
        answer = scores_to_answer([0.5, 3.0, 0.1], q)
        assert answer["type"] == "choice"
        assert answer["choice"] == "b"
        assert abs(sum(answer["probabilities"].values()) - 1.0) < 1e-6
        assert 0 <= answer["confidence"] <= 1

    def test_score(self):
        q = ScoreQuestion(
            type="score",
            instructions="test",
            criteria=["Low", "High"],
        )
        answer = scores_to_answer([0.0, 5.0], q)
        assert answer["type"] == "score"
        assert cast(Any, answer)["score"] > 0.5
        assert answer["legend"] == {"0": "Low", "1": "High"}
        assert abs(sum(answer["probabilities"].values()) - 1.0) < 1e-6

    def test_score_equal(self):
        q = ScoreQuestion(
            type="score",
            instructions="test",
            criteria=["A", "B", "C"],
        )
        answer = scores_to_answer([0.0, 0.0, 0.0], q)
        assert cast(Any, answer)["score"] == pytest.approx(1.0, abs=0.01)
