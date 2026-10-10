"""Tests for decision score operations (shared scoring logic)."""

from typing import Any, cast

import pytest

from llm_rosetta.converters.decision.score_ops import (
    build_context,
    get_option_texts,
    scores_to_answer,
    softmax,
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
    def test_assertion_with_criteria(self):
        q = {
            "type": "assertion",
            "instructions": "test",
            "criteria": [
                {"label": False, "description": "No it isn't"},
                {"label": True, "description": "Yes it is"},
            ],
        }
        assert get_option_texts(cast(Any, q)) == ["No it isn't", "Yes it is"]

    def test_assertion_without_criteria(self):
        q = {"type": "assertion", "instructions": "test"}
        assert get_option_texts(cast(Any, q)) == ["no", "yes"]

    def test_choice(self):
        q = {
            "type": "choice",
            "instructions": "test",
            "criteria": [
                {"label": "billing", "description": "Payments"},
                {"label": "tech"},
            ],
        }
        assert get_option_texts(cast(Any, q)) == ["Payments", "tech"]

    def test_score(self):
        q = {
            "type": "score",
            "instructions": "test",
            "criteria": [{"label": "Low"}, {"label": "Medium"}, {"label": "High"}],
        }
        assert get_option_texts(cast(Any, q)) == ["Low", "Medium", "High"]


class TestScoresToAnswer:
    def test_assertion_high_true(self):
        # options are ordered [false, true]; a high score on index 1 → P(true) high
        q = {"type": "assertion", "instructions": "test"}
        answer = scores_to_answer([-1.0, 5.0], cast(Any, q))
        assert answer["type"] == "assertion"
        assert cast(Any, answer)["probability"] > 0.9

    def test_assertion_high_false(self):
        q = {"type": "assertion", "instructions": "test"}
        answer = scores_to_answer([5.0, -1.0], cast(Any, q))
        assert cast(Any, answer)["probability"] < 0.1

    def test_assertion_clamped(self):
        q = {"type": "assertion", "instructions": "test"}
        assert (
            cast(Any, scores_to_answer([-100.0, 100.0], cast(Any, q)))["probability"]
            <= 0.99
        )
        assert (
            cast(Any, scores_to_answer([100.0, -100.0], cast(Any, q)))["probability"]
            >= 0.01
        )

    def test_choice(self):
        q = {
            "type": "choice",
            "instructions": "test",
            "criteria": [{"label": "a"}, {"label": "b"}, {"label": "c"}],
        }
        answer = scores_to_answer([0.5, 3.0, 0.1], cast(Any, q))
        assert answer["type"] == "choice"
        assert answer["choice"] == "b"
        assert set(answer["probabilities"]) == {"a", "b", "c"}
        assert abs(sum(answer["probabilities"].values()) - 1.0) < 1e-6

    def test_score_keyed_by_label(self):
        q = {
            "type": "score",
            "instructions": "test",
            "criteria": [{"label": "Low"}, {"label": "High"}],
        }
        answer = scores_to_answer([0.0, 5.0], cast(Any, q))
        assert answer["type"] == "score"
        assert cast(Any, answer)["score"] > 0.5
        assert "legend" not in answer
        assert set(answer["probabilities"]) == {"Low", "High"}

    def test_score_equal(self):
        q = {
            "type": "score",
            "instructions": "test",
            "criteria": [{"label": "A"}, {"label": "B"}, {"label": "C"}],
        }
        answer = scores_to_answer([0.0, 0.0, 0.0], cast(Any, q))
        assert cast(Any, answer)["score"] == pytest.approx(1.0, abs=0.01)


class TestTemperature:
    def test_low_temperature_sharpens(self):
        scores = [0.82, 0.85, 0.83]
        assert max(softmax(scores, temperature=0.1)) > max(softmax(scores))

    def test_high_temperature_flattens(self):
        scores = [1.0, 2.0, 3.0]
        assert max(softmax(scores, temperature=5.0)) < max(softmax(scores))

    def test_temperature_one_is_default(self):
        scores = [1.0, 2.0, 3.0]
        assert softmax(scores) == softmax(scores, temperature=1.0)
