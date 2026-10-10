"""Tests for derived decision-answer values (``converters.decision.derived``)."""

import pytest

from llm_rosetta.converters.decision.derived import (
    abstained,
    choice,
    confidence,
    score,
)


class TestAbstained:
    def test_unknown_is_argmax(self):
        a = {
            "type": "choice",
            "probabilities": {"a": 0.3, "b": 0.2},
            "unknown_probability": 0.5,
        }
        assert abstained(a) is True

    def test_unknown_not_argmax(self):
        a = {
            "type": "choice",
            "probabilities": {"a": 0.8, "b": 0.1},
            "unknown_probability": 0.1,
        }
        assert abstained(a) is False

    def test_no_unknown(self):
        assert abstained({"type": "choice", "probabilities": {"a": 1.0}}) is False

    def test_assertion(self):
        assert (
            abstained(
                {"type": "assertion", "probability": 0.9, "unknown_probability": 0.95}
            )
            is True
        )
        assert (
            abstained(
                {"type": "assertion", "probability": 0.9, "unknown_probability": 0.05}
            )
            is False
        )


class TestConfidence:
    def test_stored_wins(self):
        assert confidence({"confidence": 0.42, "probabilities": {"a": 1.0}}) == 0.42

    def test_derived(self):
        assert confidence({"probabilities": {"a": 1.0, "b": 0.0}}) == pytest.approx(1.0)
        assert confidence({"probabilities": {"a": 0.5, "b": 0.5}}) == pytest.approx(0.0)

    def test_no_probabilities(self):
        assert confidence({"type": "assertion", "probability": 0.9}) == 0.0


class TestChoice:
    def test_stored_wins(self):
        assert choice({"choice": "a", "probabilities": {"b": 1.0}}) == "a"

    def test_derived_argmax(self):
        assert choice({"probabilities": {"a": 0.2, "b": 0.8}}) == "b"

    def test_no_probabilities(self):
        assert choice({"type": "assertion", "probability": 0.9}) is None


class TestScore:
    def test_stored_wins(self):
        opts = [{"label": "A"}, {"label": "B"}]
        assert score({"score": 0.1, "probabilities": {"A": 1.0, "B": 0.0}}, opts) == 0.1

    def test_derived_weighted_mean(self):
        opts = [{"label": "A"}, {"label": "B"}, {"label": "C"}]
        a = {"probabilities": {"A": 0.1, "B": 0.3, "C": 0.6}}
        assert score(a, opts) == pytest.approx(1.5)

    def test_missing_label_counts_zero(self):
        opts = [{"label": "A"}, {"label": "B"}]
        assert score({"probabilities": {"A": 0.5}}, opts) == pytest.approx(0.0)
