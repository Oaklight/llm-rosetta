"""Tests for decision schema generation and answer parsing."""

from typing import Any, cast

import pytest

from llm_rosetta.converters.decision.schema_ops import (
    build_decision_schema,
    build_system_prompt,
    compute_confidence,
    extract_json,
    parse_decision_answers,
    serialize_state,
)

# ============================================================================
# Schema generation — probabilities mode
# ============================================================================


class TestBuildDecisionSchema:
    def test_assertion_question(self):
        questions = {"q": {"type": "assertion", "instructions": "Is this urgent?"}}
        schema = build_decision_schema(cast(Any, questions))
        prop = schema["properties"]["answers"]["properties"]["q"]
        assert prop["type"] == "number"
        assert prop["minimum"] == 0
        assert prop["maximum"] == 1

    def test_choice_question(self):
        questions = {
            "q": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": [
                    {"label": "billing", "description": "Payments"},
                    {"label": "tech", "description": "Bugs"},
                ],
            }
        }
        schema = build_decision_schema(cast(Any, questions))
        prop = schema["properties"]["answers"]["properties"]["q"]
        assert prop["type"] == "object"
        assert set(prop["properties"]) == {"billing", "tech"}
        assert prop["additionalProperties"] is False

    def test_score_question(self):
        questions = {
            "q": {
                "type": "score",
                "instructions": "How frustrated?",
                "criteria": [
                    {"label": "Calm"},
                    {"label": "Frustrated"},
                    {"label": "Very angry"},
                ],
            }
        }
        schema = build_decision_schema(cast(Any, questions))
        prop = schema["properties"]["answers"]["properties"]["q"]
        assert prop["type"] == "object"
        assert set(prop["properties"]) == {"0", "1", "2"}
        assert prop["additionalProperties"] is False

    def test_multi_question(self):
        questions = {
            "a": {"type": "assertion", "instructions": "q1"},
            "b": {
                "type": "choice",
                "instructions": "q2",
                "criteria": [{"label": "x"}, {"label": "y"}],
            },
        }
        schema = build_decision_schema(cast(Any, questions))
        answers_props = schema["properties"]["answers"]["properties"]
        assert set(answers_props) == {"a", "b"}
        assert set(schema["properties"]["answers"]["required"]) == {"a", "b"}

    def test_top_level_structure(self):
        questions = {"q": {"type": "assertion", "instructions": "test"}}
        schema = build_decision_schema(cast(Any, questions))
        assert schema["type"] == "object"
        assert schema["required"] == ["answers"]
        assert schema["additionalProperties"] is False


# ============================================================================
# Schema generation — discrete mode
# ============================================================================


class TestDiscreteSchema:
    def test_assertion_boolean(self):
        questions = {"q": {"type": "assertion", "instructions": "test"}}
        schema = build_decision_schema(cast(Any, questions), answer_mode="discrete")
        prop = schema["properties"]["answers"]["properties"]["q"]
        assert prop["type"] == "boolean"

    def test_choice_enum(self):
        questions = {
            "q": {
                "type": "choice",
                "instructions": "test",
                "criteria": [{"label": "a"}, {"label": "b"}],
            }
        }
        schema = build_decision_schema(cast(Any, questions), answer_mode="discrete")
        prop = schema["properties"]["answers"]["properties"]["q"]
        assert set(prop.get("enum", [])) == {"a", "b"}

    def test_score_integer(self):
        questions = {
            "q": {
                "type": "score",
                "instructions": "test",
                "criteria": [{"label": "Low"}, {"label": "High"}],
            }
        }
        schema = build_decision_schema(cast(Any, questions), answer_mode="discrete")
        prop = schema["properties"]["answers"]["properties"]["q"]
        assert prop["type"] == "integer"


# ============================================================================
# System prompt
# ============================================================================


class TestBuildSystemPrompt:
    def test_contains_question_ids(self):
        questions = {
            "is_urgent": {"type": "assertion", "instructions": "Is this urgent?"},
            "dept": {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": [
                    {"label": "billing", "description": "pay"},
                    {"label": "tech", "description": "bugs"},
                ],
            },
        }
        prompt = build_system_prompt(cast(Any, questions))
        assert "is_urgent" in prompt
        assert "dept" in prompt
        assert "(assertion)" in prompt
        assert "(choice)" in prompt

    def test_assertion_with_criteria(self):
        questions = {
            "q": {
                "type": "assertion",
                "instructions": "Is it true?",
                "criteria": [
                    {"label": False, "description": "No"},
                    {"label": True, "description": "Yes"},
                ],
            }
        }
        prompt = build_system_prompt(cast(Any, questions))
        assert "true=Yes" in prompt
        assert "false=No" in prompt

    def test_score_levels(self):
        questions = {
            "q": {
                "type": "score",
                "instructions": "Rate it",
                "criteria": [
                    {"label": "Low"},
                    {"label": "Medium"},
                    {"label": "High"},
                ],
            }
        }
        prompt = build_system_prompt(cast(Any, questions))
        assert "0=Low" in prompt
        assert "1=Medium" in prompt
        assert "2=High" in prompt

    def test_probability_mode_prompt(self):
        questions = {"q": {"type": "assertion", "instructions": "test"}}
        prompt = build_system_prompt(cast(Any, questions), answer_mode="probabilities")
        assert "sum to 1" in prompt

    def test_discrete_mode_prompt(self):
        questions = {"q": {"type": "assertion", "instructions": "test"}}
        prompt = build_system_prompt(cast(Any, questions), answer_mode="discrete")
        assert "exactly one allowed value" in prompt

    def test_prompted_fallback_embeds_schema(self):
        questions = {"q": {"type": "assertion", "instructions": "test"}}
        schema = build_decision_schema(cast(Any, questions))
        prompt = build_system_prompt(cast(Any, questions), schema=schema)
        assert "Return one JSON object that matches this schema exactly" in prompt
        assert '"answers"' in prompt

    def test_untrusted_data_warning(self):
        questions = {"q": {"type": "assertion", "instructions": "test"}}
        assert "untrusted data" in build_system_prompt(cast(Any, questions))


# ============================================================================
# State serialization + injection protection
# ============================================================================


class TestSerializeState:
    def test_string(self):
        assert "<document>" in serialize_state("hello")

    def test_dict(self):
        result = serialize_state({"key": "value"})
        assert '"key"' in result
        assert "<document>" in result

    def test_escapes_angle_brackets(self):
        result = serialize_state("<script>alert('xss')</script>")
        assert "<script>" not in result
        assert "\\u003c" in result

    def test_structured_state_escapes(self):
        result = serialize_state({"html": "<b>bold</b>"})
        assert "<b>" not in result
        assert "\\u003c" in result


# ============================================================================
# JSON extraction
# ============================================================================


class TestExtractJson:
    def test_plain_json(self):
        assert extract_json('{"a": 1}') == '{"a": 1}'

    def test_fenced_json(self):
        assert extract_json('```json\n{"a": 1}\n```') == '{"a": 1}'

    def test_fenced_no_lang(self):
        assert extract_json('```\n{"a": 1}\n```') == '{"a": 1}'

    def test_whitespace(self):
        assert extract_json('  {"a": 1}  ') == '{"a": 1}'


# ============================================================================
# Answer parsing — probabilities
# ============================================================================


class TestParseDecisionAnswers:
    def test_assertion_answer(self):
        questions = {"q": {"type": "assertion", "instructions": "test"}}
        answers = parse_decision_answers({"answers": {"q": 0.85}}, cast(Any, questions))
        assert answers["q"]["type"] == "assertion"
        assert cast(Any, answers["q"])["probability"] == 0.85

    def test_choice_answer(self):
        questions = {
            "q": {
                "type": "choice",
                "instructions": "test",
                "criteria": [{"label": "a"}, {"label": "b"}],
            }
        }
        raw = {"answers": {"q": {"a": 0.7, "b": 0.3}}}
        a = cast(Any, parse_decision_answers(raw, cast(Any, questions))["q"])
        assert a["type"] == "choice"
        assert a["choice"] == "a"
        assert a["probabilities"] == {"a": 0.7, "b": 0.3}
        assert 0 <= a["confidence"] <= 1

    def test_score_answer_rekeyed_by_label(self):
        questions = {
            "q": {
                "type": "score",
                "instructions": "test",
                "criteria": [{"label": "Low"}, {"label": "Mid"}, {"label": "High"}],
            }
        }
        raw = {"answers": {"q": {"0": 0.1, "1": 0.3, "2": 0.6}}}
        a = cast(Any, parse_decision_answers(raw, cast(Any, questions))["q"])
        assert a["type"] == "score"
        assert a["score"] == pytest.approx(0 * 0.1 + 1 * 0.3 + 2 * 0.6)
        assert a["probabilities"] == {"Low": 0.1, "Mid": 0.3, "High": 0.6}
        assert "legend" not in a

    def test_missing_answer_skipped(self):
        questions = {"q": {"type": "assertion", "instructions": "test"}}
        answers = parse_decision_answers({"answers": {}}, cast(Any, questions))
        assert "q" not in answers

    def test_multi_question(self):
        questions = {
            "n": {"type": "assertion", "instructions": "assertion q"},
            "c": {
                "type": "choice",
                "instructions": "choice q",
                "criteria": [{"label": "x"}, {"label": "y"}],
            },
        }
        raw = {"answers": {"n": 0.5, "c": {"x": 0.4, "y": 0.6}}}
        answers = parse_decision_answers(raw, cast(Any, questions))
        assert answers["n"]["type"] == "assertion"
        assert answers["c"]["type"] == "choice"


# ============================================================================
# Answer parsing — discrete
# ============================================================================


class TestParseDiscreteAnswers:
    def test_assertion_bool_true(self):
        questions = {"q": {"type": "assertion", "instructions": "test"}}
        answers = parse_decision_answers(
            {"answers": {"q": True}}, cast(Any, questions), answer_mode="discrete"
        )
        assert cast(Any, answers["q"])["probability"] == 1.0

    def test_assertion_bool_false(self):
        questions = {"q": {"type": "assertion", "instructions": "test"}}
        answers = parse_decision_answers(
            {"answers": {"q": False}}, cast(Any, questions), answer_mode="discrete"
        )
        assert cast(Any, answers["q"])["probability"] == 0.0

    def test_choice_discrete(self):
        questions = {
            "q": {
                "type": "choice",
                "instructions": "test",
                "criteria": [{"label": "a"}, {"label": "b"}],
            }
        }
        raw = {"answers": {"q": "a"}}
        a = cast(
            Any,
            parse_decision_answers(raw, cast(Any, questions), answer_mode="discrete")[
                "q"
            ],
        )
        assert a["choice"] == "a"
        assert a["confidence"] == 1.0
        assert a["probabilities"] == {"a": 1.0, "b": 0.0}

    def test_score_discrete(self):
        questions = {
            "q": {
                "type": "score",
                "instructions": "test",
                "criteria": [{"label": "Low"}, {"label": "High"}],
            }
        }
        raw = {"answers": {"q": 1}}
        a = cast(
            Any,
            parse_decision_answers(raw, cast(Any, questions), answer_mode="discrete")[
                "q"
            ],
        )
        assert a["score"] == 1.0
        assert a["confidence"] == 1.0
        assert a["probabilities"] == {"Low": 0.0, "High": 1.0}


# ============================================================================
# Confidence computation
# ============================================================================


class TestComputeConfidence:
    def test_peaked_distribution(self):
        assert compute_confidence({"a": 1.0, "b": 0.0}) == 1.0

    def test_uniform_distribution(self):
        assert compute_confidence({"a": 0.5, "b": 0.5}) == pytest.approx(0.0)

    def test_three_way_uniform(self):
        assert compute_confidence(
            {"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}
        ) == pytest.approx(0.0, abs=1e-10)

    def test_single_option(self):
        assert compute_confidence({"a": 1.0}) == 1.0

    def test_slightly_peaked(self):
        assert 0.0 < compute_confidence({"a": 0.8, "b": 0.2}) < 1.0

    def test_empty(self):
        assert compute_confidence({}) == 1.0


# ============================================================================
# Probability normalization
# ============================================================================


class TestProbabilityNormalization:
    def test_choice_normalizes(self):
        questions = {
            "q": {
                "type": "choice",
                "instructions": "test",
                "criteria": [{"label": "a"}, {"label": "b"}],
            }
        }
        raw = {"answers": {"q": {"a": 0.6, "b": 0.7}}}
        probs = cast(Any, parse_decision_answers(raw, cast(Any, questions))["q"])[
            "probabilities"
        ]
        assert abs(sum(probs.values()) - 1.0) < 1e-9

    def test_score_normalizes_expectation(self):
        questions = {
            "q": {
                "type": "score",
                "instructions": "test",
                "criteria": [{"label": "Low"}, {"label": "High"}],
            }
        }
        a = cast(
            Any,
            parse_decision_answers(
                {"answers": {"q": {"0": 0.4, "1": 0.6}}}, cast(Any, questions)
            )["q"],
        )
        assert a["score"] == pytest.approx(0.6)
        a2 = cast(
            Any,
            parse_decision_answers(
                {"answers": {"q": {"0": 0.8, "1": 1.2}}}, cast(Any, questions)
            )["q"],
        )
        assert a2["score"] == pytest.approx(0.6)
