"""Tests for the TypeSafe decision converter.

Covers bidirectional conversion between TypeSafe wire format and IR,
verifying near-passthrough conversion for all three question types.
"""

import pytest

from llm_rosetta.converters.decision.typesafe import TypeSafeDecisionConverter
from llm_rosetta.types.ir.decision import (
    NoulAnswer,
    NoulQuestion,
    ChoiceAnswer,
    ChoiceQuestion,
    IRDecisionRequest,
    IRDecisionResponse,
    ScoreAnswer,
    ScoreQuestion,
)


@pytest.fixture
def converter():
    return TypeSafeDecisionConverter()


# ============================================================================
# Wire format fixtures (TypeSafe native)
# ============================================================================

TYPESAFE_REQUEST = {
    "model": "jev-latest",
    "state": "Help! My payouts have been failing for 3 days.",
    "questions": {
        "is_urgent": {
            "type": "noul",
            "instructions": "Does this convey urgency?",
        },
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {
                "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
            },
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": ["Calm", "Frustrated", "Very angry"],
        },
    },
}

TYPESAFE_RESPONSE = {
    "model": "jev-1.13.0",
    "answers": {
        "is_urgent": {"type": "noul", "noul": 0.92},
        "department": {
            "type": "choice",
            "choice": "technical",
            "probabilities": {"billing": 0.08, "technical": 0.85, "sales": 0.07},
            "confidence": 0.82,
        },
        "frustration": {
            "type": "score",
            "score": 1.6,
            "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
            "probabilities": {"0": 0.05, "1": 0.3, "2": 0.65},
            "confidence": 0.78,
        },
    },
    "usage": {"input_tokens": 588, "output_tokens": 212},
}

# ============================================================================
# IR format fixtures
# ============================================================================

IR_REQUEST: IRDecisionRequest = {
    "model": "jev-latest",
    "state": "Help! My payouts have been failing for 3 days.",
    "questions": {
        "is_urgent": NoulQuestion(
            type="noul",
            instructions="Does this convey urgency?",
        ),
        "department": ChoiceQuestion(
            type="choice",
            instructions="Which team should handle this?",
            criteria={
                "billing": "Payments, invoicing, refunds",
                "technical": "Bugs, outages, integrations",
            },
        ),
        "frustration": ScoreQuestion(
            type="score",
            instructions="How frustrated is the customer?",
            criteria=["Calm", "Frustrated", "Very angry"],
        ),
    },
}

IR_RESPONSE: IRDecisionResponse = {
    "object": "decision",
    "model": "jev-1.13.0",
    "answers": {
        "is_urgent": NoulAnswer(type="noul", noul=0.92),
        "department": ChoiceAnswer(
            type="choice",
            choice="technical",
            probabilities={"billing": 0.08, "technical": 0.85, "sales": 0.07},
            confidence=0.82,
        ),
        "frustration": ScoreAnswer(
            type="score",
            score=1.6,
            legend={"0": "Calm", "1": "Frustrated", "2": "Very angry"},
            probabilities={"0": 0.05, "1": 0.3, "2": 0.65},
            confidence=0.78,
        ),
    },
    "usage": {"input_tokens": 588, "output_tokens": 212},
}


# ============================================================================
# Request conversion tests
# ============================================================================


class TestRequestFromProvider:
    def test_noul_passthrough(self, converter):
        ir = converter.request_from_provider(TYPESAFE_REQUEST)
        q = ir["questions"]["is_urgent"]
        assert q["type"] == "noul"
        assert q["instructions"] == "Does this convey urgency?"

    def test_choice_preserved(self, converter):
        ir = converter.request_from_provider(TYPESAFE_REQUEST)
        q = ir["questions"]["department"]
        assert q["type"] == "choice"
        assert "billing" in q["criteria"]

    def test_score_preserved(self, converter):
        ir = converter.request_from_provider(TYPESAFE_REQUEST)
        q = ir["questions"]["frustration"]
        assert q["type"] == "score"
        assert q["criteria"] == ["Calm", "Frustrated", "Very angry"]

    def test_state_and_model(self, converter):
        ir = converter.request_from_provider(TYPESAFE_REQUEST)
        assert ir["model"] == "jev-latest"
        assert ir["state"] == "Help! My payouts have been failing for 3 days."


class TestRequestToProvider:
    def test_noul_passthrough(self, converter):
        wire, warnings = converter.request_to_provider(IR_REQUEST)
        q = wire["questions"]["is_urgent"]
        assert q["type"] == "noul"

    def test_choice_preserved(self, converter):
        wire, _ = converter.request_to_provider(IR_REQUEST)
        q = wire["questions"]["department"]
        assert q["type"] == "choice"

    def test_score_preserved(self, converter):
        wire, _ = converter.request_to_provider(IR_REQUEST)
        q = wire["questions"]["frustration"]
        assert q["type"] == "score"

    def test_no_warnings(self, converter):
        _, warnings = converter.request_to_provider(IR_REQUEST)
        assert warnings == []


# ============================================================================
# Response conversion tests
# ============================================================================


class TestResponseFromProvider:
    def test_noul_answer_passthrough(self, converter):
        ir = converter.response_from_provider(TYPESAFE_RESPONSE)
        a = ir["answers"]["is_urgent"]
        assert a["type"] == "noul"
        assert a["noul"] == 0.92

    def test_choice_answer(self, converter):
        ir = converter.response_from_provider(TYPESAFE_RESPONSE)
        a = ir["answers"]["department"]
        assert a["type"] == "choice"
        assert a["choice"] == "technical"
        assert a["probabilities"]["technical"] == 0.85

    def test_score_answer(self, converter):
        ir = converter.response_from_provider(TYPESAFE_RESPONSE)
        a = ir["answers"]["frustration"]
        assert a["type"] == "score"
        assert a["score"] == 1.6

    def test_object_field(self, converter):
        ir = converter.response_from_provider(TYPESAFE_RESPONSE)
        assert ir["object"] == "decision"

    def test_usage(self, converter):
        ir = converter.response_from_provider(TYPESAFE_RESPONSE)
        assert ir["usage"]["input_tokens"] == 588
        assert ir["usage"]["output_tokens"] == 212


class TestResponseToProvider:
    def test_noul_answer_passthrough(self, converter):
        wire = converter.response_to_provider(IR_RESPONSE)
        a = wire["answers"]["is_urgent"]
        assert a["type"] == "noul"
        assert a["noul"] == 0.92

    def test_choice_answer(self, converter):
        wire = converter.response_to_provider(IR_RESPONSE)
        a = wire["answers"]["department"]
        assert a["type"] == "choice"
        assert a["choice"] == "technical"

    def test_score_answer(self, converter):
        wire = converter.response_to_provider(IR_RESPONSE)
        a = wire["answers"]["frustration"]
        assert a["type"] == "score"
        assert a["score"] == 1.6

    def test_no_object_field(self, converter):
        wire = converter.response_to_provider(IR_RESPONSE)
        assert "object" not in wire

    def test_usage(self, converter):
        wire = converter.response_to_provider(IR_RESPONSE)
        assert wire["usage"]["input_tokens"] == 588


# ============================================================================
# Round-trip tests
# ============================================================================


class TestRoundTrip:
    def test_provider_to_ir_to_provider(self, converter):
        """TypeSafe → IR → TypeSafe preserves data."""
        ir = converter.request_from_provider(TYPESAFE_REQUEST)
        wire, _ = converter.request_to_provider(ir)
        assert wire["model"] == TYPESAFE_REQUEST["model"]
        assert wire["state"] == TYPESAFE_REQUEST["state"]
        assert wire["questions"]["is_urgent"]["type"] == "noul"
        assert wire["questions"]["department"]["type"] == "choice"
        assert wire["questions"]["frustration"]["type"] == "score"

    def test_ir_to_provider_to_ir(self, converter):
        """IR → TypeSafe → IR preserves data."""
        wire, _ = converter.request_to_provider(IR_REQUEST)
        ir = converter.request_from_provider(wire)
        assert ir["questions"]["is_urgent"]["type"] == "noul"
        assert ir["questions"]["department"]["type"] == "choice"
        assert ir["questions"]["frustration"]["type"] == "score"

    def test_response_round_trip(self, converter):
        """TypeSafe response → IR → TypeSafe preserves data."""
        ir = converter.response_from_provider(TYPESAFE_RESPONSE)
        wire = converter.response_to_provider(ir)
        assert wire["answers"]["is_urgent"]["type"] == "noul"
        assert wire["answers"]["is_urgent"]["noul"] == 0.92
        assert wire["answers"]["department"]["choice"] == "technical"
        assert wire["answers"]["frustration"]["score"] == 1.6
        assert wire["usage"]["input_tokens"] == 588


# ============================================================================
# Edge cases
# ============================================================================


class TestEdgeCases:
    def test_noul_with_criteria(self, converter):
        req = {
            "model": "jev-latest",
            "state": "test",
            "questions": {
                "q": {
                    "type": "noul",
                    "instructions": "Is this true?",
                    "criteria": {"true": "Yes", "false": "No"},
                }
            },
        }
        ir = converter.request_from_provider(req)
        assert ir["questions"]["q"]["type"] == "noul"
        assert ir["questions"]["q"]["criteria"] == {"true": "Yes", "false": "No"}

    def test_structured_state(self, converter):
        req = {
            "model": "jev-latest",
            "state": {"message": "hello", "history": [1, 2, 3]},
            "questions": {
                "q": {"type": "noul", "instructions": "test"},
            },
        }
        ir = converter.request_from_provider(req)
        assert ir["state"] == {"message": "hello", "history": [1, 2, 3]}

    def test_response_without_usage(self, converter):
        resp = {
            "model": "jev-1.13.0",
            "answers": {
                "q": {"type": "noul", "noul": 0.5},
            },
        }
        ir = converter.response_from_provider(resp)
        assert "usage" not in ir

    def test_converter_tag(self, converter):
        assert converter._CONVERTER_TAG == "typesafe_decision"
