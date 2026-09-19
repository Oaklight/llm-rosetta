"""Tests for the LLM Chat decision converter."""

import json

import pytest

from llm_rosetta.converters.decision.llm_chat import LLMChatDecisionConverter
from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.types.ir.decision import (
    ChoiceQuestion,
    IRDecisionRequest,
    NoulQuestion,
    ScoreQuestion,
)


@pytest.fixture
def converter():
    return LLMChatDecisionConverter()


IR_REQUEST: IRDecisionRequest = {
    "model": "gpt-4o-mini",
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

MOCK_CHAT_RESPONSE = {
    "id": "chatcmpl-123",
    "model": "gpt-4o-mini",
    "choices": [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "answers": {
                            "is_urgent": 0.92,
                            "department": {"billing": 0.15, "technical": 0.85},
                            "frustration": {"0": 0.05, "1": 0.3, "2": 0.65},
                        }
                    }
                ),
            },
            "finish_reason": "stop",
        }
    ],
    "usage": {
        "prompt_tokens": 200,
        "completion_tokens": 50,
        "total_tokens": 250,
    },
}


class TestRequestToProvider:
    def test_produces_chat_format(self, converter):
        wire, warnings = converter.request_to_provider(IR_REQUEST)
        assert wire["model"] == "gpt-4o-mini"
        assert len(wire["messages"]) == 2
        assert wire["messages"][0]["role"] == "system"
        assert wire["messages"][1]["role"] == "user"

    def test_has_response_format(self, converter):
        wire, _ = converter.request_to_provider(IR_REQUEST)
        rf = wire["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "decision"
        assert rf["json_schema"]["strict"] is True

    def test_schema_has_all_questions(self, converter):
        wire, _ = converter.request_to_provider(IR_REQUEST)
        schema = wire["response_format"]["json_schema"]["schema"]
        answer_props = schema["properties"]["answers"]["properties"]
        assert "is_urgent" in answer_props
        assert "department" in answer_props
        assert "frustration" in answer_props

    def test_user_message_contains_state(self, converter):
        wire, _ = converter.request_to_provider(IR_REQUEST)
        assert "payouts" in wire["messages"][1]["content"]

    def test_system_prompt_contains_question_ids(self, converter):
        wire, _ = converter.request_to_provider(IR_REQUEST)
        system = wire["messages"][0]["content"]
        assert "is_urgent" in system
        assert "department" in system
        assert "frustration" in system

    def test_no_warnings(self, converter):
        _, warnings = converter.request_to_provider(IR_REQUEST)
        assert warnings == []


class TestResponseFromProvider:
    def test_parses_noul_answer(self, converter):
        ctx = ConversionContext()
        ctx.options["_decision_questions"] = IR_REQUEST["questions"]
        ir = converter.response_from_provider(MOCK_CHAT_RESPONSE, context=ctx)
        a = ir["answers"]["is_urgent"]
        assert a["type"] == "noul"
        assert a["noul"] == 0.92

    def test_parses_choice_answer(self, converter):
        ctx = ConversionContext()
        ctx.options["_decision_questions"] = IR_REQUEST["questions"]
        ir = converter.response_from_provider(MOCK_CHAT_RESPONSE, context=ctx)
        a = ir["answers"]["department"]
        assert a["type"] == "choice"
        assert a["choice"] == "technical"
        assert a["probabilities"]["technical"] == 0.85
        assert 0 < a["confidence"] <= 1

    def test_parses_score_answer(self, converter):
        ctx = ConversionContext()
        ctx.options["_decision_questions"] = IR_REQUEST["questions"]
        ir = converter.response_from_provider(MOCK_CHAT_RESPONSE, context=ctx)
        a = ir["answers"]["frustration"]
        assert a["type"] == "score"
        assert a["score"] == pytest.approx(0 * 0.05 + 1 * 0.3 + 2 * 0.65)
        assert a["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}

    def test_object_field(self, converter):
        ctx = ConversionContext()
        ctx.options["_decision_questions"] = IR_REQUEST["questions"]
        ir = converter.response_from_provider(MOCK_CHAT_RESPONSE, context=ctx)
        assert ir["object"] == "decision"

    def test_usage_mapping(self, converter):
        ctx = ConversionContext()
        ctx.options["_decision_questions"] = IR_REQUEST["questions"]
        ir = converter.response_from_provider(MOCK_CHAT_RESPONSE, context=ctx)
        assert ir["usage"]["input_tokens"] == 200
        assert ir["usage"]["output_tokens"] == 50


class TestRoundTrip:
    def test_request_then_response(self, converter):
        """IR request → chat request → (mock) chat response → IR response."""
        wire, _ = converter.request_to_provider(IR_REQUEST)
        assert wire["response_format"]["type"] == "json_schema"

        ctx = ConversionContext()
        ctx.options["_decision_questions"] = IR_REQUEST["questions"]
        ir = converter.response_from_provider(MOCK_CHAT_RESPONSE, context=ctx)

        assert ir["answers"]["is_urgent"]["type"] == "noul"
        assert ir["answers"]["department"]["type"] == "choice"
        assert ir["answers"]["frustration"]["type"] == "score"

    def test_context_carries_questions(self, converter):
        """request_to_provider stores questions in context for response parsing."""
        ctx = ConversionContext()
        converter.request_to_provider(IR_REQUEST, context=ctx)
        assert "_decision_questions" in ctx.options
        ir = converter.response_from_provider(MOCK_CHAT_RESPONSE, context=ctx)
        assert len(ir["answers"]) == 3


class TestEdgeCases:
    def test_structured_state(self, converter):
        req: IRDecisionRequest = {
            "model": "gpt-4o",
            "state": {"message": "test", "data": [1, 2, 3]},
            "questions": {"q": NoulQuestion(type="noul", instructions="test")},
        }
        wire, _ = converter.request_to_provider(req)
        user_content = wire["messages"][1]["content"]
        parsed = json.loads(user_content)
        assert parsed == {"message": "test", "data": [1, 2, 3]}

    def test_single_question(self, converter):
        req: IRDecisionRequest = {
            "model": "m",
            "state": "s",
            "questions": {"q": NoulQuestion(type="noul", instructions="yes?")},
        }
        wire, _ = converter.request_to_provider(req)
        schema = wire["response_format"]["json_schema"]["schema"]
        assert list(schema["properties"]["answers"]["properties"].keys()) == ["q"]

    def test_request_from_provider_raises(self, converter):
        with pytest.raises(NotImplementedError):
            converter.request_from_provider({"messages": []})

    def test_response_to_provider_raises(self, converter):
        with pytest.raises(NotImplementedError):
            converter.response_to_provider(
                {"object": "decision", "model": "m", "answers": {}}
            )

    def test_converter_tag(self, converter):
        assert converter._CONVERTER_TAG == "llm_chat_decision"

    def test_empty_content_raises(self, converter):
        ctx = ConversionContext()
        ctx.options["_decision_questions"] = IR_REQUEST["questions"]
        empty_response = {"choices": [{"message": {"content": ""}}]}
        with pytest.raises(ValueError, match="no message content"):
            converter.response_from_provider(empty_response, context=ctx)

    def test_malformed_json_raises(self, converter):
        ctx = ConversionContext()
        ctx.options["_decision_questions"] = IR_REQUEST["questions"]
        bad_response = {"choices": [{"message": {"content": "not json{"}}]}
        with pytest.raises(ValueError, match="Failed to parse"):
            converter.response_from_provider(bad_response, context=ctx)

    def test_missing_context_questions_raises(self, converter):
        ctx = ConversionContext()
        with pytest.raises(ValueError, match="_decision_questions"):
            converter.response_from_provider(MOCK_CHAT_RESPONSE, context=ctx)
