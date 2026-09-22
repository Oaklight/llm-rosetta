"""Tests for the LLM Chat decision converter."""

import json
from typing import Any, cast

import pytest

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.decision.llm_chat import LLMChatDecisionConverter
from llm_rosetta.types.ir.decision import (
    ChoiceQuestion,
    IRDecisionRequest,
    NoulQuestion,
    ScoreQuestion,
)


@pytest.fixture
def converter():  # openai_chat default
    return LLMChatDecisionConverter()


@pytest.fixture
def anthropic_converter():
    return LLMChatDecisionConverter(output_format="anthropic")


@pytest.fixture
def prompted_converter():
    return LLMChatDecisionConverter(output_format="prompted")


@pytest.fixture
def google_generate_converter():
    return LLMChatDecisionConverter(output_format="google_generate")


@pytest.fixture
def google_interactions_converter():
    return LLMChatDecisionConverter(output_format="google_interactions")


@pytest.fixture
def openai_responses_converter():
    return LLMChatDecisionConverter(output_format="openai_responses")


@pytest.fixture
def discrete_converter():
    return LLMChatDecisionConverter(answer_mode="discrete")


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
    "usage": {"prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250},
}

MOCK_DISCRETE_RESPONSE = {
    "model": "gpt-4o-mini",
    "choices": [
        {
            "message": {
                "content": json.dumps(
                    {
                        "answers": {
                            "is_urgent": True,
                            "department": "technical",
                            "frustration": 2,
                        }
                    }
                ),
            },
        }
    ],
}

MOCK_GOOGLE_RESPONSE = {
    "candidates": [
        {
            "content": {
                "parts": [
                    {
                        "text": json.dumps(
                            {
                                "answers": {
                                    "is_urgent": 0.92,
                                    "department": {
                                        "billing": 0.15,
                                        "technical": 0.85,
                                    },
                                    "frustration": {
                                        "0": 0.05,
                                        "1": 0.3,
                                        "2": 0.65,
                                    },
                                }
                            }
                        )
                    }
                ],
                "role": "model",
            },
            "finishReason": "STOP",
        }
    ],
    "usageMetadata": {
        "promptTokenCount": 200,
        "candidatesTokenCount": 50,
        "totalTokenCount": 250,
    },
    "modelVersion": "gemini-2.0-flash",
}

MOCK_OPENAI_RESPONSES_RESPONSE = {
    "id": "resp_123",
    "model": "gpt-4o-mini",
    "status": "completed",
    "output_text": json.dumps(
        {
            "answers": {
                "is_urgent": 0.92,
                "department": {"billing": 0.15, "technical": 0.85},
                "frustration": {"0": 0.05, "1": 0.3, "2": 0.65},
            }
        }
    ),
    "usage": {"input_tokens": 200, "output_tokens": 50},
}

MOCK_ANTHROPIC_RESPONSE = {
    "model": "claude-haiku-4-5",
    "content": [
        {
            "type": "text",
            "text": json.dumps(
                {
                    "answers": {
                        "is_urgent": 0.92,
                        "department": {"billing": 0.15, "technical": 0.85},
                        "frustration": {"0": 0.05, "1": 0.3, "2": 0.65},
                    }
                }
            ),
        }
    ],
    "usage": {"input_tokens": 200, "output_tokens": 50},
}


# ============================================================================
# OpenAI format (default)
# ============================================================================


class TestOpenAIFormat:
    def test_produces_chat_format(self, converter: LLMChatDecisionConverter):
        wire, warnings = converter.request_to_provider(IR_REQUEST)
        assert wire["model"] == "gpt-4o-mini"
        assert len(wire["messages"]) == 2
        assert wire["messages"][0]["role"] == "system"
        assert wire["messages"][1]["role"] == "user"

    def test_has_response_format(self, converter: LLMChatDecisionConverter):
        wire, _ = converter.request_to_provider(IR_REQUEST)
        rf = wire["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "decision"
        assert rf["json_schema"]["strict"] is True

    def test_no_output_config(self, converter: LLMChatDecisionConverter):
        wire, _ = converter.request_to_provider(IR_REQUEST)
        assert "output_config" not in wire

    def test_schema_has_all_questions(self, converter: LLMChatDecisionConverter):
        wire, _ = converter.request_to_provider(IR_REQUEST)
        schema = wire["response_format"]["json_schema"]["schema"]
        answer_props = schema["properties"]["answers"]["properties"]
        assert "is_urgent" in answer_props
        assert "department" in answer_props
        assert "frustration" in answer_props

    def test_parses_response(self, converter: LLMChatDecisionConverter):
        ctx = ConversionContext()
        converter.request_to_provider(IR_REQUEST, context=ctx)
        ir = converter.response_from_provider(MOCK_CHAT_RESPONSE, context=ctx)
        assert cast(Any, ir["answers"]["is_urgent"])["noul"] == 0.92
        assert cast(Any, ir["answers"]["department"])["choice"] == "technical"

    def test_no_warnings(self, converter: LLMChatDecisionConverter):
        _, warnings = converter.request_to_provider(IR_REQUEST)
        assert warnings == []


# ============================================================================
# Anthropic format
# ============================================================================


class TestAnthropicFormat:
    def test_has_output_config(self, anthropic_converter: LLMChatDecisionConverter):
        wire, _ = anthropic_converter.request_to_provider(IR_REQUEST)
        oc = wire["output_config"]
        assert oc["format"]["type"] == "json_schema"
        assert "schema" in oc["format"]

    def test_no_response_format(self, anthropic_converter: LLMChatDecisionConverter):
        wire, _ = anthropic_converter.request_to_provider(IR_REQUEST)
        assert "response_format" not in wire

    def test_system_separate_from_messages(
        self, anthropic_converter: LLMChatDecisionConverter
    ):
        wire, _ = anthropic_converter.request_to_provider(IR_REQUEST)
        assert wire["messages"][0]["role"] == "system"

    def test_parses_anthropic_response(
        self, anthropic_converter: LLMChatDecisionConverter
    ):
        ctx = ConversionContext()
        anthropic_converter.request_to_provider(IR_REQUEST, context=ctx)
        ir = anthropic_converter.response_from_provider(
            MOCK_ANTHROPIC_RESPONSE, context=ctx
        )
        assert cast(Any, ir["answers"]["is_urgent"])["noul"] == 0.92
        assert cast(Any, ir["answers"]["department"])["choice"] == "technical"
        assert ir["usage"]["input_tokens"] == 200


# ============================================================================
# Google format
# ============================================================================


class TestGoogleGenerateFormat:
    def test_has_response_schema(
        self, google_generate_converter: LLMChatDecisionConverter
    ):
        wire, _ = google_generate_converter.request_to_provider(IR_REQUEST)
        assert wire["response_mime_type"] == "application/json"
        assert "response_schema" in wire
        assert wire["response_schema"]["required"] == ["answers"]

    def test_no_response_format_or_output_config(
        self, google_generate_converter: LLMChatDecisionConverter
    ):
        wire, _ = google_generate_converter.request_to_provider(IR_REQUEST)
        assert "response_format" not in wire
        assert "output_config" not in wire

    def test_parses_google_response(
        self, google_generate_converter: LLMChatDecisionConverter
    ):
        ctx = ConversionContext()
        google_generate_converter.request_to_provider(IR_REQUEST, context=ctx)
        ir = google_generate_converter.response_from_provider(
            MOCK_GOOGLE_RESPONSE, context=ctx
        )
        assert cast(Any, ir["answers"]["is_urgent"])["noul"] == 0.92
        assert cast(Any, ir["answers"]["department"])["choice"] == "technical"


# ============================================================================
# OpenAI Responses format
# ============================================================================


class TestOpenAIResponsesFormat:
    def test_has_text_format(
        self, openai_responses_converter: LLMChatDecisionConverter
    ):
        wire, _ = openai_responses_converter.request_to_provider(IR_REQUEST)
        text = wire["text"]
        assert text["format"]["type"] == "json_schema"
        assert text["format"]["strict"] is True

    def test_no_response_format(
        self, openai_responses_converter: LLMChatDecisionConverter
    ):
        wire, _ = openai_responses_converter.request_to_provider(IR_REQUEST)
        assert "response_format" not in wire

    def test_parses_responses_output_text(
        self, openai_responses_converter: LLMChatDecisionConverter
    ):
        ctx = ConversionContext()
        openai_responses_converter.request_to_provider(IR_REQUEST, context=ctx)
        ir = openai_responses_converter.response_from_provider(
            MOCK_OPENAI_RESPONSES_RESPONSE, context=ctx
        )
        assert cast(Any, ir["answers"]["is_urgent"])["noul"] == 0.92
        assert cast(Any, ir["answers"]["department"])["choice"] == "technical"


# ============================================================================
# Google Interactions format
# ============================================================================


class TestGoogleInteractionsFormat:
    def test_has_response_format_with_mime(
        self, google_interactions_converter: LLMChatDecisionConverter
    ):
        wire, _ = google_interactions_converter.request_to_provider(IR_REQUEST)
        rf = wire["response_format"]
        assert rf["type"] == "text"
        assert rf["mime_type"] == "application/json"
        assert "response_schema" in rf

    def test_no_response_mime_type_top_level(
        self, google_interactions_converter: LLMChatDecisionConverter
    ):
        wire, _ = google_interactions_converter.request_to_provider(IR_REQUEST)
        assert "response_mime_type" not in wire

    def test_parses_google_response(
        self, google_interactions_converter: LLMChatDecisionConverter
    ):
        ctx = ConversionContext()
        google_interactions_converter.request_to_provider(IR_REQUEST, context=ctx)
        ir = google_interactions_converter.response_from_provider(
            MOCK_GOOGLE_RESPONSE, context=ctx
        )
        assert cast(Any, ir["answers"]["is_urgent"])["noul"] == 0.92


# ============================================================================
# Prompted fallback
# ============================================================================


class TestPromptedFallback:
    def test_no_response_format_or_output_config(
        self, prompted_converter: LLMChatDecisionConverter
    ):
        wire, _ = prompted_converter.request_to_provider(IR_REQUEST)
        assert "response_format" not in wire
        assert "output_config" not in wire

    def test_schema_embedded_in_prompt(
        self, prompted_converter: LLMChatDecisionConverter
    ):
        wire, _ = prompted_converter.request_to_provider(IR_REQUEST)
        system = wire["messages"][0]["content"]
        assert "Return one JSON object that matches this schema exactly" in system
        assert '"answers"' in system

    def test_parses_fenced_response(self, prompted_converter: LLMChatDecisionConverter):
        fenced_content = '```json\n{"answers": {"is_urgent": 0.8}}\n```'
        response = {"choices": [{"message": {"content": fenced_content}}]}
        req: IRDecisionRequest = {
            "model": "m",
            "state": "test",
            "questions": {"is_urgent": NoulQuestion(type="noul", instructions="test")},
        }
        ctx = ConversionContext()
        prompted_converter.request_to_provider(req, context=ctx)
        ir = prompted_converter.response_from_provider(response, context=ctx)
        assert cast(Any, ir["answers"]["is_urgent"])["noul"] == 0.8


# ============================================================================
# Discrete mode
# ============================================================================


class TestDiscreteMode:
    def test_schema_uses_boolean_for_noul(
        self, discrete_converter: LLMChatDecisionConverter
    ):
        wire, _ = discrete_converter.request_to_provider(IR_REQUEST)
        schema = wire["response_format"]["json_schema"]["schema"]
        noul_prop = schema["properties"]["answers"]["properties"]["is_urgent"]
        assert noul_prop["type"] == "boolean"

    def test_schema_uses_enum_for_choice(
        self, discrete_converter: LLMChatDecisionConverter
    ):
        wire, _ = discrete_converter.request_to_provider(IR_REQUEST)
        schema = wire["response_format"]["json_schema"]["schema"]
        choice_prop = schema["properties"]["answers"]["properties"]["department"]
        assert "enum" in choice_prop
        assert set(choice_prop.get("enum", [])) == {"billing", "technical"}

    def test_schema_uses_integer_for_score(
        self, discrete_converter: LLMChatDecisionConverter
    ):
        wire, _ = discrete_converter.request_to_provider(IR_REQUEST)
        schema = wire["response_format"]["json_schema"]["schema"]
        score_prop = schema["properties"]["answers"]["properties"]["frustration"]
        assert score_prop["type"] == "integer"

    def test_parses_discrete_response(
        self, discrete_converter: LLMChatDecisionConverter
    ):
        ctx = ConversionContext()
        discrete_converter.request_to_provider(IR_REQUEST, context=ctx)
        ir = discrete_converter.response_from_provider(
            MOCK_DISCRETE_RESPONSE, context=ctx
        )
        assert cast(Any, ir["answers"]["is_urgent"])["noul"] == 1.0
        assert cast(Any, ir["answers"]["department"])["choice"] == "technical"
        assert cast(Any, ir["answers"]["department"])["confidence"] == 1.0
        a: Any = ir["answers"]["frustration"]
        assert a["score"] == 2.0


# ============================================================================
# State injection protection
# ============================================================================


class TestStateInjection:
    def test_escapes_angle_brackets(self, converter: LLMChatDecisionConverter):
        req: IRDecisionRequest = {
            "model": "m",
            "state": "<script>alert('xss')</script>",
            "questions": {"q": NoulQuestion(type="noul", instructions="test")},
        }
        wire, _ = converter.request_to_provider(req)
        user = wire["messages"][1]["content"]
        assert "<script>" not in user
        assert "\\u003c" in user
        assert "<document>" in user
        assert "</document>" in user

    def test_wraps_in_document_tags(self, converter: LLMChatDecisionConverter):
        wire, _ = converter.request_to_provider(IR_REQUEST)
        user = wire["messages"][1]["content"]
        assert user.startswith("<document>")
        assert user.endswith("</document>")


# ============================================================================
# Corrective retry
# ============================================================================


class TestRetrySupport:
    def test_build_retry_messages(self, converter: LLMChatDecisionConverter):
        msgs = converter.build_retry_messages("bad json{", "JSONDecodeError")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "assistant"
        assert msgs[0]["content"] == "bad json{"
        assert msgs[1]["role"] == "user"
        assert "did not match" in msgs[1]["content"]
        assert "JSONDecodeError" in msgs[1]["content"]

    def test_max_retries_stored_in_context(self):
        conv = LLMChatDecisionConverter(max_retries=3)
        ctx = ConversionContext()
        conv.request_to_provider(IR_REQUEST, context=ctx)
        assert ctx.options["_decision_max_retries"] == 3


# ============================================================================
# System prompt
# ============================================================================


class TestSystemPrompt:
    def test_probability_prompt_mentions_calibration(
        self, converter: LLMChatDecisionConverter
    ):
        wire, _ = converter.request_to_provider(IR_REQUEST)
        system = wire["messages"][0]["content"]
        assert "genuine uncertainty" in system
        assert "sum to 1" in system

    def test_discrete_prompt_differs(
        self, discrete_converter: LLMChatDecisionConverter
    ):
        wire, _ = discrete_converter.request_to_provider(IR_REQUEST)
        system = wire["messages"][0]["content"]
        assert "exactly one allowed value" in system

    def test_contains_question_ids(self, converter: LLMChatDecisionConverter):
        wire, _ = converter.request_to_provider(IR_REQUEST)
        system = wire["messages"][0]["content"]
        assert "is_urgent" in system
        assert "department" in system
        assert "frustration" in system

    def test_untrusted_data_warning(self, converter: LLMChatDecisionConverter):
        wire, _ = converter.request_to_provider(IR_REQUEST)
        system = wire["messages"][0]["content"]
        assert "untrusted data" in system


# ============================================================================
# Error handling
# ============================================================================


class TestErrorHandling:
    def test_empty_content_raises(self, converter: LLMChatDecisionConverter):
        ctx = ConversionContext()
        ctx.options["_decision_questions"] = IR_REQUEST["questions"]
        empty_response = {"choices": [{"message": {"content": ""}}]}
        with pytest.raises(ValueError, match="no message content"):
            converter.response_from_provider(empty_response, context=ctx)

    def test_malformed_json_raises(self, converter: LLMChatDecisionConverter):
        ctx = ConversionContext()
        ctx.options["_decision_questions"] = IR_REQUEST["questions"]
        bad_response = {"choices": [{"message": {"content": "not json{"}}]}
        with pytest.raises(ValueError, match="Failed to parse"):
            converter.response_from_provider(bad_response, context=ctx)

    def test_missing_context_questions_raises(
        self, converter: LLMChatDecisionConverter
    ):
        ctx = ConversionContext()
        with pytest.raises(ValueError, match="_decision_questions"):
            converter.response_from_provider(MOCK_CHAT_RESPONSE, context=ctx)

    def test_request_from_provider_raises(self, converter: LLMChatDecisionConverter):
        with pytest.raises(NotImplementedError):
            converter.request_from_provider({"messages": []})

    def test_response_to_provider_raises(self, converter: LLMChatDecisionConverter):
        with pytest.raises(NotImplementedError):
            converter.response_to_provider(
                {"object": "decision", "model": "m", "answers": {}}
            )

    def test_converter_tag(self, converter: LLMChatDecisionConverter):
        assert converter._CONVERTER_TAG == "llm_chat_decision"


# ============================================================================
# Round trip
# ============================================================================


class TestRoundTrip:
    def test_context_carries_questions(self, converter: LLMChatDecisionConverter):
        ctx = ConversionContext()
        converter.request_to_provider(IR_REQUEST, context=ctx)
        assert "_decision_questions" in ctx.options
        ir = converter.response_from_provider(MOCK_CHAT_RESPONSE, context=ctx)
        assert len(ir["answers"]) == 3

    def test_usage_mapping(self, converter: LLMChatDecisionConverter):
        ctx = ConversionContext()
        converter.request_to_provider(IR_REQUEST, context=ctx)
        ir = converter.response_from_provider(MOCK_CHAT_RESPONSE, context=ctx)
        assert ir["usage"]["input_tokens"] == 200
        assert ir["usage"]["output_tokens"] == 50
