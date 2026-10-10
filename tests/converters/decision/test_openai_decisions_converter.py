"""Tests for the OpenAI Decisions converter (``/v1/decisions``)."""

import pytest

from llm_rosetta.converters.decision.openai import (
    OpenAIDecisionsConverter,
    _MARK,
)

OPENAI_REQUEST = {
    "model": "gpt-6-luna",
    "input": "I was charged twice for my order.",
    "questions": [
        {"type": "predicate", "name": "urgent", "instructions": "Is this urgent?"},
        {
            "type": "choice",
            "name": "dept",
            "instructions": "Which department?",
            "choices": [
                {"value": "billing", "description": "Payments"},
                {"value": "technical", "description": "Bugs"},
            ],
        },
        {
            "type": "score",
            "name": "severity",
            "instructions": "How severe?",
            "levels": [
                {"label": "Cosmetic", "description": "minor"},
                {"label": "Blocked"},
            ],
        },
    ],
}

OPENAI_RESPONSE = {
    "model": "gpt-6-luna",
    "answers": [
        {"type": "predicate", "name": "urgent", "probability": 0.92},
        {
            "type": "choice",
            "name": "dept",
            "choice": "billing",
            "probabilities": [
                {"value": "billing", "probability": 0.95},
                {"value": "technical", "probability": 0.02},
            ],
            "confidence": 0.93,
        },
        {
            "type": "score",
            "name": "severity",
            "score": 1.1,
            "probabilities": [
                {"value": 0, "label": "Cosmetic", "probability": 0.1},
                {"value": 1, "label": "Blocked", "probability": 0.9},
            ],
            "confidence": 0.6,
        },
        {"type": "refusal", "name": "extra"},
    ],
    "usage": {"input_tokens": 42, "output_tokens": 0, "total_tokens": 42},
}

IR_REQUEST = {
    "model": "gpt-6-luna",
    "state": "I was charged twice for my order.",
    "questions": {
        "urgent": {"type": "assertion", "instructions": "Is this urgent?"},
        "dept": {
            "type": "choice",
            "instructions": "Which department?",
            "criteria": [
                {"label": "billing", "description": "Payments"},
                {"label": "technical", "description": "Bugs"},
            ],
        },
        "severity": {
            "type": "score",
            "instructions": "How severe?",
            "criteria": [
                {"label": "Cosmetic", "description": "minor"},
                {"label": "Blocked"},
            ],
        },
    },
}


@pytest.fixture
def converter():
    return OpenAIDecisionsConverter()


class TestRequestFromProvider:
    def test_questions_list_to_map(self, converter):
        ir = converter.request_from_provider(OPENAI_REQUEST)
        assert set(ir["questions"]) == {"urgent", "dept", "severity"}
        assert ir["questions"]["urgent"]["type"] == "assertion"

    def test_choice_values_to_labels(self, converter):
        ir = converter.request_from_provider(OPENAI_REQUEST)
        assert ir["questions"]["dept"]["criteria"] == [
            {"label": "billing", "description": "Payments"},
            {"label": "technical", "description": "Bugs"},
        ]

    def test_score_levels_to_entries(self, converter):
        ir = converter.request_from_provider(OPENAI_REQUEST)
        assert [e["label"] for e in ir["questions"]["severity"]["criteria"]] == [
            "Cosmetic",
            "Blocked",
        ]

    def test_state_string(self, converter):
        ir = converter.request_from_provider(OPENAI_REQUEST)
        assert ir["state"] == "I was charged twice for my order."


class TestRequestToProvider:
    def test_map_to_list_with_names(self, converter):
        wire, _ = converter.request_to_provider(IR_REQUEST)
        names = [q["name"] for q in wire["questions"]]
        assert names == ["urgent", "dept", "severity"]
        assert wire["questions"][0]["type"] == "predicate"

    def test_choice_to_choices(self, converter):
        wire, _ = converter.request_to_provider(IR_REQUEST)
        assert wire["questions"][1]["choices"] == [
            {"value": "billing", "description": "Payments"},
            {"value": "technical", "description": "Bugs"},
        ]

    def test_score_to_levels(self, converter):
        wire, _ = converter.request_to_provider(IR_REQUEST)
        assert wire["questions"][2]["levels"] == [
            {"label": "Cosmetic", "description": "minor"},
            {"label": "Blocked"},
        ]

    def test_no_warnings(self, converter):
        _, warnings = converter.request_to_provider(IR_REQUEST)
        assert warnings == []


class TestAssertionCriteriaEncoding:
    def test_plain_instructions_pass_through(self, converter):
        wire, _ = converter.request_to_provider(IR_REQUEST)
        assert wire["questions"][0]["instructions"] == "Is this urgent?"

    def test_criteria_folded_and_reversible(self, converter):
        ir = {
            "model": "gpt-6-luna",
            "state": "x",
            "questions": {
                "q": {
                    "type": "assertion",
                    "instructions": "Is it urgent?",
                    "criteria": [
                        {"label": False, "description": "no"},
                        {"label": True, "description": "yes"},
                    ],
                }
            },
        }
        wire, _ = converter.request_to_provider(ir)
        assert wire["questions"][0]["instructions"].startswith(_MARK)
        back = converter.request_from_provider(wire)
        assert back["questions"]["q"]["instructions"] == "Is it urgent?"
        assert back["questions"]["q"]["criteria"] == [
            {"label": False, "description": "no"},
            {"label": True, "description": "yes"},
        ]


class TestStateShape:
    def test_content_parts_to_image_input(self, converter):
        ir = {
            "model": "gpt-6-luna",
            "state": [
                {"type": "text", "text": "look"},
                {"type": "image", "image_url": "data:image/png;base64,AA"},
            ],
            "questions": {"q": {"type": "assertion", "instructions": "x"}},
        }
        wire, _ = converter.request_to_provider(ir)
        content = wire["input"][0]["content"]
        assert content[0] == {"type": "input_text", "text": "look"}
        assert content[1]["type"] == "input_image"

    def test_image_input_to_parts(self, converter):
        req = {
            "model": "gpt-6-luna",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": "look"},
                        {
                            "type": "input_image",
                            "image_url": "data:image/png;base64,AA",
                        },
                    ],
                }
            ],
            "questions": [{"type": "predicate", "name": "q", "instructions": "x"}],
        }
        ir = converter.request_from_provider(req)
        assert ir["state"] == [
            {"type": "text", "text": "look"},
            {"type": "image", "image_url": "data:image/png;base64,AA"},
        ]

    def test_dict_state_stringified(self, converter):
        ir = {
            "model": "gpt-6-luna",
            "state": {"order_id": "A-1"},
            "questions": {"q": {"type": "assertion", "instructions": "x"}},
        }
        wire, _ = converter.request_to_provider(ir)
        assert wire["input"] == '{"order_id": "A-1"}'


class TestResponseFromProvider:
    def test_answers_list_to_map(self, converter):
        ir = converter.response_from_provider(OPENAI_RESPONSE)
        assert ir["object"] == "decision"
        assert set(ir["answers"]) == {"urgent", "dept", "severity", "extra"}

    def test_predicate_answer(self, converter):
        ir = converter.response_from_provider(OPENAI_RESPONSE)
        assert ir["answers"]["urgent"] == {"type": "assertion", "probability": 0.92}

    def test_choice_probabilities_array_to_dict(self, converter):
        ir = converter.response_from_provider(OPENAI_RESPONSE)
        a = ir["answers"]["dept"]
        assert a["choice"] == "billing"
        assert a["probabilities"] == {"billing": 0.95, "technical": 0.02}

    def test_score_probabilities_keyed_by_label(self, converter):
        ir = converter.response_from_provider(OPENAI_RESPONSE)
        a = ir["answers"]["severity"]
        assert a["score"] == 1.1
        assert a["probabilities"] == {"Cosmetic": 0.1, "Blocked": 0.9}

    def test_refusal(self, converter):
        ir = converter.response_from_provider(OPENAI_RESPONSE)
        assert ir["answers"]["extra"] == {"type": "refusal"}

    def test_usage(self, converter):
        ir = converter.response_from_provider(OPENAI_RESPONSE)
        assert ir["usage"]["input_tokens"] == 42


class TestResponseToProvider:
    def test_answers_map_to_list(self, converter):
        conv = OpenAIDecisionsConverter()
        ir = conv.response_from_provider(OPENAI_RESPONSE)
        wire = conv.response_to_provider(ir)
        assert isinstance(wire["answers"], list)
        by_name = {a["name"]: a for a in wire["answers"]}
        assert by_name["urgent"] == {
            "type": "predicate",
            "name": "urgent",
            "probability": 0.92,
        }

    def test_choice_probabilities_dict_to_array(self, converter):
        ir = converter.response_from_provider(OPENAI_RESPONSE)
        wire = converter.response_to_provider(ir)
        by_name = {a["name"]: a for a in wire["answers"]}
        assert by_name["dept"]["probabilities"] == [
            {"value": "billing", "probability": 0.95},
            {"value": "technical", "probability": 0.02},
        ]

    def test_score_probabilities_dict_to_array(self, converter):
        ir = converter.response_from_provider(OPENAI_RESPONSE)
        wire = converter.response_to_provider(ir)
        by_name = {a["name"]: a for a in wire["answers"]}
        assert by_name["severity"]["probabilities"] == [
            {"value": 0, "label": "Cosmetic", "probability": 0.1},
            {"value": 1, "label": "Blocked", "probability": 0.9},
        ]


class TestRoundTrip:
    def test_request_round_trip(self, converter):
        ir = converter.request_from_provider(OPENAI_REQUEST)
        wire, _ = converter.request_to_provider(ir)
        assert wire["questions"] == OPENAI_REQUEST["questions"]

    def test_response_round_trip(self, converter):
        ir = converter.response_from_provider(OPENAI_RESPONSE)
        wire = converter.response_to_provider(ir)
        assert wire["answers"] == OPENAI_RESPONSE["answers"]

    def test_converter_tag(self, converter):
        assert converter._CONVERTER_TAG == "openai_decisions"


class TestReviewFixes:
    def test_unknown_probability_round_trip(self, converter):
        resp = {
            "model": "gpt-6-luna",
            "answers": [
                {
                    "type": "predicate",
                    "name": "a",
                    "probability": 0.9,
                    "unknown_probability": 0.1,
                },
                {
                    "type": "choice",
                    "name": "c",
                    "choice": "billing",
                    "probabilities": [
                        {"value": "billing", "probability": 0.8},
                        {"value": "technical", "probability": 0.2},
                    ],
                    "unknown_probability": 0.05,
                },
                {
                    "type": "score",
                    "name": "s",
                    "score": 1.0,
                    "probabilities": [
                        {"value": 0, "label": "A", "probability": 0.4},
                        {"value": 1, "label": "B", "probability": 0.6},
                    ],
                    "unknown_probability": 0.02,
                },
            ],
        }
        ir = converter.response_from_provider(resp)
        for qid in ("a", "c", "s"):
            assert ir["answers"][qid]["unknown_probability"] is not None
        wire = converter.response_to_provider(ir)
        by_name = {x["name"]: x for x in wire["answers"]}
        assert by_name["a"]["unknown_probability"] == 0.1
        assert by_name["c"]["unknown_probability"] == 0.05
        assert by_name["s"]["unknown_probability"] == 0.02

    def test_unknown_answer_type_raises(self, converter):
        with pytest.raises(ValueError, match="Unknown OpenAI Decisions answer type"):
            converter.response_from_provider(
                {"model": "gpt-6-luna", "answers": [{"type": "weird", "name": "q"}]}
            )

    def test_score_levels_sorted_by_ordinal(self, converter):
        resp = {
            "model": "gpt-6-luna",
            "answers": [
                {
                    "type": "score",
                    "name": "s",
                    "score": 1.0,
                    "probabilities": [
                        {"value": 1, "label": "Blocked", "probability": 0.9},
                        {"value": 0, "label": "Cosmetic", "probability": 0.1},
                    ],
                },
            ],
        }
        ir = converter.response_from_provider(resp)
        # IR insertion order follows the ordinal, not the wire array order.
        assert list(ir["answers"]["s"]["probabilities"]) == ["Cosmetic", "Blocked"]
        wire = converter.response_to_provider(ir)
        assert wire["answers"][0]["probabilities"] == [
            {"value": 0, "label": "Cosmetic", "probability": 0.1},
            {"value": 1, "label": "Blocked", "probability": 0.9},
        ]

    def test_assertion_fold_warns(self, converter):
        from llm_rosetta.converters.base.context import ConversionContext

        ir = {
            "model": "gpt-6-luna",
            "state": "x",
            "questions": {
                "q": {
                    "type": "assertion",
                    "instructions": "Is it urgent?",
                    "criteria": [{"label": True, "description": "yes"}],
                }
            },
        }
        _, warnings = converter.request_to_provider(ir, context=ConversionContext())
        assert any("Folded assertion criteria" in w for w in warnings)


class TestReviewFixes2:
    def test_unknown_question_type_keeps_instructions(self, converter):
        ir = {
            "model": "m",
            "state": "x",
            "questions": {"q": {"type": "rating", "instructions": "Rate it"}},
        }
        wire, _ = converter.request_to_provider(ir)
        assert wire["questions"][0]["instructions"] == "Rate it"

    def test_unsupported_state_part_warns(self, converter):
        ir = {
            "model": "m",
            "state": [{"type": "audio", "data": "x"}, {"type": "text", "text": "hi"}],
            "questions": {},
        }
        wire, warnings = converter.request_to_provider(ir)
        assert wire["input"][0]["content"] == [{"type": "input_text", "text": "hi"}]
        assert any("unsupported state part" in w for w in warnings)

    def test_unsupported_input_part_warns(self, converter):
        from llm_rosetta.converters.base.context import ConversionContext

        req = {
            "model": "m",
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_file", "file_id": "f1"},
                        {"type": "input_text", "text": "hi"},
                    ],
                }
            ],
            "questions": [],
        }
        ctx = ConversionContext()
        ir = converter.request_from_provider(req, context=ctx)
        assert ir["state"] == [{"type": "text", "text": "hi"}]
        assert any("unsupported input part" in w for w in ctx.warnings)

    def test_image_detail_passthrough(self, converter):
        ir = {
            "model": "m",
            "state": [
                {
                    "type": "image",
                    "image_url": "data:image/png;base64,AA",
                    "detail": "low",
                }
            ],
            "questions": {},
        }
        wire, _ = converter.request_to_provider(ir)
        assert wire["input"][0]["content"][0]["detail"] == "low"
        back = converter.request_from_provider(wire)
        assert back["state"][0]["detail"] == "low"

    def test_malformed_mark_falls_back_to_plain(self, converter):
        req = {
            "model": "m",
            "input": "x",
            "questions": [
                {
                    "type": "predicate",
                    "name": "q",
                    "instructions": "<<<rosetta:assertion>>> is this urgent?",
                }
            ],
        }
        out = converter.request_from_provider(req)
        assert out["questions"]["q"]["instructions"] == (
            "<<<rosetta:assertion>>> is this urgent?"
        )
