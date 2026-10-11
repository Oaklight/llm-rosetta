"""Tests for the TypeSafe decision converter.

Covers bidirectional conversion between TypeSafe wire format and IR.  All
three primitives unify on ``criteria: list[DecisionEntry]`` in the IR; the
converter maps them to TypeSafe's per-type criteria shapes.
"""

import pytest

from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.decision.typesafe import TypeSafeDecisionConverter

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
            "criteria": {"true": "Time-sensitive", "false": "No urgency"},
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

IR_REQUEST = {
    "model": "jev-latest",
    "state": "Help! My payouts have been failing for 3 days.",
    "questions": {
        "is_urgent": {
            "type": "assertion",
            "instructions": "Does this convey urgency?",
            "criteria": [
                {"label": False, "description": "No urgency"},
                {"label": True, "description": "Time-sensitive"},
            ],
        },
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": [
                {"label": "billing", "description": "Payments, invoicing, refunds"},
                {"label": "technical", "description": "Bugs, outages, integrations"},
            ],
        },
        "frustration": {
            "type": "score",
            "instructions": "How frustrated is the customer?",
            "criteria": [
                {"label": "Calm"},
                {"label": "Frustrated"},
                {"label": "Very angry"},
            ],
        },
    },
}

IR_RESPONSE = {
    "object": "decision",
    "model": "jev-1.13.0",
    "answers": {
        "is_urgent": {"type": "assertion", "probability": 0.92},
        "department": {
            "type": "choice",
            "choice": "technical",
            "probabilities": {"billing": 0.08, "technical": 0.85, "sales": 0.07},
            "confidence": 0.82,
        },
        "frustration": {
            "type": "score",
            "score": 1.6,
            "probabilities": {"Calm": 0.05, "Frustrated": 0.3, "Very angry": 0.65},
            "confidence": 0.78,
        },
    },
    "usage": {"input_tokens": 588, "output_tokens": 212},
}


@pytest.fixture
def converter():
    return TypeSafeDecisionConverter()


class TestRequestFromProvider:
    def test_assertion_criteria(self, converter):
        ir = converter.request_from_provider(TYPESAFE_REQUEST)
        q = ir["questions"]["is_urgent"]
        assert q["type"] == "assertion"
        assert {e["label"]: e["description"] for e in q["criteria"]} == {
            False: "No urgency",
            True: "Time-sensitive",
        }

    def test_choice_criteria(self, converter):
        ir = converter.request_from_provider(TYPESAFE_REQUEST)
        q = ir["questions"]["department"]
        assert q["type"] == "choice"
        assert {e["label"]: e["description"] for e in q["criteria"]} == {
            "billing": "Payments, invoicing, refunds",
            "technical": "Bugs, outages, integrations",
        }

    def test_score_criteria(self, converter):
        ir = converter.request_from_provider(TYPESAFE_REQUEST)
        q = ir["questions"]["frustration"]
        assert q["type"] == "score"
        assert [e["label"] for e in q["criteria"]] == [
            "Calm",
            "Frustrated",
            "Very angry",
        ]

    def test_state_and_model(self, converter):
        ir = converter.request_from_provider(TYPESAFE_REQUEST)
        assert ir["model"] == "jev-latest"
        assert ir["state"] == TYPESAFE_REQUEST["state"]


class TestRequestToProvider:
    def test_assertion_to_wire(self, converter):
        wire, warnings = converter.request_to_provider(IR_REQUEST)
        q = wire["questions"]["is_urgent"]
        assert q["type"] == "noul"
        assert q["criteria"] == {"true": "Time-sensitive", "false": "No urgency"}

    def test_choice_to_wire(self, converter):
        wire, _ = converter.request_to_provider(IR_REQUEST)
        assert wire["questions"]["department"]["criteria"] == {
            "billing": "Payments, invoicing, refunds",
            "technical": "Bugs, outages, integrations",
        }

    def test_score_to_wire(self, converter):
        wire, _ = converter.request_to_provider(IR_REQUEST)
        assert wire["questions"]["frustration"]["criteria"] == [
            "Calm",
            "Frustrated",
            "Very angry",
        ]

    def test_no_warnings(self, converter):
        _, warnings = converter.request_to_provider(IR_REQUEST)
        assert warnings == []


class TestResponseFromProvider:
    def test_assertion_answer(self, converter):
        ir = converter.response_from_provider(TYPESAFE_RESPONSE)
        a = ir["answers"]["is_urgent"]
        assert a == {"type": "assertion", "probability": 0.92}

    def test_choice_answer(self, converter):
        ir = converter.response_from_provider(TYPESAFE_RESPONSE)
        a = ir["answers"]["department"]
        assert a["choice"] == "technical"
        assert a["probabilities"]["technical"] == 0.85

    def test_score_answer_rekeyed_by_label(self, converter):
        ir = converter.response_from_provider(TYPESAFE_RESPONSE)
        a = ir["answers"]["frustration"]
        assert a["score"] == 1.6
        assert a["probabilities"] == {
            "Calm": 0.05,
            "Frustrated": 0.3,
            "Very angry": 0.65,
        }
        assert "legend" not in a

    def test_usage(self, converter):
        ir = converter.response_from_provider(TYPESAFE_RESPONSE)
        assert ir["usage"]["input_tokens"] == 588


class TestResponseToProvider:
    def test_assertion_answer(self, converter):
        wire = converter.response_to_provider(IR_RESPONSE)
        assert wire["answers"]["is_urgent"] == {"type": "noul", "noul": 0.92}

    def test_score_answer_rekeyed_by_index(self, converter):
        wire = converter.response_to_provider(IR_RESPONSE)
        a = wire["answers"]["frustration"]
        assert a["probabilities"] == {"0": 0.05, "1": 0.3, "2": 0.65}
        assert a["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}

    def test_no_object_field(self, converter):
        wire = converter.response_to_provider(IR_RESPONSE)
        assert "object" not in wire


class TestRoundTrip:
    def test_request_round_trip(self, converter):
        ir = converter.request_from_provider(TYPESAFE_REQUEST)
        wire, _ = converter.request_to_provider(ir)
        assert wire["questions"] == TYPESAFE_REQUEST["questions"]

    def test_response_round_trip(self, converter):
        ir = converter.response_from_provider(TYPESAFE_RESPONSE)
        wire = converter.response_to_provider(ir)
        assert wire["answers"] == TYPESAFE_RESPONSE["answers"]


class TestEdgeCases:
    def test_array_state_wrapped(self, converter):
        req = {
            "model": "jev-latest",
            "state": [{"a": 1}, {"b": 2}],
            "questions": {"q": {"type": "noul", "instructions": "x"}},
        }
        ir = converter.request_from_provider(req)
        assert ir["state"] == {"items": [{"a": 1}, {"b": 2}]}

    def test_image_in_state_warns_and_drops(self, converter):
        req = {
            "model": "jev-latest",
            "state": {
                "photo": {"type": "image", "image_url": "data:image/png;base64,AA"}
            },
            "questions": {"q": {"type": "noul", "instructions": "x"}},
        }
        ir = converter.request_from_provider(req)
        wire, warnings = converter.request_to_provider(ir)
        assert any("image" in w.lower() for w in warnings)

    def test_choice_unknown_probability_round_trip(self, converter):
        resp = {
            "model": "jev-1.13.0",
            "answers": {
                "q": {
                    "type": "choice",
                    "choice": "a",
                    "probabilities": {"a": 0.7, "b": 0.3},
                    "confidence": 0.5,
                    "unknown_probability": 0.2,
                }
            },
        }
        ir = converter.response_from_provider(resp)
        assert ir["answers"]["q"]["unknown_probability"] == 0.2
        wire = converter.response_to_provider(ir)
        assert wire["answers"]["q"]["unknown_probability"] == 0.2

    def test_unknown_answer_type_warns_and_passes_through(self, converter):
        from llm_rosetta.converters.base.context import ConversionContext

        ctx = ConversionContext()
        resp = {
            "model": "m",
            "answers": {"q": {"type": "distribution", "probabilities": {"a": 0.9}}},
        }
        ir = converter.response_from_provider(resp, context=ctx)
        assert ir["answers"]["q"] == {
            "type": "distribution",
            "probabilities": {"a": 0.9},
        }
        assert any("Unrecognized TypeSafe answer type" in w for w in ctx.warnings)

    def test_array_state_round_trip(self, converter):
        req = {"model": "m", "state": ["alpha", "beta"], "questions": {}}
        ir = converter.request_from_provider(req)
        assert ir["state"] == {"items": ["alpha", "beta"]}
        wire, _ = converter.request_to_provider(ir)
        assert wire["state"] == ["alpha", "beta"]

    def test_converter_tag(self, converter):
        assert converter._CONVERTER_TAG == "typesafe_decision"


class TestMultimodal:
    """System One family ``images[]`` extension (Clef / classifier.dev)."""

    def test_parts_to_state_and_images(self, converter):
        ir = {
            "model": "m",
            "state": [
                {"type": "text", "text": "look"},
                {
                    "type": "image",
                    "image_data": {"media_type": "image/png", "data": "AAAA"},
                },
            ],
            "questions": {},
        }
        wire, _ = converter.request_to_provider(ir)
        assert wire["state"] == "look"
        assert wire["images"] == ["data:image/png;base64,AAAA"]

    def test_state_and_images_to_parts(self, converter):
        req = {
            "model": "m",
            "state": "look",
            "images": ["data:image/png;base64,AAAA"],
            "questions": {},
        }
        ir = converter.request_from_provider(req)
        assert ir["state"] == [
            {"type": "text", "text": "look"},
            {"type": "image", "image_url": "data:image/png;base64,AAAA"},
        ]

    def test_round_trip(self, converter):
        req = {
            "model": "m",
            "state": "look",
            "images": ["data:image/png;base64,AAAA"],
            "questions": {},
        }
        ir = converter.request_from_provider(req)
        wire, _ = converter.request_to_provider(ir)
        assert wire["state"] == "look"
        assert wire["images"] == ["data:image/png;base64,AAAA"]

    def test_no_images_key_when_absent(self, converter):
        wire, _ = converter.request_to_provider(
            {"model": "m", "state": "x", "questions": {}}
        )
        assert "images" not in wire

    def test_image_only_state(self, converter):
        req = {
            "model": "m",
            "state": "",
            "images": ["data:image/png;base64,AAAA"],
            "questions": {},
        }
        ir = converter.request_from_provider(req)
        assert ir["state"] == [
            {"type": "image", "image_url": "data:image/png;base64,AAAA"}
        ]

    def test_dict_embedded_image_warns(self, converter):
        req = {
            "model": "m",
            "state": {
                "photo": {"type": "image", "image_url": "data:image/png;base64,AA"}
            },
            "questions": {},
        }
        ir = converter.request_from_provider(req)
        _, warnings = converter.request_to_provider(ir, context=ConversionContext())
        assert any("embedded in a structured state" in w for w in warnings)

    def test_images_only_without_state_key(self, converter):
        req = {"model": "m", "questions": {}, "images": ["data:image/png;base64,AA"]}
        ir = converter.request_from_provider(req)
        assert ir["state"] == [
            {"type": "image", "image_url": "data:image/png;base64,AA"}
        ]

    def test_images_bare_string_not_char_split(self, converter):
        req = {
            "model": "m",
            "state": "look",
            "images": "data:image/png;base64,AA",
            "questions": {},
        }
        ir = converter.request_from_provider(req)
        assert ir["state"] == [
            {"type": "text", "text": "look"},
            {"type": "image", "image_url": "data:image/png;base64,AA"},
        ]

    def test_image_without_data_warns_and_omitted(self, converter):
        ir = {"model": "m", "state": [{"type": "image"}], "questions": {}}
        wire, warnings = converter.request_to_provider(ir)
        assert wire["state"] == ""
        assert "images" not in wire
        assert any("neither image_url nor image_data" in w for w in warnings)

    def test_empty_list_state_with_images_has_no_bracket_text(self, converter):
        req = {
            "model": "m",
            "state": [],
            "images": ["data:image/png;base64,AA"],
            "questions": {},
        }
        ir = converter.request_from_provider(req)
        assert ir["state"] == [
            {"type": "image", "image_url": "data:image/png;base64,AA"}
        ]

    def test_items_wrap_extracts_images(self, converter):
        ir = {
            "model": "m",
            "state": {
                "items": [
                    {"type": "text", "text": "look"},
                    {"type": "image", "image_url": "data:image/png;base64,AA"},
                ]
            },
            "questions": {},
        }
        wire, warnings = converter.request_to_provider(ir)
        assert wire["state"] == "look"
        assert wire["images"] == ["data:image/png;base64,AA"]
        assert warnings == []
