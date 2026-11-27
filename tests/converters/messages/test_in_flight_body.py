"""A Messages body written for a response still in flight (#1739)."""

from typing import Any

from llm_rosetta.converters.llm.messages import MessagesConverter


def test_an_in_flight_response_writes_a_body_with_a_null_stop_reason() -> None:
    """The Messages body shape has ``stop_reason`` as a nullable required key."""
    ir: Any = {"id": "i", "object": "response", "created": 1, "model": "m"}
    ir["choices"] = []
    ir["in_flight"] = True
    body = MessagesConverter().response_to_provider(ir)
    assert body["content"] == []
    assert "stop_reason" in body and body["stop_reason"] is None


def test_an_empty_choice_list_without_the_flag_keeps_its_old_body() -> None:
    bare: Any = {"id": "c", "object": "response", "created": 1, "model": "m"}
    bare["choices"] = []
    assert "stop_reason" not in MessagesConverter().response_to_provider(bare)
