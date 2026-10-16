"""Provider-bound request fields belong to the format that recorded them.

``provider_extensions`` holds verbatim provider parameters -- the fields a
reader could not map into the IR, kept so the *same* format's writer can put
them back.  Merging such a set into another format's body puts a foreign
parameter on the wire.  Measured before the owner stamp existed, a Responses
request carrying ``store`` / ``previous_response_id`` / ``background`` reached
a Chat Completions body with all three, and the live upstreams reject that:
``400 unknown_parameter: Unknown parameter: 'previous_response_id'`` from Chat
Completions, ``400 invalid_request_error: background: Extra inputs are not
permitted`` from Messages.
"""

from __future__ import annotations

from typing import Any

import pytest

from llm_rosetta import ConversionPipeline
from llm_rosetta.converters.chat_completions import ChatCompletionsConverter
from llm_rosetta.types.ir.request import IRRequest

REQUEST: dict[str, Any] = {
    "model": "test-model",
    "input": "hello",
    "store": True,
    "previous_response_id": "resp_abc",
    "background": True,
    "metadata": {"k": "v"},
    "safety_identifier": "s1",
    "user": "u1",
}

#: The fields the Responses reader records as its own.
FOREIGN_KEYS = (
    "store",
    "previous_response_id",
    "background",
    "metadata",
    "safety_identifier",
    "user",
)


def _body(
    target: str, request: dict[str, Any] | None = None
) -> tuple[dict[str, Any], list[str]]:
    pipe = ConversionPipeline("open_responses", target)
    body = pipe.convert_request(dict(REQUEST if request is None else request))
    return body, list(pipe.warnings)


@pytest.mark.parametrize(
    ("target", "container"),
    [
        ("chat_completions", None),
        ("messages", None),
        ("google_generate", "config"),
        ("google_interactions", None),
    ],
)
def test_a_body_never_carries_another_formats_fields(
    target: str, container: str | None
) -> None:
    body, _ = _body(target)
    # A target that nests its request options (GenerateContent) may omit the
    # container entirely once nothing belongs in it.
    fields = body.get(container, {}) if container else body
    for key in FOREIGN_KEYS:
        assert key not in fields, f"{key} reached the {target} body: {fields}"


def test_the_format_that_recorded_them_still_gets_them() -> None:
    """The stamp confines a set, it does not discard it."""
    body, warnings = _body("open_responses")
    for key in FOREIGN_KEYS:
        assert key in body, f"{key} was lost on the same-format round trip"
    assert warnings == []


def test_the_loss_is_reported_and_names_the_consequence() -> None:
    _, warnings = _body("messages")
    assert len(warnings) == 1, warnings
    # ``previous_response_id`` is not a trimmed knob: the target answers from
    # this turn alone, and the warning has to say so.
    assert warnings[0] == (
        "provider-bound request fields dropped for messages: background, "
        "metadata, previous_response_id, safety_identifier, store, user; "
        "previous_response_id refers to a stored conversation that only the "
        "source upstream holds, so the target answers from this turn alone"
    )


def test_a_report_of_a_plain_drop_stays_plain() -> None:
    """A knob that merely cannot follow is not described as a lost turn."""
    request = {"model": "test-model", "input": "hello", "metadata": {"k": "v"}}
    pipe = ConversionPipeline("open_responses", "messages")
    body = pipe.convert_request(request)
    assert "metadata" not in body
    assert any("dropped for messages: metadata" in w for w in pipe.warnings)
    assert all("stored conversation" not in w for w in pipe.warnings)


def test_hand_built_ir_is_left_permissive() -> None:
    """IR no reader produced keeps merging, so direct callers are unaffected."""
    ir: IRRequest = {
        "model": "test-model",
        "messages": [],
        "provider_extensions": {"legacy_field": 1},
    }
    body, warnings = ChatCompletionsConverter().request_to_provider(ir)
    assert body["legacy_field"] == 1
    assert warnings == []


NEVER_READ: dict[str, Any] = {
    "model": "test-model",
    "input": "hello",
    "include": ["reasoning.encrypted_content"],
    "conversation": "conv_1",
    "context_management": [{"type": "compaction", "compact_threshold": 1000}],
}


def test_the_fields_no_reader_read_ride_same_format() -> None:
    """``include``, ``conversation`` and ``context_management`` had no reader."""
    body, warnings = _body("open_responses", NEVER_READ)
    for key, value in NEVER_READ.items():
        if key not in ("model", "input"):
            assert body[key] == value, f"{key} was lost on the same-format trip"
    assert warnings == []


def test_the_fields_no_reader_read_are_contained_too() -> None:
    body, warnings = _body("chat_completions", NEVER_READ)
    for key in ("include", "conversation", "context_management"):
        assert key not in body
    assert any("dropped for chat_completions" in w for w in warnings)
