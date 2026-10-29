"""A terminal turn the target dialect has no finish reason for.

Chat Completions cannot say "the generation failed" or "it was cancelled", the
Messages API's vocabulary only describes finished turns, and Google's catch-all
is a reason its own model produced.  None of the three has a form for a failed
upstream, so the writer raises instead of telling the client the turn finished.
The payload — the reason, the response id, the upstream metadata — rides on the
exception as attributes so a caller dumps it structurally (#1234).
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from llm_rosetta.converters.base.errors import UnrepresentableFinishReason
from llm_rosetta.converters.llm.chat_completions import ChatCompletionsConverter
from llm_rosetta.converters.llm.google_generate import GoogleGenerateConverter
from llm_rosetta.converters.llm.messages import MessagesConverter

_WRITERS = [
    pytest.param(ChatCompletionsConverter, id="chat_completions"),
    pytest.param(MessagesConverter, id="messages"),
    pytest.param(GoogleGenerateConverter, id="google_generate"),
]


def _ir(reason: str, *, metadata: dict[str, Any] | None = None) -> Any:
    message: dict[str, Any] = {
        "role": "assistant",
        "content": [{"type": "text", "text": ""}],
    }
    if metadata is not None:
        message["provider_metadata"] = metadata
    return {
        "id": "r1",
        "object": "response",
        "created": 0,
        "model": "m",
        "choices": [
            {"index": 0, "message": message, "finish_reason": {"reason": reason}}
        ],
    }


@pytest.mark.parametrize("converter_class", _WRITERS)
@pytest.mark.parametrize("reason", ["error", "cancelled"])
def test_a_terminal_failure_is_a_typed_error(converter_class, reason):
    """The payload rides on the exception, not inside its message string."""
    ir = _ir(reason, metadata={"upstream": "boom"})

    with pytest.raises(UnrepresentableFinishReason) as raised:
        converter_class().response_to_provider(ir)

    assert raised.value.reason == reason
    assert raised.value.response_id == "r1"
    assert raised.value.provider_metadata == {"upstream": "boom"}
    assert "cannot represent" in str(raised.value)
    assert "boom" not in str(raised.value)


def test_the_dialect_is_named_in_the_message():
    """An operator reading a gateway log learns which side refused."""
    with pytest.raises(UnrepresentableFinishReason) as raised:
        MessagesConverter().response_to_provider(_ir("error"))

    assert "Messages API" in str(raised.value)


def test_google_writes_its_own_word_back():
    """Google's odd stop is not a foreign failure: the reader kept the word."""
    ir = _ir(
        "error",
        metadata={
            "_native_finish_reason": {
                "base": "google_generate",
                "reason": "MALFORMED_FUNCTION_CALL",
            }
        },
    )

    result = GoogleGenerateConverter().response_to_provider(ir)

    assert result["candidates"][0]["finishReason"] == "MALFORMED_FUNCTION_CALL"


@pytest.mark.parametrize("base", ["chat_completions", "open_responses", "messages"])
def test_google_ignores_another_dialects_record(base):
    """Only Google's own record explains a Google word.

    The Google converter is the only writer of ``_native_finish_reason``
    today, so a foreign failure reaches the writer with no record at all — the
    refusal comes from the record's absence.  This is the guard for a record
    another reader might write later, and it is what makes the origin tag on
    the existing one mean something.
    """
    ir = _ir(
        "error",
        metadata={"_native_finish_reason": {"base": base, "reason": "SAFETY"}},
    )

    with pytest.raises(UnrepresentableFinishReason):
        GoogleGenerateConverter().response_to_provider(ir)


def test_a_google_upstream_round_trips_through_the_ir():
    """Reader → writer: the word Google sent is the word Google gets back."""
    converter = GoogleGenerateConverter()
    provider_response: Any = {
        "candidates": [
            {
                "index": 0,
                "content": {"role": "model", "parts": [{"text": "partial"}]},
                "finishReason": "OTHER",
            }
        ],
        "responseId": "g1",
        "modelVersion": "gemini-2.5-flash",
    }

    ir = converter.response_from_provider(provider_response)
    assert ir["choices"][0]["finish_reason"]["reason"] == "error"

    written_back = converter.response_to_provider(cast(Any, ir))

    assert written_back["candidates"][0]["finishReason"] == "OTHER"


def test_a_blocked_google_prompt_keeps_its_prompt_feedback():
    """A blocked prompt has no candidate; the field itself is the answer.

    ``LANGUAGE`` maps to the IR's ``error`` and there is no candidate to carry
    a ``finishReason``, so only the record the reader leaves keeps the turn
    writeable — without it a Google client would get a 502 for a prompt its own
    API blocked, and with a candidate written instead it would get a turn the
    model never made.
    """
    converter = GoogleGenerateConverter()
    provider_response = {
        "candidates": [],
        "promptFeedback": {"blockReason": "LANGUAGE"},
    }

    ir = converter.response_from_provider(cast(Any, provider_response))
    assert ir["choices"][0]["finish_reason"]["reason"] == "error"

    written_back = converter.response_to_provider(cast(Any, ir))

    assert written_back["candidates"] == []
    assert written_back["promptFeedback"] == {"blockReason": "LANGUAGE"}


def test_a_blocked_google_prompt_keeps_its_feedback_on_the_stream():
    """The stream reader leaves the same record, for the same reason."""
    converter = GoogleGenerateConverter()
    chunk = {"candidates": [], "promptFeedback": {"blockReason": "LANGUAGE"}}

    events = converter.stream_response_from_provider(chunk)
    finish = next(e for e in events if e["type"] == "finish")
    assert finish["finish_reason"]["reason"] == "error"

    chunks = cast(list[dict[str, Any]], converter.stream_response_to_provider(finish))

    assert "candidates" not in chunks[-1]
    assert chunks[-1]["promptFeedback"] == {"blockReason": "LANGUAGE"}


@pytest.mark.parametrize(
    "word", ["RECITATION", "BLOCKLIST", "PROHIBITED_CONTENT", "SPII"]
)
def test_a_google_filter_word_round_trips(word):
    """The filtered family keeps Google's word instead of collapsing to SAFETY."""
    converter = GoogleGenerateConverter()
    provider_response = {
        "candidates": [
            {
                "index": 0,
                "content": {"role": "model", "parts": [{"text": "partial"}]},
                "finishReason": word,
            }
        ],
        "responseId": "g1",
        "modelVersion": "gemini-2.5-flash",
    }

    ir = converter.response_from_provider(cast(Any, provider_response))
    assert ir["choices"][0]["finish_reason"]["reason"] == "content_filter"

    written_back = converter.response_to_provider(cast(Any, ir))

    assert written_back["candidates"][0]["finishReason"] == word


def test_a_google_word_the_mapping_preserves_is_not_recorded():
    """Only a word the IR would lose is carried; STOP needs no record."""
    converter = GoogleGenerateConverter()
    provider_response = {
        "candidates": [
            {
                "index": 0,
                "content": {"role": "model", "parts": [{"text": "done"}]},
                "finishReason": "STOP",
            }
        ],
        "responseId": "g1",
        "modelVersion": "gemini-2.5-flash",
    }

    ir = converter.response_from_provider(cast(Any, provider_response))

    message = cast(dict[str, Any], ir["choices"][0]["message"])
    assert "_native_finish_reason" not in (message.get("provider_metadata") or {})


def test_a_google_filter_word_survives_the_stream():
    """Same record on the stream leg: the finish event carries it back."""
    converter = GoogleGenerateConverter()
    chunk = {
        "candidates": [
            {
                "index": 0,
                "content": {"role": "model", "parts": [{"text": "partial"}]},
                "finishReason": "RECITATION",
            }
        ]
    }

    events = converter.stream_response_from_provider(chunk)
    finish = next(e for e in events if e["type"] == "finish")
    assert finish["finish_reason"]["reason"] == "content_filter"

    chunks = cast(list[dict[str, Any]], converter.stream_response_to_provider(finish))

    assert chunks[-1]["candidates"][0]["finishReason"] == "RECITATION"


@pytest.mark.parametrize("word", ["MALFORMED_FUNCTION_CALL", "LANGUAGE", "OTHER"])
def test_a_google_failure_word_survives_the_stream(word):
    """The stream reader records the word too, not just the JSON one.

    Without it the failure reaches a Google target with no record, and a Google
    client is refused a word its own API said.
    """
    converter = GoogleGenerateConverter()
    chunk = {
        "candidates": [
            {
                "index": 0,
                "content": {"role": "model", "parts": [{"text": "partial"}]},
                "finishReason": word,
            }
        ]
    }

    events = converter.stream_response_from_provider(chunk)
    finish = next(e for e in events if e["type"] == "finish")
    assert finish["finish_reason"]["reason"] == "error"

    chunks = cast(list[dict[str, Any]], converter.stream_response_to_provider(finish))

    assert chunks[-1]["candidates"][0]["finishReason"] == word


def test_google_ignores_another_dialects_prompt_feedback():
    """Only Google's own record lets the writer omit a candidate.

    The guardian of the blocked-prompt shape: a record another dialect wrote
    must not make Google answer with ``promptFeedback`` and no candidate for a
    turn it produced normally.
    """
    ir = _ir(
        "content_filter",
        metadata={
            "_native_prompt_feedback": {
                "base": "chat_completions",
                "prompt_feedback": {"blockReason": "SAFETY"},
            }
        },
    )

    result = GoogleGenerateConverter().response_to_provider(ir)

    assert "promptFeedback" not in result
    assert result["candidates"][0]["finishReason"] == "SAFETY"
