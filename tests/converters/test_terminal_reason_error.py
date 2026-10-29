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


def test_google_ignores_another_dialects_record():
    """Only Google's own record explains a Google word."""
    ir = _ir(
        "error",
        metadata={
            "_native_finish_reason": {
                "base": "chat_completions",
                "reason": "content_filter",
            }
        },
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


def test_a_blocked_google_prompt_keeps_its_word():
    """A block reason the IR files under ``error`` is still Google's own.

    ``LANGUAGE`` maps to ``error``, so only the record the reader leaves keeps
    the turn writeable; without it a Google client would get a 502 for a
    prompt its own API blocked.
    """
    converter = GoogleGenerateConverter()
    provider_response = {
        "candidates": [],
        "promptFeedback": {"blockReason": "LANGUAGE"},
    }

    ir = converter.response_from_provider(cast(Any, provider_response))
    assert ir["choices"][0]["finish_reason"]["reason"] == "error"

    written_back = converter.response_to_provider(cast(Any, ir))

    assert written_back["candidates"][0]["finishReason"] == "LANGUAGE"


def test_a_blocked_google_prompt_keeps_its_word_on_the_stream():
    """The stream reader leaves the same record, for the same reason."""
    converter = GoogleGenerateConverter()
    chunk = {"candidates": [], "promptFeedback": {"blockReason": "LANGUAGE"}}

    events = converter.stream_response_from_provider(chunk)
    finish = next(e for e in events if e["type"] == "finish")
    assert finish["finish_reason"]["reason"] == "error"

    chunks = cast(list[dict[str, Any]], converter.stream_response_to_provider(finish))

    assert chunks[-1]["candidates"][0]["finishReason"] == "LANGUAGE"
