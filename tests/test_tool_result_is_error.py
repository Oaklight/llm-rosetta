"""Regression tests for ``tool_result.is_error`` across chat converters.

Anthropic Messages and Google Interactions carry ``is_error`` natively, and
Google Generate encodes it in the payload structure (``response.error``).
Chat Completions and Responses have no such field, so a ``true`` flag used to
disappear without a trace.  Those two now append a marker text block and
record a warning.

See https://github.com/Oaklight/llm-rosetta/issues/879.
"""

from typing import Any

import pytest

from llm_rosetta.converters.base.tools import TOOL_ERROR_MARKER
from llm_rosetta.pipeline import ConversionPipeline

TOOLS = [
    {
        "name": "deploy",
        "description": "deploy the service",
        "input_schema": {"type": "object", "properties": {"env": {"type": "string"}}},
    }
]


def _request(content: Any, is_error: bool | None = None) -> dict[str, Any]:
    tool_result: dict[str, Any] = {
        "type": "tool_result",
        "tool_use_id": "toolu_1",
        "content": content,
    }
    if is_error is not None:
        tool_result["is_error"] = is_error
    return {
        "model": "m",
        "max_tokens": 100,
        "tools": TOOLS,
        "messages": [
            {"role": "user", "content": "deploy to prod"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "deploy",
                        "input": {"env": "prod"},
                    }
                ],
            },
            {"role": "user", "content": [tool_result]},
        ],
    }


def _chat_payload(body: dict[str, Any]) -> Any:
    """Content of the converted ``role: "tool"`` message."""
    messages = ConversionPipeline("messages", "chat_completions").convert_request(body)[
        "messages"
    ]
    for message in messages:
        if message.get("role") == "tool":
            return message["content"]
    raise AssertionError(f"no tool message in {messages}")


def _responses_payload(body: dict[str, Any]) -> Any:
    """Output of the converted ``function_call_output`` item."""
    items = ConversionPipeline("messages", "open_responses").convert_request(body)[
        "input"
    ]
    for item in items:
        if item.get("type") == "function_call_output":
            return item["output"]
    raise AssertionError(f"no function_call_output in {items}")


# ---------------------------------------------------------------------------
# Chat Completions
# ---------------------------------------------------------------------------


def test_chat_true_flag_becomes_marker_block():
    payload = _chat_payload(_request("exit code 1", True))
    assert payload == [
        {"type": "text", "text": "exit code 1"},
        {"type": "text", "text": TOOL_ERROR_MARKER},
    ]


def test_chat_silent_failure_still_carries_the_flag():
    """The empty payload is exactly the case the flag exists for."""
    payload = _chat_payload(_request("", True))
    assert payload == [{"type": "text", "text": TOOL_ERROR_MARKER}]


@pytest.mark.parametrize("is_error", [False, None])
def test_chat_false_and_absent_flags_are_untouched(is_error: bool | None):
    assert _chat_payload(_request("exit code 0", is_error)) == "exit code 0"


def test_chat_lossy_conversion_is_reported():
    pipeline = ConversionPipeline("messages", "chat_completions")
    pipeline.convert_request(_request("exit code 1", True))
    assert len(pipeline.warnings) == 1
    assert "toolu_1" in pipeline.warnings[0]
    assert TOOL_ERROR_MARKER in pipeline.warnings[0]


def test_chat_clean_conversion_reports_nothing():
    pipeline = ConversionPipeline("messages", "chat_completions")
    pipeline.convert_request(_request("exit code 0", False))
    assert pipeline.warnings == []


# ---------------------------------------------------------------------------
# Responses
# ---------------------------------------------------------------------------


def test_responses_true_flag_becomes_marker_block():
    payload = _responses_payload(_request("exit code 1", True))
    assert payload == [
        {"type": "input_text", "text": "exit code 1"},
        {"type": "input_text", "text": TOOL_ERROR_MARKER},
    ]


def test_responses_silent_failure_still_carries_the_flag():
    payload = _responses_payload(_request("", True))
    assert payload == [{"type": "input_text", "text": TOOL_ERROR_MARKER}]


@pytest.mark.parametrize("is_error", [False, None])
def test_responses_false_and_absent_flags_are_untouched(is_error: bool | None):
    assert _responses_payload(_request("exit code 0", is_error)) == "exit code 0"


def test_responses_lossy_conversion_is_reported():
    pipeline = ConversionPipeline("messages", "open_responses")
    pipeline.convert_request(_request("exit code 1", True))
    assert len(pipeline.warnings) == 1
    assert "toolu_1" in pipeline.warnings[0]
    assert TOOL_ERROR_MARKER in pipeline.warnings[0]


# ---------------------------------------------------------------------------
# Formats that already carry the flag
# ---------------------------------------------------------------------------


def test_google_generate_encodes_the_flag_in_the_payload():
    contents = ConversionPipeline("messages", "google_generate").convert_request(
        _request("exit code 1", True)
    )["contents"]
    responses = [
        part["functionResponse"]["response"]
        for content in contents
        for part in content.get("parts", [])
        if "functionResponse" in part
    ]
    assert responses == [{"error": "exit code 1"}]


def test_google_interactions_keeps_the_flag():
    items = ConversionPipeline("messages", "google_interactions").convert_request(
        _request("exit code 1", True)
    )["input"]
    results = [item for item in items if item.get("type") == "function_result"]
    assert len(results) == 1
    assert results[0]["result"] == "exit code 1"
    assert results[0]["is_error"] is True


def test_chat_source_keeps_a_foreign_is_error_flag():
    """The converse direction: neither OpenAI source drops the flag.

    ``is_error`` is not a Chat field, but a foreign body may carry one, and
    the Responses reader already accepted it — so Chat reads it the same way
    instead of being the one silent leg.
    """
    body = {
        "model": "m",
        "messages": [
            {"role": "user", "content": "deploy to prod"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "deploy", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "exit code 1",
                "is_error": True,
            },
        ],
    }
    blocks = [
        part
        for message in ConversionPipeline(
            "chat_completions", "messages"
        ).convert_request(body)["messages"]
        for part in message.get("content", [])
        if part.get("type") == "tool_result"
    ]
    assert len(blocks) == 1
    assert blocks[0]["is_error"] is True


def _chat_body_with_is_error(value: object) -> dict[str, Any]:
    return {
        "model": "m",
        "messages": [
            {"role": "user", "content": "deploy to prod"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "deploy", "arguments": "{}"},
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call_1",
                "content": "exit code 1",
                "is_error": value,
            },
        ],
    }


def _responses_body_with_is_error(value: object) -> dict[str, Any]:
    return {
        "model": "m",
        "input": [
            {
                "type": "message",
                "role": "user",
                "content": [{"type": "input_text", "text": "deploy to prod"}],
            },
            {
                "type": "function_call",
                "call_id": "call_1",
                "name": "deploy",
                "arguments": "{}",
            },
            {
                "type": "function_call_output",
                "call_id": "call_1",
                "output": "exit code 1",
                "is_error": value,
            },
        ],
    }


# Both OpenAI sources read the field, so both have to survive a bad value.
_SOURCES = [
    ("chat_completions", _chat_body_with_is_error, "messages"),
    ("open_responses", _responses_body_with_is_error, "input"),
]


@pytest.mark.parametrize(("source", "build", "key"), _SOURCES)
@pytest.mark.parametrize("malformed", ["true", 1, {}, "False", []])
def test_malformed_foreign_flag_does_not_break_the_request(
    source: str, build: Any, key: str, malformed: object
):
    """A non-boolean is ignored rather than failing IR validation.

    Reading the field means it has to survive validation, and the IR types it
    as ``bool``.  Before the read, a malformed value was passed over; it still
    is, but now because it is checked rather than because it was invisible.
    """
    converted = ConversionPipeline(source, source).convert_request(build(malformed))
    tool_result = next(
        item
        for item in converted[key]
        if item.get("role") == "tool" or item.get("type") == "function_call_output"
    )
    payload = tool_result.get("content", tool_result.get("output"))
    assert payload == "exit code 1"


def test_intrinsic_result_has_nowhere_to_put_the_flag():
    """A native server item has no output payload, so the drop is reported.

    Chat Completions has no native server items, so the same IR does get the
    marker there — the two OpenAI targets disagree on this input, and the
    Responses side says so in ``warnings`` instead of staying quiet.
    """
    body = {
        "model": "m",
        "max_tokens": 100,
        "messages": [
            {"role": "user", "content": "search for X"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "server_tool_use",
                        "id": "srv_1",
                        "name": "web_search",
                        "input": {},
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "web_search_tool_result",
                        "tool_use_id": "srv_1",
                        "content": "search backend unreachable",
                        "is_error": True,
                    }
                ],
            },
        ],
    }

    pipeline = ConversionPipeline("messages", "open_responses")
    pipeline.convert_request(body)
    assert len(pipeline.warnings) == 1
    assert "srv_1" in pipeline.warnings[0]
    assert TOOL_ERROR_MARKER not in pipeline.warnings[0]

    # The flag is not lost between targets: Chat has no native server items,
    # so it renders the result as an ordinary tool message with the marker.
    chat = _chat_payload(body)
    assert chat[-1]["text"] == TOOL_ERROR_MARKER


def test_messages_keeps_the_flag():
    messages = ConversionPipeline("messages", "messages").convert_request(
        _request("exit code 1", True)
    )["messages"]
    blocks = [
        part
        for message in messages
        for part in message.get("content", [])
        if part.get("type") == "tool_result"
    ]
    assert len(blocks) == 1
    assert blocks[0]["is_error"] is True
    assert blocks[0]["content"] == "exit code 1"
