"""Regression for #941: the internal ``_llm_rosetta_provider_metadata`` carrier must not
reach the wire.

Converters ferry provider-specific values (Google thought signatures, opaque
origins) across an in-process IR→provider→IR hop by attaching a
``_llm_rosetta_provider_metadata`` key to provider-format objects.  That key is not part of
any provider schema, so the top-level ``*_to_provider`` conversions strip it
before returning — otherwise a caller that sends the body directly gets HTTP
400 from Anthropic, OpenAI Responses and Google (Chat Completions silently
ignores the key, which is why a leak of this class can hide).

The ops-level ``ir_*_to_p`` / ``p_*_to_ir`` functions keep the carrier: they are
the in-process round-trip mechanism, covered by
``tests/converters/test_cross_provider_metadata_roundtrip.py``.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from llm_rosetta import get_converter_for_provider
from llm_rosetta.converters.base.helpers import CARRIER_KEY
from llm_rosetta.converters.chat_completions.tool_ops import ChatCompletionsToolOps
from llm_rosetta.converters.messages.tool_ops import MessagesToolOps
from llm_rosetta.types.ir import IRRequest, IRResponse, ToolCallPart

FORMATS = [
    "chat_completions",
    "open_responses",
    "messages",
    "google_generate",
    "google_interactions",
]


def _carrier_paths(obj: Any, prefix: str = "") -> list[str]:
    """Return a path string for every ``_llm_rosetta_provider_metadata`` key in ``obj``."""
    found: list[str] = []
    if isinstance(obj, dict):
        for key, value in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if key == "_llm_rosetta_provider_metadata":
                found.append(path)
            found.extend(_carrier_paths(value, path))
    elif isinstance(obj, list):
        for index, item in enumerate(obj):
            found.extend(_carrier_paths(item, f"{prefix}[{index}]"))
    return found


def _assert_wire_clean(body: Any, fmt: str) -> None:
    """Fail with the offending paths when a carrier survived into ``body``."""
    leaks = _carrier_paths(body)
    assert not leaks, (
        f"{fmt}: _llm_rosetta_provider_metadata leaked into the body at {leaks}"
    )


def _tool_call_part() -> ToolCallPart:
    """An assistant tool call carrying Google-origin provider metadata."""
    return cast(
        ToolCallPart,
        {
            "type": "tool_call",
            "tool_call_id": "call_1",
            "tool_name": "get_weather",
            "tool_type": "function",
            "tool_input": {"city": "Paris"},
            "provider_metadata": {
                "google": {"thought_signature": "SIG"},
                "responses_item_id": "fc_123",
                "namespace": "weather",
            },
        },
    )


def _ir_request() -> IRRequest:
    """A multi-turn tool-use request, the shape that triggers the carrier."""
    return cast(
        IRRequest,
        {
            "model": "gemini-2.0-flash",
            "messages": [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Weather in Paris?"}],
                },
                {"role": "assistant", "content": [_tool_call_part()]},
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_call_id": "call_1",
                            "result": "18C",
                            "is_error": False,
                            "provider_metadata": {"google": {"some_field": "value"}},
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "Get weather.",
                    "parameters": {
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                    },
                }
            ],
        },
    )


def _ir_response() -> IRResponse:
    """An assistant response with a tool call carrying provider metadata."""
    return cast(
        IRResponse,
        {
            "id": "resp_1",
            "object": "response",
            "created": 1700000000,
            "model": "gemini-2.0-flash",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": [_tool_call_part()],
                    },
                    "finish_reason": {"reason": "tool_calls"},
                }
            ],
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
        },
    )


@pytest.mark.parametrize("fmt", FORMATS)
def test_request_to_provider_is_wire_clean(fmt: str) -> None:
    """No ``_llm_rosetta_provider_metadata`` survives ``request_to_provider`` for any format."""
    body, _ = get_converter_for_provider(fmt).request_to_provider(_ir_request())
    _assert_wire_clean(body, fmt)


@pytest.mark.parametrize("fmt", FORMATS)
def test_response_to_provider_is_wire_clean(fmt: str) -> None:
    """No ``_llm_rosetta_provider_metadata`` survives ``response_to_provider`` for any format."""
    body = get_converter_for_provider(fmt).response_to_provider(_ir_response())
    _assert_wire_clean(body, fmt)


@pytest.mark.parametrize("fmt", FORMATS)
def test_messages_to_provider_is_wire_clean(fmt: str) -> None:
    """No ``_llm_rosetta_provider_metadata`` survives ``messages_to_provider`` for any format."""
    messages, _ = get_converter_for_provider(fmt).messages_to_provider(
        _ir_request()["messages"]
    )
    _assert_wire_clean(messages, fmt)


def test_google_wire_body_keeps_native_signature() -> None:
    """Stripping the carrier leaves the native wire field intact.

    Google carries the thought signature natively as ``thoughtSignature``; the
    carrier is redundant on the request and must go without touching it.
    """
    body, _ = get_converter_for_provider("google_generate").request_to_provider(
        _ir_request()
    )
    part = body["contents"][1]["parts"][0]
    assert part["thoughtSignature"] == "SIG"
    assert "_llm_rosetta_provider_metadata" not in part


def test_caller_owned_provider_metadata_key_survives() -> None:
    """A caller's own ``_provider_metadata`` key is not eaten by the strip.

    The carrier is namespaced, so a caller-supplied tool-schema property or
    tool-argument key that happens to be named ``_provider_metadata`` must
    survive.  Before the rename the unconditional recursive strip deleted it
    on some formats and not others — silent data loss.
    """
    ir = cast(
        IRRequest,
        {
            "model": "gemini-2.0-flash",
            "messages": [
                {"role": "user", "content": [{"type": "text", "text": "hi"}]},
                {
                    "role": "assistant",
                    "content": [
                        {
                            "type": "tool_call",
                            "tool_call_id": "call_1",
                            "tool_name": "f",
                            "tool_type": "function",
                            "tool_input": {"_provider_metadata": "caller-value"},
                            "provider_metadata": {
                                "google": {"thought_signature": "SIG"}
                            },
                        }
                    ],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "name": "f",
                    "parameters": {
                        "type": "object",
                        "properties": {"_provider_metadata": {"type": "string"}},
                        "required": ["_provider_metadata"],
                    },
                }
            ],
        },
    )
    for fmt in FORMATS:
        body, _ = get_converter_for_provider(fmt).request_to_provider(ir)
        rendered = json.dumps(body)
        assert CARRIER_KEY not in rendered, f"{fmt}: carrier leaked into the body"
        assert "_provider_metadata" in rendered, f"{fmt}: caller key was eaten"


def test_ops_level_still_emits_carrier() -> None:
    """The ops-level round-trip mechanism is unchanged by the boundary strip."""
    for ops in (MessagesToolOps, ChatCompletionsToolOps):
        block = ops.ir_tool_call_to_p(_tool_call_part())
        assert "_llm_rosetta_provider_metadata" in block
        assert (
            block["_llm_rosetta_provider_metadata"]["google"]["thought_signature"]
            == "SIG"
        )
