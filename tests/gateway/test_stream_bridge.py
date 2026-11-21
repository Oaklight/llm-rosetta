"""The stream-required bridge: a body rebuilt from a stream.

A shim declaring ``connection.stream_required`` makes the gateway stream
upstream and fold the events into the one body the client asked for.  The fold
is in the core (``tests/test_fold.py``); what this file pins is the part the
gateway owns — the response-level fields a dialect's chunks repeat and no IR
event carries, which have to be handed to the fold or the client gets less than
the same upstream's JSON body would have given it.

``system_fingerprint`` is the field that made this necessary: the Chat
Completions merger this bridge used to run copied it off the first chunk, and a
fold reading IR events has no way to see it.
"""

from __future__ import annotations

import asyncio
import json

from llm_rosetta.formats import BY_PAIR
from llm_rosetta.pipeline import ConversionPipeline
from llm_rosetta.contracts.routing import ResolvedRoute
from llm_rosetta_gateway.providers import build_provider_info
from llm_rosetta_gateway.proxy.engine import (
    _gather_echo_fields,
    _non_streaming_from_stream,
    _stream_envelope,
)

#: A one-word Chat Completions answer, with the two response-level echo fields
#: a real Chat stream repeats on every chunk.
_CHAT_STREAM: list[dict] = [
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1763000000,
        "model": "m",
        "service_tier": "default",
        "system_fingerprint": "fp_abc",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hello"}}],
    },
    {
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1763000000,
        "model": "m",
        "service_tier": "default",
        "system_fingerprint": "fp_abc",
        "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
    },
]


class _Stream:
    status_code = 200
    is_error = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return None

    async def close(self):
        return None

    async def read_error(self):
        return ""

    def __aiter__(self):
        async def gen():
            for chunk in _CHAT_STREAM:
                yield chunk

        return gen()


class _Transport:
    def __init__(self):
        self.streamed: list[bool] = []

    async def send(self, *args, **kwargs):
        raise AssertionError("a stream-only upstream is never asked for JSON")

    async def send_streaming(self, provider_info, url, body, **kwargs):
        provider_info.auth_headers()
        self.streamed.append(body.get("stream"))
        return _Stream()

    async def close(self):
        return None


def _chat_pipeline() -> ConversionPipeline:
    """A Chat Completions → Chat Completions pipeline, its request converted.

    A baseline (same-format passthrough) pipeline has no IR leg, so the test
    builds the converting one the gateway builds for a cross-dialect route.
    """
    pipeline = ConversionPipeline("chat_completions", "chat_completions")
    pipeline.convert_request(
        {"model": "m", "stream": False, "messages": [{"role": "user", "content": "hi"}]}
    )
    return pipeline


def test_the_catalog_names_the_echo_fields_a_chat_stream_repeats():
    """The fact lives in the format record, not in the gateway's code."""
    assert BY_PAIR[("llm", "chat_completions")].stream_echo_fields == (
        "service_tier",
        "system_fingerprint",
    )
    assert BY_PAIR[("llm", "open_responses")].stream_echo_fields == ()
    assert BY_PAIR[("llm", "messages")].stream_echo_fields == ()


def test_the_fold_and_the_catalog_name_the_same_echo_fields():
    """Two homes, one list: a field the catalog grows must reach the fold.

    The catalog declares which response-level fields a dialect repeats; the
    fold accepts one only if it knows the name.  Adding a field to the catalog
    and to the bridge's gather is therefore not enough — it would arrive and be
    dropped in silence.  This pins the two lists to each other.
    """
    from llm_rosetta.ir.primitives.fold import _ECHO_FIELDS

    declared = {field for spec in BY_PAIR.values() for field in spec.stream_echo_fields}

    assert declared == set(_ECHO_FIELDS)


def test_every_declared_echo_field_survives_the_fold():
    """Each catalog-declared field is one the fold actually keeps.

    A name in either list that the other lacks fails here with the field in
    the message, which is the difference between a silent drop and a red test.
    """
    from llm_rosetta.ir.primitives import fold_response_events

    declared = {field for spec in BY_PAIR.values() for field in spec.stream_echo_fields}

    for field in sorted(declared):
        folded = fold_response_events([], response={field: "sentinel"})
        assert folded.get(field) == "sentinel", field


def test_the_envelope_names_the_fields_before_the_stream_states_them():
    """It starts empty-of-value, so a field the stream never states stays absent."""
    route = ResolvedRoute(
        source_provider="chat_completions",
        target_provider="chat_completions",
        provider_name="local",
    )

    envelope = _stream_envelope(route)

    assert envelope == {"service_tier": None, "system_fingerprint": None}


def test_first_chunk_that_states_a_field_wins():
    """A field is taken once; a later chunk cannot overwrite what was settled."""
    envelope = {"system_fingerprint": None}

    _gather_echo_fields({"system_fingerprint": "fp_abc"}, envelope)
    _gather_echo_fields({"system_fingerprint": "fp_other"}, envelope)
    _gather_echo_fields({"model": "m"}, {"model": None})

    assert envelope == {"system_fingerprint": "fp_abc"}


def test_a_chat_client_keeps_the_echo_fields_the_stream_stated():
    """End to end: the Chat body carries what the same upstream's JSON would.

    Without the envelope the fold answers the same turn without
    ``system_fingerprint`` and ``service_tier`` — the regression the Chat-only
    merger did not have.
    """
    route = ResolvedRoute(
        source_provider="chat_completions",
        target_provider="chat_completions",
        provider_name="local",
    )
    transport = _Transport()

    response, profile = asyncio.run(
        _non_streaming_from_stream(
            route=route,
            provider_info=build_provider_info(
                "chat_completions",
                {"base_url": "http://upstream.invalid/v1", "api_key": "k"},
            ),
            pipeline=_chat_pipeline(),
            target_body={"model": "m", "input": "hi"},
            body={
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
            },
            model="m",
            transport=transport,
            metadata_store=None,
            extra_headers=None,
            persistence=None,
            capture_state=None,
            entry_id=None,
            profile={},
        )
    )

    assert transport.streamed == [True]
    assert response.status_code == 200
    body = json.loads(response.body)
    assert body["object"] == "chat.completion"
    assert body["choices"][0]["message"]["content"] == "hello"
    assert body["system_fingerprint"] == "fp_abc"
    assert body["service_tier"] == "default"
    assert profile["usage"]["prompt_tokens"] == 5


class _RefusingStream:
    """A stream-only upstream that refuses before the stream starts."""

    status_code = 400
    is_error = True

    def __init__(self, text: str) -> None:
        self._text = text

    async def read_error(self) -> str:
        return self._text

    async def close(self) -> None:
        return None


class _RefusingTransport:
    def __init__(self, text: str) -> None:
        self._text = text

    async def send(self, *args, **kwargs):
        raise AssertionError("a stream-only upstream is never asked for JSON")

    async def send_streaming(self, provider_info, url, body, **kwargs):
        return _RefusingStream(self._text)

    async def close(self) -> None:
        return None


def test_the_stream_bridge_reports_a_refusal_in_the_clients_terms():
    """A pre-stream 400 names the client's tool, not the upstream's path.

    The bridge forwards the request leg's name map to the error path; without
    it the client is pointed at ``tools[1].function.parameters``, a key its own
    Messages body does not have.
    """
    messages_body = {
        "model": "m",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hi"}],
        "tools": [
            {"name": "one", "input_schema": {"type": "object"}},
            {
                "name": "send",
                "input_schema": {
                    "type": "object",
                    "properties": {"count": {"type": "integer"}},
                },
            },
        ],
    }
    pipeline = ConversionPipeline("messages", "chat_completions")
    target_body = pipeline.convert_request(messages_body)
    route = ResolvedRoute(
        source_provider="messages",
        target_provider="chat_completions",
        provider_name="stub",
    )
    upstream_error = json.dumps(
        {
            "error": {
                "type": "invalid_request_error",
                "message": (
                    "Invalid 'tools[1].function.parameters.properties.count': "
                    "'1' is not of type 'integer'"
                ),
            }
        }
    )

    response, _ = asyncio.run(
        _non_streaming_from_stream(
            route=route,
            provider_info=build_provider_info(
                "chat_completions",
                {"base_url": "http://upstream.invalid/v1", "api_key": "k"},
            ),
            pipeline=pipeline,
            target_body=target_body,
            body=messages_body,
            model="m",
            transport=_RefusingTransport(upstream_error),
            metadata_store=None,
            extra_headers=None,
            persistence=None,
            capture_state=None,
            entry_id=None,
            profile={},
        )
    )

    assert response.status_code == 400
    message = json.loads(response.body)["error"]["message"]
    assert "the client's tool 'send'" in message
    assert "tools[1].input_schema.properties.count" in message
