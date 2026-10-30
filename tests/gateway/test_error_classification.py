"""The JSON envelope and the in-band terminal event agree on every cell.

A client that asks with ``stream: true`` must not get a different verdict from
one that does not.  The two envelopes are built by different modules from
different tables — ``error_response_for_source`` for the body,
``build_stream_error_events`` for the stream — and they had drifted: at 502 a
Responses stream said ``server_error`` where its JSON leg said ``api_error``.

The table in ``transport/error_class.py`` owns the words now; this test asks
both envelopes for every cell a caller can reach and compares what each
dialect is told.
"""

from __future__ import annotations

import json
from typing import Any, cast

import pytest

from llm_rosetta_gateway.proxy.errors import error_response_for_source
from llm_rosetta_gateway.transport.error_class import (
    CLIENT,
    GATEWAY,
    UPSTREAM,
    classify_status,
)
from llm_rosetta_gateway.transport.sse_format import build_stream_error_events

#: One status per cell, plus the neighbours the boundaries used to blur.
STATUSES = [400, 500, 501, 502, 503]


def _json(provider: str, status: int) -> dict[str, Any]:
    response = error_response_for_source(cast(Any, provider), status, "boom")
    body = response.body
    raw = body.decode() if isinstance(body, (bytes, bytearray)) else str(body)
    return json.loads(raw)


def _in_band(provider: str, status: int) -> dict[str, Any]:
    [event] = build_stream_error_events(provider, "boom", status=status)
    return event


@pytest.mark.parametrize("status", STATUSES)
def test_chat_streams_the_same_verdict_as_its_json_leg(status: int) -> None:
    body = _json("chat_completions", status)
    event = _in_band("chat_completions", status)
    assert event["error"]["type"] == body["error"]["type"]


@pytest.mark.parametrize("status", STATUSES)
def test_messages_streams_the_same_verdict_as_its_json_leg(status: int) -> None:
    body = _json("messages", status)
    event = _in_band("messages", status)
    assert event["error"]["type"] == body["error"]["type"]


@pytest.mark.parametrize("status", STATUSES)
def test_google_streams_the_same_verdict_as_its_json_leg(status: int) -> None:
    body = _json("google_generate", status)
    event = _in_band("google_generate", status)
    assert event["error"]["status"] == body["error"]["status"]


@pytest.mark.parametrize("status", STATUSES)
def test_responses_streams_the_same_verdict_as_its_json_leg(status: int) -> None:
    """The cell that had drifted: the in-band code used to be a fixed word."""
    body = _json("open_responses", status)
    event = _in_band("open_responses", status)
    assert event["response"]["error"]["code"] == body["error"]["type"]


def test_the_boundaries_are_the_table_s() -> None:
    assert classify_status(400) is CLIENT
    assert classify_status(500) is GATEWAY
    assert classify_status(501) is GATEWAY
    assert classify_status(502) is UPSTREAM
    assert classify_status(503) is UPSTREAM
