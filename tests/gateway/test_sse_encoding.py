"""The SSE formatters emit exactly what ``json.dumps(..., ensure_ascii=False)`` does.

The formatters share one :class:`json.JSONEncoder` instead of calling
``json.dumps`` per event (#1410).  That is only safe while the options match,
so this pins the bytes: a payload that would differ (non-ASCII text, floats,
``None``, nested containers) must come out identical.
"""

from __future__ import annotations

import json

import pytest

from llm_rosetta_gateway.transport.sse_format import SSE_FORMATTERS

PAYLOADS = [
    {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta"}},
    {"type": "text", "text": 'héllo — 中文 😀\n\t"quoted"', "n": None},
    {"type": "usage", "values": [1, 2.5, -0.0, 1e-7, 2**60], "ok": True},
    {"type": "nested", "a": {"b": [{"c": "ü"}]}},
]


@pytest.mark.parametrize("format_name", sorted(SSE_FORMATTERS))
@pytest.mark.parametrize("payload", PAYLOADS)
def test_sse_payload_matches_json_dumps(format_name: str, payload: dict) -> None:
    text = SSE_FORMATTERS[format_name](payload)
    expected = json.dumps(payload, ensure_ascii=False)
    assert expected in text
    # Nothing else may appear between the prefix and the encoded body.
    data_line = next(line for line in text.splitlines() if line.startswith("data: "))
    assert data_line == f"data: {expected}"
