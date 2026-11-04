"""Tests for the vendored httpserver's chunk framing and lock."""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path

from llm_rosetta_gateway._vendor.httpserver import StreamingResponse

_ROOT = Path(__file__).resolve().parents[2]
_VENDOR = (
    _ROOT
    / "packages"
    / "llm-rosetta-gateway"
    / "src"
    / "llm_rosetta_gateway"
    / "_vendor"
)


class _Writer:
    """Record transport writes for a StreamingResponse."""

    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.data = bytearray()

    def write(self, data: bytes) -> None:
        self.writes.append(bytes(data))
        self.data.extend(data)

    async def drain(self) -> None:
        pass

    def is_closing(self) -> bool:
        return False


def _stream(*chunks: str) -> _Writer:
    async def gen():
        for chunk in chunks:
            yield chunk

    async def _run() -> _Writer:
        writer = _Writer()
        response = StreamingResponse(gen(), content_type="text/event-stream")
        await response._write(writer)  # ty: ignore[invalid-argument-type]
        return writer

    return asyncio.run(_run())


def test_small_chunks_are_framed_in_one_write():
    writer = _stream("a", "b")

    assert [w for w in writer.writes if w.startswith(b"1\r\n")] == [
        b"1\r\na\r\n",
        b"1\r\nb\r\n",
    ]
    # The size line and the trailing CRLF are not separate writes.
    assert b"\r\n" not in writer.writes
    assert writer.writes[-1] == b"0\r\n\r\n"


def test_an_empty_chunk_does_not_terminate_the_stream():
    writer = _stream("a", "", "b")

    assert bytes(writer.data).endswith(b"1\r\na\r\n1\r\nb\r\n0\r\n\r\n")
    assert bytes(writer.data).count(b"0\r\n\r\n") == 1


def test_the_coalesce_bound_is_inclusive():
    at_limit = _stream("x" * 65536)
    over = _stream("x" * 65537)

    assert at_limit.writes[1:-1] == [b"10000\r\n" + b"x" * 65536 + b"\r\n"]
    assert over.writes[1:-1] == [b"10001\r\n", b"x" * 65537, b"\r\n"]


def test_vendored_httpserver_matches_the_lock():
    lock = json.loads((_VENDOR / "zerodep.lock.json").read_text("utf-8"))

    assert lock["source"]["ref"] == "v2026.11.4"
    assert lock["modules"]["httpserver"]["version"] == "0.7.1"
    for filename, expected in lock["modules"]["httpserver"]["files"].items():
        actual = hashlib.sha256((_VENDOR / filename).read_bytes()).hexdigest()
        assert actual == expected
