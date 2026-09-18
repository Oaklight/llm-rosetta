"""Tests for soft-error detection (200-but-error responses).

Some upstream providers (e.g. Argo) return HTTP 200 with an error message
embedded in the response body.  The gateway should detect these via
shim-configured patterns and convert them to proper HTTP error responses.
"""

import json

import asyncio

import pytest

from llm_rosetta.shims.provider_shim import SoftErrorPattern
from llm_rosetta.gateway.proxy import _check_soft_errors, _stream_event_generator


# ---------------------------------------------------------------------------
# SoftErrorPattern unit tests
# ---------------------------------------------------------------------------

ARGO_PATTERN = SoftErrorPattern(
    pattern="AUTHENTICATION NOTICE FROM ARGO",
    status_code=403,
    message="ARGO authentication error",
)


class TestSoftErrorPattern:
    def test_compiled_property(self):
        assert ARGO_PATTERN.compiled.search("⚠️ AUTHENTICATION NOTICE FROM ARGO ⚠️")

    def test_compiled_case_insensitive(self):
        assert ARGO_PATTERN.compiled.search("authentication notice from argo")

    def test_compiled_no_match(self):
        assert ARGO_PATTERN.compiled.search("normal response text") is None

    def test_frozen(self):
        with pytest.raises((AttributeError, TypeError)):
            ARGO_PATTERN.pattern = "new"  # type: ignore[misc]  # ty: ignore[invalid-assignment]


# ---------------------------------------------------------------------------
# _check_soft_errors unit tests
# ---------------------------------------------------------------------------

ARGO_WARNING_BODY = {
    "content": [
        {
            "type": "text",
            "text": (
                "⚠️ **IMPORTANT AUTHENTICATION NOTICE FROM ARGO** ⚠️\n\n"
                "🚫 **ACCESS DENIED** 🚫\n\n"
                "The username 'testuser' is not authorized."
            ),
        }
    ],
    "model": "claude-opus-4-6",
    "role": "assistant",
    "stop_reason": "end_turn",
    "type": "message",
}

NORMAL_BODY = {
    "content": [{"type": "text", "text": "Hello! How can I help?"}],
    "model": "claude-opus-4-6",
    "role": "assistant",
    "stop_reason": "end_turn",
    "type": "message",
}

PATTERNS = (ARGO_PATTERN,)


class TestCheckSoftErrors:
    def test_match(self):
        result = _check_soft_errors(PATTERNS, ARGO_WARNING_BODY)
        assert result is ARGO_PATTERN

    def test_no_match(self):
        result = _check_soft_errors(PATTERNS, NORMAL_BODY)
        assert result is None

    def test_empty_patterns(self):
        result = _check_soft_errors((), ARGO_WARNING_BODY)
        assert result is None

    def test_string_body(self):
        result = _check_soft_errors(PATTERNS, "AUTHENTICATION NOTICE FROM ARGO: denied")
        assert result is ARGO_PATTERN

    def test_multiple_patterns_first_match_wins(self):
        p1 = SoftErrorPattern(pattern="FIRST", status_code=400, message="first")
        p2 = SoftErrorPattern(pattern="SECOND", status_code=500, message="second")
        body = {"text": "contains FIRST and SECOND"}
        result = _check_soft_errors((p1, p2), body)
        assert result is p1


# ---------------------------------------------------------------------------
# Streaming path tests
# ---------------------------------------------------------------------------

# OpenAI Chat format: Argo warning appears in a content delta chunk
ARGO_STREAM_CHUNK = {
    "choices": [
        {
            "delta": {
                "content": (
                    "⚠️ **IMPORTANT AUTHENTICATION NOTICE FROM ARGO** ⚠️\nACCESS DENIED"
                )
            },
            "index": 0,
        }
    ],
    "model": "gpt-4o",
}

NORMAL_STREAM_CHUNK = {
    "choices": [
        {
            "delta": {"content": "Hello! How can I help?"},
            "index": 0,
        }
    ],
    "model": "gpt-4o",
}

DONE_CHUNK = {
    "choices": [
        {
            "delta": {},
            "index": 0,
            "finish_reason": "stop",
        }
    ],
    "model": "gpt-4o",
}


class _FakeStream:
    def __init__(self, chunks):
        self._chunks = chunks
        self.status_code = 200
        self.closed = False

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        self.closed = True
        return False

    async def __aiter__(self):
        for chunk in self._chunks:
            yield chunk


class _FakeProcessor:
    def process_chunk(self, chunk):
        if "error" in chunk:
            return []
        return [{"type": "content", "data": chunk}]


def _format_sse(event):
    return f"data: {json.dumps(event)}\n\n"


async def _collect_stream(
    chunks, soft_error_patterns=(), source_provider="openai_chat"
):
    out = []
    async for piece in _stream_event_generator(
        source_provider=source_provider,
        stream=_FakeStream(chunks),
        processor=_FakeProcessor(),
        model="test-model",
        format_sse=_format_sse,
        soft_error_patterns=soft_error_patterns,
    ):
        out.append(piece)
    return out


class TestStreamSoftError:
    def test_soft_error_in_stream_breaks(self):
        events = asyncio.run(
            _collect_stream(
                [ARGO_STREAM_CHUNK, NORMAL_STREAM_CHUNK],
                soft_error_patterns=PATTERNS,
            )
        )
        texts = "".join(events)
        assert "ARGO authentication error" in texts
        assert "Hello! How can I help?" not in texts

    def test_normal_stream_passes_through(self):
        events = asyncio.run(
            _collect_stream(
                [NORMAL_STREAM_CHUNK, DONE_CHUNK],
                soft_error_patterns=PATTERNS,
            )
        )
        texts = "".join(events)
        assert "Hello! How can I help?" in texts
        assert "ARGO authentication error" not in texts

    def test_no_patterns_passes_through(self):
        events = asyncio.run(
            _collect_stream(
                [ARGO_STREAM_CHUNK],
                soft_error_patterns=(),
            )
        )
        texts = "".join(events)
        assert "ARGO authentication error" not in texts


# ---------------------------------------------------------------------------
# Shim YAML loading test
# ---------------------------------------------------------------------------


class TestArgoShimLoading:
    @pytest.fixture(autouse=True)
    def _ensure_shims_loaded(self):
        from llm_rosetta.shims import get_shim
        from llm_rosetta.shims.providers import load_providers

        if get_shim("argo--anthropic") is None:
            load_providers()

    def test_argo_anthropic_shim_has_patterns(self):
        from llm_rosetta.shims import get_shim

        shim = get_shim("argo--anthropic")
        assert shim is not None
        assert len(shim.soft_error_patterns) > 0
        p = shim.soft_error_patterns[0]
        assert "AUTHENTICATION NOTICE FROM ARGO" in p.pattern
        assert p.status_code == 403

    def test_argo_openai_chat_shim_has_patterns(self):
        from llm_rosetta.shims import get_shim

        shim = get_shim("argo--openai_chat")
        assert shim is not None
        assert len(shim.soft_error_patterns) > 0
        p = shim.soft_error_patterns[0]
        assert "AUTHENTICATION NOTICE FROM ARGO" in p.pattern
        assert p.status_code == 403
