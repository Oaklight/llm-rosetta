"""Ordering of ``content`` and ``reasoning_content`` in one Chat chunk (#950).

Some Chat Completions providers put the tail of the reasoning and the first
answer token in the same stream delta.  The reader must emit the reasoning
event first, otherwise the leftover reasoning opens a second thinking block
*after* the text block and Anthropic clients render the answer in pieces.
"""

from typing import Any, cast

from llm_rosetta.converters.chat_completions.converter import ChatCompletionsConverter
from llm_rosetta.converters.messages.converter import MessagesConverter


def _chat_chunk(
    delta: dict[str, Any],
    *,
    finish_reason: str | None = None,
    index: int = 0,
) -> dict[str, Any]:
    """Build one upstream Chat Completions streaming chunk."""
    return {
        "id": "chatcmpl-dual",
        "object": "chat.completion.chunk",
        "model": "gpt-4",
        "created": 1700000000,
        "choices": [{"index": index, "delta": delta, "finish_reason": finish_reason}],
    }


def _chat_to_anthropic(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Run upstream Chat chunks through the Chat reader and the Anthropic writer."""
    chat = ChatCompletionsConverter()
    messages = MessagesConverter()
    from_ctx = chat.create_stream_context()
    to_ctx = messages.create_stream_context()

    out: list[dict[str, Any]] = []
    for chunk in chunks:
        for event in chat.stream_response_from_provider(chunk, context=from_ctx):
            result = messages.stream_response_to_provider(event, context=to_ctx)
            if isinstance(result, list):
                out.extend(e for e in result if e)
            elif result:
                out.append(result)
    return out


def _block_types(events: list[dict[str, Any]]) -> list[str]:
    return [
        e["content_block"]["type"] for e in events if e["type"] == "content_block_start"
    ]


def _joined(events: list[dict[str, Any]], delta_type: str) -> str:
    key = "thinking" if delta_type == "thinking_delta" else "text"
    return "".join(
        e["delta"][key]
        for e in events
        if e["type"] == "content_block_delta" and e["delta"]["type"] == delta_type
    )


class TestDualFieldChunkOrder:
    """The reader's per-chunk field order."""

    def setup_method(self):
        self.converter = ChatCompletionsConverter()

    def test_reasoning_emitted_before_content_in_same_chunk(self):
        """A chunk carrying both fields emits reasoning first."""
        chunk = _chat_chunk({"content": "基本", "reasoning_content": "。"})
        events = cast(list[Any], self.converter.stream_response_from_provider(chunk))
        assert [e["type"] for e in events] == ["reasoning_delta", "text_delta"]
        assert events[0]["reasoning"] == "。"
        assert events[1]["text"] == "基本"

    def test_content_only_chunk_unchanged(self):
        """A chunk with just content still emits a single text event."""
        chunk = _chat_chunk({"content": "hello"})
        events = cast(list[Any], self.converter.stream_response_from_provider(chunk))
        assert [e["type"] for e in events] == ["text_delta"]

    def test_tool_calls_still_after_reasoning_and_content(self):
        """Tool call deltas stay last: reasoning -> text -> tool call."""
        chunk = _chat_chunk(
            {
                "content": "checking",
                "reasoning_content": "use the tool",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": ""},
                    }
                ],
            }
        )
        events = cast(list[Any], self.converter.stream_response_from_provider(chunk))
        assert [e["type"] for e in events] == [
            "reasoning_delta",
            "text_delta",
            "tool_call_start",
        ]

    def test_empty_reasoning_content_still_reported(self):
        """An explicit empty string still yields a reasoning event (PR #393).

        DeepSeek emits an initial ``reasoning_content: ""`` and requires the
        field on assistant tool-call history, so the guard must stay
        ``is not None``.
        """
        chunk = _chat_chunk({"reasoning_content": ""})
        events = cast(list[Any], self.converter.stream_response_from_provider(chunk))
        assert [e["type"] for e in events] == ["reasoning_delta"]
        assert events[0]["reasoning"] == ""

    def test_empty_reasoning_precedes_content_in_same_chunk(self):
        """The empty reasoning event still comes first inside a dual-field chunk.

        This is DeepSeek's opening chunk: ``reasoning_content: ""`` riding along
        with the first answer token.
        """
        chunk = _chat_chunk({"content": "hi", "reasoning_content": ""})
        events = cast(list[Any], self.converter.stream_response_from_provider(chunk))
        assert [e["type"] for e in events] == ["reasoning_delta", "text_delta"]
        assert events[0]["reasoning"] == ""
        assert events[1]["text"] == "hi"


class TestDualFieldChunkToAnthropic:
    """Cross-format: the leftover reasoning must not land after the answer."""

    def test_dual_field_chunk_keeps_thinking_block_before_text(self):
        chunks = [
            _chat_chunk({"role": "assistant", "reasoning_content": "先想一下"}),
            # The reported shape: tail of the reasoning + first answer token.
            _chat_chunk({"content": "基本", "reasoning_content": "。"}),
            _chat_chunk({"content": "对，前半部分是 12。"}),
            _chat_chunk({}, finish_reason="stop"),
        ]
        events = _chat_to_anthropic(chunks)

        assert _block_types(events) == ["thinking", "text"]
        starts = [e for e in events if e["type"] == "content_block_start"]
        assert starts[0]["index"] == 0
        assert starts[1]["index"] == 1

        stops = [e for e in events if e["type"] == "content_block_stop"]
        assert stops[0]["index"] == 0

        # The trailing period joins the single thinking block, not a new one.
        assert _joined(events, "thinking_delta") == "先想一下。"
        assert _joined(events, "text_delta") == "基本对，前半部分是 12。"

    def test_empty_reasoning_produces_no_anthropic_block(self):
        """Empty reasoning events must not open a thinking block."""
        chunks = [
            _chat_chunk({"reasoning_content": ""}),
            _chat_chunk({"content": "hi"}),
            _chat_chunk({}, finish_reason="stop"),
        ]
        events = _chat_to_anthropic(chunks)
        assert _block_types(events) == ["text"]

    def test_empty_reasoning_and_content_in_one_chunk(self):
        """An empty reasoning event ahead of the text inside one chunk is dropped.

        DeepSeek opens with a chunk that carries ``reasoning_content: ""``
        together with the first answer token.  The writer must skip the empty
        reasoning event rather than open a thinking block before the text.
        """
        chunks = [
            _chat_chunk({"reasoning_content": ""}),
            _chat_chunk({"content": "hi", "reasoning_content": ""}),
            _chat_chunk({}, finish_reason="stop"),
        ]
        events = _chat_to_anthropic(chunks)
        assert _block_types(events) == ["text"]
        starts = [e for e in events if e["type"] == "content_block_start"]
        assert starts[0]["index"] == 0
        assert _joined(events, "text_delta") == "hi"

    def test_reasoning_resuming_in_a_later_chunk_still_splits(self):
        """Separate-chunk interleaving keeps the #250 block boundary.

        This is upstream interleaving we forward as-is: the text block closes
        and a new thinking block opens at the next index.
        """
        chunks = [
            _chat_chunk({"content": "answer part"}),
            _chat_chunk({"reasoning_content": "wait, reconsidering"}),
            _chat_chunk({"content": "rest"}),
            _chat_chunk({}, finish_reason="stop"),
        ]
        events = _chat_to_anthropic(chunks)
        assert _block_types(events) == ["text", "thinking", "text"]
