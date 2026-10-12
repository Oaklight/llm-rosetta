"""Contract for ``ReasoningDeltaEvent`` carrying text and opaque values (#939).

``reasoning`` is a required field and ``signature`` / ``encrypted_content`` are
optional ones, so a single event may legitimately hold both.  Readers may
therefore emit combined events, and every writer must emit each field its wire
can express instead of silently dropping one.

The Google Generate reader currently splits a combined thought part into two
events (that is the shape Google's stream uses), so the combined case is
exercised here directly at the writer boundary.
"""

from typing import Any, cast

from llm_rosetta.converters.base.context import StreamContext
from llm_rosetta.converters.google_generate import GoogleGenerateConverter
from llm_rosetta.converters.google_interactions import GoogleInteractionsConverter
from llm_rosetta.converters.messages import MessagesConverter
from llm_rosetta.converters.open_responses import OpenAIResponsesConverter
from llm_rosetta.types.ir.stream import ReasoningDeltaEvent


def _combined(
    tag: str,
    *,
    reasoning: str = "visible reasoning",
    signature: str = "sig-1",
) -> ReasoningDeltaEvent:
    """A reasoning delta holding both visible text and a signature."""
    return cast(
        ReasoningDeltaEvent,
        {
            "type": "reasoning_delta",
            "reasoning": reasoning,
            "signature": signature,
            "provider_metadata": {"_opaque_origin": {"signature": tag}},
        },
    )


class TestMessagesCombinedReasoningDelta:
    """Anthropic carries thinking text and signature as separate deltas."""

    def setup_method(self):
        self.converter = MessagesConverter()

    def test_combined_event_emits_text_then_signature(self):
        result = cast(
            list[dict[str, Any]],
            self.converter.stream_response_to_provider(
                _combined("messages"), context=StreamContext()
            ),
        )
        deltas = [chunk["delta"] for chunk in result if chunk.get("delta")]
        assert [delta["type"] for delta in deltas] == [
            "thinking_delta",
            "signature_delta",
        ]
        assert deltas[0]["thinking"] == "visible reasoning"
        assert deltas[1]["signature"] == "sig-1"

    def test_combined_event_shares_one_block_index(self):
        ctx = StreamContext()
        ctx.mark_started()
        ctx.next_block_index()
        result = cast(
            list[dict[str, Any]],
            self.converter.stream_response_to_provider(
                _combined("messages"), context=ctx
            ),
        )
        deltas = [chunk for chunk in result if chunk.get("delta")]
        assert {chunk["index"] for chunk in deltas} == {0}

    def test_foreign_signature_keeps_text_only(self):
        result = cast(
            list[dict[str, Any]],
            self.converter.stream_response_to_provider(
                _combined("open_responses"), context=StreamContext()
            ),
        )
        deltas = [chunk["delta"] for chunk in result if chunk.get("delta")]
        assert [delta["type"] for delta in deltas] == ["thinking_delta"]
        assert deltas[0]["thinking"] == "visible reasoning"

    def test_foreign_signature_only_event_emits_nothing(self):
        event = cast(
            ReasoningDeltaEvent,
            {
                "type": "reasoning_delta",
                "reasoning": "",
                "signature": "sig-foreign",
                "provider_metadata": {
                    "_opaque_origin": {"signature": "open_responses"},
                },
            },
        )
        result = self.converter.stream_response_to_provider(
            event, context=StreamContext()
        )
        assert result == {}


class TestInteractionsCombinedReasoningDelta:
    """Interactions carries the summary and its signature as separate deltas."""

    def setup_method(self):
        self.converter = GoogleInteractionsConverter()

    def _deltas(self, tag: str) -> list[dict[str, Any]]:
        """Emitted ``step.delta`` payloads, ignoring the synthesised step.start."""
        result = cast(
            list[dict[str, Any]],
            self.converter.stream_response_to_provider(_combined(tag)),
        )
        return [chunk["delta"] for chunk in result if chunk.get("delta")]

    def test_combined_event_emits_summary_then_signature(self):
        deltas = self._deltas("google_interactions")
        assert [delta["type"] for delta in deltas] == [
            "thought_summary",
            "thought_signature",
        ]
        assert deltas[0]["text"] == "visible reasoning"
        assert deltas[1]["signature"] == "sig-1"

    def test_generate_signature_counts_as_own_family(self):
        deltas = self._deltas("google_generate")
        assert [delta["type"] for delta in deltas] == [
            "thought_summary",
            "thought_signature",
        ]

    def test_foreign_signature_keeps_summary_only(self):
        deltas = self._deltas("messages")
        assert [delta["type"] for delta in deltas] == ["thought_summary"]
        assert deltas[0]["text"] == "visible reasoning"


class TestResponsesCombinedReasoningDelta:
    """Responses keeps the text delta even when an opaque blob rides along."""

    def setup_method(self):
        self.converter = OpenAIResponsesConverter()

    def test_encrypted_content_does_not_swallow_the_text(self):
        event = cast(
            ReasoningDeltaEvent,
            {
                "type": "reasoning_delta",
                "reasoning": "visible reasoning",
                "encrypted_content": "enc-blob",
                "provider_metadata": {
                    "_opaque_origin": {"redacted_data": "open_responses"},
                },
            },
        )
        result = cast(
            list[dict[str, Any]],
            self.converter.stream_response_to_provider(event),
        )
        text_deltas = [
            chunk
            for chunk in result
            if chunk.get("type", "").endswith("reasoning_summary_text.delta")
        ]
        assert [chunk["delta"] for chunk in text_deltas] == ["visible reasoning"]


def _deltas(result: Any) -> list[dict[str, Any]]:
    """The ``delta`` payloads of a writer result, whether dict or list."""
    chunks = result if isinstance(result, list) else [result]
    return [
        chunk["delta"]
        for chunk in chunks
        if isinstance(chunk, dict) and chunk.get("delta")
    ]


class TestEmptySignatureIsNotASignature:
    """An empty signature is treated as absent by every writer."""

    def test_messages_drops_an_empty_signature_only_event(self):
        """No text and no real signature → no delta at all."""
        event = cast(
            ReasoningDeltaEvent,
            {
                "type": "reasoning_delta",
                "reasoning": "",
                "signature": "",
                "provider_metadata": {"_opaque_origin": {"signature": "messages"}},
            },
        )
        result = MessagesConverter().stream_response_to_provider(
            event, context=StreamContext()
        )
        assert _deltas(result) == []
        assert not result

    def test_messages_keeps_text_when_the_signature_is_empty(self):
        """An empty signature must not cost the visible reasoning text."""
        event = cast(
            ReasoningDeltaEvent,
            {
                "type": "reasoning_delta",
                "reasoning": "visible reasoning",
                "signature": "",
                "provider_metadata": {"_opaque_origin": {"signature": "messages"}},
            },
        )
        result = MessagesConverter().stream_response_to_provider(
            event, context=StreamContext()
        )
        assert [delta["type"] for delta in _deltas(result)] == ["thinking_delta"]

    def test_interactions_drops_an_empty_signature(self):
        """The Interactions writer already tested truthiness; pin it."""
        event = cast(
            ReasoningDeltaEvent,
            {
                "type": "reasoning_delta",
                "reasoning": "",
                "signature": "",
                "provider_metadata": {
                    "_opaque_origin": {"signature": "google_generate"}
                },
            },
        )
        result = GoogleInteractionsConverter().stream_response_to_provider(event)
        assert _deltas(result) == []


class TestGoogleGenerateStreamToInteractions:
    """The bug #939 warned about: a signature must not cost the visible text."""

    def setup_method(self):
        self.source = GoogleGenerateConverter()
        self.target = GoogleInteractionsConverter()

    def test_thinking_text_survives_alongside_a_captured_signature(self):
        chunks = [
            {
                "candidates": [
                    {
                        "index": 0,
                        "content": {
                            "role": "model",
                            "parts": [{"text": "let me think", "thought": True}],
                        },
                    }
                ]
            },
            {
                "candidates": [
                    {
                        "index": 0,
                        "content": {
                            "role": "model",
                            "parts": [{"text": "", "thoughtSignature": "sig-g"}],
                        },
                    }
                ]
            },
        ]
        deltas: list[dict[str, Any]] = []
        for chunk in chunks:
            for event in cast(
                list[Any], self.source.stream_response_from_provider(chunk)
            ):
                emitted = self.target.stream_response_to_provider(event)
                for piece in emitted if isinstance(emitted, list) else [emitted]:
                    if piece.get("delta"):
                        deltas.append(piece["delta"])

        assert [delta["type"] for delta in deltas] == [
            "thought_summary",
            "thought_signature",
        ]
        assert deltas[0]["text"] == "let me think"
        assert deltas[1]["signature"] == "sig-g"
