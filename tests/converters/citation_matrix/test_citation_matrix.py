"""The cross-format citation matrix, pinned.

Every dialect's citation vocabulary, its IR form, and what each target does
with it.  The cells are **measured** against the live endpoints, not assumed:

* ``url_citation`` is the one kind every dialect but Anthropic understands, and
  the four dialects that spell it (chat completions, Responses, Interactions,
  GenerateContent) convert freely among each other, streaming and not.
* Anthropic accepts exactly its own five kinds, and each has to point at a
  document or ``search_result`` block the message carries, or at an
  ``encrypted_index`` it issued — a ``text_citation`` or ``url_citation`` tag is
  rejected with a 400.  Its **reader** still exports the URL-bearing kinds, so
  an Anthropic citation can leave for any other dialect; nothing can come back.
* GenerateContent has no annotation at all: its citations live in the
  candidate's ``groundingMetadata`` and are rebuilt from the IR parts.
* All five dialects stream citations — chat completions carries them in
  ``choices[].delta.annotations`` (measured live), the two Responses surfaces in
  an annotation event, Interactions in a ``text_annotation_delta``, GenerateContent
  in the finish chunk's ``groundingMetadata``, and Anthropic in ``citations_delta``.

Spans are per source dialect, not synthesized: a citation read from a dialect
whose citations have no message-level span (Anthropic's document kinds) writes
none, rather than claiming ``0/0`` and reading as "the first characters".

A cell that changes here changes the contract, so the assertions name the kind
rather than only the boolean.
"""

from __future__ import annotations

from typing import Any, cast

import pytest

from llm_rosetta.types.ir import CitationPart
from llm_rosetta.converters.base.context import ConversionContext
from llm_rosetta.converters.base.helpers.citations import citation_url
from llm_rosetta.converters.chat_completions.converter import (
    ChatCompletionsConverter,
)
from llm_rosetta.converters.chat_completions.content_ops import (
    ChatCompletionsContentOps,
)
from llm_rosetta.converters.google_generate.converter import GoogleGenerateConverter
from llm_rosetta.converters.google_interactions.content_ops import (
    GoogleInteractionsContentOps,
)
from llm_rosetta.converters.messages.content_ops import MessagesContentOps
from llm_rosetta.converters.messages.converter import MessagesConverter
from llm_rosetta.converters.open_responses.content_ops import OpenResponsesContentOps

URL = "https://example.com/source"
SPAN = {"start_index": 5, "end_index": 25}

#: One citation as each dialect spells it, with the IR part it must read as.
NATIVE: dict[str, tuple[dict[str, Any], dict[str, Any]]] = {
    "chat": (
        {"type": "url_citation", "url_citation": {"url": URL, "title": "T", **SPAN}},
        {"citation_type": "url_citation", "url": URL, "title": "T", **SPAN},
    ),
    "open_responses": (
        {"type": "url_citation", "url": URL, "title": "T", **SPAN},
        {"citation_type": "url_citation", "url": URL, "title": "T", **SPAN},
    ),
    "google_interactions": (
        {"type": "url_citation", "url": URL, "title": "T", **SPAN},
        {"citation_type": "url_citation", "url": URL, "title": "T", **SPAN},
    ),
    "google_generate": (
        {
            "groundingChunks": [{"web": {"uri": URL, "title": "T"}}],
            "groundingSupports": [
                {
                    "segment": {"startIndex": 5, "endIndex": 25},
                    "groundingChunkIndices": [0],
                }
            ],
        },
        {"citation_type": "url_citation", "url": URL, "title": "T", **SPAN},
    ),
    "messages": (
        {
            "type": "search_result_location",
            "cited_text": "quoted",
            "source": URL,
            "title": "T",
            "search_result_index": 0,
            "start_block_index": 0,
            "end_block_index": 1,
        },
        {
            "citation_type": "search_result_location",
            "url": URL,
            "title": "T",
            "cited_text": "quoted",
            "provider_metadata": {
                "anthropic": {
                    "search_result_index": 0,
                    "start_block_index": 0,
                    "end_block_index": 1,
                }
            },
        },
    ),
}

#: Targets that can write a shared ``url_citation`` — every dialect but
#: Anthropic, whose request validator rejects that tag outright.
URL_TARGETS = ("chat", "open_responses", "google_interactions", "google_generate")


def read_native(dialect: str) -> Any:
    """Read a dialect's native citation into the IR part."""
    native = NATIVE[dialect][0]
    if dialect == "chat":
        return ChatCompletionsContentOps.p_citation_to_ir(native)
    if dialect == "open_responses":
        return OpenResponsesContentOps.p_citation_to_ir(native)
    if dialect == "google_interactions":
        return GoogleInteractionsContentOps.p_citation_to_ir(native)
    if dialect == "messages":
        return MessagesContentOps.p_citation_to_ir(native)
    ir = GoogleGenerateConverter().response_from_provider(
        {
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": "a"}]},
                    "groundingMetadata": native,
                }
            ]
        }
    )
    return ir["choices"][0]["message"]["content"][1]


def write_native(dialect: str, part: Any) -> Any:
    """Write an IR citation as a dialect's native form (``None`` = no form)."""
    if dialect == "chat":
        return ChatCompletionsContentOps.ir_citation_to_p(part)
    if dialect == "open_responses":
        return OpenResponsesContentOps.ir_citation_to_p(part)
    if dialect == "google_interactions":
        return GoogleInteractionsContentOps.ir_citation_to_p(part) or None
    if dialect == "messages":
        return MessagesContentOps.ir_citation_to_p(part)
    response = GoogleGenerateConverter().response_to_provider(
        cast(
            Any,
            {
                "id": "r",
                "object": "response",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": [part]},
                        "finish_reason": {"reason": "stop"},
                    }
                ],
            },
        )
    )
    return response["candidates"][0].get("groundingMetadata") or None


@pytest.mark.parametrize("dialect", sorted(NATIVE))
def test_each_dialect_reads_into_the_same_shared_shape(dialect: str) -> None:
    """Every dialect produces ``citation_type`` plus the shared fields."""
    expected = NATIVE[dialect][1]
    part = read_native(dialect)
    assert part["type"] == "citation"
    for key, value in expected.items():
        assert part[key] == value, key


@pytest.mark.parametrize("dialect", sorted(NATIVE))
@pytest.mark.parametrize("target", URL_TARGETS)
def test_every_dialect_reaches_the_url_annotation_targets(
    dialect: str, target: str
) -> None:
    """Every dialect's URL-bearing citation converts to the URL vocabulary.

    Including Anthropic's: its ``search_result_location`` names a source, so it
    becomes an annotation there — one way, since nothing converts back.
    """
    assert write_native(target, read_native(dialect)), f"{dialect} -> {target}"


def test_anthropic_has_no_form_for_a_foreign_citation() -> None:
    """Measured: its validator accepts its own five tags and nothing else.

    The writer returns ``None`` for every one of the other dialects' kinds, so
    the caller reports the loss rather than inventing a block to point at.
    """
    for dialect in URL_TARGETS:
        assert write_native("messages", read_native(dialect)) is None, dialect


def test_a_url_bearing_anthropic_citation_leaves_for_any_dialect() -> None:
    """The one direction that works: Anthropic's reader exports it."""
    part = read_native("messages")
    assert citation_url(part) == URL
    for target in URL_TARGETS:
        assert write_native(target, part), target


def test_messages_splits_citations_out_of_the_text_block() -> None:
    """Anthropic embeds them; the IR keeps them as their own parts."""
    text_block = {
        "type": "text",
        "text": "Answer",
        "citations": [NATIVE["messages"][0]],
    }
    converter = MessagesConverter()
    parts = converter.message_ops._p_content_part_to_ir(text_block)  # noqa: SLF001
    assert [p["type"] for p in parts] == ["text", "citation"]
    assert parts[1]["citation_type"] == "search_result_location"


def test_messages_hangs_a_citation_back_on_the_text_block() -> None:
    """The inverse: a citation is the ``citations`` array of the text before it."""
    response = MessagesConverter().response_to_provider(
        cast(
            Any,
            {
                "id": "r",
                "object": "response",
                "created": 0,
                "model": "m",
                "choices": [
                    {
                        "index": 0,
                        "message": {
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": "Answer"},
                                read_native("messages"),
                            ],
                        },
                        "finish_reason": {"reason": "stop"},
                    }
                ],
            },
        )
    )
    block = response["content"][0]
    assert block["type"] == "text"
    assert block["citations"] == [NATIVE["messages"][0]]


UNWRITABLE: dict[str, Any] = {
    "type": "citation",
    "citation_type": "char_location",
    "cited_text": "quoted",
    "title": "notes",
    "provider_metadata": {"anthropic": {"document_index": 0}},
}


def _chat_response(part: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": "r",
        "object": "response",
        "created": 0,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "Answer"}, part],
                },
                "finish_reason": {"reason": "stop"},
            }
        ],
    }


def test_a_dropped_citation_is_reported_by_the_writer() -> None:
    """A loss the target cannot avoid is not silent.

    Asserted through the response writer rather than by calling the reporting
    helper: the helper is not what can silently regress, the call site is — a
    mutation that removes the report there leaves this red.
    """
    converter = ChatCompletionsConverter()
    assert ChatCompletionsContentOps.ir_citation_to_p(cast(Any, UNWRITABLE)) is None

    context = ConversionContext()
    converter.response_to_provider(
        cast(Any, _chat_response(UNWRITABLE)), context=context
    )
    assert any("char_location" in w for w in context.warnings)

    sink: list[str] = []
    converter.response_to_provider(cast(Any, _chat_response(UNWRITABLE)), warnings=sink)
    assert any("char_location" in w for w in sink)


def test_the_write_pins_the_exact_native_spelling() -> None:
    """Values, not truthiness: a renamed kind or a leaked foreign tag fails."""
    spanned: dict[str, Any] = {
        "type": "citation",
        "citation_type": "url_citation",
        "url": URL,
        "title": "T",
        **SPAN,
    }
    assert write_native("chat", spanned) == {
        "type": "url_citation",
        "url_citation": {"title": "T", "url": URL, **SPAN},
    }
    assert write_native("open_responses", spanned) == {
        "type": "url_citation",
        "title": "T",
        "url": URL,
        **SPAN,
    }
    assert write_native("google_interactions", spanned) == {
        "type": "url_citation",
        "url": URL,
        "title": "T",
        **SPAN,
    }
    assert write_native("google_generate", spanned) == {
        "groundingChunks": [{"web": {"uri": URL, "title": "T"}}],
        "groundingSupports": [
            {
                "segment": {
                    "startIndex": SPAN["start_index"],
                    "endIndex": SPAN["end_index"],
                },
                "groundingChunkIndices": [0],
            }
        ],
    }


def test_a_spanless_citation_writes_no_span() -> None:
    """An Anthropic document citation has no message span, so none is invented.

    Writing ``0/0`` would say "the first characters" rather than "unknown".
    """
    spanless: dict[str, Any] = {
        "type": "citation",
        "citation_type": "search_result_location",
        "url": URL,
        "title": "T",
        "cited_text": "quoted",
    }
    assert write_native("chat", spanless) == {
        "type": "url_citation",
        "url_citation": {"title": "T", "url": URL},
    }
    assert write_native("open_responses", spanless) == {
        "type": "url_citation",
        "title": "T",
        "url": URL,
    }
    assert write_native("google_interactions", spanless) == {
        "type": "url_citation",
        "url": URL,
        "title": "T",
    }
    assert write_native("google_generate", spanless) == {
        "groundingChunks": [{"web": {"uri": URL, "title": "T"}}]
    }


def test_chat_completions_streams_a_citation_in_a_delta() -> None:
    """Measured live: ``choices[].delta.annotations``, nested like the message."""
    converter = ChatCompletionsConverter()
    context = converter.create_stream_context()
    native = NATIVE["chat"][0]
    events = (
        converter.stream_response_from_provider(
            {
                "choices": [
                    {
                        "index": 0,
                        "delta": {"annotations": [native]},
                        "finish_reason": None,
                    }
                ]
            },
            context,
        )
        or []
    )
    deltas = [e for e in events if e["type"] == "citation_delta"]
    assert len(deltas) == 1
    assert deltas[0]["citation"]["citation_type"] == "url_citation"

    written = cast(
        dict[str, Any], converter.stream_response_to_provider(deltas[0], context)
    )
    assert written["choices"][0]["delta"]["annotations"] == [native]


def test_google_generate_rebuilds_grounding_metadata() -> None:
    """GG -> IR -> GG keeps the sources, their spans and the search metadata."""
    metadata = {
        "groundingChunks": [
            {"web": {"uri": URL, "title": "T"}},
            {"web": {"uri": "https://b.example", "title": "B"}},
        ],
        "groundingSupports": [
            {
                "segment": {"startIndex": 5, "endIndex": 25},
                # Both sources are cited, so both are read as citations: a chunk
                # no support points at is not one (see the test below).
                "groundingChunkIndices": [0, 1],
            }
        ],
        "webSearchQueries": ["a query"],
        "searchEntryPoint": {"renderedContent": "<style>x</style>"},
    }
    converter = GoogleGenerateConverter()
    ir = converter.response_from_provider(
        {
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": "Answer"}]},
                    "finishReason": "STOP",
                    "groundingMetadata": metadata,
                }
            ]
        }
    )
    message = ir["choices"][0]["message"]
    # The search metadata is turn-level, so it is not repeated on the sources.
    assert message["provider_metadata"]["google"]["web_search_queries"] == ["a query"]
    for part in message["content"]:
        google = (part.get("provider_metadata") or {}).get("google") or {}
        assert "web_search_queries" not in google

    back = converter.response_to_provider(ir)
    restored = back["candidates"][0]["groundingMetadata"]
    assert restored["groundingChunks"] == metadata["groundingChunks"]
    assert restored["webSearchQueries"] == ["a query"]
    assert restored["searchEntryPoint"] == {"renderedContent": "<style>x</style>"}
    # One support that named two chunks comes back as one support per chunk:
    # same segment, same sources, and Gemini accepts both spellings.
    assert restored["groundingSupports"] == [
        {
            "segment": {"startIndex": 5, "endIndex": 25},
            "groundingChunkIndices": [0],
        },
        {
            "segment": {"startIndex": 5, "endIndex": 25},
            "groundingChunkIndices": [1],
        },
    ]


def test_streamed_citations_round_trip_as_their_own_event() -> None:
    """A native streamed citation becomes ``citation_delta`` and back."""
    converter = MessagesConverter()
    context = converter.create_stream_context()
    native = {
        "type": "web_search_result_location",
        "cited_text": "quoted",
        "url": URL,
        "title": "T",
        "encrypted_index": "opaque",
    }
    events: list[Any] = []
    for chunk in (
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": "", "citations": []},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "citations_delta", "citation": native},
        },
    ):
        events.extend(converter.stream_response_from_provider(chunk, context) or [])

    deltas = [e for e in events if e["type"] == "citation_delta"]
    assert len(deltas) == 1
    assert deltas[0]["citation"]["citation_type"] == "web_search_result_location"
    assert deltas[0]["block_index"] == 0

    written = cast(
        dict[str, Any], converter.stream_response_to_provider(deltas[0], context)
    )
    assert written["delta"] == {"type": "citations_delta", "citation": native}


def test_google_generate_streams_citations_as_events() -> None:
    """Google reports them on a chunk; the reader emits an event per source."""
    converter = GoogleGenerateConverter()
    context = converter.create_stream_context()
    chunk = {
        "candidates": [
            {
                "index": 0,
                "content": {"role": "model", "parts": []},
                "finishReason": "STOP",
                "groundingMetadata": NATIVE["google_generate"][0],
            }
        ]
    }
    events = converter.stream_response_from_provider(chunk, context) or []
    deltas = [e for e in events if e["type"] == "citation_delta"]
    assert [citation_url(e["citation"]) for e in deltas] == [URL]


@pytest.mark.parametrize(
    "ops",
    [ChatCompletionsContentOps, OpenResponsesContentOps, GoogleInteractionsContentOps],
)
def test_a_kind_a_dialect_does_not_know_is_never_guessed(ops: Any) -> None:
    """A kind outside the vocabulary yields no native form."""
    part: CitationPart = {
        "type": "citation",
        "citation_type": "char_location",
        "cited_text": "x",
    }
    assert ops.ir_citation_to_p(part) in ({}, None)


def test_interactions_and_responses_share_the_flat_shape() -> None:
    """Two dialects, one field set — the reason the pair converts losslessly."""
    native = NATIVE["open_responses"][0]
    assert GoogleInteractionsContentOps.p_citation_to_ir(
        native
    ) == OpenResponsesContentOps.p_citation_to_ir(native)
    assert GoogleInteractionsContentOps.ir_citation_to_p(cast(Any, native)) == native


def test_an_uncited_grounding_chunk_is_not_a_citation() -> None:
    """Google lists every source it read, but only cites some of them.

    A chunk no ``groundingSupport`` points at carries no span, so it is not read
    as a citation and does not come back on a write — the same rule the reader
    has followed since it was written.
    """
    converter = GoogleGenerateConverter()
    ir = converter.response_from_provider(
        {
            "candidates": [
                {
                    "index": 0,
                    "content": {"role": "model", "parts": [{"text": "Answer"}]},
                    "finishReason": "STOP",
                    "groundingMetadata": {
                        "groundingChunks": [
                            {"web": {"uri": URL, "title": "T"}},
                            {"web": {"uri": "https://b.example", "title": "B"}},
                        ],
                        "groundingSupports": [
                            {
                                "segment": {"startIndex": 5, "endIndex": 25},
                                "groundingChunkIndices": [0],
                            }
                        ],
                    },
                }
            ]
        }
    )
    citations = [
        p for p in ir["choices"][0]["message"]["content"] if p["type"] == "citation"
    ]
    assert [citation_url(c) for c in citations] == [URL]
