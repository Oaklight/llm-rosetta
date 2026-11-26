"""Stream usage and finish handling across the five chat dialects (Hypothesis).

The rules, as the code states them:

* ``StreamContext.buffer_usage`` says usage events are *partial updates that
  add up* ("input tokens first, output tokens later"), so the usage a stream
  reads back is the sum of the usage events written into it;
* ``StreamContext.mark_choice_finished`` says the *first* finish of a choice is
  the one recorded;
* a stream has one start, one end, nothing after the end, and one finish per
  choice;
* a provider whose usage is a running total (Google sends ``usageMetadata`` on
  every chunk, cumulative) must read to its final total, not to the sum of the
  running totals.

Every test goes through the converter classes the registry binds
(``test_the_dialects_are_the_registry_s_own``).  Tests marked ``xfail``
(non-strict) record a place where a dialect disagrees with the rule; each says
which, and none is fixed here.
"""

from __future__ import annotations

from typing import Any

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from llm_rosetta.converters.registry import _CONVERTER_CLASSES  # noqa: E402
from tests.converters.test_leg_agreement import DIALECTS, _write_events  # noqa: E402

_SETTINGS = settings(max_examples=80, deadline=None, derandomize=True)
_NAMES = sorted(DIALECTS)

_ADDITIVE = [n for n in _NAMES if n != "google_interactions"]
_REASONS = ["stop", "length"]


def test_the_dialects_are_the_registry_s_own() -> None:
    for name, cls in DIALECTS.items():
        assert _CONVERTER_CLASSES["llm"][name] is cls, name


def _stream(
    usages: list[dict[str, int]], finishes: list[str] | None = None
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = [
        {"type": "stream_start", "response_id": "r", "model": "m", "created": 1},
        {"type": "text_delta", "text": "hi"},
    ]
    events += [{"type": "usage", "usage": dict(u)} for u in usages]
    events += [
        {"type": "finish", "finish_reason": {"reason": reason}}
        for reason in (finishes if finishes is not None else ["stop"])
    ]
    events.append({"type": "stream_end"})
    return events


def _round_trip(name: str, events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converter = DIALECTS[name]()
    chunks = _write_events(converter, events)
    return list(
        converter.iter_stream_from_provider(
            chunks, context=converter.create_stream_context()
        )
    )


def _summed(events: list[dict[str, Any]]) -> tuple[int, int, int]:
    total = [0, 0, 0]
    for event in events:
        if event["type"] == "usage":
            usage = event["usage"]
            for i, key in enumerate(
                ("prompt_tokens", "completion_tokens", "total_tokens")
            ):
                total[i] += usage.get(key) or 0
    return (total[0], total[1], total[2])


@st.composite
def _partial_usages(draw) -> list[dict[str, int]]:
    """One to three partial usage events; each carries total = prompt + completion."""
    out = []
    for _ in range(draw(st.integers(1, 3))):
        prompt = draw(st.integers(0, 500))
        completion = draw(st.integers(0, 500))
        out.append(
            {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "total_tokens": prompt + completion,
            }
        )
    return out


def _expected(usages: list[dict[str, int]]) -> tuple[int, int, int]:
    return (
        sum(u["prompt_tokens"] for u in usages),
        sum(u["completion_tokens"] for u in usages),
        sum(u["total_tokens"] for u in usages),
    )


# --- usage -------------------------------------------------------------------


@pytest.mark.parametrize("name", _ADDITIVE)
@_SETTINGS
@given(usages=_partial_usages())
def test_the_usage_a_stream_reads_back_is_the_sum_of_what_was_written(
    name, usages
) -> None:
    assert _summed(_round_trip(name, _stream(usages))) == _expected(usages)


@pytest.mark.xfail(
    strict=False,
    reason="google_interactions keeps the last usage event instead of adding "
    "them (#1750)",
)
@_SETTINGS
@given(usages=_partial_usages())
def test_google_interactions_adds_partial_usage_events(usages) -> None:
    assert _summed(_round_trip("google_interactions", _stream(usages))) == _expected(
        usages
    )


@pytest.mark.parametrize("name", _NAMES)
@_SETTINGS
@given(prompt=st.integers(1, 500), completion=st.integers(1, 500))
def test_one_usage_event_reads_back_whole(name, prompt, completion) -> None:
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
    }
    assert _summed(_round_trip(name, _stream([usage]))) == _expected([usage])


@pytest.mark.parametrize("name", _NAMES)
def test_a_stream_with_no_usage_reads_back_with_none(name) -> None:
    assert _summed(_round_trip(name, _stream([]))) == (0, 0, 0)


# --- the shape of a stream ---------------------------------------------------


@pytest.mark.parametrize("name", _NAMES)
@_SETTINGS
@given(
    usages=st.one_of(st.just([]), _partial_usages()), reason=st.sampled_from(_REASONS)
)
def test_a_stream_has_one_start_one_finish_and_ends_last(name, usages, reason) -> None:
    events = _round_trip(name, _stream(usages, [reason]))
    kinds = [e["type"] for e in events]
    assert kinds.count("stream_start") == 1
    assert kinds.count("stream_end") == 1
    assert kinds[-1] == "stream_end"
    finishes = [e for e in events if e["type"] == "finish"]
    assert [f["finish_reason"]["reason"] for f in finishes] == [reason]
    assert kinds.index("finish") < kinds.index("stream_end")
    # Usage never arrives after the end, whatever order the provider sends it in.
    assert "usage" not in kinds[kinds.index("stream_end") :]


# --- a second finish for the same choice -------------------------------------

_FIRST_WINS = ["chat_completions"]
_NOT_FIRST = [n for n in _NAMES if n not in _FIRST_WINS]


def _finish_reasons(name: str) -> list[str]:
    events = _round_trip(name, _stream([], ["stop", "length"]))
    return [e["finish_reason"]["reason"] for e in events if e["type"] == "finish"]


@pytest.mark.parametrize("name", _FIRST_WINS)
def test_a_second_finish_for_a_choice_is_ignored(name) -> None:
    assert _finish_reasons(name) == ["stop"]


@pytest.mark.xfail(
    strict=False,
    reason="only chat_completions keeps the first finish of a choice, as "
    "StreamContext.mark_choice_finished says; the others keep the last "
    "(google_generate emits both)",
)
@pytest.mark.parametrize("name", _NOT_FIRST)
def test_a_second_finish_for_a_choice_is_ignored_on_every_dialect(name) -> None:
    assert _finish_reasons(name) == ["stop"]


# --- a provider whose usage is a running total --------------------------------

# Captured from a live streamGenerateContent call (gemini-3.7-flash,
# 2026-11-26, maxOutputTokens 300): usageMetadata is on every chunk and the
# counts are cumulative; the last two chunks carry the same totals.
_GOOGLE_RUNNING = [
    (15, 17, 135),
    (15, 41, 159),
    (15, 65, 183),
    (15, 89, 207),
    (15, 110, 228),
    (15, 110, 228),
]


def _google_chunks() -> list[dict[str, Any]]:
    chunks = []
    for i, (prompt, completion, total) in enumerate(_GOOGLE_RUNNING):
        candidate: dict[str, Any] = {
            "content": {"role": "model", "parts": [{"text": f"{i} "}]}
        }
        if i == len(_GOOGLE_RUNNING) - 1:
            candidate["finishReason"] = "STOP"
        chunks.append(
            {
                "candidates": [candidate],
                "usageMetadata": {
                    "promptTokenCount": prompt,
                    "candidatesTokenCount": completion,
                    "totalTokenCount": total,
                },
                "modelVersion": "m",
            }
        )
    return chunks


def _google_to(name: str) -> tuple[int, int, int]:
    reader = DIALECTS["google_generate"]()
    events = list(
        reader.iter_stream_from_provider(
            _google_chunks(), context=reader.create_stream_context()
        )
    )
    return _summed(_round_trip(name, events))


@pytest.mark.xfail(
    strict=False,
    reason="google_generate reads each chunk's running total as a separate "
    "usage event and the target writers add them (#1749)",
)
@pytest.mark.parametrize(
    "target", ["chat_completions", "open_responses", "messages", "google_generate"]
)
def test_a_google_stream_reads_to_its_final_total(target) -> None:
    assert _google_to(target) == (15, 110, 228)


@pytest.mark.parametrize("name", ["messages"])
def test_a_messages_stream_adds_start_input_and_final_output(name) -> None:
    """message_start carries input tokens, message_delta the final output count."""
    reader = DIALECTS[name]()
    chunks = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "m",
                "content": [],
                "usage": {"input_tokens": 3, "output_tokens": 1},
            },
        },
        {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        },
        {
            "type": "content_block_delta",
            "index": 0,
            "delta": {"type": "text_delta", "text": "hi"},
        },
        {"type": "content_block_stop", "index": 0},
        {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 15},
        },
        {"type": "message_stop"},
    ]
    events = list(
        reader.iter_stream_from_provider(chunks, context=reader.create_stream_context())
    )
    assert _summed(events)[:2] == (3, 15)


def test_a_chat_stream_with_one_trailing_usage_chunk_reads_it_once() -> None:
    reader = DIALECTS["chat_completions"]()
    base = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m"}
    chunks = [
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "hi"},
                    "finish_reason": None,
                }
            ],
        },
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]},
        {
            **base,
            "choices": [],
            "usage": {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5},
        },
    ]
    events = list(
        reader.iter_stream_from_provider(chunks, context=reader.create_stream_context())
    )
    assert _summed(events) == (3, 2, 5)
    kinds = [e["type"] for e in events]
    assert kinds[-1] == "stream_end" and kinds.count("finish") == 1


# --- the buffer every writer shares ------------------------------------------

_KEYS = ["prompt_tokens", "completion_tokens", "reasoning_tokens", "cache_read_tokens"]


@_SETTINGS
@given(
    events=st.lists(
        st.dictionaries(
            st.sampled_from(_KEYS), st.integers(0, 100), min_size=1, max_size=4
        ),
        min_size=1,
        max_size=5,
    )
)
def test_buffered_usage_is_the_per_field_sum_of_the_events(events) -> None:
    from llm_rosetta.ir.context import StreamContext

    context = StreamContext()
    for event in events:
        context.buffer_usage(event)
    expected: dict[str, int] = {}
    for event in events:
        for key, value in event.items():
            expected[key] = expected.get(key, 0) + value
    assert context.pop_pending_usage() == expected
    assert context.pop_pending_usage() is None


def test_buffered_usage_takes_the_latest_non_numeric_value() -> None:
    from llm_rosetta.ir.context import StreamContext

    context = StreamContext()
    context.buffer_usage({"prompt_tokens": 1, "tier": "a"})
    context.buffer_usage({"prompt_tokens": 2, "tier": "b"})
    assert context.pop_pending_usage() == {"prompt_tokens": 3, "tier": "b"}
