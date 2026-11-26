"""The usage contract over generated streams (Hypothesis): increments that add.

The contract (``UsageEvent``, ``StreamContext.usage_increment``,
``add_usage``): the usage events of a stream are *increments*; a consumer adds
them; a reader of a wire that repeats a running total emits the difference, so
the sum of its events is the highest snapshot, field by field.  The first
snapshot passes through whole (zeros included); after it a field is a
high-water mark, so a zero placeholder or a decrease resets nothing.

``test_usage_live_streams.py`` checks five real captures.  The laws here run
the same contract over generated snapshot sequences, against a reference
written in this file (a per-field maximum), on each consumer and reader that
the contract changed: ``StreamContext.usage_increment``, ``add_usage``, the
Google, Chat and Messages stream readers, the fold and the gateway record
(``StreamProcessor._merge_usage``).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytest.importorskip("hypothesis")

from hypothesis import given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from llm_rosetta.converters.registry import _CONVERTER_CLASSES  # noqa: E402
from llm_rosetta.ir import context as context_module  # noqa: E402
from llm_rosetta.ir.context import StreamContext, add_usage  # noqa: E402
from llm_rosetta.ir.primitives import fold_response_events  # noqa: E402
from llm_rosetta.ir.primitives import fold as fold_module  # noqa: E402
from llm_rosetta.pipeline.chat import streaming as streaming_module  # noqa: E402

_SETTINGS = settings(max_examples=150, deadline=None, derandomize=True)
_LLM = _CONVERTER_CLASSES["llm"]
_FIELDS = ("prompt_tokens", "completion_tokens", "total_tokens", "reasoning_tokens")


def test_the_targets_are_the_product_s_own() -> None:
    assert fold_module.add_usage is context_module.add_usage
    assert streaming_module.add_usage is context_module.add_usage
    assert StreamContext is context_module.StreamContext


# --- generated snapshot sequences -------------------------------------------


@st.composite
def _snapshots(draw, *, keys: tuple[str, ...] = _FIELDS) -> list[dict[str, int]]:
    """A running report: mostly rising, with zero placeholders and decreases.

    The first snapshot states every key (a wire states its fields at the
    start); later ones may leave a key out, repeat it, raise it, drop it to a
    lower value or to zero.
    """
    first = {k: draw(st.integers(0, 50)) for k in keys}
    out = [first]
    current = dict(first)
    for _ in range(draw(st.integers(0, 6))):
        snap: dict[str, int] = {}
        for key in keys:
            move = draw(
                st.sampled_from(["omit", "same", "rise", "rise", "drop", "zero"])
            )
            if move == "omit":
                continue
            if move == "same":
                snap[key] = current[key]
            elif move == "rise":
                snap[key] = current[key] + draw(st.integers(0, 40))
            elif move == "drop":
                snap[key] = draw(st.integers(0, max(current[key], 0)))
            else:
                snap[key] = 0
        out.append(snap)
        for key, value in snap.items():
            current[key] = max(current[key], value)
    return out


def _highest(snapshots: list[dict[str, int]]) -> dict[str, int]:
    """The reference: the per-field maximum over the report."""
    best: dict[str, int] = {}
    for snap in snapshots:
        for key, value in snap.items():
            best[key] = max(best.get(key, 0), value)
    return best


def _nonzero(usage: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in usage.items() if v}


# --- StreamContext.usage_increment and add_usage -----------------------------


@_SETTINGS
@given(snapshots=_snapshots())
def test_the_increments_of_a_report_add_up_to_its_highest_snapshot(snapshots) -> None:
    context = StreamContext()
    total: dict[str, Any] | None = None
    increments = []
    for snap in snapshots:
        increment = context.usage_increment(snap)
        if increment is not None:
            increments.append(increment)
            total = add_usage(total, increment)
    assert total is not None  # the first snapshot always passes through
    assert _nonzero(total) == _nonzero(_highest(snapshots))
    # An increment never goes down, and states the three required counts.
    for index, increment in enumerate(increments):
        assert all(v >= 0 for v in increment.values())
        if index:
            assert {"prompt_tokens", "completion_tokens", "total_tokens"} <= set(
                increment
            )


@_SETTINGS
@given(snapshots=_snapshots())
def test_the_first_snapshot_passes_through_whole_zeros_included(snapshots) -> None:
    first = StreamContext().usage_increment(snapshots[0])
    assert first == snapshots[0]


@_SETTINGS
@given(snapshot=_snapshots().map(lambda s: s[0]))
def test_a_snapshot_that_repeats_adds_nothing(snapshot) -> None:
    context = StreamContext()
    assert context.usage_increment(snapshot) is not None
    assert context.usage_increment(dict(snapshot)) is None


@_SETTINGS
@given(snapshots=_snapshots())
def test_the_report_is_independent_of_how_often_it_is_repeated(snapshots) -> None:
    once, twice = StreamContext(), StreamContext()
    total_once: dict[str, Any] | None = None
    total_twice: dict[str, Any] | None = None
    for snap in snapshots:
        for context, repeat in ((once, 1), (twice, 2)):
            for _ in range(repeat):
                increment = context.usage_increment(snap)
                if increment is not None:
                    if context is once:
                        total_once = add_usage(total_once, increment)
                    else:
                        total_twice = add_usage(total_twice, increment)
    assert total_once == total_twice


_NESTED = st.fixed_dictionaries(
    {
        "prompt_tokens": st.integers(0, 30),
        "details": st.dictionaries(
            st.sampled_from(["text", "audio"]), st.integers(0, 30), max_size=2
        ),
    },
    optional={"tier": st.sampled_from(["a", "b"])},
)


@_SETTINGS
@given(parts=st.lists(_NESTED, min_size=1, max_size=4))
def test_add_usage_sums_numbers_recursively_and_keeps_the_latest_text(parts) -> None:
    total: dict[str, Any] | None = None
    for part in parts:
        total = add_usage(total, part)
    assert total is not None
    assert total["prompt_tokens"] == sum(p["prompt_tokens"] for p in parts)
    for key in ("text", "audio"):
        stated = [p["details"][key] for p in parts if key in p["details"]]
        if stated:
            assert total["details"][key] == sum(stated)
    tiers = [p["tier"] for p in parts if "tier" in p]
    assert total.get("tier") == (tiers[-1] if tiers else None)


@_SETTINGS
@given(a=_NESTED, b=_NESTED, c=_NESTED)
def test_add_usage_is_associative_on_numbers(a, b, c) -> None:
    left = add_usage(add_usage(a, b), c)
    right = add_usage(a, add_usage(b, c))
    assert left["prompt_tokens"] == right["prompt_tokens"]
    assert left["details"] == right["details"]


def test_add_usage_changes_neither_argument_and_none_states_nothing() -> None:
    held = {"prompt_tokens": 3, "details": {"text": 1}, "tier": "a"}
    step = {"prompt_tokens": 2, "details": {"text": 4}, "tier": None, "extra": None}
    before = (dict(held), {"prompt_tokens": 2, "details": {"text": 4}})
    result = add_usage(held, step)
    assert held["prompt_tokens"] == 3 and held["details"] == {"text": 1}
    assert step["details"] == {"text": 4}
    assert result["prompt_tokens"] == 5 and result["details"] == {"text": 5}
    assert result["tier"] == "a" and result["extra"] is None
    assert before[0]["tier"] == "a"


# --- the readers -------------------------------------------------------------


def _read(name: str, chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    converter = _LLM[name]()
    return list(
        converter.iter_stream_from_provider(
            chunks, context=converter.create_stream_context()
        )
    )


def _sum_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    total: dict[str, Any] | None = None
    for event in events:
        if event["type"] == "usage":
            total = add_usage(total, event["usage"])
    return total or {}


@_SETTINGS
@given(
    snapshots=_snapshots(keys=("prompt_tokens", "completion_tokens", "total_tokens"))
)
def test_a_google_stream_reads_to_its_highest_snapshot(snapshots) -> None:
    chunks = []
    for index, snap in enumerate(snapshots):
        usage: dict[str, int] = {}
        if "prompt_tokens" in snap:
            usage["promptTokenCount"] = snap["prompt_tokens"]
        if "completion_tokens" in snap:
            usage["candidatesTokenCount"] = snap["completion_tokens"]
        if "total_tokens" in snap:
            usage["totalTokenCount"] = snap["total_tokens"]
        candidate: dict[str, Any] = {
            "content": {"role": "model", "parts": [{"text": "x"}]}
        }
        if index == len(snapshots) - 1:
            candidate["finishReason"] = "STOP"
        chunks.append({"candidates": [candidate], "usageMetadata": usage})
    total = _sum_events(_read("google_generate", chunks))
    assert _nonzero({k: total.get(k, 0) for k in _FIELDS[:3]}) == _nonzero(
        _highest(snapshots)
    )


@_SETTINGS
@given(
    snapshots=_snapshots(keys=("prompt_tokens", "completion_tokens", "total_tokens"))
)
def test_a_chat_stream_that_repeats_usage_reads_to_its_highest_snapshot(
    snapshots,
) -> None:
    base = {"id": "c", "object": "chat.completion.chunk", "created": 1, "model": "m"}
    chunks: list[dict[str, Any]] = [
        {
            **base,
            "choices": [
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": "x"},
                    "finish_reason": None,
                }
            ],
        }
    ]
    for snap in snapshots:
        chunks.append({**base, "choices": [], "usage": dict(snap)})
    chunks.append(
        {**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]}
    )
    total = _sum_events(_read("chat_completions", chunks))
    assert _nonzero({k: total.get(k, 0) for k in _FIELDS[:3]}) == _nonzero(
        _highest(snapshots)
    )


def _messages_stream(
    start: dict[str, int], deltas: list[dict[str, int]]
) -> list[dict[str, Any]]:
    chunks: list[dict[str, Any]] = [
        {
            "type": "message_start",
            "message": {
                "id": "msg_1",
                "type": "message",
                "role": "assistant",
                "model": "m",
                "content": [],
                "usage": start,
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
            "delta": {"type": "text_delta", "text": "x"},
        },
        {"type": "content_block_stop", "index": 0},
    ]
    for usage in deltas:
        chunks.append(
            {
                "type": "message_delta",
                "delta": {"stop_reason": "end_turn"},
                "usage": usage,
            }
        )
    chunks.append({"type": "message_stop"})
    return chunks


@_SETTINGS
@given(
    input_tokens=st.integers(0, 200),
    cache_read=st.integers(0, 200),
    outputs=st.lists(st.integers(0, 100), min_size=1, max_size=4),
    repeat_input=st.booleans(),
)
def test_a_messages_stream_counts_input_and_cache_once(
    input_tokens, cache_read, outputs, repeat_input
) -> None:
    """Start states input and cache; each delta states the output so far, and
    may repeat the input and cache: none of it counts twice."""
    start = {
        "input_tokens": input_tokens,
        "cache_read_input_tokens": cache_read,
        "output_tokens": 0,
    }
    deltas = []
    for output in outputs:
        usage = {"output_tokens": output}
        if repeat_input:
            usage.update(input_tokens=input_tokens, cache_read_input_tokens=cache_read)
        deltas.append(usage)
    total = _sum_events(_read("messages", _messages_stream(start, deltas)))
    assert total.get("completion_tokens", 0) == max(outputs)
    assert total.get("cache_read_tokens", 0) == cache_read
    assert total.get("total_tokens", 0) == input_tokens + cache_read + max(outputs)


def test_a_messages_cache_stream_reads_to_142() -> None:
    """input 40 + cache read 100 + output 2; the delta repeats the first two."""
    start = {"input_tokens": 40, "cache_read_input_tokens": 100, "output_tokens": 1}
    delta = {"input_tokens": 40, "cache_read_input_tokens": 100, "output_tokens": 2}
    total = _sum_events(_read("messages", _messages_stream(start, [delta])))
    assert total["total_tokens"] == 142
    assert total["completion_tokens"] == 2 and total["cache_read_tokens"] == 100


# --- the consumers -----------------------------------------------------------


@_SETTINGS
@given(
    increments=st.lists(
        st.fixed_dictionaries(
            {
                "prompt_tokens": st.integers(0, 40),
                "completion_tokens": st.integers(0, 40),
                "total_tokens": st.integers(0, 80),
            }
        ),
        min_size=1,
        max_size=5,
    )
)
def test_the_gateway_record_and_the_fold_both_add_the_events(increments) -> None:
    expected = {
        key: sum(i[key] for i in increments)
        for key in ("prompt_tokens", "completion_tokens", "total_tokens")
    }
    holder = SimpleNamespace(_usage=None)
    for usage in increments:
        merge: Any = streaming_module.StreamProcessor._merge_usage
        merge(holder, dict(usage))
    assert holder._usage == expected

    events: list[Any] = [
        {"type": "stream_start", "response_id": "r", "model": "m"},
        {"type": "text_delta", "text": "x"},
        *({"type": "usage", "usage": dict(u)} for u in increments),
        {"type": "finish", "finish_reason": {"reason": "stop"}},
        {"type": "stream_end"},
    ]
    folded = fold_response_events(events)["usage"]
    folded_usage: dict[str, Any] = dict(folded)
    assert {k: folded_usage[k] for k in expected} == expected


# --- nested details, text fields, exactness, the decrease warning -----------


@st.composite
def _rich_snapshots(draw) -> list[dict[str, Any]]:
    """Snapshots with a nested detail dict and a text field, as real wires have."""
    out: list[dict[str, Any]] = []
    prompt = draw(st.integers(0, 30))
    details = {"text": draw(st.integers(0, 30))}
    for _ in range(draw(st.integers(1, 5))):
        prompt = max(prompt, draw(st.integers(0, 60)))
        details = {
            "text": max(details["text"], draw(st.integers(0, 60))),
            **({"audio": draw(st.integers(0, 9))} if draw(st.booleans()) else {}),
        }
        snap: dict[str, Any] = {
            "prompt_tokens": prompt,
            "completion_tokens": 0,
            "total_tokens": prompt,
            "details": dict(details),
        }
        if draw(st.booleans()):
            snap["tier"] = draw(st.sampled_from(["a", "b"]))
        out.append(snap)
    return out


@_SETTINGS
@given(snapshots=_rich_snapshots())
def test_nested_details_and_text_fields_follow_the_same_contract(snapshots) -> None:
    context = StreamContext()
    total: dict[str, Any] | None = None
    last_tier = None
    for snap in snapshots:
        increment = context.usage_increment(snap)
        if increment is not None:
            total = add_usage(total, increment)
        last_tier = snap.get("tier", last_tier)
    assert total is not None
    assert total["prompt_tokens"] == max(s["prompt_tokens"] for s in snapshots)
    for key in ("text", "audio"):
        stated = [s["details"][key] for s in snapshots if key in s["details"]]
        if stated and max(stated):
            assert total["details"][key] == max(stated)
    assert total.get("tier") == last_tier


@_SETTINGS
@given(snapshots=_snapshots())
def test_an_increment_after_the_first_states_only_what_rose(snapshots) -> None:
    context = StreamContext()
    context.usage_increment(snapshots[0])
    held = dict(snapshots[0])
    for snap in snapshots[1:]:
        increment = context.usage_increment(snap)
        rose = {k: v - held[k] for k, v in snap.items() if v > held.get(k, 0)}
        for key, value in snap.items():
            held[key] = max(held.get(key, 0), value)
        if not rose:
            assert increment is None
            continue
        assert increment is not None
        assert {k: v for k, v in increment.items() if v} == rose
        assert set(increment) - set(rose) <= {
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
        }


def test_a_decrease_is_logged_once_per_stream(caplog: pytest.LogCaptureFixture) -> None:
    context = StreamContext()
    with caplog.at_level("WARNING", logger=context_module.logger.name):
        context.usage_increment({"prompt_tokens": 10, "completion_tokens": 5})
        context.usage_increment({"prompt_tokens": 4, "completion_tokens": 5})
        context.usage_increment({"prompt_tokens": 3, "completion_tokens": 5})
    warnings = [r for r in caplog.records if "went down" in r.getMessage()]
    assert len(warnings) == 1
    assert "prompt_tokens" in warnings[0].getMessage()


def test_a_rise_and_no_decrease_logs_nothing(caplog: pytest.LogCaptureFixture) -> None:
    context = StreamContext()
    with caplog.at_level("WARNING", logger=context_module.logger.name):
        context.usage_increment({"prompt_tokens": 1, "completion_tokens": 1})
        context.usage_increment({"prompt_tokens": 2, "completion_tokens": 1})
    assert not [r for r in caplog.records if "went down" in r.getMessage()]


def test_a_late_zero_alone_is_not_an_event_but_a_rise_brings_it_along() -> None:
    context = StreamContext()
    context.usage_increment({"prompt_tokens": 5, "completion_tokens": 2})
    assert (
        context.usage_increment(
            {"prompt_tokens": 5, "completion_tokens": 2, "reasoning_tokens": 0}
        )
        is None
    )
    increment = context.usage_increment(
        {"prompt_tokens": 5, "completion_tokens": 3, "reasoning_tokens": 0}
    )
    assert increment is not None and increment["completion_tokens"] == 1


def test_a_boolean_field_is_text_not_a_count() -> None:
    """``True`` is an ``int`` in Python; a usage flag must not be counted."""
    context = StreamContext()
    assert context.usage_increment({"cached": True}) == {"cached": True}
    assert context.usage_increment({"cached": True}) is None
    assert context.usage_increment({"cached": False}) == {
        "cached": False,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
    }
    assert add_usage({"cached": 1}, {"cached": True}) == {"cached": True}
    assert add_usage({"n": 2}, {"n": 3}) == {"n": 5}
