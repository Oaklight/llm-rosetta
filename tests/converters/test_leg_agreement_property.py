"""The two response legs agree on generated turns (Hypothesis).

``test_leg_agreement.py`` checks the thirteen turns of ``CASE_TABLE``.  This
module states the same law over generated turns: any turn a dialect can write,
written as a body and as a stream, reads back as the same turn, and that turn
keeps the text, the reasoning, the calls (id, name, input, order) and the usage
the IR turn carried.  It reuses the table's own helpers, so the two cannot
drift on what "the turn" means.

The generator draws the terminal state from the three reasons every dialect
writes (``stop``, ``tool_calls``, ``length``); the refusal and error rows stay
with the table, because they are decisions about one dialect each.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

pytest.importorskip("hypothesis")

from hypothesis import HealthCheck, given, settings  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

from tests.converters.test_leg_agreement import (  # noqa: E402
    DIALECTS,
    EXPECTED_CALL_KINDS,
    EXPECTED_REASON,
    KEEPS_REFUSAL_WORDS,
    Refused,
    Turn,
    TurnView,
    _body_leg,
    _ir_events,
    _ir_response,
    _stream_leg,
    _write_events,
)

_TEXT = st.text(
    alphabet=st.characters(blacklist_categories=("Cs",), blacklist_characters="\x00"),
    max_size=40,
)
_JSON_LEAF = st.one_of(
    st.none(),
    st.booleans(),
    st.integers(min_value=-1000, max_value=1000),
    st.text(max_size=12),
)
_ARGUMENTS = st.dictionaries(
    st.text(alphabet="abcdefghij_", min_size=1, max_size=6),
    st.one_of(_JSON_LEAF, st.lists(_JSON_LEAF, max_size=3)),
    max_size=4,
)


@st.composite
def _calls(draw) -> tuple[dict, ...]:
    calls = []
    for index in range(draw(st.integers(min_value=0, max_value=3))):
        kind = draw(st.sampled_from(["function", "custom", "intrinsic"]))
        call: dict[str, Any] = {
            "type": "tool_call",
            "tool_call_id": f"call_{index}",
            "tool_name": draw(st.sampled_from(["lookup", "search", "run_cmd"])),
            "tool_input": draw(_ARGUMENTS),
        }
        if kind == "custom":
            call["tool_type"] = "custom"
            # Texts that are JSON objects stay out on purpose: the IR keeps such
            # a text structured, so it is re-serialized on the way out and the
            # text ``{"input": "x"}`` reads as the dict (see
            # ``llm_rosetta.ir.tools.custom_input``).  That is a known limit, not
            # a law the legs can be held to.
            call["tool_input"] = {
                "input": draw(
                    st.one_of(
                        st.text(alphabet="abc xyz", max_size=12),
                        st.sampled_from(["0", "true", "null", "[1, 2]", '"s"']),
                    )
                )
            }
        elif kind == "intrinsic":
            call["tool_type"] = "intrinsic"
            call["intrinsic_kind"] = "web_search"
            call["tool_name"] = "web_search"
            call["tool_input"] = {"query": draw(st.text(max_size=12))}
        calls.append(call)
    return tuple(calls)


def _client_calls(turn: Turn) -> list[dict[str, Any]]:
    """The calls a client must answer: a provider-run (intrinsic) call is not one."""
    return [c for c in turn.calls if c.get("tool_type") != "intrinsic"]


@st.composite
def _turns(draw) -> Turn:
    usage = draw(
        st.one_of(
            st.none(),
            st.builds(
                lambda p, c: {
                    "prompt_tokens": p,
                    "completion_tokens": c,
                    "total_tokens": p + c,
                },
                st.integers(min_value=1, max_value=500),
                st.integers(min_value=1, max_value=500),
            ),
        )
    )
    text, reasoning, calls = draw(_TEXT), draw(_TEXT), draw(_calls())
    # ``tool_calls`` names a turn with a call the client must answer; without
    # one, the dialects that derive the reason from the content write ``stop``.
    answerable = [c for c in calls if c.get("tool_type") != "intrinsic"]
    finishes = ["stop", "length"] + (["tool_calls"] if answerable else [])
    return Turn(
        text=text,
        reasoning=reasoning,
        calls=calls,
        finish=draw(st.sampled_from(finishes)),
        usage=usage,
    )


def _expected_reason(turn: Turn) -> str:
    """A ``stop`` that carries a client call is a tool-call turn on every dialect."""
    if turn.finish == "stop" and _client_calls(turn):
        return "tool_calls"
    return turn.finish


def _expected_calls(dialect: str, turn: Turn) -> list[tuple[str, str, str]]:
    """``(id, name, kind)`` of each call the dialect keeps, as the table declares."""
    custom_kept = EXPECTED_CALL_KINDS[(dialect, "custom")] == ("custom",)
    intrinsic_kept = EXPECTED_CALL_KINDS[(dialect, "intrinsic")] != ()
    kept = []
    for call in turn.calls:
        kind = call.get("tool_type", "function")
        if kind == "intrinsic" and not intrinsic_kept:
            continue
        if kind == "custom" and not custom_kept:
            kind = "function"
        if kind == "intrinsic":
            kind = f"intrinsic:{call['intrinsic_kind']}"
        kept.append((call["tool_call_id"], call["tool_name"], kind))
    return kept


def _assert_custom_text_on_the_wire(converter: Any, turn: Turn) -> None:
    """Both Open Responses writers send a custom call's text, not its IR wrapper."""
    texts = {
        c["tool_call_id"]: c["tool_input"]["input"]
        for c in turn.calls
        if c.get("tool_type") == "custom"
    }
    body = converter.response_to_provider(_ir_response(turn))
    from_body = {
        i["call_id"]: i["input"]
        for i in body["output"]
        if i["type"] == "custom_tool_call"
    }
    chunks = _write_events(converter, _ir_events(turn))
    from_stream = {
        c["item"]["call_id"]: c["item"]["input"]
        for c in chunks
        if c.get("type") == "response.output_item.done"
        and c["item"]["type"] == "custom_tool_call"
    }
    assert from_body == texts
    assert from_stream == texts


@pytest.mark.parametrize("dialect", DIALECTS)
@settings(
    max_examples=150,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    derandomize=True,
)
@given(turn=_turns())
def test_the_legs_agree_and_keep_the_turn(dialect: str, turn: Turn) -> None:
    converter = DIALECTS[dialect]()
    body = _body_leg(converter, turn)
    stream = _stream_leg(converter, turn)

    assert body == stream, f"{dialect}: body and stream legs disagree on {turn}"
    assert isinstance(body, TurnView) and not isinstance(body, Refused)

    assert body.reason == _expected_reason(turn)
    assert body.text == turn.text
    assert body.reasoning == turn.reasoning
    expected = _expected_calls(dialect, turn)
    # ``TurnView`` sorts calls by id and a provider-run call's id is re-prefixed,
    # so the view's order is not the turn's; the calls are compared as sets.
    assert sorted((c.name, c.kind) for c in body.calls) == sorted(
        (n, k) for _, n, k in expected
    )
    # A dialect re-prefixes the id of a provider-run call (``ws_``, ``srvtoolu_``)
    # and keeps every other id as it was.
    for call_id, _, kind in expected:
        if kind.startswith("intrinsic"):
            assert any(c.call_id.endswith(call_id) for c in body.calls)
        else:
            assert call_id in [c.call_id for c in body.calls]
    kept_inputs = sorted(
        json.dumps(c["tool_input"], sort_keys=True)
        for c in turn.calls
        if c.get("tool_type") != "intrinsic"
        or EXPECTED_CALL_KINDS[(dialect, "intrinsic")]
    )
    assert (
        sorted(json.dumps(json.loads(c.arguments), sort_keys=True) for c in body.calls)
        == kept_inputs
    )
    if dialect == "open_responses":
        _assert_custom_text_on_the_wire(converter, turn)
    if turn.usage is not None:
        assert body.usage == (
            turn.usage["prompt_tokens"],
            turn.usage["completion_tokens"],
            turn.usage["total_tokens"],
        )
    else:
        assert body.usage is None


@pytest.mark.parametrize("dialect", DIALECTS)
@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    derandomize=True,
)
@given(words=st.text(alphabet="abc xyz.", min_size=1, max_size=30))
def test_a_refusal_keeps_its_words_on_both_legs(dialect: str, words: str) -> None:
    """The words survive: as a refusal where the dialect has one, else as text."""
    turn = Turn(finish="refusal", refusal=words)
    converter = DIALECTS[dialect]()
    body = _body_leg(converter, turn)
    stream = _stream_leg(converter, turn)

    assert body == stream
    assert isinstance(body, TurnView)
    assert body.reason == EXPECTED_REASON.get(
        (dialect, "refusal_with_words"), "refusal"
    )
    if dialect in KEEPS_REFUSAL_WORDS:
        assert body.refusal == words
        assert body.text == ""
    else:
        assert body.refusal == ""
        assert body.text == words


@settings(
    max_examples=80,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
    derandomize=True,
)
@given(
    value=st.one_of(
        st.none(),
        st.booleans(),
        st.integers(min_value=-1000, max_value=1000),
        st.lists(st.integers(min_value=-9, max_value=9), max_size=3),
    )
)
def test_a_non_string_custom_input_is_written_as_json_on_both_writers(
    value: Any,
) -> None:
    """``{"input": True}`` goes on the wire as ``true``, never as ``True``."""
    turn = Turn(
        calls=(
            {
                "type": "tool_call",
                "tool_call_id": "call_0",
                "tool_name": "lookup",
                "tool_input": {"input": value},
                "tool_type": "custom",
            },
        ),
        finish="tool_calls",
    )
    converter = DIALECTS["open_responses"]()
    body = converter.response_to_provider(_ir_response(turn))
    (from_body,) = [
        i["input"] for i in body["output"] if i["type"] == "custom_tool_call"
    ]
    chunks = _write_events(converter, _ir_events(turn))
    (from_stream,) = [
        c["item"]["input"]
        for c in chunks
        if c.get("type") == "response.output_item.done"
        and c["item"]["type"] == "custom_tool_call"
    ]
    assert from_body == from_stream == json.dumps(value)
