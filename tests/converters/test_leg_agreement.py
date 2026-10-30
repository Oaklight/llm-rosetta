"""The two response legs must answer the same turn the same way.

A dialect writes a turn twice: once as a whole body
(:meth:`BaseConverter.response_to_provider`) and once as a stream
(:meth:`BaseConverter.stream_response_to_provider`).  Nothing compares them
today, which is how a terminal reason fixed on one leg kept answering on the
other (#1234, #1236, #1237).

The check feeds both legs the same IR turn, reads each provider payload back
with the *same* dialect's reader, and compares the two in the IR's vocabulary.
Reading back removes the wire-shape differences (Anthropic's ``content`` vs
Responses' ``output``) without hiding a dropped part: a writer that loses a
call cannot be read back with it.  A writer and a reader wrong in exactly
compensating ways would pass; provider shape is the wire baseline's job, not
this harness's.

Scope is the terminal state and the content both legs can express, for a
single-choice turn.  Multi-choice Chat is a follow-up.  Fields that exist only
in a body are declared below and not asserted; provider-level field equality is
the conformance wire baseline's job (``tools/conformance``).

Epic: #1238.  The harness is the regression net for the finish-reason rule
(#1240).  The one remaining shared loss and the stop-sequence value gap are
pinned as strict xfails rather than excluded: fixing them turns the marker into
a failure and forces the pin out in the same change.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import pytest

from llm_rosetta.converters.base.errors import UnrepresentableFinishReason
from llm_rosetta.converters.llm.chat_completions import ChatCompletionsConverter
from llm_rosetta.converters.llm.google_generate import GoogleGenerateConverter
from llm_rosetta.converters.llm.google_interactions import GoogleInteractionsConverter
from llm_rosetta.converters.llm.messages import MessagesConverter
from llm_rosetta.converters.llm.open_responses import OpenResponsesConverter

ConverterFactory = Callable[[], Any]

DIALECTS: dict[str, ConverterFactory] = {
    "chat_completions": ChatCompletionsConverter,
    "open_responses": OpenResponsesConverter,
    "messages": MessagesConverter,
    "google_generate": GoogleGenerateConverter,
    "google_interactions": GoogleInteractionsConverter,
}

#: The compared surface is ``TurnView``; everything else is out of scope by
#: construction.  The emitted-field ledger below records the fields the writers
#: actually produce, so a field nobody modelled is still visible as a path.

USAGE = {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}

FUNCTION_CALL = {
    "type": "tool_call",
    "tool_call_id": "call_1",
    "tool_name": "lookup",
    "tool_input": {"q": "x"},
}
CUSTOM_CALL = {
    "type": "tool_call",
    "tool_call_id": "call_c",
    "tool_name": "shell",
    "tool_input": {"input": "ls"},
    "tool_type": "custom",
}
INTRINSIC_CALL = {
    "type": "tool_call",
    "tool_call_id": "call_s",
    "tool_name": "web_search",
    "tool_input": {"query": "x"},
    "tool_type": "intrinsic",
    "intrinsic_kind": "web_search",
}


@dataclass(frozen=True)
class Turn:
    """One assistant turn, described once and rendered onto both legs."""

    text: str = ""
    reasoning: str = ""
    calls: tuple[dict[str, Any], ...] = ()
    finish: str = "stop"
    stop_sequence: str | None = None
    native_finish_reason: str | None = None
    usage: dict[str, int] | None = None


@dataclass(frozen=True)
class CallView:
    """A call as both legs see it after read-back."""

    call_id: str
    name: str
    arguments: str
    kind: str  # "function", "custom" or "intrinsic:<kind>"


@dataclass(frozen=True)
class TurnView:
    """The comparable content of a turn."""

    reason: str
    text: str
    reasoning: str
    calls: tuple[CallView, ...]
    usage: tuple[int, int, int] | None


@dataclass(frozen=True)
class Refused:
    """A writer that raised: both legs must refuse for the same IR reason."""

    reason: str


#: (case id, turn, the IR reason a dialect capable of expressing the turn
#: writes).  The per-dialect overrides are in ``EXPECTED_REASON``.
CASE_TABLE: list[tuple[str, Turn, str]] = [
    ("text", Turn(text="hi", usage=USAGE), "stop"),
    ("reasoning", Turn(reasoning="think", text="hi"), "stop"),
    (
        "tool",
        Turn(text="let me", calls=(FUNCTION_CALL,), finish="tool_calls"),
        "tool_calls",
    ),
    (
        "stop_with_call",
        Turn(text="let me", calls=(FUNCTION_CALL,), finish="stop"),
        "tool_calls",
    ),
    ("custom", Turn(calls=(CUSTOM_CALL,), finish="tool_calls"), "tool_calls"),
    ("intrinsic", Turn(calls=(INTRINSIC_CALL,), finish="stop"), "stop"),
    ("length", Turn(text="hi", finish="length"), "length"),
    (
        "length_with_call",
        Turn(text="hi", calls=(FUNCTION_CALL,), finish="length"),
        "length",
    ),
    ("filter", Turn(text="hi", finish="content_filter"), "content_filter"),
    ("error", Turn(text="hi", finish="error"), "error"),
    ("cancelled", Turn(text="hi", finish="cancelled"), "cancelled"),
]

CASES = [
    pytest.param(case_id, turn, reason, id=case_id.replace("_", "-"))
    for case_id, turn, reason in CASE_TABLE
]

#: What a dialect expects instead of the capable-dialect reason above.  The
#: table is the assertion that stops a shared loss from counting as agreement:
#: if both legs drop the same thing, the two views still match, but the
#: expected value here does not.
EXPECTED_REASON: dict[tuple[str, str], str] = {
    ("messages", "filter"): "refusal",
}

#: Dialects whose writers refuse the turn instead of writing it.
REFUSED_REASON = {
    "chat_completions": {"error", "cancelled"},
    "messages": {"error", "cancelled"},
    "google_generate": {"error", "cancelled"},
}

#: Call kinds a dialect keeps, per case.  A case absent here has no call, or
#: keeps the function call every dialect keeps.
EXPECTED_CALL_KINDS: dict[tuple[str, str], tuple[str, ...]] = {
    ("chat_completions", "custom"): ("custom",),
    ("open_responses", "custom"): ("custom",),
    ("messages", "custom"): ("function",),
    ("google_generate", "custom"): ("function",),
    ("google_interactions", "custom"): ("function",),
    ("open_responses", "intrinsic"): ("intrinsic:web_search",),
    ("messages", "intrinsic"): ("intrinsic:web_search",),
    ("chat_completions", "intrinsic"): (),
    ("google_generate", "intrinsic"): (),
    ("google_interactions", "intrinsic"): (),
}

#: (dialect, case) pairs where both legs agree on the wrong answer.  Equality
#: cannot see these; the dedicated strict xfail below is the pin.  The fix is
#: a writer change, not a leg reconciliation.
KNOWN_SHARED_LOSSES = {
    ("google_interactions", "filter"): (
        "the Interactions writer maps a filtered turn to status completed, so "
        "both legs agree it finished normally (#1244)"
    ),
}


def _ir_response(turn: Turn) -> dict[str, Any]:
    """Render *turn* as an IR response body."""
    content: list[dict[str, Any]] = []
    if turn.reasoning:
        content.append({"type": "reasoning", "reasoning": turn.reasoning})
    if turn.text:
        content.append({"type": "text", "text": turn.text})
    content.extend(copy.deepcopy(turn.calls))
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if turn.native_finish_reason is not None:
        message["provider_metadata"] = {
            "_native_finish_reason": {
                "base": "google_generate",
                "reason": turn.native_finish_reason,
            }
        }
    response: dict[str, Any] = {
        "id": "turn1",
        "object": "response",
        "created": 1,
        "model": "m",
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": {
                    "reason": turn.finish,
                    **(
                        {"stop_sequence": turn.stop_sequence}
                        if turn.stop_sequence is not None
                        else {}
                    ),
                },
            }
        ],
    }
    if turn.usage is not None:
        response["usage"] = dict(turn.usage)
    return response


def _ir_events(turn: Turn) -> list[dict[str, Any]]:
    """Render *turn* as an IR stream event sequence."""
    events: list[dict[str, Any]] = [
        {"type": "stream_start", "response_id": "turn1", "model": "m", "created": 1}
    ]
    if turn.reasoning:
        events.append({"type": "reasoning_delta", "reasoning": turn.reasoning})
    if turn.text:
        events.append({"type": "text_delta", "text": turn.text})
    for index, call in enumerate(turn.calls):
        start: dict[str, Any] = {
            "type": "tool_call_start",
            "tool_call_id": call["tool_call_id"],
            "tool_name": call["tool_name"],
            "tool_call_index": index,
            "tool_type": call.get("tool_type", "function"),
        }
        if call.get("intrinsic_kind"):
            start["intrinsic_kind"] = call["intrinsic_kind"]
        events.append(start)
        events.append(
            {
                "type": "tool_call_delta",
                "tool_call_id": call["tool_call_id"],
                "arguments_delta": json.dumps(call["tool_input"]),
            }
        )
    if turn.usage is not None:
        events.append({"type": "usage", "usage": dict(turn.usage)})
    finish_reason: dict[str, Any] = {"reason": turn.finish}
    if turn.stop_sequence is not None:
        finish_reason["stop_sequence"] = turn.stop_sequence
    finish: dict[str, Any] = {"type": "finish", "finish_reason": finish_reason}
    if turn.native_finish_reason is not None:
        finish["provider_metadata"] = {
            "_native_finish_reason": {
                "base": "google_generate",
                "reason": turn.native_finish_reason,
            }
        }
    events.append(finish)
    events.append({"type": "stream_end"})
    return events


def _as_dicts(result: Any) -> list[dict[str, Any]]:
    """Flatten a stream writer's return value into a chunk list."""
    if isinstance(result, list):
        return [chunk for chunk in result if chunk]
    return [result] if result else []


def _body_leg(converter: Any, turn: Turn) -> TurnView | Refused:
    """Write the turn as a body and read it back."""
    try:
        body = converter.response_to_provider(_ir_response(turn))
    except UnrepresentableFinishReason as exc:
        return Refused(reason=exc.reason)
    return _view_of_body(converter.response_from_provider(body))


def _raw_stream_chunks(converter: Any, turn: Turn) -> list[dict[str, Any]]:
    """Write the turn as a stream and return the provider chunks."""
    to_context = converter.create_stream_context()
    chunks: list[dict[str, Any]] = []
    for event in _ir_events(turn):
        chunks.extend(
            _as_dicts(converter.stream_response_to_provider(event, context=to_context))
        )
    return chunks


def _raw_stream_events(converter: Any, turn: Turn) -> list[dict[str, Any]]:
    """Write the turn as a stream and read the chunks back as IR events."""
    chunks = _raw_stream_chunks(converter, turn)
    from_context = converter.create_stream_context()
    return list(converter.iter_stream_from_provider(chunks, context=from_context))


def _stream_leg(converter: Any, turn: Turn) -> TurnView | Refused:
    """Write the turn as a stream and read it back."""
    try:
        events = _raw_stream_events(converter, turn)
    except UnrepresentableFinishReason as exc:
        return Refused(reason=exc.reason)
    return _view_of_events(events)


def _stop_sequence(converter: Any, turn: Turn) -> tuple[str | None, str | None]:
    """Return the stop sequence each leg carries after read-back."""
    body = converter.response_from_provider(
        converter.response_to_provider(_ir_response(turn))
    )
    stream = _raw_stream_events(converter, turn)
    body_sequence = body["choices"][0]["finish_reason"].get("stop_sequence")
    finish = next(event for event in stream if event.get("type") == "finish")
    return body_sequence, (finish.get("finish_reason") or {}).get("stop_sequence")


def _call_view(
    call_id: str,
    name: str,
    arguments: Any,
    tool_type: str | None,
    intrinsic_kind: str | None,
) -> CallView:
    """Normalise one call part/event into a comparable value."""
    kind = tool_type or "function"
    if kind == "intrinsic":
        kind = f"intrinsic:{intrinsic_kind or ''}"
    if isinstance(arguments, str):
        try:
            arguments = json.dumps(json.loads(arguments), sort_keys=True)
        except json.JSONDecodeError:
            pass
    else:
        arguments = json.dumps(arguments, sort_keys=True)
    return CallView(call_id=call_id, name=name, arguments=arguments, kind=kind)


def _usage_view(raw: dict[str, Any] | None) -> tuple[int, int, int] | None:
    """Compare the three core token counts; an all-zero report is absence."""
    if not raw:
        return None
    triple: tuple[int, int, int] = (
        int(raw.get("prompt_tokens") or 0),
        int(raw.get("completion_tokens") or 0),
        int(raw.get("total_tokens") or 0),
    )
    return None if triple == (0, 0, 0) else triple


def _view_of_body(ir_response: dict[str, Any]) -> TurnView:
    """Compare a read-back body."""
    choice = ir_response["choices"][0]
    parts = (choice.get("message") or {}).get("content") or []
    text = "".join(part.get("text", "") for part in parts if part.get("type") == "text")
    reasoning = "".join(
        part.get("reasoning", "") for part in parts if part.get("type") == "reasoning"
    )
    calls = tuple(
        sorted(
            (
                _call_view(
                    part.get("tool_call_id", ""),
                    part.get("tool_name", ""),
                    part.get("tool_input", {}),
                    part.get("tool_type"),
                    part.get("intrinsic_kind"),
                )
                for part in parts
                if part.get("type") == "tool_call"
            ),
            key=lambda call: call.call_id,
        )
    )
    return TurnView(
        reason=(choice.get("finish_reason") or {}).get("reason", ""),
        text=text,
        reasoning=reasoning,
        calls=calls,
        usage=_usage_view(ir_response.get("usage")),
    )


def _view_of_events(events: list[dict[str, Any]]) -> TurnView:
    """Compare a read-back stream by folding its events into one turn."""
    text = ""
    reasoning = ""
    reason = ""
    usage: dict[str, int] = {}
    calls: dict[str, dict[str, Any]] = {}
    for event in events:
        event_type = event.get("type")
        if event_type == "text_delta":
            text += event.get("text", "")
        elif event_type == "reasoning_delta":
            reasoning += event.get("reasoning", "")
        elif event_type == "tool_call_start":
            call_id = event.get("tool_call_id", "")
            calls[call_id] = {
                "name": event.get("tool_name", ""),
                "arguments": "",
                "tool_type": event.get("tool_type"),
                "intrinsic_kind": event.get("intrinsic_kind"),
            }
        elif event_type == "tool_call_delta":
            call = calls.get(event.get("tool_call_id", ""))
            if call is not None:
                call["arguments"] += event.get("arguments_delta") or ""
        elif event_type == "finish":
            reason = (event.get("finish_reason") or {}).get("reason", "")
        elif event_type == "usage":
            for key, value in (event.get("usage") or {}).items():
                if value:
                    usage[key] = value
    call_views = tuple(
        sorted(
            (
                _call_view(
                    call_id,
                    call["name"],
                    call["arguments"],
                    call["tool_type"],
                    call["intrinsic_kind"],
                )
                for call_id, call in calls.items()
            ),
            key=lambda call: call.call_id,
        )
    )
    return TurnView(
        reason=reason,
        text=text,
        reasoning=reasoning,
        calls=call_views,
        usage=_usage_view(usage),
    )


def _assert_call_contents(view: TurnView, turn: Turn) -> None:
    """Every call the turn asked for is there, with the same name and input."""
    assert len(view.calls) == len(turn.calls)
    for call, requested in zip(view.calls, turn.calls):
        assert call.name == requested["tool_name"]
        assert json.loads(call.arguments) == requested["tool_input"]


# ---------------------------------------------------------------------------
# The emitted-field ledger
# ---------------------------------------------------------------------------

#: Provider-specific metadata that is not part of the turn's comparable
#: content.  A leaf classified ``provider_only`` must say why it is not
#: compared; the ledger test enforces the reason.
PROVIDER_ONLY_REASONS: dict[str, str] = {
    "annotations": (
        "whole-block annotations are provider-specific and body-only in Chat; "
        "streamed annotation deltas are not compared"
    ),
    "citations": (
        "whole-block citations are body-only; the stream carries citation "
        "deltas, which the IR models as their own event"
    ),
    "container": (
        "Anthropic container metadata is provider-specific and not part of "
        "the turn's content"
    ),
    "logprobs": (
        "per-token logprobs are body-only in Chat and have a different "
        "streamed shape; they are not part of the comparable content"
    ),
    "service_tier": "response-level service tier has no streamed counterpart",
    "signature": (
        "opaque reasoning signature; provider-specific and carried on "
        "different shapes per leg"
    ),
    "system_fingerprint": (
        "response-level system fingerprint has no streamed counterpart"
    ),
    "thoughtSignature": (
        "opaque Google signature; provider-specific and carried on different "
        "shapes per leg"
    ),
}

_ENVELOPE_LEAVES = frozenset(
    {
        "id",
        "object",
        "type",
        "model",
        "created",
        "created_at",
        "completed_at",
        "updated",
        "responseId",
        "modelVersion",
        "index",
        "sequence_number",
        "output_index",
        "content_index",
        "block_index",
        "item_id",
        "event_type",
        "role",
        "call_id",
    }
)
_TERMINAL_LEAVES = frozenset(
    {
        "finish_reason",
        "stop_reason",
        "stop_sequence",
        "stop_details",
        "incomplete_details",
        "error",
        "status",
        "finishReason",
    }
)
_CLASSIFICATIONS = frozenset(
    {"content", "envelope", "terminal", "usage", "provider_only"}
)


def _payload_paths(value: Any, prefix: str = "") -> set[str]:
    """Return every dotted field path in a JSON-like payload."""
    paths: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else key
            paths.add(path)
            paths |= _payload_paths(child, path)
    elif isinstance(value, list):
        for item in value:
            paths |= _payload_paths(item, prefix + "[]")
    return paths


def _classify_path(path: str) -> str:
    """Classify one emitted path for the ledger."""
    leaf = path.rsplit(".", 1)[-1].replace("[]", "")
    if leaf in _ENVELOPE_LEAVES:
        return "envelope"
    if leaf in _TERMINAL_LEAVES:
        return "terminal"
    if "usage" in path.lower() or leaf.lower().endswith(("token", "tokens")):
        return "usage"
    if leaf in PROVIDER_ONLY_REASONS:
        return "provider_only"
    return "content"


def _emitted_field_paths(converter: Any) -> tuple[set[str], set[str]]:
    """Return the body and stream field paths the fixture matrix emits."""
    body_paths: set[str] = set()
    stream_paths: set[str] = set()
    for _case_id, turn, _reason in CASE_TABLE:
        try:
            body_paths |= _payload_paths(
                converter.response_to_provider(_ir_response(turn))
            )
        except UnrepresentableFinishReason:
            pass
        context = converter.create_stream_context()
        chunks: list[dict[str, Any]] = []
        try:
            for event in _ir_events(turn):
                chunks.extend(
                    _as_dicts(
                        converter.stream_response_to_provider(event, context=context)
                    )
                )
        except UnrepresentableFinishReason:
            chunks = []
        stream_paths |= _payload_paths(chunks)
    # A stream is a list of chunks; its paths start with ``[].`` while a body
    # starts at its root object.  Normalise so both read from the root.
    stream_paths = {path.removeprefix("[].") for path in stream_paths}
    return body_paths, stream_paths


def build_field_ledger() -> dict[str, Any]:
    """Build the committed ledger; ``scripts/dev/regen_leg_agreement_fields.py``
    calls this."""
    dialects: dict[str, Any] = {}
    for dialect, factory in DIALECTS.items():
        body_paths, stream_paths = _emitted_field_paths(factory())
        dialects[dialect] = {
            "body": {path: _classify_path(path) for path in sorted(body_paths)},
            "stream": {path: _classify_path(path) for path in sorted(stream_paths)},
        }
    return {
        "schema_version": 1,
        "suite": "leg-agreement-fields",
        "source": "tests/converters/test_leg_agreement.py fixture matrix",
        "provider_only_reasons": dict(PROVIDER_ONLY_REASONS),
        "dialects": dialects,
    }


def _ledger_path() -> Path:
    return Path(__file__).with_name("leg_agreement_fields.json")


def _ledger() -> dict[str, Any]:
    return json.loads(_ledger_path().read_text())


@pytest.mark.parametrize(("dialect", "factory"), DIALECTS.items())
@pytest.mark.parametrize(("case_id", "turn", "capable_reason"), CASES)
def test_the_two_legs_answer_the_same_turn(
    dialect: str,
    factory: ConverterFactory,
    case_id: str,
    turn: Turn,
    capable_reason: str,
) -> None:
    """Same IR turn in, same content out, whichever leg wrote it."""
    if (dialect, case_id) in KNOWN_SHARED_LOSSES:
        pytest.skip(
            "pinned as a strict xfail below: " + KNOWN_SHARED_LOSSES[(dialect, case_id)]
        )

    converter = factory()
    body = _body_leg(converter, turn)
    stream = _stream_leg(converter, turn)

    assert body == stream, f"{dialect} legs disagree on {case_id}"

    if case_id in REFUSED_REASON.get(dialect, set()):
        assert isinstance(body, Refused), f"{dialect} should refuse {case_id}"
        assert body.reason == turn.finish
        return

    assert isinstance(body, TurnView), f"{dialect} should write {case_id}"
    expected_reason = EXPECTED_REASON.get((dialect, case_id), capable_reason)
    assert body.reason == expected_reason, f"{dialect} terminal reason on {case_id}"

    expected_kinds = EXPECTED_CALL_KINDS.get(
        (dialect, case_id), tuple("function" for _ in turn.calls)
    )
    assert tuple(call.kind for call in body.calls) == expected_kinds, (
        f"{dialect} call kinds on {case_id}"
    )
    if expected_kinds:
        _assert_call_contents(body, turn)

    if turn.text:
        assert body.text == turn.text, f"{dialect} text on {case_id}"
    if turn.reasoning:
        assert body.reasoning == turn.reasoning, f"{dialect} reasoning on {case_id}"
    if turn.usage is not None:
        assert body.usage == (
            turn.usage["prompt_tokens"],
            turn.usage["completion_tokens"],
            turn.usage["total_tokens"],
        ), f"{dialect} usage on {case_id}"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the Interactions writer maps a filtered turn to status completed, so "
        "both legs agree it finished normally (#1244)"
    ),
)
def test_the_known_interactions_shared_loss() -> None:
    """Pin the shared loss equality cannot see, so fixing it is visible."""
    body = _body_leg(
        DIALECTS["google_interactions"](), Turn(text="hi", finish="content_filter")
    )

    assert isinstance(body, TurnView)
    assert body.reason == "content_filter"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "the Messages stream writer drops finish_reason.stop_sequence; the body "
        "keeps it (#1245)"
    ),
)
def test_a_messages_stop_sequence_survives_both_legs() -> None:
    """Pin the value gap a path ledger cannot see."""
    turn = Turn(text="hi", finish="stop", stop_sequence="END")

    assert _stop_sequence(MessagesConverter(), turn) == ("END", "END")


def test_a_google_native_word_survives_both_legs() -> None:
    """A Google failure the IR cannot name keeps Google's own word on both legs."""
    converter = GoogleGenerateConverter()
    turn = Turn(
        text="hi", finish="error", native_finish_reason="MALFORMED_FUNCTION_CALL"
    )

    body = _body_leg(converter, turn)
    stream = _stream_leg(converter, turn)

    assert body == stream
    assert isinstance(body, TurnView)
    assert body.reason == "error"

    body_payload = converter.response_to_provider(cast(Any, _ir_response(turn)))
    assert body_payload["candidates"][0]["finishReason"] == "MALFORMED_FUNCTION_CALL"

    stream_reasons = [
        candidate["finishReason"]
        for chunk in _raw_stream_chunks(converter, turn)
        for candidate in chunk.get("candidates", [])
        if candidate.get("finishReason")
    ]
    assert stream_reasons == ["MALFORMED_FUNCTION_CALL"]


@pytest.mark.parametrize("dialect", DIALECTS)
def test_every_emitted_field_is_accounted_for(dialect: str) -> None:
    """A new or vanished field is a behaviour change, not a passing test."""
    ledger = _ledger()
    recorded = ledger["dialects"][dialect]
    body_paths, stream_paths = _emitted_field_paths(DIALECTS[dialect]())

    for leg, observed in (("body", body_paths), ("stream", stream_paths)):
        recorded_paths = set(recorded[leg])
        assert observed == recorded_paths, (
            f"{dialect} {leg} fields changed; regenerate "
            f"{_ledger_path().name} and review the diff.\n"
            f"new: {sorted(observed - recorded_paths)}\n"
            f"missing: {sorted(recorded_paths - observed)}"
        )
        for path, kind in recorded[leg].items():
            assert kind in _CLASSIFICATIONS, f"{dialect} {leg} {path}: {kind}"
            assert kind == _classify_path(path), (
                f"{dialect} {leg} {path}: {kind} != {_classify_path(path)}"
            )


def test_every_provider_only_field_says_why() -> None:
    """A field excluded from the comparison must carry a reason."""
    ledger = _ledger()
    reasons = ledger["provider_only_reasons"]
    for dialect, legs in ledger["dialects"].items():
        for leg, fields in legs.items():
            for path, kind in fields.items():
                if kind != "provider_only":
                    continue
                leaf = path.rsplit(".", 1)[-1].replace("[]", "")
                assert leaf in reasons, f"{dialect} {leg} {path} has no reason"
