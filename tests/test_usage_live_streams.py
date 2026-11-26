"""Live streams through every reader, writer, fold and the gateway record (#1749).

The captures in ``tests/fixtures/usage_live_streams`` are real streams.  Whatever
dialect they come from and whatever dialect the client speaks, the gateway's
record, the fold of the IR events and the usage the client receives must be the
one total the upstream reported.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from llm_rosetta import get_converter_for_provider
from llm_rosetta.ir.primitives import fold_response_events
from llm_rosetta.pipeline.chat.streaming import StreamProcessor

_DIR = Path(__file__).parent / "fixtures" / "usage_live_streams"
_FORMATS = [
    "chat_completions",
    "messages",
    "open_responses",
    "google_generate",
    "google_interactions",
]

#: capture -> (upstream format, wrapped in ``{"data": ...}``, prompt, completion, total)
_CAPTURES = {
    "google_generate": ("google_generate", False, 16, 149, 466),
    "google_interactions": ("google_interactions", True, 16, 149, 511),
    "anthropic": ("messages", False, 24, 83, 107),
    "anthropic_thinking_tool": ("messages", False, 601, 130, 731),
    "openai_chat": ("chat_completions", False, 20, 536, 556),
}


def _chunks(name: str, wrapped: bool) -> list[dict[str, Any]]:
    chunks = []
    for line in (_DIR / f"{name}.sse").read_text(encoding="utf-8").splitlines():
        if line.startswith("data:"):
            data = line[5:].strip()
            if data and data != "[DONE]":
                body = json.loads(data)
                chunks.append({"data": body} if wrapped else body)
    return chunks


def _usages(node: Any, found: list[dict[str, Any]]) -> None:
    if isinstance(node, dict):
        for key, value in node.items():
            if key in ("usage", "usageMetadata", "usage_metadata") and isinstance(
                value, dict
            ):
                found.append(value)
            else:
                _usages(value, found)
    elif isinstance(node, list):
        for item in node:
            _usages(item, found)


def _client_triple(client: str, out: list[Any]) -> tuple[int, int, int | None]:
    """What the client was told: the last usage object it received."""
    found: list[dict[str, Any]] = []
    _usages(out, found)
    last = found[-1]
    if client == "chat_completions":
        return last["prompt_tokens"], last["completion_tokens"], last["total_tokens"]
    if client == "open_responses":
        return last["input_tokens"], last["output_tokens"], last["total_tokens"]
    if client == "google_generate":
        return (
            last["promptTokenCount"],
            last["candidatesTokenCount"],
            last["totalTokenCount"],
        )
    if client == "google_interactions":
        return (
            last["total_input_tokens"],
            last["total_output_tokens"],
            last["total_tokens"],
        )
    return last["input_tokens"], last["output_tokens"], None  # messages: no total


def _run(upstream: str, client: str, chunks: list[dict[str, Any]]):
    target = get_converter_for_provider(upstream)
    source = get_converter_for_provider(client)
    ir_events: list[dict[str, Any]] = []
    processor = StreamProcessor(
        target_converter=target,
        source_converter=source,
        from_ctx=target.create_stream_context(),
        to_ctx=source.create_stream_context(),
        on_ir_event=ir_events.append,
    )
    out: list[Any] = []
    for chunk in chunks:
        out += processor.process_chunk(copy.deepcopy(chunk))
    out += processor.process_chunk(None)
    return processor.get_accumulated_usage(), ir_events, out


_PAIRS = [
    (name, client)
    for name, (upstream, *_rest) in _CAPTURES.items()
    for client in _FORMATS
    if client != upstream
]


@pytest.mark.parametrize(("name", "client"), _PAIRS)
def test_record_fold_and_client_agree_on_the_captured_total(name, client):
    upstream, wrapped, prompt, completion, total = _CAPTURES[name]
    acc, ir_events, out = _run(upstream, client, _chunks(name, wrapped))

    assert acc is not None
    assert (acc["prompt_tokens"], acc["completion_tokens"], acc["total_tokens"]) == (
        prompt,
        completion,
        total,
    )
    folded: Any = fold_response_events(copy.deepcopy(ir_events))
    assert (
        folded["usage"]["prompt_tokens"],
        folded["usage"]["completion_tokens"],
        folded["usage"]["total_tokens"],
    ) == (prompt, completion, total)

    got_prompt, got_completion, got_total = _client_triple(client, out)
    assert (got_prompt, got_completion) == (prompt, completion)
    if got_total is not None:
        assert got_total == total


def test_the_zeros_a_wire_reports_stay_reported():
    """OpenAI's final usage states ``cached_tokens: 0`` and zero detail fields; a
    first snapshot passes whole, so the record and the client keep them."""
    upstream, wrapped, *_ = _CAPTURES["openai_chat"]
    acc, _events, out = _run(
        upstream, "google_generate", _chunks("openai_chat", wrapped)
    )
    assert acc is not None
    assert acc["cache_read_tokens"] == 0
    assert acc["prompt_tokens_details"]["cached_tokens"] == 0
    assert acc["completion_tokens_details"]["rejected_prediction_tokens"] == 0
    found: list[dict[str, Any]] = []
    _usages(out, found)
    assert found[-1]["cachedContentTokenCount"] == 0


def test_google_into_interactions_is_not_overwritten_by_the_last_event():
    """The Interactions writer adds usage events (#1750): the client sees the
    total, not the last increment (18 on the live capture)."""
    upstream, wrapped, prompt, completion, total = _CAPTURES["google_generate"]
    _acc, _events, out = _run(
        upstream, "google_interactions", _chunks("google_generate", wrapped)
    )
    assert _client_triple("google_interactions", out) == (prompt, completion, total)
