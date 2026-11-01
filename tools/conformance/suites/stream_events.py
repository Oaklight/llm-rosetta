"""Stream-event coverage — one canonical IR stream through every dialect.

The body ledgers (:mod:`.fields`, :mod:`.request_fields`) record what each
format declares and what the conversion does with it.  A stream has no single
declared type to walk: a dialect's wire is a sequence of events, and what
matters is which IR events survive a trip through it.

So the unit here is a **sequence**, not an event.  A probe of 2026-10-31 fed
each IR event type to every writer on its own and read the result back; the
state-dependent ones (`finish`, `usage`, `stream_end`) produced nothing,
because they need an open response or a started block first.  Those results
were artifacts of the probe, not coverage gaps.

This suite writes one canonical IR stream — an opening, a reasoning block, a
text block with a refusal and a citation, a tool call with its result, a
finish and a usage — through every chat dialect, reads the wire back, and
records what came out:

- ``chunks`` — how many wire messages the writer produced;
- ``recovered`` — the IR event types the reader produced, in order.

A cell that changes fails the case: a type that reappears is an improvement
worth recording, and a type that disappears is a loss.  The ledger is the
reviewed statement of what each dialect carries today.

Run it with ``python -m tools.conformance run --suite stream-events``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..model import Probe, ProbeFailure

_BASELINE = Path(__file__).resolve().parents[1] / "baselines/stream-events.json"

#: One canonical IR stream.  Every IR event type appears once, in an order a
#: reader can follow: blocks open and close around their deltas, and the tool
#: call's delta follows its start.
SEQUENCE: tuple[dict[str, Any], ...] = (
    {"type": "stream_start", "response_id": "r1", "model": "m"},
    {"type": "content_block_start", "block_index": 0, "block_type": "reasoning"},
    {"type": "reasoning_delta", "reasoning": "think"},
    {"type": "content_block_end", "block_index": 0},
    {"type": "content_block_start", "block_index": 1, "block_type": "text"},
    {"type": "text_delta", "text": "hi"},
    {"type": "refusal_delta", "refusal": "no"},
    {
        "type": "citation_delta",
        "citation": {"type": "citation", "url": "https://x", "title": "t"},
    },
    {"type": "content_block_end", "block_index": 1},
    {"type": "tool_call_start", "tool_call_id": "call_1", "tool_name": "f"},
    {"type": "tool_call_delta", "tool_call_id": "call_1", "arguments_delta": '{"a":1}'},
    {"type": "tool_result", "tool_call_id": "call_1", "result": {"ok": True}},
    {"type": "finish", "finish_reason": {"reason": "stop"}},
    {
        "type": "usage",
        "usage": {"prompt_tokens": 1, "completion_tokens": 2, "total_tokens": 3},
    },
    {"type": "stream_end"},
)


def _dialects() -> dict[str, Any]:
    """Return the chat dialect converters, keyed by format name."""
    from llm_rosetta.converters.llm.chat_completions import ChatCompletionsConverter
    from llm_rosetta.converters.llm.google_generate import GoogleGenerateConverter
    from llm_rosetta.converters.llm.google_interactions import (
        GoogleInteractionsConverter,
    )
    from llm_rosetta.converters.llm.messages import MessagesConverter
    from llm_rosetta.converters.llm.open_responses import OpenResponsesConverter

    return {
        "chat_completions": ChatCompletionsConverter,
        "open_responses": OpenResponsesConverter,
        "messages": MessagesConverter,
        "google_generate": GoogleGenerateConverter,
        "google_interactions": GoogleInteractionsConverter,
    }


def observe(format_name: str) -> dict[str, Any]:
    """Round-trip the canonical stream through one dialect.

    Args:
        format_name: The dialect to drive.

    Returns:
        ``{"chunks": <int>, "recovered": [<IR event type>, ...]}``.
    """
    from llm_rosetta.converters.base.context import StreamContext

    converter = _dialects()[format_name]()

    write_context = StreamContext()
    chunks: list[dict[str, Any]] = []
    for event in SEQUENCE:
        produced = converter.stream_response_to_provider(
            json.loads(json.dumps(event)), context=write_context
        )
        items = produced if isinstance(produced, list) else [produced]
        chunks.extend(item for item in items if item)

    read_context = StreamContext()
    recovered: list[str] = []
    for chunk in chunks:
        events = converter.stream_response_from_provider(chunk, context=read_context)
        recovered.extend(str(event.get("type", "?")) for event in events)

    return {"chunks": len(chunks), "recovered": recovered}


def inventory() -> dict[str, dict[str, Any]]:
    """Return the observation for every dialect."""
    return {name: observe(name) for name in _dialects()}


def read_baseline() -> dict[str, Any]:
    """Read the committed ledger, or an empty one when it is absent."""
    if not _BASELINE.exists():
        return {}
    return json.loads(_BASELINE.read_text())


class _StreamEventsProbe:
    """One dialect's observed stream against the ledger."""

    def __init__(self, name: str, current: dict[str, Any], baseline: dict[str, Any]):
        self.name = name
        self._current = current
        self._baseline = baseline

    def run(self) -> dict[str, Any]:
        recorded = self._baseline.get("formats", {}).get(self.name)
        if recorded is None:
            raise ProbeFailure("format missing from the ledger; regenerate it")
        if self._current != recorded:
            added = sorted(set(self._current["recovered"]) - set(recorded["recovered"]))
            lost = sorted(set(recorded["recovered"]) - set(self._current["recovered"]))
            detail = []
            if self._current["chunks"] != recorded["chunks"]:
                detail.append(
                    f"chunk count {recorded['chunks']} -> {self._current['chunks']}"
                )
            if added:
                detail.append(f"recovered types added: {added}")
            if lost:
                detail.append(f"recovered types lost: {lost}")
            if not detail:
                detail.append("the recovered order changed")
            raise ProbeFailure(
                f"{self.name}: {'; '.join(detail)}; regenerate with write_baseline() "
                "and review the diff"
            )
        return {
            "chunks": self._current["chunks"],
            "recovered": len(self._current["recovered"]),
        }


class _BrokenProbe:
    name = "inventory"

    def __init__(self, error: Exception) -> None:
        self._error = error

    def run(self) -> None:
        raise self._error


def build() -> tuple[list[Probe], dict[str, Any]]:
    try:
        current = inventory()
    except Exception as exc:  # noqa: BLE001 - reported as an error case
        return [_BrokenProbe(exc)], {}
    baseline = read_baseline()
    probes: list[Probe] = [
        _StreamEventsProbe(name, data, baseline) for name, data in current.items()
    ]
    return probes, {"formats": sorted(current)}


def write_baseline() -> Path:
    """Regenerate the ledger from the current behaviour."""
    payload = {
        "schema_version": 1,
        "suite": "stream-events",
        "formats": inventory(),
    }
    _BASELINE.parent.mkdir(parents=True, exist_ok=True)
    _BASELINE.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return _BASELINE
