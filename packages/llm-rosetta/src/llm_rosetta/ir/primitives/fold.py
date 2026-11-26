"""A stream of IR events, folded back into the response it renders.

:func:`~llm_rosetta.ir.primitives.expand_response` renders a whole response as
the events a stream of it would emit.  This module is the inverse, and it
exists for one caller shape: an upstream that refuses ``stream: false``
(``connection.stream_required`` on a shim) must answer a client that asked for
one body, so the events its stream produced have to become that body again.

Why the IR and not the source wire: the readers already fold a provider's
events into the IR, and every writer already folds an IR response back to its
own dialect.  Folding here means one reassembly serves every source format and
every target.

What the fold cannot recover is what no event carries.  A body keeps a turn's
``logprobs``, ``service_tier`` and ``system_fingerprint``; the event vocabulary
has no event for them, which is exactly the split ``expand_response`` documents.
The writer supplies its own defaults for those, so a synthesized body omits
them rather than inventing them.

The one place a fold has to interpret rather than copy: ``content_block_start``
announces a block, and a delta addresses it by ``block_index``.  A dialect that
numbers its blocks (Anthropic, Responses) always sends the index, so the fold
follows it.  A dialect that does not (Chat Completions) sends bare deltas, and
the fold keeps the block that is currently open when the delta's kind matches
it — the same rule the writers use in reverse.

Response-level echo fields — ``service_tier``, ``system_fingerprint`` — are the
exception the caller can close: no event carries them, but a dialect's chunks
repeat them, so a caller holding the source stream passes what the first chunk
said and the fold puts it back (``FormatSpec.stream_echo_fields`` names them per
dialect).  A caller with nothing to pass gets a body without them, which is what
a body that never carried them looks like.
"""

from __future__ import annotations

import json
from typing import Any, cast
from collections.abc import Callable

from llm_rosetta.conversion_warnings import report_loss
from llm_rosetta.ir.tools.custom_input import custom_input_from_text
from llm_rosetta.ir.context import add_usage
from llm_rosetta.ir.model import IRResponse, UsageInfo

__all__ = ["fold_response_events"]


#: Block types that own a content part, and the empty part each accumulates
#: into.  A ``tool_use`` block is absent on purpose: its call arrives as a
#: ``tool_call_start`` / ``tool_call_delta`` pair, so the block announces a
#: place in the numbering and nothing else.
_BLOCK_PARTS: dict[str, dict[str, Any]] = {
    "text": {"type": "text", "text": ""},
    "thinking": {"type": "reasoning", "reasoning": ""},
    "refusal": {"type": "refusal", "refusal": ""},
}

#: The content field each delta kind appends to, and the block kind it belongs
#: to.  Together they let one handler serve all three delta events.
_DELTA_FIELDS: dict[str, str] = {
    "text_delta": "text",
    "reasoning_delta": "reasoning",
    "refusal_delta": "refusal",
}
_DELTA_BLOCKS: dict[str, str] = {
    "text_delta": "text",
    "reasoning_delta": "thinking",
    "refusal_delta": "refusal",
}

#: Optional fields a call or a result part copies from its event verbatim.
_CALL_FIELDS = ("tool_type", "tool_group", "intrinsic_kind", "cache_hint")
_RESULT_FIELDS = ("tool_type", "is_error", "intrinsic_kind", "cache_hint")

#: Response-level fields a caller may hand in, because their own stream's
#: chunks repeat them even though no IR event does.  The IR response spells
#: them (:class:`~llm_rosetta.ir.model.IRResponse`), so a seed that carries one
#: is not an invention — it is the same fact the source body would have had.
_ECHO_FIELDS = ("service_tier", "system_fingerprint")


def _metadata(event: dict[str, Any]) -> dict[str, Any] | None:
    """Return the event's ``provider_metadata`` when it carries a non-empty one."""
    metadata = event.get("provider_metadata")
    return dict(metadata) if isinstance(metadata, dict) and metadata else None


def _int_or_none(value: Any) -> int | None:
    """Read an event's ``block_index``, which a dialect may omit or null."""
    return value if isinstance(value, int) else None


class _Choice:
    """One choice being assembled, with its open-block bookkeeping."""

    __slots__ = (
        "blocks",
        "finish",
        "index",
        "message_metadata",
        "next_block",
        "open_block",
        "parts",
    )

    def __init__(self, index: int) -> None:
        self.index = index
        self.parts: list[dict[str, Any]] = []
        #: ``block_index`` → position in :attr:`parts`.
        self.blocks: dict[int, int] = {}
        self.open_block: int | None = None
        self.next_block = 0
        self.finish: dict[str, Any] = {"reason": "stop"}
        self.message_metadata: dict[str, Any] | None = None

    def block_part(self, block_index: int, block_type: str) -> dict[str, Any] | None:
        """Return the part at *block_index*, creating it when its start was lost.

        A reader can drop a block's start — a dialect that omits it, or a
        stream that began mid-turn — and the deltas that follow still name the
        block.  Creating it on demand keeps those deltas.
        """
        existing = self.blocks.get(block_index)
        if existing is not None:
            return self.parts[existing]
        template = _BLOCK_PARTS.get(block_type)
        if template is None:
            return None
        part = dict(template)
        self.parts.append(part)
        self.blocks[block_index] = len(self.parts) - 1
        self.next_block = max(self.next_block, block_index + 1)
        return part

    def delta_part(
        self, block_index: int | None, block_type: str
    ) -> dict[str, Any] | None:
        """Return the part a delta of *block_type* belongs to.

        A delta that names its block follows that index.  A delta that names
        none joins the open block when the kinds agree, and opens a new block
        when they do not — the reverse of how the writers close a block before
        switching kinds.
        """
        if block_index is not None:
            part = self.block_part(block_index, block_type)
            if part is not None:
                self.open_block = block_index
            return part

        current = self._part_of_open_block(block_type)
        if current is not None:
            return current

        index = self.next_block
        part = self.block_part(index, block_type)
        if part is not None:
            self.open_block = index
        return part

    def _part_of_open_block(self, block_type: str) -> dict[str, Any] | None:
        """The open block's part when it is one a delta of *block_type* extends."""
        if self.open_block is None:
            return None
        position = self.blocks.get(self.open_block)
        if position is None:
            return None
        current = self.parts[position]
        expected = _BLOCK_PARTS.get(block_type) or {}
        return current if current.get("type") == expected.get("type") else None


class _Folder:
    """The fold's state, and one method per event it understands."""

    def __init__(self, response: dict[str, Any] | None) -> None:
        seed = dict(response or {})
        self.folded: dict[str, Any] = {
            "id": seed.get("id", ""),
            "object": "response",
            "created": seed.get("created", 0),
            "model": seed.get("model", ""),
            "choices": [],
        }
        for key in _ECHO_FIELDS:
            if seed.get(key) is not None:
                self.folded[key] = seed[key]
        self.choices: dict[int, _Choice] = {}
        self.usage: dict[str, Any] | None = None
        self.passthrough: list[dict[str, Any]] = []
        #: Every tool call being assembled, keyed by ``tool_call_id``.
        self.calls: dict[str, dict[str, Any]] = {}
        #: ``tool_call_index`` → the key in :attr:`calls`, for a dialect whose
        #: deltas name the position instead of the id.
        self.calls_by_index: dict[int, str] = {}
        self.last_call_id: str | None = None

    # -- entry ---------------------------------------------------------

    def feed(self, event: dict[str, Any]) -> None:
        """Fold one event; an event with no rule is skipped."""
        handler = _HANDLERS.get(str(event.get("type", "")))
        if handler is not None:
            handler(self, event)

    def finish(self) -> IRResponse:
        """Resolve the tool arguments and assemble the response."""
        for entry in self.calls.values():
            _resolve_arguments(entry)
        self.folded["choices"] = [
            self._choice_body(self.choices[position])
            for position in sorted(self.choices)
        ]
        if self.usage:
            self.folded["usage"] = cast(UsageInfo, self.usage)
        if self.passthrough:
            self.folded["provider_passthrough_items"] = self.passthrough
        return cast(IRResponse, self.folded)

    @staticmethod
    def _choice_body(choice: _Choice) -> dict[str, Any]:
        """Render one assembled choice as the body's choice entry."""
        message: dict[str, Any] = {
            "role": "assistant",
            "content": choice.parts,
        }
        if choice.message_metadata:
            message["provider_metadata"] = choice.message_metadata
        return {
            "index": choice.index,
            "message": message,
            "finish_reason": choice.finish,
        }

    def choice_for(self, event: dict[str, Any]) -> _Choice:
        """Return (and remember) the choice the event belongs to."""
        index = event.get("choice_index")
        key = index if isinstance(index, int) else 0
        if key not in self.choices:
            self.choices[key] = _Choice(key)
        return self.choices[key]

    # -- lifecycle ------------------------------------------------------

    def start(self, event: dict[str, Any]) -> None:
        """``stream_start`` names the response the events belong to."""
        self.folded["id"] = event.get("response_id") or self.folded["id"]
        self.folded["model"] = event.get("model") or self.folded["model"]
        if event.get("created"):
            self.folded["created"] = event["created"]

    def block_start(self, event: dict[str, Any]) -> None:
        """``content_block_start`` opens a block a later delta addresses."""
        choice = self.choice_for(event)
        block_index = event.get("block_index", choice.next_block)
        part = choice.block_part(block_index, str(event.get("block_type", "")))
        if part is None:
            # A tool block, or one this vocabulary has no part for.  It is not
            # a delta target, so a following bare text delta must not join it.
            choice.open_block = None
            return
        metadata = _metadata(event)
        if metadata:
            part["provider_metadata"] = metadata
        choice.open_block = block_index

    def block_end(self, event: dict[str, Any]) -> None:
        """``content_block_end`` closes the block it names."""
        choice = self.choice_for(event)
        if choice.open_block == event.get("block_index"):
            choice.open_block = None

    # -- content --------------------------------------------------------

    def delta(self, event: dict[str, Any]) -> None:
        """Append one text, reasoning or refusal fragment to its block."""
        kind = str(event.get("type"))
        field = _DELTA_FIELDS[kind]
        part = self.choice_for(event).delta_part(
            _int_or_none(event.get("block_index")), _DELTA_BLOCKS[kind]
        )
        if part is None:
            return
        part[field] = part.get(field, "") + str(event.get(field, ""))
        if kind == "reasoning_delta":
            _reasoning_extras(part, event)

    def citation(self, event: dict[str, Any]) -> None:
        """A citation is a part of its own, in the order the reader emitted it."""
        citation = event.get("citation")
        if isinstance(citation, dict):
            self.choice_for(event).parts.append(dict(citation))

    # -- tools ----------------------------------------------------------

    def call_start(self, event: dict[str, Any]) -> None:
        """``tool_call_start`` opens a call the deltas fill in."""
        choice = self.choice_for(event)
        call_id = str(event.get("tool_call_id") or "")
        part: dict[str, Any] = {
            "type": "tool_call",
            "tool_call_id": call_id,
            "tool_name": str(event.get("tool_name") or ""),
            "tool_input": {},
        }
        for key in _CALL_FIELDS:
            if event.get(key) is not None:
                part[key] = event[key]
        metadata = _metadata(event)
        if metadata:
            part["provider_metadata"] = metadata
        choice.parts.append(part)

        key = call_id or f"__index_{event.get('tool_call_index')}"
        self.calls[key] = {"part": part, "args": ""}
        position = event.get("tool_call_index")
        if isinstance(position, int):
            self.calls_by_index[position] = key
        self.last_call_id = key
        # A tool call is its own thing, not a text or reasoning block.
        choice.open_block = None

    def call_delta(self, event: dict[str, Any]) -> None:
        """Append an argument fragment to the call it belongs to."""
        entry = self.calls.get(self._call_key(event))
        if entry is None:
            return
        entry["args"] += str(event.get("arguments_delta") or "")
        part = entry["part"]
        if event.get("intrinsic_kind") and not part.get("intrinsic_kind"):
            part["intrinsic_kind"] = event["intrinsic_kind"]
        metadata = _metadata(event)
        if metadata:
            part["provider_metadata"] = {
                **(part.get("provider_metadata") or {}),
                **metadata,
            }

    def _call_key(self, event: dict[str, Any]) -> str:
        """Resolve which call a delta belongs to.

        The id is authoritative when the dialect sends one.  Otherwise the
        index names it, and failing that the call most recently started is the
        only one the fragment can belong to.
        """
        key = str(event.get("tool_call_id") or "")
        if key in self.calls:
            return key
        position = event.get("tool_call_index")
        return self.calls_by_index.get(
            position if isinstance(position, int) else -1, self.last_call_id or ""
        )

    def tool_result(self, event: dict[str, Any]) -> None:
        """A provider-executed result arrives whole, so it is one part."""
        part: dict[str, Any] = {
            "type": "tool_result",
            "tool_call_id": str(event.get("tool_call_id") or ""),
            "result": event.get("result"),
        }
        for key in _RESULT_FIELDS:
            if event.get(key) is not None:
                part[key] = event[key]
        metadata = _metadata(event)
        if metadata:
            part["provider_metadata"] = metadata
        self.choice_for(event).parts.append(part)

    # -- turn -----------------------------------------------------------

    def finish_turn(self, event: dict[str, Any]) -> None:
        """``finish`` carries the choice's terminal reason and message facts.

        Some readers carry a refusal's words on the finish rather than in a
        content block: the Chat reader accumulates ``delta.refusal`` on its
        context and only the finish event holds the text.  A body spells a
        refusal as a part, so the fold writes one when nothing opened a block
        for it — otherwise the words are dropped and the terminal reason with
        them.
        """
        choice = self.choice_for(event)
        reason = event.get("finish_reason")
        if isinstance(reason, dict):
            choice.finish = dict(reason)
        metadata = _metadata(event)
        if metadata:
            choice.message_metadata = metadata
            _refusal_from_finish(choice, metadata)

    def usage_event(self, event: dict[str, Any]) -> None:
        """``usage`` is cumulative: a later report refines the earlier one."""
        self.usage = _merge_usage(self.usage, event.get("usage"))

    def passthrough_event(self, event: dict[str, Any]) -> None:
        """A dialect-private event rides the response's opaque item list."""
        payload = event.get("payload")
        if isinstance(payload, dict):
            self.passthrough.append(
                {
                    "type": "provider_passthrough_item",
                    "provider": str(event.get("provider") or ""),
                    "payload": dict(payload),
                }
            )


def _refusal_from_finish(choice: _Choice, metadata: dict[str, Any]) -> None:
    """Turn ``provider_metadata.refusal_text`` into the part a body keeps it in.

    Only when the choice has no refusal part already: a dialect that streamed
    ``refusal_delta`` events has one, and its text is the authoritative
    spelling of the same fact.
    """
    text = metadata.get("refusal_text")
    if not isinstance(text, str) or not text:
        return
    if any(part.get("type") == "refusal" for part in choice.parts):
        return
    choice.parts.append({"type": "refusal", "refusal": text})


def _reasoning_extras(part: dict[str, Any], event: dict[str, Any]) -> None:
    """Carry the opaque and descriptive fields a reasoning delta may hold."""
    signature = event.get("signature")
    if signature:
        part["signature"] = part.get("signature", "") + str(signature)
    encrypted = event.get("encrypted_content")
    if encrypted:
        # The part keeps the whole opaque blob, not a fragment: the writers
        # read it as one value.
        part["redacted_data"] = encrypted
    incoming = event.get("provider_metadata")
    if isinstance(incoming, dict) and incoming:
        part["provider_metadata"] = {
            **(part.get("provider_metadata") or {}),
            **incoming,
        }


def _resolve_arguments(entry: dict[str, Any]) -> None:
    """Turn one accumulated argument string into the call's ``tool_input``."""
    part = entry["part"]
    text = entry["args"]
    if part.get("tool_type") == "custom":
        # A custom call's deltas are its plain text, however it parses.
        part["tool_input"] = custom_input_from_text(text)
        return
    if not text.strip():
        return
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        # A fragment that never closed is a call the client cannot run.
        # Keeping the raw text is what the Chat reader does with the same
        # shape, and it names the loss rather than passing it off as input.
        report_loss(
            "A streamed tool call's arguments did not parse as JSON; the "
            "raw text is kept under 'raw_arguments'",
            subject=str(part.get("tool_name") or ""),
            leg="response",
        )
        part["tool_input"] = {"raw_arguments": text}
        return
    part["tool_input"] = parsed if isinstance(parsed, dict) else {}


def _merge_usage(accumulated: dict[str, Any] | None, raw: Any) -> dict[str, Any]:
    """Add one usage event to the running usage.

    Usage events are increments (see
    :class:`~llm_rosetta.ir.model.stream.UsageEvent`), so they add; a zero or a
    missing field changes nothing.
    """
    if not isinstance(raw, dict):
        return dict(accumulated or {})
    return add_usage(accumulated, raw)


#: Event type → the method that folds it.  A type absent here is one this
#: vocabulary has no part to put in the response (``stream_end``,
#: ``response.in_progress``), and skipping it is the rule.
_HANDLERS: dict[str, Callable[[_Folder, dict[str, Any]], None]] = {
    "stream_start": _Folder.start,
    "content_block_start": _Folder.block_start,
    "content_block_end": _Folder.block_end,
    "text_delta": _Folder.delta,
    "reasoning_delta": _Folder.delta,
    "refusal_delta": _Folder.delta,
    "citation_delta": _Folder.citation,
    "tool_call_start": _Folder.call_start,
    "tool_call_delta": _Folder.call_delta,
    "tool_result": _Folder.tool_result,
    "finish": _Folder.finish_turn,
    "usage": _Folder.usage_event,
    "provider_passthrough": _Folder.passthrough_event,
}


def fold_response_events(
    events: list[Any],
    *,
    response: dict[str, Any] | None = None,
) -> IRResponse:
    """Fold IR stream events into the one response a stream of them describes.

    Args:
        events: IR stream events, in the order the reader emitted them.
        response: Envelope facts the caller already knows — ``id``, ``model``,
            ``created``.  A reader publishes them on ``stream_start``; a caller
            that kept its own stream context can still pass them, so a stream
            whose first event was dropped does not lose its identity.

    Returns:
        The IR response.  Choices appear in ascending index order, each
        carrying the parts its blocks accumulated and the terminal reason its
        ``finish`` event named.  A stream that ended without a ``finish``
        event leaves the reason at ``stop``.

    Raises:
        Nothing for a malformed event: an event this fold has no rule for is
        skipped, because a stream may carry dialect-private events its own
        reader already reduced to nothing.
    """
    folder = _Folder(response)
    for event in events:
        if isinstance(event, dict):
            folder.feed(event)
    return folder.finish()
