"""A whole response, rendered as the event sequence a stream of it would emit.

Every dialect writes a turn twice: once as a body, once as a stream.  They are
two views of one content model (epic #1238), but each dialect derived that
model again in its body writer, so a rule fixed on one leg kept answering on
the other.  The readers fold a provider's events into the IR; this module
un-folds an :class:`IRResponse` back into the IR event vocabulary, so a body
writer can be a fold over the same per-event code its stream writer already
uses.

What it emits is the content model, not the envelope.  A fact that exists only
in a whole body — Chat's ``logprobs``, ``service_tier``, ``system_fingerprint``
and ``object``, Messages' ``citations: null`` on a text block, Google's usage
detail arrays, Interactions' ``created`` / ``updated`` — has no event to ride
on and stays the fold's tail.

Block boundaries are emitted for the content that carries deltas (text,
reasoning, refusal), because a block is where a dialect keeps the item identity
a body keeps on the part.  A tool call is its own ``tool_call_start`` /
``tool_call_delta`` pair, and a server tool result is one ``tool_result``
event.  A citation is a ``citation_delta`` on the block of the *nearest
preceding text*, not of the part immediately before it: Google appends its
citations after the answer's last part, so a turn that reads text, tool call,
citation still annotates the text.  When no text precedes it there is no block
to name, and the event carries none.

The message's ``provider_metadata`` rides on the choice's ``finish`` event: a
body keeps the facts about the turn on the message, and a stream keeps them on
the event that ends it.  That is the same correspondence the cross-leg harness
pins (``tests/converters/test_leg_agreement.py``), which is also this module's
first witness.

Multi-choice: ``choice_index`` is written on the events that carry it — the
deltas, ``tool_call_start``, ``finish`` — only when a response has more than one
choice, the IR's convention being that an absent index means the only one.
Block boundaries deliberately carry none: no reader produces a choice index on
one, and a field nothing reads is a field with no meaning.  A fold that needs
to attribute a block to a choice can read the deltas inside it.
"""

from __future__ import annotations

import json
from typing import Any, cast

from llm_rosetta.types.ir import (
    IRResponse,
    IRStreamEvent,
    UsageInfo,
    is_citation_part,
    is_refusal_part,
    is_reasoning_part,
    is_text_part,
    is_tool_call_part,
    is_tool_result_part,
)

__all__ = ["expand_response"]


def expand_response(ir_response: IRResponse) -> list[IRStreamEvent]:
    """Render *ir_response* as the events a stream of that turn would emit.

    Choices are expanded in order and block indices run across them.
    ``choice_index`` is written on the events that declare the field, and only
    when the response has more than one choice.

    Args:
        ir_response: The response body to expand.

    Returns:
        The events, always ending with ``stream_end``.  A response with no
        choices still yields ``stream_start`` and ``stream_end``, plus its
        usage when it carries one.  No event dict is the response's own dict;
        the values inside them are the response's own objects, as parts are
        everywhere else in the IR, so a caller that mutates a nested value
        mutates the response.
    """
    choices = ir_response.get("choices") or []
    multi = len(choices) > 1

    events: list[IRStreamEvent] = [
        {
            "type": "stream_start",
            "response_id": ir_response.get("id", ""),
            "model": ir_response.get("model", ""),
            "created": ir_response.get("created", 0),
        }
    ]
    finishes: list[IRStreamEvent] = []
    block_index = 0

    for position, choice in enumerate(choices):
        # The choice's own index when it declares one: a fold back into a
        # body renumbers with `enumerate` otherwise, and the two must agree.
        choice_index = choice.get("index", position)
        choice_events, block_index = _expand_choice(
            choice, choice_index, multi, block_index
        )
        events.extend(choice_events)
        finishes.append(_finish_event(choice, choice_index, multi))

    if ir_response.get("usage"):
        usage_event: IRStreamEvent = {
            "type": "usage",
            "usage": cast(UsageInfo, dict(ir_response["usage"])),
        }
        events.append(usage_event)
    events.extend(finishes)
    events.append({"type": "stream_end"})
    return events


def _expand_choice(
    choice: Any, choice_index: int, multi: bool, block_index: int
) -> tuple[list[IRStreamEvent], int]:
    """Expand one choice's parts, starting at *block_index*.

    Returns:
        The events and the next free block index.
    """
    events: list[IRStreamEvent] = []
    # The nearest preceding text block, which is what a citation annotates.
    # It is not reset by the parts in between: Google appends its citations
    # after the answer's last part, so text → call → citation still names the
    # text block the citation belongs to.
    text_block: int | None = None
    # A tool call has no index of its own in the IR — its position among the
    # choice's calls is the only identity — and the Chat writer indexes deltas
    # by it, so two parallel calls announced at the same index would have
    # their arguments folded into one.
    call_index = 0

    for part in (choice.get("message") or {}).get("content") or []:
        metadata = part.get("provider_metadata")

        if is_text_part(part):
            events.append(_block_start(block_index, "text", metadata))
            events.append(
                _tagged(
                    {
                        "type": "text_delta",
                        "text": part.get("text", ""),
                        "block_index": block_index,
                    },
                    choice_index,
                    multi,
                )
            )
            events.append(_block_end(block_index))
            text_block = block_index
            block_index += 1
        elif is_reasoning_part(part):
            events.extend(_reasoning_events(part, block_index, choice_index, multi))
            block_index += 1
        elif is_refusal_part(part):
            events.append(_block_start(block_index, "refusal", metadata))
            events.append(
                _tagged(
                    {
                        "type": "refusal_delta",
                        "refusal": part.get("refusal", ""),
                        "block_index": block_index,
                    },
                    choice_index,
                    multi,
                )
            )
            events.append(_block_end(block_index))
            block_index += 1
        elif is_citation_part(part):
            events.append(
                _citation_event(part, text_block, metadata, choice_index, multi)
            )
        elif is_tool_call_part(part):
            events.extend(_tool_call_events(part, call_index, choice_index, multi))
            call_index += 1
        elif is_tool_result_part(part):
            events.append(_tool_result_event(part))

    return events, block_index


def _citation_event(
    part: Any,
    text_block: int | None,
    metadata: Any,
    choice_index: int,
    multi: bool,
) -> IRStreamEvent:
    """One citation, on the block of the text it annotates.

    The part's own ``provider_metadata`` rides the event whether or not there
    is a block to name: it describes the citation, not the text, and a dialect
    that keeps a citation's identity there needs it on every citation event.
    """
    event: dict[str, Any] = {"type": "citation_delta", "citation": dict(part)}
    if text_block is not None:
        event["block_index"] = text_block
    if metadata:
        event["provider_metadata"] = dict(metadata)
    return _tagged(event, choice_index, multi)


def _finish_event(choice: Any, choice_index: int, multi: bool) -> IRStreamEvent:
    """The event that ends one choice, carrying the message's own facts."""
    finish: dict[str, Any] = {
        "type": "finish",
        "finish_reason": dict(choice.get("finish_reason") or {"reason": "stop"}),
    }
    metadata = (choice.get("message") or {}).get("provider_metadata")
    if metadata:
        finish["provider_metadata"] = dict(metadata)
    return _tagged(finish, choice_index, multi)


def _reasoning_events(
    part: Any, block_index: int, choice_index: int, multi: bool
) -> list[IRStreamEvent]:
    """One reasoning block: a start, a delta, an end."""
    metadata = part.get("provider_metadata")
    delta: dict[str, Any] = {
        "type": "reasoning_delta",
        "reasoning": part.get("reasoning", ""),
        "block_index": block_index,
    }
    if part.get("signature"):
        delta["signature"] = part["signature"]
    if part.get("redacted_data"):
        # The stream spelling of the same opaque blob: Anthropic's
        # redacted_thinking and OpenAI's encrypted_content are one field on
        # the wire side.
        delta["encrypted_content"] = part["redacted_data"]
    if metadata:
        delta["provider_metadata"] = dict(metadata)
    return [
        _block_start(block_index, "thinking", metadata),
        _tagged(delta, choice_index, multi),
        _block_end(block_index),
    ]


def _tool_call_events(
    part: Any, tool_call_index: int, choice_index: int, multi: bool
) -> list[IRStreamEvent]:
    """The start/delta pair that carries one tool call."""
    start: dict[str, Any] = {
        "type": "tool_call_start",
        "tool_call_id": part["tool_call_id"],
        "tool_name": part["tool_name"],
        "tool_type": part.get("tool_type", "function"),
        "tool_call_index": tool_call_index,
    }
    for key in ("tool_group", "intrinsic_kind", "cache_hint"):
        if part.get(key):
            start[key] = part[key]
    if part.get("provider_metadata"):
        start["provider_metadata"] = dict(part["provider_metadata"])

    delta: dict[str, Any] = {
        "type": "tool_call_delta",
        "tool_call_id": part["tool_call_id"],
        "arguments_delta": json.dumps(part.get("tool_input") or {}, ensure_ascii=False),
        "tool_call_index": tool_call_index,
    }
    if part.get("intrinsic_kind"):
        # The kind rides every event of a provider-executed call, not only the
        # start: a reader that sees the fragment alone still knows the tool.
        delta["intrinsic_kind"] = part["intrinsic_kind"]
    return [_tagged(start, choice_index, multi), _tagged(delta, choice_index, multi)]


def _tool_result_event(part: Any) -> IRStreamEvent:
    """One provider-executed tool result, delivered whole."""
    event: dict[str, Any] = {
        "type": "tool_result",
        "tool_call_id": part["tool_call_id"],
        "result": part.get("result"),
    }
    for key in ("tool_type", "is_error", "intrinsic_kind", "cache_hint"):
        if part.get(key) is not None:
            event[key] = part[key]
    if part.get("provider_metadata"):
        event["provider_metadata"] = dict(part["provider_metadata"])
    return cast(IRStreamEvent, event)


def _tagged(event: dict[str, Any], choice_index: int, multi: bool) -> Any:
    """Add ``choice_index`` when the response has more than one choice."""
    if multi:
        event["choice_index"] = choice_index
    return event


def _block_start(block_index: int, block_type: str, metadata: Any) -> Any:
    event: dict[str, Any] = {
        "type": "content_block_start",
        "block_index": block_index,
        "block_type": block_type,
    }
    if isinstance(metadata, dict) and metadata:
        event["provider_metadata"] = dict(metadata)
    return event


def _block_end(block_index: int) -> Any:
    return {"type": "content_block_end", "block_index": block_index}
