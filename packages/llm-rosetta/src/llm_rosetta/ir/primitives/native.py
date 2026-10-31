"""One record for what a provider said in its own words.

The IR has one finish reason and one shape for a blocked turn; a provider
distinguishes several.  A word the mapping would replace with a coarser IR
reason — Google's ``MALFORMED_FUNCTION_CALL``, its ``promptFeedback`` — is
kept verbatim in ``provider_metadata`` so a same-format write can hand it
back instead of the IR's approximation.

That record used to be two keys with two shapes: ``_native_finish_reason``
holding ``{"base": ..., "reason": ...}`` and ``_native_prompt_feedback``
holding ``{"base": ..., "prompt_feedback": ...}``.  Neither said what it was
beyond its key, both repeated the origin, and a third provider fact would
have added a third key.  There is now one record::

    provider_metadata["_native_response"] = {
        "origin": "google_generate",
        "finish_reason": "MALFORMED_FUNCTION_CALL",   # when one was recorded
        "prompt_feedback": {...},                     # when one was recorded
    }

The key is ``_native_response``, not ``_native``: ``_native`` already holds a
provider's native *part* payload in three dialects (``{"base": ...,
"part"|"block"|"item": ...}``), and one key cannot mean two shapes at once.
The part carriers are content a writer replays; this record is what the
provider said about the turn.  Folding them together is P6's job, when the
writers thread one carrier instead of three.

Old records stay readable: :func:`native_record` folds both legacy spellings
into the new shape, field by field, so a value produced by an earlier release
converts the same as one produced today.  A write consumes the legacy keys it
folded — the dict leaves with one spelling, not two.

One holder, one dialect: the key holds a single record, so a write by one
dialect replaces a record another dialect left on the same dict, while
:func:`native_record` refuses to read that foreign record.  The asymmetry is
deliberate and currently unreachable — a record is written by the reader that
produced the turn, so a dict carries at most one dialect's — and a
per-origin shape is a decision for P6, where the carriers are merged.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "NATIVE_RESPONSE_KEY",
    "LEGACY_FINISH_REASON_KEY",
    "LEGACY_PROMPT_FEEDBACK_KEY",
    "native_record",
    "record_native",
]

#: Where the record lives on ``provider_metadata``.
NATIVE_RESPONSE_KEY = "_native_response"
#: The pre-#1238 key for a provider-spelled finish reason.
LEGACY_FINISH_REASON_KEY = "_native_finish_reason"
#: The pre-#1238 key for a provider's blocked-prompt field.
LEGACY_PROMPT_FEEDBACK_KEY = "_native_prompt_feedback"


def native_record(metadata: Any, *, origin: str) -> dict[str, Any]:
    """Return *origin*'s native record in ``metadata``, legacy shapes included.

    The two sources are folded field by field, and the current record wins a
    field it holds.  A record written today that carries only a prompt
    feedback does not hide a legacy finish reason recorded beside it.

    Args:
        metadata: A message's or a stream event's ``provider_metadata``.  Any
            value is accepted; a non-mapping yields an empty record.
        origin: The converter tag the record must belong to.  A record written
            by another dialect is not this dialect's to read — the origin is
            what stops a foreign turn from making this provider speak a word
            it never said.

    Returns:
        The record, or an empty dict when there is none for *origin*.  A
        field recorded with no value (``None``) is not a fact and is left
        out, so ``if native_record(...)`` and ``"finish_reason" in record``
        both answer "was this said".  The caller mutates nothing by reading
        it; use :func:`record_native` to write.
    """
    if not isinstance(metadata, Mapping):
        return {}

    folded: dict[str, Any] = {}

    # A record written before this key existed.  Both spellings name the
    # dialect in ``base``; one carried a reason, the other a prompt feedback.
    legacy_reason = metadata.get(LEGACY_FINISH_REASON_KEY)
    if isinstance(legacy_reason, Mapping) and legacy_reason.get("base") == origin:
        reason = legacy_reason.get("reason")
        if reason is not None:
            folded["finish_reason"] = reason
    legacy_feedback = metadata.get(LEGACY_PROMPT_FEEDBACK_KEY)
    if isinstance(legacy_feedback, Mapping) and legacy_feedback.get("base") == origin:
        feedback = legacy_feedback.get("prompt_feedback")
        if feedback is not None:
            folded["prompt_feedback"] = feedback

    # The current record wins a field it holds.  A field with no value is not
    # held: a dict with ``finish_reason: None`` must not read as a word.
    record = metadata.get(NATIVE_RESPONSE_KEY)
    if isinstance(record, Mapping) and record.get("origin") == origin:
        folded.update(
            {
                key: value
                for key, value in record.items()
                if key != "origin" and value is not None
            }
        )

    # ``origin`` alone names no fact, so an empty record reads as absent:
    # ``if native_record(...)`` is then an honest "is there a fact" test.
    if not folded:
        return {}
    folded["origin"] = origin
    return folded


def record_native(
    metadata: dict[str, Any],
    *,
    origin: str,
    finish_reason: Any = None,
    prompt_feedback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Add what *origin* said to the record in *metadata*, in its one shape.

    A field passed as ``None`` leaves the field already recorded alone, so a
    reader that learned the word and a reader that learned the blocked field
    can both write without clobbering each other.

    The legacy keys the record was folded from are removed: the dict leaves
    with one spelling, so a consumer reading the old keys cannot see a stale
    copy of what the new one holds.

    A record with no fact is not written.  A call that adds no field to a
    record that holds none — normalising a legacy key that carried nothing —
    leaves the dict as empty as it found it rather than storing an
    ``origin``-only record that means nothing.

    Args:
        metadata: The ``provider_metadata`` dict to write into.  It is created
            by the caller and mutated in place.
        origin: The converter tag the record belongs to.
        finish_reason: The provider's own word, when it has one.
        prompt_feedback: The provider's blocked-prompt field, verbatim.

    Returns:
        The record as written, or ``{}`` when it holds no fact.
    """
    record = native_record(metadata, origin=origin)
    record["origin"] = origin
    if finish_reason is not None:
        record["finish_reason"] = finish_reason
    if prompt_feedback is not None:
        record["prompt_feedback"] = dict(prompt_feedback)
    metadata.pop(LEGACY_FINISH_REASON_KEY, None)
    metadata.pop(LEGACY_PROMPT_FEEDBACK_KEY, None)
    if len(record) == 1:  # the origin alone
        metadata.pop(NATIVE_RESPONSE_KEY, None)
        return {}
    metadata[NATIVE_RESPONSE_KEY] = record
    return record
