"""Provenance helpers for opaque, provider-specific reasoning values.

Opaque reasoning values — Anthropic ``thinking.signature`` /
``redacted_thinking.data``, OpenAI ``encrypted_content``, Google
``thoughtSignature`` / ``function_call.signature`` — are only valid for the
provider that issued them.  The IR stores them under provider-agnostic names
(:pyattr:`~llm_rosetta.types.ir.parts.ReasoningPart.signature` and
``redacted_data``), so a converter must know a value's *origin* before it can
decide whether to emit it.

Origin is recorded under a reserved key inside the IR part's
``provider_metadata``::

    provider_metadata["_opaque_origin"] = {
        "signature": "<converter_tag>",
        "redacted_data": "<converter_tag>",
    }

An absent key (or absent field) means the origin is *unknown*; callers treat
unknown as same-origin so legacy and hand-built IR keeps round-tripping.

Emission policy derived from the origin (issue #881):

- Anthropic / OpenAI targets drop a foreign opaque value.
- Google targets emit the real value when it is Google-origin, otherwise inject
  the documented dummy signature (:data:`DUMMY_THOUGHT_SIGNATURE`) on the first
  ``functionCall`` of a turn.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "DUMMY_THOUGHT_SIGNATURE",
    "GOOGLE_NS",
    "OPAQUE_ORIGIN_KEY",
    "OPAQUE_SAME_ORIGIN",
    "carrier_metadata",
    "get_opaque_origin",
    "google_thought_signature",
    "inject_google_function_call_signature",
    "inject_interactions_function_call_signature",
    "is_own_origin",
    "set_google_thought_signature",
    "set_opaque_origin",
]

#: Reserved ``provider_metadata`` key holding opaque-value provenance.
OPAQUE_ORIGIN_KEY = "_opaque_origin"

#: ``provider_metadata`` namespace used for Google-specific carrier values.
GOOGLE_NS = "google"

#: Key inside the Google namespace holding the thought signature.
GOOGLE_SIGNATURE_FIELD = "thought_signature"

#: Documented Google bypass value for replayed function-call history that has
#: no native Gemini thought signature (cross-model traces).  Google accepts it
#: on both Gemini 2.5 and 3; arbitrary values are rejected.
DUMMY_THOUGHT_SIGNATURE = "skip_thought_signature_validator"

#: Converter tags that share opaque-value semantics (same provider family).
#: A value whose recorded origin is in the emitting converter's family is its
#: own and may be preserved; any other recorded origin is foreign (dropped, or
#: adapted to the Google dummy).
OPAQUE_SAME_ORIGIN: dict[str, frozenset[str]] = {
    "messages": frozenset({"messages"}),
    "open_responses": frozenset({"open_responses"}),
    "chat_completions": frozenset({"chat_completions"}),
    "google_generate": frozenset({"google_generate", "google_interactions"}),
    "google_interactions": frozenset({"google_generate", "google_interactions"}),
}


def _ensure_metadata(part: Any) -> dict[str, Any]:
    """Return a mutable ``provider_metadata`` dict, tolerating ``None``."""
    metadata = part.get("provider_metadata")
    if not isinstance(metadata, dict):
        metadata = {}
        part["provider_metadata"] = metadata
    return metadata


def set_opaque_origin(
    part: Any,
    *,
    signature: str | None = None,
    redacted_data: str | None = None,
) -> None:
    """Record the originating converter tag for an opaque reasoning value.

    Existing origins are **not** overwritten: an origin ferried through a
    converter hop is more authoritative than one re-derived while reading a
    provider payload, so readers only fill in the unknown case.

    Args:
        part: IR reasoning part (or any IR dict carrying ``provider_metadata``).
        signature: Converter tag owning ``part["signature"]``, if set.
        redacted_data: Converter tag owning ``part["redacted_data"]``, if set.
    """
    if signature is None and redacted_data is None:
        return
    origins = _ensure_metadata(part).setdefault(OPAQUE_ORIGIN_KEY, {})
    if signature is not None:
        origins.setdefault("signature", signature)
    if redacted_data is not None:
        origins.setdefault("redacted_data", redacted_data)


def get_opaque_origin(part: Any, field: str) -> str | None:
    """Return the recorded origin tag for ``field`` (``signature``/``redacted_data``).

    Args:
        part: IR part carrying ``provider_metadata``.
        field: ``"signature"`` or ``"redacted_data"``.

    Returns:
        The originating converter tag, or ``None`` when unknown.
    """
    metadata = part.get("provider_metadata") or {}
    origins = metadata.get(OPAQUE_ORIGIN_KEY) or {}
    origin = origins.get(field)
    return origin if isinstance(origin, str) else None


def is_own_origin(origin: str | None, tag: str) -> bool:
    """Whether a value with ``origin`` belongs to the ``tag`` converter's family.

    Unknown origin (``None``) is treated as own for back-compat with legacy and
    hand-built IR that never recorded provenance.

    Args:
        origin: Result of :func:`get_opaque_origin`.
        tag: The emitting converter's own tag (e.g. ``"messages"``).

    Returns:
        ``True`` when the value may be preserved, ``False`` when it is foreign
        and must be dropped or adapted.
    """
    if origin is None:
        return True
    return origin in OPAQUE_SAME_ORIGIN.get(tag, frozenset({tag}))


def carrier_metadata(part: Any) -> dict[str, Any] | None:
    """Return ``provider_metadata`` for the ``_provider_metadata`` carrier.

    The reserved :data:`OPAQUE_ORIGIN_KEY` is converter-internal bookkeeping,
    not provider data, so it is filtered out here — it must not ride along into
    a provider payload for callers that bypass the gateway's
    ``_strip_internal_metadata``.

    Args:
        part: IR part carrying ``provider_metadata``.

    Returns:
        A copy without the reserved key, or ``None`` when nothing remains.
    """
    metadata = part.get("provider_metadata") or {}
    remaining = {k: v for k, v in metadata.items() if k != OPAQUE_ORIGIN_KEY}
    return remaining or None


def google_thought_signature(part: Any) -> str | None:
    """Read the Google thought signature carried on an IR part, if any."""
    metadata = part.get("provider_metadata") or {}
    value = (metadata.get(GOOGLE_NS) or {}).get(GOOGLE_SIGNATURE_FIELD)
    return value if isinstance(value, str) and value else None


def set_google_thought_signature(part: Any, value: str | None) -> None:
    """Mirror a Google thought signature into ``provider_metadata["google"]``.

    This lets a Google-origin value survive a hop through a foreign converter
    (which drops the flat ``signature`` field but ferries ``provider_metadata``).
    """
    if not value:
        return
    namespace = _ensure_metadata(part).setdefault(GOOGLE_NS, {})
    namespace[GOOGLE_SIGNATURE_FIELD] = value


def inject_google_function_call_signature(
    parts: list[dict[str, Any]],
    *,
    function_key: str = "functionCall",
    signature_key: str = "thoughtSignature",
) -> bool:
    """Inject the dummy thought signature on the first ``functionCall`` part.

    No-op when the turn has no function-call part, or when its first
    function-call already carries a signature (real or previously injected).

    Args:
        parts: Outbound Google ``contents[].parts`` list for one model turn.
        function_key: Part key holding the function call (``functionCall``).
        signature_key: Part key holding the thought signature.

    Returns:
        ``True`` when the dummy was written.
    """
    for part in parts:
        if function_key not in part and "function_call" not in part:
            continue
        if part.get(signature_key) or part.get("thought_signature"):
            return False
        part[signature_key] = DUMMY_THOUGHT_SIGNATURE
        return True
    return False


def inject_interactions_function_call_signature(
    steps: list[dict[str, Any]],
    *,
    step_type: str = "function_call",
    signature_key: str = "signature",
) -> bool:
    """Inject the dummy signature on the first ``function_call`` step.

    No-op when the turn has no function-call step, or when its first
    function-call step already carries a signature.

    Args:
        steps: Outbound Interactions ``steps`` list for one assistant turn.
        step_type: Step type marking a function call.
        signature_key: Step key holding the signature.

    Returns:
        ``True`` when the dummy was written.
    """
    for step in steps:
        if step.get("type") != step_type:
            continue
        if step.get(signature_key):
            return False
        step[signature_key] = DUMMY_THOUGHT_SIGNATURE
        return True
    return False
