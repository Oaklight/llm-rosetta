"""Body-shape detection for the format catalog.

Every chat format in :mod:`llm_rosetta.formats.catalog` names its detector in
its ``detect`` field and gives it a ``detect_priority``.  :func:`detect` walks
that ordered list instead of a hand-written ``if``/``elif`` chain, so a new
format participates in auto-detection by adding one catalog record.

The module is a leaf: it reads the catalog and nothing else.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llm_rosetta.formats import catalog as _catalog

#: A predicate over a raw request body.
Detector = Callable[[dict[str, Any]], bool]


_RESPONSES_ITEM_TYPES = frozenset(
    {
        "message",
        "function_call",
        "function_call_output",
        "mcp_call",
        "mcp_call_output",
        "reasoning",
        "system_event",
        "input_text",
        "output_text",
    }
)

_ANTHROPIC_CONTENT_TYPES = frozenset(
    {"image", "tool_use", "tool_result", "thinking", "document"}
)

_INTERACTIONS_STEP_TYPES = frozenset(
    {"user_input", "model_output", "thought", "function_call", "function_result"}
)

# Wire-format type names (not IR names — the IR calls the "noul" wire type
# "assertion").
_DECISION_QUESTION_TYPES = frozenset({"noul", "choice", "score"})

# Decision is detected at the model-type level: it names a model type rather
# than a format, because its two formats (``systemone``, ``openai_decisions``)
# are told apart by the route, not by the body shape.  It therefore has no
# catalog record and runs ahead of every chat format.
_DECISION_PRIORITY = 10


def is_decision_body(body: dict[str, Any]) -> bool:
    """Check if body matches the System One decision format: state + questions."""
    if "state" not in body or "questions" not in body:
        return False
    questions = body["questions"]
    if not isinstance(questions, dict) or not questions:
        return False
    first_q = next(iter(questions.values()))
    return isinstance(first_q, dict) and first_q.get("type") in _DECISION_QUESTION_TYPES


def is_chat_completions_body(body: dict[str, Any]) -> bool:
    """Check if body matches OpenAI Chat Completions: a ``messages`` array.

    Runs last among the messages-based formats, so Anthropic (also
    ``messages``-based) is claimed first; whatever is left is Chat.
    """
    return "messages" in body


def is_messages_body(body: dict[str, Any]) -> bool:
    """Check if a messages-based body is Anthropic rather than OpenAI Chat.

    Both Anthropic and OpenAI Chat use ``messages``, so this inspects
    top-level fields and content-block types to disambiguate.
    """
    # Anthropic-specific top-level fields
    if "system" in body and isinstance(body["system"], (str, list)):
        return True
    if "anthropic_version" in body or "max_tokens_to_sample" in body:
        return True

    messages = body.get("messages")
    if not isinstance(messages, list) or len(messages) == 0:
        return False

    first_message = messages[0]
    if not isinstance(first_message, dict):
        return False

    content = first_message.get("content")
    if not isinstance(content, list) or len(content) == 0:
        return False

    return any(
        isinstance(part, dict) and part.get("type") in _ANTHROPIC_CONTENT_TYPES
        for part in content
    )


def is_open_responses_body(body: dict[str, Any]) -> bool:
    """Check if body matches the Responses API (input/output with typed items)."""
    items = body.get("input") or body.get("output")
    if not isinstance(items, list) or len(items) == 0:
        return False
    first = items[0]
    return isinstance(first, dict) and first.get("type") in _RESPONSES_ITEM_TYPES


def is_google_generate_body(body: dict[str, Any]) -> bool:
    """Check if body matches Google generateContent (``contents`` with parts)."""
    contents = body.get("contents")
    if not isinstance(contents, list) or len(contents) == 0:
        return False
    first = contents[0]
    return isinstance(first, dict) and "parts" in first


def is_google_interactions_body(body: dict[str, Any]) -> bool:
    """Check if body matches the Google Interactions API.

    Interactions uses ``input`` (string, Content list, or Step list) without
    Responses API item types (``message``, ``function_call_output``, etc.).
    """
    inp = body.get("input")
    if inp is None:
        return False
    if isinstance(inp, str):
        return "contents" not in body and "messages" not in body
    if isinstance(inp, list) and inp:
        first = inp[0]
        if isinstance(first, dict):
            ftype = first.get("type", "")
            if ftype in _INTERACTIONS_STEP_TYPES:
                return True
            if ftype in ("text", "image", "audio", "document", "video"):
                return "contents" not in body and "messages" not in body
    return False


#: Detector name (the catalog's ``detect`` field) → predicate.
DETECTORS: dict[str, Detector] = {
    "chat_completions": is_chat_completions_body,
    "messages": is_messages_body,
    "open_responses": is_open_responses_body,
    "google_generate": is_google_generate_body,
    "google_interactions": is_google_interactions_body,
}


def _ordered_detectors() -> tuple[tuple[int, str, Detector], ...]:
    """Return ``(priority, name, predicate)`` entries, highest priority first.

    Built from the catalog — read at call time, so a catalog that grows a
    record at runtime is honoured — so registering a format there is enough to
    put it in the detection order.
    """
    entries: list[tuple[int, str, Detector]] = [
        (_DECISION_PRIORITY, "decision", is_decision_body)
    ]
    for spec in _catalog.FORMAT_SPECS:
        if spec.detect is None:
            continue
        detector = DETECTORS.get(spec.detect)
        if detector is None:  # pragma: no cover - catalog/registry mismatch
            raise RuntimeError(
                f"Format {spec.format!r} names unknown detector {spec.detect!r}"
            )
        entries.append((spec.detect_priority, spec.detect, detector))
    return tuple(sorted(entries, key=lambda entry: entry[0]))


def detect_format(body: dict[str, Any]) -> str | None:
    """Return the detected provider type for *body*, or ``None``.

    Runs the catalog-ordered detectors; the first matching predicate wins.
    """
    if not isinstance(body, dict):
        return None
    for _, name, predicate in _ordered_detectors():
        if predicate(body):
            return name
    return None


__all__ = [
    "DETECTORS",
    "Detector",
    "detect_format",
    "is_chat_completions_body",
    "is_decision_body",
    "is_google_generate_body",
    "is_google_interactions_body",
    "is_messages_body",
    "is_open_responses_body",
]
