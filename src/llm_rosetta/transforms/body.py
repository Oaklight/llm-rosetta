"""Body-level transform primitives.

A **Transform** operates on the raw provider request/response body dict:
``dict → dict``.  Handles provider-specific field-level quirks (rename,
strip, inject defaults).

Design principles:

* **Idempotent**: applying the same transform twice should be harmless.
* **Non-overlapping**: transforms should operate on different fields by
  convention.
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core type
# ---------------------------------------------------------------------------

Transform = Callable[[dict[str, Any]], dict[str, Any]]
"""A pure data transformation: receives a provider body dict and returns
a (possibly mutated) body dict."""


# ---------------------------------------------------------------------------
# Helper class
# ---------------------------------------------------------------------------


class _NamedTransform:
    """Thin wrapper that gives a factory-produced callable a readable repr."""

    __slots__ = ("_fn", "_repr")

    def __init__(self, fn: Transform, repr_str: str) -> None:
        self._fn = fn
        self._repr = repr_str

    def __call__(self, body: dict[str, Any]) -> dict[str, Any]:
        return self._fn(body)

    def __repr__(self) -> str:
        return self._repr


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------


def strip_fields(*keys: str) -> Transform:
    """Return a transform that removes *keys* from the body.

    No-op for keys that do not exist (idempotent).
    """

    def _strip(body: dict[str, Any]) -> dict[str, Any]:
        for k in keys:
            body.pop(k, None)
        return body

    return _NamedTransform(_strip, f"strip_fields({', '.join(repr(k) for k in keys)})")


def rename_field(old: str, new: str, *, pattern: str | None = None) -> Transform:
    """Return a transform that renames a top-level field.

    No-op if *old* does not exist (idempotent).

    When *pattern* is set, the rename only fires if ``body["model"]``
    matches the regex (search on normalised model string: lowercase,
    non-alphanumeric stripped).

    Example::

        rename_field("max_tokens", "max_completion_tokens")
        rename_field("max_tokens", "max_completion_tokens", pattern=r"^gemini")
    """
    compiled = re.compile(pattern) if pattern else None

    def _rename(body: dict[str, Any]) -> dict[str, Any]:
        if old not in body:
            return body
        if compiled is not None:
            model = body.get("model")
            if not model or not isinstance(model, str):
                return body
            normalised = re.sub(r"[^a-z0-9]", "", model.lower())
            if not compiled.search(normalised):
                return body
        body[new] = body.pop(old)
        return body

    label = f"rename_field({old!r}, {new!r}"
    if pattern:
        label += f", pattern={pattern!r}"
    label += ")"
    return _NamedTransform(_rename, label)


def set_defaults(**defaults: Any) -> Transform:
    """Return a transform that sets fields only when they are absent.

    Idempotent: existing values are never overwritten.
    """

    def _defaults(body: dict[str, Any]) -> dict[str, Any]:
        for k, v in defaults.items():
            body.setdefault(k, v)
        return body

    return _NamedTransform(
        _defaults,
        f"set_defaults({', '.join(f'{k}={v!r}' for k, v in defaults.items())})",
    )


# ---------------------------------------------------------------------------
# Message-level factory functions
# ---------------------------------------------------------------------------


def replace_message_field(field: str, old_value: Any, new_value: Any) -> Transform:
    """Return a transform that replaces *old_value* with *new_value* in
    ``messages[].field``.

    Iterates all entries in ``body["messages"]`` and replaces the field
    value where it matches *old_value*.  No-op if ``messages`` is absent
    or no entry has the matching value (idempotent).

    Example::

        replace_message_field("role", "developer", "system")
    """

    def _replace(body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            return body
        for msg in messages:
            if isinstance(msg, dict) and msg.get(field) == old_value:
                msg[field] = new_value
        return body

    return _NamedTransform(
        _replace,
        f"replace_message_field({field!r}, {old_value!r}, {new_value!r})",
    )


def default_message_field(field: str, default: Any) -> Transform:
    """Return a transform that sets a default for ``messages[].field``
    when the field is absent or its value is ``None``.

    No-op if ``messages`` is absent or no entry needs defaulting
    (idempotent).

    Example::

        default_message_field("content", "")
    """

    def _default(body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            return body
        for msg in messages:
            if isinstance(msg, dict) and msg.get(field) is None:
                msg[field] = default
        return body

    return _NamedTransform(
        _default,
        f"default_message_field({field!r}, {default!r})",
    )


def backfill_reasoning_content(field: str = "reasoning_content") -> Transform:
    """Fill a missing reasoning field on assistant messages.

    DeepSeek's thinking mode requires the model's reasoning to be replayed once
    a request involves tools: "for requests carrying the ``tools`` parameter,
    the ``reasoning_content`` must be fully passed back to the API in all
    subsequent requests — even for turns where the model did not perform a tool
    call."  A request that violates this returns HTTP 400 "The
    ``reasoning_content`` in the thinking mode must be passed back to the API".
    An empty string satisfies the check (verified against the live API), so a
    client that drops the reasoning when replaying history — a cross-format
    agent harness, for example — can still round-trip.

    The requirement bites whenever the message list does not end on a ``user``
    message and the history involves tools, so the transform fills **every**
    assistant message that lacks the field — not only tool-call ones.  Verified
    live on ``deepseek-flash``: a history ending on a tool result (with or
    without the ``tools`` parameter) and a history ending on an assistant turn
    with ``tools`` present both fail as-is and pass once every assistant
    message carries the field; filling only the tool-call message, or only the
    trailing one, still fails.  Ending on a ``user`` message needs nothing.

    The gate is provider-wide even though the requirement is specific to
    thinking mode, so a non-thinking request routed through the same shim also
    gets the field.  DeepSeek accepts it there: both models the shim serves
    (``deepseek-flash``, ``deepseek-v4-pro``) return 200 with
    ``reasoning_content`` set to ``""`` or to text, with and without
    ``reasoning_effort``, and with thinking explicitly disabled.  No model
    guard is needed for the reachable models; one would only be warranted if a
    deployment pinned to the older "never echo ``reasoning_content``"
    behaviour were added.

    Idempotent: a message that already carries a non-empty value is left
    untouched.

    The backfill keeps the request alive but also makes the client-side drop
    invisible — the wire body shows a clean ``reasoning_content`` and nothing
    points at the harness that lost the reasoning.  Body-level transforms have
    no warning sink (the pipeline's ``ctx.warnings`` is converter-level), so
    the count is logged at DEBUG on this module's logger instead.

    Example::

        backfill_reasoning_content()
    """

    def _backfill(body: dict[str, Any]) -> dict[str, Any]:
        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            return body
        last = messages[-1]
        if not isinstance(last, dict) or last.get("role") == "user":
            return body
        # Ending on a tool result trips the check even without the tools
        # parameter; a trailing assistant turn only does when tools are sent.
        if last.get("role") != "tool" and not body.get("tools"):
            return body
        filled = 0
        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "assistant":
                continue
            if not msg.get(field):
                msg[field] = ""
                filled += 1
        if filled:
            logger.debug(
                "backfilled %s on %d assistant message(s); the client replayed "
                "history without the model's reasoning",
                field,
                filled,
            )
        return body

    return _NamedTransform(_backfill, f"backfill_reasoning_content({field!r})")


def strip_fields_for_model(pattern: str, *keys: str) -> Transform:
    """Return a transform that removes *keys* from the body only when
    ``body["model"]`` matches *pattern* (regex search).

    No-op if ``model`` is absent or doesn't match (idempotent).

    Example::

        strip_fields_for_model(r"^claudeopus47", "temperature")
    """
    compiled = re.compile(pattern)

    def _strip(body: dict[str, Any]) -> dict[str, Any]:
        model = body.get("model")
        if not model or not isinstance(model, str):
            return body
        normalised = re.sub(r"[^a-z0-9]", "", model.lower())
        if not compiled.search(normalised):
            return body
        for k in keys:
            body.pop(k, None)
        return body

    return _NamedTransform(
        _strip,
        f"strip_fields_for_model({pattern!r}, {', '.join(repr(k) for k in keys)})",
    )


def apply_transforms(
    transforms: tuple[Transform, ...], body: dict[str, Any]
) -> dict[str, Any]:
    """Apply *transforms* sequentially to *body*, returning the result."""
    for t in transforms:
        body = t(body)
    return body


# ---------------------------------------------------------------------------
# System message flattening
# ---------------------------------------------------------------------------


def flatten_system_content(pattern: str | None = None) -> Transform:
    """Return a body-level transform that flattens system message content
    arrays to plain strings.

    Converts ``messages[].content`` from
    ``[{"type": "text", "text": "..."}]`` to a joined string for system
    messages.  Idempotent — no-op if content is already a string.

    When *pattern* is set, flattening only fires if ``body["model"]``
    matches the regex (search on raw model string).

    Example::

        flatten_system_content()                     # always flatten
        flatten_system_content(pattern=r"gemini")    # only for gemini models
    """
    compiled = re.compile(pattern) if pattern else None

    def _flatten(body: dict[str, Any]) -> dict[str, Any]:
        if compiled is not None:
            model = body.get("model", "")
            if not compiled.search(model):
                return body

        messages = body.get("messages")
        if not messages or not isinstance(messages, list):
            return body

        for msg in messages:
            if not isinstance(msg, dict) or msg.get("role") != "system":
                continue
            content = msg.get("content")
            if isinstance(content, list):
                text = " ".join(
                    p.get("text", "") if isinstance(p, dict) else str(p)
                    for p in content
                )
                msg["content"] = text

        return body

    label = "flatten_system_content("
    if pattern:
        label += f"pattern={pattern!r}"
    label += ")"
    return _NamedTransform(_flatten, label)


# ---------------------------------------------------------------------------
# Tool defaults
# ---------------------------------------------------------------------------


def default_tool_description(default: str = "") -> Transform:
    """Return a transform that fills missing/null tool ``description`` fields.

    vLLM (< 0.28) rejects tool definitions with ``description: null``
    via pydantic validation.  This transform ensures every tool's
    ``function.description`` has a string value.

    Idempotent — no-op when descriptions are already present or there
    are no tools.

    Example::

        default_tool_description()       # fill with ""
        default_tool_description("n/a")  # fill with "n/a"
    """

    def _default(body: dict[str, Any]) -> dict[str, Any]:
        tools = body.get("tools")
        if not tools or not isinstance(tools, list):
            return body
        for tool in tools:
            if not isinstance(tool, dict):
                continue
            func = tool.get("function")
            if isinstance(func, dict) and func.get("description") is None:
                func["description"] = default
        return body

    return _NamedTransform(_default, f"default_tool_description({default!r})")


# ---------------------------------------------------------------------------
# Harmony token rewriting
# ---------------------------------------------------------------------------

_HARMONY_RE = re.compile(
    r"<\|content_invoke_tool_json\|>\s*(\{.*?\})\s*"
    r"(?:<\|end_message\|>|<\|call\|>|$)",
    re.DOTALL,
)


def _parse_harmony_token(text: str) -> tuple[str, list[dict] | None]:
    """Extract a tool call from a raw harmony text token, if present."""

    m = _HARMONY_RE.search(text or "")
    if not m:
        return text, None
    try:
        obj = json.loads(m.group(1))
    except (json.JSONDecodeError, ValueError):
        return text, None
    name = obj.get("name")
    raw_args = obj.get("args", obj.get("arguments", {}))
    if not name:
        return text, None
    if not isinstance(raw_args, str):
        raw_args = json.dumps(raw_args)
    tc = [
        {
            "id": "call_" + uuid.uuid4().hex[:20],
            "type": "function",
            "function": {"name": name, "arguments": raw_args},
        }
    ]
    clean = _HARMONY_RE.sub("", text).strip()
    return clean, tc


def rewrite_harmony_tool_calls() -> Transform:
    """Return a transform that rewrites raw harmony tool-call tokens into
    proper ``tool_calls`` fields.

    Some vLLM-served models (e.g. inkling-bf16) intermittently emit tool
    calls as a raw text token in ``message.content`` (or ``delta.content``
    for streaming) instead of a structured ``tool_calls`` field::

        <|content_invoke_tool_json|>{"name":"fn","args":{...}}<|end_message|>

    This transform detects that pattern and rewrites it into the standard
    OpenAI ``tool_calls`` structure.  Handles both non-streaming
    (``message``) and streaming (``delta``) response shapes.

    No-op when ``tool_calls`` is already present or no harmony token is
    found.
    """

    def _rewrite(body: dict[str, Any]) -> dict[str, Any]:
        choices = body.get("choices")
        if not choices or not isinstance(choices, list):
            return body
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            msg = choice.get("message") or choice.get("delta")
            if not isinstance(msg, dict):
                continue
            if msg.get("tool_calls"):
                continue
            content = msg.get("content")
            if not content or not isinstance(content, str):
                continue
            clean, tc = _parse_harmony_token(content)
            if tc:
                msg["content"] = clean or None
                msg["tool_calls"] = tc
                choice["finish_reason"] = "tool_calls"
        return body

    return _NamedTransform(_rewrite, "rewrite_harmony_tool_calls()")
