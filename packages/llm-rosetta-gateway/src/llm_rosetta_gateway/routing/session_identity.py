"""What identifies one conversation, and where that answer came from.

A gateway sees only requests.  Two of them belong to the same conversation
when the *client* says so, and only the client can say it: an agentic CLI's
first turn is a fixed preamble (project instructions, a ``<system-reminder>``
block), so the opening message names a *project*, not a conversation.  Measured
over 600 requests of a real capture, one such preamble was the first message
of 395 of them.

The identity is therefore read from a **carrier**, in a fixed order, and each
answer records which carrier produced it:

1. a request header the client sets — ``x-opencode-session``,
   ``x-claude-code-session-id`` and others the shims know, in a declared order;
2. any other ``x-<vendor>-session-id`` header whose value has the shape of an
   id, for a client that follows the convention without our listing it;
3. the body's ``metadata.user_id.session_id`` (a JSON object or a JSON string
   under ``metadata.user_id``);
4. the body's ``prompt_cache_key``, which the Responses family uses as a
   thread id;
5. failing all of those, a hash of how the conversation *opens*, with the
   agent's own preamble skipped.

Only the first four are **declared**: they are the client telling us which
conversation this is.  The fifth is **derived** — a guess that is right often
enough for a log column and never good enough to key stored state on, because
two conversations in one project can share an opening.  Callers that would
store something against the id must read :attr:`SessionIdentity.declared` and
treat a derived id as "unknown".

Every value is hashed before it is returned, so the raw client id is never
stored or logged, and a header longer than :data:`MAX_CARRIER_LENGTH` is
truncated before hashing: an unbounded header is an unbounded log line.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: Headers a client uses to name the conversation, in the order they win.
#: Only names with evidence behind them: seen in captured traffic, or declared
#: by a shim's ``connection.session_header``.  A generic ``x-session-id`` is
#: deliberately absent — middleware and SDKs set it for their own purposes.
SESSION_HEADERS: tuple[str, ...] = (
    "x-opencode-session",
    "x-claude-code-session-id",
    "x-deepseek-harness-session-id",
    "x-conversation-id",
)

#: The convention the rest of the world follows: ``x-<vendor>-session-id``.
#: Read only after :data:`SESSION_HEADERS`, and only when the value has the
#: shape of an id — that shape test is what keeps a middleware's own
#: ``x-request-session-id``-style header from being believed.  When several
#: match, the first by name wins, so one request always answers the same way.
SESSION_HEADER_PATTERN = re.compile(r"^x-[a-z0-9]+(?:-[a-z0-9]+)*-session-id$")

#: What a patterned carrier must look like to be believed: short, and made of
#: the characters an id uses.  A sentence, a path or a base64 blob is not an id.
_PATTERNED_VALUE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.:@+-]{7,127}$")

#: The most of a carrier value that reaches the hash.  Longer is truncated,
#: not refused: a client that appends to its id still gets a stable answer.
MAX_CARRIER_LENGTH = 128

#: Where a ``SessionIdentity`` came from.  ``header:<name>`` and
#: ``body:<path>`` are declared by the client; ``derived:opening`` is this
#: module's own guess; ``none`` means the request carried nothing to key on.
_HEADER = "header"
_BODY = "body"
_DERIVED = "derived"
_NONE = "none"


@dataclass(frozen=True, slots=True)
class SessionIdentity:
    """Which conversation a request belongs to, and how we know.

    Attributes:
        value: A digest of the carrier's value, or ``None`` when the request
            carries nothing to key a conversation on.
        source: ``header:<name>``, ``body:<path>``, ``derived:opening`` or
            ``none``.
    """

    value: str | None = None
    source: str = _NONE

    @property
    def declared(self) -> bool:
        """Whether the client itself named this conversation.

        A consumer that stores state against the id must require this: a
        derived id can name two conversations at once.
        """
        return self.source.startswith((_HEADER + ":", _BODY + ":"))


def identify(
    headers: Mapping[str, str] | None,
    body: Mapping[str, Any] | None = None,
    *,
    header_names: Sequence[str] = SESSION_HEADERS,
    body_fallback: SessionIdentity | None = None,
    client: str | None = None,
) -> SessionIdentity:
    """Return the conversation identity for one request.

    Args:
        headers: The request's headers, any casing.
        body: The request body as the client sent it, before conversion.  When
            ``None``, no body carrier and no derived id is produced.
        header_names: Carrier headers, in precedence order.
        body_fallback: What to use when neither a carrier nor a body produces
            an id — typically a value another layer already derived, so a
            caller that has no raw body keeps its previous behaviour.
        client: The authenticated client's key hash, mixed into the *derived*
            id only.  Two clients that send the same preset preamble are two
            conversations, and this is the only thing that separates them; a
            declared carrier is the client's own answer and needs no help.

    Returns:
        The identity, with its source.
    """
    found = _from_headers(headers, header_names)
    if found is not None:
        name, value = found
        return SessionIdentity(_digest(value), f"{_HEADER}:{name}")

    found_body = _from_body(body)
    if found_body is not None:
        path, value = found_body
        return SessionIdentity(_digest(value), f"{_BODY}:{path}")

    if body is not None:
        messages = _messages_of(body)
        if messages:
            digest = _opening_digest(messages, client)
            if digest is not None:
                return SessionIdentity(digest, f"{_DERIVED}:opening")

    if body_fallback is not None and body_fallback.value is not None:
        return body_fallback
    return SessionIdentity()


def _from_headers(
    headers: Mapping[str, str] | None, names: Sequence[str]
) -> tuple[str, str] | None:
    """Return ``(header name, value)`` for the first carrier that has one."""
    if not headers:
        return None
    lowered = {str(key).lower(): value for key, value in headers.items()}
    for name in names:
        value = lowered.get(name.lower())
        if isinstance(value, str) and value.strip():
            return name, value.strip()
    for name in sorted(lowered):
        if not SESSION_HEADER_PATTERN.match(name):
            continue
        value = lowered[name]
        if isinstance(value, str) and _PATTERNED_VALUE.match(value.strip()):
            return name, value.strip()
    return None


def _from_body(body: Mapping[str, Any] | None) -> tuple[str, str] | None:
    """Return ``(body path, value)`` for the first body carrier present."""
    if not body:
        return None
    metadata = body.get("metadata")
    if isinstance(metadata, Mapping):
        user_id = metadata.get("user_id")
        session = _session_of_user_id(user_id)
        if session:
            return "metadata.user_id.session_id", session
    key = body.get("prompt_cache_key")
    if isinstance(key, str) and key.strip():
        return "prompt_cache_key", key.strip()
    return None


def _session_of_user_id(user_id: Any) -> str:
    """Read ``session_id`` out of a ``metadata.user_id`` that may be JSON text."""
    if isinstance(user_id, Mapping):
        value = user_id.get("session_id")
        return value.strip() if isinstance(value, str) else ""
    if isinstance(user_id, str) and user_id.strip().startswith("{"):
        try:
            parsed = json.loads(user_id)
        except ValueError:
            return ""
        if isinstance(parsed, Mapping):
            value = parsed.get("session_id")
            return value.strip() if isinstance(value, str) else ""
    return ""


def _digest(value: str) -> str:
    """Hash a carrier value so the raw id is never stored or logged."""
    return hashlib.sha256(value[:MAX_CARRIER_LENGTH].encode("utf-8")).hexdigest()[:32]


# ---------------------------------------------------------------------------
# The derived id: how a conversation opens
# ---------------------------------------------------------------------------

#: Agent text wrapped in these says nothing the user wrote.
_REMINDER = re.compile(r"<system-reminder(?:\s[^>]*)?>.*?</system-reminder>", re.S)
#: Codex puts the user's own words after this heading, below its context.
_MY_REQUEST = re.compile(r"(?m)^## My request(?: for Codex)?:[ \t]*\n?")
#: Codex opens with an instruction file's heading rather than a user's words.
_AGENTS_FOR = re.compile(r"^# (\S+\.md) instructions for (\S.*?)[ \t]*\n")

#: How a text the agent put in as the user begins.  The list is a heuristic:
#: a new agent release with a new preamble degrades the derived id to the
#: preamble, which is why this id is never a state key.
_INJECTED_LEADS: tuple[str, ...] = (
    "<system-reminder",
    "<environment_context>",
    "<user_instructions>",
    "<turn_aborted>",
    "<user_shell_command>",
    "<subagent_notification>",
    "<in-app-browser-context",
    "<session-context",
    "<task-notification>",
    "<subagent-notification>",
    "<plugin_reference>",
    "Available agent types for the Agent tool:",
    "The TodoWrite tool hasn't been used recently.",
    "Conversation rewind applied.",
    "Workspace rewind applied.",
    "The following deferred tools are",
    "# MCP Server Instructions",
)

#: Per-message keys that move between turns without the content changing.
_VOLATILE_KEYS = ("cache_control",)

#: How many messages the opening may span before it is given up on.  Bounded
#: so a conversation that opens with many agent messages hashes fast.
_MAX_OPENING = 8


def _messages_of(body: Mapping[str, Any]) -> list[Any]:
    """Return the request's history, whichever field the family uses."""
    for key in ("messages", "input", "contents"):
        items = body.get(key)
        if isinstance(items, (list, tuple)) and items:
            return list(items)
    return []


def _opening_digest(messages: Sequence[Any], client: str | None = None) -> str | None:
    """Hash the conversation's opening, skipping what the agent wrote.

    The client is part of the hash because two tenants can send the same
    preset preamble and no user words at all: the opening alone would name
    both, and the log would report one conversation where there are two.
    """
    opening = _opening(messages)
    if not opening:
        return None
    blobs = [_stable(message) for message in opening]
    joined = "\n".join(blobs)
    if client:
        joined = f"{client}\n{joined}"
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]


def _opening(messages: Sequence[Any]) -> list[Any]:
    """The messages that name the conversation.

    Starts at the first message and ends at the first one the user wrote.
    Leading agent messages are carried along, so two projects that share a
    preamble but differ in it still separate; a preamble with no user words
    after it falls back to that preamble alone, which is the known collision.
    """
    opening: list[Any] = [messages[0]]
    for message in messages[1:_MAX_OPENING]:
        role = _role_of(message)
        if role in ("system", "developer"):
            continue
        if role != "user":
            break
        if len(opening) == 1 and _user_wrote(opening[0]):
            break
        opening.append(message)
        if _user_wrote(message):
            return opening
    return opening[:1]


def _role_of(message: Any) -> str:
    if isinstance(message, Mapping):
        role = message.get("role")
        return role if isinstance(role, str) else ""
    return ""


def _user_wrote(message: Any) -> bool:
    """Whether a message holds anything the user wrote."""
    for kind, part in _parts_of(message):
        if kind == "text" and _user_words(part.get("text") or ""):
            return True
        if kind in _USER_MEDIA:
            return True
    return False


#: Part types only the user produces.
_USER_MEDIA = frozenset(
    {
        "image",
        "image_url",
        "input_image",
        "document",
        "file",
        "input_file",
        "input_audio",
    }
)


def _parts_of(message: Any) -> list[tuple[str, Mapping[str, Any]]]:
    content = message.get("content") if isinstance(message, Mapping) else None
    if isinstance(content, str):
        return [("text", {"type": "text", "text": content})]
    if not isinstance(content, list):
        return []
    return [
        (str(part.get("type") or ""), part)
        for part in content
        if isinstance(part, Mapping)
    ]


def _user_words(text: str) -> str:
    """What a text says in the user's own words; empty when the agent wrote it."""
    cleaned = _REMINDER.sub("", text).strip()
    if "## My request" in cleaned:
        hits = list(_MY_REQUEST.finditer(cleaned))
        if hits:
            cleaned = cleaned[hits[-1].end() :].strip()
    if not cleaned or _AGENTS_FOR.match(cleaned):
        return ""
    for lead in _INJECTED_LEADS:
        if cleaned.startswith(lead):
            return ""
    return cleaned


def _stable(node: Any) -> str:
    """Serialise a value so per-turn volatile fields do not change the hash."""
    return json.dumps(_drop_volatile(node), sort_keys=True, ensure_ascii=False)


def _drop_volatile(node: Any) -> Any:
    if isinstance(node, Mapping):
        return {
            key: _drop_volatile(value)
            for key, value in node.items()
            if key not in _VOLATILE_KEYS
        }
    if isinstance(node, (list, tuple)):
        return [_drop_volatile(value) for value in node]
    return node


__all__ = [
    "MAX_CARRIER_LENGTH",
    "SESSION_HEADERS",
    "SESSION_HEADER_PATTERN",
    "SessionIdentity",
    "identify",
]
