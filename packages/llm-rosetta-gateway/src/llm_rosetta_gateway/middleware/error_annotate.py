"""Annotate upstream error bodies with gateway-side diagnostics.

Some requests the gateway forwards carry a defect no converter can repair —
a tool list that declares one name twice with different definitions, a tool
schema an upstream type-checks and rejects.  The upstream's own rejection
names the tool the way *it* saw the request, which is not how the client
wrote it: it points at an index in a list the client never built, or at a
name a rename pass produced.  The failure then reads like a gateway fault
even though the client's request caused it.

This module appends a short attribution to the error body the client receives:
which of its own declarations the upstream meant, and where that declaration
sits in the request it wrote.  The rewrite is deliberately narrow: of the
parsed document, only the ``error.message`` string changes, and only when the
request actually carried the defect.  The body is re-serialized, so whitespace
and escaping may come back normalized (``ensure_ascii=False`` keeps non-ASCII
text readable) — harmless to a JSON client, visible to one that diffs the raw
bytes.  A body that is not JSON, or whose error envelope carries no string
message, is returned untouched.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from llm_rosetta.capabilities import ToolNameMap
from llm_rosetta.formats.tool_locations import ToolDeclaration, read_tool_declarations

# Marks a message this module already annotated, so a chained gateway or a
# client retry does not stack a second copy of the hint.
_MARKER = "[llm-rosetta]"

# Names listed before the hint summarizes the rest.  Each collision needs two
# definitions in the client's own request, so the input bounds this — the cap
# only keeps a pathological tool list from producing a multi-kilobyte message.
_MAX_HINT_NAMES = 10


def collision_hint(collisions: Mapping[str, int]) -> str:
    """Build the attribution sentence for a set of conflicting tool names.

    Args:
        collisions: Conflicting tool name -> number of distinct definitions
            declared for it.

    Returns:
        One sentence naming the tools in sorted order, capped at
        :data:`_MAX_HINT_NAMES` names with a count of the remainder.
    """
    names = sorted(collisions.items())
    parts = "; ".join(
        f"declared tool name {name!r} with {count} distinct definitions"
        for name, count in names[:_MAX_HINT_NAMES]
    )
    if len(names) > _MAX_HINT_NAMES:
        parts += f"; and {len(names) - _MAX_HINT_NAMES} more"
    return (
        f"{_MARKER} This request {parts}. Some upstreams require unique tool "
        "names, so this rejection may come from the request's own tool list."
    )


def annotate_tool_collisions(
    raw: str | bytes, collisions: Mapping[str, int] | None
) -> str | bytes:
    """Append a tool-name-collision hint to a serialized error body.

    Args:
        raw: The serialized error body, from an upstream or a synthesized
            soft-error message.
        collisions: Conflicting tool names to report.  An empty or ``None``
            mapping makes this a no-op.

    Returns:
        The body with the hint appended to ``error.message``, matching the
        input type, or *raw* unchanged when there is nothing to annotate or
        the body cannot be rewritten.
    """
    return annotate_tool_errors(raw, collisions)


def annotate_tool_errors(
    raw: str | bytes,
    collisions: Mapping[str, int] | None = None,
    *,
    hint_for: Callable[[str], str | None] | None = None,
) -> str | bytes:
    """Append gateway-side tool diagnostics to a serialized error body.

    The body is treated as opaque: it may be in the upstream's format rather
    than the client's, and any envelope whose ``error`` object carries a
    string ``message`` (OpenAI, Anthropic and Google all do) is accepted
    without format-specific handling.  Only the parsed ``error.message``
    changes; the returned body is re-serialized, so whitespace and escaping
    may differ from the input.

    Both diagnostics travel in one rewrite.  The marker that keeps a chained
    gateway from stacking a second copy is checked against the message as a
    whole, so running the two hints separately would suppress whichever came
    second.

    Args:
        raw: The serialized error body.
        collisions: Conflicting tool names to report, or ``None``.
        hint_for: Builds a further hint from the upstream's message — the
            tool-schema attribution, whose sentence depends on what the
            upstream named.  ``None``, or a callable returning ``None``, adds
            nothing.

    Returns:
        The body with the hints appended to ``error.message``, matching the
        input type, or *raw* unchanged when there is nothing to annotate or
        the body cannot be rewritten.
    """
    if not collisions and hint_for is None:
        return raw
    if isinstance(raw, bytes):
        try:
            text: str | None = raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw
    else:
        text = raw
    annotated = _annotate_text(text, collisions, hint_for)
    if annotated is None:
        return raw
    return annotated.encode("utf-8") if isinstance(raw, bytes) else annotated


def _annotate_text(
    text: str,
    collisions: Mapping[str, int] | None,
    hint_for: Callable[[str], str | None] | None,
) -> str | None:
    """Rewrite one JSON error body, or return ``None`` if it does not fit.

    Returns ``None`` for anything this module should leave alone: a body that
    does not parse, is not a JSON object, has no ``error`` object, has a
    non-string or already-annotated message.
    """
    try:
        data: Any = json.loads(text)
    except (ValueError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    error = data.get("error")
    if not isinstance(error, dict):
        return None
    message = error.get("message")
    if not isinstance(message, str) or _MARKER in message:
        return None
    hints: list[str] = []
    if collisions:
        hints.append(collision_hint(collisions))
    if hint_for is not None:
        extra = hint_for(message)
        if extra:
            hints.append(extra)
    if not hints:
        return None
    joined = "\n\n".join(hints)
    error["message"] = f"{message}\n\n{joined}" if message else joined
    return json.dumps(data, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tool-schema attribution
# ---------------------------------------------------------------------------

#: A path-shaped reference to a tool list in an upstream message.  Both the
#: bracket spelling (``tools[16]``, Google's ``tools[0]``) and the dot spelling
#: (Anthropic's ``tools.0``) occur in the wild, and Google names its nested list
#: in snake_case while its REST writer uses camelCase.
_PATH_HEAD = re.compile(
    r"(?:tools|function_declarations|functionDeclarations)"
    r"(?:\s*\[\s*\d+\s*\]|\.\d+)"
)

#: An upstream message that names the tool instead of indexing it, e.g.
#: ``Invalid schema for function 'send_message': ...``.
_NAME_REF = re.compile(r"\b(?:function|tool)\s+['\"]([^'\"]+)['\"]")

#: OpenAI's schema validator reports the position as a tuple, e.g.
#: ``In context=('properties', 'count')``.
_CONTEXT = re.compile(r"In context=\(([^)]*)\)")

_INDEX_SEGMENT = re.compile(r"\[\s*(\d+)\s*\]")
_KEY_SEGMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True)
class _Reference:
    """What an upstream message points at, before it is attributed.

    Attributes:
        raw: The message's own spelling of the reference, quoted back to the
            client so it can match the upstream's words.
        name: The upstream's name for the tool.
        segments: The path segments after the declaration, as the upstream
            spelled them (``("function", "name")``).
        declaration: The declaration in the forwarded body that the reference
            resolved to, when the forwarded body declares the tool.
        positional: Whether the message pointed at a position in the forwarded
            list rather than naming the tool.  A position is meaningless to
            the client, so the name has to be stated; a name it already wrote
            needs no restating.
    """

    raw: str
    name: str
    segments: tuple[str, ...]
    declaration: ToolDeclaration | None
    positional: bool


class ToolSchemaLocator:
    """Maps an upstream tool reference back to the client's own declaration.

    An upstream rejection names a tool the way the *forwarded* body spells it:
    by its position in the list it received, or by a name a rename pass may
    have produced.  The client holds a different body and a different spelling
    — it may have sent its declarations inside ``functionDeclarations``, and an
    MCP name it wrote as ``srv/fs__read`` reaches the wire as
    ``srv_fs__read``.  Both bodies and the pipeline's name map are what turn
    one into the other.

    The locator reads declarations lazily and caches them: a caller constructs
    one per error, and a body with no tools costs one dict lookup.
    """

    def __init__(
        self,
        *,
        upstream_format: str,
        client_format: str,
        upstream_body: Any,
        client_body: Any,
        name_map: ToolNameMap | None = None,
    ) -> None:
        """Record the two bodies and the formats they are written in.

        Args:
            upstream_format: The format of *upstream_body* (the forwarded
                request), e.g. ``chat_completions``.
            client_format: The format of *client_body* (what the client sent).
            upstream_body: The request the upstream received.
            client_body: The request the client sent.
            name_map: The request leg's client↔upstream tool-name pairing.
                Absent or empty, a name is taken to be the client's own.
        """
        self._upstream_format = upstream_format
        self._client_format = client_format
        self._upstream_body = upstream_body
        self._client_body = client_body
        self._name_map = name_map
        self._upstream_cache: list[ToolDeclaration] | None = None
        self._client_cache: list[ToolDeclaration] | None = None

    def hint(self, message: str) -> str | None:
        """Attribute *message*'s tool reference to the client, when it has one.

        Args:
            message: The upstream's own error message, verbatim.

        Returns:
            One attribution sentence, or ``None`` when the message names no
            tool, or names one the client's own request does not declare.
        """
        reference = self._reference(message)
        if reference is None:
            return None
        client_name = self._client_name(reference.name)
        client_declaration = self._client_declaration(client_name)
        if client_declaration is None:
            return None
        segments = self._client_segments(reference, client_declaration, message)
        return self._sentence(reference, client_name, client_declaration, segments)

    # -- reading the two bodies ------------------------------------------

    @property
    def _upstream_declarations(self) -> list[ToolDeclaration]:
        if self._upstream_cache is None:
            self._upstream_cache = read_tool_declarations(
                self._upstream_format, self._upstream_body
            )
        return self._upstream_cache

    @property
    def _client_declarations(self) -> list[ToolDeclaration]:
        if self._client_cache is None:
            self._client_cache = read_tool_declarations(
                self._client_format, self._client_body
            )
        return self._client_cache

    def _client_name(self, upstream_name: str) -> str:
        """The client's spelling of *upstream_name*, when a rename happened."""
        if self._name_map is None:
            return upstream_name
        client_name, _namespace = self._name_map.to_client(upstream_name)
        return client_name or upstream_name

    def _client_declaration(self, name: str) -> ToolDeclaration | None:
        """The client's declaration that carries *name*, if it declares one.

        A name several declarations share — the same child name in two
        containers — resolves to the first, which is the one a client reading
        its own request top to bottom meets first.
        """
        for declaration in self._client_declarations:
            if declaration.name == name:
                return declaration
        return None

    # -- resolving the reference -----------------------------------------

    def _reference(self, message: str) -> _Reference | None:
        """Extract the first tool reference *message* makes, if any.

        A positional reference is preferred: it identifies the declaration
        unambiguously, and the name can be read off the forwarded body.  A
        name reference is the fallback, for messages that name the function
        without indexing the list (OpenAI's schema validator does).
        """
        for match in _PATH_HEAD.finditer(message):
            raw = _scan_path(message, match.start())
            segments = _split_path(raw)
            declaration = self._declaration_at(segments)
            if declaration is None or declaration.name is None:
                continue
            remainder = tuple(segments[len(_split_path(declaration.path)) :])
            return _Reference(raw, declaration.name, remainder, declaration, True)
        for match in _NAME_REF.finditer(message):
            name = match.group(1)
            return _Reference(name, name, (), self._declaration_named(name), False)
        return None

    def _declaration_at(self, segments: tuple[str, ...]) -> ToolDeclaration | None:
        """The declaration *segments* reaches, or the one they can only mean.

        A reference may stop short of the declaration — Google nests its
        declarations, so ``tools[0]`` names a container rather than a tool.  It
        still identifies one when exactly one declaration extends the
        reference.  A container with several children names none of them, so it
        resolves to ``None``: a guess would point the client at a declaration
        the upstream never named.
        """
        best: ToolDeclaration | None = None
        best_length = 0
        extending: list[ToolDeclaration] = []
        for declaration in self._upstream_declarations:
            declared = _split_path(declaration.path)
            if _is_prefix(declared, segments):
                if len(declared) > best_length:
                    best, best_length = declaration, len(declared)
                continue
            if _is_prefix(segments, declared):
                extending.append(declaration)
        if best is not None:
            return best
        return extending[0] if len(extending) == 1 else None

    def _declaration_named(self, name: str) -> ToolDeclaration | None:
        """The forwarded declaration that carries *name*."""
        for declaration in self._upstream_declarations:
            if declaration.name == name:
                return declaration
        return None

    # -- rendering the client side ---------------------------------------

    def _client_segments(
        self,
        reference: _Reference,
        client: ToolDeclaration,
        message: str,
    ) -> tuple[str, ...]:
        """Translate the upstream's within-tool path into the client's spelling.

        Two steps are wire-only and do not survive the translation: the
        declaration's wrapper (Chat's ``function``) and the upstream's name for
        the schema field (``input_schema``, ``parameters``).  What remains is
        the same JSON Schema path, which the client's own body spells with its
        own schema field.
        """
        segments = list(reference.segments)
        upstream = reference.declaration
        if upstream is not None:
            if (
                upstream.wrapper
                and segments
                and _same_token(segments[0], upstream.wrapper)
            ):
                segments.pop(0)
            if (
                upstream.schema_field
                and segments
                and _same_token(segments[0], upstream.schema_field)
            ):
                if client.schema_field is None:
                    return ()
                segments[0] = client.schema_field
        return tuple(segments) or _context_segments(message, client)

    def _sentence(
        self,
        reference: _Reference,
        client_name: str,
        client: ToolDeclaration,
        segments: tuple[str, ...],
    ) -> str:
        """Compose the attribution sentence that is appended to the error.

        A name the client also wrote needs no translation, so the sentence only
        restates it when the spelling differs or when the upstream pointed at a
        position the client cannot see — there the name is the whole answer.
        """
        location = f"declaration #{client.number} ({client.path})"
        repeat_name = reference.positional or reference.name != client_name
        identity = (
            f"the client's tool {client_name!r}, {location}"
            if repeat_name
            else f"the client's {location}"
        )
        head = f"{_MARKER} {reference.raw!r} is {identity}"
        if segments:
            return (
                f"{head}; the rejected position in the request is "
                f"{_render(client, segments)}."
            )
        if client.schema_field:
            return (
                f"{head}; its schema is at {_render(client, (client.schema_field,))}."
            )
        return f"{head}."


def _context_segments(message: str, client: ToolDeclaration) -> tuple[str, ...]:
    """The path OpenAI states as ``In context=(...)``, under the client's schema.

    The tuple is the schema path itself (``('properties', 'count')``), so it is
    prefixed with the field the client's declaration keeps its schema in.
    """
    match = _CONTEXT.search(message)
    if match is None or not client.schema_field:
        return ()
    items = re.findall(r"['\"]([^'\"]*)['\"]", match.group(1))
    return (client.schema_field, *items)


def _render(declaration: ToolDeclaration, segments: tuple[str, ...]) -> str:
    """Render *segments* as a path into *declaration*, in the client's spelling."""
    path = declaration.path
    if declaration.wrapper:
        path += f".{declaration.wrapper}"
    for segment in segments:
        path += segment if segment.startswith("[") else f".{segment}"
    return path


def _split_path(text: str) -> tuple[str, ...]:
    """Split a wire path into segments, normalizing every index spelling.

    ``tools[0].functionDeclarations[3].parameters``, ``tools.0.input_schema``
    and ``tools[ 0 ].x`` split the same way; keys keep their spelling and
    indices come back as ``[3]``.
    """
    segments: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        if char == ".":
            index += 1
            continue
        match = _INDEX_SEGMENT.match(text, index)
        if match is not None:
            segments.append(f"[{int(match.group(1))}]")
            index = match.end()
            continue
        if char.isdigit():
            stop = index
            while stop < len(text) and text[stop].isdigit():
                stop += 1
            segments.append(f"[{int(text[index:stop])}]")
            index = stop
            continue
        match = _KEY_SEGMENT.match(text, index)
        if match is None:
            break
        segments.append(match.group(0))
        index = match.end()
    return tuple(segments)


def _scan_path(text: str, start: int) -> str:
    """Consume the path-shaped run of *text* that begins at *start*.

    Stops at the first character a path cannot contain — the space or quote
    that ends the reference in the upstream's sentence.  Spaces are allowed
    only inside brackets, so ``tools[ 16 ].function`` stays one token.
    """
    index = start
    depth = 0
    while index < len(text):
        char = text[index]
        if char == "[":
            depth += 1
        elif char == "]":
            if depth == 0:
                break
            depth -= 1
        elif char == " ":
            following = index
            while following < len(text) and text[following] == " ":
                following += 1
            inside = depth > 0 and following < len(text)
            if not (inside and (text[following].isdigit() or text[following] == "]")):
                break
        elif not (char.isalnum() or char in "._"):
            break
        index += 1
    return text[start:index]


def _is_prefix(prefix: tuple[str, ...], whole: tuple[str, ...]) -> bool:
    """Whether *prefix* leads *whole*, comparing keys case- and underscore-blind."""
    return len(prefix) <= len(whole) and all(
        _same_token(a, b) for a, b in zip(prefix, whole, strict=False)
    )


def _same_token(left: str, right: str) -> bool:
    """Compare two path segments, ignoring key case and underscore spelling."""
    if left.startswith("[") or right.startswith("["):
        return left == right
    return left.replace("_", "").lower() == right.replace("_", "").lower()
