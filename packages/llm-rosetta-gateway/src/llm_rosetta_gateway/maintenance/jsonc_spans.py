"""A JSONC reader that remembers where every value sits in the text.

The gateway's config files are JSONC: comments and trailing commas are legal
and operators use them.  A ``json`` round-trip would drop every comment and
reflow every line, so a tool that edits such a file must edit the *text*.  This
module parses a document into a small tree whose nodes carry their character
offsets, and :class:`Edits` applies replacements and insertions to those
offsets.  Everything outside an edited span is copied through byte for byte.

It is a reader for the subset the gateway accepts (``//`` and ``/* */``
comments, trailing commas); it does not validate numbers beyond scanning them.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "Edits",
    "Member",
    "Node",
    "ParseError",
    "append_item",
    "append_member",
    "line_of",
    "parse",
    "remove_member",
    "rename_member",
    "trailing_comment",
]


class ParseError(ValueError):
    """The document is not JSONC the reader understands.

    Attributes:
        offset: Character offset of the problem.
    """

    def __init__(self, message: str, offset: int) -> None:
        super().__init__(message)
        self.offset = offset


@dataclass
class Node:
    """One value and the span ``text[start:end]`` it occupies.

    Attributes:
        kind: ``obj``, ``arr``, ``str`` or ``lit`` (number, bool, null).
        start: Offset of the first character (the quote for a string).
        end: Offset one past the last character.
        value: The decoded string for ``str``, the literal text for ``lit``.
        members: The members of an ``obj``, in document order.
        items: The elements of an ``arr``, in document order.
    """

    kind: str
    start: int
    end: int
    value: Any = None
    members: list[Member] = field(default_factory=list)
    items: list[Node] = field(default_factory=list)

    def member(self, key: str) -> Member | None:
        """The first member of an object named *key*, or ``None``."""
        for member in self.members:
            if member.key.value == key:
                return member
        return None


@dataclass
class Member:
    """One ``"key": value`` pair of an object."""

    key: Node
    value: Node


def line_of(text: str, offset: int) -> int:
    """The 1-based line number of *offset*."""
    return text.count("\n", 0, offset) + 1


class _Reader:
    def __init__(self, text: str) -> None:
        self.text = text
        self.pos = 0

    def _skip(self) -> None:
        text, n = self.text, len(self.text)
        while self.pos < n:
            ch = text[self.pos]
            if ch in " \t\r\n":
                self.pos += 1
            elif text.startswith("//", self.pos):
                end = text.find("\n", self.pos)
                self.pos = n if end < 0 else end
            elif text.startswith("/*", self.pos):
                end = text.find("*/", self.pos + 2)
                if end < 0:
                    raise ParseError("unterminated comment", self.pos)
                self.pos = end + 2
            else:
                return

    def document(self) -> Node:
        self._skip()
        node = self._value()
        self._skip()
        if self.pos != len(self.text):
            raise ParseError("unexpected text after the document", self.pos)
        return node

    def _value(self) -> Node:
        if self.pos >= len(self.text):
            raise ParseError("unexpected end of document", self.pos)
        ch = self.text[self.pos]
        if ch == "{":
            return self._object()
        if ch == "[":
            return self._array()
        if ch == '"':
            return self._string()
        return self._literal()

    def _string(self) -> Node:
        text, start = self.text, self.pos
        i = start + 1
        while i < len(text):
            if text[i] == "\\":
                i += 2
            elif text[i] == '"':
                raw = text[start : i + 1]
                try:
                    value = json.loads(raw)
                except ValueError as exc:
                    raise ParseError(f"bad string: {exc}", start) from exc
                self.pos = i + 1
                return Node("str", start, i + 1, value)
            elif text[i] == "\n":
                break
            else:
                i += 1
        raise ParseError("unterminated string", start)

    def _literal(self) -> Node:
        start = self.pos
        while self.pos < len(self.text) and self.text[self.pos] not in ' \t\r\n,]}/"':
            self.pos += 1
        if self.pos == start:
            raise ParseError(f"unexpected {self.text[start]!r}", start)
        return Node("lit", start, self.pos, self.text[start : self.pos])

    def _array(self) -> Node:
        node = Node("arr", self.pos, self.pos)
        self.pos += 1
        while True:
            self._skip()
            if self.pos >= len(self.text):
                raise ParseError("unterminated array", node.start)
            if self.text[self.pos] == "]":
                self.pos += 1
                node.end = self.pos
                return node
            node.items.append(self._value())
            self._skip()
            if self.pos < len(self.text) and self.text[self.pos] == ",":
                self.pos += 1

    def _object(self) -> Node:
        node = Node("obj", self.pos, self.pos)
        self.pos += 1
        while True:
            self._skip()
            if self.pos >= len(self.text):
                raise ParseError("unterminated object", node.start)
            if self.text[self.pos] == "}":
                self.pos += 1
                node.end = self.pos
                return node
            if self.text[self.pos] != '"':
                raise ParseError("expected a string key", self.pos)
            key = self._string()
            self._skip()
            if self.pos >= len(self.text) or self.text[self.pos] != ":":
                raise ParseError("expected ':' after a key", self.pos)
            self.pos += 1
            self._skip()
            node.members.append(Member(key, self._value()))
            self._skip()
            if self.pos < len(self.text) and self.text[self.pos] == ",":
                self.pos += 1


def parse(text: str) -> Node:
    """Parse *text* as JSONC into a span-carrying tree.

    Raises:
        ParseError: When the text is not a JSONC document.
    """
    return _Reader(text).document()


class Edits:
    """A set of span replacements and insertions over one text."""

    def __init__(self, text: str) -> None:
        self.text = text
        self._edits: list[tuple[int, int, str]] = []

    def __bool__(self) -> bool:
        return bool(self._edits)

    def replace_string(self, node: Node, new: str) -> None:
        """Replace a string node (a value or a key) with the string *new*."""
        self._edits.append((node.start, node.end, json.dumps(new)))

    def replace_span(self, node: Node, text: str) -> None:
        """Replace the span of *node* (a literal, for example) with *text*."""
        self._edits.append((node.start, node.end, text))

    def insert(self, offset: int, text: str) -> None:
        """Insert *text* at *offset*."""
        self._edits.append((offset, offset, text))

    def apply(self) -> str:
        """The text with every edit applied; the rest is copied through.

        Raises:
            ValueError: When two edits overlap, which is a bug in the planner.
        """
        out: list[str] = []
        cursor = 0
        for start, end, text in sorted(self._edits, key=lambda e: (e[0], e[1])):
            if start < cursor:
                raise ValueError("overlapping edits")
            out.append(self.text[cursor:start])
            out.append(text)
            cursor = end
        out.append(self.text[cursor:])
        return "".join(out)


def _indent_of(text: str, node: Node) -> str:
    """The leading whitespace of the line *node* starts on, if only blanks."""
    line_start = text.rfind("\n", 0, node.start) + 1
    prefix = text[line_start : node.start]
    return prefix if prefix.strip() == "" else ""


def append_member(edits: Edits, obj: Node, key: str, value_json: str) -> None:
    """Add ``"key": value_json`` as the last member of *obj*.

    The new member copies the indentation of the last existing member.  It goes
    right after that member's value, so an existing trailing comma and any
    comment after it keep their place.
    """
    member = json.dumps(key) + ": " + value_json
    if not obj.members:
        edits.insert(obj.start + 1, " " + member + " ")
        return
    last = obj.members[-1]
    indent = _indent_of(edits.text, last.key)
    sep = ",\n" + indent if indent else ", "
    edits.insert(last.value.end, sep + member)


def append_item(edits: Edits, arr: Node, item_json: str) -> None:
    """Add *item_json* as the last element of *arr*, on the same line."""
    if not arr.items:
        edits.insert(arr.start + 1, item_json)
    else:
        edits.insert(arr.items[-1].end, ", " + item_json)


def trailing_comment(text: str, member: Member) -> tuple[str | None, int]:
    """The comment on the rest of *member*'s last line, and where it ends.

    Returns ``(comment, end)``: the ``//`` or same-line ``/* */`` comment that
    follows the member's value (and its comma), or ``(None, offset after the
    comma)`` when the line has none.
    """
    probe = member.value.end
    while probe < len(text) and text[probe] in " \t":
        probe += 1
    if probe < len(text) and text[probe] == ",":
        probe += 1
    after_comma = probe
    while probe < len(text) and text[probe] in " \t":
        probe += 1
    if text.startswith("//", probe):
        end = text.find("\n", probe)
        end = len(text) if end < 0 else end
        return text[probe:end].rstrip(), end
    if text.startswith("/*", probe):
        end = text.find("*/", probe)
        if end >= 0 and "\n" not in text[probe:end]:
            return text[probe : end + 2], end + 2
    return None, after_comma


def remove_member(
    edits: Edits, member: Member, previous: Member | None = None
) -> str | None:
    """Delete ``"key": value``, the comma after it and its same-line comment.

    The blank run before the key goes too when the member sits on its own line,
    so the object does not keep an empty line.  Pass *previous* (the member
    before it) when nothing after it stays: its comma is then removed too, so
    the object does not end on a trailing comma.

    Returns:
        The comment that was on the member's line, so the caller can carry it
        to the key that replaces this one.
    """
    text = edits.text
    start = member.key.start
    line_start = text.rfind("\n", 0, start) + 1
    if text[line_start:start].strip() == "" and line_start > 0:
        start = line_start - 1
    comment, end = trailing_comment(text, member)
    edits._edits.append((start, end, ""))
    if previous is not None:
        probe = previous.value.end
        while probe < len(text) and text[probe] in " \t":
            probe += 1
        if probe < start and text[probe] == ",":
            edits._edits.append((probe, probe + 1, ""))
    return comment


def rename_member(edits: Edits, member: Member, key: str, value_json: str) -> None:
    """Reuse *member*'s slot for ``"key": value_json`` (key and value replaced)."""
    edits.replace_string(member.key, key)
    edits.replace_span(member.value, value_json)
