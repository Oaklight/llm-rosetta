# /// zerodep
# version = "0.3.1"
# deps = []
# tier = "subsystem"
# category = "serialization"
# note = "Install/update via `zerodep add yaml`"
# ///

"""YAML parser and serializer (common subset) — zero dependencies, stdlib only, Python 3.10+.

Part of zerodep: https://github.com/Oaklight/zerodep
Copyright (c) 2026 Peng Ding. MIT License.

Supports the most commonly used YAML features: mappings, sequences,
scalars (str/int/float/bool/null), flow style, block scalars,
multi-document streams, and comments.

Does NOT implement: anchors/aliases, tags, merge keys, complex keys.

Example::

    data = load("name: Alice\nage: 30")
    # {'name': 'Alice', 'age': 30}
    print(dump(data))
    # age: 30
    # name: Alice
"""

from __future__ import annotations

import math
import re
from typing import IO, Any, Iterator, overload

__all__ = [
    "YAMLError",
    "load",
    "load_all",
    "dump",
    "dump_all",
]

# ── Exceptions ─────────────────────────────────────────────────────────────────


class YAMLError(Exception):
    """Raised when YAML parsing fails."""


# ── Scalar type resolution ─────────────────────────────────────────────────────

_NULL_RE = re.compile(r"\A(?:null|Null|NULL|~)\Z")
_BOOL_TRUE_RE = re.compile(r"\A(?:true|True|TRUE|yes|Yes|YES|on|On|ON)\Z")
_BOOL_FALSE_RE = re.compile(r"\A(?:false|False|FALSE|no|No|NO|off|Off|OFF)\Z")
_INT_RE = re.compile(r"\A[-+]?(?:0|[1-9][0-9_]*)\Z")
_INT_HEX_RE = re.compile(r"\A0x[0-9a-fA-F_]+\Z")
_INT_OCT_RE = re.compile(r"\A0[0-7_]+\Z")  # YAML 1.1 octal: 0777 (not 0o777)
_INT_BIN_RE = re.compile(r"\A0b[01_]+\Z")
# YAML 1.1 requires explicit sign (+/-) in exponent when base has decimal point
_FLOAT_RE = re.compile(r"\A[-+]?(?:\.[0-9]+|[0-9]+\.[0-9]*)(?:[eE][-+][0-9]+)?\Z")
_INF_RE = re.compile(r"\A[-+]?\.(?:inf|Inf|INF)\Z")
_NAN_RE = re.compile(r"\A\.(?:nan|NaN|NAN)\Z")


def _resolve_scalar(value: str) -> str | int | float | bool | None:
    """Resolve a plain (unquoted) scalar string to a typed Python value."""
    if not value:
        return None
    if _NULL_RE.match(value):
        return None
    if _BOOL_TRUE_RE.match(value):
        return True
    if _BOOL_FALSE_RE.match(value):
        return False
    if _INT_RE.match(value):
        return int(value.replace("_", ""))
    if _INT_HEX_RE.match(value):
        return int(value.replace("_", ""), 16)
    if _INT_OCT_RE.match(value):
        return int(value.replace("_", ""), 8)  # YAML 1.1: 0777 -> 511
    if _INT_BIN_RE.match(value):
        return int(value.replace("_", ""), 2)
    if _FLOAT_RE.match(value):
        return float(value.replace("_", ""))
    if _INF_RE.match(value):
        return float("-inf") if value.startswith("-") else float("inf")
    if _NAN_RE.match(value):
        return float("nan")
    return value


# ── String unquoting ───────────────────────────────────────────────────────────

_DQ_ESCAPE_MAP = {
    "\\": "\\",
    '"': '"',
    "n": "\n",
    "r": "\r",
    "t": "\t",
    "0": "\0",
    "a": "\a",
    "b": "\b",
    "e": "\x1b",
    "v": "\v",
    "/": "/",
    " ": " ",
    "N": "\x85",
    "_": "\xa0",
}


def _unescape_double_quoted(s: str) -> str:
    """Process escape sequences in a double-quoted YAML string."""
    result: list[str] = []
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            nxt = s[i + 1]
            if nxt in _DQ_ESCAPE_MAP:
                result.append(_DQ_ESCAPE_MAP[nxt])
                i += 2
            elif nxt == "x" and i + 3 < len(s):
                result.append(chr(int(s[i + 2 : i + 4], 16)))
                i += 4
            elif nxt == "u" and i + 5 < len(s):
                result.append(chr(int(s[i + 2 : i + 6], 16)))
                i += 6
            elif nxt == "U" and i + 9 < len(s):
                result.append(chr(int(s[i + 2 : i + 10], 16)))
                i += 10
            else:
                result.append(s[i])
                i += 1
        else:
            result.append(s[i])
            i += 1
    return "".join(result)


def _unquote(s: str) -> str:
    """Remove quotes from a YAML scalar and process escapes if needed."""
    if len(s) >= 2:
        if s[0] == "'" and s[-1] == "'":
            # Single-quoted: only '' escapes to '
            return s[1:-1].replace("''", "'")
        if s[0] == '"' and s[-1] == '"':
            return _unescape_double_quoted(s[1:-1])
    return s


# ── Scanner ────────────────────────────────────────────────────────────────────


class _Line:
    """A logical line with tracked indentation and line number."""

    __slots__ = ("indent", "text", "lineno")

    def __init__(self, indent: int, text: str, lineno: int):
        self.indent = indent
        self.text = text
        self.lineno = lineno

    def __repr__(self) -> str:
        return f"_Line({self.lineno}: indent={self.indent}, {self.text!r})"


def _strip_inline_comment(text: str) -> str:
    """Remove inline comments from a line, respecting quoted strings."""
    in_single = False
    in_double = False
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and in_double and i + 1 < len(text):
            i += 2
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
        elif ch == '"' and not in_single:
            in_double = not in_double
        elif ch == "#" and not in_single and not in_double:
            # Must be preceded by whitespace to be a comment
            if i == 0 or text[i - 1] in (" ", "\t"):
                return text[:i].rstrip()
        i += 1
    return text


def _scan(text: str) -> list[_Line]:
    """Convert raw YAML text into a list of logical lines."""
    lines: list[_Line] = []
    for lineno, raw in enumerate(text.splitlines(), 1):
        # Preserve completely empty lines for block scalar detection
        stripped = raw.lstrip()
        if not stripped or stripped[0] == "#":
            continue
        indent = len(raw) - len(stripped)
        # Don't strip comments from block scalar indicators or document markers
        if stripped.startswith("---") or stripped.startswith("..."):
            lines.append(_Line(indent, stripped, lineno))
        else:
            cleaned = _strip_inline_comment(stripped)
            if cleaned:
                lines.append(_Line(indent, cleaned, lineno))
    return lines


# ── Parser ─────────────────────────────────────────────────────────────────────


class _Parser:
    """Recursive-descent YAML parser for the common subset."""

    def __init__(self, lines: list[_Line], raw_lines: list[str]):
        self._lines = lines
        self._raw_lines = raw_lines  # original text lines for block scalars
        self._pos = 0

    def _peek(self) -> _Line | None:
        if self._pos < len(self._lines):
            return self._lines[self._pos]
        return None

    def _advance(self) -> _Line:
        line = self._lines[self._pos]
        self._pos += 1
        return line

    def _error(self, msg: str, lineno: int | None = None) -> YAMLError:
        if lineno is None:
            cur = self._peek()
            lineno = cur.lineno if cur else -1
        return YAMLError(f"line {lineno}: {msg}")

    # ── Document parsing ──

    def parse_stream(self) -> list[Any]:
        """Parse the entire stream, returning a list of documents."""
        documents: list[Any] = []

        # Skip leading document marker if present
        cur = self._peek()
        if cur is not None and cur.text == "---":
            self._advance()

        while self._pos < len(self._lines):
            doc = self._parse_node(-1)
            documents.append(doc)

            # Check for document separator
            cur = self._peek()
            if cur is not None and cur.text in ("---", "..."):
                self._advance()
                # '...' without following '---' ends the stream
                if cur.text == "...":
                    cur2 = self._peek()
                    if cur2 is None or cur2.text != "---":
                        break
                    self._advance()  # skip the ---

        if not documents:
            documents.append(None)

        return documents

    def _parse_node(self, parent_indent: int) -> Any:
        """Parse a YAML node (mapping, sequence, or scalar)."""
        cur = self._peek()
        if cur is None:
            return None

        # Document end markers
        if cur.text in ("---", "..."):
            return None

        text = cur.text

        # Flow collections
        if text.startswith("{"):
            return self._parse_flow_mapping()
        if text.startswith("["):
            return self._parse_flow_sequence()

        # Block scalar
        if text.startswith("|") or text.startswith(">"):
            return self._parse_block_scalar()

        # Block sequence
        if text.startswith("- ") or text == "-":
            return self._parse_block_sequence(cur.indent)

        # Check if it's a mapping (contains ': ' or ends with ':')
        if self._is_mapping_line(text):
            return self._parse_block_mapping(cur.indent)

        # Plain scalar
        self._advance()
        return self._parse_scalar_value(text)

    @staticmethod
    def _skip_quoted_key(text: str) -> int:
        """Return the index after the quoted prefix of *text*.

        If *text* starts with a single- or double-quoted string the returned
        index points just past the closing quote.  For unquoted text ``0`` is
        returned so callers can scan from the beginning.

        Returns ``-1`` when a single-quoted string has no closing quote (the
        caller should treat this as "not a mapping line").
        """
        if text.startswith("'"):
            end = text.find("'", 1)
            return -1 if end < 0 else end + 1
        if text.startswith('"'):
            i = 1
            while i < len(text):
                if text[i] == "\\" and i + 1 < len(text):
                    i += 2
                    continue
                if text[i] == '"':
                    return i + 1
                i += 1
        return 0

    @staticmethod
    def _find_mapping_colon(text: str, start: int) -> int:
        """Find the position of the mapping separator colon starting from *start*.

        Returns the index of the ``:`` that acts as a mapping separator, or
        ``-1`` if none is found.  A colon qualifies when it is either the last
        character or is immediately followed by a space or tab.
        """
        i = start
        while i < len(text):
            if text[i] == ":":
                if i + 1 == len(text) or text[i + 1] in (" ", "\t"):
                    return i
            i += 1
        return -1

    def _is_mapping_line(self, text: str) -> bool:
        """Check if a line represents a mapping key."""
        i = self._skip_quoted_key(text)
        if i < 0:
            return False
        return self._find_mapping_colon(text, i) >= 0

    def _split_mapping_line(self, text: str) -> tuple[str, str]:
        """Split a mapping line into key and value parts."""
        i = self._skip_quoted_key(text)
        if i < 0:
            return text, ""
        colon = self._find_mapping_colon(text, i)
        if colon < 0:
            return text, ""
        if colon + 1 == len(text):
            return text[:colon], ""
        return text[:colon], text[colon + 2 :].strip()

    # ── Block mapping ──

    def _parse_inline_value(self, raw_value: str, lineno: int) -> Any:
        """Parse an inline mapping or sequence value."""
        if raw_value.startswith(("{", "[")):
            return self._parse_flow_from_text(raw_value)
        if raw_value.startswith(("|", ">")):
            return self._parse_block_scalar_from_indicator(raw_value, lineno)
        return self._parse_scalar_value(raw_value)

    def _at_block_end(self, cur: _Line | None, indent: int) -> bool:
        """Return True when the current line should stop a block collection."""
        if cur is None or cur.indent < indent:
            return True
        return cur.text in ("---", "...") or cur.indent != indent

    def _parse_block_mapping(self, indent: int) -> dict:
        result: dict[Any, Any] = {}
        while True:
            cur = self._peek()
            if self._at_block_end(cur, indent):
                break
            assert cur is not None  # guaranteed by _at_block_end
            if not self._is_mapping_line(cur.text):
                break

            self._advance()
            raw_key, raw_value = self._split_mapping_line(cur.text)
            key = self._parse_scalar_value(raw_key)

            if raw_value:
                value = self._parse_inline_value(raw_value, cur.lineno)
            else:
                nxt = self._peek()
                if nxt is None or nxt.indent <= indent or nxt.text in ("---", "..."):
                    value = None
                else:
                    value = self._parse_node(indent)
            result[key] = value
        return result

    # ── Block sequence ──

    def _parse_sequence_mapping_item(
        self, item_text: str, cur: _Line, indent: int
    ) -> dict:
        """Parse a mapping that starts on a sequence item line (``- key: val``)."""
        raw_key, raw_value = self._split_mapping_line(item_text)
        key = self._parse_scalar_value(raw_key)
        mapping: dict[Any, Any] = {}

        if raw_value:
            mapping[key] = self._parse_inline_value(raw_value, cur.lineno)
        else:
            nxt = self._peek()
            if (
                nxt is not None
                and nxt.indent > indent
                and nxt.text not in ("---", "...")
            ):
                mapping[key] = self._parse_node(indent)
            else:
                mapping[key] = None

        # Continue reading mapping entries at deeper indent
        nxt = self._peek()
        if nxt is not None and nxt.indent > indent and self._is_mapping_line(nxt.text):
            rest = self._parse_block_mapping(nxt.indent)
            mapping.update(rest)

        return mapping

    def _parse_sequence_item(self, item_text: str, cur: _Line, indent: int) -> Any:
        """Parse the value part of a single sequence item."""
        if not item_text:
            nxt = self._peek()
            if nxt is None or nxt.indent <= indent or nxt.text in ("---", "..."):
                return None
            return self._parse_node(indent)
        if item_text.startswith(("{", "[")):
            return self._parse_flow_from_text(item_text)
        if self._is_mapping_line(item_text):
            return self._parse_sequence_mapping_item(item_text, cur, indent)
        return self._parse_scalar_value(item_text)

    def _parse_block_sequence(self, indent: int) -> list:
        result: list[Any] = []
        while True:
            cur = self._peek()
            if self._at_block_end(cur, indent):
                break
            assert cur is not None
            if not (cur.text.startswith("- ") or cur.text == "-"):
                break

            self._advance()
            item_text = cur.text[2:].strip() if cur.text.startswith("- ") else ""
            result.append(self._parse_sequence_item(item_text, cur, indent))

        return result

    # ── Block scalars ──

    def _parse_block_scalar(self) -> str:
        cur = self._advance()
        return self._parse_block_scalar_from_indicator(cur.text, cur.lineno)

    @staticmethod
    def _parse_chomping_indicator(header: str) -> tuple[str, int]:
        """Parse the block scalar header for chomping mode and explicit indent.

        Args:
            header: The portion of the indicator line after ``|`` or ``>``.

        Returns:
            A ``(chomp, explicit_indent)`` tuple.
        """
        chomp = "clip"
        explicit_indent = 0
        for ch in header:
            if ch == "-":
                chomp = "strip"
            elif ch == "+":
                chomp = "keep"
            elif ch.isdigit():
                explicit_indent = int(ch)
        return chomp, explicit_indent

    def _collect_scalar_lines(self, raw_lineno: int, explicit_indent: int) -> list[str]:
        """Collect raw content lines for a block scalar.

        Args:
            raw_lineno: 1-based line number of the indicator (content starts
                on the *next* raw line).
            explicit_indent: Explicit indentation width from the header, or
                ``0`` to auto-detect.

        Returns:
            List of content lines with leading indentation stripped.
        """
        if raw_lineno >= len(self._raw_lines):
            return []

        content_indent = self._detect_scalar_indent(raw_lineno, explicit_indent)
        if content_indent == 0:
            return []

        content_lines: list[str] = []
        for j in range(raw_lineno, len(self._raw_lines)):
            raw = self._raw_lines[j]
            if not raw.strip():
                content_lines.append("")
                continue
            line_indent = len(raw) - len(raw.lstrip())
            if line_indent < content_indent:
                break
            content_lines.append(raw[content_indent:])
        return content_lines

    def _detect_scalar_indent(self, raw_lineno: int, explicit: int) -> int:
        """Determine the content indentation for a block scalar."""
        if explicit > 0:
            return explicit
        for j in range(raw_lineno, len(self._raw_lines)):
            raw = self._raw_lines[j]
            stripped = raw.lstrip()
            if stripped and not stripped.startswith("#"):
                return len(raw) - len(stripped)
        return 0

    @staticmethod
    def _fold_lines(content_lines: list[str]) -> str:
        """Fold content lines (``>`` mode): replace single newlines with spaces."""
        parts: list[str] = []
        for line in content_lines:
            if not line:
                parts.append("\n")
            elif parts and parts[-1] != "\n" and not parts[-1].endswith("\n"):
                parts.append(" " + line)
            else:
                parts.append(line)
        return "".join(parts)

    @staticmethod
    def _apply_chomping(text: str, chomp: str) -> str:
        """Apply the chomping rule (strip / keep / clip) to block scalar text."""
        if chomp == "strip":
            return text
        if chomp == "keep":
            return text + "\n"
        # clip
        return text + "\n" if text else ""

    def _parse_block_scalar_from_indicator(
        self, indicator_line: str, lineno: int
    ) -> str:
        """Parse a | or > block scalar, reading content lines from raw_lines."""
        indicator = indicator_line[0]
        header = indicator_line[1:].strip()

        chomp, explicit_indent = self._parse_chomping_indicator(header)
        content_lines = self._collect_scalar_lines(lineno, explicit_indent)

        # Skip consumed content lines in the scanner
        while True:
            nxt = self._peek()
            if nxt is None:
                break
            if nxt.lineno <= lineno + len(content_lines):
                self._advance()
            else:
                break

        # Remove trailing empty lines
        while content_lines and not content_lines[-1]:
            content_lines.pop()

        if indicator == "|":
            text = "\n".join(content_lines)
        else:
            text = self._fold_lines(content_lines)

        return self._apply_chomping(text, chomp)

    # ── Flow collections ──

    def _parse_flow_mapping(self) -> dict:
        cur = self._advance()
        return self._parse_flow_from_text(cur.text)

    def _parse_flow_sequence(self) -> list:
        cur = self._advance()
        return self._parse_flow_from_text(cur.text)

    def _parse_flow_from_text(self, text: str) -> Any:
        """Parse a flow collection from raw text."""
        tokens = _FlowTokenizer(text)
        return tokens.parse()

    # ── Scalar value ──

    def _parse_scalar_value(self, text: str) -> Any:
        """Parse a scalar value, handling quoting and type resolution."""
        if not text:
            return None
        # Quoted strings
        if (text.startswith("'") and text.endswith("'")) or (
            text.startswith('"') and text.endswith('"')
        ):
            return _unquote(text)
        # Plain scalar
        return _resolve_scalar(text)


# ── Flow tokenizer ────────────────────────────────────────────────────────────


class _FlowTokenizer:
    """Character-level parser for flow-style YAML collections."""

    def __init__(self, text: str):
        self._text = text
        self._pos = 0

    def _skip_ws(self) -> None:
        while self._pos < len(self._text) and self._text[self._pos] in (
            " ",
            "\t",
            "\n",
            "\r",
        ):
            self._pos += 1

    def _peek(self) -> str:
        if self._pos < len(self._text):
            return self._text[self._pos]
        return ""

    def parse(self) -> Any:
        self._skip_ws()
        ch = self._peek()
        if ch == "{":
            return self._parse_mapping()
        if ch == "[":
            return self._parse_sequence()
        return self._parse_value()

    def _parse_mapping(self) -> dict:
        self._pos += 1  # skip {
        result: dict[Any, Any] = {}
        self._skip_ws()
        if self._peek() == "}":
            self._pos += 1
            return result

        while True:
            self._skip_ws()
            key = self._parse_value()
            self._skip_ws()
            if self._peek() == ":":
                self._pos += 1
                self._skip_ws()
                value = self.parse()
            else:
                value = None
            result[key] = value
            self._skip_ws()
            if self._peek() == ",":
                self._pos += 1
                self._skip_ws()
                if self._peek() == "}":
                    self._pos += 1
                    break
            elif self._peek() == "}":
                self._pos += 1
                break
            else:
                break
        return result

    def _parse_sequence(self) -> list:
        self._pos += 1  # skip [
        result: list[Any] = []
        self._skip_ws()
        if self._peek() == "]":
            self._pos += 1
            return result

        while True:
            self._skip_ws()
            item = self.parse()
            result.append(item)
            self._skip_ws()
            if self._peek() == ",":
                self._pos += 1
                self._skip_ws()
                if self._peek() == "]":
                    self._pos += 1
                    break
            elif self._peek() == "]":
                self._pos += 1
                break
            else:
                break
        return result

    def _parse_value(self) -> Any:
        self._skip_ws()
        ch = self._peek()
        if ch == "{":
            return self._parse_mapping()
        if ch == "[":
            return self._parse_sequence()
        if ch == "'":
            return self._parse_single_quoted()
        if ch == '"':
            return self._parse_double_quoted()
        return self._parse_plain_scalar()

    def _parse_single_quoted(self) -> str:
        self._pos += 1  # skip opening '
        parts: list[str] = []
        while self._pos < len(self._text):
            ch = self._text[self._pos]
            if ch == "'":
                if self._pos + 1 < len(self._text) and self._text[self._pos + 1] == "'":
                    parts.append("'")
                    self._pos += 2
                else:
                    self._pos += 1
                    break
            else:
                parts.append(ch)
                self._pos += 1
        return "".join(parts)

    def _parse_double_quoted(self) -> str:
        self._pos += 1  # skip opening "
        parts: list[str] = []
        while self._pos < len(self._text):
            ch = self._text[self._pos]
            if ch == "\\" and self._pos + 1 < len(self._text):
                nxt = self._text[self._pos + 1]
                if nxt in _DQ_ESCAPE_MAP:
                    parts.append(_DQ_ESCAPE_MAP[nxt])
                    self._pos += 2
                else:
                    parts.append(ch)
                    self._pos += 1
            elif ch == '"':
                self._pos += 1
                break
            else:
                parts.append(ch)
                self._pos += 1
        return "".join(parts)

    def _parse_plain_scalar(self) -> Any:
        start = self._pos
        # Read until a flow indicator or end
        while self._pos < len(self._text):
            ch = self._text[self._pos]
            if ch in (",", "}", "]", "{", "[", ":"):
                break
            self._pos += 1
        raw = self._text[start : self._pos].strip()
        if not raw:
            return None
        return _resolve_scalar(raw)


# ── Dumper ─────────────────────────────────────────────────────────────────────

# Characters that force quoting in a plain scalar
_NEEDS_QUOTE_RE = re.compile(
    r"[:\#{}\[\],&*?|>!%@`]"
    r"|^[-?]$"
    r"|^\s"
    r"|\s$"
    r"|\n"
)


class _Dumper:
    """YAML serializer."""

    def __init__(
        self,
        indent: int = 2,
        default_flow_style: bool | None = None,
        sort_keys: bool = True,
        allow_unicode: bool = True,
    ):
        self._indent = indent
        self._flow = default_flow_style
        self._sort_keys = sort_keys
        self._allow_unicode = allow_unicode
        self._seen: set[int] = set()

    def dump(self, data: Any) -> str:
        self._seen.clear()
        result = self._represent(data, 0)
        if result.endswith("\n"):
            return result
        return result + "\n"

    def _represent(self, data: Any, level: int) -> str:
        if isinstance(data, dict):
            return self._represent_mapping(data, level)
        if isinstance(data, (list, tuple)):
            return self._represent_sequence(data, level)
        return self._represent_scalar(data)

    def _represent_mapping(self, data: dict, level: int) -> str:
        if id(data) in self._seen:
            raise YAMLError("Circular reference detected")
        self._seen.add(id(data))

        if not data:
            return "{}"

        if self._flow is True:
            items = []
            keys = sorted(data.keys(), key=str) if self._sort_keys else data.keys()
            for key in keys:
                k = self._represent_scalar(key)
                v = self._represent(data[key], level)
                items.append(f"{k}: {v}")
            self._seen.discard(id(data))
            return "{" + ", ".join(items) + "}"

        lines: list[str] = []
        prefix = " " * (self._indent * level)
        keys = sorted(data.keys(), key=str) if self._sort_keys else data.keys()
        for key in keys:
            k = self._represent_scalar(key)
            val = data[key]
            if isinstance(val, dict) and val:
                lines.append(f"{prefix}{k}:")
                lines.append(self._represent_mapping(val, level + 1))
            elif isinstance(val, (list, tuple)) and val:
                lines.append(f"{prefix}{k}:")
                lines.append(self._represent_sequence(val, level + 1))
            else:
                v = self._represent(val, level + 1)
                lines.append(f"{prefix}{k}: {v}")
        self._seen.discard(id(data))
        return "\n".join(lines)

    def _represent_key_value_line(
        self, key_str: str, val: Any, line_prefix: str, level: int
    ) -> list[str]:
        """Represent a single key-value pair for use inside a sequence item."""
        if isinstance(val, (dict, list, tuple)) and val:
            return [f"{line_prefix}{key_str}:", self._represent(val, level)]
        v = self._represent(val, level)
        return [f"{line_prefix}{key_str}: {v}"]

    def _represent_dict_in_sequence(
        self, item: dict, prefix: str, level: int
    ) -> list[str]:
        """Represent a dict that appears as a sequence item (``- key: val``)."""
        keys = sorted(item.keys(), key=str) if self._sort_keys else list(item.keys())
        first_key = keys[0]
        k = self._represent_scalar(first_key)
        lines = self._represent_key_value_line(
            k, item[first_key], f"{prefix}- ", level + 2
        )
        inner_prefix = prefix + " " * self._indent
        for rk in keys[1:]:
            rk_s = self._represent_scalar(rk)
            lines.extend(
                self._represent_key_value_line(rk_s, item[rk], inner_prefix, level + 2)
            )
        return lines

    def _represent_sequence(self, data: list | tuple, level: int) -> str:
        if id(data) in self._seen:
            raise YAMLError("Circular reference detected")
        self._seen.add(id(data))

        if not data:
            return "[]"

        if self._flow is True:
            items = [self._represent(item, level) for item in data]
            self._seen.discard(id(data))
            return "[" + ", ".join(items) + "]"

        lines: list[str] = []
        prefix = " " * (self._indent * level)
        for item in data:
            if isinstance(item, dict) and item:
                lines.extend(self._represent_dict_in_sequence(item, prefix, level))
            elif isinstance(item, (list, tuple)) and item:
                lines.append(f"{prefix}-")
                lines.append(self._represent_sequence(item, level + 1))
            else:
                v = self._represent(item, level + 1)
                lines.append(f"{prefix}- {v}")

        self._seen.discard(id(data))
        return "\n".join(lines)

    def _represent_scalar(self, data: Any) -> str:
        if data is None:
            return "null"
        if isinstance(data, bool):
            return "true" if data else "false"
        if isinstance(data, int):
            return str(data)
        if isinstance(data, float):
            if math.isnan(data):
                return ".nan"
            if math.isinf(data):
                return ".inf" if data > 0 else "-.inf"
            return str(data)
        if isinstance(data, str):
            return self._represent_str(data)
        return str(data)

    def _represent_str(self, s: str) -> str:
        if not s:
            return "''"

        # Check if it looks like a special YAML value
        if (
            _NULL_RE.match(s)
            or _BOOL_TRUE_RE.match(s)
            or _BOOL_FALSE_RE.match(s)
            or _INT_RE.match(s)
            or _INT_HEX_RE.match(s)
            or _FLOAT_RE.match(s)
            or _INF_RE.match(s)
            or _NAN_RE.match(s)
        ):
            return f"'{s}'"

        # Check if quoting is needed
        if _NEEDS_QUOTE_RE.search(s):
            if "\n" in s:
                # Use literal block scalar for multiline
                return "|\n" + "\n".join("  " + line for line in s.split("\n"))
            # Use single quotes, escaping internal single quotes
            return "'" + s.replace("'", "''") + "'"

        return s


# ── Public API ─────────────────────────────────────────────────────────────────


def load(text: str) -> Any:
    """Parse a YAML string and return a Python object.

    Only produces safe types: dict, list, str, int, float, bool, None.
    Equivalent to PyYAML's ``yaml.safe_load()``.

    Args:
        text: YAML document string.

    Returns:
        Parsed Python object.

    Raises:
        YAMLError: If the YAML is malformed.
    """
    if not text or not text.strip():
        return None
    raw_lines = text.splitlines()
    lines = _scan(text)
    if not lines:
        return None
    parser = _Parser(lines, raw_lines)
    docs = parser.parse_stream()
    return docs[0] if docs else None


def load_all(text: str) -> Iterator[Any]:
    """Parse a multi-document YAML string.

    Yields one Python object per YAML document (separated by ``---``).

    Args:
        text: Multi-document YAML string.

    Yields:
        Parsed Python objects, one per document.
    """
    if not text or not text.strip():
        yield None
        return
    raw_lines = text.splitlines()
    lines = _scan(text)
    if not lines:
        yield None
        return
    parser = _Parser(lines, raw_lines)
    docs = parser.parse_stream()
    yield from docs


@overload
def dump(
    data: Any,
    stream: None = None,
    *,
    default_flow_style: bool | None = None,
    indent: int = 2,
    sort_keys: bool = True,
    allow_unicode: bool = True,
) -> str: ...


@overload
def dump(
    data: Any,
    stream: IO[str],
    *,
    default_flow_style: bool | None = None,
    indent: int = 2,
    sort_keys: bool = True,
    allow_unicode: bool = True,
) -> None: ...


def dump(
    data: Any,
    stream: IO[str] | None = None,
    *,
    default_flow_style: bool | None = None,
    indent: int = 2,
    sort_keys: bool = True,
    allow_unicode: bool = True,
) -> str | None:
    """Serialize a Python object to a YAML string.

    Args:
        data: Python object to serialize.
        stream: If provided, write to this stream and return None.
        default_flow_style: True for flow (inline) style, False for block,
            None for auto (empty collections use flow).
        indent: Number of spaces per indentation level.
        sort_keys: Sort mapping keys alphabetically.
        allow_unicode: Allow unicode characters in output.

    Returns:
        YAML string if *stream* is None, otherwise None.
    """
    dumper = _Dumper(
        indent=indent,
        default_flow_style=default_flow_style,
        sort_keys=sort_keys,
        allow_unicode=allow_unicode,
    )
    result = dumper.dump(data)
    if stream is not None:
        stream.write(result)
        return None
    return result


@overload
def dump_all(
    documents: list[Any] | tuple[Any, ...],
    stream: None = None,
    *,
    default_flow_style: bool | None = None,
    indent: int = 2,
    sort_keys: bool = True,
    allow_unicode: bool = True,
) -> str: ...


@overload
def dump_all(
    documents: list[Any] | tuple[Any, ...],
    stream: IO[str],
    *,
    default_flow_style: bool | None = None,
    indent: int = 2,
    sort_keys: bool = True,
    allow_unicode: bool = True,
) -> None: ...


def dump_all(
    documents: list[Any] | tuple[Any, ...],
    stream: IO[str] | None = None,
    *,
    default_flow_style: bool | None = None,
    indent: int = 2,
    sort_keys: bool = True,
    allow_unicode: bool = True,
) -> str | None:
    """Serialize multiple Python objects as a multi-document YAML string.

    Args:
        documents: Iterable of Python objects to serialize.
        stream: If provided, write to this stream and return None.
        default_flow_style: True for flow style, False for block, None for auto.
        indent: Number of spaces per indentation level.
        sort_keys: Sort mapping keys alphabetically.
        allow_unicode: Allow unicode characters in output.

    Returns:
        YAML string if *stream* is None, otherwise None.
    """
    dumper = _Dumper(
        indent=indent,
        default_flow_style=default_flow_style,
        sort_keys=sort_keys,
        allow_unicode=allow_unicode,
    )
    parts: list[str] = []
    for i, doc in enumerate(documents):
        if i > 0:
            parts.append("---\n")
        parts.append(dumper.dump(doc))
    result = "".join(parts)
    if stream is not None:
        stream.write(result)
        return None
    return result
