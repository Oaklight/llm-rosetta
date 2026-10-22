# /// zerodep
# version = "1.0.0"
# deps = []
# tier = "simple"
# category = "serialization"
# note = "Install/update via: https://zerodep.readthedocs.io/en/latest/guide/cli/"
# ///

"""Extended JSON parser — zero dependencies, stdlib only, Python 3.10+.

Part of zerodep: https://github.com/Oaklight/zerodep
Copyright (c) 2026 Peng Ding. MIT License.

Drop-in replacement for commentjson / stdlib json with JSONC support,
plus JSONL (JSON Lines) helpers.

Supports:
    - Single-line comments: ``//`` and ``#``
    - Block comments: ``/* ... */``
    - Trailing commas in objects and arrays
    - All standard JSON types
    - JSONL (one JSON value per line, with optional comments)

Example::

    loads('{"a": 1, // comment\n"b": 2}')
    # {'a': 1, 'b': 2}
    load(open("config.jsonc"))
    # {...}

    # JSONL with comments
    loads_lines('{"a":1}\n// skip\n{"b":2}')
    # [{'a': 1}, {'b': 2}]
"""

from __future__ import annotations

import json
import re
from typing import IO, Any

__all__ = [
    "JSONCDecodeError",
    "loads",
    "load",
    "dumps",
    "dump",
    "loads_lines",
    "load_lines",
    "dumps_lines",
    "dump_lines",
]

# ── Comment stripping ─────────────────────────────────────────────────────────

# Matches (in priority order):
#   1. Double-quoted strings (group 1) — preserved as-is
#   2. Single-line // comments — removed
#   3. Single-line # comments — removed
#   4. Block /* ... */ comments — removed
_COMMENT_RE = re.compile(
    r"""
    ( "(?:[^"\\]|\\.)*" )   # group 1: double-quoted string (keep)
    | //[^\n]*              # single-line // comment
    | \#[^\n]*              # single-line # comment
    | /\*[\s\S]*?\*/        # block /* */ comment
    """,
    re.VERBOSE | re.MULTILINE,
)

# Trailing comma before closing ] or }
_TRAILING_COMMA_RE = re.compile(
    r"""
    ( "(?:[^"\\]|\\.)*" )   # group 1: double-quoted string (keep)
    | ,\s*(?=[}\]])         # comma followed by optional whitespace then } or ]
    """,
    re.VERBOSE,
)


def _strip_comments(text: str) -> str:
    """Remove ``//``, ``#``, and ``/* */`` comments, preserving strings."""

    def _replace(m: re.Match[str]) -> str:
        if m.group(1) is not None:
            return m.group(1)
        return ""

    return _COMMENT_RE.sub(_replace, text)


def _strip_trailing_commas(text: str) -> str:
    """Remove trailing commas before ``}`` and ``]``, preserving strings."""

    def _replace(m: re.Match[str]) -> str:
        if m.group(1) is not None:
            return m.group(1)
        return ""

    return _TRAILING_COMMA_RE.sub(_replace, text)


def _preprocess(text: str) -> str:
    """Strip comments and trailing commas from JSONC text."""
    text = _strip_comments(text)
    text = _strip_trailing_commas(text)
    return text


# ── Line-number error mapping ────────────────────────────────────────────────


def _remap_error_position(
    original: str, cleaned: str, clean_pos: int
) -> tuple[int, int]:
    """Map a character offset in *cleaned* text back to *original* line/col.

    Returns (line, column), both 1-based.
    """
    # Build a mapping from cleaned-offset → original-offset by replaying the
    # comment-stripping regex.  Each matched region in the original either maps
    # 1:1 (strings) or collapses to zero length (comments).
    orig_idx = 0
    clean_idx = 0
    mapping: list[tuple[int, int, int]] = []  # (clean_start, clean_end, orig_start)

    for m in _COMMENT_RE.finditer(original):
        # Characters before this match map 1:1
        pre_len = m.start() - orig_idx
        if pre_len > 0:
            mapping.append((clean_idx, clean_idx + pre_len, orig_idx))
            clean_idx += pre_len
        orig_idx = m.start()

        if m.group(1) is not None:
            # Preserved string — maps 1:1
            span_len = m.end() - m.start()
            mapping.append((clean_idx, clean_idx + span_len, orig_idx))
            clean_idx += span_len
        # else: comment — removed, clean_idx doesn't advance

        orig_idx = m.end()

    # Tail after last match
    tail_len = len(original) - orig_idx
    if tail_len > 0:
        mapping.append((clean_idx, clean_idx + tail_len, orig_idx))

    # Look up clean_pos in the mapping
    orig_offset = clean_pos  # fallback
    for cs, ce, os_ in mapping:
        if cs <= clean_pos < ce:
            orig_offset = os_ + (clean_pos - cs)
            break
        if clean_pos < cs:
            orig_offset = os_
            break

    # Convert original offset to line/col
    line = original[:orig_offset].count("\n") + 1
    last_nl = original.rfind("\n", 0, orig_offset)
    col = orig_offset - last_nl  # 1-based
    return line, col


class JSONCDecodeError(json.JSONDecodeError):
    """Error raised when JSONC parsing fails.

    Provides line and column numbers relative to the original JSONC source
    (before comment/trailing-comma stripping).
    """


# ── Public API ────────────────────────────────────────────────────────────────


def loads(
    text: str,
    *,
    cls: type[json.JSONDecoder] | None = None,
    object_hook: Any = None,
    parse_float: Any = None,
    parse_int: Any = None,
    parse_constant: Any = None,
    object_pairs_hook: Any = None,
    **kwargs: Any,
) -> Any:
    """Deserialize a JSONC string to a Python object.

    Strips ``//``, ``#``, and ``/* */`` comments and trailing commas before
    delegating to :func:`json.loads`.

    Args:
        text: JSONC source string.
        cls: Custom JSON decoder class.
        object_hook: Called with the result of any object literal decoded.
        parse_float: Called with every JSON float string decoded.
        parse_int: Called with every JSON int string decoded.
        parse_constant: Called with ``-Infinity``, ``Infinity``, ``NaN``.
        object_pairs_hook: Called with an ordered list of pairs.
        **kwargs: Additional keyword arguments passed to :func:`json.loads`.

    Returns:
        Deserialized Python object.

    Raises:
        JSONCDecodeError: If the text is not valid JSONC.
    """
    cleaned = _preprocess(text)
    try:
        return json.loads(
            cleaned,
            cls=cls,
            object_hook=object_hook,
            parse_float=parse_float,
            parse_int=parse_int,
            parse_constant=parse_constant,
            object_pairs_hook=object_pairs_hook,
            **kwargs,
        )
    except json.JSONDecodeError as exc:
        line, col = _remap_error_position(text, cleaned, exc.pos)
        raise JSONCDecodeError(exc.msg, text, exc.pos) from None


def load(
    fp: IO[str],
    *,
    cls: type[json.JSONDecoder] | None = None,
    object_hook: Any = None,
    parse_float: Any = None,
    parse_int: Any = None,
    parse_constant: Any = None,
    object_pairs_hook: Any = None,
    **kwargs: Any,
) -> Any:
    """Deserialize a JSONC file to a Python object.

    Args:
        fp: A text file-like object containing JSONC.
        cls: Custom JSON decoder class.
        object_hook: Called with the result of any object literal decoded.
        parse_float: Called with every JSON float string decoded.
        parse_int: Called with every JSON int string decoded.
        parse_constant: Called with ``-Infinity``, ``Infinity``, ``NaN``.
        object_pairs_hook: Called with an ordered list of pairs.
        **kwargs: Additional keyword arguments passed to :func:`json.loads`.

    Returns:
        Deserialized Python object.

    Raises:
        JSONCDecodeError: If the content is not valid JSONC.
    """
    return loads(
        fp.read(),
        cls=cls,
        object_hook=object_hook,
        parse_float=parse_float,
        parse_int=parse_int,
        parse_constant=parse_constant,
        object_pairs_hook=object_pairs_hook,
        **kwargs,
    )


def dumps(
    obj: Any,
    *,
    skipkeys: bool = False,
    ensure_ascii: bool = True,
    check_circular: bool = True,
    allow_nan: bool = True,
    cls: type[json.JSONEncoder] | None = None,
    indent: int | str | None = None,
    separators: tuple[str, str] | None = None,
    default: Any = None,
    sort_keys: bool = False,
    **kwargs: Any,
) -> str:
    """Serialize a Python object to a JSON string.

    This is a pass-through to :func:`json.dumps` for API compatibility.

    Args:
        obj: Python object to serialize.
        skipkeys: Skip keys that are not basic types.
        ensure_ascii: Escape non-ASCII characters.
        check_circular: Check for circular references.
        allow_nan: Allow ``NaN``, ``Infinity``, ``-Infinity``.
        cls: Custom JSON encoder class.
        indent: Indentation level for pretty-printing.
        separators: Item and key separators.
        default: Called for objects that are not serializable.
        sort_keys: Sort dictionary keys.
        **kwargs: Additional keyword arguments passed to :func:`json.dumps`.

    Returns:
        JSON string.
    """
    return json.dumps(
        obj,
        skipkeys=skipkeys,
        ensure_ascii=ensure_ascii,
        check_circular=check_circular,
        allow_nan=allow_nan,
        cls=cls,
        indent=indent,
        separators=separators,
        default=default,
        sort_keys=sort_keys,
        **kwargs,
    )


def dump(
    obj: Any,
    fp: IO[str],
    *,
    skipkeys: bool = False,
    ensure_ascii: bool = True,
    check_circular: bool = True,
    allow_nan: bool = True,
    cls: type[json.JSONEncoder] | None = None,
    indent: int | str | None = None,
    separators: tuple[str, str] | None = None,
    default: Any = None,
    sort_keys: bool = False,
    **kwargs: Any,
) -> None:
    """Serialize a Python object to a JSON file.

    This is a pass-through to :func:`json.dump` for API compatibility.

    Args:
        obj: Python object to serialize.
        fp: A text file-like object to write to.
        skipkeys: Skip keys that are not basic types.
        ensure_ascii: Escape non-ASCII characters.
        check_circular: Check for circular references.
        allow_nan: Allow ``NaN``, ``Infinity``, ``-Infinity``.
        cls: Custom JSON encoder class.
        indent: Indentation level for pretty-printing.
        separators: Item and key separators.
        default: Called for objects that are not serializable.
        sort_keys: Sort dictionary keys.
        **kwargs: Additional keyword arguments passed to :func:`json.dump`.
    """
    json.dump(
        obj,
        fp,
        skipkeys=skipkeys,
        ensure_ascii=ensure_ascii,
        check_circular=check_circular,
        allow_nan=allow_nan,
        cls=cls,
        indent=indent,
        separators=separators,
        default=default,
        sort_keys=sort_keys,
        **kwargs,
    )


# ── JSONL (JSON Lines) API ────────────────────────────────────────────────────────────

# JSONL = one JSON value per line.  Each line is preprocessed through the
# JSONC comment/trailing-comma stripper, so "JSONL with comments" works
# out of the box.


def _is_blank_or_comment(line: str) -> bool:
    """Return *True* if *line* is empty or a pure-comment line."""
    s = line.strip()
    return s == "" or s.startswith("//") or s.startswith("#")


def loads_lines(
    text: str,
    **kwargs: Any,
) -> list[Any]:
    """Deserialize a JSONL string to a list of Python objects.

    Uses a batch fast-path for clean JSONL: non-empty lines are joined
    into a JSON array and parsed in a single :func:`json.loads` call,
    matching ``ndjson``-level performance.  If that fails (comments,
    trailing commas, etc.), falls back to per-line JSONC processing.

    Args:
        text: JSONL source string (one JSON value per line).
        **kwargs: Keyword arguments forwarded to :func:`json.loads`
            (or :func:`loads` on fallback).

    Returns:
        List of deserialized Python objects.

    Raises:
        JSONCDecodeError: If any line contains invalid JSON/JSONC.
    """
    lines = text.split("\n")
    stripped = [ln for ln in lines if ln.strip()]
    if not stripped:
        return []
    # Fast path: join non-empty lines into a JSON array, single parse.
    # This avoids per-line Python overhead and lets the C json parser
    # handle everything in one call.
    _json_loads = json.loads
    try:
        return _json_loads("[" + ",".join(stripped) + "]", **kwargs)
    except json.JSONDecodeError:
        pass
    # Slow path: comments, trailing commas, or JSONC syntax present.
    results: list[Any] = []
    for line in lines:
        if _is_blank_or_comment(line):
            continue
        try:
            results.append(_json_loads(line, **kwargs))
        except json.JSONDecodeError:
            results.append(loads(line, **kwargs))
    return results


def load_lines(
    fp: IO[str],
    **kwargs: Any,
) -> list[Any]:
    """Deserialize a JSONL file to a list of Python objects.

    Args:
        fp: A text file-like object containing JSONL.
        **kwargs: Keyword arguments forwarded to :func:`loads`.

    Returns:
        List of deserialized Python objects.

    Raises:
        JSONCDecodeError: If any line contains invalid JSON/JSONC.
    """
    return loads_lines(fp.read(), **kwargs)


def dumps_lines(
    objects: list[Any],
    **kwargs: Any,
) -> str:
    """Serialize a list of Python objects to a JSONL string.

    Each object is serialized on its own line using :func:`json.dumps`.
    No trailing newline is emitted (consistent with ``"\\n".join(...)``
    semantics).

    Args:
        objects: Iterable of Python objects to serialize.
        **kwargs: Keyword arguments forwarded to :func:`json.dumps`.

    Returns:
        JSONL string (one JSON value per line).
    """
    return "\n".join(json.dumps(obj, **kwargs) for obj in objects)


def dump_lines(
    objects: list[Any],
    fp: IO[str],
    **kwargs: Any,
) -> None:
    """Serialize a list of Python objects to a JSONL file.

    Args:
        objects: Iterable of Python objects to serialize.
        fp: A text file-like object to write to.
        **kwargs: Keyword arguments forwarded to :func:`json.dumps`.
    """
    fp.write(dumps_lines(objects, **kwargs))
    fp.write("\n")
