"""A small JSON Schema interpreter, shared by the contracts that ship a schema.

:func:`_check` interprets the subset of keywords :data:`SUPPORTED_KEYWORDS`
names, so a schema file is the rule and no copy of it lives in code; no
third-party validator is needed.  Standard library only.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

#: The keywords :func:`_check` understands.  Annotation keywords are
#: read and ignored.  ``tests/contracts`` checks that the schema uses nothing
#: else, so a keyword added to the schema cannot be silently skipped.
SUPPORTED_KEYWORDS: frozenset[str] = frozenset(
    {
        "type",
        "const",
        "enum",
        "pattern",
        "minLength",
        "maxLength",
        "minimum",
        "minItems",
        "uniqueItems",
        "required",
        "properties",
        "additionalProperties",
        "propertyNames",
        "items",
        "$ref",
    }
)
ANNOTATION_KEYWORDS: frozenset[str] = frozenset(
    {"$schema", "$id", "$defs", "title", "description", "default"}
)

_TYPE_CHECKS = {
    "object": lambda v: isinstance(v, Mapping),
    "array": lambda v: isinstance(v, list),
    "string": lambda v: isinstance(v, str),
    "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
    "number": lambda v: isinstance(v, (int, float)) and not isinstance(v, bool),
    "boolean": lambda v: isinstance(v, bool),
    "null": lambda v: v is None,
}


def _resolve_ref(ref: str, root: Mapping[str, Any]) -> Mapping[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"only local $ref is supported, got {ref!r}")
    node: Any = root
    for part in ref[2:].split("/"):
        node = node[part]
    return node


def _check(
    value: Any,
    schema: Mapping[str, Any],
    path: str,
    root: Mapping[str, Any],
    problems: list[str],
) -> None:
    """Append every violation of *schema* by *value* to *problems*."""
    if "$ref" in schema:
        _check(value, _resolve_ref(schema["$ref"], root), path, root, problems)
    if "type" in schema:
        types = schema["type"]
        types = [types] if isinstance(types, str) else list(types)
        if not any(_TYPE_CHECKS[t](value) for t in types):
            problems.append(f"{path}: expected {' or '.join(types)}")
            return
    if "const" in schema and value != schema["const"]:
        problems.append(f"{path}: must be {schema['const']!r}")
    if "enum" in schema and value not in schema["enum"]:
        allowed = ", ".join(repr(x) for x in schema["enum"])
        problems.append(f"{path}: must be one of {allowed}")
    if isinstance(value, str) and len(value) < schema.get("minLength", 0):
        problems.append(f"{path}: shorter than {schema['minLength']} characters")
    if isinstance(value, str) and len(value) > schema.get("maxLength", len(value)):
        problems.append(f"{path}: longer than {schema['maxLength']} characters")
    if isinstance(value, str) and "pattern" in schema:
        if re.search(schema["pattern"], value) is None:
            problems.append(f"{path}: {value!r} does not match {schema['pattern']}")
    if (
        "minimum" in schema
        and _TYPE_CHECKS["number"](value)
        and value < schema["minimum"]
    ):
        problems.append(f"{path}: must be >= {schema['minimum']}")
    if isinstance(value, list):
        _check_array(value, schema, path, root, problems)
    if isinstance(value, Mapping):
        _check_object(value, schema, path, root, problems)


def _check_array(
    value: list[Any],
    schema: Mapping[str, Any],
    path: str,
    root: Mapping[str, Any],
    problems: list[str],
) -> None:
    if len(value) < schema.get("minItems", 0):
        problems.append(f"{path}: needs at least {schema['minItems']} item(s)")
    if schema.get("uniqueItems"):
        seen: list[Any] = []
        for item in value:
            if item in seen:
                problems.append(f"{path}: duplicate item {item!r}")
            seen.append(item)
    if "items" in schema:
        for index, item in enumerate(value):
            _check(item, schema["items"], f"{path}[{index}]", root, problems)


def _check_object(
    value: Mapping[str, Any],
    schema: Mapping[str, Any],
    path: str,
    root: Mapping[str, Any],
    problems: list[str],
) -> None:
    for key in schema.get("required", ()):
        if key not in value:
            problems.append(f"{path}: missing required key {key!r}")
    properties = schema.get("properties", {})
    names = schema.get("propertyNames")
    extra = schema.get("additionalProperties", True)
    for key, item in value.items():
        where = f"{path}.{key}"
        if names is not None:
            _check(key, names, f"{path} key {key!r}", root, problems)
        if key in properties:
            _check(item, properties[key], where, root, problems)
        elif extra is False:
            problems.append(f"{path}: unknown key {key!r}")
        elif isinstance(extra, Mapping):
            _check(item, extra, where, root, problems)
