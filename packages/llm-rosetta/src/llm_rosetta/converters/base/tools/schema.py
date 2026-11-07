"""JSON Schema sanitization for LLM provider compatibility.

Recursively strips unsupported keywords, resolves ``$ref`` references,
normalizes ``anyOf``/``oneOf``/``allOf`` combination patterns, and reads
stringified numeric constraints back into numbers, for broad provider
compatibility.

Used by all provider tool-definition converters to clean parameter schemas
before sending them upstream.
"""

import json
import math
import re
from typing import Any

from llm_rosetta.conversion_warnings import DOWNGRADED, report_loss

# JSON Schema keywords not supported by major LLM provider endpoints.
# These are valid per the JSON Schema spec but upstream servers reject
# them (e.g. with Pydantic ``extra='forbid'`` validation errors).
UNSUPPORTED_SCHEMA_KEYS: set[str] = {
    "propertyNames",
    "const",
    "$schema",
    "$comment",
    "$id",
    "$anchor",
    "$dynamicAnchor",
    "$dynamicRef",
    "ref",
    "contentEncoding",
    "contentMediaType",
    "contentSchema",
    "deprecated",
    "readOnly",
    "writeOnly",
    "examples",
    # JSON Schema draft 6+ numeric constraints — not supported by
    # Google GenAI and some other providers.
    "exclusiveMinimum",
    "exclusiveMaximum",
}

# Keys that hold definition maps (consumed for $ref resolution, then removed).
_DEFS_KEYS: set[str] = {"$defs", "definitions"}

# Schema keywords whose value is a count — a non-negative integer.
_INTEGER_CONSTRAINT_KEYWORDS: frozenset[str] = frozenset(
    {
        "minLength",
        "maxLength",
        "minItems",
        "maxItems",
        "minProperties",
        "maxProperties",
    }
)

# Schema keywords whose value is any JSON number, negative or fractional.
_NUMBER_CONSTRAINT_KEYWORDS: frozenset[str] = frozenset(
    {"minimum", "maximum", "multipleOf"}
)

# ``^[0-9]+$``, spelled out rather than ``\d+``: ``\d`` also matches digits of
# other scripts, and a count written in one is not what these keywords take.
_INTEGER_TEXT = re.compile(r"[0-9]+")


def _parse_json_number(text: str) -> int | float | None:
    """Read *text* as a JSON number, or return ``None`` when it is not one.

    ``json.loads`` is the arbiter, with three spellings it admits that JSON
    does not mean here: ``True``/``False`` (not numbers at all), ``NaN`` and
    ``Infinity`` (not JSON), and a leading zero such as ``"01"`` (which
    ``json.loads`` rejects outright).

    Args:
        text: The string spelling to read.

    Returns:
        The number, or ``None``.
    """
    try:
        value = json.loads(text)
    except ValueError:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _normalized_constraint(keyword: str, value: Any) -> Any:
    """Return *value* as the number *keyword* takes, or unchanged.

    The keyword decides what can be read with certainty.  A count keyword
    takes a non-negative integer and nothing else, so ``"1.0"``, ``"1e2"``
    and ``"-1"`` stay as they came — each is a number, but not a count.
    ``minimum``/``maximum``/``multipleOf`` take any JSON number, so those
    spellings are read.

    Args:
        keyword: The schema keyword the value belongs to.
        value: Its value, possibly a string.

    Returns:
        The number, or *value* itself when nothing can be read from it.
    """
    if not isinstance(value, str):
        return value
    if keyword in _INTEGER_CONSTRAINT_KEYWORDS:
        return int(value) if _INTEGER_TEXT.fullmatch(value) else value
    parsed = _parse_json_number(value)
    return value if parsed is None else parsed


def _is_constraint_keyword(key: str) -> bool:
    """Whether *key* is one of the numeric constraint keywords."""
    return key in _INTEGER_CONSTRAINT_KEYWORDS or key in _NUMBER_CONSTRAINT_KEYWORDS


def _unreadable_constraint(path: str, keyword: str, value: str) -> str:
    """Describe a constraint string that could not be read as its number."""
    kind = "integer" if keyword in _INTEGER_CONSTRAINT_KEYWORDS else "number"
    return (
        f"{path}: {value!r} is not a valid JSON {kind} for this keyword; "
        "forwarded unchanged, so a target that type-checks the schema may "
        "reject it"
    )


def _normalized_entry(
    keyword: str, value: Any, path: str, issues: list[str] | None
) -> tuple[Any, bool]:
    """Read one constraint keyword's value, reporting it when it cannot be read.

    Returns:
        The value to use, and whether it differs from the one passed in.
    """
    normalized = _normalized_constraint(keyword, value)
    if normalized is value:
        if issues is not None and isinstance(value, str):
            issues.append(_unreadable_constraint(path, keyword, value))
        return value, False
    return normalized, True


def normalize_numeric_constraints(
    schema: Any, *, path: str = "", issues: list[str] | None = None
) -> Any:
    """Return *schema* with stringified numeric constraints read as numbers.

    Some clients serialize every JSON Schema keyword as a string, so
    ``{"minLength": 1}`` arrives as ``{"minLength": "1"}``.  A strict target
    rejects that with a schema validation error, though the intent is not in
    doubt (#1426).

    Only what can be read with certainty is rewritten; see
    :func:`_normalized_constraint` for what each keyword class admits.
    Anything else — ``"1.5"`` for a count, ``"abc"``, ``"true"`` — is returned
    untouched and, when *issues* is given, noted there: forwarding a value we
    cannot parse is the source's problem to see resolved, and guessing at it
    would change a constraint into a different one.

    The input is not mutated, and a subtree needing no change comes back as
    the same object, so a caller may compare by identity to tell whether
    anything was read.

    Args:
        schema: A JSON Schema node, or any value inside one.
        path: JSON path of *schema* at its root, used in *issues*.
        issues: Optional list receiving one line per string that could not be
            read as the number its keyword takes.  The value is forwarded
            unchanged in every such case.

    Returns:
        The normalized node.
    """
    if isinstance(schema, dict):
        changed = False
        result: dict[str, Any] = {}
        for key, value in schema.items():
            child_path = f"{path}.{key}" if path else key
            if _is_constraint_keyword(key):
                value, read = _normalized_entry(key, value, child_path, issues)
                changed = changed or read
            child = normalize_numeric_constraints(value, path=child_path, issues=issues)
            if child is not value:
                changed = True
            result[key] = child
        return result if changed else schema
    if isinstance(schema, list):
        changed = False
        items: list[Any] = []
        for index, item in enumerate(schema):
            child = normalize_numeric_constraints(
                item, path=f"{path}[{index}]", issues=issues
            )
            if child is not item:
                changed = True
            items.append(child)
        return items if changed else schema
    return schema


def _deep_merge_schema(base: dict[str, Any], overlay: dict[str, Any]) -> None:
    """Merge overlay into base, deep-merging 'properties' dicts.

    Regular keys are overwritten by overlay values. The 'properties' key
    is special-cased: if both base and overlay contain a 'properties' dict,
    they are merged (overlay wins on conflict) instead of replaced.

    Args:
        base: Target dict to merge into (mutated in place).
        overlay: Source dict whose entries are merged into base.
    """
    for key, value in overlay.items():
        if (
            key == "properties"
            and key in base
            and isinstance(base[key], dict)
            and isinstance(value, dict)
        ):
            base[key] = {**base[key], **value}
        else:
            base[key] = value


def _flatten_combination(schema: dict[str, Any]) -> dict[str, Any]:
    """Normalize ``anyOf``/``oneOf``/``allOf`` combination schemas.

    Single-variant nullable unions (e.g. ``[{"type": "string"}, {"type": "null"}]``)
    are unwrapped to ``{"type": "string", "nullable": true}``.

    Genuine multi-type unions (2+ non-null branches) are preserved as ``anyOf``
    so that providers receive the full type information.  The null branch is
    stripped and replaced with ``"nullable": true``.

    ``allOf`` with a single element is simply unwrapped.

    Args:
        schema: A schema dict that may contain ``anyOf``/``oneOf``/``allOf``.

    Returns:
        A new dict with combination keywords resolved.
    """
    for keyword in ("anyOf", "oneOf"):
        variants = schema.get(keyword)
        if not isinstance(variants, list):
            continue

        non_null = [v for v in variants if v.get("type") != "null"]
        has_null = len(non_null) < len(variants)

        # Preserve sibling metadata (description, title, etc.)
        base: dict[str, Any] = {
            k: v for k, v in schema.items() if k not in ("anyOf", "oneOf", "allOf")
        }

        if len(non_null) == 1:
            # Common nullable pattern: merge the single real type
            _deep_merge_schema(base, non_null[0])
        elif len(non_null) > 1:
            # Genuine multi-type union: preserve all non-null branches
            base["anyOf"] = non_null
        # else: all variants are null → just mark nullable

        if has_null:
            base["nullable"] = True

        return base

    # allOf with a single element: unwrap
    all_of = schema.get("allOf")
    if isinstance(all_of, list) and len(all_of) == 1 and isinstance(all_of[0], dict):
        base = {k: v for k, v in schema.items() if k != "allOf"}
        _deep_merge_schema(base, all_of[0])
        return base

    return schema


def _resolve_ref(ref: str, defs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Resolve a JSON Schema ``$ref`` pointer against collected definitions.

    Only local definition references (``#/$defs/Name`` or
    ``#/definitions/Name``) are supported.  Unresolvable refs return an
    empty dict so the caller can proceed without crashing.

    Args:
        ref: The ``$ref`` string value.
        defs: Merged definitions from ``$defs`` and ``definitions``.

    Returns:
        The referenced schema dict, or ``{}`` if unresolvable.
    """
    for prefix in ("#/$defs/", "#/definitions/"):
        if ref.startswith(prefix):
            name = ref[len(prefix) :]
            return defs.get(name, {})
    return {}


def _sanitize_value(
    value: Any,
    defs: dict[str, dict[str, Any]],
    extra_strip_keys: set[str] | None,
) -> Any:
    """Recursively sanitize a single schema value (dict, list, or passthrough)."""
    if isinstance(value, dict):
        return _sanitize_schema_impl(value, defs, extra_strip_keys)
    if isinstance(value, list):
        return [
            _sanitize_schema_impl(item, defs, extra_strip_keys)
            if isinstance(item, dict)
            else item
            for item in value
        ]
    return value


def _strip_orphaned_required(result: dict[str, Any]) -> None:
    """Remove ``required`` entries that reference non-existent properties."""
    if "required" not in result:
        return
    if "properties" not in result:
        del result["required"]
        return
    props = result["properties"]
    if not isinstance(props, dict) or not isinstance(result["required"], list):
        return
    valid = [r for r in result["required"] if r in props]
    if valid:
        result["required"] = valid
    else:
        del result["required"]


def _sanitize_schema_impl(
    schema: dict[str, Any],
    defs: dict[str, dict[str, Any]] | None = None,
    extra_strip_keys: set[str] | None = None,
) -> dict[str, Any]:
    """Core implementation of schema sanitization (uncached).

    Recursively removes unsupported JSON Schema keywords, resolves
    ``$ref`` references, and flattens ``anyOf``/``oneOf``/``allOf``.
    """
    if defs is None:
        defs = {}
        for key in _DEFS_KEYS:
            d = schema.get(key)
            if isinstance(d, dict):
                defs.update(d)

    # Resolve $ref: inline the referenced definition (merge siblings).
    ref = schema.get("$ref")
    if isinstance(ref, str) and defs:
        resolved = _resolve_ref(ref, defs)
        if resolved:
            merged = {k: v for k, v in schema.items() if k != "$ref"}
            _deep_merge_schema(merged, resolved)
            return _sanitize_schema_impl(merged, defs, extra_strip_keys)

    strip_keys = UNSUPPORTED_SCHEMA_KEYS | (extra_strip_keys or set())
    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key in strip_keys or key in _DEFS_KEYS or key == "$ref":
            continue
        result[key] = _sanitize_value(value, defs, extra_strip_keys)

    # Flatten combination keywords (anyOf/oneOf/allOf) into simple types.
    if result.keys() & {"anyOf", "oneOf", "allOf"}:
        result = _flatten_combination(result)

    _strip_orphaned_required(result)
    return result


def sanitize_schema(
    schema: dict[str, Any],
    defs: dict[str, dict[str, Any]] | None = None,
    extra_strip_keys: set[str] | None = None,
    *,
    path: str = "",
    issues: list[str] | None = None,
) -> dict[str, Any]:
    """Recursively remove unsupported JSON Schema keywords.

    Also resolves ``$ref`` references by inlining the referenced definition,
    normalizes ``anyOf``/``oneOf``/``allOf`` combination keywords —
    single-variant nullable unions are unwrapped, while genuine multi-type
    unions are preserved — and reads stringified numeric constraints
    (``"minLength": "1"``) back into numbers.

    Results are cached at the top-level entry point (where ``defs is None``)
    to avoid redundant work when the same tool schemas are converted
    repeatedly across conversation turns.  The cache is keyed on the
    *normalized* schema, so a stringified constraint and its number reach the
    same entry; *issues* is filled before the lookup, so it does not depend on
    a cache miss.

    Args:
        schema: A JSON Schema dict (or sub-schema).
        defs: Collected ``$defs``/``definitions`` from the top-level schema.
            Populated automatically on the first call if the schema contains
            definition maps.
        extra_strip_keys: Additional provider-specific keys to strip
            (e.g. ``{"additionalProperties"}`` for Google GenAI).
        path: JSON path of *schema* within the tool definition, used in
            *issues* (e.g. ``"tools[0].parameters"``).
        issues: Optional list receiving one line per constraint string that
            could not be read as a number, with its *path*.  The value itself
            is forwarded unchanged.

    Returns:
        A new dict with unsupported keys removed at every level.
    """
    # Only cache top-level calls (defs=None is the public entry point).
    # Recursive calls from _sanitize_value go through _sanitize_schema_impl
    # directly and already have defs populated.
    if defs is not None:
        return _sanitize_schema_impl(schema, defs, extra_strip_keys)

    from ..helpers.cache import _SENTINEL, sanitize_cache, schema_cache_key

    schema = normalize_numeric_constraints(schema, path=path, issues=issues)
    frozen_extra = frozenset(extra_strip_keys) if extra_strip_keys else None
    key = schema_cache_key(schema, frozen_extra)
    cached = sanitize_cache.get(key)
    if cached is not _SENTINEL:
        return cached

    result = _sanitize_schema_impl(schema, defs, extra_strip_keys)
    sanitize_cache.put(key, result)
    return result


def tool_name_for_warning(tool: Any) -> tuple[str, str | None]:
    """The tool's client-facing name, and the upstream one when it differs.

    A name pass may have rewritten the client's spelling — a slash in an MCP
    tool name, a collision qualified — before the tool reaches this boundary,
    and a warning about the tool has to be actionable: the operator holds the
    request the client wrote, so the name in it is the one to lead with.  The
    upstream spelling is kept beside it because that is what appears in the
    body the target rejected.

    Args:
        tool: An IR tool definition.

    Returns:
        ``(client_name, upstream_name)``, the second ``None`` when the two
        agree.
    """
    upstream = tool.get("name") or ""
    metadata = tool.get("metadata")
    client = metadata.get("_original_name") if isinstance(metadata, dict) else None
    client = client or upstream
    return client, (upstream if upstream != client else None)


def sanitize_tool_parameters(
    parameters: dict[str, Any],
    *,
    tool: Any = None,
    tool_name: str = "",
    upstream_name: str | None = None,
    tool_index: int | None = None,
    extra_strip_keys: set[str] | None = None,
    warnings: Any = None,
) -> dict[str, Any]:
    """Sanitize one tool's ``parameters``, reporting what could not be repaired.

    The tool-definition boundary every converter writes its schema through.
    A constraint string that cannot be read is forwarded exactly as it came —
    the source's intent is unknown, and a guess would quietly change what the
    schema constrains — so the conversion warning is the only sign the target
    may reject the tool.  It names the tool and the JSON path of the value
    (#1426).

    Args:
        parameters: The IR tool's ``parameters`` schema.
        tool: The IR tool definition, when the caller has it.  The names the
            warning carries are taken from it — see
            :func:`tool_name_for_warning` — and this wins over *tool_name*.
        tool_name: The tool's client-facing name, for a caller holding only the
            schema.
        upstream_name: The name the tool goes upstream under, when a name pass
            rewrote it; the warning then carries both spellings.
        tool_index: The tool's position in the list being converted, used to
            spell the path ``tools[i].parameters``.  Omitted by a caller that
            has only one tool in hand; the path is then ``parameters``.
        extra_strip_keys: Additional provider-specific keys to strip.
        warnings: The conversion's warning sink, when the caller has one.

    Returns:
        The sanitized schema.
    """
    if tool is not None:
        tool_name, upstream_name = tool_name_for_warning(tool)
    path = (
        f"tools[{tool_index}].parameters"
        if isinstance(tool_index, int)
        else "parameters"
    )
    label = f"Tool {tool_name!r}"
    if upstream_name and upstream_name != tool_name:
        label = f"Tool {tool_name!r} (upstream {upstream_name!r})"
    issues: list[str] = []
    result = sanitize_schema(
        parameters, extra_strip_keys=extra_strip_keys, path=path, issues=issues
    )
    for issue in issues:
        report_loss(
            f"{label} carries a schema constraint that cannot be read as "
            f"written: {issue}",
            code=DOWNGRADED,
            subject=tool_name,
            warnings=warnings,
        )
    return result


# ---- nullable → type-array conversion ----


def _inject_null_type(result: dict[str, Any]) -> None:
    """Inject ``"null"`` into an already-copied schema node's type declaration."""
    if "type" in result:
        current_type = result["type"]
        if isinstance(current_type, str):
            result["type"] = [current_type, "null"]
        elif isinstance(current_type, list) and "null" not in current_type:
            result["type"] = [*current_type, "null"]
    else:
        for kw in ("anyOf", "oneOf"):
            if kw in result and isinstance(result[kw], list):
                null_variant = {"type": "null"}
                if null_variant not in result[kw]:
                    result[kw] = [*result[kw], null_variant]
                break


def _nullable_to_type_array(schema: dict[str, Any]) -> dict[str, Any]:
    """Recursively convert ``"nullable": true`` to standard JSON Schema type arrays.

    Transforms ``{"type": "string", "nullable": true}`` into
    ``{"type": ["string", "null"]}`` and removes the ``nullable`` key.
    When ``type`` is already a list, appends ``"null"`` if absent.
    When ``type`` is missing, only strips the ``nullable`` key.
    """
    result: dict[str, Any] = {}
    for key, value in schema.items():
        if key == "nullable":
            continue
        if isinstance(value, dict):
            result[key] = _nullable_to_type_array(value)
        elif isinstance(value, list):
            result[key] = [
                _nullable_to_type_array(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value

    if schema.get("nullable") is True:
        _inject_null_type(result)

    return result


def convert_nullable_to_type_array(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert ``"nullable": true`` to standard JSON Schema ``"type": [T, "null"]``.

    Providers like Anthropic do not support the OpenAPI ``nullable`` extension.
    This function recursively replaces it with the equivalent JSON Schema
    representation using type arrays.

    Args:
        schema: A sanitized JSON Schema dict.

    Returns:
        A new dict with ``nullable`` fields replaced by type arrays.
    """
    return _nullable_to_type_array(schema)
