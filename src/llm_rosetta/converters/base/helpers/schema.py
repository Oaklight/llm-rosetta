"""JSON Schema sanitization for LLM provider compatibility.

Recursively strips unsupported keywords, resolves ``$ref`` references,
and flattens ``anyOf``/``oneOf``/``allOf`` combination patterns that
providers like Vertex AI reject.

Used by all provider tool-definition converters to clean parameter schemas
before sending them upstream.
"""

from typing import Any

# JSON Schema keywords not supported by OpenAI / Vertex AI compatible
# endpoints.  These are valid per the JSON Schema spec but upstream servers
# (e.g. Vertex AI's OpenAI-compatible layer) reject them with Pydantic
# ``extra='forbid'`` validation errors.
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
}

# Keys that hold definition maps (consumed for $ref resolution, then removed).
_DEFS_KEYS: set[str] = {"$defs", "definitions"}


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
    """Flatten ``anyOf``/``oneOf`` nullable patterns into a simple typed schema.

    Vertex AI's OpenAI-compatible layer does not support ``anyOf``/``oneOf``
    at all.  The most common pattern is a nullable union like
    ``{"anyOf": [{"type": "string"}, {"type": "null"}]}``, which we convert to
    ``{"type": "string", "nullable": true}``.

    For single-variant unions we unwrap directly.  For multi-type (non-null)
    unions we keep only the first non-null variant (lossy but safe).

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
            # Multiple non-null types: pick the first (lossy but avoids rejection)
            _deep_merge_schema(base, non_null[0])
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
    if "required" not in result or "properties" not in result:
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
) -> dict[str, Any]:
    """Recursively remove unsupported JSON Schema keywords.

    Also resolves ``$ref`` references by inlining the referenced definition,
    and flattens ``anyOf``/``oneOf``/``allOf`` combination keywords into
    simple typed schemas, as required by Vertex AI's OpenAI-compatible layer
    which does not support these constructs at all.

    Results are cached at the top-level entry point (where ``defs is None``)
    to avoid redundant work when the same tool schemas are converted
    repeatedly across conversation turns.

    Args:
        schema: A JSON Schema dict (or sub-schema).
        defs: Collected ``$defs``/``definitions`` from the top-level schema.
            Populated automatically on the first call if the schema contains
            definition maps.
        extra_strip_keys: Additional provider-specific keys to strip
            (e.g. ``{"additionalProperties"}`` for Google GenAI).

    Returns:
        A new dict with unsupported keys removed at every level.
    """
    # Only cache top-level calls (defs=None is the public entry point).
    # Recursive calls from _sanitize_value go through _sanitize_schema_impl
    # directly and already have defs populated.
    if defs is not None:
        return _sanitize_schema_impl(schema, defs, extra_strip_keys)

    from .cache import _SENTINEL, sanitize_cache, schema_cache_key

    frozen_extra = frozenset(extra_strip_keys) if extra_strip_keys else None
    key = schema_cache_key(schema, frozen_extra)
    cached = sanitize_cache.get(key)
    if cached is not _SENTINEL:
        return cached

    result = _sanitize_schema_impl(schema, defs, extra_strip_keys)
    sanitize_cache.put(key, result)
    return result
