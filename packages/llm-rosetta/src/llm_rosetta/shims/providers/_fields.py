"""The ``fields:`` block: validation and manifest/leaf merging."""

from __future__ import annotations

from pathlib import Path
from typing import Any


from ..auth_exchange import FIELD_KEY_RE, FIELD_LIMIT, FIELD_TEXT_LIMIT
from ._constants import (
    _IDENTITY_KEYS,
)


def _declares_identity(cfg: Any) -> bool:
    """Whether a parsed ``provider.yaml`` names a shim."""
    return isinstance(cfg, dict) and any(key in cfg for key in _IDENTITY_KEYS)


def _declared_fields(raw: Any, source: Path) -> tuple[dict[str, Any], ...]:
    """Validate a shim's ``fields:`` block into the shape the gateway renders.

    The declaration is data, so a typo in it is a load-time error rather than a
    field that silently never appears.  The rules are the panel's own
    (``FIELD_KEY_RE``, ``FIELD_LIMIT``, ``FIELD_TEXT_LIMIT`` from
    :mod:`llm_rosetta.shims.auth_exchange`), because a key the loader accepts
    and the panel drops is exactly the silent failure the block exists to
    prevent: every entry must be a mapping with an identifier ``key`` and a
    ``label``, its descriptive strings must fit, and an unknown key is refused
    rather than ignored — ``secret`` was advertised once and would do nothing.

    Args:
        raw: The ``fields:`` value as parsed from YAML, or ``None``.
        source: The file it came from, for the error message.

    Returns:
        One mapping per declared input, keys and optional strings preserved.

    Raises:
        ValueError: The block is not a list of mappings, an entry has no
            usable ``key``/``label``, a key or description breaks a bound, a
            key is repeated, or a key is one this schema does not define.
    """
    if raw is None:
        return ()
    if not isinstance(raw, list):
        raise ValueError(f"{source}: fields must be a list of mappings")
    if len(raw) > FIELD_LIMIT:
        raise ValueError(
            f"{source}: fields declares {len(raw)} entries; at most {FIELD_LIMIT} "
            "are rendered. A credential with more inputs than that is a record — "
            "keep it in the exchange's own store"
        )
    fields: list[dict[str, Any]] = []
    seen: set[str] = set()
    for entry in raw:
        cleaned = _declared_field(entry, source)
        if cleaned["key"] in seen:
            raise ValueError(
                f"{source}: the field {cleaned['key']!r} is declared twice"
            )
        seen.add(cleaned["key"])
        fields.append(cleaned)
    return tuple(fields)


#: The keys one declared input may carry.  Anything else is a declaration the
#: panel would silently ignore, which is what makes ``secret`` a load error.
_FIELD_KEYS = frozenset({"key", "label", "placeholder", "help", "required"})


def _declared_field(entry: Any, source: Path) -> dict[str, Any]:
    """Validate one ``fields:`` entry and return it in the panel's shape.

    Args:
        entry: One list element from the block, as parsed.
        source: The file it came from, for the error message.

    Returns:
        The entry with ``key`` and ``label`` required, the optional strings
        kept, and an explicit ``required: false`` dropped.

    Raises:
        ValueError: The entry is not a mapping, its key or label is missing or
            unusable, its descriptive strings are too long, or it names a key
            this schema does not define.
    """
    if not isinstance(entry, dict):
        raise ValueError(f"{source}: every fields entry must be a mapping")
    key = entry.get("key")
    if not isinstance(key, str) or not key.strip():
        raise ValueError(f"{source}: a fields entry declares no key")
    key = key.strip()
    if not FIELD_KEY_RE.match(key):
        raise ValueError(
            f"{source}: the field key {key!r} is not a plain identifier "
            "(letters, digits, '_.-', at most 64); the panel renders it as "
            "an element id"
        )
    label = entry.get("label")
    if not isinstance(label, str) or not label.strip():
        raise ValueError(f"{source}: the field {key!r} declares no label")
    if len(label) > FIELD_TEXT_LIMIT:
        raise ValueError(
            f"{source}: the field {key!r} has a label longer than "
            f"{FIELD_TEXT_LIMIT} characters"
        )
    unknown = set(entry) - _FIELD_KEYS
    if unknown:
        raise ValueError(
            f"{source}: the field {key!r} declares {', '.join(sorted(unknown))}, "
            "which this schema does not define (key, label, placeholder, help, "
            "required)"
        )
    cleaned: dict[str, Any] = {"key": key, "label": label.strip()}
    for name in ("placeholder", "help"):
        value = entry.get(name)
        if value is None:
            continue
        if not isinstance(value, str):
            raise ValueError(f"{source}: the field {key!r} has a non-string {name}")
        if len(value) > FIELD_TEXT_LIMIT:
            raise ValueError(
                f"{source}: the field {key!r} has {name} longer than "
                f"{FIELD_TEXT_LIMIT} characters"
            )
        cleaned[name] = value
    required = entry.get("required")
    if required is not None and not isinstance(required, bool):
        raise ValueError(f"{source}: the field {key!r} has a non-boolean required")
    if required:
        cleaned["required"] = True
    return cleaned


def _merged_fields(
    leaf: tuple[dict[str, Any], ...], vendor: tuple[dict[str, Any], ...]
) -> tuple[dict[str, Any], ...]:
    """Merge a leaf's declared fields over its vendor's, keyed by ``key``.

    A vendor declares the inputs every lane shares and a leaf overrides the
    attributes it words differently; order follows the vendor first so the form
    reads the same way for every lane.  A leaf entry merges *over* the vendor
    entry attribute by attribute, so rewording the label does not drop the
    vendor's ``help`` or its ``required`` — a whole-entry replacement would.
    """
    if not leaf:
        return tuple(vendor)
    if not vendor:
        return tuple(leaf)
    by_key = {entry["key"]: entry for entry in leaf}
    merged: list[dict[str, Any]] = []
    for entry in vendor:
        override = by_key.pop(entry["key"], None)
        merged.append({**entry, **override} if override else entry)
    merged.extend(by_key.values())
    return tuple(merged)
