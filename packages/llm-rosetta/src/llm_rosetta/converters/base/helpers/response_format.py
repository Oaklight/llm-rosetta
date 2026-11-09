"""The two shapes a response-format JSON schema takes across the IR (#1327).

The IR spells ``json_schema`` as the OpenAI **envelope** — ``{name,
description, schema, strict}`` — because that is what Chat Completions and
Responses define on the wire, and the field is theirs to fill.  Both Google
dialects (``google_generate`` and ``google_interactions``) take the **raw JSON
Schema** instead, under ``response_schema``.

Reading one shape into a field the other writer expects loses the schema: a
Responses writer that only knows the envelope sees a raw ``{type: object}``,
finds no ``schema`` key, and writes a format that asks for "some JSON".  So
every reader stores the envelope and every writer unwraps it; a bare schema
still arrives from an IR producer built before the envelope was the contract,
which is why the unwrap tolerates a value that is already raw.

Which shape a value is, is decided in exactly one place — :func:`is_envelope`
— because a writer that reported "these keys are dropped" on its own reading
told the caller a ``description`` was lost while writing it to the wire.
"""

from __future__ import annotations

from typing import Any

#: The envelope keys a raw JSON Schema has no place for.
ENVELOPE_ONLY_KEYS = ("name", "description", "strict")


def is_envelope(value: Any) -> bool:
    """Whether *value* is the IR envelope rather than a raw JSON Schema.

    The test is the presence of ``schema``, which is the one key of the
    envelope a raw JSON Schema has no use for.  It is a convention, not a
    proof: a hand-written schema could carry a non-standard top-level
    ``schema`` keyword and be read as an envelope — the IR had no marker to
    tell them apart, and inventing one would change the field's shape again.

    Args:
        value: The IR ``json_schema`` value.

    Returns:
        ``True`` for a mapping that carries a ``schema`` key.
    """
    return isinstance(value, dict) and "schema" in value


def schema_to_ir(json_schema: Any) -> dict[str, Any]:
    """Return *json_schema* as the IR envelope.

    A value that already carries a ``schema`` key is the envelope and is
    returned as it is.  Anything else is a raw JSON Schema and is wrapped, with
    no name invented: the Open Responses definition makes ``name`` optional,
    and a value the source did not send is not this layer's to add.

    Args:
        json_schema: The schema as a provider body carries it.

    Returns:
        The envelope.  An empty or non-mapping value yields ``{}``.
    """
    if not isinstance(json_schema, dict):
        return {}
    if is_envelope(json_schema):
        return json_schema
    return {"schema": json_schema}


def schema_to_raw(json_schema: Any) -> dict[str, Any]:
    """Return the raw JSON Schema that *json_schema* describes.

    An envelope yields its ``schema``; a dict without that key is already raw
    and is returned unchanged, so a reader that stored the pre-#1327 shape
    keeps working.  The other envelope keys have no JSON Schema spelling and
    are dropped — a dialect that cannot carry them reports that as a loss on
    its own leg, using :func:`dropped_envelope_keys`.

    Args:
        json_schema: The IR ``json_schema`` value.

    Returns:
        The raw schema.  An empty or non-mapping value yields ``{}``.
    """
    if not isinstance(json_schema, dict):
        return {}
    if is_envelope(json_schema):
        nested = json_schema["schema"]
        return nested if isinstance(nested, dict) else {}
    return json_schema


def dropped_envelope_keys(json_schema: Any) -> list[str]:
    """The envelope keys a dialect that takes only the raw schema loses.

    A raw schema loses nothing: ``description`` and ``name`` are ordinary
    keywords there, and they are written to the wire as part of it.  Only an
    envelope has keys with no raw spelling — and ``strict: false`` is not one
    of them: it asks for *less* enforcement than the target's constrained
    decoding already applies, so dropping it changes nothing.

    Args:
        json_schema: The IR ``json_schema`` value about to be written raw.

    Returns:
        The keys that will not reach the upstream, in a stable order.
    """
    if not is_envelope(json_schema):
        return []
    return [
        key
        for key in ENVELOPE_ONLY_KEYS
        if key in json_schema and not (key == "strict" and json_schema[key] is not True)
    ]
