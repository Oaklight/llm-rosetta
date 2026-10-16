"""The ledger's columns, and the canonical total that fills one of them.

Every usage figure the project stores or shows is one of six columns.  This
module is the only place that says what they are, which IR field each one comes
from, and which single value counts as "the total".

A column does not carry the same IR field name for every model type.  Chat asks
the provider for ``prompt_tokens``; the decision IR reports ``input_tokens``
already.  So the field map is declared per model type, and the four types are
the registry's own vocabulary (``gateway/model_types.py``); this package cannot
import the registry, so the names are restated here and a test pins them
together.

A count is an integer, or a string that spells one.  Anything else refuses the
usage whole rather than yielding a total that is quietly short.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

__all__ = [
    "DEFAULT_MODEL_TYPE",
    "IR_FIELDS",
    "LEDGER_COLUMNS",
    "MODEL_TYPES",
    "MODEL_TYPE_FIELDS",
    "token_total",
    "usage_columns",
]

#: The token columns of a ledger row, in display order.  ``total_tokens`` is the
#: canonical total; the rest are a breakdown of it, not parts that add up to it
#: (reasoning rides inside the output, and cache is additive for some standards
#: and a subset of the input for others — see
#: :mod:`llm_rosetta.observability.usage.semantics`).
LEDGER_COLUMNS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
    "reasoning_tokens",
)

#: What a ledger row is about when it does not say.  Rows written before the
#: column existed were chat, and the column's own default is the same value.
DEFAULT_MODEL_TYPE = "llm"

#: The model types whose usage this module can normalize, matching the names
#: ``gateway.model_types`` registers — ``llm``, ``embedding``, ``rerank``,
#: ``decision``.  A column value outside this set is a writer's bug, not a
#: usage shape to guess at.
MODEL_TYPES: tuple[str, ...] = ("llm", "embedding", "rerank", "decision")

#: Ledger column -> the IR usage field behind it, per model type.
#:
#: Only the fields a family's IR type actually declares are listed: the
#: embedding and rerank IR types have no output side, and the decision IR type
#: has neither cache counters nor a total.  A family that reports no total
#: falls back to the additive rule below.
MODEL_TYPE_FIELDS: Mapping[str, Mapping[str, str]] = {
    "llm": {
        "input_tokens": "prompt_tokens",
        "output_tokens": "completion_tokens",
        "total_tokens": "total_tokens",
        "cache_read_tokens": "cache_read_tokens",
        "cache_creation_tokens": "cache_creation_tokens",
        "reasoning_tokens": "reasoning_tokens",
    },
    "embedding": {
        "input_tokens": "prompt_tokens",
        "total_tokens": "total_tokens",
    },
    "rerank": {
        "input_tokens": "prompt_tokens",
        "total_tokens": "total_tokens",
        "cache_read_tokens": "cached_tokens",
    },
    "decision": {
        "input_tokens": "input_tokens",
        "output_tokens": "output_tokens",
    },
}

#: The chat mapping, under the name it had when chat was the only family.
IR_FIELDS: Mapping[str, str] = MODEL_TYPE_FIELDS[DEFAULT_MODEL_TYPE]

#: The columns the additive rule sums, in this order.  Reasoning is absent on
#: purpose: it rides inside the output for every provider that reports it.
ADDITIVE_COLUMNS: tuple[str, ...] = (
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_creation_tokens",
)


def _count(value: Any) -> int | None:
    """The token count behind a usage value, or ``None`` when it is not one.

    A provider that JSON-strings its numbers still sent a count, so a string
    spelling an integer is one.  Anything else — a float, a boolean, a nested
    object — is not, and :func:`token_total` refuses a usage that carries one
    rather than measuring part of it.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _additive_counts(usage: Mapping[str, Any], fields: Mapping[str, str]) -> list[int]:
    """The counts the additive rule sums; ``[]`` when one defies counting.

    An empty list is what a payload that reports none of the additive counters
    also returns, which is normal for a total the provider did not send — the
    caller decides.  A counter that is present and not a count is the other
    case, and is refused by :func:`_refused` instead.
    """
    counts: list[int] = []
    for column in ADDITIVE_COLUMNS:
        field = fields.get(column)
        if field is None or usage.get(field) is None:
            continue
        count = _count(usage.get(field))
        if count is None:
            return []
        counts.append(count)
    return counts


def _refused(usage: Mapping[str, Any], fields: Mapping[str, str]) -> bool:
    """True when a counter the total depends on is present and not a count.

    A payload that names one countable counter and one uncountable one is not
    half-measured: the total is refused, and so is the row — a breakdown
    written without the total it belongs to reports a figure the payload never
    supported.
    """
    total_field = fields.get("total_tokens")
    if total_field is not None and usage.get(total_field) is not None:
        return _count(usage.get(total_field)) is None
    for column in ADDITIVE_COLUMNS:
        field = fields.get(column)
        if field is None or usage.get(field) is None:
            continue
        if _count(usage.get(field)) is None:
            return True
    return False


def token_total(
    usage: Mapping[str, Any] | None, model_type: str = DEFAULT_MODEL_TYPE
) -> int | None:
    """The canonical token total of one model type's IR usage dict.

    The provider-reported ``total_tokens`` wins whenever the provider sends one
    — including one it sent as a string, which is the shape that would
    otherwise be recomputed *with* the cache counters on top of a figure the
    provider had already counted them into.  A total the provider stands behind
    is not ours to recompute, and for the OpenAI-shaped standards it sits below
    the additive sum on purpose (their cached counters are part of the prompt
    count).  Only when no total is sent — the Messages API never sends one — is
    it computed additively: input + output + cache read + cache write, each read
    through the field the model type declares for that role.

    :meth:`~llm_rosetta.observability.persistence.PersistenceManager.backfill_total_tokens`
    applies the same rule to rows an older writer left blank or derived too low,
    which is why a stored total survives a restart unchanged.

    ``reasoning_tokens`` is deliberately never added: it is a subset of the
    output tokens for every provider that reports it.

    A counter that is not a count — a float, a boolean, a string that does not
    spell an integer — refuses the whole usage rather than being skipped:
    summing the countable fields of a payload that also carries an uncountable
    one would report a total that is quietly short.

    Args:
        usage: The IR usage dict of one response.
        model_type: The family that produced *usage*; picks the field names.
            Defaults to chat, which is what a row written before the column
            existed means.

    Returns:
        The total, or ``None`` when *usage* names no usable token count.

    Raises:
        KeyError: If *model_type* is not one of :data:`MODEL_TYPES`.
    """
    fields = MODEL_TYPE_FIELDS[model_type]
    if not usage or _refused(usage, fields):
        return None
    total_field = fields.get("total_tokens")
    total = usage.get(total_field) if total_field else None
    if total is not None:
        return _count(total)
    counts = _additive_counts(usage, fields)
    if not counts:
        return None
    return sum(counts)


def usage_columns(
    usage: Mapping[str, Any] | None, model_type: str = DEFAULT_MODEL_TYPE
) -> dict[str, int]:
    """The ledger columns of one model type's IR usage dict.

    A column the provider did not report is left out rather than zeroed, so a
    caller can tell "no cache in play" from "cache reported as zero".  A
    counter is read through :func:`_count`, the same way the total reads it, so
    a row cannot record a breakdown its own total was not computed from; a
    usage the total refuses is refused entirely, and reports no columns.

    Args:
        usage: The IR usage dict of one response.
        model_type: The family that produced *usage*; picks the field names.

    Returns:
        A subset of :data:`LEDGER_COLUMNS`, suitable for splatting into
        ``RequestLogEntry.create`` or ``MetricsCollector.record_request``.

    Raises:
        KeyError: If *model_type* is not one of :data:`MODEL_TYPES`.
    """
    fields = MODEL_TYPE_FIELDS[model_type]
    if not usage:
        return {}
    total = token_total(usage, model_type)
    if total is None and _refused(usage, fields):
        return {}
    columns: dict[str, int] = {}
    for column, field in fields.items():
        count = _count(usage.get(field))
        if count is not None:
            columns[column] = count
    if total is not None:
        columns["total_tokens"] = total
    return columns
